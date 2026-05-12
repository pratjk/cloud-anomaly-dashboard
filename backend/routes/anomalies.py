"""
Anomaly Detection Routes

FastAPI routes for anomaly prediction and history retrieval.
Provides structured API responses with comprehensive input validation,
error handling, and proper HTTP status codes.

Routes:
    - POST /anomalies/predict: Predict anomaly for given metrics
    - GET /anomalies/history: Retrieve prediction history

Validation:
    - Input type checking (dict, numeric values)
    - Metric range validation with auto-correction
    - Log message normalization
    - Request timeout handling

Error Handling:
    - 400 Bad Request: Invalid inputs
    - 422 Unprocessable Entity: Validation errors
    - 500 Internal Server Error: Processing failures
    - All errors include human-readable messages
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from backend.database import get_db_session
from database.models import Prediction
from ml.predict import predict_anomaly
from ml.retrain import retrain_model
import os

# ============================================================================
# Configuration
# ============================================================================

MODEL_RETRAIN_INTERVAL: int = 50
MODEL_VERSION_FILE: str = "ml/model_version.txt"
MAX_LOG_MESSAGE_LENGTH: int = 500
REQUEST_TIMEOUT: int = 30

# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================


class MetricsInput(BaseModel):
    """Validated metrics input model with auto-correction."""
    
    cpu_usage: float = Field(default=0.0, ge=0, le=100, description="CPU usage 0-100%")
    memory_usage: float = Field(default=0.0, ge=0, le=100, description="Memory usage 0-100%")
    disk_io: float = Field(default=0.0, ge=0, description="Disk I/O in MB/s")
    network_traffic: float = Field(default=0.0, ge=0, description="Network traffic in Mbps")
    log_message: Optional[str] = Field(default=None, max_length=MAX_LOG_MESSAGE_LENGTH, description="Optional log message")

    @validator("cpu_usage", "memory_usage", pre=True)
    def validate_percentages(cls, v):
        """Validate and clamp percentage values."""
        if v is None:
            return 0.0
        try:
            val = float(v)
            if val < 0:
                logger.warning(f"Percentage value below 0: {val}, clamping to 0")
                return 0.0
            if val > 100:
                logger.warning(f"Percentage value above 100: {val}, clamping to 100")
                return 100.0
            return val
        except (ValueError, TypeError) as e:
            raise ValueError(f"Percentage must be numeric: {e}") from e

    @validator("disk_io", "network_traffic", pre=True)
    def validate_non_negative(cls, v):
        """Validate non-negative values."""
        if v is None:
            return 0.0
        try:
            val = float(v)
            if val < 0:
                logger.warning(f"Negative value provided: {val}, setting to 0")
                return 0.0
            return val
        except (ValueError, TypeError) as e:
            raise ValueError(f"Value must be numeric: {e}") from e

    @validator("log_message", pre=True)
    def validate_log_message(cls, v):
        """Normalize log message."""
        if v is None:
            return "INFO Normal operation"
        if not isinstance(v, str):
            v = str(v)
        return v.strip() or "INFO Normal operation"


class AnomalyPredictionRequest(BaseModel):
    """Structured request for anomaly prediction."""
    
    cpu_usage: Optional[float] = Field(default=0.0, description="CPU usage 0-100%")
    memory_usage: Optional[float] = Field(default=0.0, description="Memory usage 0-100%")
    disk_io: Optional[float] = Field(default=0.0, description="Disk I/O in MB/s")
    network_traffic: Optional[float] = Field(default=0.0, description="Network traffic in Mbps")
    log_message: Optional[str] = Field(default=None, description="Optional log message")

    class Config:
        schema_extra = {
            "example": {
                "cpu_usage": 45.5,
                "memory_usage": 60.2,
                "disk_io": 125.3,
                "network_traffic": 500.0,
                "log_message": "ERROR Database connection timeout"
            }
        }


class PredictionResponse(BaseModel):
    """Structured response with prediction result."""
    
    status: str = Field(description="Success or Error")
    prediction: str = Field(description="Anomaly or Normal")
    cause: str = Field(description="Reason for prediction")
    timestamp: Optional[datetime] = Field(default=None, description="Prediction timestamp")
    metrics_received: Dict[str, float] = Field(description="Validated metrics used")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "prediction": "Anomaly",
                "cause": "High CPU usage detected",
                "timestamp": "2026-03-31T10:30:00",
                "metrics_received": {
                    "cpu_usage": 95.0,
                    "memory_usage": 60.0,
                    "disk_io": 150.0,
                    "network_traffic": 600.0
                }
            }
        }


class PredictionHistoryResponse(BaseModel):
    """Single prediction record from history."""
    
    id: int
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_traffic: float
    prediction: str
    cause: Optional[str] = Field(default=None, description="Reason for prediction")
    timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True


class HistoryListResponse(BaseModel):
    """Paginated history response."""
    
    total_count: int = Field(description="Total number of predictions")
    predictions: List[PredictionHistoryResponse] = Field(description="Prediction records")

    class Config:
        schema_extra = {
            "example": {
                "total_count": 150,
                "predictions": [
                    {
                        "id": 150,
                        "cpu_usage": 45.5,
                        "memory_usage": 60.2,
                        "disk_io": 125.3,
                        "network_traffic": 500.0,
                        "prediction": "Normal",
                        "cause": "All metrics within normal range",
                        "timestamp": "2026-03-31T10:30:00"
                    }
                ]
            }
        }


# ============================================================================
# Route Definition
# ============================================================================

router = APIRouter(
    prefix="/anomalies",
    tags=["anomalies"],
    responses={
        400: {"description": "Invalid input"},
        422: {"description": "Validation error"},
        500: {"description": "Server error"}
    }
)


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_metrics_input(data: Any) -> Dict[str, float]:
    """
    Validate and normalize metrics input with comprehensive error handling.

    Args:
        data: Input data (ideally dict-like)

    Returns:
        Validated metrics dictionary

    Raises:
        ValueError: If input is invalid

    """
    try:
        # Type check
        if not isinstance(data, dict):
            raise ValueError(f"Input must be a JSON object, got {type(data).__name__}")

        if not data:
            raise ValueError("Input cannot be empty")

        # Use Pydantic for validation
        validated = MetricsInput(**data)
        logger.debug(f"Input validated: cpu={validated.cpu_usage}, "
                    f"mem={validated.memory_usage}, disk={validated.disk_io}, "
                    f"net={validated.network_traffic}")
        
        return {
            "cpu_usage": validated.cpu_usage,
            "memory_usage": validated.memory_usage,
            "disk_io": validated.disk_io,
            "network_traffic": validated.network_traffic,
            "log_message": validated.log_message
        }

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}", exc_info=True)
        raise ValueError(f"Failed to validate input: {str(e)}") from e


def _run_prediction(data: Dict[str, Any]) -> tuple[str, str]:
    """
    Execute anomaly prediction with comprehensive error handling.

    Args:
        data: Validated metrics and log message

    Returns:
        Tuple of (prediction, cause)

    """
    try:
        logger.debug("Running anomaly prediction model")
        result = predict_anomaly(data)

        # Handle various result formats
        if isinstance(result, dict):
            prediction = result.get("prediction", "Normal")
            cause = result.get("cause", "Model decision")
        elif isinstance(result, str):
            prediction = result
            cause = "Model decision"
        else:
            prediction = str(result)
            cause = "Model conversion"

        logger.info(f"Prediction result: {prediction}, Cause: {cause}")
        return (prediction, cause)

    except ValueError as e:
        logger.error(f"Prediction validation error: {e}")
        raise ValueError(f"Invalid data for prediction: {str(e)}") from e
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise RuntimeError(f"Prediction model unavailable: {str(e)}") from e
    except Exception as e:
        logger.error(f"Prediction model failed: {e}", exc_info=True)
        raise RuntimeError(f"Prediction failed: {str(e)}") from e


def _store_prediction_in_db(
    db: Session,
    metrics: Dict[str, float],
    prediction: str,
    cause: str = "Not available"
) -> bool:
    """
    Store prediction in database with error handling.

    Args:
        db: Database session
        metrics: Validated metrics dictionary
        prediction: Prediction result
        cause: Reason/explanation for the prediction

    Returns:
        True if stored successfully

    """
    try:
        logger.debug("Storing prediction in database")
        
        record = Prediction(
            cpu_usage=metrics["cpu_usage"],
            memory_usage=metrics["memory_usage"],
            disk_io=metrics["disk_io"],
            network_traffic=metrics["network_traffic"],
            prediction=prediction,
            cause=cause
        )

        db.add(record)
        db.commit()
        logger.info(f"Prediction stored successfully (prediction={prediction}, cause={cause})")
        return True

    except Exception as e:
        logger.error(f"Database storage failed: {e}", exc_info=True)
        try:
            db.rollback()
        except Exception as rollback_err:
            logger.error(f"Rollback failed: {rollback_err}")
        return False


def _check_and_retrain_model(db: Session) -> None:
    """
    Check if model retraining is needed and execute.

    Args:
        db: Database session

    """
    try:
        total_records = db.query(Prediction).count()
        logger.debug(f"Total records: {total_records}")

        if total_records == 0 or total_records % MODEL_RETRAIN_INTERVAL != 0:
            return

        logger.info(f"Retraining interval reached ({total_records} records)")

        # Check version file
        try:
            if not os.path.exists(MODEL_VERSION_FILE):
                logger.info("Version file missing, triggering retrain")
                retrain_model()
                return

            with open(MODEL_VERSION_FILE, "r") as f:
                version_content = f.read().strip()

            if not version_content.isdigit():
                logger.warning(f"Invalid version format: {version_content}")
                retrain_model()
                return

            current_version = int(version_content)
            expected_version = total_records // MODEL_RETRAIN_INTERVAL

            if current_version < expected_version:
                logger.info(f"Version stale (v{current_version} < v{expected_version}), retraining")
                retrain_model()
            else:
                logger.debug(f"Model version current (v{current_version})")

        except (ValueError, OSError) as e:
            logger.error(f"Version check failed: {e}")
            retrain_model()

    except Exception as e:
        logger.error(f"Retrain check error: {e}", exc_info=True)


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict anomaly for metrics",
    responses={
        200: {"description": "Prediction successful"},
        400: {"description": "Invalid metrics provided"},
        422: {"description": "Validation failed"},
        500: {"description": "Prediction failed"}
    }
)
def predict_anomaly_endpoint(
    request: AnomalyPredictionRequest,
    db: Session = Depends(get_db_session)
) -> PredictionResponse:
    """
    Predict anomaly status for cloud metrics.

    Validates input, runs hybrid anomaly detection, stores result,
    and checks if model retraining is needed.

    Args:
        request: Prediction request with metrics and optional log message
        db: Database session

    Returns:
        PredictionResponse with status, prediction, and cause

    Raises:
        HTTPException: On validation, model, or database errors

    """
    try:
        logger.info("=" * 70)
        logger.info("Received anomaly prediction request")

        # ===== Step 1: Validate and normalize input =====
        try:
            input_dict = request.dict()
            validated_metrics = _validate_metrics_input(input_dict)
        except ValueError as e:
            logger.warning(f"Input validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to process input"
            ) from e

        # ===== Step 2: Run prediction =====
        try:
            prediction, cause = _run_prediction(validated_metrics)
        except ValueError as e:
            logger.error(f"Prediction validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metrics for prediction: {str(e)}"
            ) from e
        except RuntimeError as e:
            logger.error(f"Prediction execution failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            ) from e
        except Exception as e:
            logger.critical(f"Unexpected prediction error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal prediction error"
            ) from e

        # ===== Step 3: Store in database =====
        stored = False
        try:
            stored = _store_prediction_in_db(db, validated_metrics, prediction, cause)
            if not stored:
                logger.warning("Prediction not stored in database (non-fatal)")
        except Exception as e:
            logger.error(f"Database storage error: {e}")
            # Continue - storage failure shouldn't block response

        # ===== Step 4: Check and trigger retraining if needed =====
        if stored:
            try:
                _check_and_retrain_model(db)
            except Exception as e:
                logger.error(f"Retrain check failed: {e}")
                # Silently fail - don't interrupt response

        logger.info("=" * 70)
        return PredictionResponse(
            status="success",
            prediction=prediction,
            cause=cause,
            timestamp=datetime.now(),
            metrics_received={
                "cpu_usage": validated_metrics["cpu_usage"],
                "memory_usage": validated_metrics["memory_usage"],
                "disk_io": validated_metrics["disk_io"],
                "network_traffic": validated_metrics["network_traffic"]
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"Unexpected endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        ) from e


@router.get(
    "/history",
    response_model=HistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get prediction history",
    responses={
        200: {"description": "History retrieved successfully"},
        500: {"description": "Database error"}
    }
)
def get_prediction_history(
    limit: int = 100,
    db: Session = Depends(get_db_session)
) -> HistoryListResponse:
    """
    Retrieve prediction history with pagination.

    Args:
        limit: Maximum number of records to return (default: 100, max: 1000)
        db: Database session

    Returns:
        HistoryListResponse with total count and prediction records

    Raises:
        HTTPException: On database query failure

    """
    try:
        # Validate limit parameter
        if limit < 1:
            raise ValueError("Limit must be > 0")
        if limit > 1000:
            logger.warning(f"Limit {limit} exceeds max (1000), clamping")
            limit = 1000

        logger.info(f"Fetching prediction history (limit: {limit})")

        # Query database
        total_count = db.query(Prediction).count()
        records = db.query(Prediction).order_by(
            Prediction.id.desc()
        ).limit(limit).all()

        logger.info(f"Retrieved {len(records)} records (total: {total_count})")

        return HistoryListResponse(
            total_count=total_count,
            predictions=[PredictionHistoryResponse.from_orm(r) for r in records]
        )

    except ValueError as e:
        logger.warning(f"Parameter validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameter: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Database query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch prediction history"
        ) from e
