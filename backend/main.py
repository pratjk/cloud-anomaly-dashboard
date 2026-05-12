"""
Cloud Anomaly Detection - FastAPI Backend

HTTP REST API for anomaly detection predictions on cloud metrics.
Integrates hybrid metric and log-based anomaly detection models.

Endpoints:
    - POST /predict: Submit metrics/logs, get anomaly prediction
    - GET /predictions: Retrieve all stored predictions from database
    - GET /health: Comprehensive health check (database, model availability)
    - GET /: Basic health check/status endpoint

Request Flow for /predict:
    1. Validate request structure (dict, non-empty)
    2. Extract and validate metrics (cpu, memory, disk, network)
    3. Extract log message
    4. Run hybrid prediction model
    5. Store prediction in database
    6. Check if model retraining is needed (every 50 records)
    7. Retrain model if version stale
    8. Return prediction result with cause

Configuration:
    - MODEL_RETRAIN_INTERVAL: Records between retraining (default: 50)
    - MODEL_VERSION_FILE: Version tracking file path
    - DATABASE_TIMEOUT: Request timeout to database

Returns:
    JSON responses with anomaly prediction and cause/error details
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import engine, SessionLocal, get_db_session, init_db
from database.models import Prediction, QuarantinedDevice
from ml.predict import predict_anomaly
from ml.retrain import retrain_model

# ============================================================================
# Configuration Section
# ============================================================================

# Environment-based configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = ["*"]  # Allow all origins for development

MODEL_RETRAIN_INTERVAL: int = int(os.getenv("MODEL_RETRAIN_INTERVAL", "50"))
MODEL_VERSION_FILE: str = os.getenv("MODEL_VERSION_FILE", "ml/model_version.txt")
METRIC_MODEL_PATH: str = os.getenv("METRIC_MODEL_PATH", "ml/model.pkl")
LOG_MODEL_PATH: str = os.getenv("LOG_MODEL_PATH", "ml/log_model.pkl")

# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/system.log"),
        logging.StreamHandler()
    ]
)

# ============================================================================
# FastAPI App Initialization
# ============================================================================

app = FastAPI(
    title="Cloud Anomaly Detection API",
    description="Hybrid metric and log-based anomaly detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================================
# Global Exception Handler
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> dict:
    """
    Global exception handler for all unhandled exceptions.
    Ensures responses are always valid JSON.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "error": "Internal server error",
        "message": str(exc),
        "status": 500,
        "timestamp": datetime.now().isoformat()
    }

# Initialize database tables on startup
try:
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize database: {e}", exc_info=True)
    raise


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_request_data(data: Any) -> None:
    """
    Validate request body structure before processing.

    Args:
        data: Request body data

    Raises:
        ValueError: If request data is invalid

    """
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    
    if not data:
        raise ValueError("Request body cannot be empty")
    
    logger.debug(f"Request data validated: {list(data.keys())}")


def _extract_and_validate_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metric values from request and validate ranges.

    Args:
        data: Request data dictionary

    Returns:
        Dictionary with validated metrics (cpu_usage, memory_usage, disk_io, network_traffic)

    Raises:
        ValueError: If metrics are missing, non-numeric, or out of valid range

    """
    try:
        cpu = float(data.get("cpu_usage", 0))
        memory = float(data.get("memory_usage", 0))
        disk = float(data.get("disk_io", 0))
        network = float(data.get("network_traffic", 0))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Metrics must be numeric values: {e}") from e

    # Validate metric ranges and clamp if needed
    if not (0 <= cpu <= 100):
        logger.warning(f"CPU usage out of valid range [0-100]: {cpu}, clamping")
        cpu = max(0, min(100, cpu))
    
    if not (0 <= memory <= 100):
        logger.warning(f"Memory usage out of valid range [0-100]: {memory}, clamping")
        memory = max(0, min(100, memory))
    
    if disk < 0:
        logger.warning(f"Disk I/O is negative: {disk}, setting to 0")
        disk = 0
    
    if network < 0:
        logger.warning(f"Network traffic is negative: {network}, setting to 0")
        network = 0

    metrics = {
        "cpu_usage": cpu,
        "memory_usage": memory,
        "disk_io": disk,
        "network_traffic": network,
    }

    logger.debug(f"Metrics extracted and validated: {metrics}")
    return metrics


def _extract_log_message(data: Dict[str, Any]) -> str:
    """
    Extract and normalize log message from request.

    Args:
        data: Request data dictionary

    Returns:
        Log message string

    """
    log_message = data.get("log_message", "INFO Normal operation")
    
    if not isinstance(log_message, str):
        log_message = str(log_message)
    
    logger.debug(f"Log message extracted: {log_message}")
    return log_message


def _run_prediction(data: Dict[str, Any]) -> tuple[str, str, str]:
    """
    Execute anomaly prediction with error handling.

    Args:
        data: Request data with metrics and optional log message

    Returns:
        Tuple of (prediction: str, cause: str, remediation: str)

    """
    try:
        logger.debug("Running anomaly prediction")
        result = predict_anomaly(data)
        logger.debug(f"Model returned raw result: {result}")

        # Handle multiple result formats
        if isinstance(result, dict):
            prediction = result.get("prediction", "Normal")
            cause = result.get("cause", "Model decision")
            remediation = result.get("remediation", "None")
        else:
            prediction = str(result)
            cause = "Model decision"
            remediation = "None"

        logger.info(f"Prediction: {prediction}, Cause: {cause}, Remediation: {remediation}")
        return (prediction, cause, remediation)

    except Exception as e:
        error_msg = f"Prediction model failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Return error status with full error cause
        return ("Error", f"Model Error: {str(e)}", "None")


def _store_prediction_in_db(
    db: Session,
    device_id: str,
    cpu: float,
    memory: float,
    disk: float,
    network: float,
    prediction: str,
    cause: str = "Not available",
    remediation: str = "None",
    latitude: float = 0.0,
    longitude: float = 0.0
) -> bool:
    """
    Store prediction record in database with error handling.

    Args:
        db: Database session
        device_id: Source device ID
        cpu: CPU usage value
        memory: Memory usage value
        disk: Disk I/O value
        network: Network traffic value
        prediction: Prediction result
        cause: Reason/explanation for the prediction (optional, default "Not available")
        remediation: Tactical action performed (optional, default "None")
        latitude: Client latitude coordinate (optional, default 0.0)
        longitude: Client longitude coordinate (optional, default 0.0)

    Returns:
        True if stored successfully, False otherwise

    """
    try:
        logger.debug("Creating prediction record")
        
        record = Prediction(
            device_id=device_id,
            cpu_usage=cpu,
            memory_usage=memory,
            disk_io=disk,
            network_traffic=network,
            prediction=prediction,
            cause=cause,
            remediation=remediation,
            latitude=latitude,
            longitude=longitude
        )

        db.add(record)
        db.commit()
        logger.info(f"Prediction record stored in database (prediction={prediction}, cause={cause}, remediation={remediation}, geo=({latitude}, {longitude}))")
        return True

    except Exception as e:
        logger.error(f"Failed to store prediction in database: {e}", exc_info=True)
        db.rollback()
        return False


def _check_and_retrain_model(db: Session) -> None:
    """
    Check if model retraining is needed and execute if necessary.

    Retrains model when total records count reaches multiple of RETRAIN_INTERVAL
    and model version is stale.

    Args:
        db: Database session

    """
    try:
        total_records = db.query(Prediction).count()
        logger.debug(f"Total prediction records in database: {total_records}")

        # Check if retraining interval reached
        if total_records % MODEL_RETRAIN_INTERVAL != 0 or total_records == 0:
            return

        logger.info(f"Model retraining interval reached ({total_records} records)")

        # Check model version
        try:
            if not os.path.exists(MODEL_VERSION_FILE):
                logger.info("Model version file not found, initiating retraining")
                retrain_model()
                return

            with open(MODEL_VERSION_FILE, "r") as f:
                version_content = f.read().strip()

            if not version_content.isdigit():
                logger.warning(f"Invalid model version file format: '{version_content}'")
                retrain_model()
                return

            current_version = int(version_content)
            expected_version = total_records // MODEL_RETRAIN_INTERVAL

            if current_version < expected_version:
                logger.info(
                    f"Model version stale (v{current_version} < v{expected_version}), "
                    f"triggering retraining"
                )
                retrain_model()
            else:
                logger.debug(f"Model version current (v{current_version})")

        except (ValueError, OSError) as e:
            logger.error(f"Failed to check model version: {e}", exc_info=True)
            # Attempt retrain on version check failure
            retrain_model()

    except Exception as e:
        logger.error(f"Error in model retraining check: {e}", exc_info=True)


def _check_database_health() -> Tuple[bool, str]:
    """Check database connectivity and availability.
    
    Returns:
        Tuple of (is_healthy: bool, status_message: str)
    """
    try:
        with SessionLocal() as db:
            # Try a simple query to verify connection
            count = db.query(Prediction).count()
            return True, f"Connected - {count} records in database"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Database health check failed: {error_msg}")
        return False, f"Connection failed: {error_msg}"


def _check_model_availability() -> Tuple[bool, Dict[str, bool]]:
    """Check if required model files are available.
    
    Returns:
        Tuple of (all_models_healthy: bool, model_status_dict)
    """
    models_status = {
        "metric_model": os.path.exists(METRIC_MODEL_PATH),
        "log_model": os.path.exists(LOG_MODEL_PATH),
        "model_version_file": os.path.exists(MODEL_VERSION_FILE)
    }
    
    # Metric model is required, others are optional
    all_healthy = models_status["metric_model"]
    
    logger.debug(f"Model availability: {models_status}")
    return all_healthy, models_status


def _check_model_version() -> Tuple[bool, str]:
    """Check model version information.
    
    Returns:
        Tuple of (version_valid: bool, version_info: str)
    """
    try:
        if not os.path.exists(MODEL_VERSION_FILE):
            return False, "Version file not found"
        
        with open(MODEL_VERSION_FILE, "r") as f:
            version_content = f.read().strip()
        
        if not version_content:
            return False, "Version file empty"
        
        if not version_content.isdigit():
            return False, f"Invalid version format: {version_content}"
        
        return True, f"v{version_content}"
    
    except Exception as e:
        logger.error(f"Failed to read model version: {e}")
        return False, f"Error reading version: {str(e)}"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/quarantine")
def list_quarantine(db: Session = Depends(get_db_session)):
    """Retrieve all devices in the cyber jail."""
    try:
        devices = db.query(QuarantinedDevice).all()
        return [d.to_dict() for d in devices]
    except Exception as e:
        logger.error(f"Failed to fetch quarantine list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quarantine/release/{device_id}")
def release_device(device_id: str, db: Session = Depends(get_db_session)):
    """Release a device from the cyber jail."""
    try:
        device = db.query(QuarantinedDevice).filter(QuarantinedDevice.device_id == device_id).first()
        if not device:
            raise HTTPException(status_code=44, detail="Device not found in quarantine")
        
        db.delete(device)
        db.commit()
        logger.info(f"Device {device_id} released from quarantine.")
        return {"status": "success", "message": f"Device {device_id} pardoned."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to release device: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(data: dict, db: Session = Depends(get_db_session)) -> dict:
    """
    Predict anomaly status for given cloud metrics and logs.

    Request body should contain:
        - cpu_usage (float, 0-100): CPU utilization percentage
        - memory_usage (float, 0-100): Memory utilization percentage
        - disk_io (float, >= 0): Disk I/O throughput
        - network_traffic (float, >= 0): Network traffic volume
        - log_message (str, optional): Log message for context

    Returns:
        JSON with prediction result and cause

    Args:
        data: Request body as dictionary

    Raises:
        HTTPException: On validation or processing errors

    """
    try:
        logger.info("=" * 60)
        logger.info("Received prediction request")

        # ===== Step 1: Validate request structure =====
        try:
            _validate_request_data(data)
        except ValueError as e:
            logger.warning(f"Request validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e

        device_id = data.get("device_id", "Unknown")
        
        # --- ZERO TRUST QUARANTINE CHECK ---
        is_jailed = db.query(QuarantinedDevice).filter(QuarantinedDevice.device_id == device_id).first()
        if is_jailed:
            logger.warning(f"BLOCKED: Attempted access from quarantined device {device_id}")
            raise HTTPException(
                status_code=403, 
                detail=f"ACCESS DENIED: Device {device_id} is quarantined due to malicious activity signature."
            )
        # ------------------------------------

        # ===== Step 2: Extract and validate metrics =====
        try:
            metrics = _extract_and_validate_metrics(data)
        except ValueError as e:
            logger.error(f"Metric validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e

        # ===== Step 3: Extract log message and device_id =====
        log_message = _extract_log_message(data)
        device_id = data.get("device_id", "Unknown")

        # ===== Step 4: Run prediction =====
        prediction, cause, remediation = _run_prediction(data)

        # ===== Step 5: Store in database =====
        stored = False
        try:
            with SessionLocal() as db:
                stored = _store_prediction_in_db(
                    db,
                    device_id,
                    metrics["cpu_usage"],
                    metrics["memory_usage"],
                    metrics["disk_io"],
                    metrics["network_traffic"],
                    prediction,
                    cause,  # Pass cause to be stored in database
                    remediation,
                    data.get("latitude", 0.0),
                    data.get("longitude", 0.0)
                )
                
                # --- AUTO-JAIL LOGIC ---
                # If it's a critical anomaly (DDoS or high memory stress), automate the quarantine
                if prediction == "Anomaly":
                    critical_causes = ["DDoS Attack", "Severe Memory Leak", "Network Flood"]
                    is_critical = any(c in cause for c in critical_causes)
                    
                    if is_critical:
                        logger.warning(f"AUTO-JAIL: Critical threat detected from {device_id}. Initiating quarantine.")
                        # Check if already jailed (defensive)
                        if not db.query(QuarantinedDevice).filter(QuarantinedDevice.device_id == device_id).first():
                            jail_record = QuarantinedDevice(
                                device_id=device_id,
                                reason=cause
                            )
                            db.add(jail_record)
                            db.commit()
                # -----------------------
        except Exception as e:
            logger.error(f"Database operation failed: {e}", exc_info=True)
            # Continue without failing - return prediction even if storage fails

        # ===== Step 6: Check and trigger model retraining if needed =====
        if stored:
            try:
                with SessionLocal() as db:
                    _check_and_retrain_model(db)
            except Exception as e:
                logger.error(f"Retrain check failed: {e}", exc_info=True)
                # Silently fail - don't interrupt prediction response

        logger.info("=" * 60)
        return {
            "status": "success",
            "prediction": prediction,
            "cause": cause,
            "remediation": remediation,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "stored": stored
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in /predict endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error",
                "error": str(e)
            }
        ) from e


@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db_session)) -> dict:
    """
    Retrieve all stored anomaly predictions from database.

    Returns:
        Dictionary with list of prediction records, all metrics and results

    Raises:
        HTTPException: On database query failure

    """
    try:
        logger.info("Fetching all prediction records from database")
        
        records = db.query(Prediction).all()
        
        # Convert all SQLAlchemy objects to dictionaries for JSON serialization
        predictions_list = [r.to_dict() for r in records]
        
        logger.info(f"Successfully retrieved {len(predictions_list)} prediction records")
        return {
            "status": "success",
            "count": len(predictions_list),
            "predictions": predictions_list
        }
    except Exception as e:
        logger.error(f"Failed to fetch predictions from database: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to fetch predictions",
                "message": str(e)
            }
        ) from e


@app.get("/health", tags=["health"], status_code=200)
def health_check() -> dict:
    """Comprehensive health check endpoint.
    
    Verifies:
        - Database connectivity and availability
        - Required model files presence
        - Model version information
        - Overall system health status
    
    Returns:
        JSON with detailed health status of all components
        - status: healthy/degraded/unhealthy
        - database: connection status and record count
        - models: availability of metric and log models
        - model_version: current version information
        - timestamp: when check was performed
    
    Status Levels:
        - healthy: All components operational
        - degraded: Optional components missing (e.g., log model)
        - unhealthy: Critical components failed (e.g., database/metric model)
    """
    logger.info("Comprehensive health check requested")
    
    # Check database health
    db_healthy, db_status = _check_database_health()
    
    # Check model availability
    models_healthy, models_status = _check_model_availability()
    
    # Check model version
    version_valid, version_info = _check_model_version()
    
    # Determine overall health status
    if db_healthy and models_healthy:
        # All critical components are healthy
        overall_status = "healthy"
        http_status = 200
    elif db_healthy or models_healthy:
        # At least one critical component is down but not both
        overall_status = "degraded"
        http_status = 200  # Still return 200 but with degraded status
    else:
        # Critical components are down
        overall_status = "unhealthy"
        http_status = 503  # Service Unavailable
    
    response = {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "database": {
            "healthy": db_healthy,
            "status": db_status
        },
        "models": {
            "metric_model_available": models_status["metric_model"],
            "log_model_available": models_status["log_model"],
            "model_version_file_available": models_status["model_version_file"]
        },
        "model_version": {
            "valid": version_valid,
            "version": version_info
        },
        "summary": {
            "api_running": True,
            "critical_components_healthy": db_healthy and models_healthy,
            "ready_for_predictions": db_healthy and models_healthy
        }
    }
    
    logger.info(f"Health check result: {overall_status}")
    
    # Return with appropriate HTTP status
    if http_status == 503:
        raise HTTPException(status_code=503, detail=response)
    
    return response


@app.get("/", tags=["health"])
def root() -> dict:
    """
    Basic health check endpoint.

    Returns:
        Status message confirming API is running
        For detailed health information, use /health endpoint

    """
    logger.debug("Basic health check requested")
    return {
        "message": "Cloud Anomaly Detection API Running",
        "status": "running",
        "info": "Use /health endpoint for detailed system status"
    }


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    logger.info("FastAPI application starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown."""
    logger.info("FastAPI application shutting down")