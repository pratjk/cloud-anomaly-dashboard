"""
Metrics Management Routes

FastAPI routes for metrics analysis, aggregation, and monitoring.
Provides comprehensive metrics handling with validation, anomaly detection,
and safe processing of edge cases.

Routes:
    - GET /metrics/summary: Get metrics summary statistics
    - GET /metrics/aggregates: Get aggregated metrics over time
    - POST /metrics/validate: Validate metrics input

Features:
    - Comprehensive metrics validation
    - Detection of abnormal values (out of range, NaN, Inf)
    - Logging of edge cases and concerning values
    - Safe processing with fallback values
    - Statistical aggregation (mean, min, max, stddev)
"""

import logging
import math
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db_session
from database.models import Prediction

# ============================================================================
# Configuration
# ============================================================================

# Metric range definitions
METRIC_RANGES = {
    "cpu_usage": {"min": 0, "max": 100, "label": "CPU Usage (%)", "critical_high": 90},
    "memory_usage": {"min": 0, "max": 100, "label": "Memory Usage (%)", "critical_high": 90},
    "disk_io": {"min": 0, "max": None, "label": "Disk I/O (MB/s)", "critical_high": 500},
    "network_traffic": {"min": 0, "max": None, "label": "Network Traffic (Mbps)", "critical_high": 1000}
}

# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================


class MetricValue(BaseModel):
    """Single metric value with validation."""
    
    name: str = Field(description="Metric name")
    value: float = Field(description="Metric value")
    unit: str = Field(description="Measurement unit")
    is_valid: bool = Field(description="Whether value passed validation")
    is_abnormal: bool = Field(description="Whether value is outside normal range")
    corrected_value: Optional[float] = Field(default=None, description="Corrected value if invalid")

    class Config:
        schema_extra = {
            "example": {
                "name": "cpu_usage",
                "value": 45.5,
                "unit": "%",
                "is_valid": True,
                "is_abnormal": False,
                "corrected_value": None
            }
        }


class MetricsValidationRequest(BaseModel):
    """Request to validate metrics."""
    
    cpu_usage: Optional[float] = Field(default=0.0, description="CPU usage")
    memory_usage: Optional[float] = Field(default=0.0, description="Memory usage")
    disk_io: Optional[float] = Field(default=0.0, description="Disk I/O")
    network_traffic: Optional[float] = Field(default=0.0, description="Network traffic")


class MetricsValidationResponse(BaseModel):
    """Response with validation results."""
    
    status: str = Field(description="Validation status (success, warning, error)")
    timestamp: datetime = Field(description="Validation timestamp")
    validated_metrics: Dict[str, float] = Field(description="Validated metric values")
    metadata: Dict[str, Any] = Field(description="Validation metadata")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    errors: List[str] = Field(default_factory=list, description="Validation errors")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2026-03-31T10:30:00",
                "validated_metrics": {
                    "cpu_usage": 45.5,
                    "memory_usage": 60.0,
                    "disk_io": 125.0,
                    "network_traffic": 500.0
                },
                "metadata": {
                    "cpu_usage": {"is_valid": True, "is_abnormal": False},
                    "memory_usage": {"is_valid": True, "is_abnormal": False},
                    "disk_io": {"is_valid": True, "is_abnormal": False},
                    "network_traffic": {"is_valid": True, "is_abnormal": False}
                },
                "warnings": [],
                "errors": []
            }
        }


class MetricStatistics(BaseModel):
    """Statistical summary of a metric."""
    
    name: str = Field(description="Metric name")
    count: int = Field(description="Number of samples")
    mean: float = Field(description="Average value")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    stddev: float = Field(description="Standard deviation")
    latest: float = Field(description="Latest value")
    unit: str = Field(description="Measurement unit")


class MetricsSummaryResponse(BaseModel):
    """Summary response with statistics."""
    
    status: str = Field(description="Summary status")
    timestamp: datetime = Field(description="Summary timestamp")
    time_period: str = Field(description="Time period covered")
    total_records: int = Field(description="Total prediction records")
    anomaly_count: int = Field(description="Count of anomalous predictions")
    anomaly_percentage: float = Field(description="Percentage of anomalies")
    metrics_stats: Dict[str, MetricStatistics] = Field(description="Statistics per metric")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2026-03-31T10:30:00",
                "time_period": "Last 24 hours",
                "total_records": 150,
                "anomaly_count": 12,
                "anomaly_percentage": 8.0,
                "metrics_stats": {
                    "cpu_usage": {
                        "name": "cpu_usage",
                        "count": 150,
                        "mean": 45.5,
                        "min": 10.0,
                        "max": 95.0,
                        "stddev": 20.5,
                        "latest": 50.0,
                        "unit": "%"
                    }
                }
            }
        }


class AggregateMetricsResponse(BaseModel):
    """Aggregated metrics over time."""
    
    status: str = Field(description="Aggregation status")
    timestamp: datetime = Field(description="Response timestamp")
    interval: str = Field(description="Aggregation interval")
    aggregates: List[Dict[str, Any]] = Field(description="Aggregated data points")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2026-03-31T10:30:00",
                "interval": "Hourly",
                "aggregates": [
                    {
                        "period": "2026-03-31 10:00:00",
                        "cpu_mean": 45.5,
                        "memory_mean": 60.0,
                        "disk_mean": 125.0,
                        "network_mean": 500.0,
                        "count": 50
                    }
                ]
            }
        }


# ============================================================================
# Route Definition
# ============================================================================

router = APIRouter(
    prefix="/metrics",
    tags=["metrics"],
    responses={
        400: {"description": "Invalid input"},
        500: {"description": "Server error"}
    }
)


# ============================================================================
# Validation Functions
# ============================================================================


def _validate_single_metric(
    name: str,
    value: Any
) -> tuple[bool, Optional[float], List[str]]:
    """
    Validate a single metric value comprehensively.

    Args:
        name: Metric name
        value: Metric value to validate

    Returns:
        Tuple of (is_valid, corrected_value, warnings)

    """
    warnings = []

    # Check metric exists in definitions
    if name not in METRIC_RANGES:
        logger.warning(f"Unknown metric: {name}")
        return False, None, [f"Unknown metric: {name}"]

    # Type conversion
    try:
        numeric_value = float(value) if value is not None else 0.0
    except (ValueError, TypeError) as e:
        warnings.append(f"Cannot convert {name}={value} to float: {e}")
        logger.warning(f"Type conversion failed for {name}: {e}")
        return False, None, warnings

    # Check for special float values
    if math.isnan(numeric_value):
        warnings.append(f"{name} is NaN, setting to 0")
        logger.warning(f"NaN detected in {name}")
        return True, 0.0, warnings

    if math.isinf(numeric_value):
        warnings.append(f"{name} is Inf, setting to 0")
        logger.warning(f"Inf detected in {name}")
        return True, 0.0, warnings

    # Get metric definition
    metric_def = METRIC_RANGES[name]
    min_val = metric_def["min"]
    max_val = metric_def["max"]
    critical_high = metric_def.get("critical_high")

    # Range validation
    corrected = numeric_value
    is_valid = True

    if min_val is not None and numeric_value < min_val:
        logger.warning(f"{name} below minimum ({numeric_value} < {min_val}), clamping")
        warnings.append(f"{name} below minimum range ({min_val}), clamped to {min_val}")
        corrected = min_val
        is_valid = False

    if max_val is not None and numeric_value > max_val:
        logger.warning(f"{name} above maximum ({numeric_value} > {max_val}), clamping")
        warnings.append(f"{name} above maximum range ({max_val}), clamped to {max_val}")
        corrected = max_val
        is_valid = False

    # Abnormal value detection
    if critical_high and numeric_value > critical_high:
        logger.warning(f"{name} critically high: {numeric_value} (critical: {critical_high})")
        warnings.append(f"{name} critically high ({numeric_value} > {critical_high})")

    return is_valid, corrected, warnings


def _validate_all_metrics(
    cpu: Optional[float] = None,
    memory: Optional[float] = None,
    disk: Optional[float] = None,
    network: Optional[float] = None
) -> tuple[Dict[str, float], Dict[str, Any], List[str], List[str]]:
    """
    Validate all metrics comprehensively.

    Args:
        cpu: CPU usage
        memory: Memory usage
        disk: Disk I/O
        network: Network traffic

    Returns:
        Tuple of (validated_metrics, metadata, warnings, errors)

    """
    metrics_input = {
        "cpu_usage": cpu or 0.0,
        "memory_usage": memory or 0.0,
        "disk_io": disk or 0.0,
        "network_traffic": network or 0.0,
    }

    validated_metrics = {}
    metadata = {}
    all_warnings = []
    all_errors = []

    for metric_name, metric_value in metrics_input.items():
        is_valid, corrected_value, warnings = _validate_single_metric(metric_name, metric_value)

        validated_metrics[metric_name] = corrected_value if corrected_value is not None else 0.0
        metadata[metric_name] = {
            "is_valid": is_valid,
            "is_abnormal": corrected_value != metric_value if corrected_value is not None else False,
            "original_value": metric_value,
            "corrected_value": corrected_value
        }

        if warnings:
            all_warnings.extend(warnings)

        if not is_valid and corrected_value is None:
            all_errors.extend(warnings)

    logger.debug(f"Metrics validation complete: {validated_metrics}")
    return validated_metrics, metadata, all_warnings, all_errors


def _calculate_statistics(db: Session, metric_name: str) -> Optional[MetricStatistics]:
    """
    Calculate statistics for a metric from database records.

    Args:
        db: Database session
        metric_name: Name of metric column

    Returns:
        MetricStatistics object or None if no data

    """
    try:
        # Get column object
        column = getattr(Prediction, metric_name, None)
        if column is None:
            logger.warning(f"Column {metric_name} not found in Prediction model")
            return None

        # Calculate statistics
        result = db.query(
            func.count(column).label("count"),
            func.avg(column).label("mean"),
            func.min(column).label("min_val"),
            func.max(column).label("max_val")
        ).first()

        if not result or result.count == 0:
            logger.debug(f"No data available for metric {metric_name}")
            return None

        # Calculate standard deviation
        values = db.query(column).all()
        values_list = [v[0] for v in values if v[0] is not None]

        if len(values_list) < 2:
            stddev = 0.0
        else:
            mean = result.mean or 0
            variance = sum((x - mean) ** 2 for x in values_list) / len(values_list)
            stddev = math.sqrt(variance)

        # Get latest value
        latest_record = db.query(column).order_by(Prediction.id.desc()).first()
        latest = latest_record[0] if latest_record else 0.0

        # Map metric to unit
        unit_map = {
            "cpu_usage": "%",
            "memory_usage": "%",
            "disk_io": "MB/s",
            "network_traffic": "Mbps"
        }

        return MetricStatistics(
            name=metric_name,
            count=result.count or 0,
            mean=float(result.mean or 0),
            min=float(result.min_val or 0),
            max=float(result.max_val or 0),
            stddev=stddev,
            latest=float(latest or 0),
            unit=unit_map.get(metric_name, "")
        )

    except Exception as e:
        logger.error(f"Failed to calculate statistics for {metric_name}: {e}", exc_info=True)
        return None


# ============================================================================
# API Endpoints
# ============================================================================


@router.post(
    "/validate",
    response_model=MetricsValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate metrics input",
    responses={
        200: {"description": "Metrics validated"},
        400: {"description": "Invalid metrics"},
        500: {"description": "Validation error"}
    }
)
def validate_metrics(
    request: MetricsValidationRequest
) -> MetricsValidationResponse:
    """
    Validate metrics input with comprehensive error detection.

    Checks for:
    - Type validity (numeric values)
    - Range validation (within acceptable bounds)
    - Special values (NaN, Inf)
    - Abnormal values (critically high)

    Args:
        request: Metrics to validate

    Returns:
        MetricsValidationResponse with validation status and details

    """
    try:
        logger.info("Validating metrics input")

        # Validate all metrics
        validated, metadata, warnings, errors = _validate_all_metrics(
            request.cpu_usage,
            request.memory_usage,
            request.disk_io,
            request.network_traffic
        )

        # Determine overall status
        if errors:
            status_val = "error"
        elif warnings:
            status_val = "warning"
        else:
            status_val = "success"

        logger.info(f"Validation complete: status={status_val}, warnings={len(warnings)}, errors={len(errors)}")

        return MetricsValidationResponse(
            status=status_val,
            timestamp=datetime.now(),
            validated_metrics=validated,
            metadata=metadata,
            warnings=warnings,
            errors=errors
        )

    except Exception as e:
        logger.error(f"Metrics validation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        ) from e


@router.get(
    "/summary",
    response_model=MetricsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get metrics summary",
    responses={
        200: {"description": "Summary retrieved"},
        500: {"description": "Database error"}
    }
)
def get_metrics_summary(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    db: Session = Depends(get_db_session)
) -> MetricsSummaryResponse:
    """
    Get comprehensive metrics summary with statistics.

    Retrieves aggregated statistics over specified time period.

    Args:
        hours: Number of hours to analyze (1-168, default 24)
        db: Database session

    Returns:
        MetricsSummaryResponse with statistics and anomaly rates

    """
    try:
        logger.info(f"Generating metrics summary for last {hours} hours")

        # Query recent records
        cutoff_time = datetime.now() - timedelta(hours=hours)
        records = db.query(Prediction).filter(
            Prediction.timestamp >= cutoff_time
        ).all()

        if not records:
            logger.warning(f"No records found in last {hours} hours")
            return MetricsSummaryResponse(
                status="success",
                timestamp=datetime.now(),
                time_period=f"Last {hours} hours",
                total_records=0,
                anomaly_count=0,
                anomaly_percentage=0.0,
                metrics_stats={}
            )

        # Count anomalies
        anomaly_count = sum(1 for r in records if r.prediction.lower() == "anomaly")
        total = len(records)
        anomaly_pct = (anomaly_count / total * 100) if total > 0 else 0

        logger.info(f"Found {total} records, {anomaly_count} anomalies ({anomaly_pct:.1f}%)")

        # Calculate statistics for each metric
        metrics_stats = {}
        for metric_name in ["cpu_usage", "memory_usage", "disk_io", "network_traffic"]:
            stats = _calculate_statistics(db, metric_name)
            if stats:
                metrics_stats[metric_name] = stats

        return MetricsSummaryResponse(
            status="success",
            timestamp=datetime.now(),
            time_period=f"Last {hours} hours",
            total_records=total,
            anomaly_count=anomaly_count,
            anomaly_percentage=round(anomaly_pct, 2),
            metrics_stats=metrics_stats
        )

    except Exception as e:
        logger.error(f"Failed to generate metrics summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summary generation failed: {str(e)}"
        ) from e


@router.get(
    "/aggregates",
    response_model=AggregateMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get aggregated metrics",
    responses={
        200: {"description": "Aggregates retrieved"},
        500: {"description": "Database error"}
    }
)
def get_aggregated_metrics(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    interval: str = Query("hourly", regex="^(hourly|6hourly|daily)$", description="Aggregation interval"),
    db: Session = Depends(get_db_session)
) -> AggregateMetricsResponse:
    """
    Get metrics aggregated over specified time intervals.

    Args:
        hours: Number of hours to analyze (1-168, default 24)
        interval: Aggregation interval (hourly, 6hourly, daily)
        db: Database session

    Returns:
        AggregateMetricsResponse with time-series data

    """
    try:
        logger.info(f"Generating aggregated metrics ({interval}, {hours} hours)")

        # Map interval to timedelta
        interval_map = {
            "hourly": timedelta(hours=1),
            "6hourly": timedelta(hours=6),
            "daily": timedelta(days=1)
        }
        interval_delta = interval_map.get(interval, timedelta(hours=1))

        # Query records
        cutoff_time = datetime.now() - timedelta(hours=hours)
        records = db.query(Prediction).filter(
            Prediction.timestamp >= cutoff_time
        ).order_by(Prediction.timestamp).all()

        if not records:
            logger.warning(f"No records found for aggregation")
            return AggregateMetricsResponse(
                status="success",
                timestamp=datetime.now(),
                interval=interval,
                aggregates=[]
            )

        # Aggregate by interval
        aggregates = []
        current_period = cutoff_time

        while current_period <= datetime.now():
            period_end = current_period + interval_delta
            period_records = [
                r for r in records
                if current_period <= r.timestamp < period_end
            ]

            if period_records:
                cpu_values = [r.cpu_usage for r in period_records if r.cpu_usage is not None]
                mem_values = [r.memory_usage for r in period_records if r.memory_usage is not None]
                disk_values = [r.disk_io for r in period_records if r.disk_io is not None]
                net_values = [r.network_traffic for r in period_records if r.network_traffic is not None]

                aggregate = {
                    "period": current_period.isoformat(),
                    "cpu_mean": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "memory_mean": sum(mem_values) / len(mem_values) if mem_values else 0,
                    "disk_mean": sum(disk_values) / len(disk_values) if disk_values else 0,
                    "network_mean": sum(net_values) / len(net_values) if net_values else 0,
                    "count": len(period_records)
                }
                aggregates.append(aggregate)

            current_period = period_end

        logger.info(f"Generated {len(aggregates)} aggregate periods")

        return AggregateMetricsResponse(
            status="success",
            timestamp=datetime.now(),
            interval=interval,
            aggregates=aggregates
        )

    except ValueError as e:
        logger.warning(f"Invalid parameter: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameter: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Failed to generate aggregates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aggregation failed: {str(e)}"
        ) from e
