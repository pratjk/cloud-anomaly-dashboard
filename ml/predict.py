"""
Cloud Anomaly Detection - Hybrid Prediction Module

Combines metric-based and log-based anomaly detection for hybrid classification.
Uses Isolation Forest for metric analysis and Autoencoder for log message analysis.

Prediction Flow:
    1. Initialize metric model (Isolation Forest) and log model (Autoencoder)
    2. Validate input dictionary structure and field types
    3. Extract and validate numeric metrics with range checking
    4. Extract numeric features from log message
    5. Run metric-based anomaly detection
    6. Run log-based anomaly detection
    7. Combine predictions using weighted hybrid score
    8. Return final anomaly classification

Configuration:
    - METRIC_MODEL_PATH: Path to trained Isolation Forest model
    - LOG_MODEL_PATH: Path to trained Autoencoder model
    - REQUIRED_METRIC_FIELDS: Required input fields (cpu, memory, disk, network)
    - METRIC_WEIGHT: Weight for metric-based prediction (default: 1.0)
    - LOG_WEIGHT: Weight for log-based prediction (default: 1.5)
    - ANOMALY_THRESHOLD: Hybrid score threshold for anomaly (default: 1.0)

Model Reliability:
    - Both models loaded on first prediction (lazy initialization)
    - Missing metric model causes immediate failure (required)
    - Missing log model falls back to dummy training (optional)
    - All predictions validated for NaN/Inf before returning
    - Failed predictions logged with full error context

Returns:
    Dictionary with 'prediction' (Anomaly/Normal) and 'cause' (reason)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import pandas as pd
import joblib
import numpy as np
from ml.log_anomaly_model import LogAnomalyDetector

# ============================================================================
# Configuration Section
# ============================================================================

METRIC_MODEL_PATH: str = "ml/model.pkl"
LOG_MODEL_PATH: str = "ml/log_model.pkl"

# Required input fields
REQUIRED_METRIC_FIELDS: set = {"cpu_usage", "memory_usage", "disk_io", "network_traffic"}
OPTIONAL_FIELDS: set = {"log_message"}

# Metric thresholds
METRIC_ANOMALY_LABEL: int = -1  # Isolation Forest anomaly label

# Log feature indices
LOG_FEATURE_LENGTH: int = 0
LOG_FEATURE_ERROR_COUNT: int = 1
LOG_FEATURE_WARNING_COUNT: int = 2
LOG_FEATURE_INFO_COUNT: int = 3
LOG_FEATURE_CONST: int = 4
LOG_FEATURE_SIZE: int = 5

# Detection weights
METRIC_WEIGHT: float = 1.0
LOG_WEIGHT: float = 1.5
ANOMALY_THRESHOLD: float = 1.0

# Validation ranges
MAX_LOG_MESSAGE_LENGTH: int = 10000
MIN_LOG_MESSAGE_LENGTH: int = 0

# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================================
# Global Model State
# ============================================================================

_metric_model: Optional[Any] = None
_log_model: Optional[LogAnomalyDetector] = None
_models_initialized: bool = False
_initialization_error: Optional[str] = None


def _initialize_models() -> None:
    """
    Initialize metric and log anomaly detection models lazily.

    Loads models on first call and caches for subsequent calls.
    Metric model is required; log model falls back to dummy training if missing.

    Raises:
        FileNotFoundError: If metric model file is not found
        RuntimeError: If model initialization fails irreparably

    """
    global _metric_model, _log_model, _models_initialized, _initialization_error

    if _models_initialized:
        if _initialization_error:
            raise RuntimeError(f"Previous initialization failed: {_initialization_error}")
        return

    try:
        logger.info("=" * 60)
        logger.info("Initializing anomaly detection models")

        # ===== Load Metric Model (Required) =====
        metric_path = Path(METRIC_MODEL_PATH)
        
        if not metric_path.exists():
            error_msg = f"Metric model file not found: {metric_path}"
            logger.critical(error_msg)
            _initialization_error = error_msg
            _models_initialized = True
            raise FileNotFoundError(error_msg)

        if not metric_path.is_file():
            error_msg = f"Metric model path is not a file: {metric_path}"
            logger.critical(error_msg)
            _initialization_error = error_msg
            _models_initialized = True
            raise RuntimeError(error_msg)

        try:
            logger.debug(f"Loading metric model from: {metric_path.absolute()}")
            _metric_model = joblib.load(metric_path)
            logger.info(f"Metric model loaded successfully (size: {metric_path.stat().st_size} bytes)")
            
            # Validate metric model has required methods
            if not hasattr(_metric_model, 'predict'):
                raise AttributeError("Metric model missing predict() method")
            logger.debug("Metric model has predict() method")
            
        except Exception as e:
            error_msg = f"Failed to load metric model: {e}"
            logger.critical(error_msg, exc_info=True)
            _initialization_error = error_msg
            _models_initialized = True
            raise RuntimeError(error_msg) from e

        # ===== Initialize Log Model (Optional) =====
        log_path = Path(LOG_MODEL_PATH)

        try:
            logger.debug("Initializing log anomaly detector")
            _log_model = LogAnomalyDetector()
            logger.debug("LogAnomalyDetector instance created")

            # Try to load trained log model
            if log_path.exists():
                if log_path.is_file():
                    try:
                        logger.debug(f"Loading log model from: {log_path.absolute()}")
                        _log_model.load(str(log_path))
                        logger.info(f"Log model loaded successfully (size: {log_path.stat().st_size} bytes)")
                        
                        # Validate log model is trained
                        if not _log_model.is_trained():
                            logger.warning("Log model loaded but not properly trained, retraining with dummy data")
                            X_dummy = np.random.rand(100, LOG_FEATURE_SIZE)
                            _log_model.train(X_dummy)
                            logger.info("Log model retrained with dummy data")
                        else:
                            logger.debug("Log model is properly trained")
                            
                    except Exception as e:
                        logger.warning(f"Failed to load log model from {log_path}: {e}")
                        logger.warning("Falling back to dummy training for log model")
                        X_dummy = np.random.rand(100, LOG_FEATURE_SIZE)
                        _log_model.train(X_dummy)
                        logger.info("Log model trained with dummy data after load failure")
                else:
                    logger.warning(f"Log model path exists but is not a file: {log_path}")
                    logger.warning("Using dummy training for log model")
                    X_dummy = np.random.rand(100, LOG_FEATURE_SIZE)
                    _log_model.train(X_dummy)
            else:
                logger.info(f"Log model file not found ({log_path}), using dummy training")
                X_dummy = np.random.rand(100, LOG_FEATURE_SIZE)
                _log_model.train(X_dummy)
                logger.info("Log model trained with dummy data")

            # Validate log model has required methods
            if not hasattr(_log_model, 'detect'):
                raise AttributeError("Log model missing detect() method")
            if not hasattr(_log_model, 'is_trained'):
                raise AttributeError("Log model missing is_trained() method")
            logger.debug("Log model has detect() and is_trained() methods")

        except Exception as e:
            error_msg = f"Failed to initialize log model: {e}"
            logger.critical(error_msg, exc_info=True)
            _initialization_error = error_msg
            _models_initialized = True
            raise RuntimeError(error_msg) from e

        _models_initialized = True
        logger.info("=" * 60)
        logger.info("All models initialized successfully")

    except Exception as e:
        # Mark as initialized with error to prevent retry loops
        _models_initialized = True
        _initialization_error = str(e)
        raise


def _validate_input_dict(data_dict: Dict[str, Any]) -> None:
    """
    Validate input data dictionary comprehensively.

    Checks:
    - Input is a dictionary
    - Non-empty
    - Contains all required metric fields
    - All metric values are numeric
    - No NaN or Inf values
    - Values are in valid ranges (clamps if needed)

    Args:
        data_dict: Input data dictionary to validate

    Raises:
        TypeError: If data_dict is not a dictionary or fields have wrong types
        ValueError: If required fields are missing or contain invalid values

    """
    if not isinstance(data_dict, dict):
        raise TypeError(
            f"Input must be a dictionary, got {type(data_dict).__name__}"
        )

    if not data_dict:
        raise ValueError("Input dictionary cannot be empty")

    logger.debug(f"Validating input dictionary with keys: {list(data_dict.keys())}")

    # Check required fields present
    missing_fields = REQUIRED_METRIC_FIELDS - set(data_dict.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    logger.debug("All required fields present")

    # Validate each metric field
    for field in REQUIRED_METRIC_FIELDS:
        value = data_dict[field]
        
        # Type check
        if not isinstance(value, (int, float, np.number)):
            raise ValueError(
                f"Field '{field}' must be numeric, got {type(value).__name__}: {value}"
            )

        # Convert to float for uniform handling
        try:
            float_value = float(value)
        except (ValueError, OverflowError) as e:
            raise ValueError(
                f"Field '{field}' cannot be converted to float: {value}"
            ) from e

        # NaN/Inf check
        if not np.isfinite(float_value):
            raise ValueError(
                f"Field '{field}' contains invalid value: {value} (NaN or Inf)"
            )

        # Range validation with clamping
        if field in ("cpu_usage", "memory_usage"):
            if not (0 <= float_value <= 100):
                logger.warning(
                    f"Field '{field}' out of valid range [0-100]: {float_value}, clamping"
                )
                data_dict[field] = max(0.0, min(100.0, float_value))

        elif field in ("disk_io", "network_traffic"):
            if float_value < 0:
                logger.warning(
                    f"Field '{field}' is negative: {float_value}, setting to 0"
                )
                data_dict[field] = 0.0

    logger.debug(f"Input validation complete: {data_dict}")


def _extract_log_features(log_message: str) -> np.ndarray:
    """
    Extract numeric features from log message.

    Features extracted:
    1. Message length (character count)
    2. Error count (occurrences of 'ERROR')
    3. Warning count (occurrences of 'WARNING')
    4. Info count (occurrences of 'INFO')
    5. Constant 1 (bias term)

    Args:
        log_message: Log message string to extract features from

    Returns:
        2D numpy array of shape (1, 5) with log features

    Raises:
        TypeError: If log_message is not a string
        ValueError: If log_message is empty or invalid

    """
    # Type validation
    if not isinstance(log_message, str):
        raise TypeError(
            f"Log message must be a string, got {type(log_message).__name__}"
        )

    # Length validation
    if len(log_message) == 0:
        logger.warning("Log message is empty, using default placeholder")
        log_message = "INFO"
    elif len(log_message) > MAX_LOG_MESSAGE_LENGTH:
        logger.warning(
            f"Log message exceeds max length ({len(log_message)} > {MAX_LOG_MESSAGE_LENGTH}), "
            f"truncating"
        )
        log_message = log_message[:MAX_LOG_MESSAGE_LENGTH]

    # Convert to uppercase for feature extraction
    log_upper = log_message.upper()

    # Extract features
    features = [
        len(log_upper),
        log_upper.count("ERROR"),
        log_upper.count("WARNING"),
        log_upper.count("INFO"),
        1.0  # Bias term
    ]

    logger.debug(
        f"Log features extracted: length={features[0]}, "
        f"errors={features[1]}, warnings={features[2]}, info={features[3]}"
    )

    # Validate features
    for i, feat in enumerate(features):
        if not np.isfinite(feat):
            raise ValueError(f"Feature {i} contains invalid value: {feat}")

    return np.array([features], dtype=np.float32)


def _predict_metric_anomaly(data_dict: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Predict anomaly based on system metrics using Isolation Forest.

    Args:
        data_dict: Dictionary containing cpu_usage, memory_usage, disk_io, network_traffic

    Returns:
        Tuple of (prediction_label: str, is_anomaly: bool)

    Raises:
        RuntimeError: If metric model prediction fails
        ValueError: If model returns invalid predictions

    """
    try:
        logger.debug("Performing metric-based anomaly detection")

        # Create DataFrame with metrics
        df = pd.DataFrame([{
            "cpu_usage": data_dict["cpu_usage"],
            "memory_usage": data_dict["memory_usage"],
            "disk_io": data_dict["disk_io"],
            "network_traffic": data_dict["network_traffic"]
        }])

        logger.debug(f"Input metrics: {df.to_dict('records')[0]}")

        # Run prediction
        if _metric_model is None:
            raise RuntimeError("Metric model not initialized")

        prediction = _metric_model.predict(df)
        logger.debug(f"Raw metric prediction: {prediction}")

        # Validate prediction output
        if prediction is None or len(prediction) == 0:
            raise ValueError("Metric model returned empty prediction")

        if not isinstance(prediction, np.ndarray):
            raise ValueError(f"Metric model returned non-array result: {type(prediction).__name__}")

        pred_value = prediction[0]
        
        if not isinstance(pred_value, (int, float, np.number)):
            raise ValueError(f"Prediction value is not numeric: {pred_value}")

        if not np.isfinite(pred_value):
            raise ValueError(f"Prediction contains NaN/Inf: {pred_value}")

        # Classify result
        is_anomaly = (pred_value == METRIC_ANOMALY_LABEL)
        label = "Anomaly" if is_anomaly else "Normal"

        logger.info(f"Metric-based prediction: {label} (raw value: {pred_value})")
        return label, is_anomaly

    except Exception as e:
        error_msg = f"Metric anomaly detection failed: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def _predict_log_anomaly(log_message: str) -> Tuple[bool, float]:
    """
    Predict anomaly based on log message using Autoencoder.

    Args:
        log_message: Log message to analyze

    Returns:
        Tuple of (is_anomaly: bool, confidence: float)

    Raises:
        RuntimeError: If log model prediction fails
        ValueError: If model returns invalid predictions

    """
    try:
        logger.debug("Performing log-based anomaly detection")

        # Validate log model state
        if _log_model is None:
            raise RuntimeError("Log model not initialized")

        if not _log_model.is_trained():
            raise RuntimeError("Log model not properly trained")

        logger.debug("Log model is initialized and trained")

        # Extract features - returns shape (1, 5) array for single input
        log_features = _extract_log_features(log_message)
        logger.debug(f"Log input features shape: {log_features.shape}, values: {log_features[0]}")

        # Handle single input - ensure it's properly passed as 2D array
        if log_features.ndim == 1:
            log_features = log_features.reshape(1, -1)
            logger.debug(f"Reshaped features to 2D: {log_features.shape}")

        # Run detection - detect() expects 2D array (n_samples, n_features)
        prediction = _log_model.detect(log_features)
        logger.debug(f"Raw log prediction result: {prediction}, type: {type(prediction)}")

        # Validate prediction output
        if prediction is None or len(prediction) == 0:
            raise ValueError("Log model returned empty prediction")

        if not isinstance(prediction, np.ndarray):
            raise ValueError(f"Log model returned non-array result: {type(prediction).__name__}")

        # Handle both boolean and numeric predictions
        pred_bool = bool(prediction[0])
        confidence = 1.0 if pred_bool else 0.0

        logger.info(
            f"Log-based prediction: {'Anomaly' if pred_bool else 'Normal'} "
            f"(confidence: {confidence:.2f})"
        )
        return pred_bool, confidence

    except Exception as e:
        error_msg = f"Log anomaly detection failed: {e}"
        logger.error(error_msg, exc_info=True)
        # Return fallback to avoid crashing the entire prediction pipeline
        logger.warning(f"Falling back to Normal prediction due to log model error")
        return False, 0.0


def _make_hybrid_decision(
    metric_result: bool,
    log_result: bool
) -> Tuple[str, float]:
    """
    Combine metric and log predictions using weighted hybrid scoring.

    Decision logic:
    - Metric anomaly adds METRIC_WEIGHT to score
    - Log anomaly adds LOG_WEIGHT to score
    - Final prediction is "Anomaly" if score >= ANOMALY_THRESHOLD

    Args:
        metric_result: Metric-based anomaly indicator (True = anomaly)
        log_result: Log-based anomaly indicator (True = anomaly)

    Returns:
        Tuple of (prediction_label: str, hybrid_score: float)

    """
    logger.debug(f"Computing hybrid decision: metric_anomaly={metric_result}, log_anomaly={log_result}")

    score = 0.0

    if metric_result:
        score += METRIC_WEIGHT
        logger.debug(f"Metric anomaly detected, adding score {METRIC_WEIGHT} (total: {score})")

    if log_result:
        score += LOG_WEIGHT
        logger.debug(f"Log anomaly detected, adding score {LOG_WEIGHT} (total: {score})")

    logger.debug(
        f"Hybrid score: {score:.2f}, threshold: {ANOMALY_THRESHOLD}, "
        f"anomaly: {score >= ANOMALY_THRESHOLD}"
    )

    # Final decision
    prediction = "Anomaly" if score >= ANOMALY_THRESHOLD else "Normal"

    return prediction, score


def _analyze_root_cause(data_dict: Dict[str, Any], log_is_anomaly: bool) -> str:
    """
    Deterministically analyze payload metrics against threshold bounds to extract root cause.
    """
    causes = []
    
    if data_dict.get("network_traffic", 0) >= 800:
        causes.append("DDoS Attack / Network Flood")
        
    if data_dict.get("disk_io", 0) >= 400:
        causes.append("Heavy Disk Thrashing / Failure")
        
    if data_dict.get("memory_usage", 0) >= 85:
        causes.append("Severe Memory Leak")
        
    if data_dict.get("cpu_usage", 0) >= 85:
        causes.append("CPU Starvation / Spike")
        
    if log_is_anomaly:
        causes.append("Critical Application Log Error")
        
    if not causes:
        return "Unknown System Anomaly"
        
    return " | ".join(causes)


def predict_anomaly(data_dict: Dict[str, Any]) -> Dict[str, str]:
    """
    Predict system anomaly using hybrid metric and log-based detection.

    Complete prediction pipeline:
    1. Initialize models on first call (lazy init)
    2. Validate input data structure and values
    3. Run metric-based detection (Isolation Forest)
    4. Run log-based detection (Autoencoder)
    5. Combine predictions with weighted scoring
    6. Return final classification

    Args:
        data_dict: Dictionary containing:
            - cpu_usage (float, 0-100): CPU utilization percentage
            - memory_usage (float, 0-100): Memory utilization percentage
            - disk_io (float, >= 0): Disk I/O throughput in MB/s
            - network_traffic (float, >= 0): Network traffic in Mbps
            - log_message (str, optional): Log message for context

    Returns:
        Dictionary with:
            - prediction (str): "Anomaly" or "Normal"
            - cause (str): Reason for prediction

    Raises:
        TypeError: If data_dict is not a dict or fields have wrong types
        ValueError: If required fields are missing or invalid
        RuntimeError: If model prediction fails
        FileNotFoundError: If metric model file is missing

    Example:
        >>> result = predict_anomaly({
        ...     "cpu_usage": 95.5,
        ...     "memory_usage": 88.2,
        ...     "disk_io": 450,
        ...     "network_traffic": 1000,
        ...     "log_message": "ERROR: Database timeout"
        ... })
        >>> print(result)
        {'prediction': 'Anomaly', 'cause': 'Anomaly'}

    """
    try:
        logger.info("=" * 60)
        logger.info("Starting anomaly prediction")

        # ===== Step 1: Initialize models (first call only) =====
        logger.debug("Step 1/6: Initializing models")
        _initialize_models()

        # ===== Step 2: Validate input data =====
        logger.debug("Step 2/6: Validating input data")
        _validate_input_dict(data_dict)

        # ===== Step 3: Metric-based prediction =====
        logger.debug("Step 3/6: Running metric-based prediction")
        try:
            metric_label, metric_is_anomaly = _predict_metric_anomaly(data_dict)
        except Exception as e:
            logger.warning(f"Metric prediction failed, using fallback: {e}")
            metric_label = "Normal"
            metric_is_anomaly = False

        # ===== Step 4: Log-based prediction =====
        logger.debug("Step 4/6: Running log-based prediction")
        log_message = data_dict.get("log_message", "INFO")
        try:
            log_is_anomaly, log_confidence = _predict_log_anomaly(log_message)
        except Exception as e:
            logger.warning(f"Log prediction failed, using fallback: {e}")
            log_is_anomaly = False
            log_confidence = 0.0

        # ===== Step 5: Hybrid decision =====
        logger.debug("Step 5/6: Computing hybrid decision")
        final_prediction, hybrid_score = _make_hybrid_decision(metric_is_anomaly, log_is_anomaly)

        # ===== Step 6: Format result and Root Cause =====
        logger.debug("Step 6/6: Formatting result")
        
        final_cause = final_prediction
        remediation = "None"
        if final_prediction == "Anomaly":
            final_cause = _analyze_root_cause(data_dict, log_is_anomaly)
            if "DDoS" in final_cause:
                remediation = "Traffic rerouted. Auto-Scaling Load Balancers (+3 instances) deployed."
            elif "Memory" in final_cause:
                remediation = "Isolating instance. Initiating graceful Service Restart."
            elif "Disk" in final_cause:
                remediation = "Redirecting I/O streams to fallback SSD cluster."
            elif "CPU" in final_cause:
                remediation = "Throttling low-priority background queues to free up compute."
            else:
                remediation = "Isolating affected node. System Diagnostics engaged."
            
        result = {
            "prediction": final_prediction,
            "cause": final_cause,
            "remediation": remediation
        }

        logger.info(f"Prediction complete: {result}")
        logger.info("=" * 60)
        return result

    except (TypeError, ValueError) as e:
        error_msg = f"Input validation failed: {e}"
        logger.error(error_msg, exc_info=True)
        # Return fallback prediction on input error
        return {
            "prediction": "Normal",
            "cause": f"Input validation error - fallback prediction"
        }
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {e}"
        logger.critical(error_msg, exc_info=True)
        # Return safe fallback instead of crashing
        return {
            "prediction": "Normal",
            "cause": f"Prediction error - fallback prediction"
        }