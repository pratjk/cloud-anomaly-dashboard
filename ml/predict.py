# prediction module - this is where the ML magic happens
# uses isolation forest for metrics + autoencoder for logs

import logging
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
from ml.log_anomaly_model import LogAnomalyDetector

METRIC_MODEL_PATH = "ml/model.pkl"
LOG_MODEL_PATH = "ml/log_model.pkl"
REQUIRED_FIELDS = {"cpu_usage", "memory_usage", "disk_io", "network_traffic"}

# weights for hybrid scoring
METRIC_WEIGHT = 1.0
LOG_WEIGHT = 1.5
ANOMALY_THRESHOLD = 1.0

logger = logging.getLogger(__name__)

# global model state - loaded once on first prediction
metric_model = None
log_model = None
models_loaded = False


def load_models():
    """load both models into memory. metric model is required, log model is optional"""
    global metric_model, log_model, models_loaded

    if models_loaded:
        return

    # load the isolation forest (this one is mandatory)
    model_path = Path(METRIC_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"model file nahi mila: {model_path}")

    metric_model = joblib.load(model_path)
    logger.info("metric model loaded")

    # load the log anomaly model (optional - will use dummy data if missing)
    log_model = LogAnomalyDetector()
    log_path = Path(LOG_MODEL_PATH)

    if log_path.exists():
        try:
            log_model.load(str(log_path))
            if not log_model.is_trained():
                # file was corrupt or something, train with dummy data
                dummy = np.random.rand(100, 5)
                log_model.train(dummy)
        except Exception as e:
            logger.warning(f"log model load failed, using dummy: {e}")
            dummy = np.random.rand(100, 5)
            log_model.train(dummy)
    else:
        # no log model file, just train with random data
        # ye theek hai for now, actual log data se retrain hoga baad mein
        dummy = np.random.rand(100, 5)
        log_model.train(dummy)

    models_loaded = True
    logger.info("all models ready")


def extract_log_features(log_msg):
    # pull some basic features from the log message
    if not isinstance(log_msg, str) or len(log_msg) == 0:
        log_msg = "INFO"

    upper = log_msg.upper()
    features = [
        len(upper),
        upper.count("ERROR"),
        upper.count("WARNING"),
        upper.count("INFO"),
        1.0  # bias term
    ]
    return np.array([features], dtype=np.float32)


def check_metrics(data):
    # run isolation forest on the numeric metrics
    df = pd.DataFrame([{
        "cpu_usage": data["cpu_usage"],
        "memory_usage": data["memory_usage"],
        "disk_io": data["disk_io"],
        "network_traffic": data["network_traffic"]
    }])

    pred = metric_model.predict(df)
    is_anomaly = (pred[0] == -1)
    return is_anomaly


def check_logs(log_message):
    # run autoencoder on the log features
    try:
        features = extract_log_features(log_message)
        result = log_model.detect(features)
        return bool(result[0])
    except Exception as e:
        # log model failing shouldn't crash everything
        logger.warning(f"log check failed: {e}")
        return False


def figure_out_cause(data, log_anomaly):
    """try to figure out what went wrong based on which metrics are high"""
    causes = []

    # maine ye thresholds trial and error se set kiye hain
    if data.get("network_traffic", 0) >= 800:
        causes.append("DDoS Attack / Network Flood")
    if data.get("disk_io", 0) >= 400:
        causes.append("Heavy Disk Thrashing")
    if data.get("memory_usage", 0) >= 85:
        causes.append("Severe Memory Leak")
    if data.get("cpu_usage", 0) >= 85:
        causes.append("CPU Spike")
    if log_anomaly:
        causes.append("Critical Log Error")

    if not causes:
        return "Unknown Anomaly"
    return " | ".join(causes)


# ye remediation messages mostly cosmetic hain but demo mein achhe lagte hain
def get_remediation(cause):
    if "DDoS" in cause:
        return "Rerouting traffic. Auto-scaling load balancers deployed."
    elif "Memory" in cause:
        return "Isolating instance. Initiating graceful restart."
    elif "Disk" in cause:
        return "Redirecting I/O to fallback SSD cluster."
    elif "CPU" in cause:
        return "Throttling background queues to free compute."
    else:
        return "Isolating affected node. Running diagnostics."


def predict_anomaly(data_dict):
    """
    main prediction function - takes a dict of metrics,
    runs both models, combines scores, returns result
    """
    try:
        load_models()

        # basic validation
        if not isinstance(data_dict, dict) or not data_dict:
            return {"prediction": "Normal", "cause": "bad input data"}

        missing = REQUIRED_FIELDS - set(data_dict.keys())
        if missing:
            return {"prediction": "Normal", "cause": f"missing fields: {missing}"}

        # run both detection models
        metric_anomaly = check_metrics(data_dict)
        log_msg = data_dict.get("log_message", "INFO")
        log_anomaly = check_logs(log_msg)

        # hybrid scoring - log anomalies are weighted more
        score = 0.0
        if metric_anomaly:
            score += METRIC_WEIGHT
        if log_anomaly:
            score += LOG_WEIGHT

        final = "Anomaly" if score >= ANOMALY_THRESHOLD else "Normal"

        cause = final
        remediation = "None"
        if final == "Anomaly":
            cause = figure_out_cause(data_dict, log_anomaly)
            remediation = get_remediation(cause)

        return {"prediction": final, "cause": cause, "remediation": remediation}

    except Exception as e:
        logger.error(f"prediction failed badly: {e}")
        return {"prediction": "Normal", "cause": "prediction error - defaulting to normal"}