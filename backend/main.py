# main backend file - FastAPI server for anomaly detection

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, get_db_session, init_db
from database.models import Prediction, QuarantinedDevice
from ml.predict import predict_anomaly
from ml.retrain import retrain_model

# config
RETRAIN_EVERY = int(os.getenv("MODEL_RETRAIN_INTERVAL", "50"))
VERSION_FILE = os.getenv("MODEL_VERSION_FILE", "ml/model_version.txt")

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Cloud Anomaly Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# setup db on startup
try:
    init_db()
    logger.info("DB ready")
except Exception as e:
    logger.error(f"DB init failed: {e}")
    raise


# --- helper functions ---

def validate_request(data):
    if not isinstance(data, dict) or not data:
        raise ValueError("request body should be a non-empty JSON object")

def extract_metrics(data):
    """pull metrics out of request data and do basic sanity checks"""
    try:
        cpu = float(data.get("cpu_usage", 0))
        mem = float(data.get("memory_usage", 0))
        disk = float(data.get("disk_io", 0))
        net = float(data.get("network_traffic", 0))
    except (ValueError, TypeError) as e:
        raise ValueError(f"metrics should be numbers yaar: {e}")

    # clamp values to valid ranges
    cpu = max(0, min(100, cpu))
    mem = max(0, min(100, mem))
    disk = max(0, disk)
    net = max(0, net)

    return {"cpu_usage": cpu, "memory_usage": mem, "disk_io": disk, "network_traffic": net}

def run_prediction(data):
    try:
        result = predict_anomaly(data)
        if isinstance(result, dict):
            pred = result.get("prediction", "Normal")
            cause = result.get("cause", "Model decision")
            remediation = result.get("remediation", "None")
        else:
            pred = str(result)
            cause = "Model decision"
            remediation = "None"
        return pred, cause, remediation
    except Exception as e:
        logger.error(f"prediction failed: {e}")
        return "Error", f"Model Error: {e}", "None"

def save_prediction(db, device_id, metrics, prediction, cause, remediation, lat, lon):
    # shove the prediction into the database
    try:
        record = Prediction(
            device_id=device_id,
            cpu_usage=metrics["cpu_usage"],
            memory_usage=metrics["memory_usage"],
            disk_io=metrics["disk_io"],
            network_traffic=metrics["network_traffic"],
            prediction=prediction,
            cause=cause,
            remediation=remediation,
            latitude=lat,
            longitude=lon
        )
        db.add(record)
        db.commit()
        return True
    except Exception as e:
        logger.error(f"failed to save prediction: {e}")
        db.rollback()
        return False

def maybe_retrain(db):
    # check if we have enough new data to retrain
    total = db.query(Prediction).count()
    if total == 0 or total % RETRAIN_EVERY != 0:
        return

    # mujhe lagta hai version check zaroori hai idhar
    try:
        if not os.path.exists(VERSION_FILE):
            retrain_model()
            return

        with open(VERSION_FILE, "r") as f:
            ver = f.read().strip()

        if not ver.isdigit():
            retrain_model()
            return

        current_ver = int(ver)
        expected_ver = total // RETRAIN_EVERY
        if current_ver < expected_ver:
            logger.info(f"model outdated (v{current_ver}), retraining...")
            retrain_model()
    except Exception as e:
        logger.error(f"retrain check failed: {e}")


# --- API routes ---

@app.get("/quarantine")
def list_quarantine(db: Session = Depends(get_db_session)):
    try:
        devices = db.query(QuarantinedDevice).all()
        return [d.to_dict() for d in devices]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quarantine/release/{device_id}")
def release_device(device_id: str, db: Session = Depends(get_db_session)):
    device = db.query(QuarantinedDevice).filter(
        QuarantinedDevice.device_id == device_id
    ).first()
    if not device:
        raise HTTPException(status_code=404, detail=f"device {device_id} nahi mila quarantine mein")

    db.delete(device)
    db.commit()
    return {"status": "success", "message": f"device {device_id} released"}


@app.post("/predict")
def predict(data: dict, db: Session = Depends(get_db_session)):
    """main prediction endpoint - takes cloud metrics and returns anomaly prediction"""
    try:
        validate_request(data)
        device_id = data.get("device_id", "Unknown")

        # check if device is quarantined
        jailed = db.query(QuarantinedDevice).filter(
            QuarantinedDevice.device_id == device_id
        ).first()
        if jailed:
            raise HTTPException(
                status_code=403,
                detail=f"device {device_id} is quarantined - request blocked"
            )

        metrics = extract_metrics(data)
        log_msg = data.get("log_message", "INFO Normal operation")
        prediction, cause, remediation = run_prediction(data)

        # save to db
        stored = False
        try:
            conn = SessionLocal()
            stored = save_prediction(
                conn, device_id, metrics, prediction, cause, remediation,
                data.get("latitude", 0.0), data.get("longitude", 0.0)
            )

            # auto-quarantine for critical stuff
            if prediction == "Anomaly":
                bad_stuff = ["DDoS Attack", "Severe Memory Leak", "Network Flood"]
                if any(c in cause for c in bad_stuff):
                    logger.warning(f"quarantining device {device_id} - {cause}")
                    existing = conn.query(QuarantinedDevice).filter(
                        QuarantinedDevice.device_id == device_id
                    ).first()
                    if not existing:
                        conn.add(QuarantinedDevice(device_id=device_id, reason=cause))
                        conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"db error: {e}")

        # check if model needs retraining
        if stored:
            try:
                conn = SessionLocal()
                maybe_retrain(conn)
                conn.close()
            except Exception as e:
                pass  # nahi hua toh theek hai, next time hoga

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
        logger.error(f"prediction endpoint crashed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db_session)):
    try:
        records = db.query(Prediction).all()
        preds = [r.to_dict() for r in records]
        return {"status": "success", "count": len(preds), "predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    # quick health check
    db_ok = False
    try:
        conn = SessionLocal()
        count = conn.query(Prediction).count()
        conn.close()
        db_ok = True
        db_msg = f"connected - {count} records"
    except Exception as e:
        db_msg = f"connection failed: {e}"

    model_exists = os.path.exists("ml/model.pkl")

    status = "healthy" if db_ok and model_exists else "degraded"
    return {
        "status": status,
        "database": {"ok": db_ok, "info": db_msg},
        "model_available": model_exists,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
def root():
    return {"message": "Cloud Anomaly Detection API Running", "status": "ok"}