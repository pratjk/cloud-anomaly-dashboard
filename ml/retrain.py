# retrain the model using data collected from predictions

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

DB_PATH = "cloud.db"
MODEL_PATH = "ml/model.pkl"
VERSION_FILE = "model_version.txt"
RETRAIN_FILE = "last_retrain.txt"
FEATURES = ["cpu_usage", "memory_usage", "disk_io", "network_traffic"]
MIN_SAMPLES = 50

logger = logging.getLogger(__name__)


def get_ml_dir():
    return Path(__file__).parent.absolute()


def retrain_model():
    """
    grabs all prediction data from the db, retrains the isolation forest,
    and saves the updated model. also bumps the version number.
    """
    try:
        # fetch data from sqlite
        if not Path(DB_PATH).exists():
            logger.error(f"database nahi mila: {DB_PATH}")
            return False, "database not found"

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()

        if len(df) < MIN_SAMPLES:
            msg = f"not enough data yet ({len(df)} rows, need {MIN_SAMPLES})"
            logger.info(msg)
            return False, msg

        # check columns
        missing = set(FEATURES) - set(df.columns)
        if missing:
            return False, f"missing columns: {missing}"

        X = df[FEATURES].copy()
        logger.info(f"retraining on {len(X)} samples")

        # train new model
        # contamination thodi zyada rakhte hain retrain mein (0.15)
        # kyunki ab actual anomalies bhi hain data mein
        model = IsolationForest(
            contamination=0.15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)

        # save model
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        # bump version
        ml_dir = get_ml_dir()
        ver_file = ml_dir / VERSION_FILE
        current = 0
        if ver_file.exists():
            txt = ver_file.read_text().strip()
            if txt.isdigit():
                current = int(txt)

        new_ver = current + 1
        ver_file.write_text(str(new_ver))

        # save retrain timestamp
        time_file = ml_dir / RETRAIN_FILE
        time_file.write_text(datetime.now().isoformat())

        logger.info(f"retrain complete! version: v{new_ver}")
        return True, f"retrained to v{new_ver}"

    except Exception as e:
        logger.error(f"retrain failed: {e}")
        return False, str(e)