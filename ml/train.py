# train the initial isolation forest model from sample data

import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

DATASET_PATH = "data/sample_dataset.csv"
MODEL_OUTPUT = "ml/model.pkl"
FEATURES = ["cpu_usage", "memory_usage", "disk_io", "network_traffic"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_initial_model():
    """load sample data, train isolation forest, save the model"""

    # load dataset
    if not Path(DATASET_PATH).exists():
        logger.error(f"dataset nahi mila: {DATASET_PATH}")
        return False

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        logger.error("dataset is empty bro")
        return False

    logger.info(f"loaded {len(df)} rows from {DATASET_PATH}")

    # check if all required columns exist
    missing = set(FEATURES) - set(df.columns)
    if missing:
        logger.error(f"missing columns: {missing}")
        return False

    X = df[FEATURES]

    # check for NaN - ye bahut baar issue hota hai
    if X.isna().any().any():
        logger.error("data mein NaN values hain, pehle clean karo")
        return False

    if len(X) < 10:
        logger.error("not enough data to train (need at least 10 rows)")
        return False

    # train the model
    # contamination=0.05 means we expect ~5% of data to be anomalies
    model = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    logger.info("model training done")

    # save it
    out_path = Path(MODEL_OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    logger.info(f"model saved to {MODEL_OUTPUT}")

    return True


if __name__ == "__main__":
    success = train_initial_model()
    if success:
        print("model trained successfully!")
    else:
        print("training failed :(")
        exit(1)
