# log anomaly model - autoencoder approach using MLPRegressor
# basically trains a neural net to reconstruct normal data,
# if reconstruction error is high = anomaly

import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

MODEL_PATH = "ml/log_model.pkl"
HIDDEN_LAYERS = (16, 8, 16)
MAX_ITER = 500
# 99th percentile pe threshold rakhte hain - iske upar anomaly
ANOMALY_PERCENTILE = 99

logger = logging.getLogger(__name__)


class LogAnomalyDetector:
    """
    autoencoder for detecting weird patterns in log data.
    trains by learning to reconstruct normal logs, then flags
    anything with high reconstruction error as anomalous.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=HIDDEN_LAYERS,
            max_iter=MAX_ITER,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self._trained = False

    def train(self, X):
        # X should be a 2D numpy array
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("need a 2D numpy array for training")

        if X.shape[0] < 2:
            # duplicate data if we dont have enough samples
            # hacky but it works for now
            X = np.tile(X, (3, 1))

        X_scaled = self.scaler.fit_transform(X)

        # autoencoder trick: train to reconstruct its own input
        self.model.fit(X_scaled, X_scaled)
        self._trained = True
        logger.info(f"log model trained on {X.shape[0]} samples")

    def detect(self, X):
        if not self._trained:
            raise RuntimeError("model train nahi hua hai abhi, pehle train() call karo")

        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled)

        # reconstruction error = how badly the model failed to reconstruct
        error = np.mean((X_scaled - reconstructed) ** 2, axis=1)

        # anything above 99th percentile is suspicious
        threshold = np.percentile(error, ANOMALY_PERCENTILE)
        return error > threshold

    def save(self, path=None):
        if not self._trained:
            raise RuntimeError("cant save untrained model")

        save_path = Path(path or MODEL_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.scaler, self.model), save_path)
        # ye line debug karte waqt 2 ghante lagi thi samajhne mein ki tuple save karna padta hai
        logger.info(f"model saved to {save_path}")

    def load(self, path=None):
        load_path = Path(path or MODEL_PATH)
        if not load_path.exists():
            raise FileNotFoundError(f"model file nahi mila: {load_path}")

        loaded = joblib.load(load_path)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise IOError("model file ka format galat hai")

        self.scaler, self.model = loaded
        self._trained = True

    def is_trained(self):
        return self._trained