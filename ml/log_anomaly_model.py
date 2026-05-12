"""
Log Anomaly Detection Model
Detects anomalies in log data using neural network-based autoencoder approach.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

# ================== Configuration ==================
DEFAULT_MODEL_PATH = "ml/log_model.pkl"
HIDDEN_LAYER_SIZES = (16, 8, 16)
MAX_ITERATIONS = 500
ANOMALY_PERCENTILE_THRESHOLD = 99
MINIMUM_SAMPLES = 2
MINIMUM_FEATURES = 1

# ================== Logging Setup ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogAnomalyDetector:
    """
    Detects anomalies in log data using autoencoder neural network.
    
    The model learns to reconstruct normal log patterns. High reconstruction
    error indicates anomalous behavior. Uses MSE-based threshold at 99th percentile.
    """

    def __init__(self) -> None:
        """Initialize the anomaly detector with scaler and neural network model."""
        try:
            self.scaler = StandardScaler()
            self.model = MLPRegressor(
                hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                max_iter=MAX_ITERATIONS,
                random_state=42,
                n_iter_no_change=10,
                early_stopping=True,
                validation_fraction=0.1
            )
            self._is_trained = False
            logger.info("LogAnomalyDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LogAnomalyDetector: {e}", exc_info=True)
            raise

    def _validate_input_data(self, X: np.ndarray, operation: str) -> None:
        """
        Validate input data array.
        
        Args:
            X: Input data array to validate
            operation: Name of operation (e.g., 'train', 'detect')
            
        Raises:
            TypeError: If X is not a numpy array
            ValueError: If X has invalid shape or contains NaN/Inf values
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{operation}: Input must be a numpy array, got {type(X).__name__}")
        
        if X.ndim != 2:
            raise ValueError(f"{operation}: Expected 2D array, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        
        # For single sample detection, allow at least 1 sample
        # For training, require minimum samples
        min_samples = MINIMUM_SAMPLES if operation == 'train' else 1
        
        if n_samples < min_samples:
            raise ValueError(f"{operation}: Need at least {min_samples} samples, got {n_samples}")
        
        if n_features < MINIMUM_FEATURES:
            raise ValueError(f"{operation}: Need at least {MINIMUM_FEATURES} features, got {n_features}")
        
        if not np.isfinite(X).all():
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            raise ValueError(
                f"{operation}: Input contains NaN ({nan_count}) or Inf ({inf_count}) values. "
                "Please clean the data."
            )

    def train(self, X: np.ndarray) -> None:
        """
        Train the autoencoder model on log data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Raises:
            TypeError: If X is not a numpy array
            ValueError: If X has invalid shape or contains NaN/Inf
            RuntimeError: If training fails
        """
        try:
            logger.info(f"Starting training with data shape {X.shape}")
            
            # Validate input
            logger.debug("Validating input data")
            self._validate_input_data(X, "train")
            
            # Handle insufficient samples by duplicating input
            if X.shape[0] < MINIMUM_SAMPLES:
                logger.warning(
                    f"Insufficient training samples ({X.shape[0]} < {MINIMUM_SAMPLES}), "
                    f"duplicating data to meet minimum requirement"
                )
                # Duplicate rows to reach minimum samples
                repetitions = (MINIMUM_SAMPLES // X.shape[0]) + 1
                X = np.tile(X, (repetitions, 1))
                X = X[:MINIMUM_SAMPLES]  # Trim to exact minimum
                logger.info(f"Data duplicated to shape {X.shape}")
            
            # Scale the data
            logger.debug(f"Scaling {X.shape[0]} training samples")
            X_scaled = self.scaler.fit_transform(X)
            logger.debug(f"Scaled data range: min={X_scaled.min():.6f}, max={X_scaled.max():.6f}")
            
            # Train the model (autoencoder: reconstruct input)
            logger.debug("Initiating neural network training (autoencoder mode: input reconstruction)")
            self.model.fit(X_scaled, X_scaled)
            logger.debug(f"Training completed. Model iterations: {self.model.n_iter_}")
            
            self._is_trained = True
            logger.info(f"Training completed successfully. Model trained on {X.shape[0]} samples with {X.shape[1]} features")
            
        except (TypeError, ValueError) as e:
            logger.error(f"Validation failed during training: {e}")
            self._is_trained = False
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self._is_trained = False
            raise RuntimeError(f"Failed to train model: {str(e)}") from e

    def detect(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in log data.
        
        Args:
            X: Data to analyze of shape (n_samples, n_features)
            
        Returns:
            Boolean array indicating anomalies (True = anomalous, False = normal)
            
        Raises:
            TypeError: If X is not a numpy array
            ValueError: If X has invalid shape or contains NaN/Inf
            RuntimeError: If model is not trained or prediction fails
        """
        try:
            logger.debug(f"Starting anomaly detection with data shape {X.shape}")
            
            # Validate model is trained
            if not self._is_trained:
                raise RuntimeError("Model must be trained before detection. Call train() first.")
            logger.debug("Model training status verified")
            
            # Validate input
            logger.debug("Validating input data")
            self._validate_input_data(X, "detect")
            
            # Scale the data using training scaler
            logger.debug(f"Transforming {X.shape[0]} samples using fitted scaler")
            try:
                X_scaled = self.scaler.transform(X)
                logger.debug(f"Scaled data range: min={X_scaled.min():.6f}, max={X_scaled.max():.6f}")
            except Exception as e:
                logger.error(f"Scaling transformation failed: {e}", exc_info=True)
                raise RuntimeError(f"Failed to scale input data: {str(e)}") from e
            
            # Get reconstruction
            logger.debug("Predicting reconstruction from neural network")
            try:
                reconstructed = self.model.predict(X_scaled)
                logger.debug(f"Reconstruction output shape: {reconstructed.shape}")
                if not np.isfinite(reconstructed).all():
                    nan_count = np.isnan(reconstructed).sum()
                    inf_count = np.isinf(reconstructed).sum()
                    raise ValueError(f"Reconstruction contains NaN ({nan_count}) or Inf ({inf_count}) values")
            except Exception as e:
                logger.error(f"Model prediction failed: {e}", exc_info=True)
                raise RuntimeError(f"Failed to predict reconstruction: {str(e)}") from e
            
            # Calculate reconstruction error (MSE per sample)
            logger.debug("Calculating reconstruction error (MSE)")
            try:
                error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
                if not np.isfinite(error).all():
                    nan_count = np.isnan(error).sum()
                    inf_count = np.isinf(error).sum()
                    logger.warning(f"Error array contains NaN ({nan_count}) or Inf ({inf_count}) values")
                    error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
                logger.debug(f"Error statistics - min: {error.min():.6f}, max: {error.max():.6f}, mean: {error.mean():.6f}")
            except Exception as e:
                logger.error(f"Error calculation failed: {e}", exc_info=True)
                raise RuntimeError(f"Failed to calculate reconstruction error: {str(e)}") from e
            
            # Determine threshold from error distribution
            logger.debug(f"Determining anomaly threshold at {ANOMALY_PERCENTILE_THRESHOLD}th percentile")
            try:
                threshold = np.percentile(error, ANOMALY_PERCENTILE_THRESHOLD)
                if not np.isfinite(threshold):
                    raise ValueError(f"Threshold is not finite: {threshold}")
                logger.debug(f"Anomaly threshold determined: {threshold:.6f}")
            except Exception as e:
                logger.error(f"Threshold calculation failed: {e}", exc_info=True)
                raise RuntimeError(f"Failed to determine anomaly threshold: {str(e)}") from e
            
            # Return boolean anomaly indicators
            logger.debug("Computing anomaly predictions")
            anomalies = error > threshold
            anomaly_count = anomalies.sum()
            logger.info(f"Detected {anomaly_count} anomalies out of {len(X)} samples "
                       f"({100*anomaly_count/len(X):.2f}%) - threshold: {threshold:.6f}")
            
            return anomalies
            
        except (TypeError, ValueError) as e:
            logger.error(f"Validation failed during detection: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Detection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Anomaly detection failed with unexpected error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to detect anomalies: {str(e)}") from e

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the trained model and scaler to disk.
        
        Args:
            path: File path for saving. Defaults to DEFAULT_MODEL_PATH
            
        Raises:
            RuntimeError: If model is not trained
            IOError: If file save operation fails
        """
        try:
            # Verify model is trained
            if not self._is_trained:
                raise RuntimeError("Cannot save untrained model. Call train() first.")
            logger.debug("Model training status verified for save operation")
            
            # Validate scaler and model exist
            if self.scaler is None or self.model is None:
                raise RuntimeError("Model components (scaler, model) are None. Model may be corrupted.")
            logger.debug("Model components validated - scaler and model are not None")
            
            save_path = Path(path or DEFAULT_MODEL_PATH)
            logger.debug(f"Target save path: {save_path}")
            
            # Ensure parent directory exists
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Parent directory ensured: {save_path.parent}")
            except Exception as e:
                logger.error(f"Failed to create parent directory {save_path.parent}: {e}", exc_info=True)
                raise IOError(f"Cannot create save directory: {str(e)}") from e
            
            logger.info(f"Saving model and scaler to {save_path}")
            try:
                joblib.dump((self.scaler, self.model), save_path)
                logger.debug(f"Joblib dump completed to {save_path}")
            except Exception as e:
                logger.error(f"Joblib serialization failed: {e}", exc_info=True)
                raise IOError(f"Failed to serialize model: {str(e)}") from e
            
            # Verify file was created and has content
            if not save_path.exists():
                raise IOError(f"Save file was not created: {save_path}")
            
            file_size = save_path.stat().st_size
            if file_size == 0:
                raise IOError(f"Save file is empty (0 bytes): {save_path}")
            
            logger.info(f"Model saved successfully to {save_path} ({file_size} bytes)")
            
        except RuntimeError as e:
            logger.error(f"Model state error: {e}")
            raise
        except IOError as e:
            logger.error(f"File I/O error during save: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}", exc_info=True)
            raise IOError(f"Failed to save model: {str(e)}") from e

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a trained model and scaler from disk.
        
        Args:
            path: File path to load from. Defaults to DEFAULT_MODEL_PATH
            
        Raises:
            FileNotFoundError: If model file does not exist
            IOError: If file load operation fails
        """
        try:
            load_path = Path(path or DEFAULT_MODEL_PATH)
            logger.debug(f"Attempting to load model from: {load_path}")
            
            # Check file existence
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")
            logger.debug(f"Model file found: {load_path} ({load_path.stat().st_size} bytes)")
            
            # Load model and scaler
            logger.info(f"Loading model from {load_path}")
            try:
                loaded_objects = joblib.load(load_path)
                logger.debug(f"Joblib load completed, received {len(loaded_objects) if isinstance(loaded_objects, tuple) else 1} object(s)")
            except Exception as e:
                logger.error(f"Joblib deserialization failed: {e}", exc_info=True)
                raise IOError(f"Failed to deserialize model file: {str(e)}") from e
            
            # Validate loaded objects
            if not isinstance(loaded_objects, tuple) or len(loaded_objects) != 2:
                raise IOError(f"Invalid model file format: expected tuple of 2 objects, got {type(loaded_objects).__name__}")
            logger.debug("Loaded objects format validated (tuple of 2)")
            
            loaded_scaler, loaded_model = loaded_objects
            
            # Validate scaler
            if loaded_scaler is None:
                raise IOError("Loaded scaler is None - model file may be corrupted")
            if not hasattr(loaded_scaler, 'transform'):
                raise IOError(f"Loaded scaler is invalid: missing 'transform' method. Type: {type(loaded_scaler).__name__}")
            logger.debug(f"Scaler validated - type: {type(loaded_scaler).__name__}")
            
            # Validate model
            if loaded_model is None:
                raise IOError("Loaded model is None - model file may be corrupted")
            if not hasattr(loaded_model, 'predict'):
                raise IOError(f"Loaded model is invalid: missing 'predict' method. Type: {type(loaded_model).__name__}")
            logger.debug(f"Model validated - type: {type(loaded_model).__name__}")
            
            # Assign loaded components
            self.scaler = loaded_scaler
            self.model = loaded_model
            self._is_trained = True
            logger.info(f"Model loaded successfully from {load_path}. Scaler: {type(loaded_scaler).__name__}, Model: {type(loaded_model).__name__}")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            self._is_trained = False
            raise
        except IOError as e:
            logger.error(f"File I/O error during load: {e}")
            self._is_trained = False
            raise
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}", exc_info=True)
            self._is_trained = False
            raise IOError(f"Failed to load model: {str(e)}") from e

    def is_trained(self) -> bool:
        """
        Check if the model has been trained.
        
        Returns:
            True if model has been trained, False otherwise
        """
        return self._is_trained