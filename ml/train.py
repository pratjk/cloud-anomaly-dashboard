"""
Model Training Module
Trains initial Isolation Forest model for anomaly detection on system metrics.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# ================== Configuration ==================
DATASET_PATH = "data/sample_dataset.csv"
MODEL_OUTPUT_PATH = "ml/model.pkl"

# Feature configuration
REQUIRED_FEATURES = ["cpu_usage", "memory_usage", "disk_io", "network_traffic"]

# Model parameters
MODEL_CONTAMINATION = 0.05  # Expected contamination rate (5% anomalies)
RANDOM_STATE = 42
MODEL_N_JOBS = -1  # Use all available processors

# Data validation
MINIMUM_SAMPLES = 10

# ================== Logging Setup ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _validate_dataset_file(dataset_path: str) -> None:
    """
    Validate that the dataset file exists.
    
    Args:
        dataset_path: Path to the dataset CSV file
        
    Raises:
        FileNotFoundError: If dataset file does not exist
        ValueError: If path is empty or invalid
    """
    if not dataset_path:
        raise ValueError("Dataset path cannot be empty")
    
    file_path = Path(dataset_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path.absolute()}"
        )
    
    if not file_path.suffix.lower() == '.csv':
        logger.warning(f"Dataset file has unexpected extension: {file_path.suffix}")
    
    logger.debug(f"Dataset file validated: {file_path}")


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        dataset_path: Path to the dataset CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be read or is empty
        RuntimeError: If data loading fails
    """
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        
        _validate_dataset_file(dataset_path)
        
        df = pd.read_csv(dataset_path)
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file: {e}")
        raise RuntimeError(f"CSV parsing failed: {str(e)}") from e
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        raise RuntimeError(f"Dataset loading failed: {str(e)}") from e


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract required numeric features from dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame containing only the required features
        
    Raises:
        ValueError: If required features are missing
        TypeError: If df is not a DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    
    try:
        logger.debug(f"Extracting features: {REQUIRED_FEATURES}")
        
        missing_features = set(REQUIRED_FEATURES) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Available features: {set(df.columns)}"
            )
        
        X = df[REQUIRED_FEATURES].copy()
        logger.debug(f"Extracted features shape: {X.shape}")
        
        return X
        
    except ValueError as e:
        logger.error(f"Feature extraction failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during feature extraction: {e}", exc_info=True)
        raise RuntimeError(f"Feature extraction failed: {str(e)}") from e


def _validate_features(X: pd.DataFrame) -> None:
    """
    Validate feature data for training.
    
    Args:
        X: Feature matrix to validate
        
    Raises:
        ValueError: If data is invalid
        TypeError: If X is not a DataFrame
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(X).__name__}")
    
    # Check number of samples
    if len(X) < MINIMUM_SAMPLES:
        raise ValueError(
            f"Insufficient training samples. Have {len(X)}, need {MINIMUM_SAMPLES}"
        )
    
    # Check for NaN values
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Missing values (NaN) found in columns {nan_cols}: {nan_count} missing values total. "
            "Please clean the data before training."
        )
    
    # Check for non-numeric values
    for col in X.columns:
        try:
            pd.to_numeric(X[col])
        except (ValueError, TypeError):
            raise ValueError(f"Column '{col}' contains non-numeric values. All features must be numeric.")
    
    # Check for infinite values
    inf_mask = X.applymap(lambda x: pd.isna(x) or (isinstance(x, (int, float)) and pd.isnull(x)))
    has_inf = False
    for col in X.columns:
        if np.isinf(X[col]).any():
            has_inf = True
            logger.warning(f"Infinite values detected in column '{col}'")
    
    if has_inf:
        raise ValueError(
            "Infinite values (Inf) detected in features. "
            "Please clean the data before training."
        )
    
    logger.debug(f"Feature validation passed: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Data ready for training: {X.shape[0]} records, {X.shape[1]} features")


def _create_model() -> IsolationForest:
    """
    Create and configure the Isolation Forest model.
    
    Returns:
        Configured IsolationForest model (not yet trained)
    """
    try:
        logger.debug(
            f"Creating Isolation Forest model: contamination={MODEL_CONTAMINATION}, "
            f"random_state={RANDOM_STATE}, n_jobs={MODEL_N_JOBS}"
        )
        
        model = IsolationForest(
            contamination=MODEL_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=MODEL_N_JOBS
        )
        
        logger.info("Isolation Forest model created successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}", exc_info=True)
        raise RuntimeError(f"Model creation failed: {str(e)}") from e


def _train_model(model: IsolationForest, X: pd.DataFrame) -> IsolationForest:
    """
    Train the Isolation Forest model.
    
    Args:
        model: IsolationForest model to train
        X: Feature matrix for training
        
    Returns:
        Trained model
        
    Raises:
        RuntimeError: If training fails
    """
    try:
        logger.info(f"Training model on {len(X)} samples with {X.shape[1]} features")
        
        model.fit(X)
        
        logger.info("Model training completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        raise RuntimeError(f"Training failed: {str(e)}") from e


def _save_model(model: IsolationForest, output_path: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained IsolationForest model
        output_path: Path where model should be saved
        
    Raises:
        IOError: If model save fails
    """
    try:
        out_path = Path(output_path)
        
        # Create parent directory if needed
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Saving model to {output_path}")
        joblib.dump(model, output_path)
        logger.info(f"Model saved successfully to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save model to {output_path}: {e}", exc_info=True)
        raise IOError(f"Model save failed: {str(e)}") from e


def train_initial_model() -> bool:
    """
    Train initial Isolation Forest model from sample dataset.
    
    Workflow:
    1. Load dataset from CSV
    2. Extract required features
    3. Validate feature data
    4. Create and train model
    5. Save trained model to disk
    
    Returns:
        True if training succeeded, False otherwise
        
    Example:
        >>> if train_initial_model():
        ...     print("Model trained successfully")
        ... else:
        ...     print("Model training failed")
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting initial model training")
        logger.info("=" * 60)
        
        # Step 1: Load dataset
        logger.info("Step 1/5: Loading dataset from " + DATASET_PATH)
        try:
            df = _load_dataset(DATASET_PATH)
            logger.info(f"  ✓ Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        except FileNotFoundError as e:
            logger.error(f"  ✗ Dataset file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"  ✗ Failed to load dataset: {e}")
            return False
        
        # Step 2: Extract features
        logger.info("Step 2/5: Extracting required features")
        try:
            X = _extract_features(df)
            logger.info(f"  ✓ Features extracted: {X.shape[0]} rows x {X.shape[1]} features")
        except ValueError as e:
            logger.error(f"  ✗ Feature extraction failed: {e}")
            return False
        except Exception as e:
            logger.error(f"  ✗ Unexpected error during feature extraction: {e}")
            return False
        
        # Step 3: Validate features
        logger.info("Step 3/5: Validating feature data")
        try:
            _validate_features(X)
            logger.info(f"  ✓ Data validation passed")
        except ValueError as e:
            logger.error(f"  ✗ Data validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"  ✗ Unexpected error during validation: {e}")
            return False
        
        # Step 4: Create model
        logger.info("Step 4/5: Creating Isolation Forest model")
        try:
            model = _create_model()
            logger.info(f"  ✓ Model created (contamination={MODEL_CONTAMINATION})")
        except Exception as e:
            logger.error(f"  ✗ Failed to create model: {e}")
            return False
        
        # Step 5a: Train model
        logger.info("Step 5a/5: Training model")
        try:
            model = _train_model(model, X)
            logger.info(f"  ✓ Model training completed")
        except Exception as e:
            logger.error(f"  ✗ Model training failed: {e}")
            return False
        
        # Step 5b: Save model
        logger.info("Step 5b/5: Saving trained model")
        try:
            _save_model(model, MODEL_OUTPUT_PATH)
            logger.info(f"  ✓ Model saved to {MODEL_OUTPUT_PATH}")
        except IOError as e:
            logger.error(f"  ✗ Failed to save model: {e}")
            return False
        except Exception as e:
            logger.error(f"  ✗ Unexpected error during save: {e}")
            return False
        
        message = "✓ Model trained and saved successfully!"
        logger.info(message)
        logger.info("=" * 60)
        
        return True
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Validation failed: {e}")
        return False
    except (RuntimeError, IOError) as e:
        logger.error(f"Operation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    try:
        success = train_initial_model()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        exit(130)
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        exit(1)
