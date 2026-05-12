"""
Model Retraining Module
Periodically retrains anomaly detection model on collected prediction data.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# ================== Configuration ==================
DATABASE_PATH = "cloud.db"
MODEL_OUTPUT_PATH = "ml/model.pkl"

# Model file paths (relative to module directory)
MODEL_VERSION_FILE = "model_version.txt"
LAST_RETRAIN_FILE = "last_retrain.txt"

# Model parameters
MIN_TRAINING_SAMPLES = 50
ISOLATION_FOREST_CONTAMINATION = 0.15
RANDOM_STATE = 42

# Feature columns
FEATURE_COLUMNS = ["cpu_usage", "memory_usage", "disk_io", "network_traffic"]
SELECT_QUERY = "SELECT * FROM predictions"

# ================== Logging Setup ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_module_dir() -> Path:
    """
    Get the absolute path to the module directory.
    
    Returns:
        Path object pointing to the ml/ directory
    """
    return Path(__file__).parent.absolute()


def _fetch_training_data(database_path: str) -> Optional[pd.DataFrame]:
    """
    Fetch prediction data from SQLite database.
    
    Args:
        database_path: Path to SQLite database file
        
    Returns:
        DataFrame with prediction records or None if fetch fails
        
    Raises:
        FileNotFoundError: If database file does not exist
        sqlite3.OperationalError: If query execution fails
    """
    try:
        db_path = Path(database_path)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")
        
        logger.debug(f"Connecting to database: {database_path}")
        
        with sqlite3.connect(database_path) as conn:
            logger.debug(f"Executing query: {SELECT_QUERY}")
            df = pd.read_sql_query(SELECT_QUERY, conn)
        
        logger.info(f"Successfully fetched {len(df)} records from database")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"Database file not found: {e}")
        raise
    except sqlite3.OperationalError as e:
        logger.error(f"Database query failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch training data: {e}", exc_info=True)
        raise RuntimeError(f"Database fetch failed: {str(e)}") from e


def _validate_training_data(df: pd.DataFrame) -> None:
    """
    Validate training data has sufficient samples and required columns.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If data is insufficient or columns are missing
        TypeError: If df is not a DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    
    if len(df) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Insufficient training data. Have {len(df)} samples, need {MIN_TRAINING_SAMPLES}"
        )
    
    missing_columns = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.debug(f"Data validation passed: {len(df)} samples, {len(df.columns)} columns")


def _train_isolation_forest(X: pd.DataFrame) -> IsolationForest:
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        X: Feature matrix with shape (n_samples, n_features)
        
    Returns:
        Trained IsolationForest model
        
    Raises:
        RuntimeError: If model training fails
    """
    try:
        logger.info(f"Training Isolation Forest on {len(X)} samples, {X.shape[1]} features")
        logger.debug(f"Model parameters: contamination={ISOLATION_FOREST_CONTAMINATION}, "
                    f"random_state={RANDOM_STATE}")
        
        model = IsolationForest(
            contamination=ISOLATION_FOREST_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1  # Use all available processors
        )
        
        model.fit(X)
        logger.info("Model training completed successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to train model: {str(e)}") from e


def _save_model(model: IsolationForest, output_path: str) -> None:
    """
    Save trained model to disk.
    
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


def _read_version_file(version_file: Path) -> int:
    """
    Read current model version from file.
    
    Args:
        version_file: Path to version file
        
    Returns:
        Current version number (0 if file doesn't exist)
    """
    try:
        if not version_file.exists():
            logger.debug(f"Version file not found: {version_file}. Starting at version 0.")
            return 0
        
        with open(version_file, "r") as f:
            content = f.read().strip()
            if not content.isdigit():
                logger.warning(f"Invalid version format in {version_file}: '{content}'. Resetting to 0.")
                return 0
            return int(content)
            
    except Exception as e:
        logger.warning(f"Failed to read version file: {e}. Resetting to 0.")
        return 0


def _update_version_file(version_file: Path, current_version: int) -> int:
    """
    Increment and save model version.
    
    Args:
        version_file: Path to version file
        current_version: Current version number
        
    Returns:
        New version number
        
    Raises:
        IOError: If version file write fails
    """
    try:
        new_version = current_version + 1
        
        # Create parent directory if needed
        version_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Writing version {new_version} to {version_file}")
        with open(version_file, "w") as f:
            f.write(str(new_version))
        
        logger.info(f"Model version updated: {current_version} -> {new_version}")
        return new_version
        
    except Exception as e:
        logger.error(f"Failed to update version file: {e}", exc_info=True)
        raise IOError(f"Version file update failed: {str(e)}") from e


def _save_retrain_timestamp(time_file: Path) -> None:
    """
    Save current timestamp of model retraining.
    
    Args:
        time_file: Path to timestamp file
        
    Raises:
        IOError: If timestamp file write fails
    """
    try:
        # Create parent directory if needed
        time_file.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        logger.debug(f"Writing timestamp {timestamp} to {time_file}")
        
        with open(time_file, "w") as f:
            f.write(timestamp)
        
        logger.info(f"Retrain timestamp saved: {timestamp}")
        
    except Exception as e:
        logger.error(f"Failed to save retrain timestamp: {e}", exc_info=True)
        raise IOError(f"Timestamp file save failed: {str(e)}") from e


def retrain_model() -> Tuple[bool, str]:
    """
    Retrain the anomaly detection model with collected data.
    
    Workflow:
    1. Fetch prediction records from database
    2. Validate data has sufficient samples
    3. Extract features and train Isolation Forest
    4. Save trained model to disk
    5. Update version and timestamp files
    
    Returns:
        Tuple of (success: bool, message: str) indicating result
        
    Example:
        >>> success, message = retrain_model()
        >>> if success:
        ...     print(f"Retraining completed: {message}")
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting model retraining cycle")
        logger.info("=" * 60)
        
        # Fetch training data
        logger.info("Step 1/5: Fetching training data")
        df = _fetch_training_data(DATABASE_PATH)
        
        # Validate data
        logger.info("Step 2/5: Validating data")
        _validate_training_data(df)
        
        # Extract features
        logger.info("Step 3/5: Extracting features")
        X = df[FEATURE_COLUMNS].copy()
        logger.debug(f"Feature matrix shape: {X.shape}")
        
        # Train model
        logger.info("Step 4/5: Training model")
        model = _train_isolation_forest(X)
        
        # Save model
        logger.info("Step 5a/5: Saving model")
        _save_model(model, MODEL_OUTPUT_PATH)
        
        # Update metadata
        logger.info("Step 5b/5: Updating metadata (version and timestamp)")
        module_dir = _get_module_dir()
        
        version_file = module_dir / MODEL_VERSION_FILE
        current_version = _read_version_file(version_file)
        new_version = _update_version_file(version_file, current_version)
        
        time_file = module_dir / LAST_RETRAIN_FILE
        _save_retrain_timestamp(time_file)
        
        # Success
        message = f"Model retrained successfully! Version: {new_version}"
        logger.info(message)
        logger.info("=" * 60)
        
        return True, message
        
    except ValueError as e:
        message = f"Data validation failed: {e}"
        logger.warning(message)
        logger.info("Retraining skipped - insufficient data")
        return False, message
        
    except (FileNotFoundError, IOError) as e:
        message = f"File operation failed: {e}"
        logger.error(message)
        return False, message
        
    except RuntimeError as e:
        message = f"Model training failed: {e}"
        logger.error(message)
        return False, message
        
    except Exception as e:
        message = f"Unexpected error during retraining: {e}"
        logger.error(message, exc_info=True)
        return False, message