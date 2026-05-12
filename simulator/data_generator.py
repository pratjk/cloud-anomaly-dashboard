"""
Cloud Anomaly Detection - Synthetic Dataset Generator

Generates synthetic cloud infrastructure metrics with random anomalies for model training.
Creates a CSV file with timestamps, server IDs, metrics, and anomaly flags.

Module Flow:
    1. Validate configuration parameters (record count, anomaly rate)
    2. Initialize data collection structures
    3. Generate normal behavior metrics for each record
    4. Randomly inject anomalies (high CPU, memory, disk, network)
    5. Build DataFrame with all metrics
    6. Save to CSV with error handling

Configuration:
    - NUM_RECORDS: Number of synthetic records to generate
    - ANOMALY_INJECTION_RATE: Probability of injecting anomaly (0.0-1.0)
    - NORMAL_CPU_MEAN, NORMAL_CPU_STD: Normal CPU metrics (mean, std dev)
    - NORMAL_MEMORY_MEAN, NORMAL_MEMORY_STD: Normal memory metrics
    - NORMAL_DISK_MEAN, NORMAL_DISK_STD: Normal disk I/O metrics
    - NORMAL_NETWORK_MEAN, NORMAL_NETWORK_STD: Normal network metrics
    - ANOMALY_CPU_RANGE, ANOMALY_MEMORY_RANGE: Anomalous metric ranges
    - ANOMALY_DISK_RANGE, ANOMALY_NETWORK_RANGE: Anomalous metric ranges
    - OUTPUT_PATH: Path to save generated dataset CSV
    - RANDOM_SEED: Seed for reproducible randomness (None for non-deterministic)
    - SERVER_ID_RANGE: Range of server IDs to simulate
    - DECIMAL_PRECISION: Decimal places for rounding metric values

Returns:
    DataFrame saved to CSV with columns:
    - timestamp: Timestamp of each record
    - server_id: Simulated server identifier
    - cpu_usage: CPU utilization percentage
    - memory_usage: Memory utilization percentage
    - disk_io: Disk I/O throughput (MB/s)
    - network_traffic: Network traffic (Mbps)
    - anomaly_flag: 0 for normal, 1 for anomalous
"""

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# Configuration Section
# ============================================================================

NUM_RECORDS: int = 2000
ANOMALY_INJECTION_RATE: float = 0.05
RANDOM_SEED: int = 42

# Normal behavior thresholds (mean, std deviation)
NORMAL_CPU_MEAN: float = 50.0
NORMAL_CPU_STD: float = 10.0
NORMAL_MEMORY_MEAN: float = 60.0
NORMAL_MEMORY_STD: float = 8.0
NORMAL_DISK_MEAN: float = 100.0
NORMAL_DISK_STD: float = 20.0
NORMAL_NETWORK_MEAN: float = 300.0
NORMAL_NETWORK_STD: float = 50.0

# Anomaly ranges (min, max)
ANOMALY_CPU_RANGE: Tuple[float, float] = (85.0, 100.0)
ANOMALY_MEMORY_RANGE: Tuple[float, float] = (85.0, 100.0)
ANOMALY_DISK_RANGE: Tuple[float, float] = (300.0, 500.0)
ANOMALY_NETWORK_RANGE: Tuple[float, float] = (800.0, 1200.0)

SERVER_ID_RANGE: Tuple[int, int] = (1, 5)
DECIMAL_PRECISION: int = 2
OUTPUT_PATH: str = "data/sample_dataset.csv"

# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_parameters(
    num_records: int,
    anomaly_rate: float,
    random_seed: int | None = None
) -> None:
    """
    Validate dataset generation parameters before processing.

    Args:
        num_records: Number of records to generate (must be > 0)
        anomaly_rate: Anomaly injection probability (0.0 to 1.0)
        random_seed: Random seed value (None for non-deterministic)

    Raises:
        ValueError: If parameters are invalid

    """
    if not isinstance(num_records, int) or num_records <= 0:
        raise ValueError(
            f"num_records must be positive integer, got {num_records}"
        )

    if not isinstance(anomaly_rate, (int, float)) or not (0.0 <= anomaly_rate <= 1.0):
        raise ValueError(
            f"anomaly_rate must be between 0.0 and 1.0, got {anomaly_rate}"
        )

    if random_seed is not None and not isinstance(random_seed, int):
        raise ValueError(f"random_seed must be integer or None, got {random_seed}")

    logger.debug(
        f"Parameters validated: num_records={num_records}, "
        f"anomaly_rate={anomaly_rate}, random_seed={random_seed}"
    )


def _initialize_random_state(seed: int | None = None) -> None:
    """
    Initialize random number generators for reproducibility.

    Args:
        seed: Random seed (None for non-deterministic behavior)

    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed initialized to {seed}")
    else:
        logger.debug("Using non-deterministic random state")


def _generate_normal_metrics() -> Tuple[float, float, float, float]:
    """
    Generate normal (non-anomalous) metric values.

    Returns:
        Tuple of (cpu_usage, memory_usage, disk_io, network_traffic)

    """
    cpu = np.random.normal(NORMAL_CPU_MEAN, NORMAL_CPU_STD)
    memory = np.random.normal(NORMAL_MEMORY_MEAN, NORMAL_MEMORY_STD)
    disk = np.random.normal(NORMAL_DISK_MEAN, NORMAL_DISK_STD)
    network = np.random.normal(NORMAL_NETWORK_MEAN, NORMAL_NETWORK_STD)

    return (
        round(float(cpu), DECIMAL_PRECISION),
        round(float(memory), DECIMAL_PRECISION),
        round(float(disk), DECIMAL_PRECISION),
        round(float(network), DECIMAL_PRECISION),
    )


def _generate_anomalous_metrics() -> Tuple[float, float, float, float]:
    """
    Generate anomalous metric values (simulating system stress).

    Returns:
        Tuple of (cpu_usage, memory_usage, disk_io, network_traffic)

    """
    cpu = np.random.uniform(ANOMALY_CPU_RANGE[0], ANOMALY_CPU_RANGE[1])
    memory = np.random.uniform(ANOMALY_MEMORY_RANGE[0], ANOMALY_MEMORY_RANGE[1])
    disk = np.random.uniform(ANOMALY_DISK_RANGE[0], ANOMALY_DISK_RANGE[1])
    network = np.random.uniform(ANOMALY_NETWORK_RANGE[0], ANOMALY_NETWORK_RANGE[1])

    return (
        round(float(cpu), DECIMAL_PRECISION),
        round(float(memory), DECIMAL_PRECISION),
        round(float(disk), DECIMAL_PRECISION),
        round(float(network), DECIMAL_PRECISION),
    )


def _generate_records(
    num_records: int,
    anomaly_rate: float,
    start_time: datetime
) -> List[Tuple]:
    """
    Generate synthetic dataset records with metrics and anomaly flags.

    Args:
        num_records: Number of records to generate
        anomaly_rate: Probability of injecting anomaly per record
        start_time: Timestamp of first record

    Returns:
        List of tuples containing all metrics for each record

    """
    logger.info(f"Step 1/3: Generating {num_records} records")
    records = []

    for i in range(num_records):
        timestamp = start_time + timedelta(seconds=i)
        server_id = random.randint(SERVER_ID_RANGE[0], SERVER_ID_RANGE[1])

        # Determine if this record is anomalous
        is_anomaly = random.random() < anomaly_rate
        anomaly_flag = 1 if is_anomaly else 0

        # Generate appropriate metrics
        if is_anomaly:
            cpu, memory, disk, network = _generate_anomalous_metrics()
        else:
            cpu, memory, disk, network = _generate_normal_metrics()

        records.append((timestamp, server_id, cpu, memory, disk, network, anomaly_flag))

    logger.info(f"Generated {num_records} records "
                f"({int(num_records * anomaly_rate)} anomalies)")
    return records


def _create_dataframe(records: List[Tuple]) -> pd.DataFrame:
    """
    Convert record list to pandas DataFrame with proper schema.

    Args:
        records: List of tuples containing metric values

    Returns:
        DataFrame with validated columns and schema

    Raises:
        ValueError: If records list is empty

    """
    if not records:
        raise ValueError("Records list is empty, cannot create DataFrame")

    logger.info("Step 2/3: Creating DataFrame from records")

    df = pd.DataFrame(
        records,
        columns=[
            "timestamp",
            "server_id",
            "cpu_usage",
            "memory_usage",
            "disk_io",
            "network_traffic",
            "anomaly_flag",
        ],
    )

    # Validate DataFrame integrity
    if df.isnull().any().any():
        raise ValueError("DataFrame contains NaN values")

    if not all(col in df.columns for col in ["timestamp", "server_id", "cpu_usage"]):
        raise ValueError("DataFrame missing required columns")

    logger.debug(f"DataFrame created: {len(df)} rows, {len(df.columns)} columns")
    return df


def _save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to CSV with error handling and validation.

    Args:
        df: DataFrame to save
        output_path: Path where CSV should be saved

    Raises:
        IOError: If file write fails
        ValueError: If DataFrame is invalid

    """
    if df is None or df.empty:
        raise ValueError("Cannot save empty DataFrame")

    try:
        logger.info("Step 3/3: Saving dataset to CSV")
        output_file = Path(output_path)

        # Create parent directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_file, index=False)

        logger.info(f"Dataset saved successfully: {output_path}")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Anomalies: {df['anomaly_flag'].sum()} "
                f"({100*df['anomaly_flag'].mean():.1f}%)")

    except IOError as e:
        error_msg = f"Failed to write CSV to {output_path}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error saving dataset: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# ============================================================================
# Main Function
# ============================================================================

def generate_synthetic_dataset(
    num_records: int = NUM_RECORDS,
    anomaly_rate: float = ANOMALY_INJECTION_RATE,
    output_path: str = OUTPUT_PATH,
    random_seed: int | None = RANDOM_SEED,
) -> bool:
    """
    Generate synthetic cloud metrics dataset with anomaly injection.

    Complete workflow:
    1. Validate all input parameters
    2. Initialize random state for reproducibility
    3. Generate metric records with timestamps
    4. Create DataFrame with proper schema
    5. Save to CSV file

    Args:
        num_records: Number of records to generate (default: 2000)
        anomaly_rate: Anomaly injection rate 0.0-1.0 (default: 0.05)
        output_path: Output CSV file path (default: data/sample_dataset.csv)
        random_seed: Random seed for reproducibility (default: 42, None for random)

    Returns:
        True if dataset successfully generated and saved
        False if generation failed

    """
    try:
        logger.info("=" * 60)
        logger.info("Starting synthetic dataset generation")

        # Step 1: Validate parameters
        _validate_parameters(num_records, anomaly_rate, random_seed)

        # Step 2: Initialize random state
        _initialize_random_state(random_seed)

        # Step 3: Generate records
        start_time = datetime.now()
        records = _generate_records(num_records, anomaly_rate, start_time)

        # Step 4: Create DataFrame
        df = _create_dataframe(records)

        # Step 5: Save dataset
        _save_dataset(df, output_path)

        logger.info("=" * 60)
        logger.info("Dataset generation completed successfully")
        return True

    except (ValueError, IOError) as e:
        logger.error(f"Dataset generation failed: {e}")
        return False
    except RuntimeError as e:
        logger.error(f"Unexpected error during generation: {e}")
        return False
    except Exception as e:
        logger.error(f"Unknown error occurred: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    try:
        success = generate_synthetic_dataset()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Dataset generation interrupted by user")
        exit(130)
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        exit(1)
