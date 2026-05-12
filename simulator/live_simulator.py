"""
Cloud Anomaly Detection - Live Simulator

Continuously sends synthetic cloud metrics to the anomaly detection API.
Simulates real-time metric streams with random variations and optional anomalies.

Module Flow:
    1. Validate API endpoint and connection
    2. Generate random metric data within configured ranges
    3. Send POST request to prediction API with error handling
    4. Log response and metrics for monitoring
    5. Sleep for configured interval
    6. Repeat indefinitely with graceful shutdown

Configuration:
    - API_URL: Endpoint for prediction API (default: http://127.0.0.1:8000/predict)
    - API_TIMEOUT: Request timeout in seconds
    - POLL_INTERVAL: Time between requests (seconds)
    - REQUEST_RETRIES: Number of retry attempts on failure
    - RETRY_BACKOFF: Exponential backoff multiplier for retries
    - ANOMALY_INJECTION_RATE: Probability of injecting anomaly (0.0-1.0)

Metric Ranges:
    - CPU usage: 30-100% (normal), 85-100% (anomalous)
    - Memory usage: 40-100% (normal), 85-100% (anomalous)
    - Disk I/O: 80-500 MB/s (normal), 300-500 MB/s (anomalous)
    - Network traffic: 200-1200 Mbps (normal), 800-1200 Mbps (anomalous)

Returns:
    Continuously runs until KeyboardInterrupt with exit code:
    - 0: Normal shutdown
    - 130: User interruption (Ctrl+C)
    - 1: Critical error or connection failure
"""

import json
import logging
import random
import time
from typing import Dict, Any, Tuple, Optional

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration Section
# ============================================================================

API_URL: str = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
API_TIMEOUT: float = 10.0
POLL_INTERVAL: float = 3.0

# Retry configuration
REQUEST_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0

# Anomaly injection
ANOMALY_INJECTION_RATE: float = 0.35  # 35% of requests will be anomalous

# Normal metric ranges (min, max)  — kept lower so IF can distinguish from anomalous
CPU_USAGE_RANGE: Tuple[int, int] = (10, 65)
MEMORY_USAGE_RANGE: Tuple[int, int] = (20, 70)
DISK_IO_RANGE: Tuple[int, int] = (20, 150)
NETWORK_TRAFFIC_RANGE: Tuple[int, int] = (50, 400)

# Anomalous metric ranges
ANOMALY_CPU_RANGE: Tuple[int, int] = (85, 100)
ANOMALY_MEMORY_RANGE: Tuple[int, int] = (85, 100)
ANOMALY_DISK_RANGE: Tuple[int, int] = (300, 500)
ANOMALY_NETWORK_RANGE: Tuple[int, int] = (800, 1200)

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

def _validate_api_url(url: str) -> None:
    """
    Validate that API URL is properly formatted.

    Args:
        url: API endpoint URL

    Raises:
        ValueError: If URL format is invalid

    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"API_URL must be non-empty string, got {url}")

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"API_URL must start with http:// or https://, got {url}")

    logger.debug(f"API URL validated: {url}")


def _validate_config(
    timeout: float,
    poll_interval: float,
    max_retries: int,
    anomaly_rate: float
) -> None:
    """
    Validate simulator configuration parameters.

    Args:
        timeout: Request timeout in seconds (must be > 0)
        poll_interval: Time between requests in seconds (must be >= 0)
        max_retries: Number of retry attempts (must be >= 0)
        anomaly_rate: Anomaly injection rate (0.0 to 1.0)

    Raises:
        ValueError: If any parameter is invalid

    """
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    if not isinstance(poll_interval, (int, float)) or poll_interval < 0:
        raise ValueError(f"poll_interval must be non-negative, got {poll_interval}")

    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError(f"max_retries must be non-negative integer, got {max_retries}")

    if not isinstance(anomaly_rate, (int, float)) or not (0.0 <= anomaly_rate <= 1.0):
        raise ValueError(f"anomaly_rate must be 0.0-1.0, got {anomaly_rate}")

    logger.debug(
        f"Config validated: timeout={timeout}s, "
        f"poll_interval={poll_interval}s, max_retries={max_retries}, "
        f"anomaly_rate={anomaly_rate}"
    )


def _generate_normal_metrics() -> Dict[str, int]:
    """
    Generate normal (non-anomalous) metric values.

    Returns:
        Dictionary with cpu_usage, memory_usage, disk_io, network_traffic

    """
    return {
        "cpu_usage": random.randint(CPU_USAGE_RANGE[0], CPU_USAGE_RANGE[1]),
        "memory_usage": random.randint(MEMORY_USAGE_RANGE[0], MEMORY_USAGE_RANGE[1]),
        "disk_io": random.randint(DISK_IO_RANGE[0], DISK_IO_RANGE[1]),
        "network_traffic": random.randint(NETWORK_TRAFFIC_RANGE[0], NETWORK_TRAFFIC_RANGE[1]),
    }


def _generate_anomalous_metrics() -> Dict[str, int]:
    """
    Generate anomalous metric values (simulating system stress).

    Returns:
        Dictionary with cpu_usage, memory_usage, disk_io, network_traffic

    """
    return {
        "cpu_usage": random.randint(ANOMALY_CPU_RANGE[0], ANOMALY_CPU_RANGE[1]),
        "memory_usage": random.randint(ANOMALY_MEMORY_RANGE[0], ANOMALY_MEMORY_RANGE[1]),
        "disk_io": random.randint(ANOMALY_DISK_RANGE[0], ANOMALY_DISK_RANGE[1]),
        "network_traffic": random.randint(ANOMALY_NETWORK_RANGE[0], ANOMALY_NETWORK_RANGE[1]),
    }


def _generate_metrics(anomaly_rate: float) -> Dict[str, int]:
    """
    Generate metric data with optional anomaly injection.

    Args:
        anomaly_rate: Probability of generating anomalous metrics (0.0-1.0)

    Returns:
        Dictionary with metric values

    """
    if random.random() < anomaly_rate:
        metrics = _generate_anomalous_metrics()
        logger.debug(f"Generated ANOMALOUS metrics: {metrics}")
    else:
        metrics = _generate_normal_metrics()
        logger.debug(f"Generated normal metrics: {metrics}")

    return metrics


def _validate_metrics(metrics: Dict[str, Any]) -> None:
    """
    Validate metric data structure and values.

    Args:
        metrics: Metric dictionary to validate

    Raises:
        ValueError: If metrics are invalid

    """
    required_fields = {"cpu_usage", "memory_usage", "disk_io", "network_traffic"}

    if not isinstance(metrics, dict):
        raise ValueError(f"metrics must be dictionary, got {type(metrics)}")

    missing = required_fields - set(metrics.keys())
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    for field in required_fields:
        value = metrics[field]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Field '{field}' must be numeric, got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"Field '{field}' must be non-negative, got {value}")

    logger.debug(f"Metrics validated: {metrics}")


def _send_request(
    url: str,
    metrics: Dict[str, int],
    timeout: float,
    max_retries: int,
    retry_backoff: float
) -> Optional[Dict[str, Any]]:
    """
    Send metric data to API with retry logic and error handling.

    Args:
        url: API endpoint URL
        metrics: Metric dictionary to send
        timeout: Request timeout in seconds
        max_retries: Number of retry attempts
        retry_backoff: Backoff multiplier for exponential retry

    Returns:
        Response JSON if successful, None if all retries exhausted

    """
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Request attempt {attempt + 1}/{max_retries + 1} to {url}")

            response = requests.post(
                url,
                json=metrics,
                timeout=timeout
            )
            response.raise_for_status()

            response_data = response.json()
            logger.info(f"Sent: {metrics} | Response: {response_data}")
            return response_data

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout (>{timeout}s) - attempt {attempt + 1}/{max_retries + 1}"
            logger.warning(error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {e} - attempt {attempt + 1}/{max_retries + 1}"
            logger.warning(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {e.response.status_code} - {e}"
            logger.warning(error_msg)
            return None  # Don't retry on HTTP errors

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response: {e}"
            logger.error(error_msg)
            return None

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            return None

        # Exponential backoff on retry
        if attempt < max_retries:
            wait_time = RETRY_BACKOFF ** attempt
            logger.debug(f"Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)

    logger.error(f"Failed to send request after {max_retries + 1} attempts")
    return None


# ============================================================================
# Main Function
# ============================================================================

def run_live_simulator(
    api_url: str = API_URL,
    timeout: float = API_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
    max_retries: int = REQUEST_RETRIES,
    retry_backoff: float = RETRY_BACKOFF,
    anomaly_rate: float = ANOMALY_INJECTION_RATE,
) -> None:
    """
    Run continuous metric stream simulator sending to anomaly detection API.

    Generates random metrics and sends to API endpoint in infinite loop.
    Handles network errors gracefully with retry logic.

    Args:
        api_url: API endpoint URL
        timeout: Request timeout in seconds
        poll_interval: Time between requests in seconds
        max_retries: Number of retry attempts on failure
        retry_backoff: Backoff multiplier for exponential retry
        anomaly_rate: Probability of injecting anomalies (0.0-1.0)

    Raises:
        ValueError: If configuration is invalid
        KeyboardInterrupt: Caught and logged for graceful shutdown

    """
    try:
        logger.info("=" * 60)
        logger.info("Initializing live simulator")

        # Validate configuration
        _validate_api_url(api_url)
        _validate_config(timeout, poll_interval, max_retries, anomaly_rate)

        logger.info(f"API endpoint: {api_url}")
        logger.info(f"Poll interval: {poll_interval}s")
        logger.info(f"Anomaly injection rate: {anomaly_rate * 100:.1f}%")
        logger.info("=" * 60)

        # Main simulation loop
        request_count = 0
        failure_count = 0

        while True:
            try:
                # Generate metrics
                metrics = _generate_metrics(anomaly_rate)

                # Validate metrics before sending
                _validate_metrics(metrics)

                # Send to API with retries
                response = _send_request(
                    api_url,
                    metrics,
                    timeout,
                    max_retries,
                    retry_backoff
                )

                if response:
                    request_count += 1
                else:
                    failure_count += 1

                # Sleep before next request
                time.sleep(poll_interval)

            except ValueError as e:
                logger.error(f"Metric validation failed: {e}")
                time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Unexpected error in simulation loop: {e}", exc_info=True)
                failure_count += 1
                time.sleep(poll_interval)

    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except KeyboardInterrupt:
        logger.info(f"Simulation interrupted by user")
        logger.info(f"Total requests sent: {request_count}, Failures: {failure_count}")
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        run_live_simulator()
        exit(0)
    except KeyboardInterrupt:
        exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        exit(1)
