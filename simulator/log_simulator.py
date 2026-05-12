"""
Cloud Anomaly Detection - Log Simulator

Generates synthetic log messages with metric data and sends to anomaly detection API.
Simulates batch log analysis with random metric variations and contextual log messages.

Module Flow:
    1. Validate API endpoint and configuration
    2. For each request (up to NUM_REQUESTS):
    a. Generate random metrics within configured ranges
    b. Select random log template
    c. Construct payload with metrics and log message
    d. Validate payload structure and values
    e. Send POST request to API with error handling
    f. Log response status and details
    3. Insert delay after batch completes
    4. Exit with appropriate status code

Configuration:
    - API_URL: Endpoint for prediction API
    - API_TIMEOUT: Request timeout in seconds
    - NUM_REQUESTS: Number of log entries to generate and send
    - POST_BATCH_DELAY: Delay after batch completes (seconds)
    - REQUEST_RETRIES: Number of retry attempts on network error

Metric Ranges:
    - CPU usage: 10-90%
    - Memory usage: 20-95%
    - Disk I/O: 5-80 MB/s
    - Network traffic: 10-100 Mbps

Log Template Categories:
    - INFO: Service events (startup, login, backup)
    - WARNING: Performance degradation (memory, CPU spikes)
    - ERROR: System failures (connection, read, access)

Returns:
    Exit code:
    - 0: All requests completed successfully
    - 1: Request/validation error
    - 130: User interruption (Ctrl+C)
"""

import json
import logging
import random
import time
from typing import Dict, Any, List, Optional, Tuple

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration Section
# ============================================================================

API_URL: str = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
API_TIMEOUT: float = 5.0
NUM_REQUESTS: int = 1000000
POST_BATCH_DELAY: float = 2.0
REQUEST_RETRIES: int = 2
RETRY_BACKOFF: float = 1.5

# Metric ranges (min, max)
CPU_USAGE_RANGE: Tuple[int, int] = (10, 90)
MEMORY_USAGE_RANGE: Tuple[int, int] = (20, 95)
DISK_IO_RANGE: Tuple[int, int] = (5, 80)
NETWORK_TRAFFIC_RANGE: Tuple[int, int] = (10, 100)

# Log message templates organized by severity
LOG_TEMPLATES: List[str] = [
    "INFO Service started successfully",
    "INFO User login request",
    "WARNING High memory usage detected",
    "ERROR Database connection failed",
    "ERROR Disk read failure",
    "INFO Scheduled backup completed",
    "WARNING CPU usage spike",
    "ERROR Unauthorized access attempt",
]

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
        raise ValueError(f"API_URL must be non-empty string, got '{url}'")

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"API_URL must start with http:// or https://, got '{url}'")

    logger.debug(f"API URL validated: {url}")


def _validate_config(
    num_requests: int,
    timeout: float,
    post_delay: float,
    retries: int
) -> None:
    """
    Validate simulator configuration parameters.

    Args:
        num_requests: Number of requests to send (must be > 0)
        timeout: Request timeout in seconds (must be > 0)
        post_delay: Delay after batch (must be >= 0)
        retries: Number of retry attempts (must be >= 0)

    Raises:
        ValueError: If any parameter is invalid

    """
    if not isinstance(num_requests, int) or num_requests <= 0:
        raise ValueError(f"num_requests must be positive integer, got {num_requests}")

    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    if not isinstance(post_delay, (int, float)) or post_delay < 0:
        raise ValueError(f"post_delay must be non-negative, got {post_delay}")

    if not isinstance(retries, int) or retries < 0:
        raise ValueError(f"retries must be non-negative integer, got {retries}")

    logger.debug(
        f"Config validated: num_requests={num_requests}, "
        f"timeout={timeout}s, post_delay={post_delay}s, retries={retries}"
    )


def _validate_log_templates(templates: List[str]) -> None:
    """
    Validate log template list before use.

    Args:
        templates: List of log message templates

    Raises:
        ValueError: If templates list is invalid or empty

    """
    if not isinstance(templates, list) or len(templates) == 0:
        raise ValueError(f"Log templates must be non-empty list, got {templates}")

    for i, template in enumerate(templates):
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"Template {i} must be non-empty string, got '{template}'")

    logger.debug(f"Validated {len(templates)} log templates")


def _generate_log_message(templates: List[str]) -> str:
    """
    Select random log message from templates.

    Args:
        templates: List of available log message templates

    Returns:
        Randomly selected log message template

    """
    message = random.choice(templates)
    logger.debug(f"Generated log message: {message}")
    return message


def _generate_metrics() -> Dict[str, int]:
    """
    Generate random metric values within configured ranges.

    Returns:
        Dictionary with cpu_usage, memory_usage, disk_io, network_traffic

    """
    metrics = {
        "cpu_usage": random.randint(CPU_USAGE_RANGE[0], CPU_USAGE_RANGE[1]),
        "memory_usage": random.randint(MEMORY_USAGE_RANGE[0], MEMORY_USAGE_RANGE[1]),
        "disk_io": random.randint(DISK_IO_RANGE[0], DISK_IO_RANGE[1]),
        "network_traffic": random.randint(NETWORK_TRAFFIC_RANGE[0], NETWORK_TRAFFIC_RANGE[1]),
    }
    logger.debug(f"Generated metrics: {metrics}")
    return metrics


def _construct_payload(
    metrics: Dict[str, int],
    log_message: str
) -> Dict[str, Any]:
    """
    Construct API payload from metrics and log message.

    Args:
        metrics: Dictionary of metric values
        log_message: Log message string

    Returns:
        Complete payload dictionary for API request

    """
    payload = {
        **metrics,
        "log_message": log_message,
    }
    logger.debug(f"Constructed payload: {payload}")
    return payload


def _validate_payload(payload: Dict[str, Any]) -> None:
    """
    Validate payload structure and values before sending.

    Args:
        payload: Payload dictionary to validate

    Raises:
        ValueError: If payload is invalid

    """
    required_fields = {"cpu_usage", "memory_usage", "disk_io", "network_traffic", "log_message"}

    if not isinstance(payload, dict):
        raise ValueError(f"Payload must be dictionary, got {type(payload)}")

    missing = required_fields - set(payload.keys())
    if missing:
        raise ValueError(f"Payload missing fields: {missing}")

    # Validate numeric fields
    metric_fields = {"cpu_usage", "memory_usage", "disk_io", "network_traffic"}
    for field in metric_fields:
        value = payload[field]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Field '{field}' must be numeric, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"Field '{field}' must be non-negative, got {value}")

    # Validate log message
    if not isinstance(payload["log_message"], str) or not payload["log_message"].strip():
        raise ValueError(f"log_message must be non-empty string")

    logger.debug(f"Payload validated successfully")


def _send_request(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
    max_retries: int,
    retry_backoff: float
) -> Optional[Tuple[int, str]]:
    """
    Send payload to API with error handling and retry logic.

    Args:
        url: API endpoint URL
        payload: Request payload dictionary
        timeout: Request timeout in seconds
        max_retries: Number of retry attempts
        retry_backoff: Backoff multiplier for exponential retry

    Returns:
        Tuple of (status_code, response_text) if successful, None otherwise

    """
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Sending request (attempt {attempt + 1}/{max_retries + 1})")

            response = requests.post(
                url,
                json=payload,
                timeout=timeout
            )

            status_code = response.status_code
            response_text = response.text

            logger.info(f"Request sent - Status: {status_code}")
            logger.debug(f"Response: {response_text[:200]}")  # Truncate long responses

            return (status_code, response_text)

        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout (>{timeout}s): {str(e)} - attempt {attempt + 1}/{max_retries + 1}"
            logger.warning(error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)} - attempt {attempt + 1}/{max_retries + 1}"
            logger.warning(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            logger.error(error_msg)
            return None

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return None

        # Exponential backoff on retry
        if attempt < max_retries:
            wait_time = retry_backoff ** attempt
            logger.debug(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    logger.error(f"Failed after {max_retries + 1} attempts")
    return None


# ============================================================================
# Main Function
# ============================================================================

def run_log_simulator(
    api_url: str = API_URL,
    num_requests: int = NUM_REQUESTS,
    timeout: float = API_TIMEOUT,
    post_delay: float = POST_BATCH_DELAY,
    max_retries: int = REQUEST_RETRIES,
    retry_backoff: float = RETRY_BACKOFF,
    log_templates: List[str] | None = None,
    continuous: bool = True,
) -> bool:
    """
    Run log simulator sending log entries with metrics to API continuously.

    Generates synthetic log messages with random metrics and sends to prediction 
    API. Runs in infinite/continuous mode by default (keeps running indefinitely).
    Validates all data before sending.

    Args:
        api_url: API endpoint URL
        num_requests: Number of log entries per batch (ignored in continuous mode)
        timeout: Request timeout in seconds
        post_delay: Delay between requests/batches
        max_retries: Number of retry attempts per request
        retry_backoff: Backoff multiplier for retries
        log_templates: Custom log templates (default: predefined)
        continuous: If True, run indefinitely; if False, send num_requests then stop

    Returns:
        True if running successfully, False on critical error
        (In continuous mode, only returns False on fatal errors)

    """
    try:
        logger.info("=" * 60)
        logger.info("Starting log simulator")

        # Use default templates if none provided
        templates = log_templates or LOG_TEMPLATES

        # Validate configuration
        _validate_api_url(api_url)
        _validate_log_templates(templates)
        
        if not continuous:
            _validate_config(num_requests, timeout, post_delay, max_retries)

        logger.info(f"API endpoint: {api_url}")
        logger.info(f"Mode: {'CONTINUOUS (infinite loop)' if continuous else f'BATCH ({num_requests} requests)'}")
        logger.info(f"Delay between requests: {post_delay}s")
        logger.info("=" * 60)

        # Track statistics
        stats = {
            "total_sent": 0,
            "successful": 0,
            "failed": 0,
            "batch_count": 0
        }

        # Main loop - infinite if continuous=True, fixed if continuous=False
        request_count = 0
        
        while True:  # Infinite loop for continuous mode
            batch_num = stats["batch_count"] + 1
            
            if not continuous and request_count >= num_requests:
                logger.info(f"Reached requested limit of {num_requests} requests. Stopping.")
                break
            
            try:
                request_count += 1
                stats["total_sent"] += 1
                
                mode_str = f"[Batch {batch_num}] " if continuous else ""
                logger.info(f"{mode_str}Request #{request_count}")

                # Generate data
                metrics = _generate_metrics()
                log_message = _generate_log_message(templates)

                # Construct and validate payload
                payload = _construct_payload(metrics, log_message)
                _validate_payload(payload)

                # Send request
                result = _send_request(
                    api_url,
                    payload,
                    timeout,
                    max_retries,
                    retry_backoff
                )

                if result:
                    status_code, response_text = result
                    stats["successful"] += 1
                    logger.debug(f"Response: {response_text[:200]}")
                else:
                    stats["failed"] += 1
                    logger.warning(f"Request #{request_count} failed")

            except ValueError as e:
                logger.error(f"Validation error in request #{request_count}: {e}")
                stats["failed"] += 1
            except Exception as e:
                logger.error(f"Unexpected error in request #{request_count}: {e}", exc_info=True)
                stats["failed"] += 1
            
            # Delay before next request (in continuous mode, this is between requests)
            if post_delay > 0:
                time.sleep(post_delay)

        # Summary (only reached in non-continuous mode)
        logger.info("=" * 60)
        logger.info(f"Simulation complete (non-continuous mode):")
        logger.info(f"  Total requests: {stats['total_sent']}")
        logger.info(f"  Successful: {stats['successful']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info("=" * 60)
        return stats["failed"] == 0

    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning("Simulator interrupted by user (Ctrl+C)")
        logger.info("=" * 60)
        logger.info(f"Statistics before interrupt:")
        logger.info(f"  Total requests sent: {stats.get('total_sent', 0)}")
        logger.info(f"  Successful: {stats.get('successful', 0)}")
        logger.info(f"  Failed: {stats.get('failed', 0)}")
        logger.info("=" * 60)
        # Return False to indicate interrupted state
        return False
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    try:
        # Run in continuous mode by default (infinite loop)
        # Set continuous=False to run fixed number of requests via NUM_REQUESTS
        success = run_log_simulator(continuous=True)
        # In continuous mode, this only exits on error or interrupt
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Main: Simulator interrupted by user (Ctrl+C)")
        exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        exit(1)