"""
Cloud Anomaly Detection - Prediction Module Test

Tests the hybrid anomaly detection prediction module with sample metrics.
Validates both normal and anomalous metric combinations.

Module Flow:
    1. Define test cases with expected outcomes
    2. Validate test data structure and values
    3. Execute predictions on each test case
    4. Compare results with expected values
    5. Log detailed results and summary
    6. Return overall test status

Configuration:
    - TEST_CASES: Named test samples with expected results and descriptions

Test Suite:
    - normal_low_utilization: Low to moderate resource usage → normal
    - anomaly_high_stress: Very high resource usage on all metrics → anomaly

Returns:
    Exit code:
    - 0: All tests passed
    - 1: Test failed or execution error
    - 130: User interruption (Ctrl+C)
"""

import logging
from typing import Dict, Any, Tuple, List, Optional

from ml.predict import predict_anomaly

# ============================================================================
# Configuration Section
# ============================================================================

# Test cases with metadata (name -> {data, expected, description})
TEST_CASES: Dict[str, Dict[str, Any]] = {
    "normal_low_utilization": {
        "data": {
            "cpu_usage": 45,
            "memory_usage": 55,
            "disk_io": 110,
            "network_traffic": 280,
        },
        "expected": "normal",
        "description": "Low to moderate resource utilization"
    },
    "anomaly_high_stress": {
        "data": {
            "cpu_usage": 98,
            "memory_usage": 95,
            "disk_io": 450,
            "network_traffic": 1100,
        },
        "expected": "anomaly",
        "description": "Very high resource utilization on all metrics"
    }
}

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

def _validate_test_data(test_data: Dict[str, Any]) -> None:
    """
    Validate test case data structure before execution.

    Args:
        test_data: Test case dictionary with 'data', 'expected', 'description'

    Raises:
        ValueError: If test data is invalid or incomplete

    """
    # Validate required keys
    required_keys = {"data", "expected"}
    if not isinstance(test_data, dict):
        raise ValueError(f"Test data must be dictionary, got {type(test_data).__name__}")

    missing = required_keys - set(test_data.keys())
    if missing:
        raise ValueError(f"Test data missing keys: {missing}")

    # Validate metrics dictionary
    metrics = test_data["data"]
    required_fields = {"cpu_usage", "memory_usage", "disk_io", "network_traffic"}

    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics must be dictionary, got {type(metrics).__name__}")

    missing_fields = required_fields - set(metrics.keys())
    if missing_fields:
        raise ValueError(f"Metrics missing fields: {missing_fields}")

    # Validate metric values
    for field, value in metrics.items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Metric '{field}' must be numeric, got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"Metric '{field}' must be non-negative, got {value}")

    # Validate expected result
    expected = test_data["expected"]
    if expected not in ("normal", "anomaly"):
        raise ValueError(f"Expected must be 'normal' or 'anomaly', got '{expected}'")

    logger.debug(f"Test data validated successfully")


def _interpret_prediction(prediction_result: Any) -> Tuple[bool, float]:
    """
    Interpret prediction result from predict_anomaly function.

    Handles different return types (dict, bool, numeric).

    Args:
        prediction_result: Result from predict_anomaly() function

    Returns:
        Tuple of (is_anomaly: bool, confidence: float)

    Raises:
        ValueError: If result cannot be interpreted

    """
    try:
        if isinstance(prediction_result, dict):
            # Dict result: extract prediction and confidence
            is_anomaly = prediction_result.get("anomaly", 
                                            prediction_result.get("prediction", False))
            confidence = float(prediction_result.get("confidence", 0.0))
            is_anomaly = bool(is_anomaly)

        elif isinstance(prediction_result, bool):
            # Boolean result
            is_anomaly = prediction_result
            confidence = 1.0

        elif isinstance(prediction_result, (int, float)):
            # Numeric result: threshold at 0.5
            is_anomaly = float(prediction_result) > 0.5
            confidence = abs(float(prediction_result))

        else:
            # Fallback: convert to bool
            is_anomaly = bool(prediction_result)
            confidence = 1.0

        logger.debug(f"Interpreted prediction: anomaly={is_anomaly}, confidence={confidence:.3f}")
        return (is_anomaly, confidence)

    except Exception as e:
        raise ValueError(f"Failed to interpret prediction: {e}") from e


def _run_test(
    test_name: str,
    test_data: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Execute single test case and return result.

    Args:
        test_name: Name/identifier of test case
        test_data: Test data with 'data', 'expected', 'description' keys

    Returns:
        Tuple of (passed: bool, message: str, result_dict: Dict)

    """
    try:
        logger.info(f"Running test: {test_name}")

        # Validate test data
        _validate_test_data(test_data)

        # Extract test components
        metrics = test_data["data"]
        expected = test_data["expected"]
        description = test_data.get("description", "")

        # Execute prediction
        logger.debug(f"Test data: {metrics}")
        prediction_result = predict_anomaly(metrics)
        logger.debug(f"Raw prediction result: {prediction_result}")

        # Interpret result
        is_anomaly, confidence = _interpret_prediction(prediction_result)

        # Determine expected as boolean
        expected_bool = expected.lower() == "anomaly"

        # Check if test passed
        passed = (is_anomaly == expected_bool)

        predicted_label = "anomaly" if is_anomaly else "normal"
        status = "PASS" if passed else "FAIL"
        message = (
            f"{status}: {test_name} - "
            f"Expected '{expected}', got '{predicted_label}' "
            f"(confidence: {confidence:.3f}) - {description}"
        )

        if passed:
            logger.info(message)
        else:
            logger.error(message)

        # Return result tuple
        return (
            passed,
            message,
            {
                "test_name": test_name,
                "expected": expected,
                "predicted": predicted_label,
                "confidence": confidence,
                "passed": passed,
                "description": description,
            }
        )

    except ValueError as e:
        error_msg = f"Test validation failed: {e}"
        logger.error(error_msg)
        return (
            False,
            error_msg,
            {
                "test_name": test_name,
                "error": str(e),
                "passed": False,
            }
        )
    except Exception as e:
        error_msg = f"Test execution failed: {e}"
        logger.error(error_msg, exc_info=True)
        return (
            False,
            error_msg,
            {
                "test_name": test_name,
                "error": str(e),
                "passed": False,
            }
        )


# ============================================================================
# Main Function
# ============================================================================

def run_prediction_tests(
    test_cases: Dict[str, Dict[str, Any]] | None = None
) -> bool:
    """
    Run all prediction tests and log results.

    Executes each test case, validates results, and provides summary.

    Args:
        test_cases: Dictionary of test cases (default: predefined TEST_CASES)

    Returns:
        True if all tests passed, False if any failed

    """
    try:
        test_suite = test_cases or TEST_CASES

        if not test_suite:
            logger.warning("No test cases defined")
            return False

        logger.info("=" * 60)
        logger.info(f"Starting prediction tests ({len(test_suite)} test case(s))")
        logger.info("=" * 60)

        # Execute all tests
        results: List[Dict[str, Any]] = []
        passed_count = 0
        failed_count = 0

        for test_name, test_data in test_suite.items():
            passed, message, result = _run_test(test_name, test_data)
            results.append(result)

            if passed:
                passed_count += 1
            else:
                failed_count += 1

        # Print results summary
        logger.info("=" * 60)
        logger.info(f"Test Results:")
        logger.info(f"  Passed: {passed_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Total:  {passed_count + failed_count}")
        logger.info("=" * 60)

        return failed_count == 0

    except Exception as e:
        logger.critical(f"Critical error in test execution: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    try:
        success = run_prediction_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Tests interrupted by user")
        exit(130)
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        exit(1)
