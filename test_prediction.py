# quick test script to check if prediction is working

from ml.predict import predict_anomaly

# test case 1: normal metrics - should be "Normal"
normal_data = {
    "cpu_usage": 45,
    "memory_usage": 55,
    "disk_io": 110,
    "network_traffic": 280,
}

# test case 2: everything maxed out - should be "Anomaly"
crazy_data = {
    "cpu_usage": 98,
    "memory_usage": 95,
    "disk_io": 450,
    "network_traffic": 1100,
    "log_message": "ERROR OutOfMemoryError: Java heap space"
}


def run_tests():
    print("Testing prediction module...\n")

    # normal test
    result1 = predict_anomaly(normal_data)
    pred1 = result1.get("prediction", "?")
    print(f"Normal metrics -> {pred1}")
    if pred1.lower() == "normal":
        print("  PASS")
    else:
        print("  FAIL (expected Normal)")

    print()

    # anomaly test
    result2 = predict_anomaly(crazy_data)
    pred2 = result2.get("prediction", "?")
    cause = result2.get("cause", "?")
    print(f"Anomaly metrics -> {pred2} (cause: {cause})")
    if pred2.lower() == "anomaly":
        print("  PASS")
    else:
        print("  FAIL (expected Anomaly)")

    print("\ndone!")


if __name__ == "__main__":
    run_tests()
