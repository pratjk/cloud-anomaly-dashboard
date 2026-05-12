# live simulator - keeps sending fake cloud metrics to the API
# run this in a separate terminal to see the dashboard come alive

import random
import time
import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
POLL_INTERVAL = 3.0  # seconds between requests
ANOMALY_RATE = 0.35  # 35% chance of anomaly

# normal ranges
NORMAL_CPU = (10, 65)
NORMAL_MEM = (20, 70)
NORMAL_DISK = (20, 150)
NORMAL_NET = (50, 400)

# anomaly ranges - push everything to the extreme
ANOM_CPU = (85, 100)
ANOM_MEM = (85, 100)
ANOM_DISK = (300, 500)
ANOM_NET = (800, 1200)

# kuch device IDs for variety
DEVICE_IDS = ["server-alpha", "server-beta", "server-gamma", "edge-node-1", "edge-node-2"]

# kuch fake log messages
NORMAL_LOGS = [
    "INFO System health check passed",
    "INFO Request processed successfully",
    "INFO Cache hit ratio: 94%",
    "INFO Backup completed",
]
ANOMALY_LOGS = [
    "ERROR Connection timeout - database unreachable",
    "ERROR OutOfMemoryError: heap space exhausted",
    "WARNING Disk usage exceeding 95%",
    "ERROR Segmentation fault in worker thread",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# random geo coordinates for the 3D map visualization
# maine bas kuch major cities ke coords daal diye hain
LOCATIONS = [
    (19.07, 72.87),   # Mumbai
    (28.61, 77.20),   # Delhi
    (37.77, -122.41), # San Francisco
    (51.50, -0.12),   # London
    (35.68, 139.69),  # Tokyo
    (1.35, 103.81),   # Singapore
]


def make_metrics():
    is_anomaly = random.random() < ANOMALY_RATE

    if is_anomaly:
        metrics = {
            "cpu_usage": random.randint(*ANOM_CPU),
            "memory_usage": random.randint(*ANOM_MEM),
            "disk_io": random.randint(*ANOM_DISK),
            "network_traffic": random.randint(*ANOM_NET),
            "log_message": random.choice(ANOMALY_LOGS),
        }
    else:
        metrics = {
            "cpu_usage": random.randint(*NORMAL_CPU),
            "memory_usage": random.randint(*NORMAL_MEM),
            "disk_io": random.randint(*NORMAL_DISK),
            "network_traffic": random.randint(*NORMAL_NET),
            "log_message": random.choice(NORMAL_LOGS),
        }

    # add device id and location
    metrics["device_id"] = random.choice(DEVICE_IDS)
    lat, lon = random.choice(LOCATIONS)
    metrics["latitude"] = lat + random.uniform(-2, 2)
    metrics["longitude"] = lon + random.uniform(-2, 2)

    return metrics


def send_to_api(metrics):
    try:
        resp = requests.post(API_URL, json=metrics, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pred = data.get("prediction", "?")
        cause = data.get("cause", "?")
        logger.info(f"[{metrics['device_id']}] cpu={metrics['cpu_usage']}% | prediction={pred} | cause={cause}")
        return data
    except requests.exceptions.ConnectionError:
        logger.warning("API se connect nahi ho raha - server chalu hai kya?")
        return None
    except Exception as e:
        logger.error(f"request failed: {e}")
        return None


def main():
    print("=" * 50)
    print("Live Simulator Started")
    print(f"API: {API_URL}")
    print(f"Interval: {POLL_INTERVAL}s | Anomaly Rate: {ANOMALY_RATE*100:.0f}%")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    count = 0
    fails = 0

    try:
        while True:
            metrics = make_metrics()
            result = send_to_api(metrics)

            if result:
                count += 1
            else:
                fails += 1

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\nStopped. Sent {count} requests, {fails} failed.")


if __name__ == "__main__":
    main()
