"""
Cloud Anomaly Detection - Real-world Anomaly Injector

Simulates specific real-world anomalies:
- Memory Leak
- Crypto-Mining / CPU Spike
- DDoS / Network Flood
- Disk Failure
"""

import requests
import time
import logging
import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

API_URL: str = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

def generate_memory_leak(device_id: str = "Unknown", latitude: float = 0.0, longitude: float = 0.0) -> Dict[str, Any]:
    return {
        "device_id": device_id,
        "cpu_usage": 45.0,
        "memory_usage": 98.5,
        "disk_io": 120.0,
        "network_traffic": 300.0,
        "latitude": latitude,
        "longitude": longitude,
        "log_message": "WARNING OutOfMemory exception imminent"
    }

def generate_crypto_mining(device_id: str = "Unknown", latitude: float = 0.0, longitude: float = 0.0) -> Dict[str, Any]:
    return {
        "device_id": device_id,
        "cpu_usage": 99.9,
        "memory_usage": 60.0,
        "disk_io": 80.0,
        "network_traffic": 950.0,
        "latitude": latitude,
        "longitude": longitude,
        "log_message": "WARNING Unrecognized high CPU process detected"
    }

def generate_ddos_attack(device_id: str = "Unknown", latitude: float = 0.0, longitude: float = 0.0) -> Dict[str, Any]:
    return {
        "device_id": device_id,
        "cpu_usage": 80.0,
        "memory_usage": 70.0,
        "disk_io": 150.0,
        "network_traffic": 1800.0,
        "latitude": latitude,
        "longitude": longitude,
        "log_message": "ERROR Connection pool exhausted"
    }

def generate_disk_failure(device_id: str = "Unknown", latitude: float = 0.0, longitude: float = 0.0) -> Dict[str, Any]:
    return {
        "device_id": device_id,
        "cpu_usage": 50.0,
        "memory_usage": 85.0,
        "disk_io": 0.0,
        "network_traffic": 100.0,
        "latitude": latitude,
        "longitude": longitude,
        "log_message": "CRITICAL read-only file system error"
    }

def request_anomaly(anomaly_data: Dict[str, Any], url: str = API_URL) -> bool:
    try:
        logger.info(f"Injecting Anomaly Payload: {anomaly_data}")
        response = requests.post(url, json=anomaly_data, timeout=5.0)
        
        if response.status_code == 403:
            logger.error(f"❌ ACCESS DENIED: {response.json().get('detail', 'Device Quarantined')}")
            return False
            
        response.raise_for_status()
        logger.info(f"API Response: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"Failed to inject anomaly: {e}")
        return False

# For testing standalone
if __name__ == "__main__":
    request_anomaly(generate_memory_leak("TestDevice"))
