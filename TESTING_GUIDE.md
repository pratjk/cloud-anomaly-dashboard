# 🧪 Cloud Anomaly Detection - Testing & Validation Guide

This guide provides a comprehensive framework for validating the entire observability ecosystem, from low-level ML inference to high-level security quarantine protocols.

---

## 1. Automated Unit Testing

These tests validate the core logic of the system without requiring the full platform to be running.

### Prerequisite: Setup Test Database
```powershell
# Set an isolated database for testing
$env:DATABASE_URL="sqlite:///./test.db"
pip install pytest pytest-asyncio
```

### Run All Core Tests
Executing this command will validate database connectivity, model serialization, and basic ML fallback logic.
```powershell
pytest tests/ -v
```

---

## 2. Manual Anomaly Injection (UI-Based)

The most effective way to test the system is to simulate a "Day-in-the-Life" using the [**Injector UI**](http://localhost:8502).

### Protocol: Triggering a Memory Leak
1.  **Launch the Dashboard** and monitor the "Inference History" tab.
2.  **Open the Injector UI** (Terminal 3).
3.  **Action**: Select "Memory Leak" from the payload list.
4.  **Click**: `Inject Memory Leak`.
5.  **Observation**: 
    -   The Dashboard should instantly show a red **Anomaly** record.
    -   The "System Logs" tab will show: `WARNING OutOfMemory exception imminent`.
    -   The "Geographical Map" will show a pulse coming from the origin region.

---

## 3. Security Validation: The "Cyber Jail"

This test validates the **Zero-Trust Security** layer of the backend.

### Protocol: Testing the Quarantine Loop
1.  **Trigger a High-Severity Anomaly**: Perform the "Memory Leak" injection twice within 2 minutes.
2.  **Observe the Block**: On the third attempt, the Injector UI will display an **Access Denied (403)** error.
3.  **Backend Logs**: Verify that the backend console shows:
    ```text
    BLOCKED: Attempted access from quarantined device Client-Laptop-1
    ```
4.  **Confirm Persistence**: Stop and restart the Injector UI. Try to inject again. You should still be blocked—the quarantine is persistent in the database.

### Protocol: Emergency Release
To reset the environment and release your device from custody:
```powershell
python unquarantine.py
```
Verify that the output confirms: `All devices have been released from quarantine successfully!`

---

## 4. Cross-Network Verification

If you are using **ngrok** to test across networks, use this protocol to ensure connectivity.

1.  **Connectivity Check**: Ping your public URL:
    ```powershell
    curl https://your-ngrok-url.ngrok-free.app/health
    ```
    Expected result: `{"status": "healthy", ...}`
2.  **External Injection**: Run the simulator from a different computer or even a mobile device using the same public URL in its `.env` file.
3.  **Latency Check**: Verify the Dashboard updates within 5-10 seconds of an external data push.

---

## 📂 Testing Artifacts
- [**logs/system.log**](file:///c:/Users/Solgaleo/Downloads/cloud-anomaly-detection/cloud-anomaly-detection/logs/system.log): Detailed audit trail of all detections and blocks.
- [**unquarantine.py**](file:///c:/Users/Solgaleo/Downloads/cloud-anomaly-detection/cloud-anomaly-detection/unquarantine.py): Utility for environment resets.
- [**tests/**](file:///c:/Users/Solgaleo/Downloads/cloud-anomaly-detection/cloud-anomaly-detection/tests/): Directory containing automated test suites.

---

**Last Updated:** 2026-04-15  
**Version:** 2.0 (Security & Network Enhanced)
