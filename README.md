# Cloud Anomaly Detection Dashboard

A full-stack monitoring project that simulates cloud infrastructure data and detects anomalies in real-time using Machine Learning (Isolation Forests & Autoencoders). 

It features a FastAPI backend for data processing and a Streamlit dashboard for visualization.

---

## 🏛️ Architectural Overview

The system utilizes a strictly **decoupled client-server architecture**, mimicking distributed production environments:

1.  **FastAPI Backend**: The core server. It receives simulated telemetry data, runs ML inference to detect anomalies, and manages a local SQLite database.
2.  **Monitoring Dashboard**: A Streamlit web app for real-time data visualization and system controls.
3.  **Telemetry Source (Simulator)**: Continuous background data feed (`live_simulator.py`) or manual edge-injection (`injector_ui.py`). **The dashboard will appear static unless a simulator is running.**

---

## 🚀 Quick Start Guide

To see the platform in action with **live moving data**, launch these in **three separate terminals**:

### 1. The Inference Server (Terminal 1)
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. The Command Center (Terminal 2)
```powershell
streamlit run dashboard/app.py
```

### 3. The Live Data Feed (Terminal 3)
Choose **ONE** of the following depending on your needs:

*   **Option A: Continuous Live Simulation (Recommended for UI testing)**
    ```powershell
    python simulator/live_simulator.py
    ```
*   **Option B: Manual Anomaly Injection (Interactive UI)**
    ```powershell
    streamlit run simulator/injector_ui.py --server.port 8502
    ```

---

## 🛡️ Quarantine System

The project features a basic quarantine mechanism. When the ML engine detects consecutive high-severity anomalies from a specific device:

-   **Blocking**: The backend temporarily rejects requests from the blocked `device_id`.
-   **Logging**: Blocked attempts are logged.
-   **Override**: A utility script (`unquarantine.py`) can be used to reset device status.



---

## 🧪 Testing & Validation

To ensure the platform is operating at peak efficiency, follow these testing protocols:

1.  **Manual Injection**: Use the [Injector UI](http://localhost:8502) to trigger a "Memory Leak" and observe the "Cyber Jail" protocol in action.
2.  **Unit Tests**: Run `pytest tests/` to validate core ML and API logic.
3.  **End-to-End**: Run `python simulator/live_simulator.py` and verify charts move in the Dashboard.
4.  **Security Reset**: Run `python unquarantine.py` to clear the environment after a security test.

Refer to the [**TESTING_GUIDE.md**](file:///c:/Users/Solgaleo/Downloads/cloud-anomaly-detection/cloud-anomaly-detection/TESTING_GUIDE.md) for detailed test cases.

---

## 🌟 Key Features

### Automated Response Simulation
When an anomaly is detected, the backend can simulate a basic response (e.g. blocking the device) to demonstrate automated handling.

### Interactive Visualizations
The dashboard includes various charts and a 3D globe (using Pydeck) to map simulated threat origins.

---

## 🛠️ Technology Stack

-   **Backend**: FastAPI, SQLAlchemy, Pydantic
-   **Frontend**: Streamlit, Plotly, Pydeck
-   **ML Engine**: Scikit-Learn (Isolation Forest & MLPRegressor-based Autoencoder)

---

## 📂 Project Anatomy

```text
├── backend/                  # FastAPI Core
├── dashboard/                # Synthwave Monitoring UI
├── simulator/                # Edge Client & Live Simulation
├── ml/                       # Hybrid Intelligence (Predict/Retrain)
├── database/                 # SQLAlchemy Models & Schemas
├── logs/                     # Network-wide System Logs
├── unquarantine.py           # Security Reset Utility
├── .env.example              # Configuration Template
└── CROSS_NETWORK_GUIDE.md    # ngrok Setup Instructions
```

---

## 📜 License
MIT License. Built for advanced cloud observability research.
