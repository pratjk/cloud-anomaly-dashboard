# Cloud Anomaly Detection Dashboard

A full-stack monitoring system that simulates cloud infrastructure data and detects anomalies in real-time using Machine Learning. 

This project was built to demonstrate the integration of machine learning models into a live data pipeline. It features a FastAPI backend for data processing and inference, and a Streamlit dashboard for real-time visualization.

---

## Architectural Overview

The system utilizes a decoupled client-server architecture:

1. **FastAPI Backend**: The core server. It receives simulated telemetry data, runs ML inference to detect anomalies using Scikit-Learn, and manages a local SQLite database for history.
2. **Monitoring Dashboard**: A Streamlit web app for real-time data visualization and system controls.
3. **Telemetry Source (Simulator)**: Continuous background data feed (`live_simulator.py`) or manual edge-injection (`injector_ui.py`) to simulate traffic.

---

## Key Features

- **Real-Time Anomaly Detection**: Uses Isolation Forests and Autoencoder-based approaches (via MLPRegressor) to flag irregular behavior in telemetry data.
- **Automated Quarantine Simulation**: When the ML engine detects consecutive high-severity anomalies from a specific device, it simulates a quarantine by temporarily blocking requests.
- **Interactive Visualizations**: Includes various time-series charts and a 3D mapping component (using Pydeck) to track simulated threat origins.

---

## Technology Stack

- **Backend**: Python, FastAPI, SQLAlchemy, Pydantic
- **Frontend**: Streamlit, Plotly, Pydeck
- **Machine Learning**: Scikit-Learn (Isolation Forest & MLPRegressor)
- **Database**: SQLite

---

## Quick Start Guide

To run the platform locally, launch these processes in three separate terminals:

### 1. Start the Backend Server (Terminal 1)
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the Dashboard (Terminal 2)
```powershell
streamlit run dashboard/app.py
```

### 3. Start the Data Simulator (Terminal 3)
The dashboard requires live data to display charts. Choose one of the following:

* **Continuous Live Simulation** (Recommended)
  ```powershell
  python simulator/live_simulator.py
  ```
* **Manual Anomaly Injection UI** (For testing specific scenarios)
  ```powershell
  streamlit run simulator/injector_ui.py --server.port 8502
  ```

---

## Testing & Validation

To observe the system's response to threats:
1. Open the Manual Injector UI (`http://localhost:8502`).
2. Trigger a "Memory Leak" or "DDoS" simulation.
3. Watch the main dashboard flag the anomaly.
4. If a device gets blocked by the quarantine system, use `python unquarantine.py` to reset the environment.

---

## License

MIT License.
