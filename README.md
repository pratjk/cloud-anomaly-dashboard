# Cloud Anomaly Detection Dashboard

I built this because I wanted to understand how anomaly detection actually works in cloud systems. The idea is simple - simulate cloud server data (CPU, memory, disk, network), feed it into ML models, and flag anything that looks suspicious.

It turned out to be way more interesting than I expected. I ended up building a full-stack app with a FastAPI backend, a Streamlit dashboard, and even a basic auto-quarantine system that blocks devices if they keep sending malicious-looking data.

---

## What it does

The system has 3 main parts:

1. **Backend (FastAPI)** - receives telemetry data, runs the ML models on it, stores results in SQLite
2. **Dashboard (Streamlit)** - shows real-time charts, a 3D globe for tracking where threats come from, and controls for the quarantine system
3. **Simulator** - generates fake cloud data and sends it to the backend continuously so the dashboard has something to display

The ML side uses two models:
- **Isolation Forest** for detecting anomalies in numeric metrics (CPU spikes, memory leaks etc)
- **Autoencoder** (MLPRegressor) for detecting weird patterns in log messages

Both scores are combined with a weighted hybrid system to make the final call.

---

## Tech Stack

- Python, FastAPI, SQLAlchemy, SQLite
- Streamlit, Plotly, Pydeck
- Scikit-Learn (Isolation Forest + MLPRegressor)

---

## How to run it

You need 3 terminals open:

**Terminal 1 - Start the backend:**
```
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Start the dashboard:**
```
streamlit run dashboard/app.py
```

**Terminal 3 - Start the simulator:**
```
python simulator/live_simulator.py
```

The dashboard won't show any data until the simulator is running.

---

## Quarantine System

If the model detects something really bad (like a DDoS pattern or extreme memory leak), the backend automatically blocks that device from sending more requests. You can see blocked devices on the dashboard and release them manually, or run `python unquarantine.py` to clear everything.

---

## Testing

1. Run the simulator and watch the dashboard update in real-time
2. You can also use the manual injector UI to trigger specific anomalies:
   ```
   streamlit run simulator/injector_ui.py --server.port 8502
   ```
3. Try triggering a "Memory Leak" or "DDoS" and see the quarantine kick in

---

## What I learned

- How Isolation Forests work for anomaly detection (they basically isolate outliers by randomly splitting the data)
- How autoencoders detect anomalies (train them to reconstruct normal data, high reconstruction error = anomaly)
- Building REST APIs with FastAPI and connecting them to a frontend
- Working with SQLAlchemy and SQLite for data persistence
- Real-time data visualization with Streamlit and Plotly

---

## License

MIT
