# Cloud Anomaly Detection - Quick Reference Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
```bash
python --version  # 3.9+
pip install -r requirements.txt
```

### Run Locally
```bash
# Terminal 1: Start backend
python -m uvicorn backend.main:app --reload

# Terminal 2: Start dashboard
streamlit run dashboard/app.py

# Terminal 3: (Optional) Run data simulator
python simulator/log_simulator.py
```

### Verify Setup
```bash
curl http://localhost:8000/health  # Should return healthy status
open http://localhost:8501         # Dashboard in browser
```

---

##  API Reference

### POST /predict
Send metrics for anomaly detection.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_usage": 75.5,
    "memory_usage": 85.2,
    "disk_io": 250.0,
    "network_traffic": 800.0,
    "log_message": "ERROR: High resource usage"
  }'
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Anomaly",
  "cause": "Anomaly",
  "timestamp": "2026-04-01T10:30:00",
  "metrics": {...},
  "stored": true
}
```

### GET /predictions
Retrieve all stored predictions.

```bash
curl http://localhost:8000/predictions
```

### GET /health
Check system health.

```bash
curl http://localhost:8000/health
```

---

## 📁 Project Structure Quick Reference

```
cloud-anomaly-detection/
├── backend/                 # FastAPI application
│   ├── main.py             # API endpoints & logic
│   ├── database.py         # SQLAlchemy setup
│   ├── schemas.py          # (Empty - for future validation)
│   └── routes/             # Additional route handlers
│
├── ml/                      # Machine learning pipeline
│   ├── predict.py          # Main prediction function
│   ├── train.py            # Model training
│   ├── retrain.py          # Model retraining
│   ├── log_anomaly_model.py # Log analysis model
│   ├── utils.py            # Helper functions
│   ├── model.pkl           # Trained Isolation Forest
│   └── log_model.pkl       # Trained Autoencoder
│
├── dashboard/              # Streamlit frontend
│   ├── app.py              # Main dashboard
│   └── components/         # UI components
│
├── simulator/              # Data generation
│   ├── log_simulator.py    # Log producer
│   ├── kafka_producer.py   # Kafka integration
│   ├── data_generator.py   # Metric generation
│   └── live_simulator.py   # Live data source
│
├── streaming/              # Stream processing (optional)
│   ├── kafka_consumer.py   # Kafka consumer
│   └── spark_streaming.py  # Spark integration
│
├── database/               # Database models
│   ├── models.py           # SQLAlchemy ORM models
│   └── init.sql            # Database schema
│
├── configs/                # Configuration files
│   ├── app_config.yaml
│   ├── kafka_config.yaml
│   └── spark_config.yaml
│
├── data/                   # Data storage
│   ├── sample_dataset.csv
│   ├── raw/
│   └── processed/
│
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks
├── logs/                   # Application logs
├── PRODUCTION_STATUS_REPORT.md
├── DEPLOYMENT_GUIDE.md
├── TESTING_GUIDE.md
├── REFACTORING_SUMMARY.md
└── README.md
```

---

## ⚙️ Configuration Quick Reference

### Environment Variables
```bash
# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=sqlite:///./cloud.db
LOG_LEVEL=INFO

# Model Configuration
MODEL_RETRAIN_INTERVAL=50
METRIC_MODEL_PATH=ml/model.pkl
LOG_MODEL_PATH=ml/log_model.pkl

# Dashboard Configuration
API_BASE_URL=http://localhost:8000
API_TIMEOUT=5
API_RETRY_ATTEMPTS=3
REFRESH_INTERVAL=5000
```

### database.py Settings
```python
POOL_SIZE = 5                    # Connection pool size
MAX_OVERFLOW = 10               # Max overflow connections
POOL_RECYCLE = 3600             # Recycle after 1 hour
CONNECTION_RETRY_ATTEMPTS = 3   # Retry count
CONNECTION_RETRY_DELAY = 1.0    # Initial retry delay
```

### main.py Settings
```python
MODEL_RETRAIN_INTERVAL = 50     # Retrain after N predictions
DATABASE_URL = "sqlite:///./cloud.db"
MODEL_RETRAIN_INTERVAL = 50
API_TIMEOUT = 30                # Request timeout
```

---

## 🔧 Common Tasks

### Train/Retrain Model
```bash
python ml/train.py
```

### Check Database
```bash
sqlite3 cloud.db
sqlite> SELECT COUNT(*) FROM predictions;
```

### View Recent Logs
```bash
tail -f logs/anomaly_detection.log
```

### Clear Old Predictions
```bash
sqlite3 cloud.db "DELETE FROM predictions WHERE id NOT IN (SELECT id FROM predictions ORDER BY id DESC LIMIT 1000);"
```

### Reset Everything
```bash
rm cloud.db
rm ml/model.pkl ml/log_model.pkl
python ml/train.py
```

### Performance Test
```bash
pip install locust
locust -f tests/locustfile.py --host=http://localhost:8000
```

---

## 🐛 Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| "port 8000 already in use" | `lsof -i :8000` then `kill -9 <PID>` |
| "Model file not found" | Run `python ml/train.py` |
| "Database locked" | Close other connections, or restart API |
| "Dashboard not responding" | Check backend: `curl http://localhost:8000/health` |
| "High CPU usage" | Check if simulator is running, reduce frequency |
| "Predictions taking long" | Increase `API_TIMEOUT` environment variable |
| "Out of memory" | Delete old predictions or restart services |
| "Log messages saying "WARNING"" | Normal - check log levels, not errors |

---

## 📊 Monitoring Quick Commands

### Health Check
```bash
curl http://localhost:8000/health | jq '.'
```

### Recent Predictions
```bash
curl http://localhost:8000/predictions | jq '.predictions[-5:]'
```

### Model Version
```bash
cat ml/model_version.txt
```

### Last Retraining
```bash
ls -la ml/model.pkl
```

### Database Size
```bash
ls -lh cloud.db
sqlite3 cloud.db "SELECT COUNT(*) FROM predictions;"
```

### Active Logs
```bash
tail -20 logs/anomaly_detection.log
```

---

## 🤝 API Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | All good |
| 400 | Bad Request | Check input format |
| 422 | Validation Error | Check metric ranges |
| 500 | Server Error | Check logs |
| 503 | Service Unavailable | Check database connectivity |

---

## 📈 Scaling Quick Reference

### Single Machine
```
✓ Up to 100 req/s
✓ 2GB RAM sufficient
✓ SQLite database OK
✓ Good for dev/test
```

### Multiple Instances (3 replicas)
```
✓ Up to 300 req/s
✓ 4GB RAM per instance
✓ Need shared database (PostgreSQL)
✓ Need load balancer (Nginx)
✓ Good for production
```

### Kubernetes
```
✓ Auto-scaling
✓ Unlimited throughput
✓ Self-healing
✓ Easy updates
✓ Best for enterprise
```

---

## 🔐 Security Checklist

- [ ] Change CORS origins from "*" to specific domains
- [ ] Use HTTPS in production
- [ ] Add API authentication/API keys
- [ ] Configure database credentials
- [ ] Set up firewalls
- [ ] Enable request logging
- [ ] Use strong database passwords
- [ ] Regular security updates
- [ ] Backup configurations regularly

---

## 📚 Documentation Map

| Document | Purpose | Read When |
|----------|---------|-----------|
| REFACTORING_SUMMARY.md | Project overview | First |
| PRODUCTION_STATUS_REPORT.md | Quality assessment | Before deployment |
| DEPLOYMENT_GUIDE.md | Setup instructions | Deploying |
| TESTING_GUIDE.md | Testing procedures | Development |
| README.md | Project description | Initial setup |

---

## 🎓 Learning Resources

### FastAPI
- Docs: https://fastapi.tiangolo.com
- Tutorial: https://fastapi.tiangolo.com/tutorial/

### Streamlit
- Docs: https://docs.streamlit.io
- Gallery: https://streamlit.io/gallery

### SQLAlchemy
- Docs: https://docs.sqlalchemy.org
- Tutorial: https://docs.sqlalchemy.org/tutorial/

### scikit-learn
- IsolationForest: https://scikit-learn.org/stable/modules/ensemble.html#isolation-forest
- MLPRegressor: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

---

## 📞 Support

### Getting Help

**Code Issues?**
```bash
grep -r "TODO\|FIXME\|BUG" . --include="*.py"
```

**Error Messages?**
```bash
tail -f logs/anomaly_detection.log | grep ERROR
```

**Test Coverage?**
```bash
pytest --cov=backend --cov=ml --cov-report=html
```

**Performance Issues?**
```bash
# Check response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
```

---

## ✅ Pre-Deployment Checklist

- [ ] Run `pytest` - all tests pass
- [ ] Check `curl http://localhost:8000/health` - returns healthy
- [ ] Verify `ml/model.pkl` exists
- [ ] Review `logs/anomaly_detection.log` - no errors
- [ ] Test `/predict` endpoint with sample data
- [ ] Verify database connectivity
- [ ] Check disk space - at least 1GB free
- [ ] Review environment variables
- [ ] Update CORS origins
- [ ] Set LOG_LEVEL to INFO

---

## 🚀 Deployment Checklist

- [ ] Backup current database
- [ ] Update production environment variables
- [ ] Configure HTTPS/SSL
- [ ] Set up monitoring/alerts
- [ ] Configure backups
- [ ] Test health endpoint
- [ ] Run load tests
- [ ] Verify logging
- [ ] Document rollback procedure
- [ ] Notify stakeholders

---

**Last Updated:** 2026-04-01  
**Version:** 1.0  
**Status:** PRODUCTION-READY ✅
