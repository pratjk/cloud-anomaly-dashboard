# Cloud Anomaly Detection - Production Deployment Guide

## Quick Start

### Prerequisites
```bash
python 3.9+
pip install -r requirements.txt
```

### Initial Setup
```bash
# 1. Train initial model
python ml/train.py

# 2. Start backend API
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Start dashboard (in another terminal)
streamlit run dashboard/app.py

# 4. Start log simulator (in another terminal) - OPTIONAL
python simulator/log_simulator.py
```

---

## Environment Configuration

### Backend (.env or environment variables)

```bash
# API Configuration
API_HOST=0.0.0.0              # Listen on all interfaces (production)
API_PORT=8000                 # API port
API_TIMEOUT=30                # Request timeout in seconds

# Database Configuration
DATABASE_URL=sqlite:///./cloud.db  # Or provide PostgreSQL/MySQL URL
DB_TIMEOUT=30                 # Database connection timeout

# Model Configuration
MODEL_RETRAIN_INTERVAL=50    # Retrain after N predictions
MODEL_VERSION_FILE=ml/model_version.txt
METRIC_MODEL_PATH=ml/model.pkl
LOG_MODEL_PATH=ml/log_model.pkl

# Logging
LOG_LEVEL=INFO               # INFO for production, DEBUG for troubleshooting
```

### Dashboard (.streamlit/config.toml)

```toml
[server]
port = 8501
headless = true
logger.level = "info"

[client]
showErrorDetails = false
toolbarMode = "minimal"

[theme]
primaryColor = "#00f5ff"
backgroundColor = "#020617"
secondaryBackgroundColor = "#0f172a"
textColor = "#ffffff"
font = "monospace"
```

---

## Deployment Scenarios

### 1. Single Machine Deployment

```bash
# Terminal 1: Backend
cd /path/to/project
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
streamlit run dashboard/app.py --server.port=8501

# Terminal 3: Simulator (optional, for testing)
python simulator/log_simulator.py
```

**Health Check:**
```bash
curl http://localhost:8000/health
curl http://localhost:8501
```

---

### 2. Docker Deployment

**Dockerfile (Backend)**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create database on startup
RUN mkdir -p ml data/raw data/processed logs

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: sqlite:///./cloud.db
      LOG_LEVEL: INFO
    volumes:
      - ./cloud.db:/app/cloud.db
      - ./ml:/app/ml
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    image: python:3.10-slim
    ports:
      - "8501:8501"
    working_dir: /app
    command: bash -c "pip install streamlit requests plotly pandas streamlit-autorefresh && streamlit run dashboard/app.py"
    volumes:
      - ./:/app
    depends_on:
      - backend
```

**Build and Run:**
```bash
docker-compose up -d
docker-compose logs -f
```

---

### 3. Kubernetes Deployment

**k8s/backend-deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: anomaly-backend
  template:
    metadata:
      labels:
        app: anomaly-backend
    spec:
      containers:
      - name: backend
        image: anomaly-detection:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database_url
        - name: LOG_LEVEL
          value: "INFO"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"

---
apiVersion: v1
kind: Service
metadata:
  name: anomaly-detection-service
spec:
  selector:
    app: anomaly-backend
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: LoadBalancer
```

**Deploy:**
```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl get pods
kubectl logs <pod-name>
```

---

## Monitoring & Health Checks

### Health Endpoint

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "database": {
    "healthy": true,
    "status": "Connected - 150 records in database"
  },
  "models": {
    "metric_model_available": true,
    "log_model_available": true,
    "model_version_file_available": true
  },
  "model_version": {
    "valid": true,
    "version": "v3"
  }
}
```

### Prometheus Metrics (recommended addition)

```python
# Add to backend/main.py
from prometheus_client import Counter, Histogram, generate_latest

predictions_made = Counter('predictions_total', 'Total predictions', ['result'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
errors_total = Counter('errors_total', 'Total errors', ['type'])

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

Then scrape with Prometheus:
```yaml
scrape_configs:
  - job_name: 'anomaly-detection'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

---

## Troubleshooting

### Issue: "Model file not found"
**Solution:**
```bash
python ml/train.py
# Or provide METRIC_MODEL_PATH pointing to existing model
```

### Issue: "Database connection failed"
**Solution:**
```bash
# Check database exists
ls -la cloud.db

# If using external database, verify connection string
DATABASE_URL="postgresql://user:pass@host:5432/dbname"
```

### Issue: "Predictions taking too long"
**Solution:**
```bash
# 1. Check if model is too large
ls -lh ml/model.pkl

# 2. Check database query performance
# Add indices to predictions table:
CREATE INDEX idx_timestamp ON predictions(timestamp);
CREATE INDEX idx_prediction ON predictions(prediction);

# 3. Increase API timeout
API_TIMEOUT=60
```

### Issue: "Dashboard shows 'Connecting...' infinitely"
**Solution:**
```bash
# 1. Verify backend is running
curl http://localhost:8000/health

# 2. Check firewall rules
netstat -tln | grep 8000

# 3. Check logs
tail -f logs/anomaly_detection.log

# 4. Try without auto-refresh temporarily
# Edit dashboard/app.py and comment out st_autorefresh
```

### Issue: "Out of memory"
**Solution:**
```bash
# 1. Increase swap (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. Clean old predictions (keep last 1000)
sqlite3 cloud.db "DELETE FROM predictions WHERE id NOT IN (SELECT id FROM predictions ORDER BY id DESC LIMIT 1000);"

# 3. Reduce model.pkl size (retrain with fewer features if needed)
```

---

## Performance Tuning

### Database Optimization
```sql
-- Create indices for faster queries
CREATE INDEX idx_prediction_timestamp ON predictions(timestamp);
CREATE INDEX idx_prediction_result ON predictions(prediction);
CREATE INDEX idx_prediction_cause ON predictions(cause);

-- Vacuum to reclaim space
VACUUM;

-- Analyze for query optimization
ANALYZE;
```

### FastAPI Optimization
```python
# Add to main.py
from fastapi.middleware.gzip import GZIPMiddleware

app.add_middleware(GZIPMiddleware, minimum_size=1000)

# Increase worker count in production
# uvicorn backend.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Streamlit Optimization
```python
# Add caching
@st.cache_data(ttl=60)
def fetch_predictions():
    return requests.get(f"{API_BASE_URL}/predictions").json()

# Reduce refresh rate
st_autorefresh(interval=10000)  # 10 seconds instead of 5
```

---

## Security Checklist

- [ ] Change `CORS_ORIGINS` from `"*"` to specific domains:
  ```python
  CORS_ORIGINS = ["http://localhost:3000", "https://yourdomain.com"]
  ```

- [ ] Use environment variables for sensitive data:
  ```python
  SECRET_KEY = os.getenv("SECRET_KEY")
  DATABASE_PASSWORD = os.getenv("DB_PASSWORD")
  ```

- [ ] Enable HTTPS in production:
  ```bash
  uvicorn backend.main:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
  ```

- [ ] Add rate limiting:
  ```python
  from slowapi import Limiter
  limiter = Limiter(key_func=get_remote_address)
  @app.post("/predict")
  @limiter.limit("100/minute")
  ```

- [ ] Add authentication (optional):
  ```python
  from fastapi.security import HTTPBearer
  security = HTTPBearer()
  @app.post("/predict")
  def predict(credentials: HTTPAuthCredentials = Depends(security)):
      # Validate token
  ```

- [ ] Configure database credentials securely
- [ ] Use network isolation (VPC, security groups)
- [ ] Enable encryption at rest (encrypted database)
- [ ] Set up audit logging
- [ ] Regular backups

---

## Backup & Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/anomaly-detection"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
cp cloud.db $BACKUP_DIR/cloud_$TIMESTAMP.db

# Backup models
cp ml/model.pkl $BACKUP_DIR/model_$TIMESTAMP.pkl
cp ml/log_model.pkl $BACKUP_DIR/log_model_$TIMESTAMP.pkl

# Compress
tar -czf $BACKUP_DIR/backup_$TIMESTAMP.tar.gz $BACKUP_DIR/*.db $BACKUP_DIR/*.pkl

# Keep only last 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup complete: $TIMESTAMP"
```

**Setup daily backup:**
```bash
# Add to crontab
0 2 * * * /path/to/backup.sh
```

### Recovery

```bash
# Restore database
cp /backups/anomaly-detection/cloud_20260315_020000.db ./cloud.db

# Restore models
cp /backups/anomaly-detection/model_20260315_020000.pkl ./ml/model.pkl

# Restart services
systemctl restart anomaly-detection-backend
```

---

## Scaling Strategies

### Horizontal Scaling (Multiple Instances)

```yaml
# Load balancer (Nginx)
upstream anomaly_backends {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://anomaly_backends;
    }
}
```

### Vertical Scaling (Larger Machine)

- Increase database pool size: `POOL_SIZE=20`
- Increase FastAPI workers: `--workers 8`
- Increase machine RAM: 4GB → 16GB
- Use faster CPU for model inference

### Database Scaling

```python
# Use PostgreSQL instead of SQLite
DATABASE_URL = "postgresql+psycopg2://user:pass@host:5432/anomaly_db"

# Or use a cloud database
DATABASE_URL = "mysql+pymysql://user:pass@aws-rds.amazonaws.com/anomaly_db"
```

---

## Maintenance Tasks

### Weekly
- [ ] Review error logs
- [ ] Check disk space
- [ ] Monitor prediction accuracy
- [ ] Verify backups completed

### Monthly
- [ ] Retrain model with latest data
- [ ] Update dependencies: `pip install -U -r requirements.txt`
- [ ] Review model performance metrics
- [ ] Cleanup old prediction records

### Quarterly
- [ ] Full system test
- [ ] Capacity planning
- [ ] Security audit
- [ ] Disaster recovery drill

---

## Support & Troubleshooting

**Logs Location:**
```bash
logs/anomaly_detection.log  # Application logs
cloud.db                     # SQLite database
ml/model_version.txt         # Model version tracking
```

**Debug Mode:**
```bash
# Set environment
export LOG_LEVEL=DEBUG

# View detailed logs
tail -f logs/anomaly_detection.log | grep -i error

# Check API responses
curl -v http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"cpu_usage": 50, "memory_usage": 60, "disk_io": 100, "network_traffic": 200}'
```

---

## References

- FastAPI Docs: https://fastapi.tiangolo.com
- SQLAlchemy Docs: https://docs.sqlalchemy.org
- Streamlit Docs: https://docs.streamlit.io
- Docker Docs: https://docs.docker.com
- Kubernetes Docs: https://kubernetes.io/docs/

---

**Last Updated:** 2026-04-01  
**Version:** 1.0  
**Status:** PRODUCTION-READY
