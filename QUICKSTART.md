# Quick Reference Guide

## Start the System

### Using Docker Compose (Easiest)
```bash
docker compose up -d
```

Access:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- Airflow: http://localhost:8080 (admin/admin)

### Local Development
```bash
# Terminal 1: Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && python -m http.server 3000
```

### Kubernetes/Minikube
```bash
minikube start --memory=4096 --cpus=2
eval $(minikube docker-env)
docker build -t fraud-detection-backend:latest -f Dockerfile .
docker build -t fraud-detection-airflow:latest -f Dockerfile.airflow .
kubectl apply -f k8s/
```

## API Quick Reference

### Create Transaction
```bash
curl -X POST http://localhost:8000/api/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 100.00,
    "merchant": "Store Name",
    "category": "shopping",
    "location": "City",
    "user_id": "user_123"
  }'
```

### Get All Transactions
```bash
curl http://localhost:8000/api/transactions
```

### Get Only Fraudulent Transactions
```bash
curl http://localhost:8000/api/transactions?fraud_only=true
```

### Get Statistics
```bash
curl http://localhost:8000/api/stats
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Common Commands

### Docker
```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f backend
docker compose logs -f airflow-webserver

# Stop services
docker compose down

# Rebuild and restart
docker compose up -d --build
```

### Kubernetes
```bash
# Check pod status
kubectl get pods -n fraud-detection

# View logs
kubectl logs <pod-name> -n fraud-detection

# Get service URLs
minikube service backend -n fraud-detection --url
minikube service frontend -n fraud-detection --url

# Delete all resources
kubectl delete namespace fraud-detection
```

## Fraud Detection Rules

Transactions are flagged as fraud if:
- Amount > $10,000
- Merchant contains: "unknown", "test", "suspicious"
- Amount is negative
- ML model detects anomaly

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `AIRFLOW_HOME`: Airflow directory (default: ./airflow)
- `AIRFLOW__CORE__EXECUTOR`: Airflow executor type

## Troubleshooting

### Backend won't start
```bash
# Check database
docker ps | grep postgres

# Check logs
docker compose logs backend
```

### Frontend can't connect
- Verify backend is running on port 8000
- Check browser console for errors
- Ensure CORS is configured (already done)

### Airflow DAG not appearing
```bash
# Check DAG syntax
python -m py_compile airflow/dags/fraud_detection_dag.py

# Restart scheduler
docker compose restart airflow-scheduler
```

## File Locations

- Backend code: `backend/`
- Frontend code: `frontend/`
- Airflow DAGs: `airflow/dags/`
- K8s manifests: `k8s/`
- Docker configs: `Dockerfile`, `docker-compose.yml`
- Documentation: `README.md`, `IMPLEMENTATION.md`

## Default Credentials

**Airflow:**
- Username: admin
- Password: admin (set during first-time setup)

**PostgreSQL:**
- Username: fraud_user
- Password: fraud_pass
- Database: fraud_detection

⚠️ Change these in production!
