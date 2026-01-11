# Fraud Detection System

A complete fraud detection system with ML-based transaction analysis, real-time monitoring UI, ETL pipeline managed by Airflow, and containerized deployment using Docker and Kubernetes.

## 🏗️ Architecture

The system consists of the following components:

- **Backend (FastAPI)**: RESTful API for transaction management and fraud detection
- **Frontend (HTML/CSS/JS)**: Real-time dashboard for monitoring transactions
- **ML Model**: Machine learning-based fraud detection using scikit-learn
- **Database (PostgreSQL)**: Stores transaction data
- **Airflow**: ETL pipeline orchestration for batch processing
- **Docker & Kubernetes**: Container orchestration for scalable deployment

## 📋 Features

- ✅ Real-time fraud detection using ML model
- ✅ Interactive web UI with transaction monitoring
- ✅ RESTful API for transaction processing
- ✅ Automated ETL pipeline with Airflow
- ✅ PostgreSQL database for data persistence
- ✅ Docker containerization
- ✅ Kubernetes/Minikube deployment manifests
- ✅ Scalable microservices architecture

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Minikube (for Kubernetes deployment)
- kubectl

### Option 1: Docker Compose (Recommended for Quick Start)

1. **Clone the repository**
```bash
git clone https://github.com/ayush-mangukia/fraud-detection.git
cd fraud-detection
```

2. **Start all services**
```bash
docker-compose up -d
```

3. **Access the applications**
- Frontend UI: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Airflow UI: http://localhost:8080

4. **Initialize Airflow (first time only)**
```bash
docker-compose exec airflow-webserver airflow db init
docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### Option 2: Local Development

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Start PostgreSQL**
```bash
docker run -d \
  --name postgres \
  -e POSTGRES_USER=fraud_user \
  -e POSTGRES_PASSWORD=fraud_pass \
  -e POSTGRES_DB=fraud_detection \
  -p 5432:5432 \
  postgres:15
```

3. **Start the backend**
```bash
export DATABASE_URL=postgresql://fraud_user:fraud_pass@localhost:5432/fraud_detection
uvicorn backend.main:app --reload --port 8000
```

4. **Start the frontend**
```bash
cd frontend
python -m http.server 3000
```

5. **Start Airflow**
```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow webserver --port 8080 &
airflow scheduler &
```

### Option 3: Kubernetes with Minikube

1. **Start Minikube**
```bash
minikube start --memory=4096 --cpus=2
```

2. **Build Docker images**
```bash
eval $(minikube docker-env)
docker build -t fraud-detection-backend:latest -f Dockerfile .
docker build -t fraud-detection-airflow:latest -f Dockerfile.airflow .
```

3. **Deploy to Kubernetes**
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/airflow.yaml
```

4. **Access services**
```bash
# Get service URLs
minikube service backend -n fraud-detection --url
minikube service frontend -n fraud-detection --url
minikube service airflow-webserver -n fraud-detection --url
```

## 📖 Usage

### Using the Web Interface

1. Open the frontend at http://localhost:3000
2. Click **"Start Transaction Processing"** to initiate the ETL pipeline
3. Click **"Add Test Transaction"** to create sample transactions
4. View real-time fraud detection results in the dashboard
5. Toggle **"Show Fraudulent Transactions Only"** to filter results

### API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/transactions/start` - Start transaction processing
- `POST /api/transactions` - Create a new transaction
- `GET /api/transactions` - List all transactions
- `GET /api/transactions/{id}` - Get specific transaction
- `GET /api/stats` - Get fraud detection statistics

### API Example

```bash
# Create a transaction
curl -X POST http://localhost:8000/api/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1250.50,
    "merchant": "Online Store XYZ",
    "category": "shopping",
    "location": "New York, USA",
    "user_id": "user_123"
  }'

# Get all transactions
curl http://localhost:8000/api/transactions

# Get fraud statistics
curl http://localhost:8000/api/stats
```

## 🤖 ML Model

The fraud detection model uses a combination of:
- **Isolation Forest** for anomaly detection
- **Rule-based heuristics** for known fraud patterns

### Fraud Detection Rules

- Transactions over $10,000 are flagged
- Transactions from suspicious merchants (containing "unknown", "test", "suspicious")
- Negative transaction amounts
- Anomalies detected by the ML model

### Model Training

To train the model with custom data:

```python
from backend.fraud_detector import FraudDetector

detector = FraudDetector()
detector.train(X_train, y_train)
```

## 🔄 ETL Pipeline

The Airflow DAG (`fraud_detection_etl`) runs every 15 minutes and performs:

1. **Extract**: Pull transactions from source
2. **Transform**: Validate and clean transaction data
3. **Detect Fraud**: Apply ML model to detect fraudulent transactions
4. **Load**: Store processed transactions in database
5. **Alert**: Send notifications for detected fraud

## 📊 Database Schema

```sql
Table: transactions
- id (INTEGER, PRIMARY KEY)
- amount (FLOAT)
- merchant (VARCHAR)
- category (VARCHAR)
- timestamp (DATETIME)
- location (VARCHAR)
- user_id (VARCHAR)
- is_fraud (BOOLEAN)
- fraud_score (FLOAT)
- created_at (DATETIME)
- updated_at (DATETIME)
```

## 🛠️ Configuration

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string (default: SQLite for local dev)
- `AIRFLOW_HOME`: Airflow home directory
- `AIRFLOW__CORE__EXECUTOR`: Airflow executor type

### Docker Compose Configuration

Edit `docker-compose.yml` to modify:
- Port mappings
- Database credentials
- Resource limits

### Kubernetes Configuration

Edit files in `k8s/` directory to modify:
- Replica counts
- Resource requests/limits
- Service types (LoadBalancer/NodePort)

## 🧪 Testing

```bash
# Run backend tests
pytest backend/

# Test API endpoints
curl http://localhost:8000/health
```

## 📦 Project Structure

```
fraud-detection/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Database models
│   ├── database.py          # Database configuration
│   └── fraud_detector.py    # ML fraud detection
├── frontend/
│   ├── index.html           # Main UI
│   ├── styles.css           # Styling
│   └── app.js               # Frontend logic
├── airflow/
│   ├── dags/
│   │   └── fraud_detection_dag.py  # ETL DAG
│   └── airflow.cfg          # Airflow configuration
├── k8s/
│   ├── namespace.yaml       # Kubernetes namespace
│   ├── postgres.yaml        # Database deployment
│   ├── backend.yaml         # Backend deployment
│   ├── frontend.yaml        # Frontend deployment
│   └── airflow.yaml         # Airflow deployment
├── ml_model/                # ML model storage
├── Dockerfile               # Backend container
├── Dockerfile.airflow       # Airflow container
├── docker-compose.yml       # Docker Compose config
├── requirements.txt         # Python dependencies
└── README.md
```

## 🔐 Security Notes

- Change default passwords in production
- Use environment variables for sensitive data
- Enable HTTPS for production deployments
- Implement proper authentication/authorization
- Regular security audits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

MIT License

## 🐛 Troubleshooting

### Backend won't start
- Check if PostgreSQL is running: `docker ps | grep postgres`
- Verify DATABASE_URL environment variable
- Check logs: `docker-compose logs backend`

### Frontend can't connect to API
- Ensure backend is running on port 8000
- Check CORS settings in backend/main.py
- Verify API_BASE_URL in frontend/app.js

### Airflow DAG not visible
- Check DAG file for syntax errors
- Restart scheduler: `docker-compose restart airflow-scheduler`
- Check logs: `docker-compose logs airflow-scheduler`

### Minikube issues
- Increase memory: `minikube start --memory=4096`
- Check pod status: `kubectl get pods -n fraud-detection`
- View logs: `kubectl logs <pod-name> -n fraud-detection`

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review logs for error messages

---

**Built with ❤️ using FastAPI, React, Airflow, and Kubernetes**