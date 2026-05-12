# Fraud Detection Pipeline

End-to-end credit-card fraud detection system built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset. The project covers the full MLOps lifecycle: data ingestion, cleaning, feature engineering, model training with MLflow tracking, workflow orchestration via Airflow, and a Streamlit UI for interactive scoring.

## Architecture

```
  ╔═════════════════════════════════════════════════════════════════╗
  ║                    Apache Airflow DAG                           ║
  ║                  (manually triggered)                           ║
  ║                                                                 ║
  ║  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     ║
  ║  │ Ingestion │─▶│ Cleaning  │─▶│ Features  │─▶│ Training  │     ║
  ║  └───────────┘  └───────────┘  └───────────┘  └──────┬────┘     ║
  ║                                                      │          ║
  ╚══════════════════════════════════════════════════════╪══════════╝
                                                         │ register
                                                         ▼
                                               ┌──────────────────┐
                                               │     MLflow       │
                                               │  Model Registry  │
                                               └────────┬─────────┘
                                                        │ serve latest
                                                        ▼
                                               ┌──────────────────┐
                                               │  Streamlit App   │
                                               │  (predictions)   │
                                               └──────────────────┘
```

## Features

- **LightGBM model** with K-fold cross-validation, configurable via [config.yaml](config.yaml).
- **MLflow** for experiment tracking, model registry, and serving.
- **Airflow DAGs** for orchestration of training/retraining the model— both [Docker Compose](dags/fraud_detection_dag_docker.py) and [Kubernetes](dags/fraud_detection_dag_k8s.py) variants.
- **Streamlit frontend** ([streamlit_app/app.py](streamlit_app/app.py)) that loads the latest registered model and scores a transaction.
- **Kubernetes manifests** under [k8s/](k8s/) for deploying Airflow, MLflow, and the Streamlit app on a cluster (Docker Desktop friendly).

## Project Structure

```
fraud-detection/
├── src/                    # Pipeline library code
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── scripts/                # CLI entry points for each stage
├── dags/                   # Airflow DAGs (Docker + K8s)
├── streamlit_app/          # Prediction UI
├── k8s/                    # Kubernetes manifests
├── models/                 # Saved model artifacts
├── config.yaml             # Pipeline configuration
├── docker-compose.yml      # Local stack (Airflow + Postgres + MLflow)
├── Dockerfile              # Airflow image
├── Dockerfile.pipeline     # Pipeline runner image
├── Dockerfile.mlflow       # MLflow server image
└── Dockerfile.model-training
```

## Quick Start (Docker Compose)

Prerequisites: Docker Desktop,  the IEEE-CIS dataset CSVs.

1. **Place the raw data** in `./data/unzipped/` — expecting `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`.

2. **Build and start the stack:**
   ```bash
   docker compose up -d --build
   ```
   This brings up Postgres, the Airflow webserver/scheduler, MLflow, and the Streamlit app.

3. **Open the UIs:**
   - Airflow — http://localhost:8080 (default `airflow` / `airflow`)
   - MLflow — http://localhost:5000
   - Streamlit — http://localhost:8501

4. **Trigger the pipeline** from the Airflow UI: enable and run `fraud_detection_pipeline_docker`. The DAG runs ingestion → cleaning → feature engineering → training, logging the run and registering the model in MLflow.

5. **Score transactions** in the Streamlit app — it pulls the latest registered model from MLflow and predicts on transaction.


## Kubernetes Deployment

Manifests in [k8s/](k8s/) deploy the same stack to a Kubernetes cluster.

```bash
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/pipeline-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml
# Airflow via Helm using k8s/airflow-values.yaml
helm install airflow apache-airflow/airflow -f k8s/airflow-values.yaml
```

Use the `fraud_detection_pipeline_k8s` DAG which runs each stage as a `KubernetesPodOperator`.

## Configuration

All paths, model hyperparameters, cross-validation settings, and MLflow tracking URI live in [config.yaml](config.yaml).

## Tech Stack

Python 3.12 · LightGBM · pandas · scikit-learn · MLflow · Apache Airflow · Streamlit · FastAPI · Docker · Kubernetes · PostgreSQL
