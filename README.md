# Fraud Detection Pipeline

End-to-end credit-card fraud detection system built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset. The project covers the full MLOps lifecycle: data ingestion, cleaning, feature engineering, model training with MLflow tracking, workflow orchestration via Airflow, and a Streamlit UI for interactive scoring.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Ingestion   │───▶│   Cleaning   │───▶│   Features   │───▶│   Training   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
              ┌────────────────────────────────────────────────────┘
              ▼
       ┌──────────────┐       ┌──────────────────┐
       │    MLflow    │◀─────▶│  Streamlit App   │
       │  (registry)  │       │  (predictions)   │
       └──────────────┘       └──────────────────┘
                  Orchestrated by Apache Airflow
```

## Features

- **Modular pipeline** — each stage (ingest, clean, feature-engineer, train) is an independent Python module under [src/](src/) with a matching runner in [scripts/](scripts/).
- **LightGBM model** with K-fold cross-validation, configurable via [config.yaml](config.yaml).
- **MLflow** for experiment tracking, model registry, and serving.
- **Airflow DAGs** for orchestration — both [Docker Compose](dags/fraud_detection_dag_docker.py) and [Kubernetes](dags/fraud_detection_dag_k8s.py) variants.
- **Streamlit frontend** ([streamlit_app/app.py](streamlit_app/app.py)) that loads the latest registered model and scores sample transactions.
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

Prerequisites: Docker Desktop, ~8 GB free RAM, the IEEE-CIS dataset CSVs.

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

5. **Score transactions** in the Streamlit app — it pulls the latest registered model from MLflow and predicts on sample rows.

## Running Stages Manually

Each stage can be executed standalone for local debugging:

```bash
uv sync                                  # install deps
python scripts/run_data_ingestion.py
python scripts/run_data_cleaning.py
python scripts/run_feature_engineering.py
python scripts/run_model_training.py
```

Paths and hyperparameters are read from [config.yaml](config.yaml).

## Kubernetes Deployment

Manifests in [k8s/](k8s/) deploy the same stack to a Kubernetes cluster. The `*-docker-desktop.yaml` variants are tuned for the bundled K8s in Docker Desktop:

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

All paths, model hyperparameters, cross-validation settings, and MLflow tracking URI live in [config.yaml](config.yaml). Highlights:

- `data_cleaning.null_threshold` — drop columns above this null ratio (default `0.9`)
- `model.lightgbm.*` — LightGBM hyperparameters
- `cross_validation.n_splits` — number of CV folds
- `mlflow.tracking_uri` — defaults to `http://localhost:5000`, overridable via `MLFLOW_TRACKING_URI`

## Tech Stack

Python 3.12 · LightGBM · pandas · scikit-learn · MLflow · Apache Airflow · Streamlit · FastAPI · Docker · Kubernetes · PostgreSQL
