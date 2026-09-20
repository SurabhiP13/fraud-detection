# Fraud Detection Pipeline

End-to-end credit-card fraud detection system built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset. The project covers the full MLOps lifecycle: data ingestion, cleaning, feature engineering, model training with MLflow tracking, workflow orchestration via Airflow, and a Streamlit UI for interactive scoring.

## Architecture

```
  ╔══════════════════════════════════════════════════════════════════╗
  ║                    Apache Airflow DAG                           ║
  ║                  (manually triggered)                           ║
  ║                                                                 ║
  ║  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   ║
  ║  │ Ingestion │─▶│ Cleaning  │─▶│ Features  │─▶│ Training  │   ║
  ║  └───────────┘  └───────────┘  └───────────┘  └─────┬─────┘   ║
  ║                                                      │         ║
  ╚══════════════════════════════════════════════════════╪═════════╝
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

## Kubernetes Deployment (minikube on macOS)

Full guide: **[docs/SETUP_MAC.md](docs/SETUP_MAC.md)**. What changed when the project
moved off Windows/WSL: [docs/MAC_MIGRATION_LOG.md](docs/MAC_MIGRATION_LOG.md).

Manifests in [k8s/](k8s/) deploy the same stack to a Kubernetes cluster. The
`*-docker-desktop.yaml` variants are legacy files from the Docker Desktop era and are
not used by the scripts.

### Prerequisites

```bash
brew install colima docker docker-buildx docker-compose minikube kubernetes-cli helm uv libomp
uv tool install kaggle                 # only for the real dataset download
```

Colima provides the Docker daemon (no Docker Desktop needed). Works as-is on Apple
Silicon — every base image (`apache/airflow:2.9.3`, `postgres:15`, `python:3.11-slim`,
`busybox`, `bitnami/postgresql`) publishes an arm64 manifest, and all pinned Python
deps have `aarch64` wheels or are pure-Python. No `--platform` overrides needed.

### 1. Start the cluster and build the images

```bash
./scripts/start_cluster.sh             # Colima -> minikube (docker driver) -> build 4 images into the node
```

Images must be built **into minikube's own Docker daemon** (the script runs
`eval $(minikube docker-env)` first), since the manifests use `pullPolicy: IfNotPresent`
against local tags that exist in no registry.

### 2. Load the dataset

The raw CSVs are gitignored, and the minikube node cannot see host project files.
[scripts/fetch_data.sh](scripts/fetch_data.sh) copies them into the node's persistent
`/data` directory:

```bash
./scripts/fetch_data.sh --smoke        # no Kaggle account: small synthetic dataset, runs the DAG end to end
./scripts/fetch_data.sh                # real data (50k-row sample); add --full for the complete CSVs
```

Real data requires a Kaggle API token at `~/.kaggle/kaggle.json` and acceptance of the
[competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules).

### 3. Deploy

```bash
./scripts/deploy.sh --wait
```

which is equivalent to:

```bash
kubectl create namespace airflow
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml

# Pin the chart — the version matters, see below
helm upgrade --install airflow apache-airflow/airflow \
  --version 1.15.0 -n airflow -f k8s/airflow-values.yaml
```

> **Pin `--version 1.15.0`.** That is the last chart shipping Airflow **2.9.3**, which is what
> [requirements-airflow.txt](requirements-airflow.txt) and the images are built against.
> An unpinned `helm install` now resolves to chart 1.22+ / Airflow **3.x**, which breaks both
> DAGs: `airflow.utils.dates.days_ago` was removed, `schedule_interval` was renamed to
> `schedule`, and `is_delete_operator_pod` no longer exists on `KubernetesPodOperator`.

Then port-forward and trigger `fraud_detection_pipeline_k8s`, which runs each stage as a
`KubernetesPodOperator`:

```bash
kubectl port-forward -n airflow svc/airflow-webserver 8080:8080
```

### Storage notes

`fraud-data-pv` ([k8s/data-pvc.yaml](k8s/data-pvc.yaml)) is a `hostPath` on `/data` *inside the
minikube node* — one of the few directories minikube preserves across `minikube stop/start`.
It does **not** survive `minikube delete`; re-run `fetch_data.sh` after recreating the cluster.

## Configuration

All paths, model hyperparameters, cross-validation settings, and MLflow tracking URI live in [config.yaml](config.yaml). Highlights:

- `data_cleaning.null_threshold` — drop columns above this null ratio (default `0.9`)
- `model.lightgbm.*` — LightGBM hyperparameters
- `cross_validation.n_splits` — number of CV folds
- `mlflow.tracking_uri` — defaults to `http://localhost:5000`, overridable via `MLFLOW_TRACKING_URI`

## Tech Stack

Python 3.12 · LightGBM · pandas · scikit-learn · MLflow · Apache Airflow · Streamlit · FastAPI · Docker · Kubernetes · PostgreSQL
