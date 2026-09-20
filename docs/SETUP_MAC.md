# macOS setup guide (Apple Silicon)

How to run the fraud-detection stack (Airflow retraining DAG + MLflow model
registry + Streamlit UI) on a MacBook with minikube. Everything here was
verified on an M-series Mac, macOS 26, 24 GB RAM. Intel Macs work the same;
only the notes marked *arm64* differ.

For the history of what changed when the project moved off Windows/WSL, see
[MAC_MIGRATION_LOG.md](MAC_MIGRATION_LOG.md).

---

## 0. TL;DR

```bash
brew install colima docker docker-buildx docker-compose minikube kubernetes-cli helm uv libomp

./scripts/start_cluster.sh          # Colima -> minikube -> build 4 images into the node
./scripts/fetch_data.sh --smoke     # synthetic data (or plain ./scripts/fetch_data.sh with a Kaggle token)
./scripts/deploy.sh --wait          # PVCs, MLflow, Streamlit, Airflow (Helm chart 1.15.0)

kubectl port-forward -n airflow svc/airflow-webserver 8080:8080   # admin / admin
kubectl exec -n airflow deploy/airflow-scheduler -- airflow dags trigger fraud_detection_pipeline_k8s
```

Then open MLflow (`svc/mlflow-service 5000`) to see the registered
`fraud_detection_lgbm` model and Streamlit (`svc/streamlit-service 8501`) to score
transactions with it.

---

## 1. How the pieces fit on a Mac

On Windows the stack ran as: WSL2 → Docker → minikube (docker driver). macOS has
no native Linux containers, so a small Linux VM stands in for WSL2:

```
 macOS host
 ├─ Colima VM (Linux, aarch64, 6 CPU / 12 GB)      <- replaces WSL2 + Docker Desktop
 │   └─ dockerd
 │       └─ "minikube" container (docker driver, 4 CPU / 8 GB)
 │           ├─ its own dockerd  <- images are built HERE (eval $(minikube docker-env))
 │           ├─ /data            <- hostPath PV; survives minikube stop/start
 │           └─ kubelet: airflow-*, mlflow, streamlit, task pods
 └─ docker / kubectl / helm / minikube CLIs (Homebrew)
```

Why Colima rather than Docker Desktop: it is free for any use, scriptable, already
on this machine, and gives the same `docker` socket the compose stack needs.
Docker Desktop works too; if you use it, skip the Colima lines and leave
`DOCKER_SOCK` unset.

Why the docker driver rather than `vfkit`/`qemu2`: the node is then a container on
the Colima daemon, so `docker cp` can push the 1 GB CSVs into `/data` in seconds
and `minikube docker-env` behaves exactly as it did on WSL. `--driver=vfkit`
also works (`brew install vfkit`, no Colima needed for the k8s path), but data
loading falls back to the slower `minikube cp`.

### Images on Apple Silicon (arm64)

Nothing needs `--platform` overrides. Every base image and pinned wheel has an
arm64 build (checked on 2026-09-18):

| Component | arm64 source |
|---|---|
| `apache/airflow:2.9.3`, `python:3.10/3.11-slim`, `postgres:15`, `busybox` | multi-arch manifests |
| `bitnami/postgresql:latest` (chart-managed metadata DB) | multi-arch manifest |
| `lightgbm==4.0.0`, `psycopg2-binary==2.9.9` | `manylinux2014_aarch64` wheels |

The CI workflow builds `linux/amd64,linux/arm64` so images from GHCR also run here.

---

## 2. Prerequisites

```bash
# Container runtime + Kubernetes tooling
brew install colima docker docker-buildx docker-compose minikube kubernetes-cli helm

# Local Python (tests, smoke data generator); libomp is LightGBM's OpenMP runtime on macOS
brew install uv libomp

# Only for the real dataset
uv tool install kaggle
```

Homebrew installs the compose/buildx plugins outside Docker's default plugin
directory. `scripts/start_cluster.sh` registers the directory automatically; to
do it by hand add this to `~/.docker/config.json`:

```json
{ "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"] }
```

Resources: the defaults (Colima 6 CPU / 12 GB / 60 GB, minikube 4 CPU / 8 GB /
30 GB) suit a 24 GB Mac. On 16 GB use `COLIMA_MEM=8 MK_MEM=6g`; the 50k-row
training run still fits.

---

## 3. Start the cluster and build images

```bash
./scripts/start_cluster.sh
```

What it does, step by step (each step is skipped when already done):

1. `brew install` anything missing from the list above, register the plugin dir.
2. `colima start --cpu 6 --memory 12 --disk 60` and `docker context use colima`.
3. `minikube start --driver=docker --cpus=4 --memory=8g --disk-size=30g`.
4. `eval $(minikube docker-env)` then `docker build` the four images
   **into minikube's daemon** with tag `v1`:
   `fraud-detection-airflow`, `fraud-detection-model-training`, `mlflow-server`,
   `fraud-detection-streamlit`. The manifests use `pullPolicy: IfNotPresent`
   against these local tags, which exist in no registry, so building anywhere
   else (e.g. plain Colima) leaves the pods in `ErrImagePull`.
5. `helm repo add apache-airflow`.

Manual equivalent if you prefer to type it:

```bash
colima start --cpu 6 --memory 12 --disk 60
minikube start --driver=docker --cpus=4 --memory=8g
eval $(minikube docker-env)
docker build -t fraud-detection-airflow:v1        -f Dockerfile .
docker build -t fraud-detection-model-training:v1 -f Dockerfile.model-training .
docker build -t mlflow-server:v1                  -f Dockerfile.mlflow .
docker build -t fraud-detection-streamlit:v1      -f streamlit_app/Dockerfile streamlit_app
```

Rebuild after changing code under `src/`, `scripts/`, `dags/` or `config.yaml`
(they are baked into the image), then `kubectl rollout restart deploy -n airflow`.

---

## 4. Load the dataset into the node

The raw CSVs are gitignored and the minikube node cannot see files on your Mac.
`fraud-data-pv` is a `hostPath` at `/data` *inside the node* (one of the few
paths minikube preserves across `minikube stop/start`; wiped by `minikube delete`).

```bash
./scripts/fetch_data.sh --smoke   # no Kaggle account: 3 000 synthetic rows shaped like IEEE-CIS
./scripts/fetch_data.sh           # real data, transaction CSVs truncated to 50k rows
./scripts/fetch_data.sh --full    # real data, complete CSVs (~1.7 GB in the node)
```

For the real data you need `~/.kaggle/kaggle.json` (kaggle.com → Settings →
Create New Token) and to have accepted the competition rules at
https://www.kaggle.com/c/ieee-fraud-detection/rules, otherwise the API returns 403.

The smoke set is generated from `streamlit_app/sample_data/raw_transactions.csv`
by `scripts/make_smoke_data.py`. It exercises every stage of the DAG end to end
and registers a model in MLflow, but model quality on it is meaningless (scores
will sit near 0). Streamlit can still score with it: each training run logs
`preprocess_artifacts/` (feature names, label encoders, group statistics) to the
same MLflow run as the model, and Streamlit downloads them for whichever model
version it loads, so the two always match. A model version trained before this
was added has no artifacts; retrain it.

Verify: `minikube ssh -- ls -lh /data/unzipped`

---

## 5. Deploy

```bash
./scripts/deploy.sh --wait
```

Equivalent commands:

```bash
kubectl create namespace airflow
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml
helm upgrade --install airflow apache-airflow/airflow \
  --version 1.15.0 -n airflow -f k8s/airflow-values.yaml --timeout 15m
```

> **Keep `--version 1.15.0`.** It is the last chart shipping Airflow **2.9.3**,
> which is what `requirements-airflow.txt` and both DAGs target. An unpinned
> install resolves to chart 1.22+ / Airflow **3.x**, where `days_ago`,
> `schedule_interval` and `is_delete_operator_pod` no longer exist.

First deploy takes 3–5 minutes (Postgres init, DB migration job, webserver boot).
Expected steady state:

```
airflow-postgresql-0        1/1 Running
airflow-scheduler-*         2/2 Running
airflow-statsd-*            1/1 Running
airflow-triggerer-0         2/2 Running
airflow-webserver-*         1/1 Running
mlflow-*                    1/1 Running
streamlit-*                 1/1 Running
```

---

## 6. Use it

Port-forward each UI in its own terminal:

```bash
kubectl port-forward -n airflow svc/airflow-webserver 8080:8080   # http://localhost:8080  admin / admin
kubectl port-forward -n airflow svc/mlflow-service    5000:5000   # http://localhost:5000
kubectl port-forward -n airflow svc/streamlit-service 8501:8501   # http://localhost:8501
```

Trigger a retraining run from the Airflow UI (`fraud_detection_pipeline_k8s`) or:

```bash
kubectl exec -n airflow deploy/airflow-scheduler -- airflow dags trigger fraud_detection_pipeline_k8s
kubectl get pods -n airflow -w        # setup-directories -> data-ingestion -> ... -> model-training
```

Each stage runs as a `KubernetesPodOperator` pod that mounts `fraud-data-pvc`
at `/opt/airflow/data`; the training pod also mounts `mlflow-pvc` at `/mlflow`
and logs to `http://mlflow-service.airflow.svc.cluster.local:5000`. When the run
finishes, MLflow shows a new version of the registered model
`fraud_detection_lgbm`, and Streamlit's "Reload Model" button picks it up.

Task pods are kept after completion (`is_delete_operator_pod=False`) so you can
`kubectl logs -n airflow <pod>` them. Clean up with
`kubectl delete pod -n airflow -l dag_id=fraud_detection_pipeline_k8s`.

---

## 7. Day-to-day

| Task | Command |
|---|---|
| Pause everything (keeps data, images, DB) | `minikube stop && colima stop` |
| Resume | `./scripts/start_cluster.sh` (SKIP_BUILD=1 to skip rebuilding) |
| Rebuild after a code change | `./scripts/start_cluster.sh && kubectl rollout restart deploy -n airflow` |
| Full reset | `minikube delete` (then `start_cluster.sh`, `fetch_data.sh`, `deploy.sh`) |
| Run the tests | `uv sync --group dev && uv run pytest -q` |
| Regenerate smoke data only | `uv run --group dev python scripts/make_smoke_data.py` |

---

## 8. Docker Compose alternative (no Kubernetes)

`docker-compose.yml` runs Postgres + Airflow (LocalExecutor) + MLflow + Streamlit
on the Colima daemon; the `fraud_detection_pipeline_docker` DAG launches each
stage as a sibling container through the Docker socket.

```bash
export DOCKER_SOCK="$HOME/.colima/default/docker.sock"   # Colima has no /var/run/docker.sock
./scripts/fetch_data.sh --smoke --skip-load               # Dockerfile.pipeline COPYs data/unzipped
docker compose --profile build-only build                 # also builds fraud-detection-pipeline:latest
docker compose up -d
```

UIs: Airflow http://localhost:8080 (`airflow`/`airflow`), MLflow :5000, Streamlit :8501.

---

## 9. CI/CD

* **CI** (`.github/workflows/ci.yml`) runs on every push/PR: `pytest tests/`
  (config, DAG syntax, end-to-end pipeline on synthetic data), then on branch
  pushes builds all four images for `linux/amd64,linux/arm64` and pushes them to
  `ghcr.io/<owner>/<image>:{latest,<sha>}`.
* **CD** (`.github/workflows/cd.yml`) runs on a **self-hosted runner** after CI
  succeeds on `main` (or manually). It pulls the sha-tagged images into the
  cluster node, points the manifests and the Helm values at them, runs
  `helm upgrade --install` with chart 1.15.0, and triggers the DAG.

To make CD deploy to the minikube on this Mac, register the Mac as a runner:
GitHub repo → Settings → Actions → Runners → New self-hosted runner → macOS /
ARM64, follow the `./config.sh` steps, then `./run.sh` (or install it as a
service). The runner needs Colima + minikube running and uses your
`~/.kube/config`; no `KUBECONFIG` secret is required. Set the `KUBECONFIG`
secret (base64 of a kubeconfig) only when targeting a remote cluster.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker: unknown command: docker compose` | Plugin dir not registered — see §2, or run `start_cluster.sh` |
| `Cannot connect to the Docker daemon at unix:///var/run/docker.sock` | Colima not running (`colima start`) or wrong context (`docker context use colima`) |
| Pods stuck in `ErrImagePull` / `ImagePullBackOff` | Images were built against Colima, not minikube. Re-run `start_cluster.sh` (it runs `eval $(minikube docker-env)` first) |
| `airflow-run-airflow-migrations` job never completes | Metadata Postgres not ready yet; `kubectl logs -n airflow job/airflow-run-airflow-migrations` |
| DAG import error `No module named 'airflow.utils.dates'` | Chart not pinned → Airflow 3.x. `helm uninstall airflow -n airflow` and redeploy with `--version 1.15.0` |
| `data_ingestion` pod: `FileNotFoundError: /opt/airflow/data/unzipped/train_transaction.csv` | Data not loaded into the node: `./scripts/fetch_data.sh [--smoke]` |
| Task pod `PermissionError` on `/opt/airflow/data` | `/data` ownership in node reset (after `minikube delete`). `fetch_data.sh` re-applies `chown 50000:50000` |
| DAG green but no model in MLflow; training log says `Permission denied: '/mlflow/artifacts/...'` | `/mlflow` on the PVC is root-owned (old MLflow manifest). `kubectl apply -f k8s/mlflow-deployment.yaml` — the current one runs the server as uid 50000 and chowns the volume; the training pod's init container does the same |
| `kaggle: 403 Forbidden` | Accept the competition rules on the Kaggle website first |
| `OSError: dlopen ... libomp.dylib` when running tests | `brew install libomp` |
| `minikube start` says "docker driver not healthy" | Colima stopped or `DOCKER_HOST` points at a stale socket; `unset DOCKER_HOST; colima start` |
| CD run fails with `sed: 1: "...": invalid command code` | Only affects the old workflow; the current one uses `sed -i.bak`, which BSD sed accepts |
