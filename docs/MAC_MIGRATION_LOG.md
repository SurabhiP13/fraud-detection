# Windows/WSL → macOS migration log

Date: 2026-09-18
Machine: MacBook, Apple Silicon (arm64), macOS 26.3, 24 GB RAM, 15 cores
Goal: run the whole project (Airflow retraining DAG on minikube, MLflow registry,
Streamlit UI, GitHub Actions CI/CD) on the Mac exactly as it ran under WSL2.

This is the record of what was inspected, what was changed and why, and what was
verified. The how-to lives in [SETUP_MAC.md](SETUP_MAC.md).

---

## 1. Audit — what was actually Windows/WSL-specific

| Checked | Result |
|---|---|
| CRLF line endings | none (`grep -rlI $'\r'` clean) |
| Hard-coded host paths | one: `src/data_cleaning.py` had `C:\Users\psura\...\train_merged.csv` as the standalone-run fallback |
| `/mnt/<drive>` or WSL references | only in an old comment in `k8s/data-pvc.yaml` (already replaced by the `/data` hostPath) |
| Shell tooling differences | `.github/workflows/cd.yml` used `sed -i "..."`, which errors on BSD/macOS sed |
| Docker socket | `docker-compose.yml` bind-mounted `/var/run/docker.sock`; Colima on macOS exposes `~/.colima/default/docker.sock` and creates no symlink |
| Container runtime on the Mac | no Docker Desktop installed; Colima installed but stopped (4 CPU / 6 GB, too small); `docker compose` and `docker buildx` plugins missing |
| minikube | not yet created; v1.38.1 present with drivers docker / vfkit / qemu2 |
| arm64 availability | `apache/airflow:2.9.3`, `python:3.10/3.11-slim`, `postgres:15`, `bitnami/postgresql:latest` all publish arm64; `lightgbm==4.0.0` and `psycopg2-binary==2.9.9` ship `manylinux2014_aarch64` wheels. No `--platform` overrides needed |
| Local Python | `lightgbm==4.0.0` has **no** macOS arm64 wheel (only x86_64), so a floating `>=4.5` is used for the local dev env; `libomp` needed from Homebrew |
| Latent bugs found on the way | `src/model_training.py`: `_sha256_text()` body was truncated to the literal `su` (NameError on any standalone run). Docker DAG referenced volume `fraud-detection_data` but compose created `fraud-detection_fraud-detection-data`. Compose MLflow ran `pip install mlflow` at start on `python:3.8`, giving a server version that no longer matches the 2.9.2 clients. **MLflow permission bug** (found on the first Mac run): the DAG finished green but the training log said `Failed to log to MLflow: [Errno 13] Permission denied: '/mlflow/artifacts/1'` and no model was registered — the MLflow server (root) owned `/mlflow/artifacts` while the training pod runs as uid 50000 with a non-proxied artifact root. Not macOS-specific; it would have happened on WSL too |

Git state at start: local `main` was 2 commits behind `origin/main` (README
edits only) with an untracked, more complete local README and uncommitted
minikube fixes in `k8s/`. Fast-forwarded to origin and kept the local README.

---

## 2. Changes made

### Runtime / cluster (macOS-specific)

1. **Colima is the Docker daemon.** Resized to 6 CPU / 12 GB / 60 GB
   (`colima start --cpu 6 --memory 12 --disk 60`). Replaces WSL2 + Docker Desktop.
2. **Docker CLI plugins.** `brew install docker-compose docker-buildx` and
   registered `/opt/homebrew/lib/docker/cli-plugins` in `~/.docker/config.json`
   (`cliPluginsExtraDirs`) so `docker compose` / `docker buildx` resolve.
3. **minikube on the docker driver** inside Colima:
   `minikube start --driver=docker --cpus=4 --memory=8g --disk-size=30g`.
4. **Images built into minikube's daemon** via `eval $(minikube docker-env)`,
   tags `*:v1`, on arm64 with no Dockerfile changes.
5. **`brew install libomp`** for LightGBM in the local dev env.

### New scripts (`scripts/`)

| File | Purpose |
|---|---|
| `start_cluster.sh` | Idempotent bootstrap: brew deps → plugin dir → Colima → minikube → build 4 images into the node → helm repo. Tunable via `COLIMA_*`, `MK_*`, `IMAGE_TAG`, `SKIP_BUILD` |
| `deploy.sh` | namespace, PVCs, MLflow, Streamlit, `helm upgrade --install` pinned to chart **1.15.0**; `--wait` blocks on rollouts and prints port-forward commands |
| `make_smoke_data.py` | Splits the 100-row Streamlit sample back into the 4 raw IEEE-CIS files and bootstraps to 3 000 rows, so the DAG can be run end to end without a Kaggle token |
| `fetch_data.sh` | Added `--smoke` mode (uses the generator); docker-cp fast path documented for the docker driver, `minikube cp` fallback for vfkit/qemu |

### Code fixes

| File | Change |
|---|---|
| `src/model_training.py` | Restored the truncated `_sha256_text()` |
| `src/data_cleaning.py` | Removed the `C:\Users\...` fallback; standalone runs now auto-discover the newest ingestion output under `processed_data_dir` (same pattern feature engineering and training already used) |
| `dags/fraud_detection_dag_k8s.py` | Image names and namespace read from `FRAUD_AIRFLOW_IMAGE` / `FRAUD_TRAINING_IMAGE` / `FRAUD_NAMESPACE` with the old `*:v1` defaults, so CD can point the same DAG at registry tags without `sed`-editing a file that is baked into the image. Training task's init container now also `chown`s `/mlflow` (see "MLflow permission bug" below) |
| `k8s/mlflow-deployment.yaml` | Server runs as uid/gid 50000 with an init container that chowns `/mlflow`; `HOME=/tmp` for the non-root user |
| `dags/fraud_detection_dag_docker.py` | Training task also mounts the MLflow volume (`/mlflow`) — the server uses a non-proxied artifact root, so the client writes artifacts to that path |
| `config.yaml` | `mlflow.tracking_uri` was the literal string `${MLFLOW_TRACKING_URI:-http://localhost:5000}` (YAML does not expand it). Now a plain default; the env var still wins |

### Docker Compose path

| File | Change |
|---|---|
| `docker-compose.yml` | Socket mount is `${DOCKER_SOCK:-/var/run/docker.sock}` (set to the Colima socket on macOS). MLflow builds from `Dockerfile.mlflow` (2.9.2, matching clients) instead of pip-installing at boot on Python 3.8. Explicit `name:` on volumes/network so the DockerOperator DAG's `fraud-detection_data` / `fraud-detection_fraud-detection` references resolve. Added a `pipeline` build-only profile that produces `fraud-detection-pipeline:latest` for the DAG. Scheduler reuses the webserver image. `airflow db init` → `airflow db migrate` (2.9 name). Streamlit mounts the MLflow volume |
| `Dockerfile.pipeline` | Adds `mlflow==2.9.2` so the compose DAG's training stage can register the model |

### CI/CD (`.github/workflows/`)

| File | Change |
|---|---|
| `ci.yml` | Split into `test` (Python 3.12, `pytest tests/`) and a matrix `build-and-push` job. Builds **`linux/amd64,linux/arm64`** with QEMU + Buildx so GHCR images run on the Mac's arm64 node; GHA layer cache instead of inline cache (inline cache does not support multi-platform). Action versions bumped |
| `cd.yml` | Triggers after CI succeeds (`workflow_run`) or manually. macOS-safe `sed -i.bak`. Kubeconfig: uses the runner's `~/.kube/config` unless a `KUBECONFIG` secret is set (old `runner.name == 'self-hosted'` condition never matched). Pre-pulls sha-tagged images into minikube's daemon (GHCR packages are private, and this avoids a long-lived imagePullSecret). Passes Airflow/pod-template image and the DAG's `FRAUD_*_IMAGE` env through a generated Helm values file. Chart pinned to 1.15.0 (was 1.14.0, disagreeing with the README). `helm upgrade --install` instead of uninstall/reinstall |

### Tests and local dev

| File | Change |
|---|---|
| `tests/test_smoke.py` | 6 tests: config sections, every DAG file parses, no Windows paths in `src/`/`scripts/`, full ingestion→cleaning→FE→training on synthetic data, and a regression test for the data-cleaning fallback |
| `pyproject.toml` | `dev` dependency group (pytest, pandas, numpy<2, pyyaml, scikit-learn, lightgbm>=4.5) so `uv sync --group dev && uv run pytest` works on the Mac; pytest `testpaths` |

### Docs

* `docs/SETUP_MAC.md` — the setup guide.
* `docs/MAC_MIGRATION_LOG.md` — this file.
* `README.md` — Kubernetes section now points at the scripts and docs; Docker Desktop prerequisite replaced by Colima.

---

## 3. Verification performed on the Mac (2026-09-18)

| Step | Result |
|---|---|
| `uv sync --group dev && uv run pytest -q` | 6 passed (arm64 LightGBM 4.7, full pipeline on synthetic data) |
| `colima start --cpu 6 --memory 12 --disk 60` | Ubuntu 24.04 aarch64 VM, Docker 29.5.2 |
| `minikube start --driver=docker --cpus=4 --memory=8g` | node Ready, minikube v1.38.1 |
| `docker build` ×4 inside `minikube docker-env` | airflow 2.09 GB, model-training 883 MB, mlflow 914 MB, streamlit 1.25 GB — all arm64, no Dockerfile changes |
| `docker compose --profile build-only config` (with `DOCKER_SOCK`) | valid; Colima socket resolved |
| `helm template … --version 1.15.0` with base values and with the CD override file | renders; `FRAUD_*_IMAGE` env reaches scheduler, webserver, triggerer and the KubernetesExecutor pod template |
| `./scripts/fetch_data.sh --smoke` | 4 CSVs (5.3 MB) in `/data/unzipped` on the node, owned by 50000 |
| `./scripts/deploy.sh --wait` | postgresql, scheduler, statsd, triggerer, webserver, mlflow, streamlit all Running in ~90 s |
| `airflow dags list-import-errors` | none; both DAGs listed |
| Streamlit `/_stcore/health`, MLflow `/health` | 200 / 200 |
| DAG run 1 (`fraud_detection_pipeline_k8s`) | all 5 task pods Completed, run `success`, ROC AUC 0.969 on the synthetic set — but **no model registered** (permission bug above) |
| Fix + rebuild airflow image + `kubectl apply` MLflow + rollout restart | ~2 min |
| DAG run 2 | `success`; training log: `Successfully registered model 'fraud_detection_lgbm'`; registry shows version 1 READY, artifacts at `/mlflow/artifacts/1/<run>/artifacts/model/MLmodel` |
| Model load from inside the Streamlit pod (`models:/fraud_detection_lgbm/None`) | loads. Scoring a sample row then failed with `Model n_features_ is 375 and input n_features is 314`: Streamlit shipped a stale `preprocess_artifacts/` snapshot (314 features, committed 2026-02-01) that never matched any pipeline run, smoke or real. Fixed by logging the artifacts with each model in MLflow and loading them from there (see the training and Streamlit code) |

Not executed: the GitHub Actions workflows (need a push and a registered
self-hosted runner) and the Docker Compose stack end to end (config validated
only).

## 4. Not changed / known limitations

* `Dockerfile.pipeline` still `COPY`s `data/unzipped/` into the image, so the
  compose path needs the CSVs on the host before `docker compose build`.
* `k8s/pipeline-pvc.yaml`, `k8s/mlflow-values-docker-desktop.yaml` and
  `k8s/streamlit-frontend-docker-desktop.yaml` are legacy Docker-Desktop-era
  files, not used by the scripts. Left in place, untouched.
* `models/lgb_model_last_fold.pkl` was trained on x86; LightGBM pickles are
  portable, but it is not used by the k8s path (Streamlit loads from MLflow).
* CD assumes a self-hosted runner next to the cluster. It was reviewed and
  `helm template`-checked but not executed, since that needs a registered runner.
* The DAG still uses `airflow.utils.dates.days_ago` and `schedule_interval`
  (deprecated in 2.9, removed in 3.x). Fine while the chart stays at 1.15.0;
  migrating to Airflow 3 is a separate piece of work.
