#!/usr/bin/env bash
#
# macOS bootstrap: bring up the container runtime (Colima), a minikube cluster on
# the docker driver, and build the four project images straight into minikube's
# own Docker daemon so the manifests' `pullPolicy: IfNotPresent` local tags resolve.
#
# Idempotent -- safe to re-run; each step is skipped when already done.
#
# Tunables (env vars):
#   COLIMA_CPU=6 COLIMA_MEM=12 COLIMA_DISK=60   # Colima VM size (GiB)
#   MK_CPUS=4 MK_MEM=8g MK_DISK=30g            # minikube node size
#   IMAGE_TAG=v1                               # tag for the built images
#   SKIP_BUILD=1                               # only start runtime + cluster
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLIMA_CPU="${COLIMA_CPU:-6}"
COLIMA_MEM="${COLIMA_MEM:-12}"
COLIMA_DISK="${COLIMA_DISK:-60}"
MK_CPUS="${MK_CPUS:-4}"
MK_MEM="${MK_MEM:-8g}"
MK_DISK="${MK_DISK:-30g}"
IMAGE_TAG="${IMAGE_TAG:-v1}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# --- 0. Tooling ----------------------------------------------------------------
log "Checking Homebrew tooling"
missing=()
for f in colima docker docker-buildx docker-compose minikube kubernetes-cli helm; do
  brew list --formula "$f" >/dev/null 2>&1 || missing+=("$f")
done
if ((${#missing[@]})); then
  echo "Installing: ${missing[*]}"
  brew install "${missing[@]}"
fi

# Homebrew installs the compose/buildx plugins outside Docker's default plugin
# path; register that directory once so `docker compose` / `docker buildx` work.
DOCKER_CFG="$HOME/.docker/config.json"
mkdir -p "$HOME/.docker"
if ! grep -q cliPluginsExtraDirs "$DOCKER_CFG" 2>/dev/null; then
  python3 - "$DOCKER_CFG" <<'PY'
import json, sys, os
p = sys.argv[1]
cfg = json.load(open(p)) if os.path.exists(p) else {}
cfg.setdefault("cliPluginsExtraDirs", []).append("/opt/homebrew/lib/docker/cli-plugins")
json.dump(cfg, open(p, "w"), indent=2)
PY
  echo "Registered /opt/homebrew/lib/docker/cli-plugins in $DOCKER_CFG"
fi

# --- 1. Colima (Docker daemon) ---------------------------------------------------
log "Colima"
if colima status >/dev/null 2>&1; then
  echo "already running"
else
  colima start --cpu "$COLIMA_CPU" --memory "$COLIMA_MEM" --disk "$COLIMA_DISK"
fi
docker context use colima >/dev/null

# --- 2. minikube -----------------------------------------------------------------
log "minikube"
if minikube status >/dev/null 2>&1; then
  echo "already running"
else
  minikube start --driver=docker --cpus="$MK_CPUS" --memory="$MK_MEM" --disk-size="$MK_DISK"
fi

# --- 3. Build images into minikube's daemon --------------------------------------
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  log "SKIP_BUILD=1 -- not building images"
else
  log "Building images into minikube's docker daemon (tag: $IMAGE_TAG)"
  eval "$(minikube docker-env)"
  cd "$REPO_ROOT"
  docker build -t "fraud-detection-airflow:$IMAGE_TAG"        -f Dockerfile                .
  docker build -t "fraud-detection-model-training:$IMAGE_TAG" -f Dockerfile.model-training .
  docker build -t "mlflow-server:$IMAGE_TAG"                  -f Dockerfile.mlflow         .
  docker build -t "fraud-detection-streamlit:$IMAGE_TAG"      -f streamlit_app/Dockerfile  streamlit_app
  docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' | grep -E "fraud-detection|mlflow-server"
fi

# --- 4. Helm repo ------------------------------------------------------------------
log "Helm repo"
helm repo add apache-airflow https://airflow.apache.org >/dev/null 2>&1 || true
helm repo update >/dev/null

cat <<EOF

✓ Cluster ready.
  Next:  ./scripts/fetch_data.sh [--smoke|--full]   # load data into the node
         ./scripts/deploy.sh                         # apply manifests + helm install
EOF
