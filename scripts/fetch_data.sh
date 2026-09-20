#!/usr/bin/env bash
#
# Host-side setup: get the IEEE-CIS dataset and load it into the minikube node.
#
# Unlike the scripts/run_*.py pipeline stages (which execute INSIDE pods), this runs
# on your Mac. It exists because the raw CSVs are gitignored and the minikube node
# cannot see host project files.
#
# Prerequisites (real data only):
#   - Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> Create New Token)
#   - You must accept the competition rules at
#     https://www.kaggle.com/c/ieee-fraud-detection/rules  (API downloads 403 otherwise)
#
# Usage:
#   ./scripts/fetch_data.sh              # download + push a 50k-row sample (default)
#   ./scripts/fetch_data.sh --full       # push the complete CSVs instead
#   ./scripts/fetch_data.sh --smoke      # no Kaggle: synthesise a tiny dataset and push it
#   ./scripts/fetch_data.sh --skip-load  # download/generate only, don't touch minikube
#
set -euo pipefail

COMPETITION="ieee-fraud-detection"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/unzipped"
# Must match the hostPath in k8s/data-pvc.yaml, which pods mount at /opt/airflow/data.
NODE_DATA_DIR="/data/unzipped"
# Matches the nrows= used by DataIngestion.load_transaction_data().
SAMPLE_ROWS=50000

MODE="sample"
LOAD=1
for arg in "$@"; do
  case "$arg" in
    --full)      MODE="full" ;;
    --smoke)     MODE="smoke" ;;
    --skip-load) LOAD=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# --- 1. Download (or synthesise) ---------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
  echo "Generating synthetic smoke dataset (no Kaggle needed)..."
  PY="$(command -v python3)"
  # Prefer the project venv if it exists (has pandas); else fall back to uv.
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]] && "$REPO_ROOT/.venv/bin/python" -c 'import pandas' 2>/dev/null; then
    PY="$REPO_ROOT/.venv/bin/python"
  elif command -v uv >/dev/null; then
    PY="uv run --with pandas --with numpy python"
  fi
  $PY "$REPO_ROOT/scripts/make_smoke_data.py" --out "$DATA_DIR"
elif [[ -f "$DATA_DIR/train_transaction.csv" ]]; then
  echo "✓ Raw data already present in $DATA_DIR, skipping download"
else
  command -v kaggle >/dev/null || { echo "kaggle CLI not found (uv tool install kaggle)" >&2; exit 1; }
  [[ -f "$HOME/.kaggle/kaggle.json" ]] || { echo "Missing ~/.kaggle/kaggle.json — see header" >&2; exit 1; }
  chmod 600 "$HOME/.kaggle/kaggle.json"

  mkdir -p "$DATA_DIR"
  echo "Downloading $COMPETITION (~1.2 GB)..."
  kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"
  unzip -o -q "$DATA_DIR/$COMPETITION.zip" -d "$DATA_DIR"
  rm -f "$DATA_DIR/$COMPETITION.zip"
  echo "✓ Downloaded and extracted to $DATA_DIR"
fi

# --- 2. Stage the files to push ----------------------------------------------
# The identity CSVs are read in full by load_identity_data(), so never truncate them.
STAGE="$DATA_DIR"
if [[ "$MODE" == "sample" ]]; then
  STAGE="$(mktemp -d)"
  trap 'rm -rf "$STAGE"' EXIT
  echo "Truncating transaction CSVs to $SAMPLE_ROWS rows (use --full to skip)..."
  for f in train_transaction test_transaction; do
    head -n $((SAMPLE_ROWS + 1)) "$DATA_DIR/$f.csv" > "$STAGE/$f.csv"
  done
  cp "$DATA_DIR/train_identity.csv" "$DATA_DIR/test_identity.csv" "$STAGE/"
fi

[[ "$LOAD" -eq 1 ]] || { echo "✓ Done (--skip-load); files staged in $DATA_DIR"; exit 0; }

# --- 3. Load into the minikube node ------------------------------------------
minikube status >/dev/null 2>&1 || { echo "minikube is not running — start it first (scripts/start_cluster.sh)" >&2; exit 1; }

echo "Loading data into minikube node at $NODE_DATA_DIR ..."
minikube ssh -- "sudo mkdir -p $NODE_DATA_DIR && sudo chmod 777 $NODE_DATA_DIR"

NODE="$(minikube profile 2>/dev/null || echo minikube)"
for f in train_transaction train_identity test_transaction test_identity; do
  src="$STAGE/$f.csv"
  printf '  %-22s %s\n' "$f.csv" "$(du -h "$src" | cut -f1)"
  # With the docker driver the node is a container on the host daemon, and
  # `docker cp` is far faster than `minikube cp` for large files. Fall back to
  # minikube cp for other drivers (vfkit/qemu) or when DOCKER_HOST points at
  # minikube's own daemon (after `eval $(minikube docker-env)`).
  if ! docker cp "$src" "$NODE:$NODE_DATA_DIR/$f.csv" 2>/dev/null; then
    minikube cp "$src" "$NODE_DATA_DIR/$f.csv"
  fi
done

minikube ssh -- "sudo chown -R 50000:50000 /data && sudo chmod -R 777 /data"
echo "✓ Data loaded. Verify with: minikube ssh -- ls -lh $NODE_DATA_DIR"
