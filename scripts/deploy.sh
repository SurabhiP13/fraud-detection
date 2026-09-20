#!/usr/bin/env bash
#
# Deploy (or upgrade) the whole stack on the running minikube cluster:
# namespace, PVCs, MLflow, Streamlit, and Airflow via the official Helm chart.
#
# Usage:
#   ./scripts/deploy.sh            # install / upgrade everything
#   ./scripts/deploy.sh --wait     # also block until every pod is Ready
#
# Chart version is pinned: 1.15.0 is the last release shipping Airflow 2.9.3,
# which is what requirements-airflow.txt, the images and both DAGs target.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NAMESPACE:-airflow}"
CHART_VERSION="${CHART_VERSION:-1.15.0}"
WAIT=0
[[ "${1:-}" == "--wait" ]] && WAIT=1

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

minikube status >/dev/null 2>&1 || { echo "minikube is not running — run scripts/start_cluster.sh first" >&2; exit 1; }
cd "$REPO_ROOT"

log "Namespace $NS"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

log "Storage + services"
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml

log "Airflow (chart $CHART_VERSION)"
helm repo add apache-airflow https://airflow.apache.org >/dev/null 2>&1 || true
helm upgrade --install airflow apache-airflow/airflow \
  --version "$CHART_VERSION" \
  --namespace "$NS" \
  -f k8s/airflow-values.yaml \
  --timeout 15m

if [[ "$WAIT" -eq 1 ]]; then
  log "Waiting for pods"
  kubectl rollout status deployment/mlflow      -n "$NS" --timeout=5m
  kubectl rollout status deployment/streamlit   -n "$NS" --timeout=10m
  kubectl rollout status deployment/airflow-scheduler -n "$NS" --timeout=10m
  kubectl rollout status deployment/airflow-webserver -n "$NS" --timeout=10m
fi

log "Status"
kubectl get pods -n "$NS"

cat <<EOF

✓ Deployed. Open the UIs with port-forwards (each in its own terminal):
    kubectl port-forward -n $NS svc/airflow-webserver 8080:8080   # http://localhost:8080  (admin / admin)
    kubectl port-forward -n $NS svc/mlflow-service    5000:5000   # http://localhost:5000
    kubectl port-forward -n $NS svc/streamlit-service 8501:8501   # http://localhost:8501

  Trigger the pipeline from the CLI:
    kubectl exec -n $NS deploy/airflow-scheduler -- airflow dags trigger fraud_detection_pipeline_k8s
EOF
