#!/bin/bash
# Deployment script for Kubernetes/Minikube

echo "🚀 Deploying Fraud Detection System to Kubernetes..."

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "❌ Minikube is not installed. Please install Minikube first."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Start Minikube if not running
if ! minikube status &> /dev/null; then
    echo "🔧 Starting Minikube..."
    minikube start --memory=4096 --cpus=2
fi

# Set Docker environment to use Minikube's Docker daemon
echo "🐳 Configuring Docker to use Minikube..."
eval $(minikube docker-env)

# Build Docker images
echo "🏗️ Building Docker images..."
docker build -t fraud-detection-backend:latest -f Dockerfile .
docker build -t fraud-detection-airflow:latest -f Dockerfile.airflow .

# Apply Kubernetes manifests
echo "☸️ Deploying to Kubernetes..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/airflow.yaml

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/postgres -n fraud-detection
kubectl wait --for=condition=available --timeout=300s \
  deployment/backend -n fraud-detection

# Get service URLs
echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Access your services:"
echo "  Backend API: $(minikube service backend -n fraud-detection --url)"
echo "  Frontend UI: $(minikube service frontend -n fraud-detection --url)"
echo "  Airflow UI: $(minikube service airflow-webserver -n fraud-detection --url)"
echo ""
echo "To check pod status:"
echo "  kubectl get pods -n fraud-detection"
