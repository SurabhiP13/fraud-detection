#!/bin/bash
# Quick setup script for local development

echo "🚀 Setting up Fraud Detection System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please review and update .env file with your configuration"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ml_model
mkdir -p airflow/logs

# Start PostgreSQL in Docker
echo "🐘 Starting PostgreSQL..."
docker run -d \
  --name fraud-detection-postgres \
  -e POSTGRES_USER=${POSTGRES_USER:-fraud_user} \
  -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-fraud_pass} \
  -e POSTGRES_DB=${POSTGRES_DB:-fraud_detection} \
  -p 5432:5432 \
  postgres:15

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "✅ Setup complete!"
echo ""
echo "Environment variables loaded from .env file"
echo ""
echo "To start the application:"
echo "  1. Backend: uvicorn backend.main:app --reload --port 8000"
echo "  2. Frontend: cd frontend && python -m http.server 3000"
echo "  3. Airflow: airflow standalone"
echo ""
echo "Or use Docker Compose: docker compose up"
