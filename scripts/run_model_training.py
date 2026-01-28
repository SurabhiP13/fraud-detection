#!/usr/bin/env python3
"""
Containerized entry point for model training task.
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, '/opt/airflow/src')

from model_training import run_model_training


def main():
    """Run model training task."""
    # Read run_id from shared volume
    state_dir = Path('/opt/airflow/data/pipeline_state')
    run_id_file = state_dir / 'run_id.txt'
    
    if not run_id_file.exists():
        print("ERROR: run_id.txt not found. Setup task must run first.")
        return 1
    
    run_id = run_id_file.read_text().strip()
    
    # Read upstream manifest from feature engineering
    fe_result_file = state_dir / 'feature_engineering_result.json'
    if not fe_result_file.exists():
        print("ERROR: feature_engineering_result.json not found. Feature engineering must run first.")
        return 1
    
    with open(fe_result_file, 'r') as f:
        fe_result = json.load(f)
    
    upstream_manifest_path = fe_result['manifest_path']
    print(f"Running model training with run_id: {run_id}")
    print(f"Using upstream manifest: {upstream_manifest_path}")
    
    # Run model training
    result = run_model_training(
        config_path='/opt/airflow/config.yaml',
        run_id=run_id,
        upstream_manifest_path=upstream_manifest_path
    )
    
    # Save result for downstream tasks
    with open(state_dir / 'model_training_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Model training completed")
    print(f"✓ Overall ROC AUC: {result.get('overall_roc_auc', 'N/A')}")
    print(f"✓ MLflow Run ID: {result.get('mlflow_run_id', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
