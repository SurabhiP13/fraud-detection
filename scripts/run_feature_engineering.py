#!/usr/bin/env python3
"""
Containerized entry point for feature engineering task.
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, '/opt/airflow/src')

from feature_engineering import run_feature_engineering


def main():
    """Run feature engineering task."""
    # Read run_id from shared volume
    state_dir = Path('/opt/airflow/data/pipeline_state')
    run_id_file = state_dir / 'run_id.txt'
    
    if not run_id_file.exists():
        print("ERROR: run_id.txt not found. Setup task must run first.")
        return 1
    
    run_id = run_id_file.read_text().strip()
    
    # Read upstream manifest from data cleaning
    cleaning_result_file = state_dir / 'data_cleaning_result.json'
    if not cleaning_result_file.exists():
        print("ERROR: data_cleaning_result.json not found. Data cleaning must run first.")
        return 1
    
    with open(cleaning_result_file, 'r') as f:
        cleaning_result = json.load(f)
    
    upstream_manifest_path = cleaning_result['manifest_path']
    print(f"Running feature engineering with run_id: {run_id}")
    print(f"Using upstream manifest: {upstream_manifest_path}")
    
    # Run feature engineering
    result = run_feature_engineering(
        config_path='/opt/airflow/config.yaml',
        run_id=run_id,
        upstream_manifest_path=upstream_manifest_path
    )
    
    # Save result for downstream tasks
    with open(state_dir / 'feature_engineering_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Feature engineering completed")
    print(f"✓ Manifest saved to: {result.get('manifest_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
