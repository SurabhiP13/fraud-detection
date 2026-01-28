#!/usr/bin/env python3
"""
Containerized entry point for data cleaning task.
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, '/opt/airflow/src')

from data_cleaning import run_data_cleaning


def main():
    """Run data cleaning task."""
    # Read run_id from shared volume
    state_dir = Path('/opt/airflow/data/pipeline_state')
    run_id_file = state_dir / 'run_id.txt'
    
    if not run_id_file.exists():
        print("ERROR: run_id.txt not found. Setup task must run first.")
        return 1
    
    run_id = run_id_file.read_text().strip()
    
    # Read upstream manifest from data ingestion
    ingestion_result_file = state_dir / 'data_ingestion_result.json'
    if not ingestion_result_file.exists():
        print("ERROR: data_ingestion_result.json not found. Data ingestion must run first.")
        return 1
    
    with open(ingestion_result_file, 'r') as f:
        ingestion_result = json.load(f)
    
    upstream_manifest_path = ingestion_result['manifest_path']
    print(f"Running data cleaning with run_id: {run_id}")
    print(f"Using upstream manifest: {upstream_manifest_path}")
    
    # Run data cleaning
    result = run_data_cleaning(
        config_path='/opt/airflow/config.yaml',
        run_id=run_id,
        upstream_manifest_path=upstream_manifest_path
    )
    
    # Save result for downstream tasks
    with open(state_dir / 'data_cleaning_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Data cleaning completed")
    print(f"✓ Manifest saved to: {result.get('manifest_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
