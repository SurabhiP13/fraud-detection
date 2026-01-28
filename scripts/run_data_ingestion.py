#!/usr/bin/env python3
"""
Containerized entry point for data ingestion task.
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, '/opt/airflow/src')

from data_ingestion import run_data_ingestion


def main():
    """Run data ingestion task."""
    # Read run_id from shared volume
    run_id_file = Path('/opt/airflow/data/pipeline_state/run_id.txt')
    if not run_id_file.exists():
        print("ERROR: run_id.txt not found. Setup task must run first.")
        return 1
    
    run_id = run_id_file.read_text().strip()
    print(f"Running data ingestion with run_id: {run_id}")
    
    # Run data ingestion
    result = run_data_ingestion(
        config_path='/opt/airflow/config.yaml',
        run_id=run_id
    )
    
    # Save result for downstream tasks
    output_dir = Path('/opt/airflow/data/pipeline_state')
    with open(output_dir / 'data_ingestion_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Data ingestion completed")
    print(f"✓ Manifest saved to: {result.get('manifest_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
