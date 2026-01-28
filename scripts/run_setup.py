#!/usr/bin/env python3
"""
Containerized entry point for setup task.
Creates necessary directories, copies raw data to volume, and generates run_id.
"""
import sys
import os
import hashlib
from datetime import datetime
from pathlib import Path
import json
import yaml
import pandas as pd

# Add src to path
sys.path.insert(0, '/opt/airflow/src')


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_output_dirs(config: dict) -> None:
    """
    Create output directories if they don't exist.
    
    Args:
        config: Configuration dictionary
    """
    dirs_to_create = [
        config['data']['processed_data_dir'],
        config['data']['output_dir'],
        config['data']['models_dir']
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def copy_raw_data(raw_data_dir: str = '/opt/airflow/raw_data', 
                  output_dir: str = '/opt/airflow/data/unzipped',
                  nrows: int = 50000) -> None:
    """
    Copy raw data from image to volume, keeping only first nrows.
    
    Args:
        raw_data_dir: Path to raw data in image
        output_dir: Output directory in volume
        nrows: Number of rows to keep from each CSV
    """
    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        print(f"⚠ Raw data directory not found: {raw_path}")
        return
    
    csv_files = list(raw_path.glob('*.csv'))
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        # Read first nrows
        df = pd.read_csv(csv_file, nrows=nrows)
        
        # Save to volume
        output_file = output_path / csv_file.name
        df.to_csv(output_file, index=False)
        print(f"✓ Saved {len(df)} rows to {output_file}")


def main():
    """Setup directories, copy data, and generate run_id."""
    config = load_config('/opt/airflow/config.yaml')
    create_output_dirs(config)
    
    # Copy raw data from image to volume
    copy_raw_data(nrows=50000)
    
    # Generate run_id based on config hash
    config_path = Path('/opt/airflow/config.yaml')
    cfg_text = config_path.read_text(encoding="utf-8")
    cfg_hash = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()[:10]
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"{ts}_{cfg_hash}"
    
    print(f"✓ Created output directories")
    print(f"✓ Generated run_id: {run_id}")
    
    # Write run_id to shared volume for downstream tasks
    output_dir = Path('/opt/airflow/data/pipeline_state')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'run_id.txt', 'w') as f:
        f.write(run_id)
    
    # Also save as JSON for easy parsing
    with open(output_dir / 'run_metadata.json', 'w') as f:
        json.dump({
            'run_id': run_id,
            'timestamp': ts,
            'config_hash': cfg_hash
        }, f, indent=2)
    
    print(f"✓ Saved run_id to {output_dir / 'run_id.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
