"""
Data ingestion module for fraud detection pipeline.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
from utils import load_config
import json
import hashlib
from datetime import datetime, timezone
logger = logging.getLogger(__name__)
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class DataIngestion:
    """Class for loading raw data from CSV files."""
    
    def __init__(self, config: dict):
        """
        Initialize DataIngestion with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        
    def load_transaction_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load first 50k rows of transaction data for efficient processing.
        
        Returns:
            Tuple of (train_transaction, test_transaction) DataFrames
        """
        logger.info("Loading transaction data (first 50k rows)...")
        
        # Load only first 50k rows to avoid memory issues
        train_trans = pd.read_csv(self.data_config['train_transaction'], nrows=50000, low_memory=False)
        test_trans = pd.read_csv(self.data_config['test_transaction'], nrows=50000, low_memory=False)
        
        logger.info(f"Train transaction shape: {train_trans.shape}")
        logger.info(f"Test transaction shape: {test_trans.shape}")
        
        return train_trans, test_trans
    
    def load_identity_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test identity data.
        
        Returns:
            Tuple of (train_identity, test_identity) DataFrames
        """
        logger.info("Loading identity data...")
        
        train_ident = pd.read_csv(self.data_config['train_identity'])
        test_ident = pd.read_csv(self.data_config['test_identity'])
        
        logger.info(f"Train identity shape: {train_ident.shape}")
        logger.info(f"Test identity shape: {test_ident.shape}")
        
        return train_ident, test_ident
    
    def merge_data(self, 
                   train_trans: pd.DataFrame, 
                   train_ident: pd.DataFrame,
                   test_trans: pd.DataFrame,
                   test_ident: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge transaction and identity data.
        
        Args:
            train_trans: Training transaction data
            train_ident: Training identity data
            test_trans: Test transaction data
            test_ident: Test identity data
            
        Returns:
            Tuple of (train, test) merged DataFrames
        """
        logger.info("Merging transaction and identity data...")
        
        train = train_trans.merge(train_ident, on='TransactionID', how='left')
        test = test_trans.merge(test_ident, on='TransactionID', how='left')
        
        logger.info(f"Merged train shape: {train.shape}")
        logger.info(f"Merged test shape: {test.shape}")
        
        # Reduce memory usage
        # if self.config['data_cleaning']['reduce_memory']:
        #     logger.info("Reducing memory usage...")
        #     train = reduce_mem_usage(train)
        #     test = reduce_mem_usage(test)
        
        return train, test
    
    def run(self, run_id: str) -> Dict[str, str]:
        logger.info("Starting data ingestion pipeline... run_id=%s", run_id)

        train_trans, test_trans = self.load_transaction_data()
        train_ident, test_ident = self.load_identity_data()

        train, test = self.merge_data(train_trans, train_ident, test_trans, test_ident)

        # output dir becomes per-run (and per-step)
        base_dir = Path(self.data_config["processed_data_dir"])
        step_dir = self.data_config.get("ingested_subdir", "ingestion")
        output_dir = base_dir / step_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        train_name = self.data_config.get("train_merged_name", "train_merged.csv")
        test_name  = self.data_config.get("test_merged_name", "test_merged.csv")

        train_path = output_dir / train_name
        test_path  = output_dir / test_name

        logger.info("Saving merged data to %s", output_dir)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        manifest = {
            "step": "data_ingestion",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "train_transaction": self.data_config["train_transaction"],
                "train_identity": self.data_config["train_identity"],
                "test_transaction": self.data_config["test_transaction"],
                "test_identity": self.data_config["test_identity"],
            },
            "outputs": {
                "train_path": str(train_path),
                "test_path": str(test_path),
            },
            "stats": {
                "train_shape": [int(train.shape[0]), int(train.shape[1])],
                "test_shape": [int(test.shape[0]), int(test.shape[1])],
            },
            "checksums": {
                "train_sha256": _sha256_file(train_path),
                "test_sha256": _sha256_file(test_path),
            },
        }

        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Data ingestion completed! manifest=%s", output_dir / "manifest.json")

        # Return only small stuff (Airflow friendly)
        return {
            "run_id": run_id,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "manifest_path": str(output_dir / "manifest.json"),
        }

#EXTERNAL TO RUN IT LOCALLY============================================================
#======================================================================================
def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def run_data_ingestion(config_path: str = "config.yaml", run_id: Optional[str] = None, **kwargs):
    config = load_config(config_path)

    if run_id is None:
        cfg_text = Path(config_path).read_text()
        cfg_hash = _sha256_text(cfg_text)[:10]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        run_id = f"{ts}_{cfg_hash}"

    ingestion = DataIngestion(config)
    return ingestion.run(run_id=run_id)


if __name__ == "__main__":
    # For standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    

    result = run_data_ingestion()
    print("train_path:", result["train_path"])
    print("test_path:", result["test_path"])
    print("manifest_path:", result["manifest_path"])