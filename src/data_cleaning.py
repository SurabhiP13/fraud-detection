"""
Data cleaning module for fraud detection pipeline.
"""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.utils import clean_inf_nan, load_config

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class DataCleaning:
    """Class for cleaning and preprocessing data."""

    def __init__(self, config: dict):
        self.config = config
        self.cleaning_config = config["data_cleaning"]
        self.data_config = config["data"]

    def identify_columns_to_drop(self, train: pd.DataFrame, test: pd.DataFrame) -> List[str]:
        """
        Identify columns to drop based on null values and single value dominance.

        NOTE: If you want a stricter "no peeking at test" policy, you can compute these
        stats on train only and apply to both.
        """
        logger.info("Identifying columns to drop...")

        null_threshold = float(self.cleaning_config["null_threshold"])
        top_value_threshold = float(self.cleaning_config["top_value_threshold"])

        # Columns with too many null values
        many_null_cols_train = [
            col for col in train.columns
            if train[col].isnull().mean() > null_threshold
        ]
        many_null_cols_test = [
            col for col in test.columns
            if test[col].isnull().mean() > null_threshold
        ]

        # Columns where one value dominates
        # (value_counts is expensive; this is OK for IEEE-CIS sized data, but can be optimized if needed)
        big_top_value_cols_train = [
            col for col in train.columns
            if train[col].value_counts(dropna=False, normalize=True).iloc[0] > top_value_threshold
        ]
        big_top_value_cols_test = [
            col for col in test.columns
            if test[col].value_counts(dropna=False, normalize=True).iloc[0] > top_value_threshold
        ]

        # Columns with only one unique value
        one_value_cols_train = [col for col in train.columns if train[col].nunique(dropna=False) <= 1]
        one_value_cols_test = [col for col in test.columns if test[col].nunique(dropna=False) <= 1]

        cols_to_drop = sorted(set(
            many_null_cols_train + many_null_cols_test +
            big_top_value_cols_train + big_top_value_cols_test +
            one_value_cols_train + one_value_cols_test
        ))

        # Don't drop target column
        if "isFraud" in cols_to_drop:
            cols_to_drop.remove("isFraud")

        logger.info("Columns with >%.0f%% nulls (train): %d", null_threshold * 100, len(many_null_cols_train))
        logger.info("Columns with >%.0f%% nulls (test): %d", null_threshold * 100, len(many_null_cols_test))
        logger.info("Columns with >%.0f%% single value (train): %d", top_value_threshold * 100, len(big_top_value_cols_train))
        logger.info("Columns with >%.0f%% single value (test): %d", top_value_threshold * 100, len(big_top_value_cols_test))
        logger.info("Columns with one unique value (train): %d", len(one_value_cols_train))
        logger.info("Columns with one unique value (test): %d", len(one_value_cols_test))
        logger.info("Total columns to drop: %d", len(cols_to_drop))

        return cols_to_drop

    def fix_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix column naming inconsistencies (e.g., 'id-02' -> 'id_02').

        Safer than checking 'id' in x because it could match unrelated columns like 'TransactionID'.
        """
        def _rename(col: str) -> str:
            if not isinstance(col, str):
                return col
            c = col.strip()
            # only convert 'id-xx' patterns (case-insensitive)
            if c.lower().startswith("id-"):
                return c.replace("-", "_")
            return c

        return df.rename(columns=_rename)

    def clean_data(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        logger.info("Starting data cleaning...")
        logger.info("Initial train shape: %s", train.shape)
        logger.info("Initial test shape: %s", test.shape)

        train = self.fix_column_names(train)
        test = self.fix_column_names(test)

        cols_to_drop = self.identify_columns_to_drop(train, test)

        # Drop only columns that exist (extra safety)
        train_drop = [c for c in cols_to_drop if c in train.columns]
        test_drop = [c for c in cols_to_drop if c in test.columns]

        train = train.drop(train_drop, axis=1)
        test = test.drop(test_drop, axis=1)

        logger.info("After dropping columns - train shape: %s", train.shape)
        logger.info("After dropping columns - test shape: %s", test.shape)

        train = clean_inf_nan(train)
        test = clean_inf_nan(test)

        logger.info("Data cleaning completed!")
        return train, test, cols_to_drop

    def run(
        self,
        run_id: str,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        upstream_manifest_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Execute the full data cleaning pipeline.

        Preferred usage in Airflow:
          - pass run_id + train_path/test_path from ingestion XCom
          OR
          - pass upstream_manifest_path and let cleaning read the input paths from it

        Returns: small dict (Airflow/XCom friendly)
        """
        logger.info("Starting data cleaning pipeline... run_id=%s", run_id)

        # Resolve inputs from upstream manifest if provided
        if upstream_manifest_path:
            upstream_manifest_path = str(upstream_manifest_path)
            m = json.loads(Path(upstream_manifest_path).read_text(encoding="utf-8"))
            train_path = train_path or m["outputs"]["train_path"]
            test_path = test_path or m["outputs"]["test_path"]

        if not train_path or not test_path:
            raise ValueError("train_path and test_path must be provided (or provide upstream_manifest_path).")

        train_path_p = Path(train_path)
        test_path_p = Path(test_path)

        logger.info("Loading data from %s and %s", train_path_p, test_path_p)
        train = pd.read_csv(train_path_p)
        test = pd.read_csv(test_path_p)

        train_clean, test_clean, cols_to_drop = self.clean_data(train, test)

        # Per-run output folder
        base_dir = Path(self.data_config["processed_data_dir"])
        step_dir = self.data_config.get("cleaning_subdir", "cleaning")
        output_dir = base_dir / step_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        train_name = self.data_config.get("train_cleaned_name", "train_cleaned.csv")
        test_name = self.data_config.get("test_cleaned_name", "test_cleaned.csv")

        train_out = output_dir / train_name
        test_out = output_dir / test_name

        logger.info("Saving cleaned data to %s", output_dir)
        train_clean.to_csv(train_out, index=False)
        test_clean.to_csv(test_out, index=False)

        dropped_cols_path = output_dir / "dropped_columns.json"
        dropped_cols_path.write_text(json.dumps(cols_to_drop, indent=2), encoding="utf-8")

        manifest = {
            "step": "data_cleaning",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "params": {
                "null_threshold": float(self.cleaning_config["null_threshold"]),
                "top_value_threshold": float(self.cleaning_config["top_value_threshold"]),
            },
            "inputs": {
                "train_path": str(train_path_p),
                "test_path": str(test_path_p),
                "upstream_manifest_path": str(upstream_manifest_path) if upstream_manifest_path else None,
            },
            "outputs": {
                "train_path": str(train_out),
                "test_path": str(test_out),
                "dropped_columns_path": str(dropped_cols_path),
            },
            "stats": {
                "train_shape_before": [int(train.shape[0]), int(train.shape[1])],
                "test_shape_before": [int(test.shape[0]), int(test.shape[1])],
                "train_shape_after": [int(train_clean.shape[0]), int(train_clean.shape[1])],
                "test_shape_after": [int(test_clean.shape[0]), int(test_clean.shape[1])],
                "n_cols_dropped": int(len(cols_to_drop)),
            },
            "checksums": {
                "train_clean_sha256": _sha256_file(train_out),
                "test_clean_sha256": _sha256_file(test_out),
            },
        }

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        logger.info("Data cleaning completed! manifest=%s", manifest_path)

        return {
            "run_id": run_id,
            "train_path": str(train_out),
            "test_path": str(test_out),
            "dropped_columns_path": str(dropped_cols_path),
            "manifest_path": str(manifest_path),
        }


# ===== External helper to run locally / Airflow callable =====

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_data_cleaning(
    config_path: str = "config.yaml",
    run_id: Optional[str] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    upstream_manifest_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Airflow-compatible entrypoint.
    """
    config = load_config(config_path)
###This is just for running locally; airlow will pass these paths from ingestion###
    if run_id is None and train_path is None and test_path is None:
        cfg_text = Path(config_path).read_text(encoding="utf-8")
        cfg_hash = _sha256_text(cfg_text)[:10]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        run_id = f"{ts}_{cfg_hash}"
        
        train_path = str("C:\\Users\\psura\\Repositories\\fraud-detection-system\\data\\processed\\ingestion\\20260120_0723_4aff7cc44c\\train_merged.csv")
    
        test_path = str("C:\\Users\\psura\\Repositories\\fraud-detection-system\\data\\processed\\ingestion\\20260120_0723_4aff7cc44c\\test_merged.csv")
    
    cleaning = DataCleaning(config)
    return cleaning.run(
        run_id=run_id,
        train_path=train_path,
        test_path=test_path,
        upstream_manifest_path=upstream_manifest_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example local run assuming you already have ingestion output paths:
    # result = run_data_cleaning(run_id="...", train_path="...", test_path="...")
    result = run_data_cleaning()
    print("run_id:", result["run_id"])
    print("train_path:", result["train_path"])
    print("test_path:", result["test_path"])
    print("manifest_path:", result["manifest_path"])
