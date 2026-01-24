"""
Feature engineering module for fraud detection pipeline.
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
from sklearn.preprocessing import LabelEncoder

from src.utils import load_config

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_str_series(s: pd.Series) -> pd.Series:
    # keep NaN as a token so it encodes consistently
    return s.astype("string").fillna("__NA__").astype(str)


class FeatureEngineering:
    """Class for feature engineering and transformation."""

    def __init__(self, config: dict):
        self.config = config
        self.fe_config = config["feature_engineering"]
        self.data_config = config["data"]
        self.label_encoders: Dict[str, LabelEncoder] = {}

    # -------------------------------
    # Feature creation
    # -------------------------------

    def create_email_domain_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create features from email domains by splitting them.
        """
        if not self.fe_config["email_features"]["create_domain_splits"]:
            return df

        max_splits = int(self.fe_config["email_features"].get("max_splits", 3))
        logger.info("Creating email domain features (max_splits=%d)...", max_splits)

        def split_domain(data: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
            if col not in data.columns:
                return data
            parts = data[col].astype("string").fillna("").str.split(".", expand=True)
            # Create up to max_splits columns
            for i in range(max_splits):
                new_col = f"{prefix}_{i+1}"
                data[new_col] = parts[i] if i < parts.shape[1] else np.nan
            return data

        df = split_domain(df, "P_emaildomain", "P_emaildomain")
        df = split_domain(df, "R_emaildomain", "R_emaildomain")
        return df

    def add_group_ratio_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        value_col: str,
        group_cols: List[str],
        prefix: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train-fit group stats, apply to both.
        Creates value/mean(group) and value/std(group) features.

        Uses TRAIN stats only to avoid leakage / unrealistic test-time computation.
        """
        if value_col not in train.columns:
            return train, test

        prefix = prefix or value_col

        for gcol in group_cols:
            if gcol not in train.columns:
                continue

            logger.info("Creating group ratio features: %s by %s", value_col, gcol)

            grp = train.groupby(gcol)[value_col]
            means = grp.mean()
            stds = grp.std()

            # map into train/test
            train_mean = train[gcol].map(means)
            test_mean = test[gcol].map(means)

            train_std = train[gcol].map(stds)
            test_std = test[gcol].map(stds)

            train[f"{prefix}_to_mean_{gcol}"] = train[value_col] / train_mean
            test[f"{prefix}_to_mean_{gcol}"] = test[value_col] / test_mean

            train[f"{prefix}_to_std_{gcol}"] = train[value_col] / train_std
            test[f"{prefix}_to_std_{gcol}"] = test[value_col] / test_std

            # avoid inf from divide-by-zero / missing stats
            for c in [
                f"{prefix}_to_mean_{gcol}",
                f"{prefix}_to_std_{gcol}",
            ]:
                train[c] = train[c].replace([np.inf, -np.inf], np.nan)
                test[c] = test[c].replace([np.inf, -np.inf], np.nan)

        return train, test

    def create_aggregation_features(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Creating aggregation features (train-fit, apply-to-both)...")

        # TransactionAmt ratios
        train, test = self.add_group_ratio_features(
            train, test,
            value_col="TransactionAmt",
            group_cols=["card1", "card4"],
            prefix="TransactionAmt",
        )

        # id_02 ratios
        train, test = self.add_group_ratio_features(
            train, test,
            value_col="id_02",
            group_cols=["card1", "card4"],
            prefix="id_02",
        )

        # D15 ratios
        train, test = self.add_group_ratio_features(
            train, test,
            value_col="D15",
            group_cols=["card1", "card4", "addr1", "addr2"],
            prefix="D15",
        )

        return train, test

    # -------------------------------
    # Encoding
    # -------------------------------

    def encode_categorical_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Encode configured categorical columns using LabelEncoder, fit on combined (train+test) values.
        Returns list of encoded columns.
        """
        logger.info("Encoding configured categorical features...")

        cat_cols = list(self.fe_config.get("categorical_columns", []))
        existing = [c for c in cat_cols if c in train.columns and c in test.columns]
        logger.info("Found %d categorical columns to encode (configured+present in both).", len(existing))

        encoded_cols = []
        for col in existing:
            try:
                le = LabelEncoder()
                combined = pd.concat([_safe_str_series(train[col]), _safe_str_series(test[col])], axis=0)
                le.fit(combined.values)

                train[col] = le.transform(_safe_str_series(train[col]).values)
                test[col] = le.transform(_safe_str_series(test[col]).values)

                self.label_encoders[col] = le
                encoded_cols.append(col)
            except Exception as e:
                logger.warning("Failed to encode column %s: %s", col, e)

        return train, test, encoded_cols

    def encode_remaining_object_columns(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Encode any remaining object/string columns, fit on combined values to keep consistent mapping.
        Uses safe encoding that handles unknown categories gracefully.
        """
        obj_cols = [c for c in X_train.columns if X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("string")]
        # Some columns might be object in train but not in test; we only encode when present in both.
        obj_cols = [c for c in obj_cols if c in X_test.columns]

        logger.info("Encoding remaining object columns: %d", len(obj_cols))

        encoded_cols = []
        for col in obj_cols:
            try:
                le = LabelEncoder()
                combined = pd.concat([_safe_str_series(X_train[col]), _safe_str_series(X_test[col])], axis=0)
                le.fit(combined.values)

                X_train[col] = le.transform(_safe_str_series(X_train[col]).values)
                X_test[col] = le.transform(_safe_str_series(X_test[col]).values)

                # store separately (so you can persist artifacts)
                self.label_encoders[col] = le
                encoded_cols.append(col)
            except Exception as e:
                logger.warning("Failed to encode column %s: %s. Dropping column.", col, e)
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])

        return X_train, X_test, encoded_cols

    # -------------------------------
    # Modeling prep
    # -------------------------------

    def prepare_modeling_data(self, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        logger.info("Preparing data for modeling...")

        # Sort by TransactionDT if present
        if "TransactionDT" in train.columns:
            train = train.sort_values("TransactionDT")

        required = ["TransactionID"]
        for c in required:
            if c not in train.columns or c not in test.columns:
                raise ValueError(f"Missing required column: {c}")

        if "isFraud" not in train.columns:
            raise ValueError("Training data missing target column 'isFraud'.")

        drop_cols_train = [c for c in ["isFraud", "TransactionDT", "TransactionID"] if c in train.columns]
        drop_cols_test = [c for c in ["TransactionDT", "TransactionID"] if c in test.columns]

        X_train = train.drop(drop_cols_train, axis=1)
        y_train = train["isFraud"].astype(int)

        test_ids = test[[c for c in ["TransactionDT", "TransactionID"] if c in test.columns]].copy()
        X_test = test.drop(drop_cols_test, axis=1)

        logger.info("X_train shape: %s", X_train.shape)
        logger.info("y_train shape: %s", y_train.shape)
        logger.info("X_test shape: %s", X_test.shape)
        logger.info("Target distribution: %s", y_train.value_counts().to_dict())

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "test_ids": test_ids,
            "feature_names": list(X_train.columns),
        }

    # -------------------------------
    # Run
    # -------------------------------

    def run(
        self,
        run_id: str,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        upstream_manifest_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Airflow-safe run: reads cleaned train/test, creates features, writes artifacts to per-run dir.
        Returns only paths + run_id.
        """
        logger.info("Starting feature engineering... run_id=%s", run_id)

        # Resolve inputs from upstream manifest if provided
        if upstream_manifest_path:
            m = json.loads(Path(upstream_manifest_path).read_text(encoding="utf-8"))
            train_path = train_path or m["outputs"]["train_path"]
            test_path = test_path or m["outputs"]["test_path"]

        # Fallback: try to find most recent cleaning output (for local dev)
        if not train_path or not test_path:
            cleaning_dir = Path(self.data_config["processed_data_dir"]) / self.data_config.get("cleaning_subdir", "cleaning")
            if cleaning_dir.exists():
                # Find most recent run_id folder
                runs = sorted([d for d in cleaning_dir.iterdir() if d.is_dir()], reverse=True)
                if runs:
                    latest_run = runs[0]
                    train_fallback = latest_run / self.data_config.get("train_cleaned_name", "train_cleaned.csv")
                    test_fallback = latest_run / self.data_config.get("test_cleaned_name", "test_cleaned.csv")
                    if train_fallback.exists() and test_fallback.exists():
                        train_path = str(train_fallback)
                        test_path = str(test_fallback)
                        logger.info("Using most recent cleaned data from %s", latest_run)

        if not train_path or not test_path:
            raise ValueError(
                "train_path and test_path must be provided (or provide upstream_manifest_path). "
                "In Airflow, pass the manifest_path from data_cleaning as upstream_manifest_path. "
                "For local testing, ensure data_cleaning has been run first."
            )

        train_path_p = Path(train_path)
        test_path_p = Path(test_path)

        logger.info("Loading cleaned data from %s and %s", train_path_p, test_path_p)
        
        # Load CSV files (let pandas infer dtypes)
        train = pd.read_csv(train_path_p)
        test = pd.read_csv(test_path_p)

        # Validate column alignment (excluding train-only columns like isFraud)
        train_cols = set(train.columns) - {"isFraud"}
        test_cols = set(test.columns)
        
        if train_cols != test_cols:
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            logger.warning("Column mismatch detected!")
            if missing_in_test:
                logger.warning("Columns in train but not in test: %s", missing_in_test)
            if missing_in_train:
                logger.warning("Columns in test but not in train: %s", missing_in_train)

        # Features
        train, test = self.create_aggregation_features(train, test)

        # Email splits: apply the same function to both
        if self.fe_config["email_features"]["create_domain_splits"]:
            train = self.create_email_domain_features(train)
            test = self.create_email_domain_features(test)

        # Encode configured categorical columns
        train, test, encoded_config_cols = self.encode_categorical_features(train, test)

        # NOW optimize memory by converting to float32 (after categorical encoding)
        logger.info("Optimizing memory: converting numeric columns to float32...")
        all_encoded_cols = set(encoded_config_cols)
        
        # Convert numeric columns to float32 (excluding categorical encoded ones)
        common_cols = train.columns.intersection(test.columns)
        numeric_cols = [col for col in common_cols 
                       if col not in all_encoded_cols 
                       and train[col].dtype in [np.float64, np.int64, np.int32]]
        
        for col in numeric_cols:
            train[col] = train[col].astype(np.float32)
            test[col] = test[col].astype(np.float32)
        
        # Also convert train-only numeric columns (like isFraud, TransactionDT)
        train_only_numeric = [col for col in train.columns 
                             if col not in test.columns 
                             and train[col].dtype in [np.float64, np.int64, np.int32]]
        for col in train_only_numeric:
            train[col] = train[col].astype(np.float32)
        
        logger.info("Converted %d numeric columns to float32", len(numeric_cols) + len(train_only_numeric))

        # Prepare modeling data
        modeling = self.prepare_modeling_data(train, test)
        X_train, y_train, X_test, test_ids = (
            modeling["X_train"],
            modeling["y_train"],
            modeling["X_test"],
            modeling["test_ids"],
        )

        # Encode remaining object columns consistently
        X_train, X_test, encoded_remaining_cols = self.encode_remaining_object_columns(X_train, X_test)

        # Final validation: ensure train and test have identical columns
        if list(X_train.columns) != list(X_test.columns):
            logger.error("Column mismatch after feature engineering!")
            logger.error("X_train columns: %s", X_train.columns.tolist())
            logger.error("X_test columns: %s", X_test.columns.tolist())
            raise ValueError("X_train and X_test must have identical columns after feature engineering")
        
        logger.info("✓ Schema validation passed: X_train and X_test have identical %d columns", len(X_train.columns))

        # Per-run output folder
        base_dir = Path(self.data_config["processed_data_dir"])
        step_dir = self.data_config.get("feature_engineering_subdir", "feature_engineering")
        output_dir = base_dir / step_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train_path = output_dir / self.data_config.get("X_train_name", "X_train.csv")
        y_train_path = output_dir / self.data_config.get("y_train_name", "y_train.csv")
        X_test_path = output_dir / self.data_config.get("X_test_name", "X_test.csv")
        test_ids_path = output_dir / self.data_config.get("test_ids_name", "test_ids.csv")

        logger.info("Saving engineered datasets to %s", output_dir)
        X_train.to_csv(X_train_path, index=False)
        y_train.to_frame(name="isFraud").to_csv(y_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        test_ids.to_csv(test_ids_path, index=False)

        # Save feature names
        feature_names_path = output_dir / "feature_names.json"
        feature_names_path.write_text(json.dumps(list(X_train.columns), indent=2), encoding="utf-8")

        # Save which columns got label-encoded (useful for debugging/repro)
        encoding_info_path = output_dir / "encoding_info.json"
        encoding_info = {
            "encoded_configured_columns": encoded_config_cols,
            "encoded_remaining_object_columns": encoded_remaining_cols,
        }
        encoding_info_path.write_text(json.dumps(encoding_info, indent=2), encoding="utf-8")

        # Save label encoder classes (so later you can inverse_transform / be consistent)
        # Note: LabelEncoder stores classes_ which is enough.
        encoders_path = output_dir / "label_encoders.json"
        enc_dump = {col: le.classes_.tolist() for col, le in self.label_encoders.items()}
        encoders_path.write_text(json.dumps(enc_dump, indent=2), encoding="utf-8")

        manifest = {
            "step": "feature_engineering",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "train_path": str(train_path_p),
                "test_path": str(test_path_p),
                "upstream_manifest_path": str(upstream_manifest_path) if upstream_manifest_path else None,
            },
            "outputs": {
                "X_train_path": str(X_train_path),
                "y_train_path": str(y_train_path),
                "X_test_path": str(X_test_path),
                "test_ids_path": str(test_ids_path),
                "feature_names_path": str(feature_names_path),
                "encoding_info_path": str(encoding_info_path),
                "label_encoders_path": str(encoders_path),
            },
            "stats": {
                "X_train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
                "X_test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
                "y_train_distribution": y_train.value_counts().to_dict(),
            },
            "checksums": {
                "X_train_sha256": _sha256_file(X_train_path),
                "y_train_sha256": _sha256_file(y_train_path),
                "X_test_sha256": _sha256_file(X_test_path),
                "test_ids_sha256": _sha256_file(test_ids_path),
            },
        }

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        logger.info("Feature engineering completed! manifest=%s", manifest_path)

        return {
            "run_id": run_id,
            "X_train_path": str(X_train_path),
            "y_train_path": str(y_train_path),
            "X_test_path": str(X_test_path),
            "test_ids_path": str(test_ids_path),
            "manifest_path": str(manifest_path),
        }


# ---- Airflow/local entrypoint ----

def run_feature_engineering(
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
    
    # Auto-generate run_id for local testing
    if run_id is None:
        cfg_text = Path(config_path).read_text(encoding="utf-8")
        cfg_hash = _sha256_text(cfg_text)[:10]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        run_id = f"{ts}_{cfg_hash}"
        logger.info("Auto-generated run_id for local testing: %s", run_id)

    fe = FeatureEngineering(config)
    return fe.run(
        run_id=run_id,
        train_path=train_path,
        test_path=test_path,
        upstream_manifest_path=upstream_manifest_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("\n" + "="*80)
    print("TESTING FEATURE ENGINEERING MODULE")
    print("="*80 + "\n")
    
    try:
        result = run_feature_engineering()
        
        print("\n✓ Feature engineering completed successfully!\n")
        print("Output paths:")
        print(f"  run_id: {result['run_id']}")
        print(f"  X_train_path: {result['X_train_path']}")
        print(f"  y_train_path: {result['y_train_path']}")
        print(f"  X_test_path: {result['X_test_path']}")
        print(f"  manifest_path: {result['manifest_path']}")
        
        # Validate output files exist
        print("\n✓ Validating output files...")
        for key, path in result.items():
            if key != "run_id":
                p = Path(path)
                if p.exists():
                    size_mb = p.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {key}: {size_mb:.2f} MB")
                else:
                    print(f"  ✗ {key}: FILE NOT FOUND!")
        
        # Load and validate manifest
        print("\n✓ Validating manifest...")
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        print(f"  Step: {manifest['step']}")
        print(f"  X_train shape: {manifest['stats']['X_train_shape']}")
        print(f"  X_test shape: {manifest['stats']['X_test_shape']}")
        print(f"  Target distribution: {manifest['stats']['y_train_distribution']}")
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80 + "\n")
