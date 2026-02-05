"""
Model training module for fraud detection pipeline.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import json
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import lightgbm as lgb
# import mlflow  # MLflow runs as separate service
# import mlflow.lightgbm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import load_config

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    su

class ModelTrainer:
    """Class for training and evaluating fraud detection models."""
    
    def __init__(self, config: dict):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config['model']
        self.cv_config = config['cross_validation']
        self.data_config = config['data']
        self.models = []
        
        # MLflow setup (optional - will connect to external service if available)
        self.mlflow_available = False
        try:
            import mlflow
            mlflow_uri = os.getenv('MLFLOW_TRACKING_URI') or config.get('mlflow', {}).get('tracking_uri')
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
                self.mlflow_available = True
                logger.info(f"MLflow tracking URI set to: {mlflow_uri}")
        except (ImportError, Exception) as e:
            logger.warning(f"MLflow not available: {e}. Proceeding without experiment tracking.")
        
    def get_fold_strategy(self):
        """
        Get cross-validation fold strategy based on configuration.
        
        Returns:
            Sklearn fold splitter object
        """
        strategy = self.cv_config['strategy']
        n_splits = self.cv_config['n_splits']
        random_state = self.cv_config['random_state']
        shuffle = self.cv_config['shuffle']
        
        if strategy == 'stratified':
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif strategy == 'kfold':
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            raise ValueError(f"Unsupported CV strategy: {strategy}")
    
    def train_lightgbm_model(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame) -> Dict[str, any]:
        """
        Train LightGBM model with cross-validation and log to MLflow.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            
        Returns:
            Dictionary with predictions, scores, and models
        """
        logger.info("Training LightGBM model with cross-validation...")
        
        # Get model parameters
        lgb_params = self.model_config['lightgbm'].copy()
        n_folds = self.cv_config['n_splits']
        
        # Try to connect to MLflow if available
        mlflow_available = False
        mlflow_run = None
        try:
            import mlflow
            import mlflow.lightgbm
            mlflow_uri = os.getenv('MLFLOW_TRACKING_URI') or self.config.get('mlflow', {}).get('tracking_uri')
            logger.info(f"MLflow URI from env: {os.getenv('MLFLOW_TRACKING_URI')}")
            logger.info(f"MLflow URI from config: {self.config.get('mlflow', {}).get('tracking_uri')}")
            logger.info(f"Using MLflow URI: {mlflow_uri}")
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow_run = mlflow.start_run()
                mlflow_available = True
                logger.info(f"✓ Connected to MLflow at {mlflow_uri}")
                
                # Log parameters
                mlflow.log_params({
                    'model_type': 'lightgbm',
                    'n_folds': n_folds,
                    'cv_strategy': self.cv_config['strategy'],
                    **lgb_params
                })
        except (ImportError, Exception) as e:
            logger.error(f"✗ MLflow connection failed: {type(e).__name__}: {e}", exc_info=True)
        
        # Prepare results storage
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
        fold_scores = []
        
        # Get fold strategy
        folds = self.get_fold_strategy()
        
        # Cross-validation loop
        for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
            logger.info(f"Training fold {fold_n + 1}/{n_folds}...")
            
            # Split data
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            
            # Initialize model
            model = lgb.LGBMClassifier(**lgb_params, random_state=self.model_config['random_state'])
            
            # Train model
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
                eval_metric='auc',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.model_config['early_stopping_rounds']),
                    lgb.log_evaluation(period=self.model_config['verbose_eval'])
                ]
            )
            
            # Get predictions
            oof_predictions[valid_idx] = model.predict_proba(X_val)[:, 1]
            test_predictions += model.predict_proba(X_test)[:, 1] / n_folds
            
            # Calculate fold score
            fold_score = roc_auc_score(y_val, oof_predictions[valid_idx])
            fold_scores.append(fold_score)
            logger.info(f"Fold {fold_n + 1} ROC AUC: {fold_score:.6f}")
            
            # Log fold metric to MLflow if available
            if mlflow_available:
                try:
                    mlflow.log_metric(f"fold_{fold_n + 1}_roc_auc", fold_score)
                except Exception as e:
                    logger.warning(f"Failed to log metric to MLflow: {e}")
            
            # Store model
            self.models.append(model)
        
        # Calculate overall score
        overall_score = roc_auc_score(y_train, oof_predictions)
        mean_fold_score = np.mean(fold_scores)
        std_fold_score = np.std(fold_scores)
        
        logger.info(f"Overall ROC AUC: {overall_score:.6f}")
        logger.info(f"Mean Fold ROC AUC: {mean_fold_score:.6f} ± {std_fold_score:.6f}")
        
        # Log overall metrics to MLflow if available
        run_id_mlflow = None
        model_uri = None
        if mlflow_available:
            try:
                mlflow.log_metrics({
                    'overall_roc_auc': overall_score,
                    'mean_fold_roc_auc': mean_fold_score,
                    'std_fold_roc_auc': std_fold_score
                })
                
                # Log the best model (last fold as representative)
                model_info = mlflow.lightgbm.log_model(
                    self.models[-1],
                    "model",
                    registered_model_name="fraud_detection_lgbm"
                )
                
                run_id_mlflow = mlflow.active_run().info.run_id
                model_uri = model_info.model_uri
                
                logger.info(f"Logged model to MLflow run: {run_id_mlflow}")
                logger.info(f"Model registered in MLflow Model Registry: fraud_detection_lgbm")
                
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        return {
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'fold_scores': fold_scores,
            'overall_score': overall_score,
            'models': self.models,
            'mlflow_run_id': run_id_mlflow,
            'model_uri': model_uri
        }
    
    def run(
        self,
        run_id: str,
        X_train_path: Optional[str] = None,
        y_train_path: Optional[str] = None,
        X_test_path: Optional[str] = None,
        test_ids_path: Optional[str] = None,
        upstream_manifest_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Airflow-safe run: reads feature-engineered data, trains model, writes artifacts to per-run dir.
        Returns only paths + run_id + metrics.
        
        Args:
            run_id: Unique identifier for this training run
            X_train_path: Path to training features
            y_train_path: Path to training target
            X_test_path: Path to test features
            test_ids_path: Path to test IDs
            upstream_manifest_path: Path to feature engineering manifest
            
        Returns:
            Dictionary with output paths and metrics
        """
        logger.info("Starting model training... run_id=%s", run_id)
        
        # Resolve inputs from upstream manifest if provided
        if upstream_manifest_path:
            m = json.loads(Path(upstream_manifest_path).read_text(encoding="utf-8"))
            X_train_path = X_train_path or m["outputs"]["X_train_path"]
            y_train_path = y_train_path or m["outputs"]["y_train_path"]
            X_test_path = X_test_path or m["outputs"]["X_test_path"]
            test_ids_path = test_ids_path or m["outputs"]["test_ids_path"]
        
        # Fallback: try to find most recent feature engineering output (for local dev)
        if not X_train_path or not y_train_path or not X_test_path:
            fe_dir = Path(self.data_config["processed_data_dir"]) / self.data_config.get("feature_engineering_subdir", "feature_engineering")
            if fe_dir.exists():
                runs = sorted([d for d in fe_dir.iterdir() if d.is_dir()], reverse=True)
                if runs:
                    latest_run = runs[0]
                    X_train_path = X_train_path or str(latest_run / self.data_config.get("X_train_name", "X_train.csv"))
                    y_train_path = y_train_path or str(latest_run / self.data_config.get("y_train_name", "y_train.csv"))
                    X_test_path = X_test_path or str(latest_run / self.data_config.get("X_test_name", "X_test.csv"))
                    test_ids_path = test_ids_path or str(latest_run / self.data_config.get("test_ids_name", "test_ids.csv"))
                    logger.info("Using most recent feature engineering data from %s", latest_run)
        
        if not X_train_path or not y_train_path or not X_test_path:
            raise ValueError(
                "X_train_path, y_train_path, and X_test_path must be provided (or provide upstream_manifest_path). "
                "In Airflow, pass the manifest_path from feature_engineering as upstream_manifest_path. "
                "For local testing, ensure feature_engineering has been run first."
            )
        
        # Load data
        logger.info("Loading data from feature engineering outputs...")
        logger.info("  X_train: %s", X_train_path)
        logger.info("  y_train: %s", y_train_path)
        logger.info("  X_test: %s", X_test_path)
        
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)['isFraud']
        X_test = pd.read_csv(X_test_path)
        
        if test_ids_path and Path(test_ids_path).exists():
            test_ids = pd.read_csv(test_ids_path)
        else:
            test_ids = pd.DataFrame({'TransactionID': range(len(X_test))})
        
        logger.info("X_train shape: %s", X_train.shape)
        logger.info("y_train shape: %s", y_train.shape)
        logger.info("X_test shape: %s", X_test.shape)
        
        # Set MLflow experiment (if MLflow is available)
        try:
            import mlflow
            experiment_name = self.config.get('mlflow', {}).get('experiment_name', 'fraud_detection')
            mlflow.set_experiment(experiment_name)
        except (ImportError, Exception):
            pass
        
        # Train model (MLflow logging happens inside)
        if self.model_config['type'] == 'lightgbm':
            results = self.train_lightgbm_model(X_train, y_train, X_test)
        else:
            raise ValueError(f"Unsupported model type: {self.model_config['type']}")
        
        # Create per-run output directory
        base_dir = Path(self.data_config["processed_data_dir"])
        step_dir = self.data_config.get("model_training_subdir", "model_training")
        output_dir = base_dir / step_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        oof_pred_path = output_dir / "oof_predictions.csv"
        test_pred_path = output_dir / "test_predictions.csv"
        
        logger.info("Saving predictions to %s", output_dir)
        
        # OOF predictions (for validation)
        pd.DataFrame({
            'oof_prediction': results['oof_predictions']
        }).to_csv(oof_pred_path, index=False)
        
        # Test predictions
        pd.DataFrame({
            'test_prediction': results['test_predictions']
        }).to_csv(test_pred_path, index=False)
        
        # Save fold scores
        scores_path = output_dir / "fold_scores.json"
        scores_data = {
            'fold_scores': [float(s) for s in results['fold_scores']],
            'overall_score': float(results['overall_score']),
            'mean_fold_score': float(np.mean(results['fold_scores'])),
            'std_fold_score': float(np.std(results['fold_scores']))
        }
        scores_path.write_text(json.dumps(scores_data, indent=2), encoding="utf-8")
        
        # Create manifest
        manifest = {
            "step": "model_training",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "X_train_path": str(X_train_path),
                "y_train_path": str(y_train_path),
                "X_test_path": str(X_test_path),
                "test_ids_path": str(test_ids_path) if test_ids_path else None,
                "upstream_manifest_path": str(upstream_manifest_path) if upstream_manifest_path else None,
            },
            "outputs": {
                "oof_predictions_path": str(oof_pred_path),
                "test_predictions_path": str(test_pred_path),
                "fold_scores_path": str(scores_path),
            },
            "model_info": {
                "model_type": self.model_config['type'],
                "mlflow_run_id": results['mlflow_run_id'],
                "mlflow_model_uri": results['model_uri'],
                "registered_model_name": "fraud_detection_lgbm",
            },
            "metrics": scores_data,
            "params": {
                "n_splits": self.cv_config['n_splits'],
                "cv_strategy": self.cv_config['strategy'],
                **self.model_config['lightgbm']
            },
            "checksums": {
                "oof_predictions_sha256": _sha256_file(oof_pred_path),
                "test_predictions_sha256": _sha256_file(test_pred_path),
            },
        }
        
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        
        logger.info("Model training completed! manifest=%s", manifest_path)
        logger.info("Overall ROC AUC: %.6f", results['overall_score'])
        logger.info("MLflow run ID: %s", results['mlflow_run_id'])
        
        return {
            "run_id": run_id,
            "oof_predictions_path": str(oof_pred_path),
            "test_predictions_path": str(test_pred_path),
            "manifest_path": str(manifest_path),
            "overall_roc_auc": float(results['overall_score']),
            "mlflow_run_id": results['mlflow_run_id'],
        }


def run_model_training(
    config_path: str = "config.yaml",
    run_id: Optional[str] = None,
    X_train_path: Optional[str] = None,
    y_train_path: Optional[str] = None,
    X_test_path: Optional[str] = None,
    test_ids_path: Optional[str] = None,
    upstream_manifest_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Airflow-compatible entrypoint.
    
    Args:
        config_path: Path to configuration file
        run_id: Unique identifier for this training run
        X_train_path: Path to training features
        y_train_path: Path to training target
        X_test_path: Path to test features
        test_ids_path: Path to test IDs
        upstream_manifest_path: Path to feature engineering manifest
        **kwargs: Additional keyword arguments (for Airflow context)
        
    Returns:
        Dictionary with output paths and metrics
    """
    config = load_config(config_path)
    
    # Auto-generate run_id for local testing
    if run_id is None:
        cfg_text = Path(config_path).read_text(encoding="utf-8")
        cfg_hash = _sha256_text(cfg_text)[:10]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        run_id = f"{ts}_{cfg_hash}"
        logger.info("Auto-generated run_id for local testing: %s", run_id)
    
    trainer = ModelTrainer(config)
    return trainer.run(
        run_id=run_id,
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        X_test_path=X_test_path,
        test_ids_path=test_ids_path,
        upstream_manifest_path=upstream_manifest_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("\n" + "="*80)
    print("TESTING MODEL TRAINING MODULE")
    print("="*80 + "\n")
    
    try:
        result = run_model_training()
        
        print("\n✓ Model training completed successfully!\n")
        print("Output paths:")
        print(f"  run_id: {result['run_id']}")
        print(f"  manifest_path: {result['manifest_path']}")
        print(f"\nMetrics:")
        print(f"  Overall ROC AUC: {result['overall_roc_auc']:.6f}")
        print(f"  MLflow run ID: {result['mlflow_run_id']}")
        
        # Validate output files exist
        print("\n✓ Validating output files...")
        for key, path in result.items():
            if key.endswith("_path"):
                p = Path(path)
                if p.exists():
                    size_kb = p.stat().st_size / 1024
                    print(f"  ✓ {key}: {size_kb:.2f} KB")
                else:
                    print(f"  ✗ {key}: FILE NOT FOUND!")
        
        # Load and validate manifest
        print("\n✓ Validating manifest...")
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        print(f"  Step: {manifest['step']}")
        print(f"  Model type: {manifest['model_info']['model_type']}")
        print(f"  Registered model: {manifest['model_info']['registered_model_name']}")
        print(f"  Metrics: {manifest['metrics']}")
        
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