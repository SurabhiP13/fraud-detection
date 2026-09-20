"""
Production-ready preprocessing module for fraud detection.
Replicates the exact cleaning and feature engineering pipeline.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


class FraudPreprocessor:
    """
    Handles data cleaning and feature engineering for fraud prediction.
    Loads saved artifacts from training (label encoders, feature stats).
    """
    
    def __init__(self, artifacts_dir: str):
        """
        Args:
            artifacts_dir: Directory holding feature_names.json, feature_stats.json and
                label_encoders.json, as logged by the training pipeline next to the model.
        """
        artifacts = Path(artifacts_dir)

        with open(artifacts / "feature_names.json") as f:
            self.feature_names = json.load(f)

        with open(artifacts / "feature_stats.json") as f:
            self.feature_stats = json.load(f)

        # LabelEncoder.classes_ is sorted, so a class's position is its encoded value.
        with open(artifacts / "label_encoders.json") as f:
            self.label_encoders = {
                col: {cls: i for i, cls in enumerate(classes)}
                for col, classes in json.load(f).items()
            }

    def clean_transaction(self, raw_tx: pd.Series) -> pd.Series:
        """
        Column dropping in data_cleaning.py is data-driven (null / constant columns), and
        its result is already captured by feature_names, so nothing is dropped here.
        """
        return raw_tx.copy()

    def _safe_encode(self, value: str, column: str) -> int:
        """Encode a categorical value with the training-time mapping; unseen values become -1."""
        if column not in self.label_encoders:
            return value

        value = "__NA__" if pd.isna(value) else str(value)
        return self.label_encoders[column].get(value, -1)

    def engineer_features(self, cleaned_tx: pd.Series) -> pd.Series:
        """
        Apply feature engineering (same as feature_engineering.py).
        
        Args:
            cleaned_tx: Cleaned transaction
        
        Returns:
            Transaction with engineered features
        """
        tx = cleaned_tx.copy()
        
        # 1. Email domain splits
        for email_col, prefix in [('P_emaildomain', 'P_emaildomain'), 
                                   ('R_emaildomain', 'R_emaildomain')]:
            if email_col in tx.index and pd.notna(tx[email_col]):
                parts = str(tx[email_col]).split('.')
                for i in range(3):  # max_splits = 3
                    col_name = f"{prefix}_{i+1}"
                    tx[col_name] = parts[i] if i < len(parts) else np.nan
            else:
                # Training fills missing domains with "" before splitting, so the first
                # piece is "" (a real encoder class), not NaN.
                tx[f"{prefix}_1"] = ""
                tx[f"{prefix}_2"] = np.nan
                tx[f"{prefix}_3"] = np.nan
        
        # 2. Aggregation features using SAVED statistics
        # TransactionAmt ratios
        if 'TransactionAmt' in tx.index and 'card1' in tx.index:
            card1_val = str(int(tx['card1'])) if pd.notna(tx['card1']) else 'unknown'
            
            # TransactionAmt_to_mean_card1
            if card1_val in self.feature_stats.get('TransactionAmt_mean_by_card1', {}):
                mean_val = self.feature_stats['TransactionAmt_mean_by_card1'][card1_val]
                tx['TransactionAmt_to_mean_card1'] = tx['TransactionAmt'] / mean_val
            else:
                tx['TransactionAmt_to_mean_card1'] = np.nan
            
            # TransactionAmt_to_std_card1
            if card1_val in self.feature_stats.get('TransactionAmt_std_by_card1', {}):
                std_val = self.feature_stats['TransactionAmt_std_by_card1'][card1_val]
                tx['TransactionAmt_to_std_card1'] = tx['TransactionAmt'] / std_val if std_val > 0 else np.nan
            else:
                tx['TransactionAmt_to_std_card1'] = np.nan
        
        # Similar for card4
        if 'TransactionAmt' in tx.index and 'card4' in tx.index:
            card4_val = str(tx['card4']) if pd.notna(tx['card4']) else 'unknown'
            
            if card4_val in self.feature_stats.get('TransactionAmt_mean_by_card4', {}):
                mean_val = self.feature_stats['TransactionAmt_mean_by_card4'][card4_val]
                tx['TransactionAmt_to_mean_card4'] = tx['TransactionAmt'] / mean_val
            else:
                tx['TransactionAmt_to_mean_card4'] = np.nan
            
            if card4_val in self.feature_stats.get('TransactionAmt_std_by_card4', {}):
                std_val = self.feature_stats['TransactionAmt_std_by_card4'][card4_val]
                tx['TransactionAmt_to_std_card4'] = tx['TransactionAmt'] / std_val if std_val > 0 else np.nan
            else:
                tx['TransactionAmt_to_std_card4'] = np.nan
        
        # id_02 ratios
        if 'id_02' in tx.index and pd.notna(tx['id_02']):
            if 'card1' in tx.index:
                card1_val = str(int(tx['card1'])) if pd.notna(tx['card1']) else 'unknown'
                
                if card1_val in self.feature_stats.get('id_02_mean_by_card1', {}):
                    mean_val = self.feature_stats['id_02_mean_by_card1'][card1_val]
                    tx['id_02_to_mean_card1'] = tx['id_02'] / mean_val
                else:
                    tx['id_02_to_mean_card1'] = np.nan
                
                if card1_val in self.feature_stats.get('id_02_std_by_card1', {}):
                    std_val = self.feature_stats['id_02_std_by_card1'][card1_val]
                    tx['id_02_to_std_card1'] = tx['id_02'] / std_val if std_val > 0 else np.nan
                else:
                    tx['id_02_to_std_card1'] = np.nan
            
            if 'card4' in tx.index:
                card4_val = str(tx['card4']) if pd.notna(tx['card4']) else 'unknown'
                
                if card4_val in self.feature_stats.get('id_02_mean_by_card4', {}):
                    mean_val = self.feature_stats['id_02_mean_by_card4'][card4_val]
                    tx['id_02_to_mean_card4'] = tx['id_02'] / mean_val
                else:
                    tx['id_02_to_mean_card4'] = np.nan
                
                if card4_val in self.feature_stats.get('id_02_std_by_card4', {}):
                    std_val = self.feature_stats['id_02_std_by_card4'][card4_val]
                    tx['id_02_to_std_card4'] = tx['id_02'] / std_val if std_val > 0 else np.nan
                else:
                    tx['id_02_to_std_card4'] = np.nan
        
        # D15 ratios
        if 'D15' in tx.index and pd.notna(tx['D15']):
            for group_col in ['card1', 'card4', 'addr1', 'addr2']:
                if group_col in tx.index:
                    group_val = str(tx[group_col]) if pd.notna(tx[group_col]) else 'unknown'
                    
                    if group_val in self.feature_stats.get(f'D15_mean_by_{group_col}', {}):
                        mean_val = self.feature_stats[f'D15_mean_by_{group_col}'][group_val]
                        tx[f'D15_to_mean_{group_col}'] = tx['D15'] / mean_val
                    else:
                        tx[f'D15_to_mean_{group_col}'] = np.nan
                    
                    if group_val in self.feature_stats.get(f'D15_std_by_{group_col}', {}):
                        std_val = self.feature_stats[f'D15_std_by_{group_col}'][group_val]
                        tx[f'D15_to_std_{group_col}'] = tx['D15'] / std_val if std_val > 0 else np.nan
                    else:
                        tx[f'D15_to_std_{group_col}'] = np.nan
        
        # 3. Apply label encoding to categorical columns
        for col in self.label_encoders.keys():
            if col in tx.index:
                tx[col] = self._safe_encode(tx[col], col)
        
        # 4. Convert to float32 (memory optimization)
        for col in tx.index:
            if col not in ['TransactionID', 'TransactionDT', 'isFraud']:
                try:
                    tx[col] = np.float32(tx[col])
                except (ValueError, TypeError):
                    pass
        
        # 5. Ensure all expected features are present with correct order
        feature_vector = pd.Series(index=self.feature_names, dtype=np.float32)
        for col in self.feature_names:
            if col in tx.index:
                feature_vector[col] = tx[col]
            else:
                feature_vector[col] = np.nan
        
        return feature_vector
    
    def preprocess(self, raw_tx: pd.Series) -> np.ndarray:
        """
        Full preprocessing pipeline: clean → engineer → format.
        
        Args:
            raw_tx: Raw transaction (pandas Series)
        
        Returns:
            Feature vector ready for model prediction (numpy array)
        """
        cleaned = self.clean_transaction(raw_tx)
        engineered = self.engineer_features(cleaned)
        
        # Return as 2D array (sklearn expects (n_samples, n_features))
        return engineered.values.reshape(1, -1)
