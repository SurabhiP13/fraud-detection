"""
Production-ready preprocessing module for fraud detection.
Replicates the exact cleaning and feature engineering pipeline.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.preprocessing import LabelEncoder


class FraudPreprocessor:
    """
    Handles data cleaning and feature engineering for fraud prediction.
    Loads saved artifacts from training (label encoders, feature stats).
    """
    
    def __init__(
        self,
        label_encoders_path: str = "preprocess_artifacts/label_encoders.pkl",
        feature_stats_path: str = "preprocess_artifacts/feature_stats.json",
        feature_names_path: str = "preprocess_artifacts/feature_names.json",
    ):
        """
        Initialize preprocessor with saved training artifacts.
        
        Args:
            label_encoders_path: Path to saved label encoders (pickle)
            feature_stats_path: Path to feature statistics (JSON)
            feature_names_path: Path to feature names list (JSON)
        """
        self.label_encoders = joblib.load(label_encoders_path)
        
        with open(feature_stats_path, 'r') as f:
            self.feature_stats = json.load(f)
        
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
    
    def clean_transaction(self, raw_tx: pd.Series) -> pd.Series:
        """
        Apply data cleaning (same as data_cleaning.py).
        
        Args:
            raw_tx: Single transaction as pandas Series
        
        Returns:
            Cleaned transaction
        """
        tx = raw_tx.copy()
        
        # Drop identity columns (same as config.yaml drop_columns)
        identity_cols = [
            'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 
            'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 
            'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 
            'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
        ]
        
        for col in identity_cols:
            if col in tx.index:
                tx = tx.drop(col)
        
        return tx
    
    def _safe_encode(self, value: str, column: str) -> int:
        """
        Safely encode a categorical value using saved LabelEncoder.
        Handles unknown categories by encoding as -1.
        """
        if column not in self.label_encoders:
            return value
        
        encoder = self.label_encoders[column]
        
        # Handle NaN
        if pd.isna(value):
            value = "__NA__"
        else:
            value = str(value)
        
        # Try to transform, use -1 for unknown categories
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # Unknown category not seen in training
            return -1
    
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
                for i in range(3):
                    tx[f"{prefix}_{i+1}"] = np.nan
        
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


# Helper function to save artifacts during training
def save_preprocessing_artifacts(
    label_encoders: Dict[str, LabelEncoder],
    feature_stats: Dict[str, Dict],
    feature_names: List[str],
    output_dir: str = "../models"
):
    """
    Save preprocessing artifacts for production use.
    Call this at the end of feature_engineering.py
    
    Args:
        label_encoders: Dict of column -> LabelEncoder
        feature_stats: Dict of aggregation statistics
        feature_names: List of feature names in correct order
        output_dir: Where to save artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save label encoders as pickle
    encoders_path = output_path / "label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)
    print(f"✓ Saved label encoders to {encoders_path}")
    
    # Save feature stats as JSON
    stats_path = output_path / "feature_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print(f"✓ Saved feature stats to {stats_path}")
    
    # Save feature names as JSON
    names_path = output_path / "feature_names.json"
    with open(names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Saved feature names to {names_path}")


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing FraudPreprocessor...")
    
    # Load a sample transaction
    samples_path = Path("../data/samples/raw_transactions.csv")
    if samples_path.exists():
        df = pd.read_csv(samples_path)
        sample_tx = df.iloc[0]
        
        print(f"\nLoaded sample transaction (TransactionID: {sample_tx['TransactionID']})")
        
        # Initialize preprocessor (using default paths inside streamlit_app/)
        preprocessor = FraudPreprocessor()
        
        # Preprocess
        feature_vector = preprocessor.preprocess(sample_tx)
        
        print(f"\n✓ Preprocessing successful!")
        print(f"  Feature vector shape: {feature_vector.shape}")
        print(f"  Expected features: {len(preprocessor.feature_names)}")
        print(f"  Non-null features: {np.sum(~np.isnan(feature_vector))}")
    else:
        print(f"Sample data not found at {samples_path}")
