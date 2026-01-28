"""
Utility functions for fraud detection pipeline.
"""
import numpy as np
import pandas as pd
# from numba import jit


def clean_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite values with NaN.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with infinities replaced by NaN
    """
    return df.replace([np.inf, -np.inf], np.nan)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



