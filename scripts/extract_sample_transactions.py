"""
Extract random sample transactions for demo/prediction UI.
"""

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_samples(
    input_path: str = "data/ingestion/train_merged.csv",
    output_path: str = "data/samples/raw_transactions.csv",
    n_samples: int = 100,
    random_state: int = 42,
):
    """
    Extract random sample transactions from train_merged.csv.
    
    Args:
        input_path: Path to train_merged.csv
        output_path: Where to save the samples
        n_samples: Number of random transactions to extract
        random_state: Random seed for reproducibility
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Validate input exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    logger.info(f"Total transactions available: {len(df):,}")
    
    # Sample random transactions
    if len(df) < n_samples:
        logger.warning(f"Dataset has only {len(df)} rows, sampling all of them")
        samples = df.copy()
    else:
        samples = df.sample(n=n_samples, random_state=random_state)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save samples
    logger.info(f"Saving {len(samples)} samples to {output_file}...")
    samples.to_csv(output_file, index=False)
    
    # Print statistics
    logger.info(f"✓ Sample extraction complete!")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Columns: {len(samples.columns)}")
    logger.info(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    # Check fraud distribution
    if 'isFraud' in samples.columns:
        fraud_dist = samples['isFraud'].value_counts()
        logger.info(f"  Fraud distribution: {fraud_dist.to_dict()}")
        fraud_pct = (fraud_dist.get(1, 0) / len(samples)) * 100
        logger.info(f"  Fraud percentage: {fraud_pct:.2f}%")
    
    return output_file


if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXTRACTING SAMPLE TRANSACTIONS FOR DEMO")
    print("="*80 + "\n")
    
    try:
        output = extract_samples()
        print(f"\n✓ SUCCESS! Samples saved to: {output}\n")
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
