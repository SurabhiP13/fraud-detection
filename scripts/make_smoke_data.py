#!/usr/bin/env python3
"""
Generate a small synthetic IEEE-CIS-shaped dataset for smoke-testing the pipeline.

The Streamlit sample file (streamlit_app/sample_data/raw_transactions.csv) is a
100-row slice of the *merged* training set, so it carries every column the real
train_transaction.csv + train_identity.csv pair has. This script splits it back
into the four raw files the ingestion stage expects and bootstraps it up to a
few thousand rows so 2-fold CV always sees both classes.

It exists so the Airflow DAG can be exercised end to end on a fresh machine
without a Kaggle token. It is NOT a substitute for the real dataset -- model
quality on it is meaningless.

Usage:
    python scripts/make_smoke_data.py [--out data/unzipped] [--rows 3000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "streamlit_app" / "sample_data" / "raw_transactions.csv"
IDENTITY_FIRST_COL = "id_01"  # everything from here on belongs to *_identity.csv


def _bootstrap(df: pd.DataFrame, n_rows: int, seed: int, id_start: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.sample(n=n_rows, replace=True, random_state=seed).reset_index(drop=True)
    out["TransactionID"] = np.arange(id_start, id_start + n_rows)
    # Keep TransactionDT monotone-ish so the sort in feature engineering is meaningful.
    out["TransactionDT"] = np.sort(rng.integers(86_400, 86_400 * 180, size=n_rows))
    if "TransactionAmt" in out.columns:
        out["TransactionAmt"] = (out["TransactionAmt"] * rng.uniform(0.8, 1.2, size=n_rows)).round(2)
    return out


def make_smoke_data(out_dir: Path, n_rows: int = 3000, seed: int = 42) -> list[Path]:
    src = pd.read_csv(SAMPLE, low_memory=False)
    cols = list(src.columns)
    id_idx = cols.index(IDENTITY_FIRST_COL)
    trans_cols = cols[:id_idx]
    ident_cols = ["TransactionID"] + cols[id_idx:]

    train = _bootstrap(src, n_rows, seed, id_start=2_987_000)
    test = _bootstrap(src, n_rows // 2, seed + 1, id_start=3_663_549).drop(columns=["isFraud"])

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def dump(df: pd.DataFrame, name: str) -> None:
        p = out_dir / name
        df.to_csv(p, index=False)
        written.append(p)

    dump(train[trans_cols], "train_transaction.csv")
    dump(test[[c for c in trans_cols if c != "isFraud"]], "test_transaction.csv")

    # Identity rows only exist for a subset of transactions in the real data.
    ident_mask_train = train[cols[id_idx:]].notna().any(axis=1)
    ident_mask_test = test[cols[id_idx:]].notna().any(axis=1)
    dump(train.loc[ident_mask_train, ident_cols], "train_identity.csv")
    dump(test.loc[ident_mask_test, ident_cols], "test_identity.csv")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "unzipped"), help="output directory")
    ap.add_argument("--rows", type=int, default=3000, help="rows in train_transaction.csv")
    args = ap.parse_args()

    for p in make_smoke_data(Path(args.out), n_rows=args.rows):
        print(f"  wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    print("✓ Smoke dataset ready (synthetic -- for pipeline testing only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
