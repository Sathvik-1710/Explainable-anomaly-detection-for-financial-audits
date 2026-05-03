# ============================================================
# scripts/prepare_data.py -- Combine all datasets + train/test split
#
# Merges Sparkov (real) and synthetic data into a unified schema,
# then does a stratified 80/20 split.
#
# Output:
#   data/combined_train.csv
#   data/combined_test.csv
#
# Usage:
#   python scripts/prepare_data.py
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data')

# Unified schema columns
UNIFIED_COLS = [
    'amount', 'timestamp', 'merchant', 'city', 'account_id',
    'category', 'is_fraud', 'source'
]


def load_sparkov(max_rows=50000):
    """
    Load Sparkov dataset. Samples to keep memory manageable.
    Combines fraudTrain + fraudTest.
    """
    parts = []
    for fname in ['fraudTrain.csv', 'fraudTest.csv']:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fname} not found")
            continue

        df = pd.read_csv(fpath)
        parts.append(df)
        print(f"  [OK] Loaded {fname}: {len(df):,} rows")

    if not parts:
        return pd.DataFrame(columns=UNIFIED_COLS)

    raw = pd.concat(parts, ignore_index=True)
    print(f"  [OK] Combined Sparkov: {len(raw):,} rows, fraud rate: {raw['is_fraud'].mean():.4%}")

    # Sample if too large (stratified to preserve fraud ratio)
    if len(raw) > max_rows:
        fraud = raw[raw['is_fraud'] == 1]
        normal = raw[raw['is_fraud'] == 0]

        # Keep ALL fraud rows, sample normal
        n_fraud = len(fraud)
        n_normal = max_rows - n_fraud
        normal_sample = normal.sample(n=min(n_normal, len(normal)), random_state=42)
        raw = pd.concat([fraud, normal_sample], ignore_index=True)
        print(f"  [OK] Sampled to {len(raw):,} rows (kept all {n_fraud:,} fraud)")

    # Map to unified schema
    unified = pd.DataFrame({
        'amount': raw['amt'],
        'timestamp': raw['trans_date_trans_time'],
        'merchant': raw['merchant'],
        'city': raw['city'],
        'account_id': raw['cc_num'].astype(str),
        'category': raw['category'],
        'is_fraud': raw['is_fraud'],
        'source': 'sparkov',
    })
    return unified


def load_synthetic():
    """Load the generated synthetic dataset."""
    fpath = os.path.join(SAMPLE_DIR, 'synthetic_transactions.csv')
    if not os.path.exists(fpath):
        print("  [SKIP] synthetic_transactions.csv not found")
        return pd.DataFrame(columns=UNIFIED_COLS)

    raw = pd.read_csv(fpath)
    print(f"  [OK] Loaded synthetic: {len(raw):,} rows, fraud rate: {raw['IsFraud'].mean():.4%}")

    unified = pd.DataFrame({
        'amount': raw['Amount'],
        'timestamp': raw['Timestamp'],
        'merchant': raw['MerchantName'],
        'city': raw['City'],
        'account_id': raw['AccountNumber'],
        'category': raw['Category'],
        'is_fraud': raw['IsFraud'],
        'source': 'synthetic',
    })
    return unified


def main():
    print("=" * 60)
    print("Data Preparation: Combine + Split")
    print("=" * 60)

    # --- Load all sources ---
    print("\n1. Loading datasets...")
    sparkov = load_sparkov(max_rows=50000)
    synthetic = load_synthetic()

    # --- Combine ---
    print("\n2. Combining datasets...")
    combined = pd.concat([sparkov, synthetic], ignore_index=True)

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Total rows:  {len(combined):,}")
    print(f"   From Sparkov:   {(combined['source'] == 'sparkov').sum():,}")
    print(f"   From Synthetic: {(combined['source'] == 'synthetic').sum():,}")
    print(f"   Overall fraud rate: {combined['is_fraud'].mean():.4%}")

    # --- Stratified split ---
    print("\n3. Stratified 80/20 split...")
    train_df, test_df = train_test_split(
        combined,
        test_size=0.2,
        random_state=42,
        stratify=combined['is_fraud']
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"   Train: {len(train_df):,} rows (fraud: {train_df['is_fraud'].mean():.4%})")
    print(f"   Test:  {len(test_df):,} rows (fraud: {test_df['is_fraud'].mean():.4%})")

    # --- Save ---
    print("\n4. Saving...")
    train_path = os.path.join(DATA_DIR, 'combined_train.csv')
    test_path = os.path.join(DATA_DIR, 'combined_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"   Train: {train_path}")
    print(f"   Test:  {test_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Train size:    {len(train_df):,}")
    print(f"  Test size:     {len(test_df):,}")
    print(f"  Columns:       {list(combined.columns)}")
    print(f"  Fraud (train): {train_df['is_fraud'].sum():,} / {len(train_df):,}")
    print(f"  Fraud (test):  {test_df['is_fraud'].sum():,} / {len(test_df):,}")
    print()
    print("Column mapping for our system:")
    print("   Amount     -> 'amount'")
    print("   Time       -> 'timestamp'")
    print("   Vendor     -> 'merchant'")
    print("   Location   -> 'city'")
    print("   Account ID -> 'account_id'")
    print("   Label      -> 'is_fraud'")
    print()
    print("Ready for pipeline.")


if __name__ == '__main__':
    main()
