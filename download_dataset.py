"""
Download the Credit Card Fraud dataset from OpenML (no account needed).
Saves as data/creditcard.csv with correct column names.
"""
from sklearn.datasets import fetch_openml
import pandas as pd
import os

print("Downloading Credit Card Fraud dataset from OpenML...")
print("This may take 2-3 minutes (150MB download)...")

# Dataset ID 1597 = Credit Card Fraud Detection
data = fetch_openml(data_id=1597, as_frame=True, parser='auto')

df = data.frame

# Rename target column to 'Class' (OpenML uses the target name)
if 'Class' not in df.columns:
    # The target might be the last column or named differently
    df = df.rename(columns={data.target_names[0]: 'Class'}) if hasattr(data, 'target_names') else df

# Ensure Class is integer (0/1)
df['Class'] = df['Class'].astype(int)

# Save
os.makedirs('data', exist_ok=True)
output_path = os.path.join('data', 'creditcard.csv')
df.to_csv(output_path, index=False)

print(f"\n✅ Dataset saved to: {output_path}")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"   Total transactions: {len(df):,}")
print(f"   Fraud cases: {int(df['Class'].sum()):,}")
print(f"   Fraud rate: {df['Class'].mean()*100:.3f}%")
