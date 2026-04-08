"""
Sanity-check the creditcard.csv dataset structure and basic stats.
"""
import os
import pandas as pd

data_path = os.path.join("data", "creditcard.csv")

if not os.path.exists(data_path):
    raise SystemExit(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)

required_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
missing_cols = [c for c in required_cols if c not in df.columns]

print("Dataset path:", data_path)
print("Shape:", df.shape)

if missing_cols:
    print("Required columns present: NO")
    print("Missing columns:", missing_cols)
else:
    print("Required columns present: YES")

# Fraud stats
if "Class" in df.columns:
    fraud_count = int(df["Class"].sum())
    total_rows = len(df)
    fraud_pct = (fraud_count / total_rows * 100) if total_rows else 0.0
    print(f"Total rows: {total_rows:,}")
    print(f"Fraud count: {fraud_count:,}")
    print(f"Fraud percentage: {fraud_pct:.3f}%")
else:
    print("Class column missing; cannot compute fraud stats.")

# Missing values per column
print("\nMissing values per column:")
print(df.isna().sum())

# Basic stats for Amount and Class
print("\nBasic stats:")
if "Amount" in df.columns:
    print("Amount:")
    print(df["Amount"].describe())
else:
    print("Amount column missing.")

if "Class" in df.columns:
    print("\nClass:")
    print(df["Class"].describe())
else:
    print("Class column missing.")
