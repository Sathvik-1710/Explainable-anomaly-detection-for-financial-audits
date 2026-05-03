# ============================================================
# generate_synthetic_data.py — Realistic Financial Transaction Generator
#
# Produces a labeled dataset with 5 distinct fraud patterns:
#   1. Amounts far exceeding account history
#   2. Late-night / early-morning transactions
#   3. Rare merchant + rare location combos
#   4. Rapid succession (multiple txns within minutes)
#   5. Suspiciously round amounts ($500, $1000, etc.)
#
# Column names are DIFFERENT from Sparkov on purpose —
# this proves our system's schema adaptability.
#
# Usage:
#   python scripts/generate_synthetic_data.py
#   → Outputs: sample_data/synthetic_transactions.csv
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
N_ACCOUNTS = 200
N_NORMAL_TXN = 9500
N_FRAUD_TXN = 300       # ~3% fraud rate
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'synthetic_transactions.csv')

# ============================================================
# REFERENCE DATA
# ============================================================

MERCHANTS = [
    'Walmart', 'Target', 'Amazon', 'Costco', 'Starbucks',
    'McDonalds', 'Shell Gas', 'CVS Pharmacy', 'Home Depot', 'Netflix',
    'Uber', 'Lyft', 'Whole Foods', 'Best Buy', 'Apple Store',
    'Nike', 'Zara', 'Spotify', 'Grubhub', 'DoorDash',
    'Chevron', 'Safeway', 'Trader Joes', 'Walgreens', 'Kroger',
    'Delta Airlines', 'United Airlines', 'Marriott', 'Hilton', 'Airbnb',
]

RARE_MERCHANTS = [
    'CryptoXchange Ltd', 'QuickWire Transfers', 'GoldBullion Direct',
    'OffshorePay Inc', 'FastCash ATM Services', 'UnknownVendor_X9',
]

CITIES = [
    'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin',
    'Seattle', 'Denver', 'Boston', 'Nashville', 'Portland',
    'Atlanta', 'Miami', 'Minneapolis', 'Detroit', 'San Francisco',
]

RARE_LOCATIONS = [
    'Lagos', 'Minsk', 'Karachi', 'Tirana', 'Phnom Penh',
]

CHANNELS = ['Online', 'In-Store', 'ATM', 'Mobile App', 'Phone']

OCCUPATIONS = [
    'Engineer', 'Teacher', 'Doctor', 'Student', 'Retired',
    'Manager', 'Analyst', 'Designer', 'Driver', 'Nurse',
    'Lawyer', 'Accountant', 'Sales Rep', 'Freelancer', 'Chef',
]

CATEGORIES = [
    'Groceries', 'Dining', 'Transportation', 'Entertainment',
    'Shopping', 'Utilities', 'Healthcare', 'Travel',
    'Subscriptions', 'Gas', 'Education', 'Personal Care',
]

# ============================================================
# ACCOUNT PROFILES — each account has spending patterns
# ============================================================

def generate_accounts(n):
    """Create account profiles with realistic spending behavior."""
    accounts = []
    for i in range(n):
        acct_id = f'ACCT-{1000 + i}'
        age = np.random.choice(
            [np.random.randint(18, 25),    # young
             np.random.randint(25, 45),    # working adult
             np.random.randint(45, 65),    # senior professional
             np.random.randint(65, 85)],   # retired
            p=[0.15, 0.45, 0.30, 0.10]
        )

        # Spending profile based on age bracket
        if age < 25:
            avg_amount = np.random.uniform(15, 60)
            std_amount = avg_amount * 0.5
        elif age < 45:
            avg_amount = np.random.uniform(40, 200)
            std_amount = avg_amount * 0.4
        elif age < 65:
            avg_amount = np.random.uniform(80, 400)
            std_amount = avg_amount * 0.35
        else:
            avg_amount = np.random.uniform(30, 150)
            std_amount = avg_amount * 0.3

        # Each account has 2-4 preferred merchants
        n_preferred = np.random.randint(2, 5)
        preferred_merchants = list(np.random.choice(MERCHANTS, n_preferred, replace=False))

        # Home city
        home_city = np.random.choice(CITIES)

        accounts.append({
            'account_id': acct_id,
            'age': int(age),
            'occupation': np.random.choice(OCCUPATIONS),
            'avg_amount': avg_amount,
            'std_amount': std_amount,
            'preferred_merchants': preferred_merchants,
            'home_city': home_city,
        })

    return accounts


# ============================================================
# NORMAL TRANSACTIONS
# ============================================================

def generate_normal_transactions(accounts, n):
    """Generate realistic normal transaction patterns."""
    records = []
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    for _ in range(n):
        acct = np.random.choice(accounts)

        # Amount: log-normal distribution centered on account profile
        amount = max(0.50, np.random.lognormal(
            mean=np.log(acct['avg_amount']),
            sigma=0.5
        ))
        amount = round(amount, 2)

        # Date: random within range, weighted toward business hours
        day_offset = np.random.randint(0, date_range_days)
        hour = np.random.choice(
            range(24),
            p=_business_hour_weights()
        )
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        txn_date = start_date + timedelta(
            days=int(day_offset), hours=int(hour),
            minutes=int(minute), seconds=int(second)
        )

        # Merchant: 70% from preferred, 30% from general pool
        if np.random.random() < 0.7:
            merchant = np.random.choice(acct['preferred_merchants'])
        else:
            merchant = np.random.choice(MERCHANTS)

        # Location: 85% home city, 15% other cities
        if np.random.random() < 0.85:
            city = acct['home_city']
        else:
            city = np.random.choice(CITIES)

        # Category based on merchant type
        category = np.random.choice(CATEGORIES)

        # Channel: realistic distribution
        channel = np.random.choice(
            CHANNELS,
            p=[0.35, 0.30, 0.10, 0.20, 0.05]
        )

        records.append({
            'TransactionID': f'TXN-{len(records):06d}',
            'AccountNumber': acct['account_id'],
            'Timestamp': txn_date.strftime('%Y-%m-%d %H:%M:%S'),
            'Amount': amount,
            'MerchantName': merchant,
            'City': city,
            'Category': category,
            'Channel': channel,
            'CustomerAge': acct['age'],
            'CustomerOccupation': acct['occupation'],
            'IsFraud': 0,
        })

    return records


def _business_hour_weights():
    """Probability weights for each hour — peaks at 10am-2pm, low at 2am-5am."""
    weights = [
        0.005, 0.003, 0.002, 0.002, 0.002, 0.005,  # 0-5 AM
        0.015, 0.030, 0.050, 0.070, 0.080, 0.085,   # 6-11 AM
        0.090, 0.085, 0.075, 0.065, 0.060, 0.055,   # 12-5 PM
        0.050, 0.045, 0.040, 0.035, 0.025, 0.015,   # 6-11 PM
    ]
    # Normalize to sum to 1
    total = sum(weights)
    return [w / total for w in weights]


# ============================================================
# FRAUD PATTERNS — 5 distinct types
# ============================================================

def generate_fraud_transactions(accounts, n):
    """Generate anomalous transactions with distinct fraud patterns."""
    records = []
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    fraud_types = [
        _fraud_high_amount,
        _fraud_late_night,
        _fraud_rare_merchant_location,
        _fraud_rapid_succession,
        _fraud_round_amount,
    ]

    per_type = n // len(fraud_types)
    remainder = n % len(fraud_types)

    for i, fraud_fn in enumerate(fraud_types):
        count = per_type + (1 if i < remainder else 0)
        for _ in range(count):
            acct = np.random.choice(accounts)
            day_offset = np.random.randint(0, date_range_days)
            base_date = start_date + timedelta(days=int(day_offset))

            record = fraud_fn(acct, base_date, len(records) + N_NORMAL_TXN)
            records.append(record)

    return records


def _fraud_high_amount(acct, base_date, idx):
    """Pattern 1: Amount 5-20x the account's average."""
    multiplier = np.random.uniform(5, 20)
    amount = round(acct['avg_amount'] * multiplier, 2)
    hour = np.random.randint(8, 22)

    return {
        'TransactionID': f'TXN-{idx:06d}',
        'AccountNumber': acct['account_id'],
        'Timestamp': (base_date + timedelta(hours=int(hour), minutes=int(np.random.randint(0, 60)))).strftime('%Y-%m-%d %H:%M:%S'),
        'Amount': amount,
        'MerchantName': np.random.choice(MERCHANTS),
        'City': acct['home_city'],
        'Category': np.random.choice(['Shopping', 'Travel', 'Entertainment']),
        'Channel': np.random.choice(['Online', 'In-Store']),
        'CustomerAge': acct['age'],
        'CustomerOccupation': acct['occupation'],
        'IsFraud': 1,
    }


def _fraud_late_night(acct, base_date, idx):
    """Pattern 2: Transactions between 1am-4am."""
    hour = np.random.randint(1, 5)
    amount = round(np.random.uniform(50, 500), 2)

    return {
        'TransactionID': f'TXN-{idx:06d}',
        'AccountNumber': acct['account_id'],
        'Timestamp': (base_date + timedelta(hours=int(hour), minutes=int(np.random.randint(0, 60)))).strftime('%Y-%m-%d %H:%M:%S'),
        'Amount': amount,
        'MerchantName': np.random.choice(MERCHANTS + RARE_MERCHANTS),
        'City': acct['home_city'],
        'Category': np.random.choice(['Entertainment', 'Shopping', 'Transportation']),
        'Channel': np.random.choice(['ATM', 'Online']),
        'CustomerAge': acct['age'],
        'CustomerOccupation': acct['occupation'],
        'IsFraud': 1,
    }


def _fraud_rare_merchant_location(acct, base_date, idx):
    """Pattern 3: Rare merchant + foreign location."""
    hour = np.random.randint(6, 23)
    amount = round(np.random.uniform(100, 2000), 2)

    return {
        'TransactionID': f'TXN-{idx:06d}',
        'AccountNumber': acct['account_id'],
        'Timestamp': (base_date + timedelta(hours=int(hour), minutes=int(np.random.randint(0, 60)))).strftime('%Y-%m-%d %H:%M:%S'),
        'Amount': amount,
        'MerchantName': np.random.choice(RARE_MERCHANTS),
        'City': np.random.choice(RARE_LOCATIONS),
        'Category': np.random.choice(['Shopping', 'Travel']),
        'Channel': 'Online',
        'CustomerAge': acct['age'],
        'CustomerOccupation': acct['occupation'],
        'IsFraud': 1,
    }


def _fraud_rapid_succession(acct, base_date, idx):
    """Pattern 4: Transaction shortly after another (simulated via close timestamp)."""
    hour = np.random.randint(8, 20)
    minute = np.random.randint(0, 55)
    amount = round(np.random.uniform(20, 300), 2)

    return {
        'TransactionID': f'TXN-{idx:06d}',
        'AccountNumber': acct['account_id'],
        'Timestamp': (base_date + timedelta(hours=int(hour), minutes=int(minute), seconds=int(np.random.randint(5, 30)))).strftime('%Y-%m-%d %H:%M:%S'),
        'Amount': amount,
        'MerchantName': np.random.choice(MERCHANTS),
        'City': np.random.choice(CITIES),  # Different from home
        'Category': np.random.choice(CATEGORIES),
        'Channel': np.random.choice(['Online', 'Mobile App']),
        'CustomerAge': acct['age'],
        'CustomerOccupation': acct['occupation'],
        'IsFraud': 1,
    }


def _fraud_round_amount(acct, base_date, idx):
    """Pattern 5: Suspiciously round amounts ($500, $1000, $2500, etc.)."""
    amount = float(np.random.choice([500, 750, 1000, 1500, 2000, 2500, 5000]))
    hour = np.random.randint(6, 23)

    return {
        'TransactionID': f'TXN-{idx:06d}',
        'AccountNumber': acct['account_id'],
        'Timestamp': (base_date + timedelta(hours=int(hour), minutes=int(np.random.randint(0, 60)))).strftime('%Y-%m-%d %H:%M:%S'),
        'Amount': amount,
        'MerchantName': np.random.choice(MERCHANTS + RARE_MERCHANTS),
        'City': np.random.choice(CITIES + RARE_LOCATIONS),
        'Category': np.random.choice(['Shopping', 'Travel', 'Utilities']),
        'Channel': np.random.choice(['Online', 'ATM', 'Phone']),
        'CustomerAge': acct['age'],
        'CustomerOccupation': acct['occupation'],
        'IsFraud': 1,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Synthetic Financial Transaction Dataset Generator")
    print("=" * 60)

    # Generate accounts
    accounts = generate_accounts(N_ACCOUNTS)
    print(f"[OK] Created {N_ACCOUNTS} account profiles")

    # Generate normal transactions
    normal = generate_normal_transactions(accounts, N_NORMAL_TXN)
    print(f"[OK] Generated {len(normal)} normal transactions")

    # Generate fraud transactions
    fraud = generate_fraud_transactions(accounts, N_FRAUD_TXN)
    print(f"[OK] Generated {len(fraud)} fraud transactions (5 patterns)")

    # Combine and shuffle
    all_records = normal + fraud
    np.random.shuffle(all_records)
    df = pd.DataFrame(all_records)

    # Re-index TransactionIDs after shuffle
    df['TransactionID'] = [f'TXN-{i:06d}' for i in range(len(df))]

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print()
    print(f"Dataset Summary:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Normal:             {(df['IsFraud'] == 0).sum():,}")
    print(f"   Fraud:              {(df['IsFraud'] == 1).sum():,}")
    print(f"   Fraud rate:         {df['IsFraud'].mean():.2%}")
    print(f"   Accounts:           {df['AccountNumber'].nunique()}")
    print(f"   Merchants:          {df['MerchantName'].nunique()}")
    print(f"   Cities:             {df['City'].nunique()}")
    print(f"   Date range:         {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"   Amount range:       ${df['Amount'].min():.2f} - ${df['Amount'].max():,.2f}")
    print()
    print(f"   Columns: {list(df.columns)}")
    print(f"   Saved to: {OUTPUT_FILE}")
    print()
    print("Column mapping for our system:")
    print("   Amount     -> 'Amount'")
    print("   Time       -> 'Timestamp'")
    print("   Vendor     -> 'MerchantName'")
    print("   Location   -> 'City'")
    print("   Account ID -> 'AccountNumber'")
    print("   Label      -> 'IsFraud'")


if __name__ == '__main__':
    main()
