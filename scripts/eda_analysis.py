# ============================================================
# scripts/eda_analysis.py -- Exploratory Data Analysis
#
# Generates distribution plots, correlation heatmap, class
# balance, time patterns, and saves figures to results/figures/.
#
# Usage:
#   python scripts/eda_analysis.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 10})


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'combined_train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'combined_test.csv'))
    df = pd.concat([train, test], ignore_index=True)
    print(f"Loaded {len(df):,} rows ({len(train):,} train + {len(test):,} test)")
    return df


def fig1_class_balance(df):
    """Fraud vs Normal distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    counts = df['is_fraud'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(['Normal (0)', 'Fraud (1)'], counts.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}', ha='center', fontweight='bold')
    ax1.set_title('Class Distribution', fontweight='bold')
    ax1.set_ylabel('Count')

    # Pie chart
    ax2.pie(counts.values, labels=['Normal', 'Fraud'],
            colors=colors, autopct='%1.2f%%', startangle=90,
            textprops={'fontweight': 'bold'})
    ax2.set_title('Class Proportion', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_class_balance.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def fig2_amount_distribution(df):
    """Transaction amount distribution by class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Overall histogram
    axes[0].hist(df['amount'], bins=100, color='#3498db', alpha=0.7, edgecolor='white')
    axes[0].set_title('Amount Distribution (All)', fontweight='bold')
    axes[0].set_xlabel('Amount ($)')
    axes[0].set_ylabel('Count')
    axes[0].set_xlim(0, df['amount'].quantile(0.99))

    # By class - log scale
    normal = df[df['is_fraud'] == 0]['amount']
    fraud = df[df['is_fraud'] == 1]['amount']
    axes[1].hist(normal, bins=80, alpha=0.6, label='Normal', color='#2ecc71', edgecolor='white')
    axes[1].hist(fraud, bins=80, alpha=0.6, label='Fraud', color='#e74c3c', edgecolor='white')
    axes[1].set_title('Amount by Class', fontweight='bold')
    axes[1].set_xlabel('Amount ($)')
    axes[1].legend()
    axes[1].set_xlim(0, df['amount'].quantile(0.99))

    # Box plot by class
    df_box = df[df['amount'] <= df['amount'].quantile(0.99)]
    sns.boxplot(data=df_box, x='is_fraud', y='amount', ax=axes[2],
                palette=['#2ecc71', '#e74c3c'])
    axes[2].set_xticklabels(['Normal', 'Fraud'])
    axes[2].set_title('Amount Box Plot by Class', fontweight='bold')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('Amount ($)')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_amount_distribution.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def fig3_time_patterns(df):
    """Transaction patterns by hour and day of week."""
    df_time = df.copy()
    df_time['timestamp'] = pd.to_datetime(df_time['timestamp'], errors='coerce')
    df_time = df_time.dropna(subset=['timestamp'])
    df_time['hour'] = df_time['timestamp'].dt.hour
    df_time['day_of_week'] = df_time['timestamp'].dt.day_name()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By hour
    hour_normal = df_time[df_time['is_fraud'] == 0].groupby('hour').size()
    hour_fraud = df_time[df_time['is_fraud'] == 1].groupby('hour').size()
    hours = range(24)
    axes[0].bar(hours, [hour_normal.get(h, 0) for h in hours],
                alpha=0.6, label='Normal', color='#2ecc71')
    ax2 = axes[0].twinx()
    ax2.bar(hours, [hour_fraud.get(h, 0) for h in hours],
            alpha=0.6, label='Fraud', color='#e74c3c')
    axes[0].set_title('Transactions by Hour', fontweight='bold')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Normal Count', color='#2ecc71')
    ax2.set_ylabel('Fraud Count', color='#e74c3c')
    axes[0].legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Fraud rate by hour
    hourly = df_time.groupby('hour')['is_fraud'].mean()
    axes[1].bar(hourly.index, hourly.values, color='#e74c3c', alpha=0.7, edgecolor='white')
    axes[1].axhline(y=df['is_fraud'].mean(), color='gray', linestyle='--', label='Overall rate')
    axes[1].set_title('Fraud Rate by Hour', fontweight='bold')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Fraud Rate')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_time_patterns.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def fig4_top_merchants(df):
    """Top merchants and fraud rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top 15 merchants by volume
    top_merchants = df['merchant'].value_counts().head(15)
    axes[0].barh(range(len(top_merchants)), top_merchants.values, color='#3498db', edgecolor='white')
    axes[0].set_yticks(range(len(top_merchants)))
    labels = [m[:30] for m in top_merchants.index]
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title('Top 15 Merchants by Volume', fontweight='bold')
    axes[0].set_xlabel('Transaction Count')

    # Fraud rate by top merchants
    merchant_fraud = df.groupby('merchant')['is_fraud'].agg(['mean', 'count'])
    merchant_fraud = merchant_fraud[merchant_fraud['count'] >= 50].sort_values('mean', ascending=False).head(15)
    axes[1].barh(range(len(merchant_fraud)), merchant_fraud['mean'].values, color='#e74c3c', edgecolor='white')
    axes[1].set_yticks(range(len(merchant_fraud)))
    labels = [m[:30] for m in merchant_fraud.index]
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title('Highest Fraud Rate Merchants (min 50 txns)', fontweight='bold')
    axes[1].set_xlabel('Fraud Rate')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_top_merchants.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def fig5_location_analysis(df):
    """City distribution and fraud rates."""
    fig, ax = plt.subplots(figsize=(12, 5))

    city_stats = df.groupby('city')['is_fraud'].agg(['mean', 'count'])
    city_stats = city_stats[city_stats['count'] >= 20].sort_values('mean', ascending=False).head(20)

    colors = ['#e74c3c' if r > df['is_fraud'].mean() else '#3498db' for r in city_stats['mean']]
    ax.barh(range(len(city_stats)), city_stats['mean'].values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(city_stats)))
    ax.set_yticklabels(city_stats.index, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=df['is_fraud'].mean(), color='gray', linestyle='--',
               label=f'Overall rate ({df["is_fraud"].mean():.2%})')
    ax.set_title('Fraud Rate by City (min 20 transactions)', fontweight='bold')
    ax.set_xlabel('Fraud Rate')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_location_analysis.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def fig6_correlation_heatmap(df):
    """Feature correlation matrix."""
    from engine import engineer_features

    mapping = {
        'amount': 'amount', 'time': 'timestamp',
        'vendor': 'merchant', 'location': 'city',
        'account_id': 'account_id', 'label': 'is_fraud',
    }

    # Sample for speed
    sample = df.sample(n=min(10000, len(df)), random_state=42)
    features, names, _, _ = engineer_features(sample, mapping)
    features['is_fraud'] = sample['is_fraud'].values

    fig, ax = plt.subplots(figsize=(12, 10))
    corr = features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_correlation_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def fig7_source_comparison(df):
    """Compare Sparkov vs Synthetic distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Amount by source
    for src, color in [('sparkov', '#3498db'), ('synthetic', '#e67e22')]:
        subset = df[df['source'] == src]['amount']
        axes[0].hist(subset, bins=80, alpha=0.5, label=src.title(),
                     color=color, edgecolor='white')
    axes[0].set_title('Amount Distribution by Source', fontweight='bold')
    axes[0].set_xlabel('Amount ($)')
    axes[0].set_xlim(0, df['amount'].quantile(0.98))
    axes[0].legend()

    # Fraud rate by source
    source_fraud = df.groupby('source')['is_fraud'].mean()
    bars = axes[1].bar(source_fraud.index.str.title(), source_fraud.values,
                       color=['#3498db', '#e67e22'], edgecolor='white')
    for bar, val in zip(bars, source_fraud.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.2%}', ha='center', fontweight='bold')
    axes[1].set_title('Fraud Rate by Source', fontweight='bold')
    axes[1].set_ylabel('Fraud Rate')

    # Count by source
    source_counts = df.groupby('source').size()
    axes[2].bar(source_counts.index.str.title(), source_counts.values,
                color=['#3498db', '#e67e22'], edgecolor='white')
    for i, (idx, val) in enumerate(source_counts.items()):
        axes[2].text(i, val + 200, f'{val:,}', ha='center', fontweight='bold')
    axes[2].set_title('Record Count by Source', fontweight='bold')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig7_source_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def print_statistics(df):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\n  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  --- Amount ---")
    print(f"  Mean:   ${df['amount'].mean():,.2f}")
    print(f"  Median: ${df['amount'].median():,.2f}")
    print(f"  Std:    ${df['amount'].std():,.2f}")
    print(f"  Min:    ${df['amount'].min():,.2f}")
    print(f"  Max:    ${df['amount'].max():,.2f}")
    print(f"  P95:    ${df['amount'].quantile(0.95):,.2f}")
    print(f"  P99:    ${df['amount'].quantile(0.99):,.2f}")
    print(f"  Skew:   {df['amount'].skew():.3f}")
    print(f"\n  --- Class Balance ---")
    print(f"  Normal: {(df['is_fraud'] == 0).sum():,} ({(df['is_fraud'] == 0).mean():.2%})")
    print(f"  Fraud:  {(df['is_fraud'] == 1).sum():,} ({(df['is_fraud'] == 1).mean():.2%})")
    print(f"\n  --- Sources ---")
    for src, count in df['source'].value_counts().items():
        fraud_r = df[df['source'] == src]['is_fraud'].mean()
        print(f"  {src}: {count:,} rows, fraud rate: {fraud_r:.4%}")
    print(f"\n  --- Coverage ---")
    print(f"  Unique merchants: {df['merchant'].nunique():,}")
    print(f"  Unique cities:    {df['city'].nunique():,}")
    print(f"  Unique accounts:  {df['account_id'].nunique():,}")
    print(f"\n  --- Missing Values ---")
    missing = df.isnull().sum()
    if missing.any():
        for col, n in missing[missing > 0].items():
            print(f"  {col}: {n:,} ({n/len(df):.2%})")
    else:
        print("  None")


def main():
    print("=" * 60)
    print("Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()
    print_statistics(df)

    print("\nGenerating figures...")
    fig1_class_balance(df)
    fig2_amount_distribution(df)
    fig3_time_patterns(df)
    fig4_top_merchants(df)
    fig5_location_analysis(df)
    fig6_correlation_heatmap(df)
    fig7_source_comparison(df)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == '__main__':
    main()
