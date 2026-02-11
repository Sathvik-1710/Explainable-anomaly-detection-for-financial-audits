# 🔍 Explainable Anomaly Detection for Financial Audits

An **explainable machine learning system** that detects anomalous financial transactions and provides **human-understandable explanations** for each flagged transaction — built for transparency in financial audits.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Step-by-Step Beginner Guide](#-step-by-step-beginner-guide-to-building-this-project)
  - [Step 1: Set Up Your Environment](#step-1-set-up-your-environment)
  - [Step 2: Generate or Load the Dataset](#step-2-generate-or-load-the-dataset)
  - [Step 3: Explore the Data (EDA)](#step-3-explore-the-data-eda)
  - [Step 4: Preprocess the Data](#step-4-preprocess-the-data)
  - [Step 5: Train the Anomaly Detection Model](#step-5-train-the-anomaly-detection-model)
  - [Step 6: Generate Explanations](#step-6-generate-explanations-for-anomalies)
  - [Step 7: Visualize the Results](#step-7-visualize-the-results)
  - [Step 8: Build a Summary Report](#step-8-build-a-summary-report)
- [Project Structure](#-project-structure)
- [Sample Output](#-sample-output)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Introduction

Financial institutions generate massive volumes of transaction data daily, making manual auditing nearly impossible. While automated anomaly detection can help, most existing systems act as **black boxes** — they flag transactions without explaining *why*.

This project solves that problem by building a system that:
1. **Detects anomalous transactions** using the Isolation Forest algorithm
2. **Explains each anomaly** by analyzing feature-level deviations in plain language
3. **Visualizes results** with clear, audit-friendly charts and reports

---

## 🎯 Problem Statement

> Design a machine learning–based system that detects anomalous financial transactions from historical data and provides simple, interpretable explanations for each detected anomaly.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔎 **Anomaly Detection** | Uses Isolation Forest to identify unusual transactions |
| 📝 **Explainable Results** | Generates plain-English explanations for every flagged transaction |
| 📊 **Visual Dashboards** | Distribution plots, scatter plots, and anomaly highlighting |
| 📁 **Structured Reports** | Exports anomaly reports to CSV for auditors |
| 🧪 **Synthetic Data Generator** | Includes a script to generate realistic financial transaction data |

---

## 🏗 Project Architecture

```
┌──────────────────┐
│  Transaction Data │ (CSV / Synthetic)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Data Preprocessing│
│  - Missing values  │
│  - Normalization   │
│  - Encoding        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Isolation Forest  │ (Anomaly Detection)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Explainability    │ (Feature Deviation Analysis)
│  Engine            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Visualizations &  │
│  Audit Report      │
└──────────────────┘
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Isolation Forest model |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical visualizations |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher installed on your system
- `pip` package manager
- A code editor (VS Code, PyCharm, or Jupyter Notebook)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sathvik-1710/Explainable-anomaly-detection-for-financial-audits.git
cd Explainable-anomaly-detection-for-financial-audits

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On macOS/Linux
# venv\Scripts\activate         # On Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📚 Step-by-Step Beginner Guide to Building This Project

> This guide walks you through building the entire project from scratch. Each step includes **what** you're doing, **why** you're doing it, and the **code** to do it.

---

### Step 1: Set Up Your Environment

**What:** Install Python and the required libraries.

**Why:** These libraries provide the tools for data handling, machine learning, and visualization.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Create a new Python file (e.g., `main.py`) or a Jupyter Notebook to follow along.

---

### Step 2: Generate or Load the Dataset

**What:** Create a synthetic financial transaction dataset (or load your own CSV).

**Why:** We need transaction data with features like amount, frequency, and category to train our model. Synthetic data lets us get started without needing real financial data.

```python
import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000, anomaly_ratio=0.05, random_state=42):
    """Generate synthetic financial transaction data with injected anomalies."""
    np.random.seed(random_state)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # --- Normal Transactions ---
    normal_data = pd.DataFrame({
        'transaction_id': range(1, n_normal + 1),
        'amount': np.random.normal(loc=500, scale=150, size=n_normal).clip(10),
        'transaction_frequency': np.random.poisson(lam=5, size=n_normal),
        'avg_balance': np.random.normal(loc=10000, scale=3000, size=n_normal).clip(100),
        'category': np.random.choice(['groceries', 'electronics', 'utilities', 'travel', 'dining'], size=n_normal),
        'is_weekend': np.random.choice([0, 1], size=n_normal, p=[0.7, 0.3]),
    })

    # --- Anomalous Transactions (injected outliers) ---
    anomaly_data = pd.DataFrame({
        'transaction_id': range(n_normal + 1, n_samples + 1),
        'amount': np.random.normal(loc=5000, scale=2000, size=n_anomalies).clip(1000),
        'transaction_frequency': np.random.poisson(lam=25, size=n_anomalies),
        'avg_balance': np.random.normal(loc=1000, scale=500, size=n_anomalies).clip(50),
        'category': np.random.choice(['electronics', 'travel'], size=n_anomalies),
        'is_weekend': np.random.choice([0, 1], size=n_anomalies, p=[0.3, 0.7]),
    })

    data = pd.concat([normal_data, anomaly_data], ignore_index=True)
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle

    return data

# Generate and save the dataset
df = generate_synthetic_data(n_samples=1000)
df.to_csv('transactions.csv', index=False)
print(f"Dataset shape: {df.shape}")
print(df.head())
```

**Key columns explained:**
| Column | Meaning |
|--------|---------|
| `amount` | Transaction dollar amount |
| `transaction_frequency` | How many transactions the user made recently |
| `avg_balance` | Average account balance |
| `category` | Type of purchase |
| `is_weekend` | Whether the transaction happened on a weekend |

---

### Step 3: Explore the Data (EDA)

**What:** Visualize and understand the data distribution before modeling.

**Why:** EDA helps you spot patterns, understand feature ranges, and verify that the data looks reasonable.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('transactions.csv')

# Basic statistics
print(df.describe())
print(f"\nCategory distribution:\n{df['category'].value_counts()}")

# Visualize distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['amount'], bins=50, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Transaction Amount Distribution')

sns.histplot(df['transaction_frequency'], bins=30, kde=True, ax=axes[1], color='coral')
axes[1].set_title('Transaction Frequency Distribution')

sns.histplot(df['avg_balance'], bins=50, kde=True, ax=axes[2], color='mediumseagreen')
axes[2].set_title('Average Balance Distribution')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150)
plt.show()
```

**💡 What to look for:**
- Are there clear outliers in the amount distribution?
- Do some users have unusually high transaction frequency?
- Is there a cluster of low-balance, high-amount transactions?

---

### Step 4: Preprocess the Data

**What:** Clean and transform the data so the ML model can use it.

**Why:** Machine learning models require numerical input. We need to encode categories and normalize feature scales.

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """Preprocess the transaction data for anomaly detection."""
    df_processed = df.copy()

    # 1. Handle missing values (if any)
    print(f"Missing values:\n{df_processed.isnull().sum()}")
    df_processed = df_processed.fillna(df_processed.median(numeric_only=True))

    # 2. Encode categorical features
    le = LabelEncoder()
    df_processed['category_encoded'] = le.fit_transform(df_processed['category'])

    # 3. Select features for the model
    feature_columns = ['amount', 'transaction_frequency', 'avg_balance',
                       'category_encoded', 'is_weekend']
    X = df_processed[feature_columns].copy()

    # 4. Normalize numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_columns
    )

    return X_scaled, scaler, le, feature_columns

X_scaled, scaler, label_encoder, feature_columns = preprocess_data(df)
print(f"\nPreprocessed data shape: {X_scaled.shape}")
print(X_scaled.head())
```

**Key preprocessing steps explained:**
1. **Missing values** → Filled with median (robust to outliers)
2. **Label Encoding** → Converts categories like "groceries" to numbers like 0, 1, 2...
3. **StandardScaler** → Scales all features to have mean=0 and std=1 (important for Isolation Forest)

---

### Step 5: Train the Anomaly Detection Model

**What:** Use the **Isolation Forest** algorithm to find anomalous transactions.

**Why:** Isolation Forest works by randomly partitioning data — anomalies are easier to isolate (fewer partitions needed), so they get lower scores. It's unsupervised, meaning we don't need labeled "fraud" data.

```python
from sklearn.ensemble import IsolationForest

def train_isolation_forest(X_scaled, contamination=0.05, random_state=42):
    """Train an Isolation Forest model for anomaly detection."""
    model = IsolationForest(
        n_estimators=100,         # Number of trees in the forest
        contamination=contamination,  # Expected proportion of anomalies
        random_state=random_state,
        verbose=0
    )

    # Fit the model and predict
    predictions = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    return model, predictions, scores

model, predictions, anomaly_scores = train_isolation_forest(X_scaled)

# Add results back to the original dataframe
df['anomaly_label'] = predictions            # -1 = anomaly, 1 = normal
df['anomaly_score'] = anomaly_scores         # Lower score = more anomalous
df['is_anomaly'] = df['anomaly_label'] == -1

print(f"\n--- Detection Results ---")
print(f"Total transactions:  {len(df)}")
print(f"Normal transactions: {(df['anomaly_label'] == 1).sum()}")
print(f"Anomalies detected:  {(df['anomaly_label'] == -1).sum()}")
```

**Understanding the output:**
- `anomaly_label = -1` → The model thinks this transaction is **anomalous**
- `anomaly_label = 1` → The model thinks this transaction is **normal**
- `anomaly_score` → A continuous score; **more negative = more anomalous**

---

### Step 6: Generate Explanations for Anomalies

**What:** For each flagged transaction, explain *which features* made it anomalous and *by how much*.

**Why:** This is the **core value** of the project. Auditors need to know *why* a transaction was flagged, not just *that* it was flagged.

```python
def explain_anomalies(df, feature_columns, threshold_std=2.0):
    """Generate human-readable explanations for each detected anomaly."""
    anomalies = df[df['is_anomaly']].copy()
    normal = df[~df['is_anomaly']]

    # Calculate statistics from normal transactions
    normal_stats = {}
    for col in feature_columns:
        if col in df.columns:
            normal_stats[col] = {
                'mean': normal[col].mean(),
                'std': normal[col].std()
            }

    explanations = []

    for idx, row in anomalies.iterrows():
        reasons = []
        deviations = {}

        for col in feature_columns:
            if col not in normal_stats or col == 'category_encoded':
                continue

            value = row[col] if col in row.index else None
            if value is None:
                continue

            mean = normal_stats[col]['mean']
            std = normal_stats[col]['std']

            if std == 0:
                continue

            z_score = (value - mean) / std

            if abs(z_score) > threshold_std:
                direction = "above" if z_score > 0 else "below"
                reasons.append(
                    f"  • {col}: {value:.2f} is {abs(z_score):.1f}σ {direction} normal "
                    f"(normal range: {mean - threshold_std*std:.2f} – {mean + threshold_std*std:.2f})"
                )
                deviations[col] = z_score

        explanation = {
            'transaction_id': row['transaction_id'],
            'anomaly_score': row['anomaly_score'],
            'num_deviations': len(reasons),
            'explanation': '\n'.join(reasons) if reasons else '  • Combination of mildly unusual features',
            'deviations': deviations
        }
        explanations.append(explanation)

    return pd.DataFrame(explanations)

# Use original (non-encoded) feature columns for readable explanations
explanation_features = ['amount', 'transaction_frequency', 'avg_balance', 'is_weekend']
explanations_df = explain_anomalies(df, explanation_features)

# Display some explanations
print("\n===== ANOMALY EXPLANATIONS =====\n")
for _, row in explanations_df.head(5).iterrows():
    print(f"Transaction ID: {int(row['transaction_id'])}")
    print(f"Anomaly Score:  {row['anomaly_score']:.4f}")
    print(f"Reasons:")
    print(row['explanation'])
    print("-" * 60)
```

**Example output:**
```
Transaction ID: 847
Anomaly Score:  -0.2341
Reasons:
  • amount: 6523.40 is 8.2σ above normal (normal range: 200.00 – 800.00)
  • transaction_frequency: 28 is 5.1σ above normal (normal range: 1.00 – 9.00)
  • avg_balance: 432.10 is 3.1σ below normal (normal range: 4000.00 – 16000.00)
```

---

### Step 7: Visualize the Results

**What:** Create clear, audit-friendly visualizations.

**Why:** Visual evidence makes it easier for auditors to understand and act on the flagged transactions.

```python
def plot_anomaly_results(df):
    """Create comprehensive visualizations of anomaly detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = df['is_anomaly'].map({True: 'red', False: 'steelblue'})

    # 1. Amount vs Frequency scatter plot
    axes[0, 0].scatter(df['amount'], df['transaction_frequency'],
                       c=colors, alpha=0.6, edgecolors='white', linewidth=0.5)
    axes[0, 0].set_xlabel('Transaction Amount ($)')
    axes[0, 0].set_ylabel('Transaction Frequency')
    axes[0, 0].set_title('Amount vs Frequency (Red = Anomaly)')

    # 2. Amount vs Average Balance
    axes[0, 1].scatter(df['amount'], df['avg_balance'],
                       c=colors, alpha=0.6, edgecolors='white', linewidth=0.5)
    axes[0, 1].set_xlabel('Transaction Amount ($)')
    axes[0, 1].set_ylabel('Average Balance ($)')
    axes[0, 1].set_title('Amount vs Balance (Red = Anomaly)')

    # 3. Anomaly score distribution
    axes[1, 0].hist(df[~df['is_anomaly']]['anomaly_score'], bins=40,
                    alpha=0.7, label='Normal', color='steelblue')
    axes[1, 0].hist(df[df['is_anomaly']]['anomaly_score'], bins=20,
                    alpha=0.7, label='Anomaly', color='red')
    axes[1, 0].set_xlabel('Anomaly Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Anomaly Score Distribution')
    axes[1, 0].legend()

    # 4. Feature comparison (box plots)
    anomaly_amounts = df[df['is_anomaly']]['amount']
    normal_amounts = df[~df['is_anomaly']]['amount']
    axes[1, 1].boxplot([normal_amounts, anomaly_amounts],
                       labels=['Normal', 'Anomaly'],
                       patch_artist=True,
                       boxprops=[dict(facecolor='steelblue'), dict(facecolor='red')])
    axes[1, 1].set_ylabel('Transaction Amount ($)')
    axes[1, 1].set_title('Amount Distribution: Normal vs Anomaly')

    plt.suptitle('Explainable Anomaly Detection — Results Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('anomaly_results.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_anomaly_results(df)
```

---

### Step 8: Build a Summary Report

**What:** Export the results to a clean CSV report that auditors can review.

**Why:** This is the deliverable — a structured file listing each anomaly with its explanation.

```python
def generate_audit_report(df, explanations_df, output_file='audit_report.csv'):
    """Generate a final audit report combining transaction data with explanations."""
    anomalies = df[df['is_anomaly']].copy()
    report = anomalies.merge(
        explanations_df[['transaction_id', 'explanation', 'num_deviations']],
        on='transaction_id',
        how='left'
    )
    report = report.sort_values('anomaly_score', ascending=True)

    # Select columns for the report
    report_columns = ['transaction_id', 'amount', 'transaction_frequency',
                      'avg_balance', 'category', 'is_weekend',
                      'anomaly_score', 'num_deviations', 'explanation']
    report = report[report_columns]

    report.to_csv(output_file, index=False)
    print(f"\n✅ Audit report saved to '{output_file}'")
    print(f"   Total anomalies reported: {len(report)}")
    print(f"\nTop 5 most anomalous transactions:")
    print(report.head().to_string(index=False))

    return report

report = generate_audit_report(df, explanations_df)
```

---

## 📁 Project Structure

```
Explainable-anomaly-detection-for-financial-audits/
│
├── README.md                  # This file
├── LICENSE                    # MIT License
├── main.py                    # Complete pipeline script
├── transactions.csv           # Generated/input dataset
├── audit_report.csv           # Output anomaly report
├── eda_distributions.png      # EDA visualization
├── anomaly_results.png        # Results dashboard
└── requirements.txt           # Python dependencies
```

---

## 📊 Sample Output

### Detection Summary
```
--- Detection Results ---
Total transactions:  1000
Normal transactions: 950
Anomalies detected:  50
```

### Example Explanation
```
Transaction ID: 847
Anomaly Score:  -0.2341
Reasons:
  • amount: 6523.40 is 8.2σ above normal (normal range: 200.00 – 800.00)
  • transaction_frequency: 28 is 5.1σ above normal (normal range: 1.00 – 9.00)
  • avg_balance: 432.10 is 3.1σ below normal (normal range: 4000.00 – 16000.00)
```

---

## 🔮 Future Enhancements

- [ ] Add SHAP (SHapley Additive exPlanations) for model-level explainability
- [ ] Build a Streamlit / Flask web dashboard for interactive exploration
- [ ] Support real-world datasets (e.g., Kaggle credit card fraud dataset)
- [ ] Add more anomaly detection models (LOF, DBSCAN, Autoencoders)
- [ ] Implement time-series anomaly detection for temporal patterns
- [ ] Add email/Slack alerting for newly detected anomalies
- [ ] Dockerize the application for easy deployment

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for transparent financial auditing
</p>
