# Explainable Anomaly Detection for Financial Audits

A **schema-adaptive** anomaly detection system that works with **any** financial
transaction dataset. Upload a CSV or Excel file, map your columns, and get
explainable anomaly flags powered by Isolation Forest + SHAP + rule-based reasoning.

## Key Features

- **Schema-Adaptive**: Works with any dataset — no fixed column requirements
- **Column Mapping UI**: Tell the system what each column means
- **3-Layer Explainability**:
  - SHAP TreeExplainer (model-faithful feature contributions)
  - Rule-based reasoning (business-meaningful thresholds)
  - Natural language explanations (combined human-readable summary)
- **Meaningful Features**: Engineers hour_of_day, amount_zscore, is_rare_vendor, etc.
- **Audit-Ready Export**: CSV report with metadata, explanations, and disclaimers

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Build the Pre-Trained Offline Model
python train_offline.py

# 4. Run the app
streamlit run app.py
```

## Demo Dataset

A sample dataset is included at `sample_data/sample_transactions.csv` with 100
realistic transactions across 5 accounts, including anomalous patterns.

## How It Works

1. **Upload** any CSV or Excel file with financial transactions
2. **Map columns** — tell the system which column is "amount", "date", "vendor", etc.
3. **Run detection** — the system pads missing features and instantaneously predicts anomalies using the pre-trained offline model.
4. **Review results** — see flagged transactions with anomaly scores
5. **Read explanations** — each flag has SHAP analysis + rule-based reasoning + NL summary

## Supported Columns

| Role | Required? | Examples |
|------|-----------|---------|
| Amount | ✅ Yes | amount, total, value, price |
| Time/Date | ✅ Yes | date, timestamp, created_at |
| Vendor | Optional | merchant, vendor, payee |
| Location | Optional | city, country, region |
| Account ID | Optional | account_id, customer_id, user |
| Label | Optional | class, is_fraud, label (for evaluation) |

## Tech Stack

- Python 3.12+
- scikit-learn (Isolation Forest)
- SHAP (TreeExplainer)
- Streamlit
- pandas, numpy, matplotlib, seaborn, openpyxl

## Architecture

```
app.py      → Streamlit UI (5 screens)
engine.py   → Core logic (features, model, SHAP, rules, NL generator)
```

## Kaggle Credit Card Fraud Dataset

The system is backward compatible with the Kaggle CC Fraud dataset.
Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place at: `data/creditcard.csv`

When using this dataset, map: Amount → "Amount", Time → "Time", Label → "Class".

## License

MIT