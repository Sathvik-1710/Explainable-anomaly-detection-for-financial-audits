# Explainable Anomaly Detection for Financial Audits

Detects anomalous financial transactions using Isolation Forest and explains
every flagged transaction with SHAP TreeExplainer.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download creditcard.csv from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place at: `data/creditcard.csv`

## Run

```bash
streamlit run app.py
```

## Tech Stack

- Python 3.12+
- scikit-learn (Isolation Forest)
- SHAP (TreeExplainer)
- Streamlit
- pandas, numpy, matplotlib, seaborn