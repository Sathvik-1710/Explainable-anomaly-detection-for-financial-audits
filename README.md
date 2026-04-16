# Explainable Anomaly Detection for Financial Audits

A **schema-adaptive** machine learning system built for financial auditing. The system uses an unsupervised **Isolation Forest** to detect anomalies and a three-layer explainability architecture (SHAP, Rule-based, and Natural Language) to provide auditor-grade reasoning for every flagged transaction.

## Key Features

1. **Schema-Adaptive Engine**: Works with *any* CSV or Excel ledger. You don't need a fixed column structure; simply map your columns in the UI.
2. **Three-Layer Explainability**:
    - **SHAP (TreeExplainer)**: Mathematically exact feature contributions.
    - **Rule-Based**: Business-logic thresholds (e.g., "5.2x the account average").
    - **Natural Language**: Human-readable summaries perfect for compliance reports.
3. **Advanced Feature Engineering**: Automatically extracts 14 behavioural, temporal, and statistical features (e.g., unusual time-of-day, rare vendors, account frequency spikes).
4. **Offline Inference**: Uses a pre-trained model for zero-shot detection. Missing mapped columns are gracefully padded, meaning it handles reduced-feature datasets without breaking.
5. **Robust Quality Assurance**: Backed by a comprehensive 62-case test suite covering 100% of the core calculation engine (`engine.py`).

## Architecture

The project strictly separates the presentation layer from the computation layer:

- **`engine.py`**: Pure Python data processing, ML inference, and XAI calculations. 100% independent of Streamlit.
- **`app.py`**: The Streamlit user interface (5-screen flow: Upload → Schema Map → Detect → Explain → Export).
- **`train_offline.py`**: The offline pipeline used to generate the frozen `models/model.pkl`.

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Tests (Optional but Recommended)
Verify the core engine is operating correctly:
```bash
pytest tests/test_engine.py -v
```

### 3. Launch Application
```bash
streamlit run app.py
```

## Using the System

1. Open your browser to `http://localhost:8501`.
2. Upload the provided demo dataset: `sample_data/sample_transactions.csv`.
3. Map the required columns (`Amount` and `Time/Date`) and optional columns (`Vendor`, `Location`, `Account ID`). *Note: Mapping all columns yields the highest accuracy.*
4. Adjust the **Contamination Rate** to set the strictness of the anomaly flagging (0.05 targets 5% of data, recommended for strict audits).
5. Review the flagged anomalies, inspect their unique 3-layer explanations, and download the resulting CSV Audit Report.

## Model Performance

The pre-trained Isolation Forest model achieved the following performance on a held-out dataset of ~11,000 transactions:
- **AUC-ROC**: 0.8038
- **Precision**: 0.7576 (at 0.05 contamination limit)
- **Top Fraud Indicators**: Transactions at rarely-seen locations, spikes in account frequency, and massive deviations from an account's historical average.

*(For full methodology, results, and architecture details, refer to `report/mini_project_report.md`)*

## License
MIT