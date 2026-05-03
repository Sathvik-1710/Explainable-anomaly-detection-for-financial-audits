# 🌟 North Star Document
## Explainable, Schema-Adaptive Anomaly Detection for Financial Audits

> **Purpose**: This document is the single source of truth for the project's vision, scope, architecture, milestones, and success criteria. Every development decision should trace back to this document.

---

## 1. Vision Statement

> Build an **end-to-end machine learning system** that detects anomalous financial transactions from **ANY uploaded dataset** and generates **human-readable, audit-grade explanations** for every flag — empowering auditors to make informed decisions with confidence, not blind trust.

The system must NOT depend on a fixed dataset or fixed column names. It must adapt to new CSV/Excel files dynamically.

### Core Principles

| Principle | Meaning |
|-----------|---------|
| 🔄 **Schema Adaptability** | System works with ANY dataset — user maps columns, system adapts |
| 🔍 **Explainability First** | Every anomaly has 3 layers: SHAP (model), rules (business), NL (human) |
| 🎯 **Audit-Ready Output** | Results are formatted for real-world auditor consumption |
| 🧩 **Modular Pipeline** | upload → mapping → features → model → explain → report |
| 📐 **Reproducibility** | Any run is reproducible given the same input and config |
| ⚖️ **Intellectual Honesty** | Transparent about what the system can and cannot explain |
| 🛡️ **Graceful Degradation** | Missing optional columns → system still works in reduced mode |

---

## 2. Problem Framing

### The Gap

| Current State | Desired State |
|---------------|---------------|
| Anomaly systems depend on fixed schemas | System adapts to any CSV/Excel |
| ML fraud systems are black boxes | 3-layer explanation for every anomaly |
| Auditors distrust opaque AI flags | Evidence-backed, interpretable reports |
| Systems break on new datasets | Column mapping ensures reliability |
| Explanations use technical jargon | Plain English explanations with business context |

### Target Users

| User | Need |
|------|------|
| **Financial Auditors** | Understand *why* a transaction is suspicious before escalating |
| **Compliance Officers** | Evidence trail for regulatory submissions |
| **Data Science Students** | Learn applied ML with real-world explainability |
| **Academic Evaluators** | Assess end-to-end ML pipeline design and presentation |

---

## 3. Critical Architecture Decisions

### 3.1 Dataset Approach — SCHEMA-ADAPTIVE

| Decision | Detail |
|----------|--------|
| **Primary mode** | Any CSV/Excel file via Column Mapping UI |
| **Kaggle CC Fraud** | Supported but not the main demo — backward compatible |
| **Required columns** | Only amount + time/date must be mapped |
| **Optional columns** | vendor, location, account_id, label (for evaluation) |
| **File formats** | CSV (.csv) and Excel (.xlsx, .xls) |

### 3.2 Explainability — 3-LAYER APPROACH

| Layer | Method | Purpose |
|-------|--------|---------|
| **Model-level** | SHAP TreeExplainer | Feature contributions from the model's perspective |
| **Rule-based** | Threshold comparisons | Business-meaningful flags (amount, time, vendor) |
| **Natural language** | Combined generator | Readable paragraph combining SHAP + rules |

> [!IMPORTANT]
> **Honesty requirement**: Every explanation must include a disclaimer: "Isolation Forest is unsupervised — anomaly ≠ confirmed fraud. Human review required."

### 3.3 Tech Stack

| Technology | Why |
|------------|-----|
| **Python 3.12+** | Industry standard for ML |
| **Pandas** | Tabular data processing |
| **scikit-learn** | Isolation Forest |
| **SHAP** | TreeExplainer for model-faithful explanations |
| **Streamlit** | Fastest Python → web app path |
| **Matplotlib + Seaborn** | Publication-quality plots |
| **openpyxl** | Excel file support |

### 3.4 Architecture — 2-FILE SPLIT

| File | Responsibility |
|------|---------------|
| `app.py` | Streamlit UI — 5 screens, session state, plots |
| `engine.py` | Core logic — parsing, features, model, SHAP, rules, NL, export |

This split is safe because `engine.py` has no Streamlit imports. Streamlit only re-executes `app.py`.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                                 │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ ANY CSV or Excel file                                     │   │
│  │ Examples: bank_transactions.csv, expenses.xlsx, etc.      │   │
│  └──────────────────────────┬────────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   COLUMN MAPPING LAYER                           │
│                                                                  │
│  User maps:  amount → "transaction_amount"                      │
│              time   → "date"                                     │
│              vendor → "merchant_name"      (optional)            │
│              location → "city"             (optional)            │
│              account_id → "customer_id"    (optional)            │
│              label → "is_fraud"            (optional)            │
│                                                                  │
│  Auto-detection: guesses best matches from column names          │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                         │
│                                                                  │
│  From Amount:  amount, log_amount, amount_zscore,               │
│                amount_deviation_from_mean                         │
│  From Time:    hour_of_day, day_of_week, is_weekend,            │
│                high_risk_time                                     │
│  From Account: amount_vs_account_avg, account_tx_frequency       │
│  From Vendor:  vendor_frequency, is_rare_vendor                  │
│  From Location: location_frequency, is_rare_location             │
│                                                                  │
│  ⚠️ MEANINGFUL features only — no raw PCA columns               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION LAYER                                │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ Isolation Forest (Pre-Trained Offline)                     │   │
│  │ • Pre-trained on historical data via train_offline.py     │   │
│  │ • Model & Scaler loaded via joblib                        │   │
│  │ • Automatically pads missing features for new data        │   │
│  └───────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXPLAINABILITY LAYER                             │
│                                                                  │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐  │
│  │ SHAP             │ │ Rule-Based       │ │ Natural Language │  │
│  │ TreeExplainer    │ │ Explanations     │ │ Generator        │  │
│  │ (model-faithful) │ │ (business logic) │ │ (combined output)│  │
│  │                  │ │                  │ │                  │  │
│  │ Per-feature      │ │ "Amount is 3.5×  │ │ "This $4,500    │  │
│  │ SHAP values +    │ │  the average"    │ │  transaction at  │  │
│  │ waterfall plots  │ │ "Unusual hour"   │ │  2 AM from an   │  │
│  │                  │ │ "Rare vendor"    │ │  unknown vendor  │  │
│  │                  │ │                  │ │  was flagged..." │  │
│  └──────────────────┘ └──────────────────┘ └─────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Streamlit     │  │ CSV Report   │  │ Visual Charts         │  │
│  │ Dashboard     │  │ (with meta   │  │ (Matplotlib/Seaborn)  │  │
│  │ (5 screens)   │  │  + NL expl.) │  │                       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Streamlit Dashboard — 5 Screens

| # | Screen | What it does |
|---|--------|-------------|
| 1 | **Upload** | Accepts CSV/Excel, shows preview, column types, missing values |
| 2 | **Column Mapping** | Dropdowns for each semantic role, auto-guesses from column names, feature preview |
| 3 | **Detection** | Contamination slider, pipeline overview, run button with progress |
| 4 | **Results** | Metrics (if labels), global SHAP, score distribution, flagged table, CSV export |
| 5 | **Explanation** | NL summary, rule-based findings, SHAP waterfall, top features, disclaimer |

---

## 6. Feature Engineering (Critical)

| Feature | Source | Required? |
|---------|--------|-----------|
| `amount` | Amount column | ✅ Yes |
| `log_amount` | Amount column | ✅ Yes |
| `amount_zscore` | Amount column | ✅ Yes |
| `amount_deviation_from_mean` | Amount column | ✅ Yes |
| `hour_of_day` | Time column | ✅ Yes |
| `day_of_week` | Time column | ✅ Yes |
| `is_weekend` | Time column | ✅ Yes |
| `high_risk_time` | Time column (0–5 AM) | ✅ Yes |
| `amount_vs_account_avg` | Amount + Account ID | Optional |
| `account_tx_frequency` | Time + Account ID | Optional |
| `vendor_frequency` | Vendor | Optional |
| `is_rare_vendor` | Vendor | Optional |
| `location_frequency` | Location | Optional |
| `is_rare_location` | Location | Optional |

---

## 7. Core Schema

**Required:**
- `amount` — transaction value (numeric)
- `time` — timestamp or date column

**Optional:**
- `vendor` — merchant/payee name
- `location` — city/country
- `account_id` — customer/account identifier
- `label` — ground truth (0=normal, 1=anomaly) for evaluation

---

## 8. Evaluation Framework

### When Labels Are Available
| Metric | How to Measure |
|--------|----------------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under ROC curve |
| **Confusion Matrix** | Full TP/FP/TN/FN breakdown |

> [!IMPORTANT]
> We do NOT use accuracy. Class imbalance makes it meaningless.

### When Labels Are NOT Available
| Metric | How to Measure |
|--------|----------------|
| **Anomaly count** | Number of flagged transactions |
| **Score distribution** | Histogram of anomaly scores |
| **Explanation coverage** | 100% of anomalies have NL explanations |

---

## 9. Risk Matrix

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Unsupervised model limitation** | Medium | Mandatory disclaimer everywhere |
| **Contamination misleads users** | Medium | Exposed in UI with explanation |
| **Time parsing fails** | Medium | Multiple strategies + fallback |
| **Too few features** | Low | Min 8 features from amount+time alone |
| **Excel formatting issues** | Low | openpyxl handles most formats |
| **SHAP computation time** | Medium | 100 background samples, not full set |

---

## 10. File Structure

```
d:\crt\
├── app.py                       # Streamlit dashboard (5 screens)
├── engine.py                    # Core logic (features, model, SHAP, rules, NL)
├── requirements.txt             # Pinned dependencies
├── NORTH_STAR.md                # This document
├── README.md                    # Project documentation
├── BUILD.md                     # Build guide for agents
├── LICENSE                      # MIT License
├── .gitignore
├── data/
│   └── .gitkeep                 # User places datasets here
├── sample_data/
│   └── sample_transactions.csv  # Demo dataset (100 transactions)
├── outputs/
│   └── .gitkeep                 # Exported reports saved here
└── report/
    └── mini_project_report.md   # Academic report
```

---

## 11. Documented Limitations

1. **Unsupervised Pre-Trainer** — Isolation Forest is pre-trained offline without fraud labels. SHAP explains the model's anomaly scoring, but anomaly ≠ confirmed fraud.
2. **Feature Alignment Fallback** — While the system is schema-adaptive, if an uploaded dataset lacks a column the pre-trained model expects (e.g. Vendor), the system fills it with generic zeroes, which marginally impacts those specific feature weights.
3. **Static Inference** — The model is frozen. Fraud patterns evolve (concept drift), so offline retraining (`train_offline.py`) is required periodically.
4. **Single-User, Local** — No auth, no multi-user. Not for production.
5. **Feature Quality** — With only amount + time, features are limited. More mapped columns = better detection.
6. **Time Parsing** — Complex or non-standard date formats may fail to parse.

---

## 12. Constraints

- Must support CSV and Excel
- Must be single-user local system
- Must validate inputs
- Must not crash on unknown schema
- Must never call plt.show() in Streamlit
- Must fit scaler on train data only

---

## 13. Mini Project vs Major Project Scope

| Feature | Mini Project (NOW) | Major Project (FUTURE) |
|---------|-------------------|----------------------|
| **Input** | CSV + Excel via mapping | API endpoints, real-time streams |
| **Model** | Isolation Forest | XGBoost + ensemble supervised |
| **Explainability** | SHAP + rules + NL | LIME + counterfactuals |
| **Dashboard** | Streamlit (2-file) | React + FastAPI (decoupled) |
| **Scope** | Single-user, local | Multi-user, authenticated, cloud |
| **Fairness** | Acknowledged limitation | Full fairness audit |

---

> **Last Updated**: April 15, 2026
> **Next Action**: Install dependencies → Run app → Test with sample_data/sample_transactions.csv
