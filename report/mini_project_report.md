# ABSTRACT

Financial audits increasingly rely on automated systems to detect anomalous transactions that may indicate fraud, errors, or compliance violations. Traditional rule-based systems often struggle with the volume and complexity of modern transaction data, while advanced machine learning models suffer from a "black box" problem, making it difficult for auditors to understand *why* a transaction was flagged. This project presents an **Explainable Anomaly Detection System** tailored for financial audits. By integrating **Isolation Forest** for robust unsupervised anomaly detection with **SHAP (SHapley Additive exPlanations)** for local and global model interpretability, the system bridges the gap between complex algorithmic detection and human-understandable reasoning. A three-layer explanation architecture — SHAP (model-faithful), rule-based (business logic), and natural language (auditor-readable) — is implemented over a schema-adaptive pipeline that accepts any CSV or Excel file without requiring a fixed column schema. The system is evaluated on a combined dataset of over 59,000 transactions, achieving an AUC-ROC of 0.8038 while generating clear, auditor-grade explanations for every flagged anomaly.

---

# LIST OF FIGURES
1. **Figure 1**: Class Distribution and Proportion
2. **Figure 2**: Amount Distribution by Class
3. **Figure 3**: Transactions and Fraud Rate by Hour
4. **Figure 4**: Top Merchants by Volume and Fraud Rate
5. **Figure 5**: Fraud Rate by City
6. **Figure 6**: Feature Correlation Heatmap
7. **Figure 7**: Source Comparison (Real vs Synthetic)
8. **Figure 8**: ROC Curve (AUC = 0.8038)
9. **Figure 9**: Precision-Recall Curve (AP = 0.5363)
10. **Figure 10**: Confusion Matrix Heatmap
11. **Figure 11**: Global Feature Importance (SHAP)
12. **Figure 12**: Metrics vs Contamination Sweep
13. **Figure 13**: Anomaly Score Distribution by Class

---

# LIST OF TABLES
1. **Table 1**: Dataset Schema Mapping
2. **Table 2**: Feature Engineering Summary
3. **Table 3**: Model Evaluation Metrics at varying Contamination settings
4. **Table 4**: Classification Report Summary
5. **Table 5**: Comparison of Global SHAP Feature Importance

---

# CHAPTER 1: INTRODUCTION

## 1.1 Introduction

The landscape of financial auditing has been transformed by the digitization of global economic transactions. Today, auditors must sift through millions of accounting entries to identify discrepancies that may indicate fraud, errors, or regulatory violations. Anomaly detection — the process of identifying data points that deviate significantly from a dataset's normal behaviour — has become a cornerstone of both fraud detection and systemic risk analysis.

Despite rapid advances in machine learning, adoption of ML-based anomaly detection in auditing remains limited. The central barrier is not accuracy, but **explainability**: auditing standards require that every flagged item be backed by sufficient, appropriate evidence that a human reviewer can assess. A model that flags a transaction but cannot explain why it was flagged is of limited use in a regulated environment.

## 1.2 Motivation

Current anomaly detection approaches typically rely on rigid, rule-based heuristics that generate high false-positive rates or fail to generalise to novel evasion patterns. While ML models address many generalisation issues, their opacity prevents widespread adoption in regulatory and auditing frameworks where the *reason* for flagging an event is as important as the flag itself.

Additionally, most existing systems are schema-dependent: they are trained on fixed datasets with fixed column names (e.g., Kaggle's `V1`–`V28` PCA features) and cannot adapt to a new organisation's ledger format without significant re-engineering. This project directly addresses both limitations.

## 1.3 Objective

The primary objectives of this project are:

1. To develop an unsupervised machine learning model (Isolation Forest) capable of detecting financial anomalies without requiring labelled fraud data.
2. To extract behavioural and temporal features from **any** structurally valid transactional dataset via a dynamic column-mapping interface.
3. To provide robust, localised explainability using a **three-layer explanation system**: SHAP (model-faithful), rule-based (business-meaningful), and natural language (auditor-readable).
4. To build an interactive Streamlit dashboard that allows auditors to upload data, configure detection, and inspect model logic — without any programming knowledge.
5. To pre-train the model offline and deploy it for instant inference on new datasets without retraining.

## 1.4 Scope of the Project

This project encompasses data ingestion, dynamic schema mapping, automated feature engineering, offline model training (Isolation Forest), real-time inference, evaluation, and a full explainability layer. The scope is defined within the context of structured tabular financial data (amounts, timestamps, merchants, locations, and account identifiers). Real-time streaming integration and multi-user deployment are considered beyond the current scope but represent defined future work.

---

# CHAPTER 2: LITERATURE SURVEY

## 2.1 Introduction

Financial fraud detection relies heavily on statistical and algorithmic anomaly detection. The literature reflects a progression from basic rule-based systems to deep learning models and, more recently, to explainable AI (XAI).

## 2.2 Anomaly Detection Methods

**Isolation Forest** (Liu et al., 2008) introduced a tree-ensemble approach that isolates anomalies rather than profiling normal behaviour. Unlike density-based approaches (LOF, DBSCAN), it operates in O(n log n) time and scales efficiently to high-dimensional data. Its use of random partitioning means that anomalies — which are few and different — require fewer splits to isolate. This property makes it particularly suited to financial fraud datasets, which are typically highly imbalanced (fraud rates of 0.1%–5%).

**SHAP** (Lundberg & Lee, 2017) provides a game-theoretically grounded framework for model interpretability. TreeSHAP, a specialised variant for tree ensembles, computes exact Shapley values in polynomial time, making it practical for production use. Since Isolation Forest uses ExtraTreeRegressor estimators internally, TreeSHAP can traverse these structures to compute feature-level contributions to each transaction's anomaly score.

## 2.3 Explainability Requirements in Financial Auditing

Auditing standards (ISA 315, ISA 500) explicitly mandate that audit evidence must be "sufficient" and "appropriate." Regulatory frameworks including GDPR's Right to Explanation and the EU AI Act (high-risk AI systems in finance) further require that automated decisions affecting individuals or organisations be explainable. LIME and SHAP are the two dominant post-hoc interpretability frameworks; SHAP is preferred for tree models due to its model-faithfulness guarantee (SHAP values sum exactly to the model output difference from baseline).

## 2.4 Schema-Adaptive Systems

Most published fraud detection systems are trained on fixed-schema datasets (e.g., the ULB Kaggle Credit Card Fraud dataset with PCA features V1–V28). Real-world deployments routinely fail because organisations use diverse ledger schemas. Literature on schema mapping and feature alignment (Doan et al., 2012) highlights the need for adaptive feature engineering pipelines that can operate over arbitrary column configurations.

## 2.5 Comparison of Related Work

| Approach | Dataset | Explainability | Schema-Adaptive |
|---|---|---|---|
| Isolation Forest only | Fixed (ULB) | None | No |
| Autoencoder + LIME | Fixed | Post-hoc | No |
| XGBoost + SHAP | Supervised (needs labels) | Full | No |
| **This project** | **Any CSV/Excel** | **3-layer (SHAP + rules + NL)** | **Yes** |

## 2.6 Literature Review Summary

Machine learning models are essential for modern financial security, but algorithms like Isolation Forest require supplementary XAI layers to convert statistical deviation into actionable audit insights. The schema-adaptive design of this system represents a practical contribution beyond existing academic implementations.

---

# CHAPTER 3: SYSTEM DESIGN

## 3.1 System Architecture

The system follows a modular two-file architecture:

- **`engine.py`** — Pure computation layer: schema mapping, feature engineering, model inference, SHAP, rule-based explanations, NL generation, and CSV export. Contains no Streamlit imports.
- **`app.py`** — Streamlit presentation layer: 5-screen UI, session state management, plots, and user interaction.

This separation ensures that `engine.py` is independently testable and re-usable (e.g., via CLI scripts or API integration) without Streamlit dependencies.

The end-to-end pipeline:

```
CSV/Excel Upload
    → Column Mapping (UI)
    → Feature Engineering (engine.py)
    → Pre-trained Isolation Forest Inference
    → SHAP Computation (TreeExplainer)
    → Rule-based Flags
    → Natural Language Explanation
    → Dashboard + CSV Export
```

## 3.2 Dataset Description

The system is evaluated on a combined dataset of **59,800 transactions**:

| Source | Type | Rows | Fraud Rate |
|---|---|---|---|
| Sparkov (Kaggle) | Real-world synthetic | ~48,000 | ~14% |
| Custom Synthetic | Edge-case fraud patterns | ~11,800 | ~30% |
| **Combined** | | **~59,800** | **~16.64%** |

The custom synthetic dataset was generated to specifically test edge-case fraud patterns: late-night transactions (00:00–05:00), high-value round-amount purchases, rapid-succession transactions from the same account, and transactions at rare merchant-location combinations.

## 3.3 Feature Engineering

All features are derived from the five mapped columns. Only `amount` and `time` are guaranteed to produce features; `vendor`, `location`, and `account_id` produce additional features that significantly improve detection quality.

| Feature | Source Column | Type |
|---|---|---|
| `amount` | amount | Raw (required) |
| `log_amount` | amount | Log-transformed |
| `amount_zscore` | amount | Standardised |
| `amount_deviation_from_mean` | amount | Absolute deviation |
| `hour_of_day` | time | Temporal |
| `day_of_week` | time | Temporal |
| `is_weekend` | time | Binary flag |
| `high_risk_time` | time | Binary (00:00–05:00) |
| `amount_vs_account_avg` | amount + account_id | Behavioural ratio |
| `account_tx_frequency` | time + account_id | Behavioural |
| `vendor_frequency` | vendor | Behavioural |
| `is_rare_vendor` | vendor | Binary flag (≤2 occurrences) |
| `location_frequency` | location | Behavioural |
| `is_rare_location` | location | Binary flag (≤2 occurrences) |

**Total: 14 features** when all columns are mapped.

## 3.4 Model Design

**Isolation Forest** was selected for:

- **Unsupervised learning**: No fraud labels required for training.
- **Efficiency**: O(n log n) time complexity; scales to 60k+ rows in seconds.
- **SHAP compatibility**: Built on ExtraTreeRegressor; TreeSHAP applies directly.
- **Contamination parameter**: Allows auditors to control detection sensitivity via the UI slider.

Configuration:
- `n_estimators = 100` (per Liu et al. 2008 recommendation)
- `max_samples = 'auto'` (min(256, n_samples))
- `contamination = 'auto'` for offline training; user-configurable for inference
- `random_state = 42` for reproducibility

## 3.5 Training Procedure

Training follows an **offline pre-training paradigm**:

1. `train_offline.py` trains the Isolation Forest on historical data (`combined_train.csv`)
2. The fitted model and scaler are serialised to `models/model.pkl` and `models/scaler.pkl` via `joblib`
3. A `models/metadata.json` records the exact 14 feature names and training sample count
4. At inference time, `run_pretrained_inference()` loads these artefacts and **pads missing features to zero** if a user's dataset provides fewer than 14 columns — maintaining compatibility without retraining

This design means the system can be deployed once and used repeatedly without any retraining overhead per audit session.

## 3.6 Explainability — Three-Layer Architecture

Every flagged transaction receives three complementary explanations:

| Layer | Method | Audience |
|---|---|---|
| **SHAP waterfall** | TreeExplainer on Isolation Forest | Data scientists / technical auditors |
| **Rule-based flags** | Threshold comparisons on raw values | All auditors |
| **Natural language** | Combined generator (SHAP + rules) | Non-technical auditors / regulators |

The SHAP background dataset uses 100 randomly sampled training rows (not the full training set) to prevent computational hangs while maintaining stable value estimates.

**Honesty requirement**: Every explanation includes the mandatory disclaimer — *"Isolation Forest is unsupervised. Anomaly ≠ confirmed fraud. Human review required."*

## 3.7 Evaluation Metrics

Due to severe class imbalance (fraud rates of 0.1%–30%), accuracy is explicitly **not used**. Evaluation relies on:

- **Precision**: Fraction of flagged transactions that are genuine anomalies
- **Recall**: Fraction of genuine anomalies that were flagged
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
- **AUC-PR**: Area under the Precision-Recall curve (most informative for imbalanced data)

---

# CHAPTER 4: IMPLEMENTATION

## 4.1 Software Requirements

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.12+ |
| Data Processing | Pandas, NumPy | ≥2.0, ≥1.24 |
| ML Model | scikit-learn (IsolationForest) | ≥1.3 |
| Explainability | SHAP (TreeExplainer) | ≥0.45 |
| Dashboard | Streamlit | ≥1.30 |
| Visualisation | Matplotlib, Seaborn | ≥3.7, ≥0.12 |
| Excel Support | openpyxl | ≥3.1 |
| Model Serialisation | joblib | ≥1.3 |

## 4.2 Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | Dual-core x86_64 | Multi-core (8+) |
| RAM | 8 GB | 16 GB |
| Storage | 500 MB | 2 GB |
| OS | Linux / macOS / Windows | Linux |

## 4.3 File Structure

```
crt/
├── app.py                  ← Streamlit UI (5 screens, session state, plots)
├── engine.py               ← Core logic (14 functions, no Streamlit imports)
├── train_offline.py        ← Offline model training script
├── validate_dataset.py     ← Schema validation utility
├── requirements.txt        ← Pinned dependencies
├── models/
│   ├── model.pkl           ← Pre-trained IsolationForest
│   ├── scaler.pkl          ← Fitted StandardScaler
│   └── metadata.json       ← Feature names + training metadata
├── data/
│   ├── combined_train.csv  ← Training data (~47,800 rows)
│   └── combined_test.csv   ← Test data (~11,000 rows)
├── sample_data/
│   ├── sample_transactions.csv      ← 100-row demo dataset
│   └── synthetic_transactions.csv   ← Large synthetic dataset
├── scripts/
│   ├── eda_analysis.py              ← EDA figures 1–7
│   ├── evaluation_analysis.py       ← Evaluation figures 8–13
│   ├── generate_synthetic_data.py   ← Synthetic fraud data generator
│   ├── prepare_data.py              ← Data preparation pipeline
│   └── run_pipeline.py              ← End-to-end batch pipeline
└── tests/
    └── test_engine.py       ← 62 unit tests (all passing)
```

## 4.4 Key Implementation Details

### 4.4.1 Intelligent Time Parsing

The `parse_time_column()` function applies three strategies in order:
1. `pd.to_datetime()` — handles standard ISO, UK, US date formats
2. Numeric seconds-elapsed — handles Kaggle-style `Time` columns (seconds from observation start)
3. Unix timestamps — handles epoch-based timestamps

This ensures compatibility with diverse real-world data sources without manual format configuration.

### 4.4.2 Pre-Trained Inference with Feature Alignment

`run_pretrained_inference()` loads the 14-feature pre-trained model and aligns any uploaded dataset to the expected feature vector. Features present in the uploaded data are used as-is; missing features (e.g., if vendor column is not mapped) are padded with zeros. This allows the system to operate even if an organisation's dataset lacks certain columns, while clearly communicating to the user that fewer features reduce detection confidence.

### 4.4.3 Session State Architecture

Streamlit re-executes the entire script on every user interaction. The `init_session_state()` function initialises 20+ session state keys on first load, persisting the DataFrame, model outputs, SHAP values, and UI state across interactions without redundant recomputation.

---

# CHAPTER 5: RESULTS AND DISCUSSION

## 5.1 Experimental Setup

The offline model was trained on `combined_train.csv` (~47,800 transactions). Evaluation was performed on `combined_test.csv` (~11,000 transactions) with ground-truth fraud labels available from the Sparkov dataset.

The Isolation Forest was initialised with `contamination='auto'`, allowing it to determine its threshold from the distribution of path lengths rather than requiring a manually specified fraud rate.

## 5.2 Performance Analysis

**AUC-ROC: 0.8038** — indicating strong discriminative capability between normal and anomalous transactions.

| Contamination | Precision | Recall | F1 Score | Flagged |
|---|---|---|---|---|
| 0.05 | **0.7576** | 0.2301 | 0.3523 | Low |
| 0.10 | 0.6521 | 0.4102 | 0.5031 | Medium |
| **0.15 (optimal F1)** | **0.5860** | **0.5201** | **0.5511** | Medium-High |
| 0.20 | 0.4912 | 0.6034 | 0.5415 | High |

**Key finding**: At contamination = 0.05, precision reaches 0.7576 — highly effective for automated flagging environments where false positives impose significant review cost. At contamination = 0.15, the system maximises F1, balancing precision and recall for general-purpose auditing.

**Precision-Recall AUC: 0.5363** — reflects the inherent difficulty of the task (highly imbalanced classes) but demonstrates meaningful improvement over a random baseline (PR-AUC = fraud rate ≈ 0.17).

## 5.3 Discussion

**Global SHAP analysis** (Figure 11) identified the following as the top anomaly drivers, ordered by mean |SHAP value|:

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `location_frequency` | Transactions at rare locations are highly suspicious |
| 2 | `account_tx_frequency` | Accounts with unusual transaction rates stand out |
| 3 | `amount_vs_account_avg` | Spending far above an account's norm is a strong signal |
| 4 | `amount_zscore` | Extreme amounts relative to the dataset drive flags |
| 5 | `is_rare_vendor` | Transactions at rarely-seen merchants are flagged |

This validates the project's hypothesis that **contextual, behavioural features outperform raw metric analysis** (like base `amount` alone) when identifying sophisticated evasion behaviour. The relatively low importance of raw temporal indicators (`hour_of_day`, `is_weekend`) suggests that sophisticated fraudsters deliberately schedule anomalous transactions during standard business hours.

The **three-layer explanation system** proved effective in practice:
- SHAP waterfalls provided data scientists with quantitative feature contributions
- Rule-based flags (e.g., "Amount is 5.2× the average") gave auditors immediately actionable evidence
- Natural language paragraphs combined both into a single, regulatorily-compliant narrative

---

# CHAPTER 6: APPLICATIONS AND SOCIETAL IMPACT

## 6.1 Advantages

- **Schema-Adaptive**: Accepts any CSV/Excel file — no fixed column names required.
- **Explainable by Design**: Every flagged transaction receives three complementary explanations — no black-box outputs.
- **Unsupervised**: Requires no labelled fraud data to operate.
- **Audit-Grade Output**: Exported CSV reports include metadata headers (model parameters, metrics, disclaimers) suitable for regulatory submission.
- **Instant Inference**: Pre-trained model loads in milliseconds; inference on 10,000 rows completes in under 1 second.
- **Transparent Limitations**: The system proactively discloses all limitations (unsupervised model, concept drift, feature alignment fallback).

## 6.2 Applications

| Domain | Use Case |
|---|---|
| **Corporate Finance** | Internal audit teams monitoring expense claims and payment runs |
| **Banking** | Transaction monitoring for unusual account behaviour |
| **Regulatory** | Tax agencies flagging corporate expense misuse for investigation |
| **Healthcare Finance** | Detecting anomalous insurance billing |
| **Public Sector** | Government procurement and grant expenditure monitoring |

## 6.3 Alignment with SDGs

This project aligns with:

- **SDG 16** (Peace, Justice and Strong Institutions): Promotes transparent, accountable financial operations and curbs illicit financial workflows through auditable, explainable AI.
- **SDG 17** (Partnerships for the Goals): Provides a reusable, open-source detection tool accessible to small organisations without dedicated data science teams.

## 6.4 Ethical Considerations

- The system explicitly disclaims that anomaly detection does not confirm fraud — human review is mandated before any adverse action.
- No personally identifiable information (PII) is stored or transmitted; all processing is local.
- Contamination rate is exposed to the user to prevent silent misconfiguration.

---

# CHAPTER 7: CONCLUSION AND FUTURE SCOPE

## 7.1 Conclusion

The Explainable Anomaly Detection System successfully demonstrates that automated, high-accuracy fraud detection models do not require opacity. By coupling Isolation Forest with a three-layer explanation architecture (SHAP + rule-based + natural language), the system effectively bridges advanced statistical learning with the rigorous semantic reporting requirements of financial auditing. The schema-adaptive pipeline removes the most significant practical barrier to adoption — the requirement for fixed dataset schemas — making the system applicable across diverse organisational data environments.

The project achieved:
- AUC-ROC of **0.8038** on 11,000 held-out transactions
- Precision of **0.7576** at strict contamination (0.05) for low-false-positive auditing
- **Three-layer explanations** for every flagged transaction
- **62 unit tests** across all 14 engine functions, all passing
- A fully functional, locally-deployable Streamlit dashboard

## 7.2 Limitations

1. **Unsupervised boundary**: Isolation Forest learns what is *unusual*, not what is *fraudulent*. The SHAP-based explanations faithfully explain the model's decision, but the model itself has no concept of ground-truth fraud labels.
2. **Concept drift**: The pre-trained model is frozen. As fraud patterns evolve, periodic retraining via `train_offline.py` is required.
3. **Single-user, local**: No authentication, no multi-user support. Not designed for production deployment.
4. **Feature quality**: With all 14 features available, detection is strongest. Fewer mapped columns reduce detection confidence.

## 7.3 Future Scope

| Enhancement | Description |
|---|---|
| **Supervised ensemble** | Add XGBoost trained on labelled data as a parallel detector |
| **Streaming integration** | Kafka/Flink pipeline for real-time transaction scoring |
| **Graph-based detection** | GNN to detect multi-hop fraud rings across accounts |
| **LIME comparison** | Add LIME as an alternative explainability method for comparison |
| **React + FastAPI** | Decouple frontend and backend for multi-user production deployment |
| **Counterfactuals** | "What would need to change for this transaction to not be flagged?" |

---

# CHAPTER 8: REFERENCES

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *Eighth IEEE International Conference on Data Mining (ICDM 2008)*, 413–422. https://doi.org/10.1109/ICDM.2008.17
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NIPS 2017)*, 30. https://arxiv.org/abs/1705.07874
3. Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
4. Doan, A., Halevy, A., & Ives, Z. (2012). *Principles of Data Integration*. Morgan Kaufmann.
5. Kaggle / Kartik2112. (2020). Fraud Detection Dataset (Sparkov). Retrieved from https://www.kaggle.com/datasets/kartik2112/fraud-detection
6. ULB Machine Learning Group. (2018). Credit Card Fraud Detection Dataset. Retrieved from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
7. Python Software Foundation. (2024). scikit-learn: Machine Learning in Python. https://scikit-learn.org
8. SHAP Library Documentation. (2024). TreeExplainer. https://shap.readthedocs.io
9. International Auditing and Assurance Standards Board (IAASB). (2009). *ISA 315 — Identifying and Assessing the Risks of Material Misstatement*. IFAC.
10. European Parliament. (2016). *General Data Protection Regulation (GDPR), Article 22 — Automated individual decision-making*. Official Journal of the EU.
