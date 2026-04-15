# 🌟 North Star Document
## Explainable Anomaly Detection for Financial Audits

> **Purpose**: This document is the single source of truth for the project's vision, scope, architecture, milestones, and success criteria. Every development decision should trace back to this document.

---

## 1. Vision Statement

> Build an **end-to-end machine learning system** that detects anomalous financial transactions and generates **human-readable, audit-grade explanations** for every flag — empowering auditors to make informed decisions with confidence, not blind trust.

### Core Principles

| Principle | Meaning |
|-----------|---------|
| 🔍 **Transparency First** | Every detection must be explainable in plain English — no black boxes |
| 🎯 **Audit-Ready Output** | Results are formatted for real-world auditor consumption, not just academic metrics |
| 🧩 **Modular Pipeline** | Each stage (data → preprocess → detect → explain → report) is independently testable and swappable |
| 📐 **Reproducibility** | Any run should be fully reproducible given the same input data and config |
| ⚖️ **Intellectual Honesty** | The system is transparent about what it can and cannot explain |

---

## 2. Problem Framing

### The Gap

| Current State | Desired State |
|---------------|---------------|
| Manual auditing is slow and scales poorly | Automated detection of suspicious transactions |
| ML fraud systems are black boxes | Every anomaly has a plain-English explanation |
| Auditors distrust opaque AI flags | Auditors receive evidence-backed, interpretable reports |
| No structured output format | Clean CSV + visual dashboard for audit trails |

### Target Users

| User | Need |
|------|------|
| **Financial Auditors** | Understand *why* a transaction is suspicious before escalating |
| **Compliance Officers** | Evidence trail for regulatory submissions |
| **Data Science Students** | Learn applied ML with real-world explainability |
| **Academic Evaluators** | Assess end-to-end ML pipeline design and presentation |

---

## 3. Critical Decisions (From Architect Review)

> These decisions were made after a comprehensive **Senior Architect Evaluation** and **Project Review** that raised 28 critical questions. All are binding for the Mini Project (MVP).

### 3.1 Explainability Approach — FINALIZED

> [!IMPORTANT]
> **The single most important architectural decision in this project.**

| Decision | Detail |
|----------|--------|
| **MVP Explainability** | **SHAP TreeExplainer** on Isolation Forest — model-faithful explanations |
| **Why SHAP works on IF** | Isolation Forest is built from `ExtraTreeRegressor` estimators. `TreeExplainer` traverses these tree structures to compute exact Shapley values for each feature's contribution to the anomaly score. This is officially supported in the SHAP library's test suite. Multiple 2024–2025 peer-reviewed papers validate this combination for financial fraud detection. |
| **Honesty requirement** | Every explanation in the UI and export **must include a disclaimer**: *"SHAP values are derived from the Isolation Forest model's internal tree structure. They are model-faithful, but Isolation Forest is unsupervised — a high anomaly score means statistically unusual, not confirmed fraud. Human review required."* |
| **Major Project enhancement** | Potential model switch to XGBoost (supervised) for even richer SHAP analysis |

**Architecture note (from the architect review):**
> *Unlike z-score explanations (which explain data distribution), SHAP TreeExplainer explains the model's actual decision process. SHAP values sum to the difference between the transaction's anomaly score and the baseline expected value — this is verifiable and model-faithful.*

**Our approach:** SHAP TreeExplainer provides true model-level explanations. However, we acknowledge that Isolation Forest is unsupervised (trained without fraud labels), so a high anomaly score means "statistically unusual" — not "confirmed fraud". This distinction is documented in the UI, exports, and project report.

---

### 3.2 Tech Stack — RESOLVED

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Dashboard** | **Streamlit** (NOT React+FastAPI) | Single-stack Python, no HTML/JS needed, fastest path to MVP |
| **File formats** | **CSV only** (NOT Excel, NOT Parquet) | True MVP simplicity, defer others |
| **Dataset** | **Kaggle Credit Card Fraud** (labeled, 284,807 rows) | Has ground truth labels → enables honest evaluation with Precision/Recall/F1 |
| **Scope** | **Single-user, local, offline tool** | Explicitly stated — no auth, no multi-user, no internet |

---

### 3.3 Evaluation — REVISED

| Decision | Detail |
|----------|--------|
| **Must have labeled data** | Using Kaggle CC Fraud dataset (0.172% fraud rate) for honest evaluation |
| **No accuracy metric** | Accuracy is misleading with 99.83% class imbalance — use **Precision, Recall, F1, AUC-ROC only** |
| **Baseline comparison required** | Must compare Isolation Forest against a rule-based baseline (e.g., "flag top N by amount") |
| **Contamination transparency** | The `contamination` parameter must be exposed in the UI with a plain-English explanation. Users must understand this is a **preset**, not a discovered rate. |
| **Performance benchmark** | Actual detection time must be measured and reported — not assumed to meet <5s target |

---

### 3.4 Mandatory Disclaimers & Transparency

Every output must include:

1. **Scope banner**: "This is a local, single-user, offline tool. CSV files only."
2. **Explanation disclaimer**: "SHAP values are model-faithful (derived from Isolation Forest's tree structure), but anomaly ≠ confirmed fraud. Human review required."
3. **Contamination notice**: "Anomaly count is controlled by the contamination parameter, not discovered by the model."
4. **Export metadata**: Run timestamp, model version, dataset name, explanation method (SHAP TreeExplainer), disclaimer.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Kaggle Credit Card Fraud Dataset (CSV, 284,807 rows)        │   │
│  │ Labeled: Class=0 (normal), Class=1 (fraud, 0.172%)          │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                               │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Missing Value │  │ Feature      │  │ Feature Scaling           │  │
│  │ Imputation    │  │ Engineering  │  │ (StandardScaler)          │  │
│  │               │  │ (Hour from   │  │ ⚠️ Fit on TRAIN only     │  │
│  │               │  │  Time col)   │  │ (no data leakage)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│                                                                      │
│  Train/Test Split: 80/20, stratified by Class                        │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DETECTION LAYER                                   │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Isolation Forest                            │  │
│  │  • contamination = ~0.00172 (matches actual fraud rate)       │  │
│  │  • n_estimators = 100                                         │  │
│  │  • random_state = 42                                          │  │
│  │  • Output: anomaly_label (-1/1) + anomaly_score               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  BASELINE: Rule-based (flag top-N by Amount)                  │  │
│  │  Purpose: Prove ML adds value over simple rules               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│  │ Major Project: XGBoost (supervised) + LOF + DBSCAN            │  │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   EXPLAINABILITY LAYER                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ SHAP TreeExplainer (model-faithful)                          │   │
│  │ • 100 background samples via shap.sample()                   │   │
│  │ • Computes Shapley values per feature per transaction        │   │
│  │ • Waterfall plots (per-transaction) + bar charts (global)    │   │
│  │ • Top-N feature contributions with direction indicators      │   │
│  │                                                              │   │
│  │ ⚠️ DISCLAIMER: SHAP values are model-faithful but IF is     │   │
│  │    unsupervised — anomaly ≠ confirmed fraud.                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│  │ Major Project: XGBoost (supervised) for richer SHAP analysis │  │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                     │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Streamlit     │  │ CSV Report   │  │ Visual Charts            │  │
│  │ Dashboard     │  │ (with meta-  │  │ (Matplotlib/Seaborn)     │  │
│  │ (6 screens)   │  │  data header │  │                          │  │
│  │               │  │  + disclaimer│  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Streamlit Dashboard — 6 Screens

| # | Screen | What it does |
|---|--------|-------------|
| 1 | **Data Upload** | Drag-drop CSV, preview data, validate format, file size/shape metrics |
| 2 | **EDA Dashboard** | Class distribution, amount histograms (fraud vs normal), hour-of-day patterns, missing value check |
| 3 | **Detection** | Contamination slider with explanation, "Run Detection" button, detection time benchmark |
| 4 | **Results Overview** | Metrics table (IF vs baseline: Precision/Recall/F1), anomaly distribution chart |
| 5 | **Transaction Explorer** | Flagged transactions table, sortable by anomaly score, CSV export |
| 6 | **Explanation Panel** | Select a transaction → SHAP waterfall plot + top 5 features + disclaimer |

**Plus**: CSV export with metadata header, error handling for bad inputs.

---

## 6. Module Breakdown

### Module 1: Streamlit App (Single-File MVP)
| Aspect | Detail |
|--------|--------|
| **File** | `app.py` |
| **Responsibility** | Full Streamlit dashboard — all 6 screens in one file |
| **Sections** | Upload → EDA → Detection → Results → Explanation → Export |

### Module 2: SHAP Explainability Engine (Functions within app.py)
| Aspect | Detail |
|--------|--------|
| **Functions** | `compute_shap()`, `make_waterfall()`, `top_features()`, `make_global_importance()` |
| **Method** | SHAP TreeExplainer on Isolation Forest (100 background samples) |
| **Output** | Per-transaction waterfall plots, top-N feature contributions, global importance bar chart |
| **Disclaimer** | Always included: "SHAP values are model-faithful but IF is unsupervised — anomaly ≠ fraud" |

---

## 7. 6-Sprint Execution Roadmap

> From the approved **Agile Sprint Plan**. Each sprint ends with a runnable prototype.

### Sprint 1: Foundation 🏗️ *(App Shell + Data Loading)*
| Task | Status |
|------|--------|
| Install Python 3.12+, create project, install dependencies | ⬜ |
| Download Kaggle CC Fraud dataset (284,807 rows) | ⬜ |
| Build `app.py` v0.1: file upload, data preview, scope banner | ⬜ |
| Restrict uploads to CSV only | ⬜ |

**Exit**: `streamlit run app.py` opens, loads CSV, shows data preview.

---

### Sprint 2: Understand Your Data 📊 *(EDA Dashboard)*
| Task | Status |
|------|--------|
| Class distribution chart + imbalance warning | ⬜ |
| Amount distribution: fraud vs normal | ⬜ |
| Temporal feature: Hour-of-day from Time column | ⬜ |
| Data quality check (missing values) | ⬜ |
| Feature statistics table | ⬜ |

**Exit**: App shows EDA with class imbalance warning and charts.

---

### Sprint 3: Detect Anomalies 🎯 *(Model + Baseline + Metrics)*
| Task | Status |
|------|--------|
| Train/test split (80/20, stratified) | ⬜ |
| StandardScaler fit on train only (no data leakage) | ⬜ |
| Isolation Forest training with exposed contamination slider | ⬜ |
| Rule-based baseline (flag top-N by Amount) | ⬜ |
| Metrics: Precision, Recall, F1 for both IF and baseline | ⬜ |
| Detection time benchmark | ⬜ |
| Flagged transactions table | ⬜ |

**Exit**: Side-by-side IF vs baseline metrics visible. Detection time reported.

---

### Sprint 4: Explain WHY 🔍 *(Core Feature — SHAP TreeExplainer)*
| Task | Status |
|------|--------|
| `compute_shap()` with TreeExplainer + 100 background samples | ⬜ |
| SHAP waterfall plot per transaction | ⬜ |
| Top 5 contributing features with direction indicators | ⬜ |
| Global feature importance bar chart (mean |SHAP|) | ⬜ |
| **Red disclaimer box**: unsupervised model caveat | ⬜ |

**Exit**: Click any flagged transaction → see SHAP waterfall + top features + disclaimer.

---

### Sprint 5: Full UI + Export 📦 *(Feature Complete)*
| Task | Status |
|------|--------|
| CSV export with metadata header (timestamp, model params, disclaimer) | ⬜ |
| Anomaly distribution donut chart | ⬜ |
| Contamination notice in results ("preset, not discovered") | ⬜ |
| Full end-to-end flow polish | ⬜ |

**Exit**: Upload → EDA → Detect → Explain → Export works seamlessly.

---

### Sprint 6: Test, Benchmark, Report 📝 *(Production-Ready)*
| Task | Status |
|------|--------|
| Edge case testing (empty CSV, missing columns, .txt rejection) | ⬜ |
| Error handling wrappers (try/except with user-friendly messages) | ⬜ |
| Project report with 7 sections (see Section 10 below) | ⬜ |
| Limitations section (z-score gap, contamination, concept drift, scope, fairness, legal) | ⬜ |
| Performance benchmark documented | ⬜ |

**Exit**: App handles bad inputs gracefully. Report written. Ready for evaluation.

---

## 8. Evaluation Framework (Revised)

### 8.1 ML Model Metrics (Against Labeled Ground Truth)

| Metric | How to Measure |
|--------|----------------|
| **Precision** | TP / (TP + FP) — of all flagged, how many are real fraud |
| **Recall** | TP / (TP + FN) — of all real fraud, how many did we catch |
| **F1 Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Full TP/FP/TN/FN breakdown |

> [!IMPORTANT]
> We do NOT use accuracy. A model predicting "not fraud" for everything gets 99.83% accuracy on this dataset. Accuracy is a misleading metric with extreme class imbalance.

### 8.2 Baseline Comparison (Mandatory)

| Method | What It Does |
|--------|-------------|
| **Isolation Forest** | ML-based multi-dimensional anomaly detection |
| **Rule-Based Baseline** | Flag top-N transactions by Amount |

Both are evaluated with the same metrics. If IF doesn't beat the baseline, the ML approach must be justified through multi-dimensional pattern detection arguments.

### 8.3 Explainability Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Explanation Coverage** | 100% of anomalies have explanations | Count anomalies with non-empty explanation |
| **Disclaimer Presence** | 100% of outputs include disclaimer | Automated check |
| **Explanation Readability** | Plain English, no jargon | Human review |

### 8.4 System Metrics

| Metric | How to Measure |
|--------|----------------|
| **Detection Time** | `time.time()` around model train+predict |
| **Edge Case Handling** | 5 test scenarios all handled without crashes |

---

## 9. Risk Matrix (Updated from Architect Review)

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **SHAP values explain model, not ground truth** | 🟡 Medium | 🟡 Medium | Mandatory disclaimer: IF is unsupervised, anomaly ≠ fraud. Human review required. |
| **SHAP computation time on large datasets** | 🟡 Medium | 🟡 Medium | Using 100 background samples (not full train set) to keep SHAP under 60s. |
| **Contamination parameter misleads users** | 🟡 Medium | 🟡 Medium | Exposed in UI with plain-English tooltip + "preset, not discovered" notice |
| **Isolation Forest underfits on complex patterns** | 🟡 Medium | 🟡 Medium | Baseline comparison validates ML adds value |
| **< 5s detection SLA violated on 284k rows** | 🟡 Medium | 🟡 Medium | Actual benchmark measured and reported honestly |
| **Model trained without temporal features** | 🟡 Medium | 🟢 Low | Hour-of-day feature engineered from Time column |
| **Concept drift on frozen model** | 🟢 Low (MVP) | 🔴 High (prod) | Documented as limitation. Retraining pipeline is Major Project scope. |
| **Adversarial robustness** | 🟢 Low (MVP) | 🟡 Medium | Acknowledged in limitations. Static model has zero adversarial defense. |
| **Fairness / bias in flagging** | 🟡 Medium | 🟡 Medium | Dataset uses PCA features (partial anonymization). Fairness audit is Major Project scope. |

---

## 10. Academic Deliverables / Report Structure

> From the 6-sprint plan. Report has **7 mandatory sections**.

| # | Section | Content | Status |
|---|---------|---------|--------|
| 1 | **Introduction** (~200 words) | Problem, users, why XAI matters. Cite GDPR Article 22. | ⬜ |
| 2 | **Dataset** (~150 words) | Kaggle CC Fraud: 284,807 rows, 0.172% fraud, 28 PCA features + Time + Amount. Pre-anonymized. | ⬜ |
| 3 | **Methodology** (~300 words) | Preprocessing (no data leakage), Hour feature, Isolation Forest, z-score explanations, evaluation with P/R/F1 | ⬜ |
| 4 | **Results** (~200 words) | Actual metrics (IF vs baseline), detection time, what worked/didn't | ⬜ |
| 5 | **Limitations** (~300 words) | ⚠️ **Strongest section** — z-score gap, contamination dependency, concept drift, single-user scope, no fairness audit, legal defensibility | ⬜ |
| 6 | **Future Work** (~150 words) | Major Project: XGBoost + SHAP, React + FastAPI, real-time, fairness, multi-user auth | ⬜ |
| 7 | **Conclusion** (~100 words) | Summary of what was built and learned | ⬜ |

---

## 11. Documented Limitations (Must Appear in Report)

These were identified by the architect review and must be explicitly stated:

1. **Unsupervised Model Limitation** — Isolation Forest is trained without fraud labels. SHAP explains the model's anomaly scoring, but anomaly ≠ confirmed fraud. The model detects statistical unusualness, not fraud intent.
2. **Contamination Parameter Dependency** — Anomaly count is preset, not discovered. The system cannot self-calibrate to varying fraud rates.
3. **Offline, Static Model** — No retraining pipeline. Fraud patterns evolve (concept drift), degrading performance over time.
4. **Single-User, Local Scope** — No authentication, no multi-user support. Not suitable for production.
5. **No Fairness Audit** — PCA-transformed features may still encode demographic proxies. Disparate impact not assessed.
6. **PCA Feature Opacity** — SHAP identifies which PCA component matters, not the original business attribute. V1–V28 are anonymized.
7. **Adversarial Robustness** — A fraudster who knows the system exists can craft transactions that appear normal to the Isolation Forest.

---

## 12. File Structure Target

```
Explainable-anomaly-detection-for-financial-audits/
│
├── README.md                    # Project documentation
├── NORTH_STAR.md                # This document (vision & roadmap)
├── LICENSE                      # MIT License
├── requirements.txt             # Pinned dependencies
├── .gitignore                   # Python/IDE ignores
│
├── app.py                       # Streamlit dashboard (all 6 screens)
│
├── data/
│   └── creditcard.csv           # Kaggle CC Fraud dataset (284,807 rows)
│
├── outputs/
│   └── anomaly_report_*.csv     # Exported reports with metadata
│
├── notebooks/                   # (Optional) Jupyter notebooks
│   └── eda.ipynb                # Interactive exploration
│
└── report/
    └── mini_project_report.md   # 7-section academic report
```

---

## 13. Tech Stack Rationale (Finalized)

| Technology | Why This Choice | Alternatives Ruled Out |
|------------|----------------|------------------------|
| **Python 3.12+** | Industry standard for ML | R (less deployment-friendly) |
| **Pandas** | De facto for tabular data | Polars (less ecosystem) |
| **NumPy** | Foundation for numerical computation | — |
| **Scikit-learn** | Best for classical ML, includes Isolation Forest | PyOD (less stable) |
| **Matplotlib + Seaborn** | Publication-quality static plots | Plotly (heavier) |
| **Streamlit** | Fastest path from Python → web app, no HTML needed | React+FastAPI (too complex for MVP) |
| **joblib** | Save/load trained models | pickle (less safe) |

---

## 14. Key References

| # | Reference | Relevance |
|---|-----------|-----------|
| 1 | Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation forest*. ICDM | Core algorithm |
| 2 | Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS | SHAP (Major Project) |
| 3 | Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly detection: A survey*. ACM Computing Surveys | Literature survey |
| 4 | Goldstein, M., & Uchida, S. (2016). *A comparative evaluation of unsupervised anomaly detection algorithms*. PLoS ONE | Model comparison |
| 5 | Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?"*. KDD | LIME (alternative) |
| 6 | EU AI Act (2024) + GDPR Article 22 | Legal basis for XAI in finance |

---

## 15. Mini Project vs Major Project Scope

| Feature | Mini Project (NOW) | Major Project (FUTURE) |
|---------|-------------------|----------------------|
| **Model** | Isolation Forest (unsupervised) | XGBoost + ensemble (supervised) |
| **Explainability** | SHAP TreeExplainer (model-faithful) | SHAP on XGBoost (supervised + richer) |
| **Dataset** | Kaggle CC Fraud (static) | Real-time streaming data |
| **Dashboard** | Streamlit (single-file) | React + FastAPI (decoupled) |
| **Scope** | Single-user, local, offline | Multi-user, authenticated, cloud |
| **File formats** | CSV only | CSV, Excel, Parquet |
| **Fairness** | Acknowledged as limitation | Full fairness audit |
| **Legal** | GDPR cited, not compliant | GDPR-compliant explanations |

---

> **Last Updated**: April 8, 2026
> **Source Conversations**: Architect Evaluation (352071b1), Research Analysis (b4f20ff9), Agile Sprint Plan (352071b1), Implementation Plans (23f1e579, 8e1fa5a7), New System Resume (b761b88b)
> **Next Action**: Sprint 2 — Dataset Download + Upload Verification (Sprint 1 complete ✅)
