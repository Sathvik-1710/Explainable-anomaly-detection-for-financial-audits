# ============================================================
# CRITICAL IMPORTS — ORDER MATTERS
# matplotlib.use('Agg') MUST come before importing pyplot
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import joblib
import time
import csv
import io
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

# ============================================================
# SECTION 1 — PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Financial Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SECTION 2 — SESSION STATE INITIALIZER
# Call this once at the top of main() before any tabs.
# Streamlit re-runs the script on every interaction.
# Session state persists data across re-runs.
# ============================================================
def init_session_state():
    defaults = {
        'df': None,
        'df_name': None,
        'X_train_scaled': None,
        'X_test_scaled': None,
        'X_test_original': None,
        'y_train': None,
        'y_test': None,
        'scaler': None,
        'feature_names': None,
        'model': None,
        'labels': None,
        'scores': None,
        'shap_values': None,
        'explainer': None,
        'train_time': None,
        'contamination': 0.00172,
        'metrics_if': None,
        'metrics_baseline': None,
        'selected_tx_idx': None,
        'detection_done': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# SECTION 3 — DATA VALIDATION
# ============================================================
def validate_dataframe(df):
    """
    Validate that the uploaded dataframe has required columns and
    is non-empty. Returns (is_valid: bool, errors: list of str).
    """
    errors = []
    required = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount', 'Class']
    missing = [c for c in required if c not in df.columns]

    if len(df) == 0:
        errors.append("Dataset is empty — upload a file with transactions.")
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if 'Class' in df.columns and df['Class'].nunique() < 2:
        errors.append(
            "Only one class found in 'Class' column. "
            "Evaluation metrics will be undefined."
        )
    return len(errors) == 0, errors


# ============================================================
# SECTION 4 — FEATURE ENGINEERING & PREPROCESSING
# The single most important rule: fit StandardScaler ONLY on
# training data. NEVER on the full dataset or test set.
# ============================================================
def preprocess(df):
    """
    Full preprocessing pipeline with zero data leakage.

    Steps:
      1. Engineer 'Hour' from Time (seconds → hour-of-day)
      2. Drop 'Time' (replaced by Hour)
      3. Separate X and y
      4. Stratified 80/20 train-test split (stratify=y preserves fraud ratio)
      5. fit_transform scaler on TRAIN only
      6. transform (no fit) on TEST only

    Returns
    -------
    X_train_scaled  : np.ndarray
    X_test_scaled   : np.ndarray
    X_test_original : pd.DataFrame  (unscaled test features, for baseline)
    y_train         : pd.Series
    y_test          : pd.Series
    scaler          : fitted StandardScaler
    feature_names   : list[str]
    """
    df = df.copy()

    # Feature engineering
    df['Hour'] = (df['Time'] % 86400) / 3600.0
    df = df.drop(columns=['Time'])

    X = df.drop(columns=['Class'])
    y = df['Class']
    feature_names = X.columns.tolist()

    # Stratified split — preserves 0.172% fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y          # REQUIRED for imbalanced data
    )

    X_test_original = X_test.copy()  # keep unscaled copy for baseline

    # Scale — fit ONLY on train, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_test_scaled  = scaler.transform(X_test)        # transform only — no fit

    return (
        X_train_scaled, X_test_scaled, X_test_original,
        y_train, y_test, scaler, feature_names
    )


# ============================================================
# SECTION 5 — ISOLATION FOREST TRAINING & PREDICTION
# ============================================================
def train_model(X_train_scaled, contamination):
    """
    Train Isolation Forest on scaled training data only.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies. This is a PRESET the user
        provides — the model does NOT discover the fraud rate itself.
        Use 0.00172 for Kaggle CC Fraud (actual fraud rate).

    Returns
    -------
    model      : fitted IsolationForest
    train_time : float (seconds)
    """
    model = IsolationForest(
        n_estimators=100,       # 100 trees — standard per original paper (Liu 2008)
        max_samples='auto',     # min(256, n_samples) — optimal per original paper
        contamination=contamination,
        random_state=42,        # reproducibility
        n_jobs=-1               # use all CPU cores
    )
    t0 = time.time()
    model.fit(X_train_scaled)
    train_time = time.time() - t0
    return model, train_time


def run_prediction(model, X_test_scaled):
    """
    Predict anomalies on test set.

    Returns
    -------
    labels : np.ndarray  — -1 = anomaly, +1 = normal
    scores : np.ndarray  — raw decision scores (lower = more anomalous)
    """
    labels = model.predict(X_test_scaled)
    scores = model.decision_function(X_test_scaled)
    return labels, scores


# ============================================================
# SECTION 6 — BASELINE MODEL
# A rule-based baseline that flags top-N transactions by Amount.
# Required to prove ML adds value over a simple heuristic.
# ============================================================
def run_baseline(X_test_original, y_test, contamination):
    """
    Rule-based baseline: flag the top-N transactions by Amount.
    N is set to match the contamination rate for fair comparison.

    Returns dict with precision, recall, f1, n_flagged.
    """
    n_flag = max(1, int(len(y_test) * contamination))
    amounts = X_test_original['Amount'].values
    threshold = np.sort(amounts)[-n_flag]
    y_pred_base = (amounts >= threshold).astype(int)
    y_true = np.array(y_test)

    return {
        'method': 'Rule-Based (Top-N by Amount)',
        'precision': float(precision_score(y_true, y_pred_base, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred_base, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred_base, zero_division=0)),
        'n_flagged': int(y_pred_base.sum()),
    }


# ============================================================
# SECTION 7 — EVALUATION METRICS
# NEVER use accuracy — 99.83% class imbalance makes it useless.
# A model predicting "never fraud" scores 99.83% accuracy.
# Use Precision, Recall, F1, AUC-ROC, AUC-PR.
# ============================================================
def evaluate_model(labels, y_test):
    """
    Evaluate Isolation Forest predictions against ground truth.

    Isolation Forest outputs:  -1 = anomaly, +1 = normal
    Ground truth (Class):       1 = fraud,   0 = normal

    Conversion: (labels == -1) gives 1 where IF flagged anomaly.
    """
    y_pred = (labels == -1).astype(int)
    y_true = np.array(y_test)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'method': 'Isolation Forest (ML)',
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc':   float(roc_auc_score(y_true, y_pred)),
        'auc_pr':    float(average_precision_score(y_true, y_pred)),
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
        'n_flagged': int((labels == -1).sum()),
    }


# ============================================================
# SECTION 8 — SHAP EXPLAINABILITY ENGINE
#
# WHY shap.TreeExplainer WORKS on Isolation Forest:
# Isolation Forest is built from ExtraTreeRegressor estimators.
# TreeExplainer traverses these tree structures to compute exact
# Shapley values for each feature's contribution to the anomaly
# score. This is model-faithful — SHAP values reconstruct the
# exact model output (verifiable via shap_values.sum() + expected).
#
# SIGN CONVENTION:
# IF decision_function: lower = more anomalous
# SHAP values follow the same convention:
#   Negative SHAP → pushes score down → toward anomaly (RED bars)
#   Positive SHAP → pushes score up  → toward normal  (BLUE bars)
#
# BACKGROUND SAMPLES:
# Use shap.sample(X_train_scaled, 100) NOT the full training set.
# Full training set (227k rows) as background = 10+ minute hang.
# 100 samples is sufficient for stable SHAP estimates.
# ============================================================
def compute_shap(model, X_train_scaled, X_test_scaled, feature_names):
    """
    Compute SHAP values for all test samples using TreeExplainer.

    Returns
    -------
    explainer   : shap.TreeExplainer
    shap_values : np.ndarray shape (n_test_samples, n_features)
    """
    # 100 background samples — DO NOT use full X_train_scaled
    background = shap.sample(
        pd.DataFrame(X_train_scaled, columns=feature_names),
        100,
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation='interventional'
    )

    # Compute SHAP values for test set
    shap_values = explainer.shap_values(
        pd.DataFrame(X_test_scaled, columns=feature_names)
    )

    # TreeExplainer on IF may return a list [array] or array directly
    # Normalize to 2D array shape (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return explainer, shap_values


def make_waterfall(idx, shap_values, X_test_scaled, feature_names, explainer):
    """
    Generate a SHAP waterfall plot for a single transaction.

    The waterfall shows:
    - Base value: average model output over background dataset
    - Each bar: feature's signed contribution to anomaly score
    - Final value: this transaction's actual anomaly score
    Red bars = pushed toward anomaly, Blue bars = pushed toward normal.
    """
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(ev[0])

    sv = shap_values[idx]
    data_row = X_test_scaled[idx] if isinstance(X_test_scaled, np.ndarray) \
        else X_test_scaled.iloc[idx].values

    explanation = shap.Explanation(
        values=sv,
        base_values=ev,
        data=data_row,
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"SHAP Explanation — Transaction #{idx}", pad=20)
    plt.tight_layout()
    return fig


def top_features(idx, shap_values, feature_names, n=5):
    """
    Return top-N features by absolute SHAP value for one transaction.
    Used for text summary in the explanation panel.
    """
    sv = shap_values[idx]
    order = np.argsort(np.abs(sv))[::-1][:n]
    result = []
    for i in order:
        result.append({
            'feature': feature_names[i],
            'shap_value': round(float(sv[i]), 5),
            'direction': 'toward anomaly' if sv[i] < 0 else 'toward normal',
            'abs': round(float(abs(sv[i])), 5),
        })
    return result


def make_global_importance(shap_values, feature_names):
    """
    Bar chart of mean |SHAP| values across all test samples.
    This is the global explanation — which features matter most overall.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [feature_names[i] for i in order[::-1]],
        mean_abs[order[::-1]],
        color='#4C72B0'
    )
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Global Feature Importance\n(mean |SHAP value| across all test samples)')
    plt.tight_layout()
    return fig


# ============================================================
# SECTION 9 — EDA PLOTS
# ============================================================
def plot_class_dist(df):
    counts = df['Class'].value_counts().sort_index()
    labels_map = {0: 'Normal', 1: 'Fraud'}
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['#4CAF50', '#F44336']
    bars = ax.bar(
        [labels_map[i] for i in counts.index],
        counts.values,
        color=colors
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    fraud_pct = df['Class'].mean() * 100
    ax.set_title(f'Class Distribution\nFraud = {fraud_pct:.3f}% of all transactions')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig


def plot_amount_dist(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    normal = df[df['Class'] == 0]['Amount']
    fraud  = df[df['Class'] == 1]['Amount']
    ax1.hist(normal, bins=60, color='#4CAF50', alpha=0.8, log=True)
    ax1.set_title('Normal — Amount Distribution\n(log scale)')
    ax1.set_xlabel('Amount (€)')
    ax2.hist(fraud, bins=40, color='#F44336', alpha=0.8)
    ax2.set_title('Fraud — Amount Distribution')
    ax2.set_xlabel('Amount (€)')
    plt.tight_layout()
    return fig


def plot_hour_dist(df):
    df2 = df.copy()
    df2['Hour'] = (df2['Time'] % 86400) / 3600.0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(df2[df2['Class']==0]['Hour'], bins=24,
            alpha=0.5, label='Normal', color='#4CAF50', density=True)
    ax.hist(df2[df2['Class']==1]['Hour'], bins=24,
            alpha=0.8, label='Fraud',  color='#F44336', density=True)
    ax.set_xlabel('Hour of Day (0–24)')
    ax.set_ylabel('Density')
    ax.set_title('Transaction Volume by Hour of Day\n(Fraud vs Normal — density normalized)')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Top 10 features most correlated with Class."""
    corr = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
    top10 = corr.head(10).index.tolist()
    sub = df[top10 + ['Class']]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sub.corr(), annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, ax=ax, linewidths=0.5
    )
    ax.set_title('Correlation Matrix — Top 10 Features vs Class')
    plt.tight_layout()
    return fig


# ============================================================
# SECTION 10 — CSV EXPORT WITH METADATA HEADER
# Every exported report must document: when, what model,
# what parameters, what explanation method, and key disclaimers.
# ============================================================
def build_export_csv(
    labels, scores, shap_values, feature_names,
    y_test, X_test_original, contamination, metrics_if
):
    """
    Build audit report CSV as a bytes buffer (for st.download_button).

    Structure:
      Lines 1–15   : metadata comments starting with #
      Line 16      : blank separator
      Line 17+     : data rows with columns:
                     test_index, anomaly_score, actual_label, correctly_flagged,
                     top1_feature, top1_shap, top2_feature, top2_shap,
                     top3_feature, top3_shap, amount
    """
    anomaly_mask = labels == -1
    anomaly_test_indices = np.where(anomaly_mask)[0]

    buf = io.StringIO()

    # Metadata header
    meta = [
        ['# === AUDIT REPORT METADATA ==='],
        ['# Generated',         datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['# Model',             'IsolationForest (scikit-learn)'],
        ['# n_estimators',      100],
        ['# contamination',     contamination],
        ['# random_state',      42],
        ['# Explainability',    'SHAP TreeExplainer (model-faithful)'],
        ['# Precision',         round(metrics_if.get('precision', 0), 4)],
        ['# Recall',            round(metrics_if.get('recall',    0), 4)],
        ['# F1_Score',          round(metrics_if.get('f1',        0), 4)],
        ['# AUC_ROC',           round(metrics_if.get('auc_roc',   0), 4)],
        ['# Disclaimer',
         'SHAP values explain model decisions. Anomaly score != confirmed fraud. '
         'Human review required.'],
        ['# Scope',             'Offline single-user tool. CSV input only.'],
        ['# ================================'],
        [],  # blank line
    ]

    writer = csv.writer(buf)
    for row in meta:
        writer.writerow(row)

    # Data header
    writer.writerow([
        'test_index', 'anomaly_score', 'actual_label', 'correctly_flagged',
        'amount',
        'top1_feature', 'top1_shap',
        'top2_feature', 'top2_shap',
        'top3_feature', 'top3_shap',
    ])

    y_arr = np.array(y_test)
    amount_arr = X_test_original['Amount'].values \
        if 'Amount' in X_test_original.columns else [None]*len(y_arr)

    for i in anomaly_test_indices:
        sv = shap_values[i]
        top3_idx = np.argsort(np.abs(sv))[::-1][:3]

        row = [
            int(i),
            round(float(scores[i]), 6),
            int(y_arr[i]),
            bool(y_arr[i] == 1),
            round(float(amount_arr[i]), 2) if amount_arr[i] is not None else '',
        ]
        for rank in top3_idx:
            row += [feature_names[rank], round(float(sv[rank]), 6)]

        writer.writerow(row)

    return buf.getvalue().encode('utf-8')


# ============================================================
# SECTION 11 — THE 6 STREAMLIT SCREENS
# ============================================================

# --- Screen 1: Data Upload ---
def screen_upload():
    st.title("🔍 Explainable Anomaly Detection for Financial Audits")

    st.info(
        "**Scope:** Local · Offline · Single-user · CSV only. "
        "Built for transparent financial auditing using "
        "Isolation Forest + SHAP TreeExplainer."
    )

    uploaded = st.file_uploader(
        "Upload financial transaction CSV",
        type=['csv'],
        help="Required columns: V1–V28, Time, Amount, Class"
    )

    if uploaded is not None:
        if not uploaded.name.endswith('.csv'):
            st.error("Only .csv files are accepted.")
            return

        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        ok, errors = validate_dataframe(df)
        if not ok:
            for err in errors:
                st.error(err)
            return

        st.session_state['df'] = df
        st.session_state['df_name'] = uploaded.name

        fraud_n   = int(df['Class'].sum())
        fraud_pct = df['Class'].mean() * 100

        st.success(f"✅ Loaded **{uploaded.name}** — {len(df):,} transactions")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{len(df):,}")
        c2.metric("Fraud Cases",        f"{fraud_n:,}")
        c3.metric("Normal Cases",       f"{len(df)-fraud_n:,}")
        c4.metric("Fraud Rate",         f"{fraud_pct:.3f}%")

        st.subheader("First 10 rows")
        st.dataframe(df.head(10), use_container_width=True)

        missing = int(df.isnull().sum().sum())
        if missing > 0:
            st.warning(
                f"⚠️ {missing} missing values detected. "
                "StandardScaler will handle them during preprocessing."
            )
        else:
            st.success("No missing values detected.")


# --- Screen 2: Detection ---
def screen_detection():
    if st.session_state['df'] is None:
        st.warning("Upload a dataset first (Tab 1).")
        return

    st.header("🎯 Anomaly Detection")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Parameters")

    contamination = st.sidebar.slider(
        "Contamination (Expected Fraud Rate)",
        min_value=0.001, max_value=0.05,
        value=0.00172, step=0.0005,
        format="%.5f",
        help=(
            "This tells the model what fraction of transactions to flag. "
            "It is a PRESET — the model does NOT discover the fraud rate. "
            "Use 0.00172 for this dataset (actual fraud rate)."
        )
    )
    st.session_state['contamination'] = contamination

    n_to_flag = int(len(st.session_state['df']) * 0.2 * contamination)
    st.sidebar.caption(
        f"At this rate, ~{n_to_flag:,} transactions will be flagged in the test set."
    )
    st.sidebar.warning(
        "⚠️ Contamination Notice: The anomaly count is controlled "
        "by this slider — not discovered by the model."
    )

    st.markdown("### Pipeline Overview")
    st.markdown("""
    **What happens when you click Run Detection:**
    1. Engineers `Hour` feature from `Time` column
    2. Stratified 80/20 train-test split (preserves fraud ratio)
    3. Fits `StandardScaler` on training data only (no data leakage)
    4. Trains `IsolationForest` on scaled training data
    5. Predicts anomaly labels on test data
    6. Computes SHAP values via `TreeExplainer` (explains model decisions)
    7. Evaluates against ground truth (Precision / Recall / F1)
    """)

    if st.button("▶ Run Detection", type="primary", use_container_width=True):

        progress = st.progress(0, text="Starting...")

        # Step 1: Preprocess
        progress.progress(10, text="Preprocessing — splitting data, scaling features...")
        try:
            (X_train_scaled, X_test_scaled, X_test_original,
             y_train, y_test, scaler, feature_names) = preprocess(
                st.session_state['df']
            )
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            return

        st.session_state.update({
            'X_train_scaled': X_train_scaled,
            'X_test_scaled':  X_test_scaled,
            'X_test_original': X_test_original,
            'y_train': y_train,
            'y_test':  y_test,
            'scaler':  scaler,
            'feature_names': feature_names,
        })

        # Step 2: Train
        progress.progress(30, text="Training Isolation Forest...")
        try:
            model, train_time = train_model(X_train_scaled, contamination)
            labels, scores    = run_prediction(model, X_test_scaled)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            return

        st.session_state.update({
            'model': model,
            'labels': labels,
            'scores': scores,
            'train_time': train_time,
        })

        # Step 3: Evaluate
        progress.progress(50, text="Evaluating model...")
        metrics_if       = evaluate_model(labels, y_test)
        metrics_baseline = run_baseline(X_test_original, y_test, contamination)
        metrics_if['precision'] = metrics_if.pop('precision', metrics_if.get('precision'))
        # Normalize key names for consistency
        metrics_if_out = {
            'precision': metrics_if['precision'],
            'recall':    metrics_if['recall'],
            'f1':        metrics_if['f1'],
            'auc_roc':   metrics_if['auc_roc'],
            'auc_pr':    metrics_if['auc_pr'],
            'tp': metrics_if['tp'], 'fp': metrics_if['fp'],
            'tn': metrics_if['tn'], 'fn': metrics_if['fn'],
            'n_flagged': metrics_if['n_flagged'],
        }

        st.session_state.update({
            'metrics_if':       metrics_if_out,
            'metrics_baseline': metrics_baseline,
        })

        # Step 4: SHAP
        progress.progress(60, text="Computing SHAP values (30–60 sec)...")
        try:
            explainer, shap_values = compute_shap(
                model, X_train_scaled, X_test_scaled, feature_names
            )
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            st.info(
                "Tip: If this fails, check that shap>=0.45 is installed. "
                "Run: pip install --upgrade shap"
            )
            return

        st.session_state.update({
            'explainer':   explainer,
            'shap_values': shap_values,
            'detection_done': True,
        })

        progress.progress(100, text="Done!")
        st.success(
            f"✅ Detection complete in **{train_time:.2f}s**. "
            f"Flagged **{(labels == -1).sum():,}** anomalies "
            f"({(labels == -1).mean()*100:.3f}% of test set)."
        )

    elif st.session_state['detection_done']:
        st.info("Detection already run. See Results and Explanation tabs. "
                "Re-run to change contamination.")


# --- Screen 3: Results ---
def screen_results():
    if not st.session_state['detection_done']:
        st.warning("Run detection first (Tab 2).")
        return

    st.header("📈 Detection Results")

    m_if   = st.session_state['metrics_if']
    m_base = st.session_state['metrics_baseline']
    labels = st.session_state['labels']
    contamination = st.session_state['contamination']

    # ── Metrics comparison table ──
    st.subheader("Isolation Forest vs Rule-Based Baseline")
    st.caption(
        "The baseline flags the same number of transactions but uses only "
        "transaction Amount (highest = most suspicious). "
        "If IF beats the baseline on Recall, ML is adding value."
    )

    comp = pd.DataFrame([
        {
            'Method':    'Isolation Forest (ML)',
            'Precision': f"{m_if['precision']:.4f}",
            'Recall':    f"{m_if['recall']:.4f}",
            'F1':        f"{m_if['f1']:.4f}",
            'AUC-ROC':   f"{m_if['auc_roc']:.4f}",
            'AUC-PR':    f"{m_if['auc_pr']:.4f}",
            'Flagged':   f"{m_if['n_flagged']:,}",
        },
        {
            'Method':    'Rule-Based (Top-N Amount)',
            'Precision': f"{m_base['precision']:.4f}",
            'Recall':    f"{m_base['recall']:.4f}",
            'F1':        f"{m_base['f1']:.4f}",
            'AUC-ROC':   '—',
            'AUC-PR':    '—',
            'Flagged':   f"{m_base['n_flagged']:,}",
        },
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # ── Why not accuracy ──
    st.error(
        "**Why no Accuracy metric?** "
        "With 99.828% normal transactions, predicting 'never fraud' achieves "
        "99.828% accuracy. Accuracy is meaningless for severe class imbalance. "
        "Recall matters most — it measures how many real fraud cases we caught."
    )

    # ── Confusion matrix metrics ──
    st.subheader("Confusion Matrix Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positives",  m_if['tp'],
              help="Real fraud cases correctly flagged")
    c2.metric("False Positives", m_if['fp'],
              help="Normal transactions wrongly flagged")
    c3.metric("True Negatives",  m_if['tn'],
              help="Normal transactions correctly cleared")
    c4.metric("False Negatives", m_if['fn'],
              help="Real fraud cases missed by model")

    # ── Contamination notice ──
    st.warning(
        f"⚠️ Contamination Notice: The model was instructed to flag "
        f"{contamination*100:.3f}% of transactions (contamination={contamination}). "
        f"It flagged {m_if['n_flagged']:,} transactions. "
        "This count is a preset — not a discovered fraud rate."
    )

    # ── Detection time ──
    if st.session_state['train_time']:
        st.metric(
            "Detection Time",
            f"{st.session_state['train_time']:.2f}s",
            help="Time to train Isolation Forest on 80% of the dataset"
        )

    # ── Global SHAP importance ──
    st.subheader("Global Feature Importance (SHAP)")
    st.caption(
        "Mean absolute SHAP value across all test samples. "
        "Larger bar = feature contributed more to anomaly scoring overall."
    )
    fig = make_global_importance(
        st.session_state['shap_values'],
        st.session_state['feature_names']
    )
    st.pyplot(fig)
    plt.close()

    # ── Score distribution ──
    st.subheader("Anomaly Score Distribution")
    scores = st.session_state['scores']
    labels_arr = st.session_state['labels']
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.hist(scores[labels_arr ==  1], bins=50, alpha=0.6,
            label='Normal',  color='#4CAF50', density=True)
    ax.hist(scores[labels_arr == -1], bins=50, alpha=0.8,
            label='Anomaly', color='#F44336', density=True)
    ax.axvline(np.percentile(scores, 100*st.session_state['contamination']),
               color='black', linestyle='--', label='Decision threshold')
    ax.set_xlabel('Anomaly Score (lower = more anomalous)')
    ax.set_ylabel('Density')
    ax.set_title('Decision Score Distribution')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# --- Screen 4: Transaction Explorer ---
def screen_explorer():
    if not st.session_state['detection_done']:
        st.warning("Run detection first (Tab 2).")
        return

    st.header("🔎 Flagged Transactions Explorer")

    labels      = st.session_state['labels']
    scores      = st.session_state['scores']
    y_test      = np.array(st.session_state['y_test'])
    feature_names = st.session_state['feature_names']
    X_test_original = st.session_state['X_test_original']

    anomaly_idx = np.where(labels == -1)[0]

    if len(anomaly_idx) == 0:
        st.info("No anomalies flagged. Try increasing the contamination slider.")
        return

    # Build display table
    rows = []
    for i in anomaly_idx:
        rows.append({
            'Test Index':      int(i),
            'Anomaly Score':   round(float(scores[i]), 5),
            'Actual Label':    '🔴 FRAUD' if y_test[i] == 1 else '🟢 Normal',
            'Correctly Flagged': '✅ Yes' if y_test[i] == 1 else '❌ False Alarm',
            'Amount (€)':      round(float(X_test_original['Amount'].iloc[i]), 2)
                               if 'Amount' in X_test_original.columns else '—',
        })

    df_display = pd.DataFrame(rows).sort_values('Anomaly Score')
    st.caption(
        f"Showing **{len(df_display):,}** flagged transactions. "
        f"Click a row or use the selector below to see its SHAP explanation."
    )
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Transaction selector
    st.subheader("Select a Transaction to Explain")
    selected = st.selectbox(
        "Transaction",
        options=anomaly_idx.tolist(),
        format_func=lambda x: (
            f"Index #{x}  |  Score: {scores[x]:.5f}  |  "
            f"{'FRAUD' if y_test[x]==1 else 'Normal'}"
        )
    )
    if selected is not None:
        st.session_state['selected_tx_idx'] = int(selected)
        st.info(f"Selected Transaction #{selected}. Go to Tab 5 (Explanation) to see SHAP details.")

    # CSV Export
    st.subheader("Export Audit Report")
    if st.button("Generate CSV Report"):
        csv_bytes = build_export_csv(
            labels=labels,
            scores=scores,
            shap_values=st.session_state['shap_values'],
            feature_names=feature_names,
            y_test=st.session_state['y_test'],
            X_test_original=X_test_original,
            contamination=st.session_state['contamination'],
            metrics_if=st.session_state['metrics_if'],
        )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="⬇ Download anomaly_report.csv",
            data=csv_bytes,
            file_name=f"anomaly_report_{timestamp}.csv",
            mime='text/csv',
        )


# --- Screen 5: Explanation Panel ---
def screen_explanation():
    if not st.session_state['detection_done']:
        st.warning("Run detection first (Tab 2).")
        return

    if st.session_state['selected_tx_idx'] is None:
        st.warning(
            "No transaction selected. "
            "Go to Tab 4 (Transaction Explorer) and select one."
        )
        return

    idx         = st.session_state['selected_tx_idx']
    shap_values = st.session_state['shap_values']
    feature_names = st.session_state['feature_names']
    X_test_scaled = st.session_state['X_test_scaled']
    explainer   = st.session_state['explainer']
    y_test      = np.array(st.session_state['y_test'])
    scores      = st.session_state['scores']
    X_test_orig = st.session_state['X_test_original']

    actual_label = 'FRAUD' if y_test[idx] == 1 else 'Normal'
    label_color  = '🔴' if y_test[idx] == 1 else '🟢'

    st.header(f"💡 SHAP Explanation — Transaction #{idx}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Anomaly Score",   round(float(scores[idx]), 5),
              help="Lower = more anomalous")
    c2.metric("Actual Label",    f"{label_color} {actual_label}")
    c3.metric("Amount (€)",
              round(float(X_test_orig['Amount'].iloc[idx]), 2)
              if 'Amount' in X_test_orig.columns else '—')

    st.subheader("SHAP Waterfall Plot")
    st.caption(
        "Each bar shows how much that feature pushed the anomaly score "
        "up or down from the baseline. "
        "🔴 Red = pushed toward anomaly. 🔵 Blue = pushed toward normal."
    )

    try:
        fig = make_waterfall(idx, shap_values, X_test_scaled, feature_names, explainer)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Could not render waterfall plot: {e}")
        st.code(str(e))

    # Top contributing features — text
    st.subheader("Top 5 Contributing Features")
    top5 = top_features(idx, shap_values, feature_names, n=5)

    for rank, f in enumerate(top5, 1):
        icon = "🔴" if f['direction'] == 'toward anomaly' else "🔵"
        direction_label = "→ anomaly" if f['direction'] == 'toward anomaly' else "→ normal"
        st.markdown(
            f"**{rank}. {f['feature']}** — "
            f"SHAP = `{f['shap_value']}` {icon} {direction_label}"
        )

    # Mandatory disclaimer
    st.divider()
    st.error(
        "**SHAP Explanation Disclaimer**\n\n"
        "These SHAP values are derived directly from the Isolation Forest model's "
        "internal tree structure. They are model-faithful — SHAP values sum to the "
        "difference between this transaction's anomaly score and the baseline expected value.\n\n"
        "However: Isolation Forest is **unsupervised** and trained without fraud labels. "
        "A high anomaly score means this transaction is statistically unusual compared to "
        "training data — it does **not** confirm fraud. "
        "All flagged transactions require human review before any action is taken.\n\n"
        "PCA features (V1–V28) are anonymized. SHAP identifies which PCA component "
        "matters, not the original business attribute."
    )


# ============================================================
# SECTION 12 — MAIN APP ENTRY POINT
# ============================================================
def main():
    init_session_state()

    tabs = st.tabs([
        "📁 1. Upload",
        "🎯 2. Detection",
        "📈 3. Results",
        "🔎 4. Explorer",
        "💡 5. Explanation",
    ])

    with tabs[0]: screen_upload()
    with tabs[1]: screen_detection()
    with tabs[2]: screen_results()
    with tabs[3]: screen_explorer()
    with tabs[4]: screen_explanation()


if __name__ == "__main__":
    main()
