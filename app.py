# ============================================================
# app.py — Schema-Adaptive Anomaly Detection Dashboard
#
# Streamlit UI for the financial anomaly detection system.
# All computation logic lives in engine.py.
#
# 5 Screens:
#   1. Upload — accepts CSV/Excel, shows data preview
#   2. Column Mapping — user maps columns to semantic roles
#   3. Detection — runs feature engineering + Isolation Forest + SHAP
#   4. Results — metrics, global importance, score distribution
#   5. Explanation — per-transaction SHAP + rules + natural language
#
# Hard constraints:
#   - NEVER call plt.show() — use st.pyplot(fig) then plt.close()
#   - ALWAYS call matplotlib.use('Agg') before importing pyplot
#   - ALWAYS fit scaler on train only
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import shap
from datetime import datetime

# Import core logic from engine.py
from engine import (
    load_file, validate_mapping, engineer_features,
    preprocess, train_model, run_prediction, evaluate_model,
    compute_shap, get_top_features,
    generate_rule_explanations, generate_nl_explanation,
    build_export_csv
)


# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Financial Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    defaults = {
        'df': None,
        'df_name': None,
        'column_mapping': {},
        'mapping_confirmed': False,
        'feature_df': None,
        'feature_names': None,
        'stats': None,
        'original_df': None,
        'X_train_scaled': None,
        'X_test_scaled': None,
        'train_idx': None,
        'test_idx': None,
        'y_train': None,
        'y_test': None,
        'scaler': None,
        'has_labels': False,
        'model': None,
        'labels': None,
        'scores': None,
        'shap_values': None,
        'explainer': None,
        'train_time': None,
        'contamination': 0.05,
        'metrics': None,
        'selected_tx_idx': None,
        'detection_done': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# SCREEN 1 — FILE UPLOAD
# ============================================================
def screen_upload():
    st.title("🔍 Explainable Anomaly Detection for Financial Audits")

    st.info(
        "**Schema-Adaptive System** — Upload ANY financial transaction file. "
        "Supports CSV and Excel (.xlsx). "
        "You'll map your columns in the next step."
    )

    uploaded = st.file_uploader(
        "Upload transaction data",
        type=['csv', 'xlsx', 'xls'],
        help="Any CSV or Excel file with at least an amount column and a time/date column."
    )

    if uploaded is not None:
        if st.session_state.get('df_name') != uploaded.name:
            df, errors = load_file(uploaded)

            if errors:
                for err in errors:
                    st.error(err)
                return

            st.session_state['df'] = df
            st.session_state['df_name'] = uploaded.name
            # Reset downstream state on new upload
            for key in ['mapping_confirmed', 'detection_done', 'model',
                         'labels', 'scores', 'shap_values', 'metrics']:
                st.session_state[key] = False if isinstance(st.session_state.get(key), bool) else None

            st.success(f"✅ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")
        else:
            df = st.session_state['df']

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{len(df.columns)}")
        missing = int(df.isnull().sum().sum())
        col3.metric("Missing Values", f"{missing:,}")

        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Column Types")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.notna().sum().values,
            'Sample Value': [str(df[c].dropna().iloc[0]) if df[c].notna().any() else '—' for c in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

        if missing > 0:
            st.warning(
                f"⚠️ {missing:,} missing values detected. "
                "They will be handled during feature engineering."
            )

        st.success("✅ Data loaded. Map your columns below, then go to **Tab 2** to run the audit.")

    elif st.session_state['df'] is not None:
        df = st.session_state['df']
        st.success(
            f"✅ **{st.session_state['df_name']}** loaded — "
            f"{len(df):,} rows × {len(df.columns)} columns"
        )
        st.dataframe(df.head(5), use_container_width=True)


# ============================================================
# SCREEN 2 — COLUMN MAPPING
# ============================================================
def screen_column_mapping():
    if st.session_state['df'] is None:
        st.warning("⬆️ Upload a dataset first (Tab 1).")
        return

    df = st.session_state['df']
    columns = ['— None —'] + df.columns.tolist()

    st.header("🗺️ Column Mapping")
    st.markdown(
        "Tell the system what each column in your dataset represents. "
        "The system will engineer meaningful features based on your mapping."
    )

    st.info(
        "**Required:** You must map **Amount**, **Time/Date**, **Vendor**, **Location**, and **Account ID**. "
        "Label is optional (enables precision/recall metrics)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Required Columns")

        amount_col = st.selectbox(
            "💰 Amount Column",
            options=columns,
            index=_guess_column_index(columns, ['amount', 'amt', 'value', 'total', 'price']),
            help="The column containing transaction amounts (numeric)."
        )

        time_col = st.selectbox(
            "🕐 Time / Date Column",
            options=columns,
            index=_guess_column_index(columns, ['date', 'time', 'timestamp', 'datetime', 'created']),
            help="The column containing transaction timestamps or dates."
        )

        vendor_col = st.selectbox(
            "🏪 Vendor / Merchant",
            options=columns,
            index=_guess_column_index(columns, ['vendor', 'merchant', 'store', 'seller', 'payee']),
            help="Who received the payment."
        )

        location_col = st.selectbox(
            "📍 Location",
            options=columns,
            index=_guess_column_index(columns, ['location', 'city', 'country', 'region', 'place']),
            help="Where the transaction occurred."
        )

        account_col = st.selectbox(
            "👤 Account ID",
            options=columns,
            index=_guess_column_index(columns, ['account', 'account_id', 'user', 'user_id', 'customer']),
            help="Groups transactions by customer/account."
        )

    with col2:
        st.subheader("Optional Columns")

        label_col = st.selectbox(
            "🏷️ Label / Class (for evaluation)",
            options=columns,
            index=_guess_column_index(columns, ['class', 'label', 'fraud', 'is_fraud', 'target']),
            help="Ground truth labels (0=normal, 1=anomaly). "
                 "If provided, the system will show precision/recall metrics."
        )

    # Build mapping
    mapping = {
        'amount': amount_col if amount_col != '— None —' else None,
        'time': time_col if time_col != '— None —' else None,
        'vendor': vendor_col if vendor_col != '— None —' else None,
        'location': location_col if location_col != '— None —' else None,
        'account_id': account_col if account_col != '— None —' else None,
        'label': label_col if label_col != '— None —' else None,
    }

    # Show mapping summary
    st.divider()
    st.subheader("Mapping Summary")

    mapped_cols = {k: v for k, v in mapping.items() if v}
    unmapped = {k: v for k, v in mapping.items() if not v}

    if mapped_cols:
        summary_df = pd.DataFrame([
            {'Role': k.replace('_', ' ').title(), 'Column': v, 'Status': '✅ Mapped'}
            for k, v in mapped_cols.items()
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if unmapped:
        unmapped_labels = [k.replace('_', ' ').title() for k in unmapped if k == 'label']
        if unmapped_labels:
            st.caption(f"Unmapped: {', '.join(unmapped_labels)}")

    # Feature preview
    st.subheader("Features That Will Be Created")
    features_list = ['amount', 'log_amount', 'amount_zscore', 'amount_deviation_from_mean',
                     'hour_of_day', 'day_of_week', 'is_weekend', 'high_risk_time']
    if mapping.get('account_id'):
        features_list += ['amount_vs_account_avg', 'account_tx_frequency']
    if mapping.get('vendor'):
        features_list += ['vendor_frequency', 'is_rare_vendor']
    if mapping.get('location'):
        features_list += ['location_frequency', 'is_rare_location']

    st.markdown("  \n".join(f"• `{f}`" for f in features_list))

    # Confirm button
    st.divider()
    if st.button("✅ Confirm Mapping", type="primary", use_container_width=True):
        ok, errors = validate_mapping(df, mapping)
        if not ok:
            for err in errors:
                st.error(err)
        else:
            st.session_state['column_mapping'] = mapping
            st.session_state['mapping_confirmed'] = True
            st.success("✅ Mapping confirmed! Go to **Tab 2 (Audit Workflow)** to run the model.")

    if st.session_state.get('mapping_confirmed'):
        st.success("✅ Mapping confirmed. Proceed to Tab 2.")


def _guess_column_index(columns, keywords):
    """Auto-guess the best column match from keywords. Returns index into columns list."""
    for i, col in enumerate(columns):
        for kw in keywords:
            if kw.lower() == col.lower():
                return i
    for i, col in enumerate(columns):
        for kw in keywords:
            if kw.lower() in col.lower():
                return i
    return 0  # '— None —'


# ============================================================
# SCREEN 3 — DETECTION
# ============================================================
def screen_detection():
    if st.session_state['df'] is None:
        st.warning("⬆️ Upload a dataset first (Tab 1).")
        return
    if not st.session_state.get('mapping_confirmed'):
        st.warning("⬆️ Confirm column mapping first (Tab 1).")
        return

    st.header("🎯 Anomaly Detection")

    mapping = st.session_state['column_mapping']

    contamination = 'auto'
    st.session_state['contamination'] = contamination

    # Pipeline description
    st.markdown("### What happens when you click Run:")
    mapped_items = [f"**{k.title()}** → `{v}`" for k, v in mapping.items() if v]
    st.markdown("**Column mapping:** " + " · ".join(mapped_items))
    st.markdown("""
    1. Engineer features from your mapped columns
    2. Load Pre-Trained Isolation Forest & Scaler
    3. Pad missing features to match training schema
    4. Predict anomalies instantaneously
    5. Compute per-transaction factor analysis
    6. Generate rule-based explanations
    """)

    if st.button("▶ Run Detection", type="primary", use_container_width=True):
        _run_detection(mapping, contamination)

    elif st.session_state.get('detection_done'):
        st.info(
            "✅ Detection complete. Review results below, or go to Tab 3 for per-transaction deep-dive. "
            "Re-run to change parameters."
        )


def _run_detection(mapping, contamination):
    """Execute the full detection pipeline with progress tracking."""
    progress = st.progress(0, text="Starting...")
    df = st.session_state['df']

    # Step 1: Feature Engineering
    progress.progress(10, text="Engineering features from your data...")
    try:
        feature_df, feature_names, stats, original_df = engineer_features(df, mapping)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        return

    st.session_state.update({
        'feature_df': feature_df,
        'feature_names': feature_names,
        'stats': stats,
        'original_df': original_df,
    })

    # Step 2: Pre-Trained Inference
    progress.progress(30, text="Running Pre-Trained Model Inference...")
    try:
        from engine import run_pretrained_inference
        (X_scaled, labels, scores,
         has_labels, y_test,
         inference_time, aligned_feature_names,
         model, scaler) = run_pretrained_inference(feature_df, original_df, mapping)
    except Exception as e:
        st.error(f"Inference failed (Is the model pre-trained?): {e}")
        return

    st.session_state.update({
        'feature_names': aligned_feature_names,
        'X_train_scaled': X_scaled,
        'X_test_scaled': X_scaled,
        'train_idx': np.arange(len(df)),
        'test_idx': np.arange(len(df)),
        'y_train': y_test,
        'y_test': y_test,
        'has_labels': has_labels,
        'model': model,
        'scaler': scaler,
        'labels': labels,
        'scores': scores,
        'train_time': inference_time,
    })

    # Step 4: Evaluate (if labels exist)
    progress.progress(55, text="Evaluating model...")
    metrics = None
    if has_labels and y_test is not None:
        try:
            metrics = evaluate_model(labels, scores, y_test)
        except Exception as e:
            st.warning(f"Evaluation skipped: {e}")
    st.session_state['metrics'] = metrics

    # Step 5: SHAP
    progress.progress(65, text="Analyzing which factors drove each flag...")
    try:
        explainer, shap_values = compute_shap(
            model, X_scaled, X_scaled, aligned_feature_names
        )
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        st.info("Tip: Ensure shap>=0.45 is installed. Run: pip install --upgrade shap")
        return

    st.session_state.update({
        'explainer': explainer,
        'shap_values': shap_values,
        'detection_done': True,
    })

    progress.progress(100, text="Done!")
    n_anomalies = int((labels == -1).sum())
    st.success(
        f"✅ Detection complete in **{inference_time:.2f}s**. "
        f"Flagged **{n_anomalies:,}** anomalies "
        f"({n_anomalies / len(labels) * 100:.1f}% of test set)."
    )


# ============================================================
# SCREEN 4 — RESULTS
# ============================================================
def screen_results():
    if not st.session_state.get('detection_done'):
        st.warning("⬆️ Run detection first (Tab 2).")
        return

    st.header("📈 Detection Results")

    labels = st.session_state['labels']
    scores = st.session_state['scores']
    feature_names = st.session_state['feature_names']
    shap_values = st.session_state['shap_values']
    metrics = st.session_state['metrics']
    has_labels = st.session_state['has_labels']
    contamination = st.session_state['contamination']
    test_idx = st.session_state['test_idx']
    original_df = st.session_state['original_df']
    mapping = st.session_state['column_mapping']

    n_anomalies = int((labels == -1).sum())
    n_normal = int((labels == 1).sum())

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Test Samples", f"{len(labels):,}")
    c2.metric("Anomalies Flagged", f"{n_anomalies:,}")
    c3.metric("Normal", f"{n_normal:,}")
    c4.metric("Detection Time", f"{st.session_state['train_time']:.2f}s")

    # Contamination notice
    st.warning(
        "⚠️ **Automatic Anomaly Threshold:** The model automatically discovered "
        "the anomaly boundary based on the statistical isolation of the transactions."
    )

    # If labels exist, show evaluation metrics
    if has_labels and metrics:
        st.subheader("Model Performance (vs Ground Truth)")
        st.caption(
            "These metrics are only available because your dataset has labels."
        )
        metrics_df = pd.DataFrame([{
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f}",
            'AUC-ROC': f"{metrics.get('auc_roc', 'N/A')}",
            'True Positives': metrics['tp'],
            'False Positives': metrics['fp'],
            'False Negatives': metrics['fn'],
        }])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Global SHAP importance
    st.subheader("Key Fraud Risk Factors")
    st.caption(
        "Which transaction characteristics most frequently triggered the alarm "
        "across the entire spreadsheet. Larger bar = stronger influence on the AI's decisions."
    )
    fig = _plot_global_importance(shap_values, feature_names)
    st.pyplot(fig)
    plt.close()

    # Score distribution
    st.subheader("System Confidence Graph")
    fig2 = _plot_score_distribution(scores, labels, contamination)
    st.pyplot(fig2)
    plt.close()

    # Flagged transactions table
    st.subheader("Flagged Transactions")
    anomaly_positions = np.where(labels == -1)[0]

    if len(anomaly_positions) == 0:
        st.info("No anomalies were flagged in this dataset.")
        return

    amount_col = mapping.get('amount')
    vendor_col = mapping.get('vendor')
    time_col = mapping.get('time')

    rows = []
    for pos in anomaly_positions:
        orig_idx = test_idx[pos]
        row_data = {
            'Position': int(pos),
            'Original Row': int(orig_idx),
            'Anomaly Score': round(float(scores[pos]), 5),
        }
        if amount_col and amount_col in original_df.columns:
            row_data['Amount'] = original_df.iloc[orig_idx][amount_col]
        if vendor_col and vendor_col in original_df.columns:
            row_data['Vendor'] = original_df.iloc[orig_idx][vendor_col]
        if time_col and time_col in original_df.columns:
            row_data['Time'] = str(original_df.iloc[orig_idx][time_col])
        if has_labels and st.session_state['y_test'] is not None:
            actual = st.session_state['y_test'][pos]
            row_data['Actual'] = '🔴 Fraud' if actual == 1 else '🟢 Normal'

        rows.append(row_data)

    df_display = pd.DataFrame(rows).sort_values('Anomaly Score')
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Transaction selector
    st.subheader("Select a Transaction to Explain")
    selected = st.selectbox(
        "Transaction",
        options=anomaly_positions.tolist(),
        format_func=lambda x: (
            f"Position #{x} | Score: {scores[x]:.5f} | "
            f"Row: {test_idx[x]}"
        )
    )
    if selected is not None:
        st.session_state['selected_tx_idx'] = int(selected)
        st.info(f"Selected position #{selected}. Go to **Tab 3 (Deep-Dive)** for details.")

    # CSV Export
    st.divider()
    st.subheader("📥 Export Audit Report")
    if st.button("Generate CSV Report", use_container_width=True):
        csv_bytes = build_export_csv(
            labels=labels,
            scores=scores,
            shap_values=shap_values,
            feature_names=feature_names,
            original_df=original_df,
            test_idx=test_idx,
            mapping=mapping,
            stats=st.session_state['stats'],
            contamination=contamination,
            metrics=metrics,
            feature_df=st.session_state['feature_df'],
        )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="⬇ Download anomaly_report.csv",
            data=csv_bytes,
            file_name=f"anomaly_report_{timestamp}.csv",
            mime='text/csv',
        )


def _plot_global_importance(shap_values, feature_names):
    """Bar chart of mean |SHAP| values."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    n_features = min(15, len(feature_names))
    order = np.argsort(mean_abs)[::-1][:n_features]

    fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.4)))
    ax.barh(
        [feature_names[i] for i in order[::-1]],
        mean_abs[order[::-1]],
        color='#4C72B0'
    )
    ax.set_xlabel('Overall Impact Severity')
    ax.set_title('Key Fraud Risk Factors\n(Which characteristics most frequently trigger the alarm across all transactions)')
    plt.tight_layout()
    return fig


def _plot_score_distribution(scores, labels, contamination):
    """Histogram of anomaly scores split by prediction."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    normal_scores = scores[labels == 1]
    anomaly_scores = scores[labels == -1]
    
    if len(normal_scores) > 0:
        ax.hist(normal_scores, bins=50, alpha=0.6,
                label='Normal', color='#4CAF50', density=True)
    if len(anomaly_scores) > 0:
        ax.hist(anomaly_scores, bins=50, alpha=0.8,
                label='Anomaly', color='#F44336', density=True)
                
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        threshold = anomaly_scores.max()
        ax.axvline(threshold, color='black', linestyle='--', label='Decision threshold')
        
    ax.set_xlabel('Pre-Trained Algorithm Confidence Score')
    ax.set_ylabel('Transaction Volume (Density)')
    ax.set_title('System Confidence Graph\n(Lower Score = Higher Risk of Fraud. Clean separation means AI found clear unusual patterns.)')
    ax.legend()
    plt.tight_layout()
    return fig


# ============================================================
# SCREEN 5 — EXPLANATION PANEL
# ============================================================
def screen_explanation():
    if not st.session_state.get('detection_done'):
        st.warning("⬆️ Run detection first (Tab 2).")
        return

    if st.session_state.get('selected_tx_idx') is None:
        st.warning(
            "No transaction selected. "
            "Go to Tab 2 (Audit Workflow) and select a flagged transaction."
        )
        return

    idx = st.session_state['selected_tx_idx']
    shap_values = st.session_state['shap_values']
    feature_names = st.session_state['feature_names']
    X_test_scaled = st.session_state['X_test_scaled']
    explainer = st.session_state['explainer']
    scores = st.session_state['scores']
    test_idx = st.session_state['test_idx']
    original_df = st.session_state['original_df']
    feature_df = st.session_state['feature_df']
    mapping = st.session_state['column_mapping']
    stats = st.session_state['stats']

    orig_idx = test_idx[idx]
    row_original = original_df.iloc[orig_idx]
    row_features = feature_df.iloc[orig_idx].to_dict()

    st.header(f"💡 Transaction Explanation — Row #{orig_idx}")

    # Key metrics
    amount_col = mapping.get('amount')
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomaly Score", round(float(scores[idx]), 5),
              help="A lower score means the AI considers this transaction more suspicious")
    if amount_col and amount_col in row_original.index:
        c2.metric("Amount", f"${float(row_original[amount_col]):,.2f}")
    else:
        c2.metric("Amount", "—")

    vendor_col = mapping.get('vendor')
    if vendor_col and vendor_col in row_original.index:
        c3.metric("Vendor", str(row_original[vendor_col]))
    else:
        time_col = mapping.get('time')
        if time_col and time_col in row_original.index:
            c3.metric("Time", str(row_original[time_col]))
        else:
            c3.metric("Row Index", str(orig_idx))

    # --- NATURAL LANGUAGE EXPLANATION ---
    st.subheader("📝 Explanation Summary")

    shap_top = get_top_features(idx, shap_values, feature_names, n=5)
    rule_explanations = generate_rule_explanations(
        row_original, row_features, mapping, stats
    )
    nl_explanation = generate_nl_explanation(
        shap_top, rule_explanations, float(scores[idx]),
        row_original, mapping
    )
    st.markdown(nl_explanation)

    # --- SHAP WATERFALL & TECHNICAL DETAILS (HIDDEN BY DEFAULT) ---
    with st.expander("🔬 View Technical Model Analysis (Advanced)"):
        st.subheader("AI Factor Breakdown")
        st.caption(
            "🔵 Blue (Negative Impact) = pushes score heavily toward Anomaly · 🔴 Red (Positive Impact) = pushes score toward Normal."
        )

        try:
            fig = _make_waterfall(idx, shap_values, X_test_scaled, feature_names, explainer)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Could not render waterfall plot: {e}")

        # --- TOP FEATURES ---
        st.subheader("Top 5 Contributing Factors")
        for rank, f in enumerate(shap_top, 1):
            icon = "🔵" if f['direction'] == 'toward anomaly' else "🔴"
            if f['direction'] == 'toward anomaly':
                direction_text = "pushed toward flagging"
            else:
                direction_text = "pushed toward normal"
            name = f['feature'].replace('_', ' ').title()
            st.markdown(
                f"**{rank}. {name}** — "
                f"Impact: `{abs(f['shap_value']):.4f}` {icon} {direction_text}"
            )

    # --- ORIGINAL TRANSACTION DATA ---
    st.subheader("📄 Original Transaction Data")
    tx_data = pd.DataFrame({
        'Field': row_original.index,
        'Value': row_original.values.astype(str)
    })
    st.dataframe(tx_data, use_container_width=True, hide_index=True)

    # --- DISCLAIMER ---
    st.divider()
    st.error(
        "**⚠️ Explanation Disclaimer**\n\n"
        "SHAP values are derived from the Isolation Forest model's internal "
        "tree structure. They are model-faithful — they explain how the model "
        "scored this transaction.\n\n"
        "However: Isolation Forest is **unsupervised**. A high anomaly score "
        "means this transaction is statistically unusual compared to training "
        "data — it does **not** confirm fraud.\n\n"
        "Rule-based explanations provide additional context using business "
        "logic thresholds. All flagged transactions require **human review** "
        "before any action is taken."
    )


def _make_waterfall(idx, shap_values, X_test_scaled, feature_names, explainer):
    """Generate SHAP waterfall plot for a single transaction."""
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
    plt.title(f"AI Factor Breakdown — Transaction (Row #{idx})", pad=20)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN APP ENTRY POINT
# ============================================================
def main():
    init_session_state()

    tabs = st.tabs([
        "📁 1. Data Setup (Upload & Map)",
        "🎯 2. Audit Workflow (Run & Review)",
        "💡 3. Deep-Dive Specific Row",
    ])

    with tabs[0]:
        screen_upload()
        if st.session_state.get('df') is not None:
             st.divider()
             screen_column_mapping()
             
    with tabs[1]:
        screen_detection()
        if st.session_state.get('detection_done'):
             st.divider()
             screen_results()
             
    with tabs[2]:
        screen_explanation()


if __name__ == "__main__":
    main()
