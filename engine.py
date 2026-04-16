# ============================================================
# engine.py — Core Logic for Schema-Adaptive Anomaly Detection
#
# This module contains ALL computation logic. No Streamlit imports.
# It is imported by app.py which handles the UI layer.
#
# Architecture:
#   1. Time parsing (intelligent format detection)
#   2. Feature engineering (adaptive to mapped columns)
#   3. Model training & prediction (Isolation Forest)
#   4. SHAP computation (TreeExplainer)
#   5. Rule-based explanations (threshold-based flags)
#   6. Natural language explanation generator
#   7. Export builder (CSV with metadata)
# ============================================================

import pandas as pd
import numpy as np
import shap
import time
import csv
import io
from datetime import datetime
import joblib
import json
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)


# ============================================================
# SECTION 1 — FILE PARSING
# ============================================================

def load_file(file_obj):
    """
    Load a CSV or Excel file into a DataFrame.

    Parameters
    ----------
    file_obj : UploadedFile (Streamlit) or file-like object
        The uploaded file. Detected by extension.

    Returns
    -------
    df : pd.DataFrame
    errors : list[str] — empty if successful
    """
    errors = []
    df = None
    name = getattr(file_obj, 'name', 'unknown')

    try:
        if name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_obj, engine='openpyxl')
        elif name.endswith('.csv'):
            df = pd.read_csv(file_obj)
        else:
            errors.append(
                f"Unsupported file type: {name}. "
                "Please upload a .csv, .xlsx, or .xls file."
            )
            return None, errors
    except Exception as e:
        errors.append(f"Could not read file: {e}")
        return None, errors

    if df is not None and len(df) == 0:
        errors.append("File is empty — no rows found.")
        return None, errors

    return df, errors


def validate_mapping(df, mapping):
    """
    Validate that the column mapping is usable.

    Parameters
    ----------
    mapping : dict
        Keys: 'amount', 'time', 'vendor', 'location', 'account_id', 'label'
        Values: column name (str) or None

    Returns
    -------
    is_valid : bool
    errors : list[str]
    """
    errors = []

    # --- Required columns ---
    required = {
        'amount': 'Amount',
        'time': 'Time/Date',
        'vendor': 'Vendor/Merchant',
        'location': 'Location',
        'account_id': 'Account ID',
    }
    for key, label in required.items():
        col = mapping.get(key)
        if not col:
            errors.append(f"'{label}' column is required. Please map it.")
        elif col not in df.columns:
            errors.append(f"Mapped {label.lower()} column '{col}' not found in data.")

    # --- Label is optional (enables evaluation metrics when present) ---
    label_col = mapping.get('label')
    if label_col and label_col not in df.columns:
        errors.append(f"Mapped label column '{label_col}' not found in data.")

    return len(errors) == 0, errors


# ============================================================
# SECTION 2 — INTELLIGENT TIME PARSING
# ============================================================

def parse_time_column(series):
    """
    Parse a time/date column into datetime, trying multiple strategies.

    Strategy order:
      1. pd.to_datetime with infer_datetime_format
      2. Numeric seconds (like Kaggle Credit Card: seconds from epoch)
      3. Numeric unix timestamp

    Returns
    -------
    parsed : pd.Series of datetime64
    method : str — which strategy succeeded
    """
    # Strategy 1: Standard datetime parsing
    # Note: infer_datetime_format was deprecated and removed in pandas 3.x;
    # pd.to_datetime() auto-infers formats by default in modern versions.
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().sum() > len(series) * 0.5:
            return parsed, 'datetime_string'
    except (ValueError, TypeError, Exception):
        pass

    # Strategy 2: Numeric — treat as seconds elapsed (Kaggle-style)
    try:
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().sum() > len(series) * 0.5:
            max_val = numeric.max()
            if max_val < 1e7:
                # Likely seconds from start of observation (Kaggle pattern)
                # Convert to datetime by adding to a reference date
                ref = pd.Timestamp('2025-01-01')
                parsed = ref + pd.to_timedelta(numeric, unit='s')
                return parsed, 'seconds_elapsed'
            else:
                # Likely Unix timestamp
                parsed = pd.to_datetime(numeric, unit='s', errors='coerce')
                if parsed.notna().sum() > len(series) * 0.5:
                    return parsed, 'unix_timestamp'
    except (ValueError, TypeError):
        pass

    # Fallback: return NaT series
    return pd.Series([pd.NaT] * len(series), index=series.index), 'failed'


# ============================================================
# SECTION 3 — ADAPTIVE FEATURE ENGINEERING
# ============================================================

def engineer_features(df, mapping):
    """
    Create derived features from the raw data based on column mapping.

    This is the core of schema adaptability. Features are created
    conditionally based on which columns the user mapped.

    Parameters
    ----------
    df : pd.DataFrame — raw uploaded data
    mapping : dict — column mapping from UI

    Returns
    -------
    feature_df : pd.DataFrame — engineered features only (no raw cols)
    feature_names : list[str]
    stats : dict — dataset statistics for rule-based explanations
    original_df : pd.DataFrame — copy of original data for display
    """
    original_df = df.copy()
    features = pd.DataFrame(index=df.index)

    # --- Amount features (REQUIRED) ---
    amount_col = mapping['amount']
    amount = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)

    features['amount'] = amount
    features['log_amount'] = np.log1p(amount.clip(lower=0))

    amount_mean = amount.mean()
    amount_std = amount.std()
    if amount_std > 0:
        features['amount_zscore'] = (amount - amount_mean) / amount_std
    else:
        features['amount_zscore'] = 0.0

    features['amount_deviation_from_mean'] = (amount - amount_mean).abs()

    # --- Time features (REQUIRED) ---
    time_col = mapping['time']
    parsed_time, time_method = parse_time_column(df[time_col])

    if time_method != 'failed':
        features['hour_of_day'] = parsed_time.dt.hour.fillna(12).astype(float)
        features['day_of_week'] = parsed_time.dt.dayofweek.fillna(3).astype(float)
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(float)
        features['high_risk_time'] = (
            (features['hour_of_day'] >= 0) & (features['hour_of_day'] <= 5)
        ).astype(float)
    else:
        # Fallback: no time features possible
        features['hour_of_day'] = 12.0
        features['day_of_week'] = 3.0
        features['is_weekend'] = 0.0
        features['high_risk_time'] = 0.0

    # --- Account-level behavioral features ---
    account_col = mapping.get('account_id')
    if account_col and account_col in df.columns:
        # Rolling average amount per account
        acct_means = df.groupby(account_col)[amount_col].transform('mean')
        acct_means = pd.to_numeric(acct_means, errors='coerce').fillna(amount_mean)
        features['amount_vs_account_avg'] = amount / acct_means.clip(lower=1)

        # Transaction frequency per account
        acct_counts = df.groupby(account_col)[amount_col].transform('count')
        total_count = len(df)
        features['account_tx_frequency'] = acct_counts.astype(float) / max(total_count, 1)

    # --- Vendor behavioral features ---
    vendor_col = mapping.get('vendor')
    if vendor_col and vendor_col in df.columns:
        # Vendor frequency (how common is this vendor overall)
        vendor_counts = df[vendor_col].value_counts()
        total = len(df)
        features['vendor_frequency'] = df[vendor_col].map(
            lambda v: vendor_counts.get(v, 0) / total
        ).astype(float)

        # Is rare vendor (seen fewer than 3 times)
        features['is_rare_vendor'] = df[vendor_col].map(
            lambda v: 1.0 if vendor_counts.get(v, 0) <= 2 else 0.0
        ).astype(float)

    # --- Location behavioral features ---
    location_col = mapping.get('location')
    if location_col and location_col in df.columns:
        loc_counts = df[location_col].value_counts()
        total = len(df)
        features['location_frequency'] = df[location_col].map(
            lambda loc: loc_counts.get(loc, 0) / total
        ).astype(float)

        features['is_rare_location'] = df[location_col].map(
            lambda loc: 1.0 if loc_counts.get(loc, 0) <= 2 else 0.0
        ).astype(float)

    # --- Compute dataset statistics for rule explanations ---
    stats = {
        'amount_mean': float(amount_mean),
        'amount_std': float(amount_std) if amount_std > 0 else 1.0,
        'amount_median': float(amount.median()),
        'amount_p95': float(amount.quantile(0.95)),
        'amount_p99': float(amount.quantile(0.99)),
        'total_transactions': len(df),
        'time_method': time_method,
    }

    feature_names = features.columns.tolist()
    return features, feature_names, stats, original_df


# ============================================================
# SECTION 4 — PREPROCESSING (SPLIT + SCALE)
# ============================================================

def preprocess(feature_df, df, mapping, allow_split=True):
    """
    Split and scale features. Handles both labeled and unlabeled data.

    Parameters
    ----------
    feature_df : pd.DataFrame — engineered features
    df : pd.DataFrame — original data (for labels)
    mapping : dict — column mapping
    allow_split: bool - whether to split into train/test (default True)

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray
    train_idx : np.ndarray — original indices for train set
    test_idx : np.ndarray — original indices for test set
    y_train : np.ndarray or None
    y_test : np.ndarray or None
    scaler : fitted StandardScaler
    has_labels : bool
    """
    label_col = mapping.get('label')
    has_labels = bool(label_col and label_col in df.columns)

    feature_df_clean = feature_df.fillna(0)
    X = feature_df_clean.values

    if not allow_split:
        y_train = y_test = df[label_col].values if has_labels else None
        train_idx = test_idx = np.arange(len(X))
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        X_test_scaled = X_train_scaled
        return (
            X_train_scaled, X_test_scaled,
            train_idx, test_idx,
            y_train, y_test,
            scaler, has_labels
        )

    if has_labels:
        y = df[label_col].values
        # Stratified split
        try:
            indices = np.arange(len(X))
            train_idx, test_idx, y_train, y_test = train_test_split(
                indices, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        except ValueError:
            # Stratification fails if a class has too few members
            train_idx, test_idx, y_train, y_test = train_test_split(
                indices, y,
                test_size=0.2,
                random_state=42
            )
    else:
        y_train = None
        y_test = None
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42
        )

    X_train = X[train_idx]
    X_test = X[test_idx]

    # Scale — fit ONLY on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled, X_test_scaled,
        train_idx, test_idx,
        y_train, y_test,
        scaler, has_labels
    )


# ============================================================
# SECTION 5 — ISOLATION FOREST TRAINING & PREDICTION
# ============================================================

def train_model(X_train_scaled, contamination='auto'):
    """
    Train Isolation Forest on scaled training data.

    Parameters
    ----------
    contamination : float or 'auto'
        Expected proportion of anomalies. Default 'auto' allows the model 
        to determine the threshold automatically.

    Returns
    -------
    model : fitted IsolationForest
    train_time : float (seconds)
    """
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    model.fit(X_train_scaled)
    train_time = time.time() - t0
    return model, train_time


def run_prediction(model, X_scaled):
    """
    Predict anomalies.

    Returns
    -------
    labels : np.ndarray — -1 = anomaly, +1 = normal
    scores : np.ndarray — raw decision scores (lower = more anomalous)
    """
    labels = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    return labels, scores

def run_pretrained_inference(feature_df, df, mapping, models_dir="models"):
    """
    Load pre-trained model/scaler and run inference on new dataset.
    Pads missing features to maintain compatibility.
    """
    t0 = time.time()
    
    # 1. Load artifacts
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Directory '{models_dir}' not found. Did you run train_offline.py?")
    
    model = joblib.load(os.path.join(models_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    expected_features = metadata["feature_names"]
    
    # 2. Align features
    aligned_df = pd.DataFrame(index=feature_df.index)
    for col in expected_features:
        if col in feature_df.columns:
            aligned_df[col] = feature_df[col].fillna(0)
        else:
            aligned_df[col] = 0.0  # fallback for missing features
            
    X = aligned_df.values
    X_scaled = scaler.transform(X)
    
    labels = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    
    label_col = mapping.get('label')
    has_labels = bool(label_col and label_col in df.columns)
    y = df[label_col].values if has_labels else None
    
    inference_time = time.time() - t0
    
    return (X_scaled, labels, scores, has_labels, y, inference_time, expected_features, model, scaler)


# ============================================================
# SECTION 6 — EVALUATION (ONLY WHEN LABELS EXIST)
# ============================================================

def evaluate_model(labels, scores, y_test):
    """
    Evaluate predictions against ground truth.
    Only callable when has_labels=True.

    Parameters
    ----------
    labels : np.ndarray — predicted labels (-1 = anomaly, +1 = normal)
    scores : np.ndarray — continuous anomaly scores from decision_function()
                          (lower = more anomalous)
    y_test : array-like — ground truth (0 = normal, 1 = anomaly)
    """
    y_pred = (labels == -1).astype(int)
    y_true = np.array(y_test).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
        'n_flagged': int((labels == -1).sum()),
    }

    # AUC requires both classes present and continuous scores
    # Negate scores because IF returns lower = more anomalous,
    # but AUC expects higher = more likely positive class
    if len(np.unique(y_true)) >= 2:
        anomaly_scores = -scores  # higher = more anomalous
        result['auc_roc'] = float(roc_auc_score(y_true, anomaly_scores))
        result['auc_pr'] = float(average_precision_score(y_true, anomaly_scores))

    return result


# ============================================================
# SECTION 7 — SHAP EXPLAINABILITY ENGINE
#
# SHAP TreeExplainer works on Isolation Forest because IF is
# built from ExtraTreeRegressor estimators. TreeExplainer
# traverses these tree structures to compute exact Shapley
# values per feature per transaction.
#
# BACKGROUND SAMPLES: Use 100, not the full training set.
# Full training set as background causes 10+ minute hangs.
# ============================================================

def compute_shap(model, X_train_scaled, X_test_scaled, feature_names):
    """
    Compute SHAP values for test samples using TreeExplainer.

    Returns
    -------
    explainer : shap.TreeExplainer
    shap_values : np.ndarray shape (n_test, n_features)
    """
    background = shap.sample(
        pd.DataFrame(X_train_scaled, columns=feature_names),
        min(100, len(X_train_scaled)),
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation='interventional'
    )

    shap_values = explainer.shap_values(
        pd.DataFrame(X_test_scaled, columns=feature_names)
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return explainer, shap_values


def get_top_features(idx, shap_values, feature_names, n=5):
    """
    Return top-N features by absolute SHAP value for one transaction.
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


# ============================================================
# SECTION 8 — RULE-BASED EXPLANATIONS
#
# These generate human-readable explanations based on simple
# thresholds and comparisons. They complement SHAP by providing
# concrete, business-meaningful context.
# ============================================================

def generate_rule_explanations(row_original, row_features, mapping, stats):
    """
    Generate rule-based explanations for a single transaction.

    Parameters
    ----------
    row_original : pd.Series — original row from uploaded data
    row_features : dict or pd.Series — engineered feature values
    mapping : dict — column mapping
    stats : dict — dataset statistics

    Returns
    -------
    explanations : list[str] — human-readable explanation strings
    """
    explanations = []
    amount_col = mapping.get('amount')

    # --- Amount rules ---
    if amount_col:
        amount = float(row_original.get(amount_col, 0))
        mean = stats['amount_mean']
        p95 = stats['amount_p95']
        p99 = stats['amount_p99']

        if mean > 0:
            ratio = amount / mean
            if ratio > 3:
                explanations.append(
                    f"💰 Amount (${amount:,.2f}) is {ratio:.1f}× higher "
                    f"than the average (${mean:,.2f})"
                )
            elif ratio > 2:
                explanations.append(
                    f"💰 Amount (${amount:,.2f}) is {ratio:.1f}× the average"
                )

        if amount > p99:
            explanations.append(
                f"📊 Amount exceeds the 99th percentile (${p99:,.2f})"
            )
        elif amount > p95:
            explanations.append(
                f"📊 Amount exceeds the 95th percentile (${p95:,.2f})"
            )

    # --- Time rules ---
    hour = row_features.get('hour_of_day', None)
    if hour is not None:
        hour = float(hour)
        if 0 <= hour <= 5:
            explanations.append(
                f"🌙 Transaction at {int(hour)}:{int((hour % 1) * 60):02d} AM "
                f"(high-risk overnight window)"
            )
        elif 23 <= hour <= 24:
            explanations.append(
                f"🌙 Late-night transaction at {int(hour)}:{int((hour % 1) * 60):02d}"
            )

    # --- Weekend rule ---
    is_weekend = row_features.get('is_weekend', 0)
    if float(is_weekend) == 1.0:
        explanations.append("📅 Weekend transaction (lower business volume period)")

    # --- Vendor rules ---
    vendor_col = mapping.get('vendor')
    if vendor_col and vendor_col in row_original.index:
        vendor = str(row_original[vendor_col])
        is_rare = row_features.get('is_rare_vendor', 0)
        if float(is_rare) == 1.0:
            explanations.append(
                f"🏪 Rare vendor: \"{vendor}\" (seen ≤2 times in dataset)"
            )
        if 'unknown' in vendor.lower():
            explanations.append(
                f"⚠️ Vendor identified as \"{vendor}\" — unknown merchant"
            )

    # --- Location rules ---
    location_col = mapping.get('location')
    if location_col and location_col in row_original.index:
        location = str(row_original[location_col])
        is_rare_loc = row_features.get('is_rare_location', 0)
        if float(is_rare_loc) == 1.0:
            explanations.append(
                f"📍 Unusual location: \"{location}\" (rarely seen in dataset)"
            )

    # --- Account deviation rules ---
    acct_ratio = row_features.get('amount_vs_account_avg', None)
    if acct_ratio is not None:
        acct_ratio = float(acct_ratio)
        if acct_ratio > 5:
            explanations.append(
                f"👤 Amount is {acct_ratio:.1f}× this account's average"
            )
        elif acct_ratio > 2:
            explanations.append(
                f"👤 Amount is {acct_ratio:.1f}× this account's typical spending"
            )

    return explanations


# ============================================================
# SECTION 9 — NATURAL LANGUAGE EXPLANATION GENERATOR
#
# Combines SHAP analysis + rule-based explanations into a
# single, auditor-readable paragraph.
# ============================================================

def generate_nl_explanation(
    shap_top_features, rule_explanations, anomaly_score, row_original, mapping
):
    """
    Generate a combined natural language explanation.

    Parameters
    ----------
    shap_top_features : list[dict] — from get_top_features()
    rule_explanations : list[str] — from generate_rule_explanations()
    anomaly_score : float — model's decision score
    row_original : pd.Series — original transaction data
    mapping : dict — column mapping

    Returns
    -------
    explanation : str — human-readable paragraph
    """
    parts = []

    # Opening
    amount_col = mapping.get('amount')
    if amount_col and amount_col in row_original.index:
        amount = float(row_original[amount_col])
        parts.append(
            f"This transaction of ${amount:,.2f} was flagged as anomalous "
            f"(anomaly score: {anomaly_score:.4f})."
        )
    else:
        parts.append(
            f"This transaction was flagged as anomalous "
            f"(anomaly score: {anomaly_score:.4f})."
        )

    # Rule-based reasons
    if rule_explanations:
        parts.append("\n\n**Key findings:**")
        for exp in rule_explanations:
            parts.append(f"\n- {exp}")

    # SHAP-based reasons
    anomaly_features = [f for f in shap_top_features if f['direction'] == 'toward anomaly']
    if anomaly_features:
        parts.append("\n\n**Model analysis (SHAP):**")
        for f in anomaly_features[:3]:
            name = f['feature'].replace('_', ' ').title()
            parts.append(
                f"\n- {name} pushed this transaction toward being flagged "
                f"(Impact: {abs(f['shap_value']):.4f} — the larger this number, "
                f"the stronger this factor contributed to the anomaly flag)"
            )

    # Disclaimer
    parts.append(
        "\n\n---\n*⚠️ Disclaimer: This is an automated analysis. "
        "Anomaly detection identifies statistically unusual patterns — "
        "it does not confirm fraud. Human review is required before "
        "any action is taken.*"
    )

    return ''.join(parts)


# ============================================================
# SECTION 10 — CSV EXPORT WITH METADATA
# ============================================================

def build_export_csv(
    labels, scores, shap_values, feature_names,
    original_df, test_idx, mapping, stats,
    contamination, metrics=None, feature_df=None
):
    """
    Build audit report CSV as bytes buffer.

    Returns
    -------
    csv_bytes : bytes — ready for st.download_button
    """
    anomaly_mask = labels == -1
    anomaly_positions = np.where(anomaly_mask)[0]

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Metadata header
    meta = [
        ['# === ANOMALY DETECTION AUDIT REPORT ==='],
        ['# Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['# Model', 'IsolationForest (scikit-learn)'],
        ['# contamination', contamination],
        ['# Explainability', 'SHAP TreeExplainer + Rule-based + NL'],
        ['# Total test samples', len(labels)],
        ['# Anomalies flagged', int(anomaly_mask.sum())],
        ['# Amount column', mapping.get('amount', 'N/A')],
        ['# Time column', mapping.get('time', 'N/A')],
    ]

    if metrics:
        meta.extend([
            ['# Precision', round(metrics.get('precision', 0), 4)],
            ['# Recall', round(metrics.get('recall', 0), 4)],
            ['# F1_Score', round(metrics.get('f1', 0), 4)],
        ])

    meta.extend([
        ['# Disclaimer',
         'Anomaly score indicates statistical unusualness, not confirmed fraud. '
         'Human review required.'],
        ['# ================================'],
        [],
    ])

    for row in meta:
        writer.writerow(row)

    # Data header
    amount_col = mapping.get('amount', '')
    header = [
        'test_position', 'original_index', 'anomaly_score',
        'amount', 'top1_feature', 'top1_shap',
        'top2_feature', 'top2_shap', 'top3_feature', 'top3_shap',
        'explanation'
    ]
    writer.writerow(header)

    for pos in anomaly_positions:
        orig_idx = test_idx[pos]
        sv = shap_values[pos]
        top3_idx = np.argsort(np.abs(sv))[::-1][:3]

        # Get amount
        amount_val = ''
        if amount_col and amount_col in original_df.columns:
            amount_val = round(float(original_df.iloc[orig_idx][amount_col]), 2)

        # Build rule explanations
        row_orig = original_df.iloc[orig_idx]
        row_feat = {}
        if feature_df is not None:
            row_feat = feature_df.iloc[orig_idx].to_dict()
        rules = generate_rule_explanations(row_orig, row_feat, mapping, stats)
        explanation_text = '; '.join(r.replace(',', ';') for r in rules) if rules else ''

        row = [
            int(pos),
            int(orig_idx),
            round(float(scores[pos]), 6),
            amount_val,
        ]
        for rank in top3_idx:
            row += [feature_names[rank], round(float(sv[rank]), 6)]

        row.append(explanation_text)
        writer.writerow(row)

    return buf.getvalue().encode('utf-8')
