# ============================================================
# scripts/evaluation_analysis.py -- Model Evaluation & Reporting
#
# Generates: ROC curve, PR curve, confusion matrix, SHAP
# feature importance, contamination sweep, and error analysis.
#
# Usage:
#   python scripts/evaluation_analysis.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine import (
    engineer_features, preprocess, train_model,
    run_prediction, compute_shap
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 10})

MAPPING = {
    'amount': 'amount', 'time': 'timestamp',
    'vendor': 'merchant', 'location': 'city',
    'account_id': 'account_id', 'label': 'is_fraud',
}


def prepare_pipeline():
    """Run pipeline and return model + data."""
    print("[1/4] Loading data and running pipeline...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'combined_train.csv'))

    features, names, stats, orig = engineer_features(train, MAPPING)
    X_train, X_test, train_idx, test_idx, y_train, y_test, scaler, has_labels = \
        preprocess(features, train, MAPPING)

    model, train_time = train_model(X_train, 0.10)
    labels, scores = run_prediction(model, X_test)

    # Anomaly scores: negate decision_function so higher = more anomalous
    raw_scores = -model.decision_function(X_test)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Fraud in test: {int(y_test.sum())} / {len(y_test)}")

    return model, X_train, X_test, y_test, labels, scores, raw_scores, names


def fig8_roc_curve(y_test, raw_scores):
    """ROC curve with AUC."""
    fpr, tpr, thresholds = roc_curve(y_test, raw_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5,
            label=f'Isolation Forest (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig8_roc_curve.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path} (AUC = {roc_auc:.4f})")


def fig9_pr_curve(y_test, raw_scores):
    """Precision-Recall curve with AP."""
    precision, recall, thresholds = precision_recall_curve(y_test, raw_scores)
    ap = average_precision_score(y_test, raw_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#3498db', lw=2.5,
            label=f'Isolation Forest (AP = {ap:.4f})')
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--',
               label=f'Baseline ({baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.1, color='#3498db')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig9_pr_curve.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path} (AP = {ap:.4f})")


def fig10_confusion_matrix(y_test, labels):
    """Confusion matrix heatmap."""
    # Convert IF labels: -1 = anomaly = predicted fraud (1), 1 = normal (0)
    y_pred = (labels == -1).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'],
                linewidths=1, linecolor='white',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix (contamination=0.10)', fontweight='bold', fontsize=13)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig10_confusion_matrix.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")

    # Print classification report
    print("\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'])
    for line in report.split('\n'):
        print(f"    {line}")


def fig11_shap_importance(model, X_train, X_test, feature_names):
    """SHAP feature importance bar chart."""
    print("\n  Computing SHAP values...")
    sample_size = min(300, len(X_test))
    bg_size = min(150, len(X_train))
    explainer, shap_values = compute_shap(
        model, X_train[:bg_size], X_test[:sample_size], feature_names
    )

    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_idx)))
    ax.barh(range(len(sorted_idx)),
            mean_abs[sorted_idx],
            color=colors, edgecolor='white')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('Feature Importance (SHAP)', fontweight='bold', fontsize=14)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig11_shap_importance.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def fig12_contamination_sweep(X_train, X_test, y_test):
    """Precision/Recall/F1 across contamination values."""
    print("\n  Running contamination sweep...")
    contaminations = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    metrics_list = []

    for c in contaminations:
        model, _ = train_model(X_train, c)
        labels, scores = run_prediction(model, X_test)
        y_pred = (labels == -1).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        metrics_list.append({
            'contamination': c, 'precision': prec,
            'recall': rec, 'f1': f1, 'flagged': int(tp + fp)
        })

    mdf = pd.DataFrame(metrics_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(mdf['contamination'], mdf['precision'], 'o-', color='#2ecc71', lw=2, label='Precision')
    ax1.plot(mdf['contamination'], mdf['recall'], 's-', color='#e74c3c', lw=2, label='Recall')
    ax1.plot(mdf['contamination'], mdf['f1'], '^-', color='#3498db', lw=2, label='F1 Score')
    ax1.set_xlabel('Contamination', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Metrics vs Contamination', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(mdf)), mdf['flagged'], color='#e67e22', edgecolor='white')
    ax2.set_xticks(range(len(mdf)))
    ax2.set_xticklabels([f'{c:.3f}' for c in mdf['contamination']], rotation=45)
    ax2.set_xlabel('Contamination', fontsize=12)
    ax2.set_ylabel('Transactions Flagged', fontsize=12)
    ax2.set_title('Flagged Count vs Contamination', fontweight='bold', fontsize=13)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig12_contamination_sweep.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def fig13_score_distribution(y_test, raw_scores):
    """Anomaly score distribution by class."""
    fig, ax = plt.subplots(figsize=(8, 5))

    normal_scores = raw_scores[y_test == 0]
    fraud_scores = raw_scores[y_test == 1]

    ax.hist(normal_scores, bins=80, alpha=0.6, label='Normal', color='#2ecc71', edgecolor='white')
    ax.hist(fraud_scores, bins=80, alpha=0.6, label='Fraud', color='#e74c3c', edgecolor='white')
    ax.set_xlabel('Anomaly Score (higher = more anomalous)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Anomaly Score Distribution by Class', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig13_score_distribution.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path}")


def main():
    print("=" * 60)
    print("Model Evaluation & Analysis")
    print("=" * 60)

    model, X_train, X_test, y_test, labels, scores, raw_scores, names = prepare_pipeline()

    print("\n[2/4] Generating evaluation figures...")
    fig8_roc_curve(y_test, raw_scores)
    fig9_pr_curve(y_test, raw_scores)
    fig10_confusion_matrix(y_test, labels)
    fig13_score_distribution(y_test, raw_scores)

    print("\n[3/4] SHAP analysis...")
    fig11_shap_importance(model, X_train, X_test, names)

    print("\n[4/4] Contamination sweep...")
    fig12_contamination_sweep(X_train, X_test, y_test)

    print(f"\nAll evaluation figures saved to: {FIG_DIR}")


if __name__ == '__main__':
    main()
