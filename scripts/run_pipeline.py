# ============================================================
# scripts/run_pipeline.py -- End-to-end pipeline on combined data
#
# Runs: load -> map -> features -> train -> predict -> evaluate -> SHAP
# Saves metrics to results/metrics.json
#
# Usage:
#   python scripts/run_pipeline.py
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine import (
    engineer_features, preprocess, train_model,
    run_prediction, evaluate_model, compute_shap, get_top_features,
    generate_rule_explanations
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def main():
    print("=" * 60)
    print("Full Pipeline Execution")
    print("=" * 60)
    t_start = time.time()

    # --- 1. Load ---
    print("\n[1/6] Loading combined training data...")
    train_path = os.path.join(DATA_DIR, 'combined_train.csv')
    df = pd.read_csv(train_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.4%}")

    # --- 2. Map ---
    mapping = {
        'amount': 'amount',
        'time': 'timestamp',
        'vendor': 'merchant',
        'location': 'city',
        'account_id': 'account_id',
        'label': 'is_fraud',
    }
    print(f"\n[2/6] Column mapping: {mapping}")

    # --- 3. Feature Engineering ---
    print("\n[3/6] Engineering features...")
    t0 = time.time()
    feature_df, feature_names, stats, original_df = engineer_features(df, mapping)
    feat_time = time.time() - t0
    print(f"  Features created: {len(feature_names)}")
    print(f"  Feature list: {feature_names}")
    print(f"  Time: {feat_time:.2f}s")

    # --- 4. Preprocess (split + scale) ---
    print("\n[4/6] Preprocessing (split + scale)...")
    (X_train, X_test, train_idx, test_idx,
     y_train, y_test, scaler, has_labels) = preprocess(feature_df, df, mapping)
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}")
    print(f"  Has labels:  {has_labels}")
    if has_labels:
        print(f"  Train fraud: {int(np.sum(y_train)):,} / {len(y_train):,}")
        print(f"  Test fraud:  {int(np.sum(y_test)):,} / {len(y_test):,}")

    # --- 5. Train + Predict ---
    contamination_values = [0.01, 0.03, 0.05, 0.10, 0.15]
    all_results = {}

    print("\n[5/6] Training Isolation Forest at multiple contamination levels...")
    for contamination in contamination_values:
        print(f"\n  --- contamination = {contamination} ---")
        model, train_time = train_model(X_train, contamination)
        labels, scores = run_prediction(model, X_test)

        n_flagged = int((labels == -1).sum())
        print(f"  Train time: {train_time:.3f}s")
        print(f"  Flagged: {n_flagged} / {len(labels)}")

        if has_labels:
            metrics = evaluate_model(labels, scores, y_test)
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            if 'auc_roc' in metrics:
                print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
                print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
            print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, "
                  f"TN: {metrics['tn']}, FN: {metrics['fn']}")

            metrics['contamination'] = contamination
            metrics['train_time'] = round(train_time, 3)
            metrics['n_test'] = len(labels)
            all_results[str(contamination)] = metrics

    # --- 6. SHAP (on best contamination) ---
    print("\n[6/6] Computing SHAP values (contamination=0.05)...")
    model_05, _ = train_model(X_train, 0.05)
    labels_05, scores_05 = run_prediction(model_05, X_test)

    # SHAP on a sample to keep it fast
    shap_sample_size = min(500, len(X_test))
    X_test_sample = X_test[:shap_sample_size]
    X_train_sample = X_train[:min(200, len(X_train))]

    t0 = time.time()
    explainer, shap_values = compute_shap(
        model_05, X_train_sample, X_test_sample, feature_names
    )
    shap_time = time.time() - t0
    print(f"  SHAP computed for {shap_sample_size} samples in {shap_time:.2f}s")

    # Top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Global Feature Importance (mean |SHAP|):")
    for fname, imp in feature_importance:
        print(f"    {fname:30s} {imp:.6f}")

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_payload = {
        'dataset': {
            'train_rows': len(df),
            'test_rows': len(X_test),
            'features': feature_names,
            'n_features': len(feature_names),
            'fraud_rate': round(df['is_fraud'].mean(), 6),
        },
        'evaluation': all_results,
        'feature_importance': [
            {'feature': f, 'mean_abs_shap': round(float(v), 6)}
            for f, v in feature_importance
        ],
        'stats': {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in stats.items()},
        'total_time': round(time.time() - t_start, 2),
    }

    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n  Results saved to: {metrics_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Contamination':>14} | {'Precision':>9} | {'Recall':>7} | {'F1':>7} | {'AUC-ROC':>8} | {'Flagged':>8}")
    print("-" * 70)
    for c_str, m in all_results.items():
        auc = m.get('auc_roc', 0)
        print(f"{float(c_str):>14.2f} | {m['precision']:>9.4f} | {m['recall']:>7.4f} | {m['f1']:>7.4f} | {auc:>8.4f} | {m['n_flagged']:>8}")

    print(f"\nTotal execution time: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
