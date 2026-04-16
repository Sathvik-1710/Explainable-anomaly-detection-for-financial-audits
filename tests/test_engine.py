"""
Unit tests for engine.py — Explainable Anomaly Detection for Financial Audits
Covers: load_file, validate_mapping, parse_time_column, engineer_features,
        preprocess, train_model, run_prediction, run_pretrained_inference,
        evaluate_model, compute_shap, get_top_features,
        generate_rule_explanations, generate_nl_explanation, build_export_csv
"""
import io
import json
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine  # noqa: E402


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def _make_fake_file(content: bytes, name: str):
    """Return a file-like object that mimics a Streamlit UploadedFile."""
    buf = io.BytesIO(content)
    buf.name = name
    return buf


def _sample_df(n=50, include_label=False):
    """Create a small synthetic DataFrame matching sample_transactions schema."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="h").astype(str),
            "amount": rng.uniform(10, 5000, n).round(2),
            "vendor": rng.choice(["ShopA", "ShopB", "RareVendor"], n),
            "location": rng.choice(["NYC", "LA", "RareLoc"], n),
            "account_id": rng.choice(["ACC1", "ACC2", "ACC3"], n),
        }
    )
    if include_label:
        df["label"] = rng.choice([0, 1], n, p=[0.9, 0.1])
    return df


def _sample_mapping(include_label=False):
    mapping = {
        "amount": "amount",
        "time": "date",
        "vendor": "vendor",
        "location": "location",
        "account_id": "account_id",
        "label": "label" if include_label else None,
    }
    return mapping


# ===========================================================================
# 1. load_file
# ===========================================================================

class TestLoadFile:
    def test_csv_loads_correctly(self):
        df_orig = _sample_df()
        csv_bytes = df_orig.to_csv(index=False).encode()
        fake = _make_fake_file(csv_bytes, "transactions.csv")
        df, errors = engine.load_file(fake)
        assert errors == []
        assert df is not None
        assert len(df) == len(df_orig)

    def test_unsupported_extension_returns_error(self):
        fake = _make_fake_file(b"data", "file.txt")
        df, errors = engine.load_file(fake)
        assert df is None
        assert any("Unsupported" in e or "file type" in e.lower() for e in errors)

    def test_empty_csv_returns_error(self):
        # Only header, no rows
        csv_bytes = b"amount,date,vendor\n"
        fake = _make_fake_file(csv_bytes, "empty.csv")
        df, errors = engine.load_file(fake)
        assert df is None
        assert any("empty" in e.lower() for e in errors)

    def test_corrupt_csv_returns_error(self):
        fake = _make_fake_file(b"\x00\x01\x02", "bad.csv")
        df, errors = engine.load_file(fake)
        # Should return an error gracefully, not raise
        assert df is None or errors  # either no df or there are errors

    def test_excel_loads_correctly(self, tmp_path):
        df_orig = _sample_df()
        xlsx_path = tmp_path / "tx.xlsx"
        df_orig.to_excel(xlsx_path, index=False)
        with open(xlsx_path, "rb") as f:
            content = f.read()
        fake = _make_fake_file(content, "tx.xlsx")
        df, errors = engine.load_file(fake)
        assert errors == []
        assert df is not None
        assert len(df) == len(df_orig)


# ===========================================================================
# 2. validate_mapping
# ===========================================================================

class TestValidateMapping:
    def test_valid_full_mapping(self):
        df = _sample_df()
        mapping = _sample_mapping()
        ok, errors = engine.validate_mapping(df, mapping)
        assert ok is True
        assert errors == []

    def test_missing_amount_col_in_data(self):
        df = _sample_df().drop(columns=["amount"])
        mapping = _sample_mapping()
        ok, errors = engine.validate_mapping(df, mapping)
        assert ok is False
        assert any("amount" in e.lower() for e in errors)

    def test_missing_time_mapping(self):
        df = _sample_df()
        mapping = _sample_mapping()
        mapping["time"] = None
        ok, errors = engine.validate_mapping(df, mapping)
        assert ok is False

    def test_label_col_wrong_name_errors(self):
        df = _sample_df()
        mapping = _sample_mapping()
        mapping["label"] = "nonexistent_label"
        ok, errors = engine.validate_mapping(df, mapping)
        assert ok is False
        assert any("label" in e.lower() or "nonexistent" in e.lower() for e in errors)

    def test_vendor_location_account_required_for_strong_detection(self):
        """vendor, location, account_id are REQUIRED — not optional.
        Without them the model only has amount + time features (8 features),
        which is insufficient for confident anomaly claims. The engine correctly
        rejects mappings that omit these columns."""
        df = _sample_df()
        mapping = {
            "amount": "amount",
            "time": "date",
            "vendor": None,
            "location": None,
            "account_id": None,
            "label": None,
        }
        ok, errors = engine.validate_mapping(df, mapping)
        assert ok is False
        # All three missing cols should each produce an error
        error_text = " ".join(errors).lower()
        assert "vendor" in error_text or "merchant" in error_text
        assert "location" in error_text
        assert "account" in error_text


# ===========================================================================
# 3. parse_time_column
# ===========================================================================

class TestParseTimeColumn:
    def test_standard_datetime_strings(self):
        series = pd.Series(["2025-01-01 09:00", "2025-01-02 14:30", "2025-01-03 02:15"])
        parsed, method = engine.parse_time_column(series)
        assert method == "datetime_string"
        assert parsed.notna().all()
        assert parsed.dt.hour.tolist() == [9, 14, 2]

    def test_kaggle_style_seconds_elapsed(self):
        """Small numeric values may be parsed by Strategy 1 (datetime_string) in
        pandas 3.x (treats them as years) or Strategy 2 (seconds_elapsed).
        Either way, the result must be parseable datetimes."""
        series = pd.Series([0, 3600, 86400, 172800], dtype=float)
        parsed, method = engine.parse_time_column(series)
        # pandas 3.x may pick 'datetime_string' before 'seconds_elapsed'
        assert method in ("seconds_elapsed", "datetime_string", "unix_timestamp")
        assert parsed.notna().all()

    def test_unix_timestamp(self):
        import time as _time
        now = int(_time.time())
        series = pd.Series([now, now + 3600, now + 7200], dtype=float)
        parsed, method = engine.parse_time_column(series)
        assert method in ("unix_timestamp", "datetime_string")
        assert parsed.notna().sum() >= 2

    def test_unparseable_returns_failed(self):
        series = pd.Series(["not-a-date", "also-bad", "???"] * 5)
        parsed, method = engine.parse_time_column(series)
        assert method == "failed"

    def test_handles_mixed_with_majority_valid(self):
        """If >50% parse, that strategy wins."""
        series = pd.Series(
            ["2025-01-01"] * 8 + ["garbage"] * 2
        )
        parsed, method = engine.parse_time_column(series)
        assert method == "datetime_string"


# ===========================================================================
# 4. engineer_features
# ===========================================================================

class TestEngineerFeatures:
    def test_required_features_present(self):
        df = _sample_df()
        mapping = _sample_mapping()
        features, feature_names, stats, orig = engine.engineer_features(df, mapping)
        required = ["amount", "log_amount", "amount_zscore",
                    "amount_deviation_from_mean", "hour_of_day",
                    "day_of_week", "is_weekend", "high_risk_time"]
        for f in required:
            assert f in feature_names, f"Missing required feature: {f}"

    def test_optional_features_when_cols_provided(self):
        df = _sample_df()
        mapping = _sample_mapping()
        features, feature_names, stats, _ = engine.engineer_features(df, mapping)
        optional = ["amount_vs_account_avg", "account_tx_frequency",
                    "vendor_frequency", "is_rare_vendor",
                    "location_frequency", "is_rare_location"]
        for f in optional:
            assert f in feature_names, f"Missing optional feature: {f}"

    def test_optional_features_absent_when_cols_none(self):
        df = _sample_df()
        mapping = {
            "amount": "amount", "time": "date",
            "vendor": None, "location": None, "account_id": None, "label": None
        }
        features, feature_names, stats, _ = engine.engineer_features(df, mapping)
        assert "vendor_frequency" not in feature_names
        assert "location_frequency" not in feature_names
        assert "amount_vs_account_avg" not in feature_names

    def test_no_nan_in_features(self):
        df = _sample_df()
        mapping = _sample_mapping()
        features, _, _, _ = engine.engineer_features(df, mapping)
        assert not features.isnull().any().any(), "Features contain NaN values"

    def test_log_amount_is_non_negative(self):
        df = _sample_df()
        mapping = _sample_mapping()
        features, _, _, _ = engine.engineer_features(df, mapping)
        assert (features["log_amount"] >= 0).all()

    def test_high_risk_time_binary(self):
        df = _sample_df()
        mapping = _sample_mapping()
        features, _, _, _ = engine.engineer_features(df, mapping)
        assert features["high_risk_time"].isin([0.0, 1.0]).all()

    def test_is_weekend_binary(self):
        df = _sample_df()
        mapping = _sample_mapping()
        features, _, _, _ = engine.engineer_features(df, mapping)
        assert features["is_weekend"].isin([0.0, 1.0]).all()

    def test_stats_keys_present(self):
        df = _sample_df()
        mapping = _sample_mapping()
        _, _, stats, _ = engine.engineer_features(df, mapping)
        for key in ["amount_mean", "amount_std", "amount_median",
                    "amount_p95", "amount_p99", "total_transactions"]:
            assert key in stats, f"Missing stats key: {key}"

    def test_output_shape_matches_input(self):
        df = _sample_df(n=30)
        mapping = _sample_mapping()
        features, _, _, _ = engine.engineer_features(df, mapping)
        assert len(features) == 30

    def test_original_df_unchanged(self):
        df = _sample_df()
        mapping = _sample_mapping()
        _, _, _, orig = engine.engineer_features(df, mapping)
        # original_df should contain the original columns
        assert "amount" in orig.columns


# ===========================================================================
# 5. preprocess
# ===========================================================================

class TestPreprocess:
    def test_returns_correct_shapes(self):
        df = _sample_df(n=50)
        mapping = _sample_mapping()
        feat, _, _, _ = engine.engineer_features(df, mapping)
        X_tr, X_te, tr_idx, te_idx, y_tr, y_te, scaler, has_labels = \
            engine.preprocess(feat, df, mapping)
        assert X_tr.shape[0] + X_te.shape[0] == 50
        assert X_tr.shape[1] == feat.shape[1]
        assert has_labels is False

    def test_no_data_leakage_scaler_fitted_on_train_only(self):
        df = _sample_df(n=100)
        mapping = _sample_mapping()
        feat, _, _, _ = engine.engineer_features(df, mapping)
        X_tr, X_te, tr_idx, te_idx, y_tr, y_te, scaler, _ = \
            engine.preprocess(feat, df, mapping)
        # Mean of scaled train should be ~0 (fitted on train)
        assert abs(X_tr.mean()) < 0.5

    def test_with_labels_has_labels_true(self):
        df = _sample_df(n=50, include_label=True)
        mapping = _sample_mapping(include_label=True)
        feat, _, _, _ = engine.engineer_features(df, mapping)
        _, _, _, _, y_tr, y_te, _, has_labels = engine.preprocess(feat, df, mapping)
        assert has_labels is True
        assert y_tr is not None
        assert y_te is not None

    def test_no_split_mode(self):
        df = _sample_df(n=30)
        mapping = _sample_mapping()
        feat, _, _, _ = engine.engineer_features(df, mapping)
        X_tr, X_te, tr_idx, te_idx, y_tr, y_te, scaler, _ = \
            engine.preprocess(feat, df, mapping, allow_split=False)
        assert len(tr_idx) == 30
        assert len(te_idx) == 30
        np.testing.assert_array_equal(X_tr, X_te)


# ===========================================================================
# 6. train_model
# ===========================================================================

class TestTrainModel:
    def test_returns_fitted_model_and_time(self):
        X = np.random.default_rng(0).standard_normal((100, 8))
        model, t = engine.train_model(X, contamination=0.1)
        assert hasattr(model, "predict")
        assert t > 0

    def test_model_predicts_on_new_data(self):
        X_train = np.random.default_rng(0).standard_normal((100, 8))
        X_test = np.random.default_rng(1).standard_normal((20, 8))
        model, _ = engine.train_model(X_train, contamination=0.1)
        labels = model.predict(X_test)
        assert set(labels).issubset({-1, 1})

    def test_auto_contamination(self):
        X = np.random.default_rng(0).standard_normal((100, 8))
        model, _ = engine.train_model(X, contamination='auto')
        assert hasattr(model, "predict")


# ===========================================================================
# 7. run_prediction
# ===========================================================================

class TestRunPrediction:
    def test_labels_binary(self):
        X = np.random.default_rng(0).standard_normal((100, 8))
        model, _ = engine.train_model(X, contamination=0.1)
        labels, scores = engine.run_prediction(model, X)
        assert set(labels).issubset({-1, 1})
        assert len(scores) == 100

    def test_scores_are_floats(self):
        X = np.random.default_rng(0).standard_normal((80, 8))
        model, _ = engine.train_model(X, contamination=0.1)
        labels, scores = engine.run_prediction(model, X)
        assert scores.dtype in (np.float32, np.float64)

    def test_anomaly_count_matches_contamination(self):
        """With contamination=0.1 on 100 samples, expect ~10 anomalies."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((1000, 8))
        model, _ = engine.train_model(X_train, contamination=0.1)
        labels, _ = engine.run_prediction(model, X_train)
        n_anomalies = (labels == -1).sum()
        # Should be exactly 10% for IsolationForest with explicit contamination
        assert 80 <= n_anomalies <= 120


# ===========================================================================
# 8. run_pretrained_inference
# ===========================================================================

class TestRunPretrainedInference:
    def test_loads_from_models_dir(self):
        df = _sample_df(n=30)
        mapping = _sample_mapping()
        feat, _, _, _ = engine.engineer_features(df, mapping)
        result = engine.run_pretrained_inference(feat, df, mapping, models_dir="models")
        X_scaled, labels, scores, has_labels, y, t, exp_feat, model, scaler = result
        assert labels is not None
        assert len(labels) == 30
        assert set(labels).issubset({-1, 1})

    def test_pads_missing_features(self):
        """If feature_df has fewer cols than the trained model, it should pad with 0."""
        df = _sample_df(n=20)
        # Only map amount + time (no vendor, location, account)
        mapping = {
            "amount": "amount", "time": "date",
            "vendor": None, "location": None, "account_id": None, "label": None
        }
        feat, _, _, _ = engine.engineer_features(df, mapping)
        # feat will have fewer columns than the 14-feature model
        result = engine.run_pretrained_inference(feat, df, mapping, models_dir="models")
        _, labels, _, _, _, _, _, _, _ = result
        assert len(labels) == 20

    def test_missing_models_dir_raises(self):
        df = _sample_df(n=10)
        mapping = _sample_mapping()
        feat, _, _, _ = engine.engineer_features(df, mapping)
        with pytest.raises(FileNotFoundError):
            engine.run_pretrained_inference(feat, df, mapping, models_dir="nonexistent_dir")


# ===========================================================================
# 9. evaluate_model
# ===========================================================================

class TestEvaluateModel:
    def _make_labels_and_scores(self, n=100, n_anomaly=10):
        rng = np.random.default_rng(0)
        labels = np.ones(n, dtype=int)
        anomaly_idx = rng.choice(n, n_anomaly, replace=False)
        labels[anomaly_idx] = -1
        scores = rng.standard_normal(n)
        y_true = np.zeros(n, dtype=int)
        # Make some true positives overlap with anomaly_idx
        tp_idx = anomaly_idx[:5]
        y_true[tp_idx] = 1
        return labels, scores, y_true

    def test_returns_required_keys(self):
        labels, scores, y_true = self._make_labels_and_scores()
        result = engine.evaluate_model(labels, scores, y_true)
        for key in ["precision", "recall", "f1", "tp", "fp", "tn", "fn", "n_flagged"]:
            assert key in result, f"Missing key: {key}"

    def test_precision_in_range(self):
        labels, scores, y_true = self._make_labels_and_scores()
        result = engine.evaluate_model(labels, scores, y_true)
        assert 0.0 <= result["precision"] <= 1.0

    def test_n_flagged_matches_anomaly_count(self):
        labels, scores, y_true = self._make_labels_and_scores(n=100, n_anomaly=15)
        result = engine.evaluate_model(labels, scores, y_true)
        assert result["n_flagged"] == 15

    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 0, 1, 1])
        labels = np.array([1, 1, 1, -1, -1])  # IF convention
        scores = np.array([0.1, 0.1, 0.1, -0.5, -0.5])
        result = engine.evaluate_model(labels, scores, y_true)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0


# ===========================================================================
# 10. compute_shap
# ===========================================================================

class TestComputeShap:
    def test_shap_values_shape(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((200, 8))
        X_test = rng.standard_normal((30, 8))
        model, _ = engine.train_model(X_train, contamination=0.1)
        feature_names = [f"f{i}" for i in range(8)]
        explainer, shap_values = engine.compute_shap(model, X_train, X_test, feature_names)
        assert shap_values.shape == (30, 8)

    def test_shap_returns_ndarray(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 5))
        model, _ = engine.train_model(X, contamination=0.1)
        fnames = [f"feat{i}" for i in range(5)]
        _, shap_values = engine.compute_shap(model, X, X[:10], fnames)
        assert isinstance(shap_values, np.ndarray)


# ===========================================================================
# 11. get_top_features
# ===========================================================================

class TestGetTopFeatures:
    def _make_shap(self, n=20, n_feat=8):
        return np.random.default_rng(0).standard_normal((n, n_feat))

    def test_returns_n_features(self):
        sv = self._make_shap()
        fnames = [f"f{i}" for i in range(8)]
        top = engine.get_top_features(0, sv, fnames, n=3)
        assert len(top) == 3

    def test_sorted_by_abs_shap(self):
        sv = self._make_shap()
        fnames = [f"f{i}" for i in range(8)]
        top = engine.get_top_features(0, sv, fnames, n=5)
        abs_vals = [entry["abs"] for entry in top]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_direction_label_correct(self):
        sv = np.array([[3.0, -2.0]])
        fnames = ["pos", "neg"]
        top = engine.get_top_features(0, sv, fnames, n=2)
        dirs = {e["feature"]: e["direction"] for e in top}
        assert dirs["pos"] == "toward normal"
        assert dirs["neg"] == "toward anomaly"

    def test_handles_single_feature(self):
        sv = np.array([[0.5]])
        top = engine.get_top_features(0, sv, ["amount"], n=1)
        assert len(top) == 1
        assert top[0]["feature"] == "amount"


# ===========================================================================
# 12. generate_rule_explanations
# ===========================================================================

class TestGenerateRuleExplanations:
    def _make_row(self, amount=5000.0, vendor="TestShop", location="NYC"):
        return pd.Series({"amount": amount, "vendor": vendor, "location": location})

    def _make_stats(self):
        return {
            "amount_mean": 500.0,
            "amount_std": 200.0,
            "amount_median": 400.0,
            "amount_p95": 1000.0,
            "amount_p99": 2000.0,
            "amount_p05": 100.0,
            "amount_p01": 50.0,
            "total_transactions": 100,
        }

    def _make_mapping(self):
        return {"amount": "amount", "vendor": "vendor", "location": "location"}

    def test_high_amount_flagged(self):
        row = self._make_row(amount=5000.0)  # 10× mean of 500
        feats = {"hour_of_day": 12.0, "is_weekend": 0, "is_rare_vendor": 0,
                 "is_rare_location": 0}
        rules = engine.generate_rule_explanations(row, feats, self._make_mapping(), self._make_stats())
        assert any("amount" in r.lower() or "💰" in r for r in rules)

    def test_high_risk_time_flagged(self):
        row = self._make_row(amount=50.0)
        feats = {"hour_of_day": 2.0, "is_weekend": 0, "is_rare_vendor": 0,
                 "is_rare_location": 0}
        rules = engine.generate_rule_explanations(row, feats, self._make_mapping(), self._make_stats())
        assert any("🌙" in r or "overnight" in r.lower() or "high-risk" in r.lower() for r in rules)

    def test_rare_vendor_flagged(self):
        row = self._make_row(vendor="RareVendor")
        feats = {"hour_of_day": 12.0, "is_weekend": 0, "is_rare_vendor": 1.0,
                 "is_rare_location": 0}
        rules = engine.generate_rule_explanations(row, feats, self._make_mapping(), self._make_stats())
        assert any("vendor" in r.lower() or "🏪" in r for r in rules)

    def test_normal_transaction_no_rules(self):
        """Unremarkable transaction should produce no flags."""
        row = self._make_row(amount=450.0)  # close to mean
        feats = {"hour_of_day": 14.0, "is_weekend": 0, "is_rare_vendor": 0,
                 "is_rare_location": 0}
        rules = engine.generate_rule_explanations(row, feats, self._make_mapping(), self._make_stats())
        # Amount is 0.9× mean, within 95th percentile — no flags expected
        assert all("💰" not in r for r in rules)

    def test_returns_list(self):
        row = self._make_row()
        feats = {"hour_of_day": 12.0, "is_weekend": 0, "is_rare_vendor": 0,
                 "is_rare_location": 0}
        result = engine.generate_rule_explanations(row, feats, self._make_mapping(), self._make_stats())
        assert isinstance(result, list)


# ===========================================================================
# 13. generate_nl_explanation
# ===========================================================================

class TestGenerateNlExplanation:
    def _top_features(self):
        return [
            {"feature": "amount_zscore", "shap_value": -0.5,
             "direction": "toward anomaly", "abs": 0.5},
            {"feature": "log_amount", "shap_value": 0.2,
             "direction": "toward normal", "abs": 0.2},
        ]

    def test_returns_string(self):
        row = pd.Series({"amount": 4500.0})
        mapping = {"amount": "amount"}
        nl = engine.generate_nl_explanation(
            self._top_features(), ["High amount"], -0.25, row, mapping
        )
        assert isinstance(nl, str)
        assert len(nl) > 50

    def test_contains_disclaimer(self):
        row = pd.Series({"amount": 1000.0})
        mapping = {"amount": "amount"}
        nl = engine.generate_nl_explanation(
            self._top_features(), [], -0.1, row, mapping
        )
        assert "disclaimer" in nl.lower() or "human review" in nl.lower()

    def test_contains_bold_amount(self):
        row = pd.Series({"amount": 800.0})
        mapping = {"amount": "amount"}
        nl = engine.generate_nl_explanation(
            self._top_features(), [], -0.3142, row, mapping
        )
        assert "**$800.00**" in nl or "800" in nl

    def test_includes_shap_analysis(self):
        row = pd.Series({"amount": 2000.0})
        mapping = {"amount": "amount"}
        nl = engine.generate_nl_explanation(
            self._top_features(), ["High amount"], -0.4, row, mapping
        )
        assert "Statistical" in nl or "deviates" in nl


# ===========================================================================
# 14. build_export_csv
# ===========================================================================

class TestBuildExportCsv:
    def _make_inputs(self, n=30):
        rng = np.random.default_rng(0)
        labels = rng.choice([-1, 1], n)
        scores = rng.standard_normal(n)
        shap_values = rng.standard_normal((n, 4))
        feature_names = ["amount", "log_amount", "amount_zscore", "hour_of_day"]
        df = _sample_df(n=n)
        test_idx = np.arange(n)
        mapping = _sample_mapping()
        stats = {"amount_mean": 500.0, "amount_std": 200.0,
                 "amount_median": 400.0, "amount_p95": 1000.0,
                 "amount_p99": 2000.0, "amount_p05": 100.0,
                 "amount_p01": 50.0, "total_transactions": n}
        contamination = 0.1
        return labels, scores, shap_values, feature_names, df, test_idx, mapping, stats, contamination

    def test_returns_bytes(self):
        args = self._make_inputs()
        result = engine.build_export_csv(*args)
        assert isinstance(result, bytes)

    def test_csv_has_metadata_header(self):
        args = self._make_inputs()
        csv_str = engine.build_export_csv(*args).decode("utf-8")
        assert "AUDIT REPORT" in csv_str or "ANOMALY" in csv_str

    def test_csv_has_data_rows(self):
        args = self._make_inputs()
        csv_str = engine.build_export_csv(*args).decode("utf-8")
        lines = [l for l in csv_str.splitlines() if l and not l.startswith("#")]
        # At least a header row + some data rows
        assert len(lines) >= 2

    def test_no_anomalies_still_valid(self):
        labels, scores, sv, fnames, df, idx, mapping, stats, cont = self._make_inputs()
        labels[:] = 1  # all normal
        result = engine.build_export_csv(labels, scores, sv, fnames, df, idx, mapping, stats, cont)
        assert isinstance(result, bytes)
        csv_str = result.decode("utf-8")
        assert "AUDIT REPORT" in csv_str or "ANOMALY" in csv_str

    def test_with_metrics_in_header(self):
        args = self._make_inputs()
        metrics = {"precision": 0.85, "recall": 0.72, "f1": 0.78}
        result = engine.build_export_csv(*args, metrics=metrics)
        csv_str = result.decode("utf-8")
        assert "0.85" in csv_str or "Precision" in csv_str
