"""
tests/test_app.py — Automated tests for the Student Performance Predictor.

Tests cover:
  1. Dataset loading and validation
  2. Feature preparation consistency
  3. Input validation (valid and invalid cases)
  4. Model training and evaluation
  5. Prediction correctness
  6. No prohibited frontend files were created
  7. Model selection (each model produces a valid result)
  8. Edge cases: boundary values, wrong types, missing absence values

Run with:
    python -m pytest tests/test_app.py -v
"""

import sys
from pathlib import Path

# Add project root to path so imports work when tests are run from any directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

import config
import modules


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def df():
    """Load the real dataset once for all tests."""
    return modules.load_dataset()


@pytest.fixture(scope="session")
def trained_results(df):
    """Train all models once for all tests."""
    return modules.train_all_models(df)


# ── 1. Dataset loading ────────────────────────────────────────────────────────

class TestDatasetLoading:
    def test_dataset_loads(self, df):
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Dataset must not be empty"

    def test_required_columns_present(self, df):
        for col in config.FEATURE_COLUMNS + [config.TARGET_COLUMN]:
            assert col in df.columns, f"Column '{col}' missing from dataset"

    def test_no_missing_values_in_features(self, df):
        for col in config.FEATURE_COLUMNS:
            assert df[col].isnull().sum() == 0, f"Column '{col}' has missing values"

    def test_target_classes_valid(self, df):
        actual = set(df[config.TARGET_COLUMN].unique())
        expected = set(config.CLASS_ORDER)
        assert actual == expected, f"Unexpected classes: {actual - expected}"

    def test_dataset_stats(self, df):
        stats = modules.get_dataset_stats(df)
        assert stats["total_records"] == len(df)
        assert stats["valid_records"] == len(df)
        assert stats["missing_records"] == 0
        assert stats["num_features"] == len(config.FEATURE_COLUMNS)
        assert stats["num_classes"] == 3
        assert stats["num_models"] == len(config.MODEL_KEYS)
        assert sum(stats["class_distribution"].values()) == stats["total_records"]


# ── 2. Feature preparation ────────────────────────────────────────────────────

class TestFeaturePreparation:
    def test_prepare_features_shape(self, df):
        X, y = modules.prepare_features(df)
        assert X.shape == (len(df), len(config.FEATURE_COLUMNS))
        assert y.shape == (len(df),)

    def test_absence_encoding_under7(self, df):
        row = df[df["StudentAbsenceDays"] == "Under-7"].iloc[:1]
        X, _ = modules.prepare_features(row)
        assert X[0, 3] == 1, "Under-7 should encode to 1"

    def test_absence_encoding_above7(self, df):
        row = df[df["StudentAbsenceDays"] == "Above-7"].iloc[:1]
        X, _ = modules.prepare_features(row)
        assert X[0, 3] == 0, "Above-7 should encode to 0"

    def test_prepare_single_input_matches_batch(self, df):
        """Single-input preparation must produce the same feature vector as batch."""
        row = df.iloc[0]
        X_batch, _ = modules.prepare_features(df.iloc[[0]])
        att = float(row["AttendancePercentage"]) if "AttendancePercentage" in row else None
        sh = float(row["StudyHours"]) if "StudyHours" in row else None
        X_single = modules.prepare_single_input(
            int(row["raisedhands"]),
            int(row["VisITedResources"]),
            int(row["Discussion"]),
            row["StudentAbsenceDays"],
            att,
            sh,
        )
        np.testing.assert_array_equal(X_batch, X_single)

    def test_feature_order_consistent(self, df):
        """Column order in X must always be raisedhands, VisITedResources, Discussion, absence."""
        X, _ = modules.prepare_features(df)
        expected_col0 = df["raisedhands"].values.astype(float)
        np.testing.assert_array_equal(X[:, 0], expected_col0)

        expected_col1 = df["VisITedResources"].values.astype(float)
        np.testing.assert_array_equal(X[:, 1], expected_col1)


# ── 3. Input validation ───────────────────────────────────────────────────────

class TestInputValidation:
    def _ok(self, **kwargs):
        defaults = {
            "raisedhands": 50,
            "visited_resources": 60,
            "discussion": 30,
            "absence_days": "Under-7",
        }
        defaults.update(kwargs)
        return modules.validate_inputs(**defaults)

    def test_valid_input_passes(self):
        valid, err, cleaned = self._ok()
        assert valid is True
        assert err == ""
        assert cleaned["raisedhands"] == 50

    def test_boundary_min(self):
        valid, _, _ = self._ok(raisedhands=0, visited_resources=0, discussion=0)
        assert valid is True

    def test_boundary_max(self):
        valid, _, _ = self._ok(raisedhands=100, visited_resources=100, discussion=100)
        assert valid is True

    def test_out_of_range_high(self):
        valid, err, _ = self._ok(raisedhands=101)
        assert valid is False
        assert "Raised Hands" in err

    def test_out_of_range_low(self):
        valid, err, _ = self._ok(discussion=-1)
        assert valid is False
        assert "Discussion" in err

    def test_non_numeric_int_field(self):
        valid, err, _ = self._ok(raisedhands="abc")
        assert valid is False
        assert "Raised Hands" in err

    def test_none_int_field(self):
        valid, err, _ = self._ok(visited_resources=None)
        assert valid is False

    def test_invalid_absence_value(self):
        valid, err, _ = self._ok(absence_days="Never")
        assert valid is False
        assert "Absence Days" in err

    def test_float_input_rounded_to_int(self):
        # int(45.9) == 45 — should pass since int() truncates
        valid, _, cleaned = self._ok(raisedhands=45.9)
        assert valid is True
        assert cleaned["raisedhands"] == 45

    def test_multiple_errors_reported(self):
        valid, err, _ = modules.validate_inputs(
            raisedhands=None,
            visited_resources=None,
            discussion=None,
            absence_days="INVALID",
        )
        assert valid is False
        assert "|" in err  # multiple errors joined


# ── 4. Model training ─────────────────────────────────────────────────────────

class TestModelTraining:
    def test_all_models_trained(self, trained_results):
        for key in config.MODEL_KEYS:
            assert key in trained_results["pipelines"], f"Pipeline missing: {key}"
            assert key in trained_results["metrics"], f"Metrics missing: {key}"

    def test_accuracy_is_realistic(self, trained_results):
        """Accuracy should be between 40% and 100% (not fabricated)."""
        for key in config.MODEL_KEYS:
            acc = trained_results["metrics"][key]["accuracy"]
            assert 0.40 <= acc <= 1.0, f"{key}: unrealistic accuracy {acc}"

    def test_metrics_not_fabricated(self, trained_results):
        """Precision, recall, f1 must be in [0,1]."""
        for key in config.MODEL_KEYS:
            m = trained_results["metrics"][key]
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["f1"] <= 1.0

    def test_confusion_matrix_sums_to_test_size(self, trained_results):
        test_size = trained_results["test_size"]
        for key in config.MODEL_KEYS:
            cm = np.array(trained_results["metrics"][key]["confusion_matrix"])
            assert cm.sum() == test_size, f"{key}: confusion matrix sum mismatch"

    def test_train_test_sizes_correct(self, df, trained_results):
        expected_test = int(len(df) * config.TEST_SIZE)
        # Allow ±1 for rounding
        assert abs(trained_results["test_size"] - expected_test) <= 2

    def test_no_data_leakage_check(self, trained_results):
        """Training and test sizes together must equal dataset size."""
        total = trained_results["train_size"] + trained_results["test_size"]
        df = modules.load_dataset()
        assert total == len(df), f"Train+test={total} != dataset size={len(df)}"


# ── 5. Prediction ─────────────────────────────────────────────────────────────

class TestPrediction:
    def test_each_model_predicts_valid_class(self, trained_results):
        class_names = trained_results["class_names"]
        X = modules.prepare_single_input(50, 60, 30, "Under-7")
        for key in config.MODEL_KEYS:
            pipeline = trained_results["pipelines"][key]
            result = modules.predict(pipeline, X, class_names)
            assert result["predicted_class"] in class_names, \
                f"{key}: predicted unknown class {result['predicted_class']}"

    def test_prediction_label_matches_class(self, trained_results):
        X = modules.prepare_single_input(10, 10, 5, "Above-7")
        class_names = trained_results["class_names"]
        pipeline = trained_results["pipelines"]["decision_tree"]
        result = modules.predict(pipeline, X, class_names)
        assert result["predicted_label"] == config.CLASS_LABELS[result["predicted_class"]]

    def test_proba_sums_to_one_for_rf(self, trained_results):
        X = modules.prepare_single_input(70, 80, 50, "Under-7")
        class_names = trained_results["class_names"]
        pipeline = trained_results["pipelines"]["random_forest"]
        result = modules.predict(pipeline, X, class_names)
        assert result["probabilities"], "Random Forest should provide probabilities"
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, expected 1.0"

    def test_high_engagement_tends_towards_high(self, trained_results):
        """
        Students with very high engagement metrics should most often be
        predicted Medium or High. This is a statistical tendency, not a guarantee.
        We assert that at least 3 of 5 models agree the student is not Low.
        """
        X = modules.prepare_single_input(90, 95, 85, "Under-7")
        class_names = trained_results["class_names"]
        not_low = 0
        for key in config.MODEL_KEYS:
            pipeline = trained_results["pipelines"][key]
            result = modules.predict(pipeline, X, class_names)
            if result["predicted_class"] != "L":
                not_low += 1
        assert not_low >= 3, "Expected most models to predict not-Low for very high engagement student"

    def test_low_engagement_tends_towards_low(self, trained_results):
        """
        Students with very low engagement should mostly be predicted Low or Medium.
        """
        X = modules.prepare_single_input(2, 3, 1, "Above-7")
        class_names = trained_results["class_names"]
        not_high = 0
        for key in config.MODEL_KEYS:
            pipeline = trained_results["pipelines"][key]
            result = modules.predict(pipeline, X, class_names)
            if result["predicted_class"] != "H":
                not_high += 1
        assert not_high >= 3, "Expected most models to predict not-High for very low engagement student"


# ── 6. No prohibited frontend files ──────────────────────────────────────────

class TestNoProhibitedFiles:
    PROHIBITED_EXTENSIONS = {".html", ".htm", ".css", ".js"}
    EXCLUDED_DIRS = {".agents", "__pycache__", ".git", "node_modules"}

    def test_no_html_css_js_files_created(self):
        found = []
        for p in PROJECT_ROOT.rglob("*"):
            if any(ex in p.parts for ex in self.EXCLUDED_DIRS):
                continue
            if p.suffix.lower() in self.PROHIBITED_EXTENSIONS:
                found.append(str(p))
        assert not found, f"Prohibited frontend files found:\n" + "\n".join(found)


# ── 7. Config integrity ───────────────────────────────────────────────────────

class TestConfig:
    def test_model_keys_have_display_names(self):
        for k in config.MODEL_KEYS:
            assert k in config.MODEL_DISPLAY_NAMES, f"No display name for {k}"

    def test_class_order_complete(self):
        assert set(config.CLASS_ORDER) == {"L", "M", "H"}

    def test_class_labels_complete(self):
        for cls in config.CLASS_ORDER:
            assert cls in config.CLASS_LABELS

    def test_absence_encoding_values(self):
        enc = config.ABSENCE_ENCODING
        assert set(enc.keys()) == {"Under-7", "Above-7"}
        assert set(enc.values()) == {0, 1}

    def test_secret_key_not_empty(self):
        assert config.SECRET_KEY, "SECRET_KEY must not be empty"

    def test_pass_fail_mapping_defined(self):
        assert "L" in config.PASS_FAIL_MAPPING
        assert "M" in config.PASS_FAIL_MAPPING
        assert "H" in config.PASS_FAIL_MAPPING
        assert config.PASS_FAIL_MAPPING["L"] == "Fail"
        assert config.PASS_FAIL_MAPPING["M"] == "Pass"
        assert config.PASS_FAIL_MAPPING["H"] == "Pass"
        assert len(config.PASS_FAIL_RULE_DESC) > 0


# ── 8. Top 3 models ranking & comparison ──────────────────────────────────────

class TestTop3Models:
    def test_top3_models_ranking(self, trained_results):
        rankings = modules.get_model_rankings(trained_results["metrics"])
        assert len(rankings) == len(config.MODEL_KEYS)
        # Verify 1-based ranks
        assert [r["rank"] for r in rankings] == list(range(1, len(config.MODEL_KEYS) + 1))
        # Verify top 3 flag
        assert sum(1 for r in rankings if r["is_top_3"]) == 3
        # Verify sorting by F1 score primarily
        for i in range(len(rankings) - 1):
            assert rankings[i]["f1"] >= rankings[i + 1]["f1"] or (
                rankings[i]["f1"] == rankings[i + 1]["f1"] and rankings[i]["accuracy"] >= rankings[i + 1]["accuracy"]
            )

    def test_top3_prediction_inference(self, trained_results):
        rankings = modules.get_model_rankings(trained_results["metrics"])
        top_3 = rankings[:3]
        X = modules.prepare_single_input(43, 76, 43, "Under-7")
        class_names = trained_results["class_names"]

        preds = modules.predict_top_models(
            trained_results["pipelines"],
            top_3,
            X,
            class_names,
        )
        assert len(preds) == 3
        for p in preds:
            assert p["predicted_class"] in class_names
            assert p["predicted_label"] in ["Low", "Medium", "High"]
            assert p["pass_fail"] in ["Pass", "Fail"]
            assert p["error"] is None

    def test_top3_charts_generate(self, trained_results):
        rankings = modules.get_model_rankings(trained_results["metrics"])
        top_3 = rankings[:3]
        X = modules.prepare_single_input(50, 60, 40, "Under-7")
        class_names = trained_results["class_names"]
        preds = modules.predict_top_models(trained_results["pipelines"], top_3, X, class_names)

        # Graph A: Categorical Prediction Comparison
        fig_a = modules.chart_top3_prediction_comparison(preds)
        assert fig_a is not None
        assert fig_a.layout.title.text == "Prediction Comparison — Top 3 Models"

        # Graph B: Class Probability Comparison
        fig_b, reason = modules.chart_top3_probability_comparison(preds)
        if all(p.get("has_proba") for p in preds):
            assert fig_b is not None
            assert reason == ""
        else:
            assert fig_b is None
            assert len(reason) > 0

        # Graph C: Metrics Comparison
        fig_c = modules.chart_top3_metrics_comparison(top_3)
        assert fig_c is not None
        assert "Performance Comparison" in fig_c.layout.title.text

    def test_top3_chart_a_has_valid_nonempty_bars(self, trained_results):
        """Regression test: Graph A must have non-empty categorical x-axis and traces."""
        rankings = modules.get_model_rankings(trained_results["metrics"])
        top_3 = rankings[:3]
        X = modules.prepare_single_input(43, 76, 43, "Under-7")
        class_names = trained_results["class_names"]
        preds = modules.predict_top_models(trained_results["pipelines"], top_3, X, class_names)

        fig_a = modules.chart_top3_prediction_comparison(preds)
        assert len(fig_a.data) == 3
        for trace in fig_a.data:
            assert len(trace.x) == 1
            assert trace.x[0] is not None
            assert trace.y[0] >= 0
            assert any(lbl in trace.text[0] for lbl in ["Low", "Medium", "High"])
        assert fig_a.layout.xaxis.type == "category"


# ── 9. Improvement recommendations & Dashboard details ────────────────────────

class TestRecommendationsAndDashboard:
    def test_recommendations_for_low_performer(self):
        recs = modules.get_improvement_recommendations(
            raisedhands=15,
            visited_resources=20,
            discussion=10,
            absence_days="Above-7",
            predicted_class="L",
            attendance_pct=60.0,
            study_hours=1.5,
        )
        assert len(recs) >= 3
        titles = [r["title"] for r in recs]
        assert "Reduce Unexcused Absences" in titles
        assert any(r["priority"] == "high" for r in recs)

    def test_recommendations_for_high_performer(self):
        recs = modules.get_improvement_recommendations(
            raisedhands=90,
            visited_resources=95,
            discussion=85,
            absence_days="Under-7",
            predicted_class="H",
            attendance_pct=98.0,
            study_hours=5.5,
        )
        assert any("Maintain" in r["title"] for r in recs)

    def test_default_prediction_panel_initialized(self):
        import app
        assert app._bootstrap_state() is True
        panel = app._build_default_result_panel()
        assert panel is not None
        assert hasattr(panel, "children")

    def test_dashboard_home_includes_details(self):
        import app
        assert app._bootstrap_state() is True
        home = app._tab_home()
        assert home is not None
        assert len(home.children) >= 5
