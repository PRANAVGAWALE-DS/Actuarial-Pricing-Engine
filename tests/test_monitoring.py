"""
Unit tests for src/insurance_ml/monitoring.py

Coverage:
  DriftMonitor.create_baseline:
    - writes JSON with correct top-level structure
    - records numeric and categorical feature stats
    - raises FileExistsError without overwrite=True
    - overwrites silently with overwrite=True

  DriftMonitor.detect_drift:
    - raises FileNotFoundError on missing baseline
    - returns empty DriftReport (no drift) on identical data
    - flags drifted numeric feature when mean shifts significantly
    - flags drifted categorical feature when distribution shifts
    - warns and returns empty report when n_new < 2
    - records missing feature in report.missing_features

  DriftMonitor.compare_batches:
    - returns (DriftReport, dict) tuple
    - detects drift between two very different batches

  DriftReport:
    - has_drift False when drifted_features is empty
    - has_drift True when drifted_features is non-empty
    - to_dict contains all required keys
    - summary() returns a non-empty string

  Private helpers:
    - _tvd: identical distributions → 0.0
    - _tvd: completely disjoint distributions → 0.5
    - _z_score: identical means → 0.0
    - _z_score: large shift → high z-score
    - _se_of_mean: correct formula
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insurance_ml.monitoring import (
    DriftMonitor,
    DriftReport,
    _se_of_mean,
    _tvd,
    _z_score,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _make_numeric_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, n).astype(float),
            "bmi": rng.uniform(18.0, 40.0, n),
            "children": rng.integers(0, 5, n).astype(float),
        }
    )


def _make_categorical_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "smoker": rng.choice(["yes", "no"], n),
            "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
        }
    )


def _make_mixed_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    num = _make_numeric_df(n, seed)
    cat = _make_categorical_df(n, seed)
    return pd.concat([num, cat], axis=1)


# ===========================================================================
# DriftMonitor.create_baseline
# ===========================================================================


@pytest.mark.unit
class TestCreateBaseline:
    def test_creates_json_file(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        out = tmp_path / "baseline.json"
        DriftMonitor.create_baseline(
            X_train=sample_df.drop(columns=["charges"]),
            y_train=sample_df["charges"],
            output_path=str(out),
            overwrite=True,
        )
        assert out.exists()

    def test_json_structure_has_required_keys(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        out = tmp_path / "baseline.json"
        baseline = DriftMonitor.create_baseline(
            X_train=sample_df.drop(columns=["charges"]),
            y_train=sample_df["charges"],
            output_path=str(out),
            overwrite=True,
        )
        assert "n_samples" in baseline
        assert "target_stats" in baseline
        assert "feature_stats" in baseline

    def test_target_stats_has_summary_quantiles(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        out = tmp_path / "baseline.json"
        baseline = DriftMonitor.create_baseline(
            X_train=sample_df.drop(columns=["charges"]),
            y_train=sample_df["charges"],
            output_path=str(out),
            overwrite=True,
        )
        ts = baseline["target_stats"]
        for key in ("mean", "std", "min", "max", "q25", "q50", "q75"):
            assert key in ts

    def test_numeric_feature_stats_recorded(self, tmp_path: Path) -> None:
        df = _make_numeric_df(30)
        y = pd.Series(np.ones(30) * 5000.0)
        out = tmp_path / "b.json"
        baseline = DriftMonitor.create_baseline(
            X_train=df, y_train=y, output_path=str(out), overwrite=True
        )
        assert "age" in baseline["feature_stats"]
        age_stats = baseline["feature_stats"]["age"]
        assert age_stats["dtype"] == "numeric"
        for key in ("mean", "std", "min", "max"):
            assert key in age_stats

    def test_categorical_feature_stats_recorded(self, tmp_path: Path) -> None:
        df = _make_categorical_df(30)
        y = pd.Series(np.ones(30) * 5000.0)
        out = tmp_path / "c.json"
        baseline = DriftMonitor.create_baseline(
            X_train=df, y_train=y, output_path=str(out), overwrite=True
        )
        assert "smoker" in baseline["feature_stats"]
        smoker_stats = baseline["feature_stats"]["smoker"]
        assert smoker_stats["dtype"] == "categorical"
        assert "distribution" in smoker_stats

    def test_raises_file_exists_error_without_overwrite(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        out = tmp_path / "baseline.json"
        out.write_text("{}")  # pre-create the file
        with pytest.raises(FileExistsError, match="overwrite=True"):
            DriftMonitor.create_baseline(
                X_train=sample_df.drop(columns=["charges"]),
                y_train=sample_df["charges"],
                output_path=str(out),
                overwrite=False,
            )

    def test_overwrites_with_overwrite_true(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        out = tmp_path / "baseline.json"
        out.write_text('{"old": true}')
        DriftMonitor.create_baseline(
            X_train=sample_df.drop(columns=["charges"]),
            y_train=sample_df["charges"],
            output_path=str(out),
            overwrite=True,
        )
        content = json.loads(out.read_text())
        assert "n_samples" in content  # new baseline written

    def test_n_samples_matches_input(self, tmp_path: Path) -> None:
        df = _make_numeric_df(42)
        y = pd.Series(np.ones(42) * 5000.0)
        out = tmp_path / "b.json"
        baseline = DriftMonitor.create_baseline(
            X_train=df, y_train=y, output_path=str(out), overwrite=True
        )
        assert baseline["n_samples"] == 42


# ===========================================================================
# DriftMonitor.detect_drift
# ===========================================================================


@pytest.mark.unit
class TestDetectDrift:
    def test_raises_file_not_found_on_missing_baseline(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Baseline not found"):
            DriftMonitor.detect_drift(
                X_new=sample_df.drop(columns=["charges"]),
                baseline_path=str(tmp_path / "does_not_exist.json"),
            )

    def test_no_drift_on_identical_data(
        self, drift_baseline_path: Path, sample_df: pd.DataFrame
    ) -> None:
        X = sample_df.drop(columns=["charges"])
        report = DriftMonitor.detect_drift(X_new=X, baseline_path=str(drift_baseline_path))
        assert isinstance(report, DriftReport)
        assert not report.has_drift

    def test_detects_drift_on_age_mean_shift(self, tmp_path: Path) -> None:
        """A batch with age shifted to [200, 250] vs baseline [18, 65] → drift."""
        n = 100
        X_train = _make_numeric_df(n, seed=1)
        y_train = pd.Series(np.ones(n) * 5000.0)
        out = tmp_path / "baseline.json"
        DriftMonitor.create_baseline(
            X_train=X_train, y_train=y_train, output_path=str(out), overwrite=True
        )

        rng = np.random.default_rng(99)
        X_new = pd.DataFrame(
            {
                "age": rng.uniform(200.0, 250.0, n),  # extreme mean shift
                "bmi": X_train["bmi"].values,  # unchanged
                "children": X_train["children"].values,  # unchanged
            }
        )
        report = DriftMonitor.detect_drift(X_new=X_new, baseline_path=str(out))
        assert "age" in report.drifted_features

    def test_no_drift_when_unchanged(self, tmp_path: Path) -> None:
        n = 100
        X_train = _make_numeric_df(n, seed=2)
        y_train = pd.Series(np.ones(n) * 5000.0)
        out = tmp_path / "b.json"
        DriftMonitor.create_baseline(
            X_train=X_train, y_train=y_train, output_path=str(out), overwrite=True
        )
        report = DriftMonitor.detect_drift(X_new=X_train, baseline_path=str(out))
        # Comparing training to itself → no drift at all
        assert len(report.drifted_features) == 0

    def test_detects_categorical_drift(self, tmp_path: Path) -> None:
        """100% smoker in new batch vs ~50% in training → TVD drift."""
        n = 100
        X_train = _make_categorical_df(n, seed=3)
        y_train = pd.Series(np.ones(n) * 5000.0)
        out = tmp_path / "cat_base.json"
        DriftMonitor.create_baseline(
            X_train=X_train, y_train=y_train, output_path=str(out), overwrite=True
        )

        X_new = pd.DataFrame(
            {
                "smoker": ["yes"] * n,  # 100% smokers — extreme shift
                "region": X_train["region"].values,  # unchanged
            }
        )
        report = DriftMonitor.detect_drift(X_new=X_new, baseline_path=str(out))
        assert "smoker" in report.drifted_features

    def test_warns_and_returns_empty_for_single_row(
        self, drift_baseline_path: Path, sample_df: pd.DataFrame
    ) -> None:
        X = sample_df.drop(columns=["charges"]).iloc[:1]
        with pytest.warns(RuntimeWarning, match="n_new >= 2"):
            report = DriftMonitor.detect_drift(X_new=X, baseline_path=str(drift_baseline_path))
        assert len(report.drifted_features) == 0
        assert report.n_new == 1

    def test_missing_feature_recorded(self, tmp_path: Path) -> None:
        n = 50
        X_train = _make_numeric_df(n)
        y_train = pd.Series(np.ones(n) * 5000.0)
        out = tmp_path / "b.json"
        DriftMonitor.create_baseline(
            X_train=X_train, y_train=y_train, output_path=str(out), overwrite=True
        )
        # New batch is missing the 'age' column
        X_new = X_train.drop(columns=["age"])
        report = DriftMonitor.detect_drift(X_new=X_new, baseline_path=str(out))
        assert "age" in report.missing_features

    def test_report_n_new_matches_input_size(
        self, drift_baseline_path: Path, sample_df: pd.DataFrame
    ) -> None:
        X = sample_df.drop(columns=["charges"])
        report = DriftMonitor.detect_drift(X_new=X, baseline_path=str(drift_baseline_path))
        assert report.n_new == len(X)


# ===========================================================================
# DriftMonitor.compare_batches
# ===========================================================================


@pytest.mark.unit
class TestCompareBatches:
    def test_returns_tuple_of_report_and_dict(self) -> None:
        X_ref = _make_numeric_df(50, seed=10)
        X_new = _make_numeric_df(50, seed=11)
        result = DriftMonitor.compare_batches(X_ref, X_new)
        assert isinstance(result, tuple)
        assert len(result) == 2
        report, baseline_dict = result
        assert isinstance(report, DriftReport)
        assert isinstance(baseline_dict, dict)

    def test_detects_drift_between_extreme_batches(self) -> None:
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(99)
        X_ref = pd.DataFrame({"age": rng1.uniform(18.0, 30.0, 100)})
        X_new = pd.DataFrame({"age": rng2.uniform(200.0, 300.0, 100)})
        report, _ = DriftMonitor.compare_batches(X_ref, X_new)
        assert "age" in report.drifted_features

    def test_no_drift_on_same_batch(self) -> None:
        X = _make_numeric_df(80)
        report, _ = DriftMonitor.compare_batches(X, X)
        assert not report.has_drift


# ===========================================================================
# DriftReport
# ===========================================================================


@pytest.mark.unit
class TestDriftReport:
    def test_has_drift_false_when_empty(self) -> None:
        r = DriftReport()
        assert r.has_drift is False

    def test_has_drift_true_when_features_listed(self) -> None:
        r = DriftReport(drifted_features=["age"])
        assert r.has_drift is True

    def test_to_dict_contains_all_keys(self) -> None:
        r = DriftReport(drifted_features=["age"], drift_scores={"age": 3.5}, n_new=50)
        d = r.to_dict()
        for key in (
            "drifted_features",
            "drift_scores",
            "n_new",
            "z_threshold",
            "tvd_threshold",
            "missing_features",
            "has_drift",
        ):
            assert key in d, f"Missing key: {key}"

    def test_summary_returns_non_empty_string(self) -> None:
        r = DriftReport(
            drifted_features=["age"],
            drift_scores={"age": 3.5},
            n_new=100,
        )
        s = r.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "age" in s


# ===========================================================================
# Private helpers
# ===========================================================================


@pytest.mark.unit
class TestTVD:
    def test_identical_distributions_tvd_is_zero(self) -> None:
        d = {"yes": 0.5, "no": 0.5}
        assert _tvd(d, d) == 0.0

    def test_completely_disjoint_distributions_tvd_is_one(self) -> None:
        """
        TVD = 0.5 × Σ|p(k) - q(k)|.
        For two disjoint single-category distributions:
          Σ|p-q| = |1-0| + |0-1| = 2  →  TVD = 2/2 = 1.0 (maximum possible TVD).
        """
        d_a = {"yes": 1.0}
        d_b = {"no": 1.0}
        assert abs(_tvd(d_a, d_b) - 1.0) < 1e-9

    def test_partial_shift_tvd_is_correct(self) -> None:
        """
        d_a = {yes:0.8, no:0.2}, d_b = {yes:0.2, no:0.8}
        TVD = 0.5 × (|0.8-0.2| + |0.2-0.8|) = 0.5 × 1.2 = 0.6
        """
        d_a = {"yes": 0.8, "no": 0.2}
        d_b = {"yes": 0.2, "no": 0.8}
        result = _tvd(d_a, d_b)
        assert abs(result - 0.6) < 1e-9

    def test_missing_category_treated_as_zero(self) -> None:
        # d_b doesn't have 'maybe'; should use 0.0 for it
        d_a = {"yes": 0.5, "no": 0.5}
        d_b = {"yes": 0.5, "no": 0.3, "maybe": 0.2}
        result = _tvd(d_a, d_b)
        assert result > 0.0


@pytest.mark.unit
class TestZScore:
    def test_identical_means_z_is_zero(self) -> None:
        z = _z_score(new_mean=5.0, baseline_mean=5.0, se=1.0)
        assert z == 0.0

    def test_large_shift_produces_high_z(self) -> None:
        z = _z_score(new_mean=100.0, baseline_mean=5.0, se=0.1)
        assert z > 10.0

    def test_z_is_absolute_value(self) -> None:
        z_pos = _z_score(new_mean=6.0, baseline_mean=5.0, se=1.0)
        z_neg = _z_score(new_mean=4.0, baseline_mean=5.0, se=1.0)
        assert z_pos == z_neg


@pytest.mark.unit
class TestSeOfMean:
    def test_formula_matches(self) -> None:
        """SE = std / sqrt(n)"""
        std = 10.0
        n = 100
        expected = std / (n**0.5)
        assert abs(_se_of_mean(std, n) - expected) < 1e-12

    def test_n_zero_does_not_raise(self) -> None:
        """max(n, 1) guard prevents ZeroDivisionError."""
        result = _se_of_mean(10.0, 0)
        assert result == 10.0  # std / sqrt(1)

    def test_larger_n_produces_smaller_se(self) -> None:
        se_small = _se_of_mean(10.0, 10)
        se_large = _se_of_mean(10.0, 1000)
        assert se_large < se_small
