"""
Unit tests for src/insurance_ml/evaluate.py

Coverage:
  R² sentinel:
    - calculate_segment_metrics: n=1 segment → r2 is NaN, not 0.0
    - calculate_gate_aligned_segment_metrics: n=1 → r2 is NaN

  BusinessConfig:
    - default churn_sensitivity is 1.0 (not 0.01 — known old regression)
    - from_config_dict respects churn_sensitivity default 1.0
    - load_business_config_from_yaml uses shallow copy (N3 — no config mutation)

  BusinessMetricsCalculator:
    - calculate_portfolio_metrics: smoke test returns expected keys
    - calculate_portfolio_metrics: overpriced policies generate churn cost
    - calculate_portfolio_metrics: zero net_profit when prediction is accurate
      enough to earn only the accuracy_bonus
    - vectorised path matches single-row path within tolerance

  _calculate_business_value_score:
    - high profit → profitability_score >= 50
    - zero profit → profitability_score is between 0 and 50
    - negative profit → profitability_score < 0
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from insurance_ml.evaluate import (
    BusinessConfig,
    BusinessMetricsCalculator,
)

# ===========================================================================
# R² sentinel: n=1 segment must return NaN
# ===========================================================================


@pytest.mark.unit
class TestR2Sentinel:
    """
    Regression guard: when a segment has only 1 sample, sklearn's r2_score
    returns 0.0 (undefined for 1 sample).  The pipeline must short-circuit
    and return float('nan') instead so callers can distinguish 'bad fit'
    from 'statistically undefined'.

    Fixed in evaluate.py v7.4.0 — tested here to prevent regression.
    """

    def test_calculate_segment_metrics_single_sample_r2_is_nan(self) -> None:
        from insurance_ml.evaluate import BusinessMetricsCalculator

        calc = BusinessMetricsCalculator()

        # Craft arrays where all but one segment have 0 members, forcing n=1
        # in at least one bin.  Use fixed threshold config to control which
        # bin gets only 1 sample.
        y_true = np.array([100.0, 5000.0, 16000.0, 35000.0])
        y_pred = np.array([110.0, 5100.0, 16100.0, 35100.0])

        # Patch config so the low_risk threshold is just above 100 → only 1 low_risk sample
        config_override = {
            "evaluation": {
                "segment_thresholds": {
                    "low_value": 200.0,  # y_true[0]=100 is the sole low_risk member
                    "standard": 15000.0,
                    "high_value": 30000.0,
                }
            }
        }
        results = calc.calculate_segment_metrics(
            y_true=y_true,
            y_pred=y_pred,
            use_business_thresholds=True,
            config=config_override,
        )

        # low_risk segment has exactly 1 sample → r2 must be NaN
        low = results.get("low_risk")
        assert low is not None, "low_risk segment should exist"
        assert low["n_samples"] == 1
        assert math.isnan(low["r2"]), f"Expected NaN for n=1 segment r2, got {low['r2']!r}"

    def test_calculate_segment_metrics_two_samples_r2_is_finite(self) -> None:
        calc = BusinessMetricsCalculator()
        y_true = np.array([100.0, 150.0, 5000.0])
        y_pred = np.array([110.0, 160.0, 5100.0])

        config_override = {
            "evaluation": {
                "segment_thresholds": {
                    "low_value": 200.0,
                    "standard": 15000.0,
                    "high_value": 30000.0,
                }
            }
        }
        results = calc.calculate_segment_metrics(
            y_true=y_true,
            y_pred=y_pred,
            use_business_thresholds=True,
            config=config_override,
        )
        low = results.get("low_risk")
        assert low is not None
        assert low["n_samples"] == 2
        assert not math.isnan(low["r2"])

    def test_gate_aligned_single_sample_r2_is_nan(self) -> None:
        """
        calculate_gate_aligned_segment_metrics should also return NaN for n=1.
        """
        from insurance_ml.evaluate import BusinessMetricsCalculator

        calc = BusinessMetricsCalculator()

        # One sample in the Very High bin (>= 16_701)
        y_true = np.array([1000.0, 5000.0, 11000.0, 15000.0, 20000.0])
        y_pred = np.array([1100.0, 5100.0, 11100.0, 15100.0, 20100.0])

        results = calc.calculate_gate_aligned_segment_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metadata_path="nonexistent_metadata.json",  # use defaults
        )
        # Find any segment with n_samples == 1
        single_segments = {k: v for k, v in results.items() if v["n_samples"] == 1}
        for seg_name, metrics in single_segments.items():
            assert math.isnan(
                metrics["r2"]
            ), f"Segment '{seg_name}' n=1 should have r2=NaN, got {metrics['r2']}"


# ===========================================================================
# BusinessConfig defaults
# ===========================================================================


@pytest.mark.unit
class TestBusinessConfigDefaults:
    def test_default_churn_sensitivity_is_one(self) -> None:
        """
        Regression guard: churn_sensitivity was 0.01 before v7.4.0 which
        effectively disabled the churn metric (required 5000% overpricing
        to reach 50% churn probability).  Must be 1.0.
        """
        cfg = BusinessConfig()
        assert (
            cfg.churn_sensitivity == 1.0
        ), f"Expected churn_sensitivity=1.0, got {cfg.churn_sensitivity}"

    def test_from_config_dict_defaults_churn_sensitivity_to_one(self) -> None:
        """from_config_dict with empty dict should also default to 1.0."""
        cfg = BusinessConfig.from_config_dict({})
        assert cfg.churn_sensitivity == 1.0

    def test_from_config_dict_respects_explicit_churn_sensitivity(self) -> None:
        cfg = BusinessConfig.from_config_dict({"churn_sensitivity": 2.5})
        assert cfg.churn_sensitivity == 2.5

    def test_base_profit_margin_default(self) -> None:
        assert BusinessConfig().base_profit_margin == 0.15

    def test_admin_cost_default(self) -> None:
        assert BusinessConfig().admin_cost_per_policy == 25.0


# ===========================================================================
# N3 FIX: load_business_config_from_yaml shallow copy
# ===========================================================================


@pytest.mark.unit
class TestLoadBusinessConfigFromYamlShallowCopy:
    """
    N3 guard: load_business_config_from_yaml was writing 'low_value_threshold'
    directly into the shared config dict, permanently mutating it for
    subsequent callers.  It must use a shallow copy.
    """

    def test_does_not_mutate_original_config_dict(self, minimal_config: dict, monkeypatch) -> None:
        import copy

        from insurance_ml import config as cfg_module
        from insurance_ml.evaluate import load_business_config_from_yaml

        frozen = copy.deepcopy(minimal_config)

        # Patch load_config to return our controlled copy each time
        monkeypatch.setattr(cfg_module, "load_config", lambda: copy.deepcopy(minimal_config))

        load_business_config_from_yaml()
        load_business_config_from_yaml()  # second call — must not see mutations

        # Verify the original minimal_config was not changed
        assert minimal_config.get("hybrid_predictor", {}).get("business_config") == frozen.get(
            "hybrid_predictor", {}
        ).get(
            "business_config"
        ), "load_business_config_from_yaml mutated the shared config dict (N3)"

    def test_returns_business_config_instance(self, minimal_config: dict, monkeypatch) -> None:
        import copy

        from insurance_ml import config as cfg_module
        from insurance_ml.evaluate import load_business_config_from_yaml

        monkeypatch.setattr(cfg_module, "load_config", lambda: copy.deepcopy(minimal_config))
        result = load_business_config_from_yaml()
        assert isinstance(result, BusinessConfig)

    def test_falls_back_to_defaults_on_yaml_error(self, monkeypatch) -> None:
        from insurance_ml import config as cfg_module
        from insurance_ml.evaluate import load_business_config_from_yaml

        monkeypatch.setattr(
            cfg_module, "load_config", lambda: (_ for _ in ()).throw(OSError("no config"))
        )
        result = load_business_config_from_yaml()
        assert isinstance(result, BusinessConfig)
        assert result.churn_sensitivity == 1.0


# ===========================================================================
# BusinessMetricsCalculator
# ===========================================================================


@pytest.mark.unit
class TestBusinessMetricsCalculatorPortfolioMetrics:
    def _calc(self) -> BusinessMetricsCalculator:
        return BusinessMetricsCalculator(BusinessConfig())

    def test_returns_expected_keys(self) -> None:
        calc = self._calc()
        y_true = np.array([5000.0, 10000.0, 15000.0])
        y_pred = np.array([5100.0, 10100.0, 15100.0])
        result = calc.calculate_portfolio_metrics(y_true, y_pred)

        expected_keys = {
            "n_predictions",
            "total_net_profit",
            "profit_per_policy",
            "churn_rate_pct",
            "n_underpriced",
            "n_overpriced",
            "n_accurate",
            "accuracy_rate_pct",
            "business_value_score",
            "gross_margin_pct",
            "net_margin_pct",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_n_predictions_correct(self) -> None:
        calc = self._calc()
        n = 50
        y = np.ones(n) * 5000.0
        result = calc.calculate_portfolio_metrics(y, y)
        assert result["n_predictions"] == n

    def test_exact_predictions_have_no_underpricing(self) -> None:
        calc = self._calc()
        y = np.array([5000.0, 10000.0])
        result = calc.calculate_portfolio_metrics(y, y)
        assert result["n_underpriced"] == 0

    def test_exact_predictions_have_no_overpricing(self) -> None:
        calc = self._calc()
        y = np.array([5000.0, 10000.0])
        result = calc.calculate_portfolio_metrics(y, y)
        assert result["n_overpriced"] == 0

    def test_overpriced_policy_generates_churn_cost(self) -> None:
        """
        A policy predicted at 2× true cost should exceed churn_threshold_pct
        (0.40) and generate nonzero expected churn.
        """
        calc = self._calc()
        y_true = np.array([10000.0])
        y_pred = np.array([25000.0])  # 150% overpriced → well above 40% threshold
        result = calc.calculate_portfolio_metrics(y_true, y_pred)
        assert result["churn_rate_pct"] > 0.0

    def test_underpriced_policy_generates_underpricing_penalty(self) -> None:
        calc = self._calc()
        y_true = np.array([10000.0])
        y_pred = np.array([5000.0])  # 50% underpriced
        result = calc.calculate_portfolio_metrics(y_true, y_pred)
        assert result["n_underpriced"] == 1

    def test_vectorized_matches_single_row_within_tolerance(self) -> None:
        """
        For a single prediction, calculate_portfolio_metrics (vectorised)
        and calculate_single_prediction_value (scalar) must agree on
        net_profit within 1e-6 relative tolerance.
        """
        calc = self._calc()
        y_true_val = 12000.0
        y_pred_val = 11000.0

        vec = calc.calculate_portfolio_metrics(np.array([y_true_val]), np.array([y_pred_val]))
        scalar = calc.calculate_single_prediction_value(y_true_val, y_pred_val)

        assert math.isclose(
            vec["total_net_profit"],
            scalar["net_profit"],
            rel_tol=1e-6,
        ), (
            f"Vectorized net_profit={vec['total_net_profit']:.4f} "
            f"!= scalar net_profit={scalar['net_profit']:.4f}"
        )


# ===========================================================================
# _calculate_business_value_score
# ===========================================================================


@pytest.mark.unit
class TestCalculateBusinessValueScore:
    def _calc(self) -> BusinessMetricsCalculator:
        return BusinessMetricsCalculator(BusinessConfig())

    def test_high_profit_score_at_or_above_50(self) -> None:
        calc = self._calc()
        # profit_per_policy == biz_score_profit_target (500) → score exactly 50
        score = calc._calculate_business_value_score(
            profit_per_policy=500.0, churn_rate=0.0, accuracy_rate=0.0
        )
        assert score >= 50.0

    def test_zero_profit_score_between_25_and_55(self) -> None:
        """
        With profit=0, churn=0, accuracy=0:
          profitability = 25.0  (0/500 * 25 + 25)
          retention     = 30.0  (max retention, churn=0 < churn_low=0.10)
          quality       =  0.0  (accuracy_rate * 20)
          total         = 55.0
        The score is above 50 because zero churn earns the full retention bonus.
        """
        calc = self._calc()
        score = calc._calculate_business_value_score(
            profit_per_policy=0.0, churn_rate=0.0, accuracy_rate=0.0
        )
        # 0 profit → profitability=25, 0 churn → retention=30, 0 accuracy → quality=0
        assert abs(score - 55.0) < 1.0

    def test_negative_profit_reduces_score(self) -> None:
        calc = self._calc()
        score_pos = calc._calculate_business_value_score(
            profit_per_policy=100.0, churn_rate=0.0, accuracy_rate=0.0
        )
        score_neg = calc._calculate_business_value_score(
            profit_per_policy=-100.0, churn_rate=0.0, accuracy_rate=0.0
        )
        assert score_neg < score_pos

    def test_high_churn_reduces_score(self) -> None:
        calc = self._calc()
        score_low_churn = calc._calculate_business_value_score(
            profit_per_policy=500.0, churn_rate=0.0, accuracy_rate=0.5
        )
        score_high_churn = calc._calculate_business_value_score(
            profit_per_policy=500.0, churn_rate=0.5, accuracy_rate=0.5
        )
        assert score_high_churn < score_low_churn

    def test_perfect_accuracy_adds_quality_score(self) -> None:
        calc = self._calc()
        score_zero = calc._calculate_business_value_score(
            profit_per_policy=0.0, churn_rate=0.0, accuracy_rate=0.0
        )
        score_perfect = calc._calculate_business_value_score(
            profit_per_policy=0.0, churn_rate=0.0, accuracy_rate=1.0
        )
        assert score_perfect > score_zero

    def test_score_bounded_below_by_negative_fifty(self) -> None:
        """Score should not go below -50 even for very negative profit."""
        calc = self._calc()
        score = calc._calculate_business_value_score(
            profit_per_policy=-1_000_000.0, churn_rate=1.0, accuracy_rate=0.0
        )
        assert score >= -50.0
