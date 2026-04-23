"""
Unit tests for src/insurance_ml/features.py

Coverage:
  BiasCorrection:
    - apply() 2-tier: correct factor per tier, returns copy
    - apply() 3-tier: correct factor per tier
    - to_dict() / from_dict() roundtrip (canonical 2-tier and 3-tier)
    - from_dict() legacy format (_bias_var_low etc.)
    - from_dict() BiasCorrectionArtifact wrapper (applied=True and applied=False)
    - from_dict() unknown keys → KeyError
    - __post_init__ validation: zero var, non-finite var, partial 3-tier,
      threshold_low >= threshold_high
    - is_2tier property
    - get_hash() determinism and sensitivity to param changes

  FeatureEngineeringConfig.validate():
    - rejects correlation_threshold outside (0, 1]
    - rejects vif_threshold <= 0
    - rejects polynomial_degree < 1
    - rejects outlier_contamination outside (0, 0.5)
    - rejects bmi_min >= bmi_max
    - rejects age_min >= age_max
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from insurance_ml.features import BiasCorrection, FeatureEngineeringConfig


# ===========================================================================
# BiasCorrection — 2-tier apply
# ===========================================================================


@pytest.mark.unit
class TestBiasCorrectionApply2Tier:
    """Tests for BiasCorrection.apply() in 2-tier mode."""

    def test_low_tier_gets_correct_factor(self, bc_2tier: BiasCorrection) -> None:
        """Predictions below threshold receive exp(var_low / 2) factor."""
        preds = np.array([1000.0, 5000.0, 10000.0])  # all < 15_000 threshold
        result = bc_2tier.apply(y_pred=preds, y_original=preds)

        expected_factor = math.exp(bc_2tier.var_low / 2.0)
        np.testing.assert_allclose(result, preds * expected_factor, rtol=1e-9)

    def test_high_tier_gets_correct_factor(self, bc_2tier: BiasCorrection) -> None:
        """Predictions at/above threshold receive exp(var_high / 2) factor."""
        preds = np.array([15000.0, 25000.0, 50000.0])
        result = bc_2tier.apply(y_pred=preds, y_original=preds)

        expected_factor = math.exp(bc_2tier.var_high / 2.0)
        np.testing.assert_allclose(result, preds * expected_factor, rtol=1e-9)

    def test_mixed_tiers(self, bc_2tier: BiasCorrection) -> None:
        """Low and high tier samples processed correctly in one call."""
        preds = np.array([5000.0, 20000.0])
        result = bc_2tier.apply(y_pred=preds, y_original=preds)

        factor_low = math.exp(bc_2tier.var_low / 2.0)
        factor_high = math.exp(bc_2tier.var_high / 2.0)

        assert math.isclose(result[0], 5000.0 * factor_low, rel_tol=1e-9)
        assert math.isclose(result[1], 20000.0 * factor_high, rel_tol=1e-9)

    def test_returns_copy_not_inplace(self, bc_2tier: BiasCorrection) -> None:
        """apply() must not mutate the input array."""
        preds = np.array([5000.0, 25000.0])
        original = preds.copy()
        bc_2tier.apply(y_pred=preds, y_original=preds)
        np.testing.assert_array_equal(preds, original)

    def test_single_sample_array(self, bc_2tier: BiasCorrection) -> None:
        """Single-element array should not raise."""
        preds = np.array([1234.56])
        result = bc_2tier.apply(y_pred=preds, y_original=preds)
        assert result.shape == (1,)
        assert result[0] > 0

    def test_routing_uses_y_original_not_y_pred(
        self, bc_2tier: BiasCorrection
    ) -> None:
        """
        Tier routing is driven by y_original (ground-truth signal at eval time).
        A prediction that is above threshold but whose y_original is below
        threshold should receive the LOW factor.
        """
        pred = np.array([20000.0])    # above threshold → would normally get high
        y_orig = np.array([5000.0])   # below threshold → should drive low routing

        result = bc_2tier.apply(y_pred=pred, y_original=y_orig)
        factor_low = math.exp(bc_2tier.var_low / 2.0)
        assert math.isclose(result[0], 20000.0 * factor_low, rel_tol=1e-9)

    def test_output_dtype_is_float(self, bc_2tier: BiasCorrection) -> None:
        preds = np.array([1000, 20000], dtype=int)
        result = bc_2tier.apply(y_pred=preds, y_original=preds)
        assert result.dtype == np.float64


# ===========================================================================
# BiasCorrection — 3-tier apply
# ===========================================================================


@pytest.mark.unit
class TestBiasCorrectionApply3Tier:
    def test_low_tier(self, bc_3tier: BiasCorrection) -> None:
        preds = np.array([5000.0])  # < threshold_low (10_000)
        result = bc_3tier.apply(y_pred=preds, y_original=preds)
        expected = 5000.0 * math.exp(bc_3tier.var_low / 2.0)
        assert math.isclose(result[0], expected, rel_tol=1e-9)

    def test_mid_tier(self, bc_3tier: BiasCorrection) -> None:
        preds = np.array([15000.0])  # threshold_low <= x < threshold_high
        result = bc_3tier.apply(y_pred=preds, y_original=preds)
        expected = 15000.0 * math.exp(bc_3tier.var_mid / 2.0)
        assert math.isclose(result[0], expected, rel_tol=1e-9)

    def test_high_tier(self, bc_3tier: BiasCorrection) -> None:
        preds = np.array([25000.0])  # >= threshold_high (20_000)
        result = bc_3tier.apply(y_pred=preds, y_original=preds)
        expected = 25000.0 * math.exp(bc_3tier.var_high / 2.0)
        assert math.isclose(result[0], expected, rel_tol=1e-9)

    def test_all_three_tiers_in_one_call(self, bc_3tier: BiasCorrection) -> None:
        preds = np.array([5000.0, 15000.0, 25000.0])
        result = bc_3tier.apply(y_pred=preds, y_original=preds)

        fl = math.exp(bc_3tier.var_low / 2.0)
        fm = math.exp(bc_3tier.var_mid / 2.0)
        fh = math.exp(bc_3tier.var_high / 2.0)

        assert math.isclose(result[0], 5000.0 * fl, rel_tol=1e-9)
        assert math.isclose(result[1], 15000.0 * fm, rel_tol=1e-9)
        assert math.isclose(result[2], 25000.0 * fh, rel_tol=1e-9)


# ===========================================================================
# BiasCorrection — __post_init__ validation
# ===========================================================================


@pytest.mark.unit
class TestBiasCorrectionPostInit:
    def test_raises_on_zero_var_low(self) -> None:
        with pytest.raises(ValueError, match="var_low"):
            BiasCorrection(var_low=0.0, var_high=0.09, threshold=15000.0)

    def test_raises_on_zero_var_high(self) -> None:
        with pytest.raises(ValueError, match="var_high"):
            BiasCorrection(var_low=0.04, var_high=0.0, threshold=15000.0)

    def test_raises_on_infinite_var_low(self) -> None:
        with pytest.raises(ValueError, match="var_low"):
            BiasCorrection(var_low=float("inf"), var_high=0.09, threshold=15000.0)

    def test_raises_on_nan_var_high(self) -> None:
        with pytest.raises(ValueError, match="var_high"):
            BiasCorrection(var_low=0.04, var_high=float("nan"), threshold=15000.0)

    def test_negative_var_is_valid(self) -> None:
        """Negative variance encodes a downward correction (model over-predicts)."""
        bc = BiasCorrection(var_low=-0.04, var_high=-0.09, threshold=15000.0)
        assert bc.var_low < 0
        assert bc.var_high < 0

    def test_raises_when_only_some_3tier_fields_set(self) -> None:
        """All three 3-tier fields must be provided together."""
        with pytest.raises(ValueError, match="3-tier"):
            BiasCorrection(
                var_low=0.04,
                var_high=0.09,
                threshold=0.0,
                var_mid=0.06,
                # threshold_low and threshold_high omitted
            )

    def test_raises_when_threshold_low_gte_threshold_high(self) -> None:
        with pytest.raises(ValueError, match="threshold_low"):
            BiasCorrection(
                var_low=0.04,
                var_high=0.09,
                threshold=0.0,
                var_mid=0.06,
                threshold_low=20000.0,
                threshold_high=10000.0,  # inverted
            )

    def test_raises_on_zero_var_mid_in_3tier(self) -> None:
        with pytest.raises(ValueError, match="var_mid"):
            BiasCorrection(
                var_low=0.04,
                var_high=0.09,
                threshold=0.0,
                var_mid=0.0,
                threshold_low=10000.0,
                threshold_high=20000.0,
            )

    def test_valid_3tier_construction(self) -> None:
        bc = BiasCorrection(
            var_low=0.04,
            var_high=0.09,
            threshold=0.0,
            var_mid=0.06,
            threshold_low=10000.0,
            threshold_high=20000.0,
        )
        assert not bc.is_2tier


# ===========================================================================
# BiasCorrection — is_2tier property
# ===========================================================================


@pytest.mark.unit
class TestBiasCorrectionIs2Tier:
    def test_true_when_var_mid_none(self, bc_2tier: BiasCorrection) -> None:
        assert bc_2tier.is_2tier is True

    def test_false_when_var_mid_set(self, bc_3tier: BiasCorrection) -> None:
        assert bc_3tier.is_2tier is False


# ===========================================================================
# BiasCorrection — to_dict / from_dict roundtrip
# ===========================================================================


@pytest.mark.unit
class TestBiasCorrectionSerialization:
    def test_2tier_to_dict_keys(self, bc_2tier: BiasCorrection) -> None:
        d = bc_2tier.to_dict()
        assert set(d.keys()) == {"var_low", "var_high", "threshold"}

    def test_2tier_to_dict_values(self, bc_2tier: BiasCorrection) -> None:
        d = bc_2tier.to_dict()
        assert d["var_low"] == bc_2tier.var_low
        assert d["var_high"] == bc_2tier.var_high
        assert d["threshold"] == bc_2tier.threshold

    def test_3tier_to_dict_includes_extra_fields(
        self, bc_3tier: BiasCorrection
    ) -> None:
        d = bc_3tier.to_dict()
        assert "var_mid" in d
        assert "threshold_low" in d
        assert "threshold_high" in d

    def test_2tier_roundtrip(self, bc_2tier: BiasCorrection) -> None:
        restored = BiasCorrection.from_dict(bc_2tier.to_dict())
        assert restored.var_low == bc_2tier.var_low
        assert restored.var_high == bc_2tier.var_high
        assert restored.threshold == bc_2tier.threshold
        assert restored.is_2tier

    def test_3tier_roundtrip(self, bc_3tier: BiasCorrection) -> None:
        restored = BiasCorrection.from_dict(bc_3tier.to_dict())
        assert restored.var_mid == bc_3tier.var_mid
        assert restored.threshold_low == bc_3tier.threshold_low
        assert restored.threshold_high == bc_3tier.threshold_high
        assert not restored.is_2tier

    def test_small_positive_var_preserved_in_serialization(self) -> None:
        """
        Regression: to_dict() must NOT drop fields with small positive values
        (old bug used truthiness checks like `if self.var_mid:` which silently
        dropped small-but-valid values).
        """
        bc = BiasCorrection(var_low=0.0001, var_high=0.0002, threshold=500.0)
        d = bc.to_dict()
        assert d["var_low"] == 0.0001
        assert d["var_high"] == 0.0002
        assert d["threshold"] == 500.0

    def test_from_dict_canonical_format(self) -> None:
        data = {"var_low": 0.04, "var_high": 0.09, "threshold": 15000.0}
        bc = BiasCorrection.from_dict(data)
        assert bc.var_low == 0.04
        assert bc.var_high == 0.09
        assert bc.threshold == 15000.0
        assert bc.is_2tier

    def test_from_dict_legacy_format(self) -> None:
        data = {
            "_bias_var_low": 0.04,
            "_bias_var_high": 0.09,
            "_bias_threshold": 12000.0,
        }
        bc = BiasCorrection.from_dict(data)
        assert bc.var_low == 0.04
        assert bc.threshold == 12000.0

    def test_from_dict_artifact_wrapper_applied_true(self) -> None:
        """BiasCorrectionArtifact wrapper with applied=True unwraps correctly."""
        data = {
            "applied": True,
            "reason": "model has log1p transform",
            "correction_params": {
                "var_low": 0.04,
                "var_high": 0.09,
                "threshold": 15000.0,
            },
        }
        bc = BiasCorrection.from_dict(data)
        assert bc is not None
        assert bc.var_low == 0.04

    def test_from_dict_artifact_wrapper_applied_false_returns_none(self) -> None:
        """BiasCorrectionArtifact with applied=False (quantile model) → None."""
        data = {
            "applied": False,
            "reason": "quantile model — no correction needed",
            "correction_params": None,
        }
        result = BiasCorrection.from_dict(data)
        assert result is None

    def test_from_dict_unknown_keys_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            BiasCorrection.from_dict({"unknown_key": 1.0})

    def test_from_dict_artifact_applied_true_empty_params_raises(self) -> None:
        data = {"applied": True, "reason": "x", "correction_params": {}}
        with pytest.raises(ValueError, match="correction_params"):
            BiasCorrection.from_dict(data)


# ===========================================================================
# BiasCorrection — get_hash
# ===========================================================================


@pytest.mark.unit
class TestBiasCorrectionGetHash:
    def test_hash_is_deterministic(self, bc_2tier: BiasCorrection) -> None:
        assert bc_2tier.get_hash() == bc_2tier.get_hash()

    def test_hash_differs_with_different_var_low(self) -> None:
        bc_a = BiasCorrection(var_low=0.04, var_high=0.09, threshold=15000.0)
        bc_b = BiasCorrection(var_low=0.05, var_high=0.09, threshold=15000.0)
        assert bc_a.get_hash() != bc_b.get_hash()

    def test_hash_differs_with_different_threshold(self) -> None:
        bc_a = BiasCorrection(var_low=0.04, var_high=0.09, threshold=15000.0)
        bc_b = BiasCorrection(var_low=0.04, var_high=0.09, threshold=10000.0)
        assert bc_a.get_hash() != bc_b.get_hash()

    def test_hash_is_string(self, bc_2tier: BiasCorrection) -> None:
        assert isinstance(bc_2tier.get_hash(), str)

    def test_3tier_hash_differs_from_2tier(
        self, bc_2tier: BiasCorrection, bc_3tier: BiasCorrection
    ) -> None:
        assert bc_2tier.get_hash() != bc_3tier.get_hash()


# ===========================================================================
# FeatureEngineeringConfig.validate()
# ===========================================================================


def _make_valid_fe_config() -> FeatureEngineeringConfig:
    """Construct a fully-valid FeatureEngineeringConfig for mutation tests."""
    return FeatureEngineeringConfig(
        smoker_binary_map={"yes": 1, "no": 0},
        smoker_risk_map={"yes": 2, "no": 0},
        variance_threshold=1e-6,
        correlation_threshold=0.90,
        vif_threshold=10.0,
        max_vif_iterations=5,
        use_optimized_vif=True,
        polynomial_degree=2,
        max_polynomial_features=50,
        outlier_contamination=0.05,
        outlier_random_state=42,
        bmi_min=10.0,
        bmi_max=100.0,
        age_min=0.0,
        age_max=120.0,
        enable_performance_logging=False,
        log_memory_usage=False,
    )


@pytest.mark.unit
class TestFeatureEngineeringConfigValidate:
    def test_valid_config_passes(self) -> None:
        _make_valid_fe_config().validate()  # must not raise

    def test_raises_on_correlation_threshold_zero(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.correlation_threshold = 0.0
        with pytest.raises(ValueError, match="correlation_threshold"):
            cfg.validate()

    def test_raises_on_correlation_threshold_above_one(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.correlation_threshold = 1.1
        with pytest.raises(ValueError, match="correlation_threshold"):
            cfg.validate()

    def test_correlation_threshold_of_one_is_valid(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.correlation_threshold = 1.0
        cfg.validate()  # le=1 is valid per docstring: (0, 1]

    def test_raises_on_vif_threshold_zero(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.vif_threshold = 0.0
        with pytest.raises(ValueError, match="vif_threshold"):
            cfg.validate()

    def test_raises_on_vif_threshold_negative(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.vif_threshold = -1.0
        with pytest.raises(ValueError, match="vif_threshold"):
            cfg.validate()

    def test_raises_on_polynomial_degree_zero(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.polynomial_degree = 0
        with pytest.raises(ValueError, match="polynomial_degree"):
            cfg.validate()

    def test_raises_on_outlier_contamination_zero(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.outlier_contamination = 0.0
        with pytest.raises(ValueError, match="outlier_contamination"):
            cfg.validate()

    def test_raises_on_outlier_contamination_point_five(self) -> None:
        """contamination must be strictly < 0.5."""
        cfg = _make_valid_fe_config()
        cfg.outlier_contamination = 0.5
        with pytest.raises(ValueError, match="outlier_contamination"):
            cfg.validate()

    def test_raises_on_bmi_min_gte_bmi_max(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.bmi_min = 50.0
        cfg.bmi_max = 50.0
        with pytest.raises(ValueError, match="bmi_min"):
            cfg.validate()

    def test_raises_on_age_min_gte_age_max(self) -> None:
        cfg = _make_valid_fe_config()
        cfg.age_min = 100.0
        cfg.age_max = 50.0
        with pytest.raises(ValueError, match="age_min"):
            cfg.validate()
