"""
Unit tests for src/insurance_ml/predict.py

Only tests that do NOT require loading model artifacts from disk are included
here.  Tests that need real model files are marked @pytest.mark.model and
skipped in fast CI.

Coverage (no model files):
  validate_prediction_scale:
    - log scale valid range → (True, message)
    - log scale value > 30 → (False, SCALE ERROR)     [FIX U-03: raised from 20]
    - log scale value < 1 → (False, SCALE WARNING)    [FIX U-03: lowered from 5]
    - original scale valid → (True, message)
    - original scale value < 100 → (False, SCALE ERROR)
    - unknown scale_type → ValueError

  HighValueSegmentRouter (disabled path):
    - enabled=False after failed specialist load
    - route() with enabled=False returns global preds unchanged with correct dict
    - routing diagnostics key "routing_enabled" is False

  HybridPredictor.predict_single input validation (mocked pipeline):
    - age out of range raises ValueError
    - bmi out of range raises ValueError
    - children out of range raises ValueError
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from insurance_ml.predict import validate_prediction_scale


# ===========================================================================
# validate_prediction_scale
# ===========================================================================


@pytest.mark.unit
class TestValidatePredictionScaleLog:
    """Tests for scale_type='log'."""

    def test_valid_log_range_returns_true(self) -> None:
        preds = np.array([8.0, 9.5, 11.2, 12.0])
        valid, msg = validate_prediction_scale(preds, scale_type="log")
        assert valid is True
        assert "correct" in msg.lower() or "LOG" in msg

    def test_value_above_30_returns_false_scale_error(self) -> None:
        # FIX U-03: ABSOLUTE_MAX_LOG raised from 20 → 30 to accommodate YJ range.
        preds = np.array([8.0, 9.5, 31.0])  # max=31 > ABSOLUTE_MAX_LOG=30
        valid, msg = validate_prediction_scale(preds, scale_type="log")
        assert valid is False
        assert "SCALE ERROR" in msg

    def test_value_above_30_message_mentions_original_scale(self) -> None:
        # FIX U-03: threshold is 30.0; must exceed it (not equal) to trigger error.
        preds = np.array([8.0, 31.0])  # max=31 > ABSOLUTE_MAX_LOG=30
        valid, msg = validate_prediction_scale(preds, scale_type="log")
        assert "ORIGINAL" in msg.upper()

    def test_value_below_1_returns_false_scale_warning(self) -> None:
        # FIX U-03: SUSPICIOUS_MIN_LOG lowered from 5.0 → 1.0 to accommodate
        # low-value non-smoker policies in YJ space.
        preds = np.array([0.1, 0.5, 0.9])  # max=0.9 < SUSPICIOUS_MIN_LOG=1.0
        valid, msg = validate_prediction_scale(preds, scale_type="log")
        assert valid is False
        assert "SCALE WARNING" in msg

    def test_boundary_exactly_5_returns_true(self) -> None:
        preds = np.array([5.0, 10.0, 15.0])
        valid, _ = validate_prediction_scale(preds, scale_type="log")
        assert valid is True

    def test_boundary_exactly_20_returns_true(self) -> None:
        preds = np.array([8.0, 20.0])  # max == 20 ≤ ABSOLUTE_MAX_LOG
        valid, _ = validate_prediction_scale(preds, scale_type="log")
        assert valid is True

    def test_single_value_valid(self) -> None:
        preds = np.array([9.5])
        valid, _ = validate_prediction_scale(preds, scale_type="log")
        assert valid is True

    def test_message_includes_range(self) -> None:
        preds = np.array([8.0, 12.0])
        _, msg = validate_prediction_scale(preds, scale_type="log")
        assert "8" in msg or "12" in msg


@pytest.mark.unit
class TestValidatePredictionScaleOriginal:
    """Tests for scale_type='original'."""

    def test_valid_original_scale_returns_true(self) -> None:
        preds = np.array([1500.0, 25000.0, 60000.0])
        valid, msg = validate_prediction_scale(preds, scale_type="original")
        assert valid is True

    def test_value_below_100_returns_false_scale_error(self) -> None:
        preds = np.array([8.5, 9.0, 11.0])  # looks like log space
        valid, msg = validate_prediction_scale(preds, scale_type="original")
        assert valid is False
        assert "SCALE ERROR" in msg

    def test_boundary_exactly_100_returns_true(self) -> None:
        preds = np.array([100.0, 5000.0])
        valid, _ = validate_prediction_scale(preds, scale_type="original")
        assert valid is True

    def test_message_includes_dollar_amounts(self) -> None:
        preds = np.array([5000.0, 20000.0])
        _, msg = validate_prediction_scale(preds, scale_type="original")
        assert "$" in msg


@pytest.mark.unit
class TestValidatePredictionScaleUnknown:
    def test_unknown_scale_type_raises_value_error(self) -> None:
        preds = np.array([8.0, 10.0])
        with pytest.raises(ValueError, match="Unknown scale_type"):
            validate_prediction_scale(preds, scale_type="unknown")

    def test_method_parameter_echoed_in_log_message(self) -> None:
        # FIX U-03: ABSOLUTE_MAX_LOG=30; use 31.0 to exceed it.
        preds = np.array([31.0])  # > ABSOLUTE_MAX_LOG=30
        valid, msg = validate_prediction_scale(
            preds, scale_type="log", method="log1p"
        )
        assert valid is False
        assert "log1p" in msg


# ===========================================================================
# HighValueSegmentRouter — disabled path (no model files needed)
# ===========================================================================


@pytest.mark.unit
class TestHighValueSegmentRouterDisabled:
    """
    When the specialist model cannot be loaded, HighValueSegmentRouter sets
    self.enabled = False.  All calls to route() should be no-ops that return
    the global predictions unchanged.
    """

    def _make_router_disabled(self):
        """
        Construct a HighValueSegmentRouter in disabled state without hitting
        the filesystem.  We patch ModelManager.load_model to raise FileNotFoundError
        so the graceful-degradation path is exercised.
        """
        from insurance_ml.predict import HighValueSegmentRouter, PredictionPipeline
        from insurance_ml.models import ModelManager

        # Build a minimal mock for PredictionPipeline that has a model_manager
        mock_pipeline = MagicMock(spec=PredictionPipeline)
        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.load_model.side_effect = FileNotFoundError("not found")
        mock_pipeline.model_manager = mock_mm

        router = HighValueSegmentRouter(global_pipeline=mock_pipeline)
        return router

    def test_enabled_is_false_when_specialist_absent(self) -> None:
        router = self._make_router_disabled()
        assert router.enabled is False

    def test_route_returns_global_preds_unchanged(self) -> None:
        router = self._make_router_disabled()
        global_preds = np.array([1000.0, 5000.0, 25000.0])
        mock_fe = MagicMock()

        returned_preds, diagnostics = router.route(
            processed_input=MagicMock(),
            global_preds_original=global_preds,
            feature_engineer=mock_fe,
        )
        np.testing.assert_array_equal(returned_preds, global_preds)

    def test_route_diagnostics_routing_enabled_is_false(self) -> None:
        router = self._make_router_disabled()
        global_preds = np.array([5000.0])
        _, diagnostics = router.route(
            processed_input=MagicMock(),
            global_preds_original=global_preds,
            feature_engineer=MagicMock(),
        )
        assert diagnostics.get("routing_enabled") is False

    def test_route_does_not_invoke_model_manager(self) -> None:
        router = self._make_router_disabled()
        global_preds = np.array([5000.0, 20000.0])
        router.route(
            processed_input=MagicMock(),
            global_preds_original=global_preds,
            feature_engineer=MagicMock(),
        )
        # Specialist model should never be called on a disabled router
        assert router.specialist_model is None

    def test_blend_bounds_computed_correctly(self) -> None:
        """
        Even when disabled, the router's internal blend bounds should follow
        the documented formula:
          _lower = threshold * BLEND_LOWER_FACTOR
          _upper = threshold * BLEND_UPPER_FACTOR
        """
        from insurance_ml.predict import HighValueSegmentRouter, PredictionPipeline
        from insurance_ml.models import ModelManager

        mock_pipeline = MagicMock(spec=PredictionPipeline)
        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.load_model.side_effect = FileNotFoundError("not found")
        mock_pipeline.model_manager = mock_mm

        threshold = 20_000.0
        router = HighValueSegmentRouter(
            global_pipeline=mock_pipeline, threshold=threshold
        )

        expected_lower = threshold * HighValueSegmentRouter.BLEND_LOWER_FACTOR
        expected_upper = threshold * HighValueSegmentRouter.BLEND_UPPER_FACTOR
        assert router._lower == pytest.approx(expected_lower)
        assert router._upper == pytest.approx(expected_upper)


# ===========================================================================
# HybridPredictor.predict_single — input validation (mocked pipeline)
# ===========================================================================


@pytest.mark.unit
class TestHybridPredictorInputValidation:
    """
    predict_single() validates age/bmi/children against config bounds BEFORE
    calling predict().  These tests mock out the entire PredictionPipeline so
    no model files are needed.
    """

    def _make_hybrid(self, age_min=18, age_max=120, bmi_min=10.0, bmi_max=100.0):
        from insurance_ml.predict import HybridPredictor, PredictionPipeline

        mock_pipeline = MagicMock(spec=PredictionPipeline)
        mock_pipeline.model_name = "xgboost_test"
        mock_pipeline.model = MagicMock()
        mock_pipeline.model.get_xgb_params = MagicMock(
            return_value={"objective": "reg:squarederror"}
        )

        config = {
            "threshold": 4500.0,
            "blend_ratio": 0.70,
            "use_soft_blending": False,
            "soft_blend_window": 500.0,
            "max_actuarial_uplift_ratio": 1.15,
            "calibration": {"enabled": False, "factor": 1.00, "apply_to_ml_only": True},
            "business_config": {},
        }
        features_config = {
            "age_min": age_min,
            "age_max": age_max,
            "bmi_min": bmi_min,
            "bmi_max": bmi_max,
            "children_min": 0,
            "children_max": 20,
        }
        # Inject features config into the pipeline config so predict_single
        # can read bounds.
        mock_pipeline.config = {"features": features_config}

        hybrid = HybridPredictor.__new__(HybridPredictor)
        hybrid.ml_predictor = mock_pipeline
        hybrid.config = config
        hybrid.threshold = 4500.0
        hybrid.blend_ratio = 0.70
        hybrid.use_soft_blending = False
        hybrid.soft_blend_window = 500.0
        hybrid.calibration_enabled = False
        hybrid.calibration_factor = 1.00
        hybrid.calibration_apply_to_ml_only = True
        hybrid.actuarial_params = {}
        hybrid.business_config = {}
        hybrid._drift_monitor_enabled = False
        hybrid._segment_router = None

        return hybrid

    def test_age_above_max_raises_value_error(self) -> None:
        hybrid = self._make_hybrid(age_max=120)
        with pytest.raises(ValueError, match="Age"):
            hybrid.predict_single(age=150, sex="male", bmi=25.0, children=0,
                                  smoker="no", region="northeast")

    def test_age_below_min_raises_value_error(self) -> None:
        hybrid = self._make_hybrid(age_min=18)
        with pytest.raises(ValueError, match="Age"):
            hybrid.predict_single(age=17, sex="male", bmi=25.0, children=0,
                                  smoker="no", region="northeast")

    def test_bmi_above_max_raises_value_error(self) -> None:
        hybrid = self._make_hybrid(bmi_max=100.0)
        with pytest.raises(ValueError, match="BMI"):
            hybrid.predict_single(age=35, sex="male", bmi=101.0, children=0,
                                  smoker="no", region="northeast")

    def test_bmi_below_min_raises_value_error(self) -> None:
        hybrid = self._make_hybrid(bmi_min=10.0)
        with pytest.raises(ValueError, match="BMI"):
            hybrid.predict_single(age=35, sex="male", bmi=9.0, children=0,
                                  smoker="no", region="northeast")

    def test_children_above_max_raises_value_error(self) -> None:
        hybrid = self._make_hybrid()
        with pytest.raises(ValueError, match="Children"):
            hybrid.predict_single(age=35, sex="male", bmi=25.0, children=25,
                                  smoker="no", region="northeast")
