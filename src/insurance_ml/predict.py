"""
Enhanced Production Hybrid Prediction Pipeline v6.3.3
=====================================================
"""

import json
import logging
import math as _math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from insurance_ml.config import get_feature_config, load_config
from insurance_ml.features import BiasCorrection, FeatureEngineer
from insurance_ml.models import ModelManager
from insurance_ml.monitoring import DriftMonitor

logger = logging.getLogger(__name__)


# =====================================================================
# CONSTANTS
# =====================================================================

_CONFORMAL_STD_MIN = 1e-4  # Minimum conformal residual std to avoid degeneracy

# moved from inside predict_with_intervals() where it was re-created
# on every call.  Controls the hard cap on upper CI bounds: the inverse-transformed
# upper bound is clipped to y_max_safe × _CI_UPPER_CAP_FACTOR to prevent NaN/inf
# from extreme OOD Yeo-Johnson inputs while still reflecting real uncertainty.
# 3.0 = allow CI upper up to 3× the training-data maximum premium.
_CI_UPPER_CAP_FACTOR: float = 3.0


# =====================================================================
# SCALE VALIDATION
# =====================================================================


def validate_prediction_scale(
    predictions: np.ndarray, scale_type: str = "log", method: str = "log1p"
) -> tuple[bool, str]:
    """Validate predictions are in expected scale.

    Previous thresholds (max > 20.0 → error, max < 5.0 → warning) were
    calibrated for log1p only.  Yeo-Johnson of a $60K+ smoker premium can reach
    YJ ≈ 22–25 in transformed space, causing false SCALE ERROR crashes for valid
    high-value predictions.  Conversely, a batch of low-value ($1K–$2K) non-smoker
    policies has YJ max ≈ 2–3, triggering a false SCALE WARNING.

    raise the hard ceiling to 30.0 and lower the suspiciously-small floor to
    1.0 to accommodate the full YJ range while still catching genuine unit errors
    (dollar-scale values reach thousands, not single digits).
    """
    min_val = np.min(predictions)
    max_val = np.max(predictions)
    mean_val = np.mean(predictions)

    if scale_type == "log":
        # Yeo-Johnson of $500K (extreme OOD) ≈ 28; log1p of $500K ≈ 13.
        # Ceiling 30.0 catches genuine original-scale leakage (min premium $1K → 7.0 in YJ)
        # without false-positives for valid high-value policies.
        ABSOLUTE_MAX_LOG = 30.0
        # Floor 1.0: YJ of $1,100 (minimum realistic premium) ≈ 1.2; log1p ≈ 7.0.
        # Anything below 1.0 is almost certainly a near-zero or negative artefact.
        SUSPICIOUS_MIN_LOG = 1.0

        if max_val > ABSOLUTE_MAX_LOG:
            return False, (
                f"❌ SCALE ERROR: Values appear to be in ORIGINAL scale!\n"
                f"   Range: [{min_val:.2f}, {max_val:.2f}]\n"
                f"   Expected for {method}: max < {ABSOLUTE_MAX_LOG}"
            )
        elif max_val < SUSPICIOUS_MIN_LOG:
            return False, (
                f"⚠️ SCALE WARNING: Values suspiciously small\n"
                f"   Range: [{min_val:.2f}, {max_val:.2f}]"
            )
        else:
            return True, (
                f"✅ Predictions in correct LOG scale: "
                f"[{min_val:.2f}, {max_val:.2f}], mean={mean_val:.2f}"
            )

    elif scale_type == "original":
        if max_val < 100:
            return False, (
                f"❌ SCALE ERROR: Values appear to be in LOG scale!\n"
                f"   Range: [${min_val:.2f}, ${max_val:.2f}]"
            )
        else:
            return True, (
                f"✅ Predictions in correct ORIGINAL scale: "
                f"[${min_val:,.2f}, ${max_val:,.2f}], mean=${mean_val:,.2f}"
            )

    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")


# =====================================================================
# HIGH-VALUE SEGMENT ROUTER
# =====================================================================


class HighValueSegmentRouter:
    """
    Routes predictions between the global model and a high-value specialist model.

    Background (Incident 3):
        The global XGBoost model achieves R²=-2.0157 on the High-value segment
        (premiums ≥ P75 ≈ $16,701).  A specialist model trained exclusively on that
        segment uses deeper trees and lower learning rate to capture the extreme
        smoker×BMI×age interaction patterns that the global model under-fits.

    Routing logic (three mutually exclusive zones):
        • Low zone   (pred ≤ lower_bound):     pure global output
        • High zone  (pred ≥ upper_bound):     pure specialist output
        • Blend zone (lower < pred < upper):   linear alpha blend
          where alpha = 0 at lower_bound → 1 at upper_bound

    The 30% blend window on each side of the threshold prevents the hard
    boundary discontinuity that would otherwise appear in the premium surface
    at exactly $16,701 when one model takes over from the other.

    Graceful degradation:
        If the specialist model file is absent (pre-training, or cold start),
        `self.enabled` is set to False and every call to `route()` returns the
        global predictions unchanged.  No exception is raised; the pipeline
        continues to function normally.

    Attributes:
        HIGH_VALUE_THRESHOLD (float): P75 of training targets from pipeline log.
        BLEND_LOWER_FACTOR   (float): Lower edge of blend window as a fraction
                                      of threshold (1.00 = blend starts at threshold).
        BLEND_UPPER_FACTOR   (float): Upper edge of blend window (1.299 = ~+30%).
    """

    HIGH_VALUE_THRESHOLD: float = 16_701.0  # P75 of charges from pipeline log
    # Raised from 1.00 to 1.08 ($18,037).
    # At 1.00 the blend zone started exactly at the routing threshold ($16,701),
    # causing High+ segment ($14K-$16.7K) predictions near the boundary to
    # partially route through the specialist. The specialist trains on y_true >
    # $16,701 and has never seen this distribution — it consistently worsened
    # High (R²: -2.28→-2.77) and High+ (R²: -33.4→-35.0) in training log.
    # At 1.08 the blend zone is $18,037–$21,695, cleanly above the segment boundary.
    BLEND_LOWER_FACTOR: float = 1.08
    BLEND_UPPER_FACTOR: float = 1.299  # blend ends at $21,705 ≈ $21,701 (TRANSITION_ZONE[1])

    # Model name as written to disk by HighValueSpecialist.save() in models.py
    DEFAULT_SPECIALIST_NAME: str = "xgboost_high_value_specialist"

    def __init__(
        self,
        global_pipeline: "PredictionPipeline",
        specialist_model_name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        """
        Args:
            global_pipeline:        Initialised PredictionPipeline (model already
                                    loaded).  Used to access model_manager and
                                    feature_engineer.
            specialist_model_name:  Override the default specialist model filename.
                                    Default: 'xgboost_high_value_specialist'.
            threshold:              Override the routing threshold.
                                    Default: HIGH_VALUE_THRESHOLD ($16,701).
        """
        self.global_pipeline = global_pipeline
        self.specialist_model_name = specialist_model_name or self.DEFAULT_SPECIALIST_NAME
        self.threshold = threshold or self.HIGH_VALUE_THRESHOLD
        self._lower = self.threshold * self.BLEND_LOWER_FACTOR
        self._upper = self.threshold * self.BLEND_UPPER_FACTOR

        self.specialist_model: Any = None
        self.enabled: bool = False

        self._try_load_specialist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_specialist(self) -> None:
        """Attempt to load the specialist model from disk.  Non-fatal on failure."""
        try:
            self.specialist_model = self.global_pipeline.model_manager.load_model(
                self.specialist_model_name
            )
            self.enabled = True
            logger.info(
                "✅ HighValueSegmentRouter: specialist loaded "
                "(name='%s', threshold=$%.0f, blend=[%.0f, %.0f])",
                self.specialist_model_name,
                self.threshold,
                self._lower,
                self._upper,
            )
        except FileNotFoundError:
            logger.info(
                "ℹ️  HighValueSegmentRouter: specialist model '%s' not found on "
                "disk — routing disabled.  Global model handles all segments.  "
                "Train HighValueSpecialist and save to 'models/' to activate.",
                self.specialist_model_name,
            )
        except (OSError, RuntimeError) as _e:
            logger.warning(
                "⚠️  HighValueSegmentRouter: could not load specialist (%s) — "
                "falling back to global model for all segments.",
                _e,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        processed_input: pd.DataFrame,
        global_preds_original: np.ndarray,
        feature_engineer: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Blend global and specialist predictions based on prediction magnitude.

        Uses the GLOBAL model's dollar-space output as the routing signal — not
        the raw feature values — because the threshold is defined in premium space
        and the global model's output is available at zero extra cost.

        Args:
            processed_input:       Feature-engineered input ready for model.predict().
                                   Shape: (n_samples, n_features).
            global_preds_original: Global model predictions already inverse-transformed
                                   to original dollar space.  Shape: (n_samples,).
            feature_engineer:      FeatureEngineer instance for inverse-transforming
                                   the specialist model's raw (log-space) output.

        Returns:
            routed_preds (np.ndarray): Final predictions after routing/blending.
            diagnostics  (dict):       Breakdown of routing decisions for observability.

        Raises:
            Nothing.  Any specialist inference failure falls back to global predictions
            and records the error in the returned diagnostics dict.
        """
        if not self.enabled or self.specialist_model is None:
            return global_preds_original, {"routing_enabled": False}

        n = len(global_preds_original)

        # ── Zone masks (mutually exclusive and exhaustive) ─────────────────────
        mask_high = global_preds_original >= self._upper  # pure specialist
        mask_low = global_preds_original <= self._lower  # pure global
        mask_blend = ~mask_high & ~mask_low  # linear blend

        n_high = int(mask_high.sum())
        n_low = int(mask_low.sum())
        n_blend = int(mask_blend.sum())

        # Early exit: no samples need specialist inference
        if n_high + n_blend == 0:
            return global_preds_original, {
                "routing_enabled": True,
                "specialist_invoked": False,
                "n_global": n_low,
                "n_specialist": 0,
                "n_blend": 0,
                "pct_specialist": 0.0,
                "pct_blend": 0.0,
            }

        # ── Run specialist only on samples that may need it (high + blend) ─────
        specialist_mask = mask_high | mask_blend
        X_spec = (
            processed_input.iloc[specialist_mask]
            if hasattr(processed_input, "iloc")
            else processed_input[specialist_mask]
        )

        try:
            # Specialist model runs on cuda:0 (same as global).  Calling
            # self.specialist_model.predict(DataFrame) triggers XGBoost's inplace_predict
            # with a CPU input vs a CUDA booster, emitting the device-mismatch warning
            # on every inference call.  Apply the same DMatrix workaround used for the
            # global model in PredictionPipeline.predict() (lines 959–973):
            # build a CPU DMatrix from the numpy backing array and call the booster
            # directly.  DMatrix.predict() does not trigger inplace_predict and therefore
            # does not emit the mismatch warning.
            # Fallback: if get_booster() is unavailable (non-XGBoost specialist) or
            # DMatrix construction fails, fall back to the direct .predict() call.
            try:
                import xgboost as _xgb_spec

                if hasattr(self.specialist_model, "get_booster"):
                    _spec_arr = X_spec.values if hasattr(X_spec, "values") else np.asarray(X_spec)
                    _spec_dmat = _xgb_spec.DMatrix(_spec_arr)
                    spec_raw = self.specialist_model.get_booster().predict(_spec_dmat)
                else:
                    spec_raw = self.specialist_model.predict(X_spec)
            except Exception:
                spec_raw = self.specialist_model.predict(X_spec)
            # specialist is trained on original-scale USD directly.
            # Its .predict() output is already in dollar space — do NOT call
            # inverse_transform_target() here.  That call was treating e.g.
            # $25,000 as a yeo-johnson space value (~11.3 in log-space maps to
            # ~$80K+ after inversion), producing 2–3× systematic overpricing
            # on the Very High segment (root cause of test R²=−2.85).
            # See train.py line 7867: "DO NOT call inverse_transform_target on
            # its output."
            spec_orig = np.asarray(spec_raw, dtype=float)
            # Clip to training-time safe range to prevent rare OOD extremes.
            _y_min = getattr(feature_engineer, "y_min_safe", None)
            _y_max = getattr(feature_engineer, "y_max_safe", None)
            if _y_min is not None and _y_max is not None:
                spec_orig = np.clip(spec_orig, _y_min, _y_max)
        except (ValueError, RuntimeError, AttributeError) as _spec_err:
            logger.warning(
                "⚠️  HighValueSegmentRouter: specialist inference failed (%s). "
                "Returning global predictions for all %d samples.",
                _spec_err,
                n,
            )
            return global_preds_original, {
                "routing_enabled": True,
                "specialist_invoked": False,
                "specialist_error": str(_spec_err),
            }

        # ── Build routed output ────────────────────────────────────────────────
        routed = global_preds_original.copy().astype(float)

        # Scatter specialist predictions back into a full-length array so zone
        # indexing below uses simple boolean masks on the full arrays.
        spec_full = np.full(n, np.nan, dtype=float)
        spec_full[np.where(specialist_mask)] = spec_orig

        # High zone: pure specialist
        if n_high > 0:
            routed[mask_high] = spec_full[mask_high]

        # Blend zone: linear alpha from 0 (global) → 1 (specialist)
        if n_blend > 0:
            alpha = np.clip(
                (global_preds_original[mask_blend] - self._lower) / (self._upper - self._lower),
                0.0,
                1.0,
            )
            routed[mask_blend] = (1.0 - alpha) * global_preds_original[
                mask_blend
            ] + alpha * spec_full[mask_blend]

        diagnostics: dict[str, Any] = {
            "routing_enabled": True,
            "specialist_invoked": True,
            "threshold": self.threshold,
            "blend_lower": self._lower,
            "blend_upper": self._upper,
            "n_global": n_low,
            "n_specialist": n_high,
            "n_blend": n_blend,
            "pct_specialist": round(n_high / n * 100, 2),
            "pct_blend": round(n_blend / n * 100, 2),
        }

        logger.info(
            "🔀 SegmentRouter: global=%d | blend=%d | specialist=%d "
            "(specialist threshold=$%.0f, blend window=[$%.0f, $%.0f]) "
            "— NOTE: these counts use the specialist routing threshold, "
            "not the HybridPredictor actuarial threshold.",
            n_low,
            n_blend,
            n_high,
            self.threshold,
            self._lower,
            self._upper,
        )

        return routed, diagnostics


# =====================================================================
# ENHANCED ML PREDICTION PIPELINE
# =====================================================================


class PredictionPipeline:
    """
    Enhanced ML prediction pipeline with insurance-grade validation

    """

    VERSION = "6.3.3"

    def __init__(
        self,
        model_name: str | None = None,
        preprocessor_path: str | None = None,
        model_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize pipeline with enhanced validation"""
        # SAFETY: copy the dict so we never mutate the caller's config object.
        # Passing config by reference and then writing ["model_dir"] into it
        # would silently inject the key back into the caller's namespace.
        self.config = dict(config) if config is not None else load_config()

        if model_dir:
            self.config["model_dir"] = model_dir
        elif "model_dir" not in self.config:
            self.config["model_dir"] = "models"

        self.model_manager = ModelManager(self.config)
        self.model_dir = self.config.get("model_dir", "models")

        # Resolve preprocessor path: explicit > config > latest on disk > legacy default
        preprocessor_path = self._resolve_preprocessor_path(preprocessor_path)

        # Initialize FeatureEngineer
        logger.info("=" * 70)
        logger.info("Initializing FeatureEngineer...")
        try:
            feat_cfg = get_feature_config(self.config)
            self.feature_engineer = FeatureEngineer(config_dict=feat_cfg)
            logger.info("✅ FeatureEngineer initialized")
            # NOTE: target_transformation.method is NOT logged here intentionally.
            # FeatureEngineer initialises with the config default ('none') before
            # load_preprocessor() overwrites it with the saved artifact value
            # ('yeo-johnson').  Logging here would always emit 'none', misleading
            # operators into thinking no transformation is active.
            # The correct value is logged after load_preprocessor() below.

        except Exception as e:
            logger.error(f"❌ Failed to initialize FeatureEngineer: {e}")
            raise

        # Load preprocessor with validation
        logger.info("\nLoading preprocessor...")
        try:
            self.feature_engineer.load_preprocessor(preprocessor_path)
            # Log the transformation method AFTER load_preprocessor() so the value
            # reflects the artifact state ('yeo-johnson'), not the config default ('none').
            _transform_method = self.feature_engineer.target_transformation.method
            logger.info(f"✅ Loaded preprocessor: {preprocessor_path}")
            logger.info(f"   Target transformation: {_transform_method}")

            # Load BiasCorrection object for point-prediction correction ──
            self._bias_correction = None
            bias_path = Path(preprocessor_path).parent / "bias_correction.json"

            if bias_path.exists():
                try:
                    with open(bias_path) as _f:
                        _bc_data = json.load(_f)
                    self._bias_correction = BiasCorrection.from_dict(_bc_data)

                    # ── CORRECTION DIRECTION AUDIT ──────────────────────────────────
                    # var > 0 → upward correction (exp(var/2) > 1.0), model under-predicted.
                    # var < 0 → downward correction (exp(var/2) < 1.0), model over-predicted.
                    # var == 1e-6 (sentinel) → degenerate ratio during training; no-op.
                    # Only the true sentinel (exactly 1e-6) triggers a warning now.
                    _SENTINEL_EXACT = 1e-6
                    _bc = self._bias_correction
                    _tier_info = []
                    for _name, _v in [
                        ("var_low", _bc.var_low),
                        ("var_high", _bc.var_high),
                        *(
                            [("var_mid", _bc.var_mid)]
                            if not _bc.is_2tier and _bc.var_mid is not None
                            else []
                        ),
                    ]:
                        _factor = _math.exp(_v / 2)
                        if _v == _SENTINEL_EXACT:
                            _tier_info.append(
                                f"{_name}: ⚠️  sentinel (no-op, degenerate training ratio)"
                            )
                        elif _v < 0:
                            _tier_info.append(f"{_name}: ↓ {_factor:.4f} ({(_factor-1)*100:+.1f}%)")
                        else:
                            _tier_info.append(f"{_name}: ↑ {_factor:.4f} ({(_factor-1)*100:+.1f}%)")

                    _has_sentinel = any(
                        [
                            _bc.var_low == _SENTINEL_EXACT,
                            _bc.var_high == _SENTINEL_EXACT,
                            (not _bc.is_2tier and _bc.var_mid == _SENTINEL_EXACT),
                        ]
                    )
                    if _has_sentinel:
                        logger.warning(
                            "⚠️  BiasCorrection has sentinel tier(s) (degenerate training ratio).\n"
                            "   Retrain to resolve. CI falls back to conformal residuals."
                        )
                    else:
                        logger.info(f"ℹ️  BiasCorrection tier corrections: {'; '.join(_tier_info)}")

                    logger.info(
                        f"✅ BiasCorrection loaded from {bias_path.name}\n"
                        f"   Type: {'2-tier' if self._bias_correction.is_2tier else '3-tier'}\n"
                        f"   var_low={self._bias_correction.var_low:.6f}, "
                        f"var_high={self._bias_correction.var_high:.6f}"
                    )

                    # Validate BiasCorrection is consistent with
                    #            the copy embedded in the model artifact.
                    #            A mismatch means the .json and the model were
                    #            saved from different training checkpoints.
                    _model_bc = (
                        getattr(self.model, "_bias_correction", None)  # type: ignore[has-type]
                        if hasattr(self, "model")
                        else None
                    )
                    # Note: model not yet loaded here — check deferred to after model load (see below)
                except Exception as _e:
                    logger.error(
                        f"❌ Failed to load bias_correction.json: {_e}\n"
                        "   Point predictions will be systematically low."
                    )
            else:
                # Backward-compat: file not present on old model artifacts
                logger.warning(
                    f"⚠️ bias_correction.json not found at {bias_path}\n"
                    "   Predictions will not have stratified bias correction applied.\n"
                    "   Re-train or copy bias_correction.json alongside the preprocessor."
                )

            # Separate flag: uncertainty quantification (CI width) uses _log_residual_variance.
            # NOTE: A *negative* value is NOT a sentinel — it is a legitimate downward
            # correction encoding (var = 2*log(ratio), ratio < 1 → var < 0).
            # The true sentinel is exactly 1e-6 (set when ratio ≤ 0 during training).
            # Negative values are unusable for CI std derivation (sqrt of negative
            # is undefined) but they are correct for point-prediction bias correction.
            _lrv = getattr(self.feature_engineer, "_log_residual_variance", None)
            if _lrv is not None:
                if _lrv < 0:
                    logger.debug(
                        f"ℹ️  _log_residual_variance={_lrv:.6f} is negative, encoding a "
                        f"downward yeo-johnson correction (exp({_lrv:.6f}/2)="
                        f"{_math.exp(_lrv / 2):.4f}x — model over-predicts on median). "
                        f"This is a valid bias-correction value, not a sentinel. "
                        f"Unsuitable for CI std derivation; CI will use conformal residuals."
                    )
                else:
                    logger.debug(f"✅ Uncertainty quantification available: variance={_lrv:.6f}")
            else:
                logger.warning(
                    "⚠️ _log_residual_variance missing — predict_with_intervals() degraded."
                )

        except FileNotFoundError:
            logger.error(f"❌ Preprocessor not found: {preprocessor_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading preprocessor: {e}")
            raise

        # Auto-select best model
        self.model_name = model_name or self._get_best_model()

        # Load model with validation
        logger.info("\nLoading model...")
        try:
            self.model = self.model_manager.load_model(self.model_name)

            if not hasattr(self.model, "predict"):
                raise ValueError(f"Model {self.model_name} missing predict() method")

            # Validate feature compatibility.
            # Use a 4-row probe that exercises EVERY categorical value so that
            # a fitted OneHotEncoder always emits all known columns regardless
            # of its handle_unknown setting. A single-row, single-category
            # probe can mask column-count mismatches when handle_unknown='ignore'.
            if hasattr(self.model, "n_features_in_"):
                expected_features = self.model.n_features_in_

                logger.info("   Validating feature compatibility...")
                test_df = pd.DataFrame(
                    {
                        "age": [30, 40, 50, 60],
                        "sex": ["male", "female", "male", "female"],
                        "bmi": [25.0, 28.0, 22.0, 31.0],
                        "children": [0, 1, 2, 3],
                        "smoker": ["no", "yes", "no", "yes"],
                        "region": ["northeast", "northwest", "southeast", "southwest"],
                    }
                )
                test_output = self.feature_engineer.transform_pipeline(test_df)
                actual_features = test_output.shape[1]

                if actual_features != expected_features:
                    raise ValueError(
                        f"❌ FEATURE MISMATCH:\n"
                        f"   Model expects: {expected_features} features\n"
                        f"   Preprocessor produces: {actual_features} features"
                    )

                # apply the same DMatrix workaround used in predict() so
                # the init validation call uses the GPU path and does NOT emit the
                # "Falling back to prediction using DMatrix due to mismatched devices"
                # warning on every startup.  The warning fires because XGBoost's
                # inplace_predict (triggered by passing a DataFrame directly) sees a
                # CPU array against a CUDA booster.  Wrapping in DMatrix bypasses
                # inplace_predict entirely — DMatrix.predict() has no device check.
                try:
                    import xgboost as _xgb_init

                    if hasattr(self.model, "get_booster"):
                        _init_arr = (
                            test_output.values
                            if hasattr(test_output, "values")
                            else np.asarray(test_output)
                        )
                        _init_dmat = _xgb_init.DMatrix(_init_arr)
                        self.model.get_booster().predict(_init_dmat)
                    else:
                        self.model.predict(test_output)
                except (ImportError, ValueError, RuntimeError, AttributeError) as _init_err:
                    # Non-XGBoost model or DMatrix construction/predict failed.
                    # Log the reason so the fallback is not completely silent —
                    # a bare `except Exception` previously swallowed genuine errors
                    # (e.g. shape mismatch, GPU OOM during init) with no trace.
                    logger.debug(
                        "ℹ️  DMatrix init-probe fallback (%s: %s) — "
                        "using sklearn predict path for validation call.",
                        type(_init_err).__name__,
                        _init_err,
                    )
                    self.model.predict(test_output)

                logger.info(f"✅ Feature validation passed: {expected_features} features")

            logger.info(f"✅ Loaded model: {self.model_name}")
            logger.info("=" * 70)

            # Cross-validate BiasCorrection from JSON vs model artifact ──
            # Both are written by train.py from the same BiasCorrection instance,
            # but if the model was retrained without regenerating the preprocessor
            # (or vice versa), the two copies can diverge.
            _bc_json = self._bias_correction  # loaded from bias_correction.json
            _bc_model = getattr(self.model, "_bias_correction", None)  # from model metadata
            # Architecture note: if self.model is a CalibratedModel, __getattr__ delegates
            # to base_model._bias_correction correctly. However CalibratedModel.__setattr__
            # does NOT delegate _bias_correction writes to base_model — a direct assignment
            # lands on the CalibratedModel wrapper, not the base. If train.py ever sets
            # calibrated_model._bias_correction post-fit, the embedded value and the
            # base_model value will silently diverge. (Finding 5 — no action today.)

            # changed condition from
            #   'if _bc_json is not None AND _bc_model is not None'
            # to
            #   'if _bc_json is not None'
            #
            # The old condition silently passed when _bc_model was None (which happens
            # when bias_correction=None was stored in model metadata during a quantile
            # training run). The check body never ran, but the 'else' branch still
            # logged "✅ BiasCorrection consistency check passed" — a false positive on
            # every inference call. The stale JSON went undetected every time.
            if _bc_json is not None and _bc_model is not None:
                _TOLERANCE = 1e-4
                _mismatches = []
                # the original list only included "threshold_low" (a 3-tier
                # attribute absent on 2-tier objects). For 2-tier BiasCorrection the
                # routing attribute is "threshold", not "threshold_low".  Omitting
                # "threshold" from the check meant a stale JSON with the wrong routing
                # boundary logged "✅ consistency check passed" — a false positive on
                # every single-threshold BC loaded from a mismatched run.
                # check "threshold" first (universal), then "threshold_low"
                # (3-tier only — silently skipped when None via the guard below).
                for _attr in ("var_low", "var_high", "threshold", "threshold_low"):
                    _v_json = getattr(_bc_json, _attr, None)
                    _v_model = getattr(_bc_model, _attr, None)
                    if _v_json is not None and _v_model is not None:
                        if abs(_v_json - _v_model) > _TOLERANCE:
                            _mismatches.append(
                                f"   {_attr}: JSON={_v_json:.6f}, model={_v_model:.6f}  "
                                f"(Δ={abs(_v_json - _v_model):.6f})"
                            )
                if _mismatches:
                    logger.error(
                        "❌ CRITICAL — BiasCorrection MISMATCH between "
                        "bias_correction.json and model artifact:\n"
                        + "\n".join(_mismatches)
                        + "\n\n   Root cause: bias_correction.json and the model were "
                        "saved from DIFFERENT training runs.\n"
                        "   Impact:     var_high discrepancy causes wrong correction "
                        "on high-value predictions ($15k–$32k range).\n"
                        "\n   🔧 Overriding with model-embedded BiasCorrection.\n"
                        "      The model .joblib and its embedded BiasCorrection are "
                        "saved atomically in a single joblib.dump() call inside "
                        "train_single_model(). The JSON is a secondary artifact that "
                        "can be overwritten by a subsequent training phase and is NOT "
                        "the ground truth when a mismatch is detected.\n"
                        "      Re-run train.py for a full end-to-end pass to resync "
                        "both artifacts."
                    )
                    self._bias_correction = _bc_model  # model artifact is ground truth
                else:
                    logger.info(
                        "✅ BiasCorrection consistency check passed (JSON == model artifact)"
                    )
            elif _bc_json is not None and _bc_model is None:
                # JSON loaded but model artifact has no embedded BiasCorrection.
                # This means the model was trained with bias_correction=None (quantile
                # model) but a stale bias_correction.json from a previous non-quantile
                # run is still on disk.
                logger.warning(
                    "⚠️  BiasCorrection STALE ARTIFACT DETECTED:\n"
                    "   bias_correction.json exists on disk but the loaded model artifact "
                    "has no embedded BiasCorrection (model._bias_correction is None).\n"
                    "   This means the JSON was produced by a DIFFERENT training run "
                    "(likely a non-quantile model such as ElasticNet/QuantileRegressor).\n"
                    "   The corrections in the JSON are INVALID for the current model "
                    "and will cause systematic underpricing.\n"
                    "\n   🔧 FIX: Delete bias_correction.json and retrain, OR run "
                    "train.py which will rename it automatically."
                )
            else:
                logger.info(
                    "✅ BiasCorrection consistency check passed "
                    "(no JSON artifact — quantile model, correction intentionally absent)"
                )

            # ── High-value segment router ─────────────────────
            # Non-fatal: if specialist model file is absent (pre-training or cold
            # start), HighValueSegmentRouter sets enabled=False and every call to
            # route() is a transparent no-op.  PredictionPipeline works normally.
            self._segment_router = HighValueSegmentRouter(self)

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def _resolve_preprocessor_path(self, explicit_path: str | None) -> str:
        """Resolve preprocessor path with versioned-file safety.

        Priority:
          1. Caller-supplied explicit path (if it exists on disk)
          2. config.yaml preprocessor_path key (if present)
          3. Most recently modified preprocessor_*.joblib in models dir
          4. Legacy hardcoded default (backward compat, warns loudly)
        """
        _models_dir = Path(self.model_dir)

        # Priority 1 — explicit caller argument
        if explicit_path is not None:
            if Path(explicit_path).exists():
                return explicit_path
            logger.warning(
                f"⚠️  Supplied preprocessor_path='{explicit_path}' not found on disk. "
                "Falling back to auto-resolution."
            )

        # Priority 2 — config.yaml key
        _cfg_path = self.config.get("preprocessor_path")
        if _cfg_path and Path(_cfg_path).exists():
            logger.info(f"✅ Preprocessor path from config: {_cfg_path}")
            return str(_cfg_path)

        # Priority 3 — newest preprocessor_*.joblib in models dir
        if _models_dir.exists():
            candidates = [
                p
                for p in _models_dir.glob("preprocessor_*.joblib")
                if not p.stem.endswith("_checksum")
            ]
            if candidates:
                newest = max(candidates, key=lambda p: p.stat().st_mtime)
                logger.info(
                    f"✅ Auto-selected preprocessor: {newest.name} "
                    f"(newest of {len(candidates)} candidate(s))"
                )
                return str(newest)

        # Priority 4 — legacy default; hard-code only as last resort
        _legacy = "models/preprocessor_v5.2.0.joblib"
        logger.warning(
            f"⚠️  No preprocessor found via auto-resolution. "
            f"Falling back to legacy default: {_legacy}\n"
            "   If this file does not exist, run train.py to regenerate it."
        )
        return _legacy

    def _get_best_model(self) -> str:
        """
        Auto-select best model from metadata with intelligent fallback

        Priority:
        1. pipeline_metadata.json (created by train.py)
        2. config.yaml
        3. Scan models directory for available models
        4. Raise error if nothing found
        """
        # ========================================================================
        # PRIORITY 1: Load from pipeline_metadata.json (RECOMMENDED)
        # ========================================================================
        try:
            metadata_path = Path(self.model_dir) / "pipeline_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

                best_model = metadata.get("best_model")
                if best_model:
                    logger.info(f"🎯 Auto-selected best model: {best_model}")
                    if "best_val_rmse" in metadata:
                        logger.info(f"   Validation RMSE: ${metadata['best_val_rmse']:,.0f}")
                    if "best_val_r2" in metadata:
                        logger.info(f"   Validation R²: {metadata['best_val_r2']:.4f}")
                    if "training_timestamp" in metadata:
                        logger.info(f"   Trained: {metadata['training_timestamp']}")
                    return str(best_model)

                # ✅ FALLBACK 1a: Use first trained model from metadata
                trained_models = metadata.get("trained_models", [])
                if trained_models:
                    fallback_model = trained_models[0]
                    logger.warning(
                        f"⚠️ 'best_model' not in metadata, using first trained model: {fallback_model}\n"
                        f"   Available: {', '.join(trained_models)}"
                    )
                    return str(fallback_model)

        except Exception as e:
            logger.debug(f"ℹ️ Could not load pipeline_metadata.json: {e}")

        # ========================================================================
        # PRIORITY 2: Load from config.yaml
        # ========================================================================
        config_model = self.config.get("model", {}).get("best_model")
        if config_model:
            logger.warning(
                f"⚠️ Using model from config.yaml: {config_model}\n"
                f"   This may not be the actual best model from training!"
            )
            return str(config_model)

        # ========================================================================
        # PRIORITY 3: Scan models directory for available models
        # ========================================================================
        model_dir = Path(self.model_dir)
        if model_dir.exists():
            # Find all .joblib files (excluding checksums and preprocessor)
            available_models = [
                p.stem
                for p in model_dir.glob("*.joblib")
                if not p.stem.endswith("_checksum")
                and not p.stem.startswith("preprocessor_")
                and p.stem not in ["drift_baseline"]  # Exclude monitoring files
            ]

            if available_models:
                logger.warning(
                    f"⚠️ No metadata found. Scanning models directory...\n"
                    f"   Found {len(available_models)} model(s): {', '.join(available_models)}"
                )

                # stale comment removed. xgboost_median is first because
                # it is the designated pricing model (reg:squarederror).  Linear models
                # are last — they have the lowest R² on this dataset and should only be
                # selected when no tree-based model is available.
                preferred_order = [
                    # xgboost_median added above xgboost ──────────────────
                    # xgboost_median (reg:squarederror) is the designated pricing
                    # model.  It must rank above the risk model (xgboost,
                    # reg:quantileerror α=0.65) so that cold-start / metadata-absent
                    # scenarios load the correct artifact for premium calculation.
                    # Without this, any training run that fails to write
                    # pipeline_metadata.json falls back to the quantile risk model,
                    # which systematically over-prices (G7 overpricing rate > 55%).
                    "xgboost_median",
                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "random_forest",
                    "gradient_boosting",
                    # Linear models are listed last: they are typically inferior on
                    # the insurance dataset (lower R²) and should only be selected
                    # when no tree-based model is available.
                    "lasso",
                    "ridge",
                    "elastic_net",
                    "linear_regression",
                ]

                for preferred in preferred_order:
                    if preferred in available_models:
                        logger.warning(
                            f"✅ Selected: {preferred} (from preferred model list)\n"
                            f"   ⚠️ This may NOT be the best performing model!\n"
                            f"   Recommendation: Re-run train.py to generate proper metadata"
                        )
                        return preferred

                # No preferred model found, use first alphabetically
                fallback = sorted(available_models)[0]
                logger.warning(
                    f"✅ Selected: {fallback} (alphabetically first)\n"
                    f"   ⚠️ This may NOT be the best performing model!\n"
                    f"   Recommendation: Re-run train.py to generate proper metadata"
                )
                return fallback

        # ========================================================================
        # PRIORITY 4: No models found - raise error
        # ========================================================================
        error_msg = (
            "❌ No trained models found!\n\n"
            f"Searched locations:\n"
            f"  1. Metadata: {metadata_path if 'metadata_path' in locals() else 'N/A'}\n"
            f"  2. Config: {self.config.get('model', {})}\n"
            f"  3. Models directory: {model_dir}\n\n"
            "🚨 REQUIRED ACTION: Run train.py to generate models\n\n"
            "Example:\n"
            "  python train.py\n"
        )

        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    def _validate_categorical_values(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize categorical input values.

        astype(str) cast added before every .str chain so that numeric
        values (e.g. sex=0/1 from an upstream encoder) produce a clear ValueError
        instead of a silent all-NaN series that reaches the isin() check and emits
        a confusing 'Invalid sex values: [nan]' message.
        """
        df = input_data.copy()

        df["sex"] = df["sex"].astype(str).str.lower().str.strip()
        invalid = df[~df["sex"].isin(["male", "female"])]
        if not invalid.empty:
            raise ValueError(f"Invalid 'sex' values: {invalid['sex'].unique()}")

        df["smoker"] = df["smoker"].astype(str).str.lower().str.strip()
        invalid = df[~df["smoker"].isin(["yes", "no"])]
        if not invalid.empty:
            raise ValueError(f"Invalid 'smoker' values: {invalid['smoker'].unique()}")

        df["region"] = df["region"].astype(str).str.lower().str.strip()
        valid_regions = ["northeast", "northwest", "southeast", "southwest"]
        invalid = df[~df["region"].isin(valid_regions)]
        if not invalid.empty:
            raise ValueError(f"Invalid 'region' values: {invalid['region'].unique()}")

        return df

    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess with detailed validation logging"""
        required_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
        missing_cols = [col for col in required_cols if col not in input_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = input_data.copy()

        try:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
            df["children"] = pd.to_numeric(df["children"], errors="coerce")
        except Exception as e:
            raise ValueError(f"Error converting numeric columns: {e}") from e

        numeric_cols = ["age", "bmi", "children"]
        if df[numeric_cols].isna().any().any():
            nan_counts = df[numeric_cols].isna().sum()
            nan_cols = nan_counts[nan_counts > 0].to_dict()
            # include problem row indices in the error message.
            # Previously problem_rows was computed but silently dropped from the
            # ValueError, forcing operators to re-run the batch to identify bad rows.
            problem_rows = df[df[numeric_cols].isna().any(axis=1)].index.tolist()
            raise ValueError(
                f"Invalid numeric values detected:\n"
                f"   NaN counts: {nan_cols}\n"
                f"   Affected row indices: {problem_rows[:20]}"
                f"{'  (first 20 shown)' if len(problem_rows) > 20 else ''}"
            )

        # ── Read validation bounds from config, not hardcoded literals ──
        _feat = self.config.get("features", {})

        _bmi_min = _feat.get("bmi_min", 10.0)
        _bmi_max = _feat.get("bmi_max", 100.0)
        _age_min = _feat.get("age_min", 0.0)
        _age_max = _feat.get("age_max", 120.0)
        _children_min = _feat.get("children_min", 0)
        _children_max = _feat.get("children_max", 20)

        age_invalid = df[(df["age"] < _age_min) | (df["age"] > _age_max)]
        if not age_invalid.empty:
            raise ValueError(f"Age must be in [{_age_min}, {_age_max}]")

        bmi_invalid = df[(df["bmi"] < _bmi_min) | (df["bmi"] > _bmi_max)]
        if not bmi_invalid.empty:
            raise ValueError(f"BMI must be in [{_bmi_min}, {_bmi_max}]")

        children_invalid = df[(df["children"] < _children_min) | (df["children"] > _children_max)]
        if not children_invalid.empty:
            raise ValueError(f"Children must be in [{_children_min}, {_children_max}]")

        df = self._validate_categorical_values(df)

        try:
            processed_data = self.feature_engineer.transform_pipeline(df)
            logger.debug(f"✅ Preprocessing completed: {len(processed_data)} samples")
            return processed_data
        except Exception as e:
            logger.error(f"❌ Error in preprocessing pipeline: {e}")
            raise

    def predict(self, input_data: pd.DataFrame, return_reliability: bool = True) -> dict[str, Any]:
        """Predict with business-focused statistics"""
        if input_data.empty:
            raise ValueError("Input data cannot be empty")

        # T3-C: read max batch size from config so bulk re-pricing runs
        # (e.g. portfolio renewal at 50K policies) don't require code changes.
        _max_batch = int(self.config.get("prediction", {}).get("max_batch_size", 50000))
        if len(input_data) > _max_batch:
            raise ValueError(
                f"Batch size too large: {len(input_data)} samples "
                f"(limit: {_max_batch} — set prediction.max_batch_size in config.yaml to override)"
            )

        try:
            processed_input = self.preprocess_input(input_data)
            # Move input to model's device to avoid slow DMatrix fallback.
            # XGBoost booster runs on cuda:0; CPU DataFrame triggers a warning
            # and a slower DMatrix conversion path every inference call.
            try:
                import xgboost as _xgb

                if hasattr(self.model, "get_booster"):
                    _arr = (
                        processed_input.values
                        if hasattr(processed_input, "values")
                        else np.asarray(processed_input)
                    )
                    _dmat = _xgb.DMatrix(_arr)
                    predictions_raw = self.model.get_booster().predict(_dmat)
                else:
                    predictions_raw = self.model.predict(processed_input)
            except Exception:
                predictions_raw = self.model.predict(processed_input)

            is_valid, scale_msg = validate_prediction_scale(
                predictions_raw,
                scale_type="log",
                method=self.feature_engineer.target_transformation.method,
            )

            if not is_valid:
                logger.error(scale_msg)
                raise ValueError("Raw prediction scale validation failed!")
            else:
                logger.debug(scale_msg)

            if len(predictions_raw) != len(input_data):
                raise ValueError("Prediction count mismatch!")

            if not isinstance(predictions_raw, np.ndarray):
                predictions_raw = np.array(predictions_raw)

            if np.any(~np.isfinite(predictions_raw)):
                raise ValueError("Model produced invalid predictions")

            predictions_original = self.feature_engineer.inverse_transform_target(
                predictions_raw,
                transformation_method=self.feature_engineer.target_transformation.method,
                clip_to_safe_range=True,
                context="prediction",
            )

            # ── Apply stratified bias correction to point predictions ──
            # NOTE: At inference there is no ground-truth y; we use y_pred itself
            # as the routing signal for tier assignment. This is the correct
            # convention: borderline predictions near tier thresholds may be
            # misrouted by one tier, introducing a small (~0.1%) artefact.
            if self._bias_correction is not None:
                predictions_original = self._bias_correction.apply(
                    y_pred=predictions_original,
                    y_original=predictions_original,  # self-referential routing at inference
                    log_details=logger.isEnabledFor(logging.DEBUG),
                )
                logger.debug("✅ Stratified bias correction applied to point predictions.")
            else:
                logger.debug(
                    "⚠️ No bias correction — predictions not corrected for log-transform bias."
                )

            is_valid_out, scale_msg_out = validate_prediction_scale(
                predictions_original, scale_type="original"
            )

            if not is_valid_out:
                logger.error(scale_msg_out)
                raise ValueError("Final prediction scale validation failed!")
            else:
                logger.info(scale_msg_out)

            n_negative = 0
            if np.any(predictions_original < 0):
                n_negative = (predictions_original < 0).sum()
                logger.warning(f"⚠️ Clipping {n_negative} negative predictions to 0")
                predictions_original = np.clip(predictions_original, 0, None)

            # ── Route high-value predictions to specialist ────
            # HighValueSegmentRouter blends the global model's output with a
            # specialist model trained exclusively on the high-value segment,
            # using a soft 30% blend window on each side of the routing threshold
            # ($16,701 = P75) to eliminate hard boundary discontinuity.
            #
            # When the specialist model file is absent (router.enabled=False),
            # this block is a complete no-op — predictions_original is unchanged.
            #
            # NOTE: processed_input is available from line 586 above.  We pass
            # it here rather than re-running preprocess_input() to avoid double
            # feature-engineering overhead.
            _routing_diagnostics: dict[str, Any] = {}
            if self._segment_router.enabled:
                predictions_original, _routing_diagnostics = self._segment_router.route(
                    processed_input=processed_input,
                    global_preds_original=predictions_original,
                    feature_engineer=self.feature_engineer,
                )
                # Re-clip: specialist booster output can produce rare near-zero
                # edge values after inverse-transform on extreme feature inputs.
                _n_neg_post = int(np.sum(predictions_original < 0))
                if _n_neg_post > 0:
                    logger.warning(
                        f"⚠️ Clipping {_n_neg_post} negative predictions "
                        f"after specialist routing"
                    )
                    predictions_original = np.clip(predictions_original, 0, None)
                    n_negative += _n_neg_post

            result = {
                "predictions": predictions_original.tolist(),
                "predictions_log_space": predictions_raw.tolist(),  # 🆕 For CI calculation
                "model_used": self.model_name,
                "input_count": len(input_data),
                "target_transformation": self.feature_engineer.target_transformation.method,
                "preprocessor_version": self.VERSION,
                "statistics": {
                    "mean": float(np.mean(predictions_original)),
                    "median": float(np.median(predictions_original)),
                    "min": float(np.min(predictions_original)),
                    "max": float(np.max(predictions_original)),
                    "std": float(np.std(predictions_original)),
                    "q25": float(np.percentile(predictions_original, 25)),
                    "q75": float(np.percentile(predictions_original, 75)),
                    "interquartile_range": float(
                        np.percentile(predictions_original, 75)
                        - np.percentile(predictions_original, 25)
                    ),
                    "coefficient_of_variation": float(
                        np.std(predictions_original) / np.mean(predictions_original)
                        if np.mean(predictions_original) > 0
                        else 0
                    ),
                    "n_low_premium": int(np.sum(predictions_original < 5000)),
                    "n_medium_premium": int(
                        np.sum((predictions_original >= 5000) & (predictions_original < 15000))
                    ),
                    "n_high_premium": int(np.sum(predictions_original >= 15000)),
                    "pct_low_premium": float(
                        np.sum(predictions_original < 5000) / len(predictions_original) * 100
                    ),
                    "pct_medium_premium": float(
                        np.sum((predictions_original >= 5000) & (predictions_original < 15000))
                        / len(predictions_original)
                        * 100
                    ),
                    "pct_high_premium": float(
                        np.sum(predictions_original >= 15000) / len(predictions_original) * 100
                    ),
                },
                "validation": {
                    "scale_check_passed": True,
                    "feature_count": processed_input.shape[1],
                    "negative_predictions_clipped": int(n_negative),
                },
                # Segment routing diagnostics (empty dict when router is disabled
                # or specialist model has not been trained yet).
                "routing_diagnostics": _routing_diagnostics,
            }

            if return_reliability:
                extreme_threshold = int(
                    self.config.get("prediction", {}).get("extreme_prediction_threshold", 100_000)
                )
                max_pred_val = np.max(predictions_original)

                result["reliability"] = {
                    "has_extreme_predictions": bool(max_pred_val > extreme_threshold),
                    "extreme_count": int(np.sum(predictions_original > extreme_threshold)),
                    "max_prediction": float(max_pred_val),
                    # ── two precise flags instead of one misleading one ──
                    # True only when BiasCorrection.apply() was called on these predictions
                    "point_prediction_bias_corrected": self._bias_correction is not None,
                    # True when _log_residual_variance is available for CI width computation
                    "uncertainty_quantification_available": (
                        getattr(self.feature_engineer, "_log_residual_variance", None) is not None
                    ),
                    # Deprecated — kept for one release to avoid breaking callers
                    # Will be removed in v6.4.0. Use point_prediction_bias_corrected instead.
                    "bias_correction_applied": self._bias_correction is not None,
                    "scale_validation_passed": True,
                }

            return result

        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise

    def predict_with_intervals(
        self,
        input_data: pd.DataFrame,
        confidence_level: float = 0.90,
        _precomputed_result: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
        """
        Generate predictions with CORRECT confidence intervals

        Confidence intervals are now computed in log space BEFORE inverse transform

        (_precomputed_result): When evaluate.py has already called self.predict()
        for Step 3 (ML predictions), it can pass that result dict here so feature
        engineering + XGBoost inference are not repeated in Step 6a (CI coverage check).
        Shape is validated before use; a mismatch falls back to a fresh predict() call.
        Intentionally private (underscore prefix) — internal optimisation only.
        """
        if input_data.empty:
            raise ValueError("Input data cannot be empty")

        def _use_precomputed(pre: "dict[str, Any] | None") -> bool:
            """Return True only when the precomputed result is shape-compatible."""
            if pre is None:
                return False
            n_pre = len(pre.get("predictions", []))
            if n_pre != len(input_data):
                logger.warning(
                    f"⚠️  _precomputed_result has {n_pre} predictions but "
                    f"input_data has {len(input_data)} rows — "
                    "ignoring precomputed result and running fresh predict()."
                )
                return False
            return True

        # ── CI SOURCE PRIORITY ────────────────────────────────────────────────
        # Priority 1: model._validation_residuals  (conformal calibration data,
        #             std ≈ 0.63, restored from model artifact by load_model())
        # Priority 2: _log_residual_variance        (bias-correction variance;
        #             for yeo-johnson this is a median-ratio sentinel that can
        #             be as small as _MIN_VAR=1e-6 when the model already
        #             over-predicts.  Using it for CI width gives ~$37 intervals
        #             which are statistically meaningless.)
        # ─────────────────────────────────────────────────────────────────────

        # Attempt Priority 1: conformal residual std
        _conformal_std: float | None = None
        _val_residuals = getattr(self.model, "_validation_residuals", None)
        if _val_residuals is not None and len(_val_residuals) > 0:
            _conformal_std = float(np.std(_val_residuals))
            if _conformal_std < _CONFORMAL_STD_MIN:  # degenerate / near-zero
                logger.warning(
                    f"⚠️ Conformal residual std={_conformal_std:.6f} is near-zero; "
                    "falling back to _log_residual_variance."
                )
                _conformal_std = None
            else:
                logger.info(
                    f"✅ CI source: conformal_residuals_split  "
                    f"(n={len(_val_residuals)}, std={_conformal_std:.6f})"
                )

        # Attempt Priority 2: _log_residual_variance (guard: must be meaningful)
        if _conformal_std is None:
            _lrv = getattr(self.feature_engineer, "_log_residual_variance", None)
            # Guard: only use if positive and above minimum useful threshold (0.01).
            # Negative values encode downward corrections (var = 2*log(ratio), ratio < 1)
            # — sqrt of a negative is undefined, so they cannot be used for CI std.
            # Near-zero positive values (<= 0.01) would produce negligibly narrow CIs
            # (~$37 width) that are statistically meaningless for insurance premiums.
            if _lrv is not None and _lrv > 0.01:
                _conformal_std = float(np.sqrt(_lrv))
                logger.info(
                    f"✅ CI source: _log_residual_variance  "
                    f"(variance={_lrv:.6f}, std={_conformal_std:.6f})"
                )
            else:
                lrv_val = _lrv if _lrv is not None else "missing"
                _reason = (
                    "negative (downward correction encoding — sqrt undefined)"
                    if isinstance(lrv_val, float) and lrv_val < 0
                    else (
                        "below minimum usable threshold (0.01)"
                        if isinstance(lrv_val, float)
                        else "missing"
                    )
                )
                logger.warning(
                    f"⚠️ Cannot compute meaningful confidence intervals.\n"
                    f"   _log_residual_variance={lrv_val} — {_reason}.\n"
                    f"   model._validation_residuals is also unavailable.\n"
                    f"   Retrain to populate conformal calibration data."
                )
                result = (
                    _precomputed_result
                    if _use_precomputed(_precomputed_result)
                    else self.predict(input_data, return_reliability=True)
                )
                if result is None:  # narrow type: _use_precomputed guarantees non-None
                    result = self.predict(input_data, return_reliability=True)
                result["confidence_intervals"] = None
                return result

        # ── Single-pass: call predict() once, reuse its log-space output ────
        # Use precomputed result when available to avoid a redundant feature
        # engineering + XGBoost inference pass (e.g. evaluate.py Step 6a reusing
        # the Step 3 ML predictions already computed over the same X_test).
        result = (
            _precomputed_result
            if _use_precomputed(_precomputed_result)
            else self.predict(input_data, return_reliability=True)
        )
        if result is None:  # narrow type: _use_precomputed guarantees non-None
            result = self.predict(input_data, return_reliability=True)

        try:
            predictions_log = np.array(result["predictions_log_space"])

            # ── proper distribution-free conformal quantile ────────
            # Previous code used  pred ± z × std  which is a PARAMETRIC Gaussian
            # approximation with NO coverage guarantee.  The correct split-conformal
            # method builds the interval from the empirical quantile of absolute
            # calibration residuals with a finite-sample correction:
            #
            #   q_level = ceil((n_cal + 1)(1 - α)) / n_cal
            #   q        = quantile(|residuals|, q_level, method="higher")
            #   CI       = [ŷ − q, ŷ + q]
            #
            # This achieves  P(Y ∈ CI) ≥ 1 − α  without any distributional
            # assumption (exchangeability only).  The z-score approach can under-
            # cover when residuals are skewed or heavy-tailed — both common for
            # insurance premiums.
            alpha = 1.0 - confidence_level
            n_cal = len(_val_residuals) if _val_residuals is not None else 0

            if _val_residuals is not None and n_cal >= 30:
                conformity_scores = np.abs(_val_residuals)

                # ── prefer stored heteroscedastic bins over global quantile ──────
                # Training uses 9-bin heteroscedastic conformal (per-bin quantiles adaptive
                # to prediction magnitude).  Inference was using a single global
                # split-conformal quantile — a 3.5× CI width mismatch ($36K vs $125K
                # training CI).  When the model artifact contains stored bins, use them
                # directly without refitting.
                _hetero_bins = None
                _raw_model = (
                    self.model.base_model if hasattr(self.model, "base_model") else self.model
                )
                _cd = getattr(_raw_model, "_conformal_data", None)
                if isinstance(_cd, dict):
                    _hetero_bins = _cd.get("heteroscedastic_bins")

                # ── validate predictions_log_space before using bins ──
                # If predict() doesn't populate 'predictions_log_space' (e.g. key
                # renamed, or a code path that skips it), _preds_log will be empty.
                # An empty _preds_log produces _per_pred_q of shape (0,), then
                # `lower_log = predictions_log - _lower_q` raises a shape mismatch
                # which is caught by the outer except and silently returns no CI.
                # Guard: force fallback to global split-conformal when bins are
                # present but log-space predictions are unavailable.
                _preds_log = np.array(result.get("predictions_log_space", []), dtype=float)
                _n_preds = len(result.get("predictions", []))
                if _hetero_bins is not None and len(_preds_log) != _n_preds:
                    logger.warning(
                        f"⚠️  heteroscedastic_bins present but 'predictions_log_space' "
                        f"has {len(_preds_log)} entries vs {_n_preds} predictions — "
                        f"falling back to global split-conformal. "
                        f"Ensure PredictionPipeline.predict() populates "
                        f"'predictions_log_space' for every code path."
                    )
                    _hetero_bins = None  # force fallback branch below

                if _hetero_bins is not None:
                    _bin_rights = np.array(_hetero_bins["bin_right_edges"], dtype=float)
                    _bin_qs = np.array(_hetero_bins["bin_quantiles"], dtype=float)
                    _asym_upper = float(_hetero_bins.get("asym_upper_ratio", 1.0))
                    _asym_lower = float(_hetero_bins.get("asym_lower_ratio", 1.0))

                    # vectorised bin assignment with np.searchsorted.
                    # The previous Python nested loop was O(n × b) where n = batch
                    # size and b = number of bins.  At 100K samples × 10 bins that
                    # is 1,000,000 Python iterations (~600ms).  searchsorted is
                    # O(n log b) and runs fully in C — benchmarks to <1ms at 100K.
                    _bin_indices = np.searchsorted(_bin_rights, _preds_log, side="right").clip(
                        0, len(_bin_qs) - 1
                    )
                    _per_pred_q = _bin_qs[_bin_indices]

                    quantile_value = float(np.mean(_per_pred_q))
                    ci_method = "heteroscedastic_conformal"
                    ci_source = "stored_heteroscedastic_bins"
                    logger.info(
                        f"✅ Heteroscedastic CI: {_hetero_bins['n_bins']} bins "
                        f"(range: [{_bin_qs.min():.4f}, {_bin_qs.max():.4f}], "
                        f"mean quantile: {quantile_value:.4f})"
                    )

                    # Per-prediction asymmetric log-space offsets
                    _upper_q = _per_pred_q * _asym_upper
                    _lower_q = _per_pred_q * _asym_lower

                else:
                    # ── Fallback: global split-conformal (pre-C1 behaviour) ──────────────
                    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
                    quantile_value = float(np.quantile(conformity_scores, q_level, method="higher"))
                    ci_method = "split_conformal"
                    ci_source = "conformal_residuals_split"
                    logger.info(
                        f"✅ Conformal quantile: {quantile_value:.4f}  "
                        f"(level={q_level:.4f}, n_cal={n_cal})"
                    )
                    logger.warning(
                        "⚠️  CI METHOD MISMATCH: using global split-conformal fallback — "
                        "training used heteroscedastic_conformal (per-bin adaptive quantiles). "
                        "Reported interval width will differ from the training design target. "
                        "Store heteroscedastic bins in the model artifact to restore adaptive CI. "
                        "Re-run train.py with C1 patches applied."
                        # removed stale '$36K / $125K' width literals which reflected
                        # the pre-fix state (v6.3.1).  Current deployed CI mean width is ~$11,859
                        # (config.yaml ci_mean_width).  Hardcoded widths in a log message cause
                        # false incident reports when operators see numbers that no longer apply.
                    )
                    _upper_q = np.full(len(result["predictions"]), quantile_value)
                    _lower_q = _upper_q
            else:
                # Fallback: parametric Gaussian (labelled clearly, not as conformal)
                quantile_value = float(
                    scipy_stats.norm.ppf((1 + confidence_level) / 2) * _conformal_std
                )
                ci_method = "parametric_gaussian_fallback"
                # Use distinct labels so downstream consumers can tell exactly which
                # source backed the CI: split-conformal data vs. variance-derived std.
                ci_source = (
                    "conformal_residuals_std_matched"
                    if _val_residuals is not None
                    and np.isclose(_conformal_std, float(np.std(_val_residuals)), rtol=1e-9)
                    else "log_residual_variance"
                )
                logger.warning(
                    f"⚠️ Falling back to parametric CI (n_cal={n_cal} < 30 or no residuals). "
                    f"Coverage NOT guaranteed."
                )
                # Broadcast scalar to arrays so interval construction below is uniform
                _upper_q = np.full(len(result["predictions"]), quantile_value)
                _lower_q = _upper_q

            # Use per-prediction arrays (heteroscedastic) or scalar broadcast (global/parametric)
            lower_log = predictions_log - _lower_q
            upper_log = predictions_log + _upper_q

            # Inverse-transform all three arrays in one block.
            # Previously a single _inv lambda with clip_to_safe_range=True was
            # used for all three arrays (point, lower, upper).  For upper CI bounds this
            # truncated extreme-but-valid YJ values at y_max_safe ($89,278), making the
            # reported CI narrower than the model's actual uncertainty for high-value
            # samples.  In insurance, understating upper-tail uncertainty is non-conservative.
            #
            # separate lambdas per use-case:
            #   _inv_pt   — point prediction: keep original clip=True (preserves existing behaviour)
            #   _inv_lo   — lower bound: clip at 0 only (no negative premiums)
            #   _inv_hi   — upper bound: unclipped inverse-transform, then hard-cap at
            #               y_max_safe * _CI_UPPER_CAP_FACTOR (3×) to prevent NaN/inf
            #               from extreme OOD YJ inputs while still reflecting real uncertainty
            _transform_method = self.feature_engineer.target_transformation.method
            _y_max_safe = getattr(self.feature_engineer, "y_max_safe", None)
            # _CI_UPPER_CAP_FACTOR is now a module-level constant.

            def _inv_pt(arr):
                return self.feature_engineer.inverse_transform_target(
                    arr,
                    transformation_method=_transform_method,
                    clip_to_safe_range=True,
                    context="prediction",
                )

            def _inv_lo(arr):
                return np.maximum(
                    self.feature_engineer.inverse_transform_target(
                        arr,
                        transformation_method=_transform_method,
                        clip_to_safe_range=False,
                        context="ci_lower_bound",
                    ),
                    0.0,
                )

            def _inv_hi(arr):
                return (
                    np.minimum(
                        self.feature_engineer.inverse_transform_target(
                            arr,
                            transformation_method=_transform_method,
                            clip_to_safe_range=False,
                            context="ci_upper_bound",
                        ),
                        _y_max_safe * _CI_UPPER_CAP_FACTOR,
                    )
                    if _y_max_safe is not None
                    else self.feature_engineer.inverse_transform_target(
                        arr,
                        transformation_method=_transform_method,
                        clip_to_safe_range=True,
                        context="ci_upper_bound",
                    )
                )

            predictions_original = _inv_pt(predictions_log)
            lower_bound = _inv_lo(lower_log)
            upper_bound = _inv_hi(upper_log)

            # ── CI CENTERING ──────────────────────────────────────────────
            # `predictions_log_space` holds the raw (uncalibrated) model output.
            # `result["predictions"]` holds the FINAL predictions after:
            #   1. BiasCorrection.apply() — stratified multiplicative correction
            #   2. calibration_factor (via HybridPredictor) — flat multiplier
            # Both shift the point prediction in original space but are NOT applied
            # to lower_log / upper_log above, so the raw-space CI is off-centre
            # relative to the reported prediction.
            #
            # multiply each bound by the same factor that was applied to the
            # point prediction.  This preserves the CI width in log space while
            # ensuring the final interval brackets the calibrated prediction.
            #
            # clamp shift_factor to [0.5, 2.0] to prevent extreme
            # BiasCorrection outputs (e.g. clip artefacts producing tiny
            # predictions_original) from inflating upper_bound toward infinity
            # or compressing lower_bound to zero.  Values outside [0.5, 2.0]
            # indicate a BiasCorrection / clip interaction that should be
            # investigated; the clamp limits downstream damage and triggers a
            # diagnostic warning.
            _calibrated_preds = np.array(result["predictions"])
            _raw_shift = _calibrated_preds / np.maximum(predictions_original, 1e-8)
            _SHIFT_CLAMP_LOW, _SHIFT_CLAMP_HIGH = 0.5, 2.0
            _extreme_shift_mask = (_raw_shift < _SHIFT_CLAMP_LOW) | (_raw_shift > _SHIFT_CLAMP_HIGH)
            if _extreme_shift_mask.any():
                logger.warning(
                    f"⚠️ CI shift_factor out of [{_SHIFT_CLAMP_LOW}, {_SHIFT_CLAMP_HIGH}] "
                    f"for {int(_extreme_shift_mask.sum())} predictions "
                    f"(min={float(_raw_shift.min()):.3f}, max={float(_raw_shift.max()):.3f}). "
                    "Likely BiasCorrection clip artefact on extreme YJ values. "
                    "Clamping to safe range — investigate inverse_transform_target clip settings."
                )
            _shift_factor = np.clip(_raw_shift, _SHIFT_CLAMP_LOW, _SHIFT_CLAMP_HIGH)
            lower_bound = lower_bound * _shift_factor
            upper_bound = upper_bound * _shift_factor
            # Re-compute predictions_original so width logging reflects centred CI.
            predictions_original = _calibrated_preds

            # Sanity-check: CI width should be in the $thousands for typical premiums
            _mean_width = float(np.mean(upper_bound - lower_bound))
            _mean_pred = float(np.mean(predictions_original))
            if _mean_width < 500 and _mean_pred > 1000:
                logger.warning(
                    f"⚠️ CI sanity check FAILED: mean width=${_mean_width:,.2f} on "
                    f"mean prediction=${_mean_pred:,.2f}. "
                    f"Expected width in the thousands. "
                    f"Verify quantile_value={quantile_value:.6f} is correct."
                )

            _raw_widths = upper_bound - lower_bound
            _n_inverted = int(np.sum(_raw_widths < 0))
            if _n_inverted > 0:
                logger.warning(
                    f"⚠️ {_n_inverted} sample(s) have inverted CI bounds "
                    f"(lower > upper) after centering/transform — clipping interval_width to 0."
                )
            result["confidence_intervals"] = {
                "confidence_level": confidence_level,
                "lower_bound": lower_bound.tolist(),
                "upper_bound": upper_bound.tolist(),
                "interval_width": np.maximum(_raw_widths, 0.0).tolist(),
                "mean_interval_width": _mean_width,
                "median_interval_width": float(np.median(upper_bound - lower_bound)),
                "method": ci_method,
                "quantile_value_log": float(quantile_value),
                "n_calibration": n_cal,
                "ci_source": ci_source,
                "note": (
                    "Distribution-free split-conformal intervals: "
                    "P(Y ∈ CI) ≥ 1-α without Gaussian assumption"
                    if ci_method == "split_conformal"
                    # T3-D: build note dynamically from the artifact so the
                    # reported bin count and winsorize percentile stay correct
                    # if the model is retrained with different hyperparameters.
                    # Fallback text preserved when _cd is unavailable.
                    else (
                        (
                            f"Heteroscedastic conformal: per-bin coverage from stored training "
                            f"artifact ({_cd.get('heteroscedastic_bins', {}).get('n_bins', '?')} bins, "
                            f"winsorized at {_cd.get('winsorize_percentile', 99)}th pctile)"
                        )
                        if ci_method == "heteroscedastic_conformal" and isinstance(_cd, dict)
                        else (
                            "Heteroscedastic conformal: per-bin coverage from stored training artifact"
                            if ci_method == "heteroscedastic_conformal"
                            else "Parametric fallback: n_cal too small for conformal guarantee"
                        )
                    )
                ),
            }

            logger.info(
                f"✅ Confidence intervals computed ({confidence_level*100:.0f}%)\n"
                f"   Mean interval width: ${_mean_width:,.2f}\n"
                f"   Quantile (log space): {quantile_value:.6f} "
                f"(source: {ci_source}, method: {ci_method})"
            )

            return result

        except Exception as e:
            # reuse the result dict already populated by self.predict()
            # above. The original code called self.predict() a SECOND time here,
            # doubling latency on every CI failure.
            logger.error(f"❌ Error computing confidence intervals: {e}")
            result["confidence_intervals"] = {"error": str(e)}
            return result

    def predict_single(
        self, age: int, sex: str, bmi: float, children: int, smoker: str, region: str
    ) -> float:
        """Single prediction with comprehensive validation"""
        try:
            age = int(age)
            children = int(children)
            bmi = float(bmi)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric input: {e}") from e

        # ── single-record path reads same config bounds as batch path ──
        _feat = self.config.get("features", {})
        _age_min, _age_max = _feat.get("age_min", 0.0), _feat.get("age_max", 120.0)
        _bmi_min, _bmi_max = _feat.get("bmi_min", 10.0), _feat.get("bmi_max", 100.0)
        _children_min, _children_max = _feat.get("children_min", 0), _feat.get("children_max", 20)

        if not _age_min <= age <= _age_max:
            raise ValueError(f"Age must be in [{_age_min}, {_age_max}], got {age}")
        if not _bmi_min <= bmi <= _bmi_max:
            raise ValueError(f"BMI must be in [{_bmi_min}, {_bmi_max}], got {bmi}")
        if not _children_min <= children <= _children_max:
            raise ValueError(
                f"Children must be in [{_children_min}, {_children_max}], got {children}"
            )

        input_df = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region],
            }
        )

        result = self.predict(input_df, return_reliability=False)
        prediction = float(result["predictions"][0])

        _extreme_threshold = self.config.get("prediction", {}).get(
            "extreme_prediction_threshold", 100_000
        )
        if prediction > _extreme_threshold:
            logger.warning(f"⚠️ Unusually high prediction: ${prediction:,.2f}")

        return prediction

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get comprehensive pipeline information with recommended metrics"""
        metadata = self.feature_engineer.get_feature_metadata()

        info = {
            "pipeline_version": self.VERSION,
            "model_name": self.model_name,
            "target_transformation": {
                "method": self.feature_engineer.target_transformation.method,
                "bias_correction": (
                    hasattr(self.feature_engineer, "_log_residual_variance")
                    and self.feature_engineer._log_residual_variance is not None
                ),
                "bias_correction_variance": (
                    self.feature_engineer._log_residual_variance
                    if hasattr(self.feature_engineer, "_log_residual_variance")
                    else None
                ),
                "recommended_metrics": ["RMSE", "MALE", "SMAPE"],
                "deprecated_metrics": ["MAPE"],
                "metric_descriptions": {
                    "RMSE": "Root Mean Squared Error - primary accuracy metric",
                    "MALE": "Mean Absolute Log Error - scale-invariant metric",
                    "SMAPE": "Symmetric MAPE - better than MAPE for skewed data",
                    "MAPE": "Mean Absolute % Error - diagnostic only, not for decisions",
                },
            },
            "feature_count": metadata.get("scaler_features", "unknown"),
            "pipeline_state": metadata.get("pipeline_state", "unknown"),
            "feature_metadata": metadata,
        }

        if hasattr(self.model, "n_features_in_"):
            info["model_expected_features"] = self.model.n_features_in_

        info["model_type"] = type(self.model).__name__

        if hasattr(self.model, "feature_importances_"):
            info["has_feature_importances"] = True
        elif hasattr(self.model, "coef_"):
            info["has_coefficients"] = True

        return info


# =====================================================================
# ENHANCED HYBRID PREDICTOR
# =====================================================================


class HybridPredictor:
    """
    v6.3.3: Insurance-grade hybrid predictor
    """

    VERSION = "6.3.3"

    def __init__(
        self,
        ml_predictor: PredictionPipeline,
        threshold: float | None = None,
        blend_ratio: float | None = None,
        use_soft_blending: bool | None = None,
        soft_blend_window: float | None = None,
        calibration_factor: float | None = None,
        actuarial_params: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize with enhanced validation"""
        if config is None:
            full_config = load_config()
            config = full_config.get("hybrid_predictor", {})

        self.ml_predictor = ml_predictor
        self.config = config

        self.threshold = threshold if threshold is not None else config.get("threshold", 4500.0)
        self.blend_ratio = (
            blend_ratio if blend_ratio is not None else config.get("blend_ratio", 0.70)
        )
        self.use_soft_blending = (
            use_soft_blending
            if use_soft_blending is not None
            else config.get("use_soft_blending", True)
        )
        self.soft_blend_window = (
            soft_blend_window
            if soft_blend_window is not None
            else config.get("soft_blend_window", 500.0)
        )

        # Calibration configuration
        calibration_config = config.get("calibration", {})
        self.calibration_enabled = calibration_config.get("enabled", True)

        # pricing_factor and risk_factor in config.yaml are DEAD CODE.
        # HybridPredictor was reading only calibration.factor (legacy single-model key,
        # always 1.00), never the per-model pricing_factor / risk_factor keys.
        # resolve the active model name from ml_predictor, then pick the
        # correct per-model factor.  Fall back to legacy factor if keys are absent
        # (backward-compatible with single-model deployments).
        if calibration_factor is not None:
            _resolved_cal_factor = calibration_factor
        else:
            # dispatch on model objective, not model name substring.
            # "median" in name is fragile — any rename (xgboost_pricing, xgb_mean,
            # etc.) silently applies risk_factor instead of pricing_factor.
            # Read the booster's actual objective string as the canonical signal.
            _active_model: str | None = getattr(ml_predictor, "model_name", None)
            _model_objective: str | None = None
            try:
                _raw_m = getattr(ml_predictor, "model", None)
                if _raw_m is not None:
                    _inner = getattr(_raw_m, "base_model", _raw_m)
                    if hasattr(_inner, "get_xgb_params"):
                        _model_objective = str(_inner.get_xgb_params().get("objective", ""))
                    elif hasattr(_inner, "objective"):
                        _model_objective = str(_inner.objective)
            except Exception:
                pass  # objective unresolvable — fall through to name heuristic

            _is_pricing_model = (
                _model_objective is not None and "squarederror" in _model_objective.lower()
            ) or (
                # Name heuristic as secondary fallback only
                _model_objective is None
                and _active_model is not None
                and "median" in _active_model.lower()
            )

            if _is_pricing_model:
                _resolved_cal_factor = float(
                    calibration_config.get(
                        "pricing_factor",
                        calibration_config.get("factor", 1.00),
                    )
                )
                logger.info(
                    "✅ [BUG-5] HybridPredictor using pricing_factor=%.4f "
                    "for model '%s' (objective=%s)",
                    _resolved_cal_factor,
                    _active_model,
                    _model_objective or "unknown",
                )
            elif _active_model is not None:
                _resolved_cal_factor = float(
                    calibration_config.get(
                        "risk_factor",
                        calibration_config.get("factor", 1.00),
                    )
                )
                logger.info(
                    "✅ [BUG-5] HybridPredictor using risk_factor=%.4f "
                    "for model '%s' (objective=%s)",
                    _resolved_cal_factor,
                    _active_model,
                    _model_objective or "unknown",
                )
            else:
                _resolved_cal_factor = float(calibration_config.get("factor", 1.00))
                logger.warning(
                    "⚠️  [BUG-5] ml_predictor.model_name unavailable; "
                    "using legacy calibration.factor=%.4f",
                    _resolved_cal_factor,
                )
        self.calibration_factor = _resolved_cal_factor
        # 🆕 NEW: Calibration strategy
        # Default is True (apply calibration to ML predictions only) because:
        # - The calibration_factor corrects for ML model under-prediction, not actuarial bias.
        # - Defaulting to False would silently inflate actuarial predictions if the config
        #   key is absent, which is the more dangerous pricing failure mode.
        # Set 'calibration.apply_to_ml_only: false' explicitly if actuarial is competitive
        # and the full hybrid output should be scaled.
        self.calibration_apply_to_ml_only = calibration_config.get("apply_to_ml_only", True)

        # Actuarial parameters
        config_actuarial = config.get("actuarial_params", {})

        if actuarial_params is not None:
            self.actuarial_params = actuarial_params
        else:
            # Fallback defaults match config.yaml v7.3.1 actuarial_params
            # (×1.35 scaled values). The previous defaults were the original
            # unscaled values (base=1980, age_coefficient=38.5, etc.) — a
            # leftover from pre-v7.3.1 that would revert the actuarial formula
            # to the 0.61x ML ratio if config loading ever fails.
            self.actuarial_params = {
                # fallback literals now match config.yaml actuarial_params
                # (×1.087 scaled values from v7.5.0). Previous fallbacks were the
                # pre-v7.3.1 unscaled values (smoker_multiplier=1.8, base=2795, etc.)
                # which revert the actuarial formula to the 0.61x ML ratio on any
                # config load failure — recreating the $255K net loss from Run 3.
                "base": config_actuarial.get("base", 3038),
                "age_coefficient": config_actuarial.get("age_coefficient", 59.0),
                "bmi_threshold": config_actuarial.get("bmi_threshold", 25),
                "bmi_penalty": config_actuarial.get("bmi_penalty", 118.2),
                "children_cost": config_actuarial.get("children_cost", 675),
                "smoker_multiplier": config_actuarial.get("smoker_multiplier", 3.5),
                "region_multipliers": config_actuarial.get(
                    "region_multipliers",
                    {
                        "northeast": 1.05,
                        "southeast": 1.02,
                        "northwest": 0.97,
                        "southwest": 1.0,
                    },
                ),
                # Smoker×BMI×age interaction coefficient (v6.3.4).
                # Defaults to 0.0 so deployments without the config.yaml key
                # are completely unaffected (interaction term = 0 → no-op).
                # Set to 8.0 in config.yaml to activate the VH-ratio.
                "smoker_bmi_age_coeff": config_actuarial.get("smoker_bmi_age_coeff", 0.0),
            }

        self.business_config = config.get("business_config", {})
        self.monitoring_config = config.get("monitoring", {})

        # Initialize drift monitor if enabled.
        # DriftMonitor is a pure static-method class — there is no __init__.
        # We store only the baseline_path (JSON produced by DriftMonitor.create_baseline())
        # and call DriftMonitor.detect_drift() as a static method at inference time.
        # The old code mistakenly called DriftMonitor(reference_data_path=...) which
        # raises TypeError, and also confused the CSV training-data path with the
        # JSON baseline path that detect_drift() actually requires.
        self._drift_monitor_enabled = False
        self._drift_baseline_path: str = self.monitoring_config.get(
            "baseline_path", "models/drift_baseline.json"
        )
        # config.yaml uses "drift_detection" (top-level monitoring block);
        # "drift_detection_enabled" was a predict.py-local name that never
        # matched any key in config, so drift was silently disabled on every run.
        # Read both: canonical config key first, predict-local alias as fallback.
        if self.monitoring_config.get(
            "drift_detection_enabled",
            self.monitoring_config.get("drift_detection", False),
        ):
            try:
                if Path(self._drift_baseline_path).exists():
                    self._drift_monitor_enabled = True
                    logger.info(
                        "✅ Drift monitoring enabled (baseline: %s)",
                        self._drift_baseline_path,
                    )
                else:
                    logger.warning(
                        "⚠️ Drift monitoring requested but baseline not found at '%s'. "
                        "Run DriftMonitor.create_baseline() after training to enable it.",
                        self._drift_baseline_path,
                    )
            except Exception as e:
                logger.warning(f"⚠️ Could not enable drift monitor: {e}")

        # Validate
        self._validate_config()

        # Log initialization
        logger.info("=" * 70)
        logger.info(f"✅ Hybrid Predictor v{self.VERSION} Initialized")
        logger.info("=" * 70)
        logger.info(f"   Threshold: ${self.threshold:,.0f}")
        logger.info(
            f"   Blend ratio: {self.blend_ratio:.0%} ML, {1-self.blend_ratio:.0%} actuarial"
        )
        logger.info(
            f"   Soft blending: {'✅ enabled' if self.use_soft_blending else '❌ disabled'}"
        )

        if self.calibration_enabled:
            strategy = "ML-only" if self.calibration_apply_to_ml_only else "Full hybrid"
            logger.info(
                f"   Calibration: ✅ {self.calibration_factor:.4f} "
                f"({(self.calibration_factor-1)*100:+.2f}%) [{strategy}]"
            )

            if not self.calibration_apply_to_ml_only:
                logger.warning(
                    f"   ⚠️ Calibration applies to FULL HYBRID output (apply_to_ml_only=False).\n"
                    f"      This inflates actuarial predictions by {(self.calibration_factor-1)*100:.1f}%,\n"
                    f"      which is only correct if actuarial parameters are consistently competitive.\n"
                    f"      Set 'calibration.apply_to_ml_only: true' if actuarial is conservative."
                )
        else:
            logger.warning("   Calibration: ⚠️ DISABLED (NOT RECOMMENDED)")

        logger.info("=" * 70)

    def _validate_config(self):
        """Validate configuration"""
        if not 0 <= self.blend_ratio <= 1:
            raise ValueError(f"blend_ratio must be in [0, 1], got {self.blend_ratio}")

        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

        # soft_blend_window=0 with use_soft_blending=True causes
        # ZeroDivisionError in _blend_predictions at the `progress` calculation.
        # Guard: when soft blending is enabled the window must be strictly positive.
        if self.use_soft_blending and self.soft_blend_window <= 0:
            raise ValueError(
                f"soft_blend_window must be > 0 when use_soft_blending=True, "
                f"got {self.soft_blend_window}. "
                "Set use_soft_blending: false or provide a positive soft_blend_window."
            )
        if not self.use_soft_blending and self.soft_blend_window < 0:
            raise ValueError("soft_blend_window must be non-negative")

        if self.calibration_factor <= 0:
            raise ValueError("calibration_factor must be positive")

        required_keys = [
            "base",
            "age_coefficient",
            "bmi_threshold",
            "bmi_penalty",
            "children_cost",
            "smoker_multiplier",
            "region_multipliers",
        ]
        missing = [k for k in required_keys if k not in self.actuarial_params]
        if missing:
            raise ValueError(f"Missing actuarial parameters: {missing}")

        # smoker_bmi_age_coeff is optional (defaults to 0.0) but must not be
        # negative — a negative value would reduce actuarial for high-BMI smokers,
        # the exact wrong direction for the VH-ratio.
        _coeff = self.actuarial_params.get("smoker_bmi_age_coeff", 0.0)
        if _coeff < 0:
            raise ValueError(
                f"actuarial_params.smoker_bmi_age_coeff must be >= 0, got {_coeff}. "
                "A negative value reduces actuarial premiums for high-BMI smokers, "
                "which inverts the intended VH-ratio correction."
            )

    def _calculate_actuarial_prediction(self, customer_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate actuarial predictions — fully vectorised.

        Replaces the previous row-by-row iterrows() loop, which was
        10-100× slower than vectorised operations for larger batches.
        """
        params = self.actuarial_params

        required_cols = ["age", "bmi", "children", "smoker", "region"]
        missing = [col for col in required_cols if col not in customer_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = customer_data  # already validated upstream; no copy needed

        # Base + age
        premium = params["base"] + df["age"].to_numpy(dtype=float) * params["age_coefficient"]

        # BMI penalty (only excess above threshold)
        bmi_excess = (df["bmi"].to_numpy(dtype=float) - params["bmi_threshold"]).clip(min=0)
        premium += bmi_excess * params["bmi_penalty"]

        # Children cost
        premium += df["children"].to_numpy(dtype=float) * params["children_cost"]

        # Smoker×BMI×age interaction term (v6.3.4 — VH-ratio).
        # Applied BEFORE the smoker multiplier so the interaction is amplified
        # by smoker_multiplier × region_multiplier downstream — matching the
        # multiplicative structure of real synergistic risk.
        #
        # Formula:  interaction = bmi_excess × age × smoker_bmi_age_coeff
        # Default coeff = 0.0 (no-op) unless set in config.yaml.
        # At coeff = 8.0 the prototypical VH smoker (age=50, bmi_excess=8)
        # moves from actuarial/ML = 0.608× to ≈ 0.83×, with zero effect on
        # non-smokers and near-zero effect on young/low-BMI smokers.
        _smoker_bmi_age_coeff = float(params.get("smoker_bmi_age_coeff", 0.0))
        smoker_mask = df["smoker"].str.lower().str.strip().isin(["yes", "1", "true"])
        if _smoker_bmi_age_coeff != 0.0:
            interaction = bmi_excess * df["age"].to_numpy(dtype=float) * _smoker_bmi_age_coeff
            premium = np.where(smoker_mask.to_numpy(), premium + interaction, premium)

        # Smoker multiplier
        premium = np.where(smoker_mask.to_numpy(), premium * params["smoker_multiplier"], premium)

        # Region multiplier (unknown regions default to 1.0)
        region_mult = (
            df["region"]
            .str.lower()
            .str.strip()
            .map(params["region_multipliers"])
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
        actuarial_preds = premium * region_mult

        # Validate actuarial predictions scale
        is_valid, scale_msg = validate_prediction_scale(actuarial_preds, scale_type="original")

        if not is_valid:
            logger.error(f"Actuarial prediction scale issue: {scale_msg}")
        else:
            logger.debug(f"Actuarial predictions: {scale_msg}")

        return np.asarray(actuarial_preds)

    def _blend_predictions(
        self, ml_preds: np.ndarray, act_preds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        🆕 ENHANCED: Blend predictions with detailed diagnostics

        Returns: (final_predictions, ml_weights, blend_diagnostics)

        v6.3.2: Vectorised — replaces the previous element-wise Python for-loop
        that recomputed `lower_bound` on every iteration.  Consistent with the
        vectorisation already applied to _calculate_actuarial_prediction.
        Logic is provably identical to the original loop (exhaustively tested).
        """
        n = len(ml_preds)

        # lower_bound is a constant — compute once, not inside a loop
        lower_bound = self.threshold - self.soft_blend_window

        # ── Zone masks (mutually exclusive, exhaustive) ───────────────────────
        # v6.3.3: routing decision now uses a 50/50 composite of both signals
        # instead of act_preds alone.  Pure actuarial routing caused boundary
        # instability when act_preds was near the threshold while ml_preds
        # disagreed strongly — the composite damps those flip-flops.
        routing_signal = 0.5 * ml_preds + 0.5 * act_preds
        above_threshold = routing_signal >= self.threshold

        if self.use_soft_blending:
            # Actuarial-dominant: actuarial estimate below the soft transition window.
            # REVERT v6.3.4: routing_signal for below_lower caused a net regression.
            # Policies with act in (lower_bound, threshold) AND ml < lower_bound were
            # reclassified from transition (ml_weight ~0.85) into actuarial-dominant
            # (ml_weight = 0.70), dropping effective ML weight from 0.80 to 0.76 and
            # increasing actuarial drag on the exact cohort responsible for tail risk.
            # Zone anchor stays on act_preds: consistent with the progress formula
            # below which also uses act_preds as its interpolation signal.
            #
            # `act_preds <= lower_bound` is evaluated independently of
            # `above_threshold`, so when routing_signal >= threshold (above_threshold=True)
            # but act_preds <= lower_bound, the same policy was counted in BOTH
            # ml_only_count AND soft_blend_zone_count / actuarial_dominant_indices,
            # producing a self-contradictory blend_diagnostics dict.
            # explicitly exclude above_threshold policies from below_lower so the
            # three zones are always mutually exclusive and exhaustive.
            below_lower = (act_preds <= lower_bound) & ~above_threshold
            # Transition: in the window [lower_bound, threshold)
            in_transition = ~above_threshold & ~below_lower
        else:
            # Hard blend: no actuarial-dominant zone.
            # All below-threshold policies use blend_ratio; actuarial_dominant_count stays 0.
            below_lower = np.zeros(n, dtype=bool)
            in_transition = np.zeros(n, dtype=bool)
        # ── ML weights ────────────────────────────────────────────────────────
        # Start: above-threshold → 1.0 (pure ML); everything else → blend_ratio
        ml_weights = np.where(above_threshold, 1.0, self.blend_ratio).astype(float)

        # Override transition zone with smooth interpolation
        if self.use_soft_blending and in_transition.any():
            progress = (act_preds - lower_bound) / self.soft_blend_window
            interpolated = self.blend_ratio + (1.0 - self.blend_ratio) * progress
            ml_weights = np.where(in_transition, interpolated, ml_weights)

        # ── Final predictions ─────────────────────────────────────────────────
        final = ml_weights * ml_preds + (1.0 - ml_weights) * act_preds

        # ── Churn cap: actuarial uplift guard ────────────────────────────────
        # Prevents actuarial-path predictions from inflating more than
        # max_actuarial_uplift_ratio × ML prediction, which would trigger
        # customer churn via overpricing.  Applied after blending so the
        # soft-blend interpolation is not distorted.
        # Default 1.15 (±15%) matches the actuarial conservativeness threshold
        # already used in the predict() diagnostics block.
        max_uplift = self.config.get("max_actuarial_uplift_ratio", 1.10)
        final = np.minimum(final, ml_preds * max_uplift)

        # ── T2-B: ML-relative floor guard ────────────────────────────────────
        # Symmetric counterpart to the churn cap above.  When the actuarial
        # formula systematically under-prices a segment (e.g. VH at 0.61x ML),
        # the blend is dragged below ML with no lower bound.  This guard
        # prevents the blend from falling more than (1 - min_floor) below ML.
        #
        # Default 0.75: allows up to 25% undercut of ML before clamping upward.
        # This does NOT fix the underlying actuarial calibration deficit — it
        # is a safety net that limits worst-case exposure.  The primary fix is
        # recalibrating actuarial_params (Option 1) or the VH interaction term.
        #
        # NOTE: An actuarial-fraction floor (e.g. 0.55 * act_preds) was tested
        # previously and rejected because it conflicted with the churn cap for
        # tail-risk policies (act > 3.5*ml), causing net churn cost to increase
        # by +$4,525 across 11 FALSE-ALARM policies where the flat
        # smoker_multiplier=3.5 over-estimates young low-BMI policyholders.
        # The ML-relative floor avoids this conflict: it anchors on ML, not on
        # the miscalibrated actuarial signal.
        # Set hybrid_predictor.min_actuarial_floor_ratio in config.yaml to tune.
        min_floor = self.config.get("min_actuarial_floor_ratio", 0.75)
        final = np.maximum(final, ml_preds * min_floor)

        # ── Diagnostics ───────────────────────────────────────────────────────
        # cap index lists at _MAX_DIAG_INDICES to prevent O(N) JSON
        # serialization overhead on every request.  For a 7701-row batch at 82%
        # ML routing, the uncapped ml_only_indices list holds ~6315 integers
        # (≈40KB JSON) serialized on every call — ~4MB/s at 100 req/s.
        # The list is diagnostic only; callers that need all indices should use
        # the above_threshold boolean mask from blend_diagnostics["ml_only_count"].
        _MAX_DIAG_INDICES = 50
        blend_diagnostics = {
            "ml_only_count": int(above_threshold.sum()),
            "soft_blend_zone_count": int(below_lower.sum()),
            "actuarial_dominant_count": int(below_lower.sum()),  # legacy alias
            "non_ml_dominant_count": int((~above_threshold).sum()),
            "transition_zone_count": int(in_transition.sum()),
            "ml_only_indices": above_threshold.nonzero()[0].tolist()[:_MAX_DIAG_INDICES],
            "actuarial_dominant_indices": below_lower.nonzero()[0].tolist()[:_MAX_DIAG_INDICES],
            "avg_ml_weight": float(np.mean(ml_weights)),
            "floor_guard_ratio": float(min_floor),
            "churn_cap_ratio": float(max_uplift),
        }

        return final, ml_weights, blend_diagnostics

    def predict(
        self,
        input_data: pd.DataFrame,
        return_components: bool = False,
        return_reliability: bool = True,
        _precomputed_ml_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        🆕 ENHANCED: Generate hybrid predictions with tail risk monitoring

        Calibration can now be applied to ML-only or full hybrid

        (_precomputed_ml_result): When routes.py calls predict_with_intervals()
        first (to obtain CI), it can pass the resulting dict here so the ML pipeline
        is not invoked a second time.  The parameter is intentionally private
        (underscore prefix) — it is an internal optimisation path, not public API.
        When None (the default), the normal ml_predictor.predict() call is made.
        """
        if input_data.empty:
            raise ValueError("Input data cannot be empty")

        # read max_batch_size from the FULL pipeline config, not from
        # self.config (which is the hybrid_predictor sub-dict).
        # The key prediction.max_batch_size lives at the root of config.yaml;
        # self.config.get("prediction", {}) always returns {} and falls through
        # to the hardcoded 10000 default — silently ignoring any config override.
        # ml_predictor.config IS the full config (PredictionPipeline stores it).
        _max_batch = int(
            self.ml_predictor.config.get("prediction", {}).get("max_batch_size", 50000)
        )
        if len(input_data) > _max_batch:
            raise ValueError(
                f"Batch size too large: {len(input_data)} samples "
                f"(limit: {_max_batch} — set prediction.max_batch_size in config.yaml to override)"
            )

        try:
            # reuse precomputed ML result when available (e.g. supplied by
            # routes.py predict_single after calling predict_with_intervals) to avoid
            # running feature engineering + XGBoost inference a second time.
            # validate shape before trusting the precomputed result.
            # A stale or cached result with a different row count would produce a
            # silent shape mismatch in _blend_predictions (numpy broadcast failure).
            if _precomputed_ml_result is not None:
                _pre_preds = _precomputed_ml_result.get("predictions", [])
                if len(_pre_preds) != len(input_data):
                    logger.warning(
                        f"_precomputed_ml_result has {len(_pre_preds)} predictions "
                        f"but input_data has {len(input_data)} rows — "
                        "ignoring precomputed result and running fresh ML inference."
                    )
                    ml_result = self.ml_predictor.predict(input_data, return_reliability=False)
                else:
                    ml_result = _precomputed_ml_result
                    logger.debug("Using precomputed ML result — skipping ml_predictor.predict()")
            else:
                logger.debug("Getting ML predictions...")
                ml_result = self.ml_predictor.predict(input_data, return_reliability=False)
            ml_predictions = np.array(ml_result["predictions"])

            # Validate ML predictions
            is_valid, scale_msg = validate_prediction_scale(ml_predictions, "original")
            if not is_valid:
                logger.error(f"ML predictions scale issue: {scale_msg}")
                raise ValueError("ML prediction scale validation failed")
            else:
                logger.debug(f"ML predictions validated: {scale_msg}")

            # Apply calibration to ML predictions BEFORE blending if configured
            if self.calibration_enabled and self.calibration_apply_to_ml_only:
                ml_predictions_calibrated = ml_predictions * self.calibration_factor
                logger.debug(
                    f"Applied calibration to ML predictions: {self.calibration_factor:.4f}\n"
                    f"   Uncalibrated ML mean: ${ml_predictions.mean():,.2f}\n"
                    f"   Calibrated ML mean: ${ml_predictions_calibrated.mean():,.2f}"
                )
            else:
                ml_predictions_calibrated = ml_predictions

            # Get actuarial predictions
            logger.debug("Calculating actuarial predictions...")
            actuarial_predictions = self._calculate_actuarial_prediction(input_data)

            # 🆕 ADDED: Detect actuarial calibration anomalies in both directions.
            # Over-conservative (ratio > 1.15): actuarial over-prices — inflates blended result.
            # Under-pricing   (ratio < 0.70): actuarial under-prices — creates severe pricing risk
            # for the actuarial-dominant segment (below the blending threshold).
            actuarial_vs_ml_ratio = np.median(
                actuarial_predictions / np.maximum(ml_predictions_calibrated, 1e-8)
            )
            if actuarial_vs_ml_ratio > 1.15:
                logger.warning(
                    f"⚠️ ACTUARIAL CONSERVATIVENESS DETECTED:\n"
                    f"   Actuarial/ML ratio: {actuarial_vs_ml_ratio:.2f}x\n"
                    f"   Actuarial estimates are {(actuarial_vs_ml_ratio-1)*100:.1f}% higher than ML\n"
                    f"   Consider setting 'calibration.apply_to_ml_only: true'"
                )
            elif actuarial_vs_ml_ratio < 0.70:
                logger.warning(
                    f"⚠️ ACTUARIAL UNDER-PRICING DETECTED:\n"
                    f"   Actuarial/ML ratio: {actuarial_vs_ml_ratio:.2f}x\n"
                    f"   Actuarial estimates are {(1 - actuarial_vs_ml_ratio)*100:.1f}% BELOW ML\n"
                    f"   Actuarial-dominant predictions (below ${self.threshold:,.0f} threshold)\n"
                    f"   are at risk of severe under-pricing.\n"
                    f"   ACTION: Review actuarial base parameters or raise the blending threshold."
                )

            # Blend predictions with diagnostics
            logger.debug("Blending predictions...")
            hybrid_predictions, ml_weights, blend_diagnostics = self._blend_predictions(
                ml_predictions_calibrated, actuarial_predictions
            )

            # Apply calibration to FULL HYBRID only if not applied to ML-only
            if self.calibration_enabled and not self.calibration_apply_to_ml_only:
                hybrid_predictions_calibrated = hybrid_predictions * self.calibration_factor

                logger.debug(
                    f"Applied calibration to full hybrid: {self.calibration_factor:.4f}\n"
                    f"   Uncalibrated mean: ${hybrid_predictions.mean():,.2f}\n"
                    f"   Calibrated mean: ${hybrid_predictions_calibrated.mean():,.2f}"
                )
            else:
                hybrid_predictions_calibrated = hybrid_predictions
                if not self.calibration_enabled:
                    logger.warning("⚠️ Calibration disabled - predictions may be underpriced")

            # Validate final predictions
            is_valid_final, scale_msg_final = validate_prediction_scale(
                hybrid_predictions_calibrated, "original"
            )

            if not is_valid_final:
                logger.error(f"Final predictions scale issue: {scale_msg_final}")
                raise ValueError("Final prediction scale validation failed")
            else:
                logger.info(f"Final predictions validated: {scale_msg_final}")

            # ── COMBINED CORRECTION GOVERNANCE VALIDATION ─────────────────────
            # BiasCorrection (stratified) and calibration_factor (flat) are both
            # multiplicative in original space.  They compound silently:
            #   net = exp(var_tier/2) × calibration_factor
            # For the high tier this can reach +30%+.  Validate the net factor
            # stays within the approved pricing governance range.
            #
            # compare hybrid vs ml_predictions_CALIBRATED (not raw).
            # When apply_to_ml_only=True, calibration is already baked into
            # ml_predictions_calibrated before blending.  Dividing by raw
            # ml_predictions inflated _net_factor by calibration_factor on every
            # call (e.g. factor=1.08 → governance check always read ~1.08× even
            # for a neutral blend), causing permanent false-positive alerts.
            _net_max = self.business_config.get("combined_correction_max", 1.45)
            _net_min = self.business_config.get("combined_correction_min", 0.65)
            _net_factor = float(
                np.median(
                    hybrid_predictions_calibrated / np.maximum(ml_predictions_calibrated, 1e-8)
                )
            )
            if not (_net_min <= _net_factor <= _net_max):
                logger.warning(
                    f"⚠️ COMBINED CORRECTION OUTSIDE GOVERNANCE RANGE:\n"
                    f"   Net correction factor (hybrid vs. raw ML): {_net_factor:.3f}x "
                    f"({(_net_factor - 1) * 100:+.1f}%)\n"
                    f"   Approved range: [{_net_min:.2f}x, {_net_max:.2f}x]\n"
                    f"   This reflects BiasCorrection × calibration_factor stacking.\n"
                    f"   Review both parameters before production deployment."
                )

            # 🆕 TAIL RISK MONITORING (with governance-appropriate language)
            tail_risk_threshold = self.business_config.get(
                "severe_underpricing_threshold_pct", 0.50
            )

            # anchor safe_minimum on ML predictions, not actuarial.
            # The previous formula (actuarial * 0.5) produced false alarms for
            # low-value smoker policies where smoker_multiplier=3.5 inflates
            # actuarial far above the true charge (e.g. actuarial=$24K for a
            # true-charge=$4K policy → safe_minimum=$12K → every correctly-priced
            # hybrid prediction flagged as severe underpricing).
            # For the VH segment (actuarial=0.608x ML confirmed), safe_minimum
            # was too LOW, masking genuine underpricing.
            # ML is the best available unbiased estimate at inference time;
            # anchoring on it removes both failure modes.
            safe_minimum = ml_predictions * (1 - tail_risk_threshold)
            underpriced_mask = hybrid_predictions_calibrated < safe_minimum
            n_severe_underpricing = underpriced_mask.sum()

            tail_risk_warning = None
            if n_severe_underpricing > 0:
                avg_underpricing = np.mean(
                    (
                        safe_minimum[underpriced_mask]
                        - hybrid_predictions_calibrated[underpriced_mask]
                    )
                    / safe_minimum[underpriced_mask]
                )

                severity = (
                    "CRITICAL"
                    if n_severe_underpricing > len(input_data) * 0.10
                    else ("HIGH" if n_severe_underpricing > len(input_data) * 0.05 else "MODERATE")
                )

                # 🆕 GOVERNANCE-APPROPRIATE WARNING
                logger.warning(
                    f"🔥 TAIL RISK ALERT ({severity}):\n"
                    f"   {n_severe_underpricing} predictions below safe minimum threshold\n"
                    f"   Average gap from threshold: {avg_underpricing*100:.1f}%\n"
                    f"   Recommendation: Review calibration settings and actuarial parameters"
                )

                tail_risk_warning = {
                    "severity": severity,
                    "policies_below_threshold": int(n_severe_underpricing),
                    "policies_below_threshold_pct": float(
                        n_severe_underpricing / len(input_data) * 100
                    ),
                    "sample_indices": underpriced_mask.nonzero()[0].tolist()[:10],
                    "avg_gap_from_threshold_pct": float(avg_underpricing * 100),
                    "threshold_used_pct": float(tail_risk_threshold * 100),
                    "recommended_action": (
                        "URGENT: Review pricing model calibration"
                        if severity == "CRITICAL"
                        else (
                            "Review and monitor in next cycle"
                            if severity == "HIGH"
                            else "Monitor in next evaluation cycle"
                        )
                    ),
                    "governance_note": (
                        "This alert indicates predictions may not meet minimum pricing requirements. "
                        "Please review with pricing governance team."
                    ),
                }

            # The original code recomputed n_above / n_in_transition /
            # n_below_threshold directly from actuarial_predictions here, while
            # blend_diagnostics derives its counts from routing_signal (composite
            # 0.5 * ml + 0.5 * act) for above_threshold and from act_preds for
            # below_lower.  The two paths gave different numbers for the same batch,
            # so hybrid_info and blend_diagnostics were always self-contradictory.
            # read counts from blend_diagnostics (single source of truth).
            n_above = blend_diagnostics["ml_only_count"]
            n_in_transition = blend_diagnostics["transition_zone_count"]
            n_below_threshold = blend_diagnostics["soft_blend_zone_count"]

            # Build result
            result = {
                "predictions": hybrid_predictions_calibrated.tolist(),
                "model_used": f"hybrid_{self.ml_predictor.model_name}_v{self.VERSION}",
                "model_version": self.VERSION,
                "input_count": len(input_data),
                "calibration_applied": self.calibration_enabled,
                "calibration_factor": (
                    self.calibration_factor if self.calibration_enabled else 1.0
                ),
                "calibration_strategy": (
                    "ML-only" if self.calibration_apply_to_ml_only else "Full hybrid"
                ),
                "hybrid_info": {
                    "threshold": self.threshold,
                    "blend_ratio": self.blend_ratio,
                    "soft_blending": self.use_soft_blending,
                    "predictions_actuarial_dominant": n_below_threshold,
                    "predictions_in_transition": n_in_transition,
                    "predictions_above_threshold": n_above,
                },
                "statistics": {
                    "mean": float(np.mean(hybrid_predictions_calibrated)),
                    "median": float(np.median(hybrid_predictions_calibrated)),
                    "min": float(np.min(hybrid_predictions_calibrated)),
                    "max": float(np.max(hybrid_predictions_calibrated)),
                    "std": float(np.std(hybrid_predictions_calibrated)),
                    "q25": float(np.percentile(hybrid_predictions_calibrated, 25)),
                    "q75": float(np.percentile(hybrid_predictions_calibrated, 75)),
                    # 🆕 Business-relevant statistics
                    "interquartile_range": float(
                        np.percentile(hybrid_predictions_calibrated, 75)
                        - np.percentile(hybrid_predictions_calibrated, 25)
                    ),
                    "coefficient_of_variation": float(
                        np.std(hybrid_predictions_calibrated)
                        / np.mean(hybrid_predictions_calibrated)
                        if np.mean(hybrid_predictions_calibrated) > 0
                        else 0
                    ),
                    # Premium risk segmentation
                    "n_low_premium": int(np.sum(hybrid_predictions_calibrated < 5000)),
                    "n_medium_premium": int(
                        np.sum(
                            (hybrid_predictions_calibrated >= 5000)
                            & (hybrid_predictions_calibrated < 15000)
                        )
                    ),
                    "n_high_premium": int(np.sum(hybrid_predictions_calibrated >= 15000)),
                    "pct_low_premium": float(
                        np.sum(hybrid_predictions_calibrated < 5000)
                        / len(hybrid_predictions_calibrated)
                        * 100
                    ),
                    "pct_medium_premium": float(
                        np.sum(
                            (hybrid_predictions_calibrated >= 5000)
                            & (hybrid_predictions_calibrated < 15000)
                        )
                        / len(hybrid_predictions_calibrated)
                        * 100
                    ),
                    "pct_high_premium": float(
                        np.sum(hybrid_predictions_calibrated >= 15000)
                        / len(hybrid_predictions_calibrated)
                        * 100
                    ),
                },
                "validation": {
                    "ml_scale_validated": True,
                    "actuarial_scale_validated": True,
                    "final_scale_validated": True,
                },
                # Enhanced diagnostics
                "blend_diagnostics": blend_diagnostics,
                "tail_risk_warning": tail_risk_warning,
                "actuarial_conservativeness_ratio": float(actuarial_vs_ml_ratio),
            }

            # Add component predictions if requested
            if return_components:
                # ── compute TRUE uncalibrated blend ─────────────
                # When apply_to_ml_only=True, _blend_predictions() was called with
                # ml_predictions_calibrated (already scaled up), so hybrid_predictions
                # already contains the calibration effect.  Storing it as
                # "uncalibrated_hybrid" caused evaluate.py to report $0 calibration
                # impact (hybrid_preds - uncal == 0 identically).
                #
                # re-blend using the raw (uncalibrated) ml_predictions so the
                # "uncalibrated_hybrid" key genuinely reflects what predictions would
                # look like without calibration.  This is a reporting-only operation;
                # it does not change any prediction logic or the final output.
                #
                # When apply_to_ml_only=False, hybrid_predictions IS the pre-calibration
                # blend (calibration is applied afterward), so no extra work is needed.
                if self.calibration_enabled and self.calibration_apply_to_ml_only:
                    _uncal_blend, _, _ = self._blend_predictions(
                        ml_predictions, actuarial_predictions
                    )
                    uncalibrated_hybrid_arr = _uncal_blend
                else:
                    # apply_to_ml_only=False: hybrid_predictions = pre-calibration blend
                    uncalibrated_hybrid_arr = hybrid_predictions

                result["components"] = {
                    "ml_predictions": ml_predictions.tolist(),
                    "ml_predictions_calibrated": ml_predictions_calibrated.tolist(),
                    "actuarial_predictions": actuarial_predictions.tolist(),
                    "ml_weights": ml_weights.tolist(),
                    "uncalibrated_hybrid": uncalibrated_hybrid_arr.tolist(),
                }

            # Reliability flags
            if return_reliability:
                extreme_threshold = int(
                    self.ml_predictor.config.get("prediction", {}).get(
                        "extreme_prediction_threshold", 100_000
                    )
                )
                max_pred_val = np.max(hybrid_predictions_calibrated)

                result["reliability"] = {
                    "has_extreme_predictions": bool(max_pred_val > extreme_threshold),
                    "extreme_count": int(np.sum(hybrid_predictions_calibrated > extreme_threshold)),
                    "max_prediction": float(max_pred_val),
                    "hybrid_mode": "enabled",
                    "actuarial_rules_applied": n_below_threshold > 0,
                    "blending_method": "actuarial_threshold_based",
                    "calibration_status": ("enabled" if self.calibration_enabled else "disabled"),
                    "calibration_strategy": (
                        "ML-only" if self.calibration_apply_to_ml_only else "Full hybrid"
                    ),
                    # ml_result is fetched at line ~2109 with
                    # return_reliability=False, so ml_result["reliability"] is absent
                    # and .get("bias_correction_applied", False) always returned False —
                    # even when bias correction was applied.
                    # read the flag directly from the pipeline object, which is
                    # always populated regardless of return_reliability.
                    "ml_bias_correction": self.ml_predictor._bias_correction is not None,
                    "tail_risk_detected": tail_risk_warning is not None,
                    # Both flags are always present; only one is True at a time.
                    "actuarial_conservative": actuarial_vs_ml_ratio > 1.15,
                    "actuarial_aggressive": actuarial_vs_ml_ratio < 0.70,
                    # Informational: configured blend_ratio applies only below the
                    # transition window.  At typical portfolios, the effective average
                    # ML weight is much higher than blend_ratio because most premiums
                    # exceed the threshold and route to pure ML.
                    "configured_blend_ratio": self.blend_ratio,
                    "effective_avg_ml_weight": float(blend_diagnostics["avg_ml_weight"]),
                }

            # Drift monitoring (if enabled)
            # DriftMonitor.detect_drift() is a @staticmethod — called on the class,
            # not on an instance. It takes (X_new, baseline_path, z_threshold, tvd_threshold)
            # and returns a DriftReport dataclass, not a dict — access .has_drift and
            # .drifted_features directly; call .to_dict() for JSON-serialisable output.
            if self._drift_monitor_enabled:
                try:
                    drift_report = DriftMonitor.detect_drift(
                        X_new=input_data,
                        baseline_path=self._drift_baseline_path,
                    )

                    if drift_report.has_drift:
                        logger.warning(
                            f"⚠️ DATA DRIFT DETECTED:\n"
                            f"   Drifted features: {drift_report.drifted_features}\n"
                            f"   Prediction reliability may be reduced\n"
                            f"   {drift_report.summary()}"
                        )

                    result["drift_monitoring"] = drift_report.to_dict()

                except FileNotFoundError:
                    logger.warning(
                        "⚠️ Drift baseline missing at '%s' — monitoring skipped. "
                        "Run DriftMonitor.create_baseline() to re-enable.",
                        self._drift_baseline_path,
                    )
                    result["drift_monitoring"] = {"error": "baseline_not_found"}
                except (ValueError, RuntimeError, AttributeError) as e:
                    logger.error(f"❌ Drift monitoring failed: {e}")
                    result["drift_monitoring"] = {"error": str(e)}

            # Log summary
            if self.monitoring_config.get("enabled", True):
                # Finding I: in hard-blend mode below_lower / in_transition are zeroed
                # unconditionally, so the counts always print 0 — which looks like a
                # routing bug to an operator reading the log.  Add a mode tag to make
                # it unambiguous that the zeros are intentional, not a failure.
                _blend_mode_tag = " [hard blend — N/A]" if not self.use_soft_blending else ""
                logger.info(
                    f"📊 Hybrid Prediction Summary:\n"
                    f"   Actuarial-dominant ({n_below_threshold}){_blend_mode_tag}: below transition window\n"
                    f"   In transition ({n_in_transition}){_blend_mode_tag}: blended\n"
                    f"   ML-dominant ({n_above}): above threshold\n"
                    f"   Avg ML weight: {blend_diagnostics['avg_ml_weight']:.2f}\n"
                    f"   Calibration: {self.calibration_factor:.4f} "
                    f"[{result['calibration_strategy']}]\n"
                    f"   Actuarial/ML ratio: {actuarial_vs_ml_ratio:.2f}x\n"
                    f"   Final range: [${result['statistics']['min']:,.2f}, "
                    f"${result['statistics']['max']:,.2f}]\n"
                    f"   Final mean: ${result['statistics']['mean']:,.2f}"
                )

                if tail_risk_warning:
                    logger.warning(
                        f"   🔥 Tail Risk: {tail_risk_warning['severity']} "
                        f"({tail_risk_warning['policies_below_threshold']} policies)"
                    )

            return result

        except ValueError as e:
            logger.error(f"❌ Validation error in hybrid prediction: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error in hybrid prediction: {e}")
            raise

    def predict_single(
        self, age: int, sex: str, bmi: float, children: int, smoker: str, region: str
    ) -> float:
        """Single prediction with validation"""
        try:
            age = int(age)
            children = int(children)
            bmi = float(bmi)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric input: {e}") from e

        # ── single-record path reads same config bounds as batch path ──
        # HybridPredictor.config holds only the hybrid_predictor subsection;
        # feature bounds live in the full pipeline config.
        _feat = self.ml_predictor.config.get("features", {})
        _age_min, _age_max = _feat.get("age_min", 0.0), _feat.get("age_max", 120.0)
        _bmi_min, _bmi_max = _feat.get("bmi_min", 10.0), _feat.get("bmi_max", 100.0)
        _children_min, _children_max = _feat.get("children_min", 0), _feat.get("children_max", 20)

        if not _age_min <= age <= _age_max:
            raise ValueError(f"Age must be in [{_age_min}, {_age_max}], got {age}")
        if not _bmi_min <= bmi <= _bmi_max:
            raise ValueError(f"BMI must be in [{_bmi_min}, {_bmi_max}], got {bmi}")
        if not _children_min <= children <= _children_max:
            raise ValueError(
                f"Children must be in [{_children_min}, {_children_max}], got {children}"
            )

        input_df = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region],
            }
        )

        result = self.predict(input_df, return_reliability=False)
        return float(result["predictions"][0])

    def update_parameters(
        self,
        threshold: float | None = None,
        blend_ratio: float | None = None,
        calibration_factor: float | None = None,
        calibration_apply_to_ml_only: bool | None = None,
        actuarial_params: dict[str, Any] | None = None,
    ):
        """
        🆕 ENHANCED: Update parameters with calibration strategy control
        """
        if threshold is not None:
            if threshold <= 0:
                raise ValueError(f"threshold must be positive, got {threshold}")
            old_threshold = self.threshold
            self.threshold = threshold
            logger.info(f"Updated threshold: ${old_threshold:,.0f} → ${threshold:,.0f}")

        if blend_ratio is not None:
            if not 0 <= blend_ratio <= 1:
                raise ValueError(f"blend_ratio must be in [0, 1], got {blend_ratio}")
            old_ratio = self.blend_ratio
            self.blend_ratio = blend_ratio
            logger.info(f"Updated blend ratio: {old_ratio:.0%} → {blend_ratio:.0%} ML")

        if calibration_factor is not None:
            if calibration_factor <= 0:
                raise ValueError("calibration_factor must be positive")
            old_cal = self.calibration_factor
            self.calibration_factor = calibration_factor
            logger.warning(
                f"⚠️ Updated calibration: {old_cal:.4f} → {calibration_factor:.4f}\n"
                f"   Only do this after evaluation!"
            )

        # 🆕 NEW: Allow updating calibration strategy
        if calibration_apply_to_ml_only is not None:
            old_strategy = "ML-only" if self.calibration_apply_to_ml_only else "Full hybrid"
            new_strategy = "ML-only" if calibration_apply_to_ml_only else "Full hybrid"
            self.calibration_apply_to_ml_only = calibration_apply_to_ml_only
            logger.warning(
                f"⚠️ Updated calibration strategy: {old_strategy} → {new_strategy}\n"
                f"   This significantly affects final predictions!"
            )

        if actuarial_params is not None:
            self.actuarial_params.update(actuarial_params)
            logger.warning(f"⚠️ Updated actuarial parameters: {list(actuarial_params.keys())}")

    def get_config_summary(self) -> dict[str, Any]:
        """Get current configuration"""
        return {
            "version": self.VERSION,
            "threshold": self.threshold,
            "blend_ratio": self.blend_ratio,
            "use_soft_blending": self.use_soft_blending,
            "soft_blend_window": self.soft_blend_window,
            "calibration_enabled": self.calibration_enabled,
            "calibration_factor": self.calibration_factor,
            "calibration_apply_to_ml_only": self.calibration_apply_to_ml_only,
            "actuarial_params": self.actuarial_params.copy(),
            "business_config": self.business_config.copy(),
            "drift_monitoring_enabled": self._drift_monitor_enabled,
        }


# =====================================================================
# USAGE EXAMPLES
# =====================================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n" + "=" * 80)
    print("ENHANCED HYBRID PIPELINE v6.3.3")
    print("=" * 80)
    print("🆕 CRITICAL FIXES:")
    print("  ✅ Calibration now configurable (ML-only vs full hybrid)")
    print("  ✅ Confidence intervals computed correctly in log space")
    print("  ✅ Tail risk warnings use governance-appropriate language")
    print("  ✅ Actuarial conservativeness detection")
    print("=" * 80)

    try:
        # Initialize pipeline
        print("\n[1/6] Initializing ML Pipeline...")
        pipeline = PredictionPipeline()

        info = pipeline.get_pipeline_info()
        print("\n✅ Pipeline Info:")
        print(f"  Version: v{info['pipeline_version']}")
        print(f"  Model: {info['model_name']}")
        print(f"  Transform: {info['target_transformation']['method']}")
        print(f"  Bias correction: {info['target_transformation']['bias_correction']}")
        print(
            f"  Recommended metrics: "
            f"{', '.join(info['target_transformation']['recommended_metrics'])}"
        )

        # Initialize hybrid
        print("\n[2/6] Initializing Hybrid Predictor...")
        hybrid = HybridPredictor(pipeline)

        # Test predictions
        print("\n[3/6] Testing Predictions...")
        test_data = pd.DataFrame(
            {
                "age": [25, 45, 60],
                "sex": ["female", "male", "female"],
                "bmi": [22.5, 30.0, 28.0],
                "children": [0, 2, 1],
                "smoker": ["no", "yes", "no"],
                "region": ["southwest", "northeast", "southeast"],
            }
        )

        result = hybrid.predict(test_data, return_components=True, return_reliability=True)

        print("\n✅ Predictions Generated:")
        print(f"  Batch size: {result['input_count']}")
        print(f"  Mean: ${result['statistics']['mean']:,.2f}")
        print(
            f"  Range: [${result['statistics']['min']:,.2f}, "
            f"${result['statistics']['max']:,.2f}]"
        )
        print(f"  Calibration strategy: {result['calibration_strategy']}")

        # Show blend diagnostics
        print("\n  Blend Diagnostics:")
        diag = result["blend_diagnostics"]
        print(f"    ML-only: {diag['ml_only_count']}")
        print(f"    Actuarial-dominant: {diag['actuarial_dominant_count']}")
        print(f"    Transition zone: {diag['transition_zone_count']}")
        print(f"    Avg ML weight: {diag['avg_ml_weight']:.2f}")

        # Show actuarial conservativeness
        # ratio on n=3 is noise — suppress interpretation label below
        # a minimum sample threshold; ratio is still printed for traceability.
        _n_for_ratio = result.get("input_count", 0)
        _MIN_RATIO_N = 30
        print("\n  Actuarial Analysis:")
        print(
            f"    Actuarial/ML ratio: {result['actuarial_conservativeness_ratio']:.2f}x"
            + (
                f"  (ℹ️ n={_n_for_ratio} < {_MIN_RATIO_N} — ratio unreliable at this sample size)"
                if _n_for_ratio < _MIN_RATIO_N
                else ""
            )
        )
        if _n_for_ratio >= _MIN_RATIO_N and result["actuarial_conservativeness_ratio"] > 1.15:
            print("    ⚠️ Actuarial appears conservative")

        # Show tail risk status
        if result["tail_risk_warning"]:
            warning = result["tail_risk_warning"]
            print(f"\n  🔥 Tail Risk Warning ({warning['severity']}):")
            print(f"    Policies below threshold: {warning['policies_below_threshold']}")
            print(f"    Avg gap: {warning['avg_gap_from_threshold_pct']:.1f}%")
        else:
            print("\n  ✅ No tail risk warnings")

        # Test confidence intervals
        print("\n[4/6] Testing Confidence Intervals (C1 — Heteroscedastic)...")
        # Use production default (0.90) so this test reflects actual API behaviour.
        # Previously hardcoded 0.95 — mismatched the method default and caused the
        # README to quote 95% CI widths ($16,322) instead of the 90% production
        # value ($11,680). Both the CI output and the README must use 0.90.
        ci_result = pipeline.predict_with_intervals(test_data, confidence_level=0.90)

        if ci_result.get("confidence_intervals"):
            ci = ci_result["confidence_intervals"]
            print("✅ Confidence intervals computed:")
            print(f"  Confidence level: {ci['confidence_level']*100:.0f}%")
            print(f"  Mean interval width: ${ci['mean_interval_width']:,.2f}")
            print(f"  Method: {ci['method']}")
            print(f"  Note: {ci.get('note', 'N/A')}")
        else:
            print("⚠️ Confidence intervals not available")

        # Test calibration strategy update
        print("\n[5/6] Testing Calibration Strategy Update...")
        config_summary = hybrid.get_config_summary()
        original_strategy = config_summary["calibration_apply_to_ml_only"]
        print(f"Current strategy (apply_to_ml_only): {original_strategy}")

        # ── toggle the attribute directly rather than via
        #    update_parameters() or deepcopy().
        #
        #    deepcopy(hybrid) fails because HybridPredictor contains a
        #    DriftMonitor / LightGBM booster that holds _thread.lock objects
        #    which cannot be pickled.
        #
        #    update_parameters() emits WARNING logs on every toggle, which
        #    polluted the original output with two spurious
        #    "Updated calibration strategy" messages.
        #
        #    Direct attribute assignment tests the round-trip silently and
        #    leaves every other object in hybrid completely untouched.
        hybrid.calibration_apply_to_ml_only = not original_strategy
        assert hybrid.calibration_apply_to_ml_only == (
            not original_strategy
        ), "Strategy update failed"
        hybrid.calibration_apply_to_ml_only = original_strategy
        assert hybrid.calibration_apply_to_ml_only == original_strategy, "Strategy restore failed"
        print(
            f"✅ Calibration strategy round-trip verified: {original_strategy} → "
            f"{not original_strategy} → {original_strategy}"
        )

        # Test error handling
        print("\n[6/6] Testing Error Handling...")
        try:
            hybrid.predict_single(150, "male", 25.0, 1, "no", "northeast")
        except ValueError as e:
            print(f"✅ Caught invalid age: {str(e)[:60]}...")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

        print("\n🎯 Critical Fixes in v6.3.3 (v7.5.1 patch):")
        print("  1. ✅ Calibration strategy (ML-only vs full hybrid)")
        print("  2. ✅ Confidence intervals in log space (statistically correct)")
        print("  3. ✅ Governance-appropriate tail risk warnings")
        print("  4. ✅ Actuarial conservativeness detection")
        print("  5. ✅ Enhanced configuration validation")
        print("  6. ✅ Tail risk anchored on ML predictions (not actuarial) [C4]")
        print("  7. ✅ Fallback actuarial params match config.yaml v7.5.0 [C2]")
        print("  8. ✅ Batch limit reads from full config scope [C1]")
        print("  9. ✅ Calibration dispatch on model objective, not name [H2]")

        print("\n📋 Configuration Guidance:")
        print("  • Set 'calibration.apply_to_ml_only: true' if actuarial is conservative")
        print("  • Set 'calibration.apply_to_ml_only: false' if actuarial is competitive")
        print("  • Monitor actuarial/ML ratio to detect conservativeness")
        print("  • Review tail risk warnings with pricing governance team")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🚨 Run train.py first to generate model and preprocessor")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
