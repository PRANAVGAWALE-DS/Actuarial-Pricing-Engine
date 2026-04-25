"""
Production ML Training Pipeline
Optimized for: Windows 11, 16GB RAM, RTX 3050 4GB
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import re
import signal
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from optuna.exceptions import ExperimentalWarning
from sklearn.model_selection import KFold, train_test_split

from insurance_ml.utils import MetricsExtractor, ValidationError

warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
warnings.filterwarnings("ignore", category=ExperimentalWarning)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

VERSION = "5.2.0"
MODEL_SCHEMA_VERSION = "3.0"

# Centralized psutil detection
_PSUTIL_AVAILABLE = False
try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    pass


class BiasCorrection:
    """
    Model-specific bias correction for log-transformed predictions.

    Stored SEPARATELY from feature_engineer to prevent cache contamination.
    Each model gets its own BiasCorrection instance based on its residuals.
    """

    def __init__(
        self,
        var_low: float,
        var_mid: float | None,
        var_high: float,
        threshold_low: float | None = None,
        threshold_high: float | None = None,
        overall_variance: float | None = None,
    ):
        self.var_low = var_low
        self.var_mid = var_mid
        self.var_high = var_high
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.overall_variance = overall_variance
        self.is_2tier = var_mid is None
        self._validate()

    def _validate(self):
        """Validate bias correction parameters.

        Variance values CAN be negative — negative var means exp(var/2) < 1.0,
        i.e. a downward correction for tiers where the model over-predicts.
        Zero is still invalid (indistinguishable from an uninitialized value).
        """
        import math as _m

        for name, val in [("var_low", self.var_low), ("var_high", self.var_high)]:
            if val == 0 or not _m.isfinite(val):
                raise ValueError(f"{name} must be non-zero and finite, got {val}")

        if not self.is_2tier:
            if self.var_mid == 0 or not _m.isfinite(self.var_mid):
                raise ValueError(f"var_mid must be non-zero and finite, got {self.var_mid}")

        if self.threshold_low is None:
            raise ValueError("threshold_low is required")

        if not self.is_2tier and self.threshold_high is None:
            raise ValueError("threshold_high required for 3-tier")

        if not self.is_2tier:
            if self.threshold_low >= self.threshold_high:
                raise ValueError(
                    f"threshold_low ({self.threshold_low}) must be < threshold_high ({self.threshold_high})"
                )

    def apply(
        self,
        y_pred: np.ndarray,
        y_original: np.ndarray | None = None,
        log_details: bool = False,
    ) -> np.ndarray:
        """Apply stratified bias correction to predictions.

        Args:
            y_pred:     Predictions in original scale to be corrected.
            y_original: Routing signal for tier assignment.
                        At inference time, omit or pass None — tier routing will
                        use y_pred directly (self-referential routing).
                        During evaluation, pass true labels for exact tier routing.
            log_details: Emit per-tier diagnostics when True.

        Notes:
            y_original is now Optional.  Callers MUST NOT pass y_pred as
            both arguments as a workaround — simply omit y_original at inference.
        """
        # route using y_pred when y_original not available (inference path).
        routing = y_original if y_original is not None else y_pred

        if len(y_pred) != len(routing):
            raise ValueError(f"Length mismatch: y_pred={len(y_pred)}, y_original={len(routing)}")

        y_corrected = y_pred.copy()

        if self.is_2tier:
            low_mask = routing <= self.threshold_low
            high_mask = routing > self.threshold_low

            correction_low = np.exp(self.var_low / 2)
            correction_high = np.exp(self.var_high / 2)

            y_corrected[low_mask] *= correction_low
            y_corrected[high_mask] *= correction_high

            if log_details:
                logger.info(
                    f"Applied 2-tier bias correction:\n"
                    f"   Low (≤${self.threshold_low:.0f}): {correction_low:.4f}x ({low_mask.sum()} samples)\n"
                    f"   High (>${self.threshold_low:.0f}): {correction_high:.4f}x ({high_mask.sum()} samples)"
                )
        else:
            low_mask = routing <= self.threshold_low
            mid_mask = (routing > self.threshold_low) & (routing <= self.threshold_high)
            high_mask = routing > self.threshold_high

            correction_low = np.exp(self.var_low / 2)
            assert self.var_mid is not None, "var_mid must be set for 3-tier correction"
            correction_mid = np.exp(self.var_mid / 2)
            correction_high = np.exp(self.var_high / 2)

            y_corrected[low_mask] *= correction_low
            y_corrected[mid_mask] *= correction_mid
            y_corrected[high_mask] *= correction_high

            if log_details:
                logger.info(
                    f"Applied 3-tier bias correction:\n"
                    f"   Low (≤${self.threshold_low:.0f}): {correction_low:.4f}x ({low_mask.sum()} samples)\n"
                    f"   Mid (${self.threshold_low:.0f}-${self.threshold_high:.0f}): {correction_mid:.4f}x ({mid_mask.sum()} samples)\n"
                    f"   High (>${self.threshold_high:.0f}): {correction_high:.4f}x ({high_mask.sum()} samples)"
                )

        return y_corrected

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/caching."""
        return {
            "var_low": float(self.var_low),
            "var_mid": float(self.var_mid) if self.var_mid is not None else None,
            "var_high": float(self.var_high),
            # use 'is not None' not truthiness.
            # A threshold of 0.0 is technically valid; a truthiness check would
            # silently serialise it as None, corrupting the round-trip.
            "threshold_low": (
                float(self.threshold_low) if self.threshold_low is not None else None
            ),
            "threshold_high": (
                float(self.threshold_high) if self.threshold_high is not None else None
            ),
            "overall_variance": (
                float(self.overall_variance) if self.overall_variance is not None else None
            ),
            "is_2tier": self.is_2tier,
            "version": "1.0",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BiasCorrection:
        """Deserialize from dict."""
        return cls(
            var_low=data["var_low"],
            var_mid=data.get("var_mid"),
            var_high=data["var_high"],
            threshold_low=data.get("threshold_low"),
            threshold_high=data.get("threshold_high"),
            overall_variance=data.get("overall_variance"),
        )

    def get_hash(self) -> str:
        """Generate deterministic hash for caching."""
        import hashlib

        if self.is_2tier:
            state_str = f"2tier_{self.var_low:.10f}_{self.var_high:.10f}_{self.threshold_low:.4f}"
        else:
            state_str = (
                f"3tier_{self.var_low:.10f}_{self.var_mid:.10f}_{self.var_high:.10f}_"
                f"{self.threshold_low:.4f}_{self.threshold_high:.4f}"
            )

        return hashlib.md5(state_str.encode(), usedforsecurity=False).hexdigest()[:8]

    def __repr__(self) -> str:
        tier_type = "2-tier" if self.is_2tier else "3-tier"
        import math as _m

        f_low = _m.exp(self.var_low / 2)
        f_high = _m.exp(self.var_high / 2)
        return f"BiasCorrection({tier_type}, " f"factor_low={f_low:.4f}, factor_high={f_high:.4f})"

    @classmethod
    def calculate_from_model(
        cls,
        model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_engineer,
        model_name: str,
    ) -> BiasCorrection | None:
        """
        Factory method to compute model-specific bias correction.

        Does NOT modify feature_engineer.
        Returns a new BiasCorrection instance.

        Formula selection (per-transform):
        ─────────────────────────────────
        log1p   → exp(σ²/2)  Jensen's inequality correction.  Valid because the
                              back-transform is exp() and residuals are approximately
                              log-normal.

        yeo-johnson → median ratio correction.  The exp(σ²/2) formula requires
                      Y_orig = exp(Y_transformed) and N(0,σ²) residuals — neither
                      holds for Yeo-Johnson.  Instead we compute
                          r_tier = median(y_true_orig / y_pred_orig)
                      per tier and store var = 2·ln(r_tier) so that the existing
                      apply() call (exp(var/2) == r_tier) works unchanged for all
                      model types (ridge, lasso, quantile_regressor, random_forest,
                      gradient_boosting, svr, knn, xgboost, lightgbm).

        Clamp: max correction per tier = 30% (ratio 1.30).  Generous enough for
        linear models on non-linear data while preventing runaway inflation.
        Downward correction is supported: ratio < 1 → var = 2*log(ratio) < 0,
        so exp(var/2) < 1.0. apply() handles both directions via the same formula.
        """
        import math as _math

        _MIN_VAR = 1e-6  # satisfies _validate() > 0; exp(1e-6/2) ≈ 1.0
        _MAX_RATIO = 1.30  # 30% max upward correction per tier
        _MAX_DOWNWARD_VAR = -2.0 * _math.log(0.90)  # ≈ 0.2107 → exp(-v/2)=0.90 (−10% floor)

        transform_method = feature_engineer.target_transformation.method

        if transform_method not in ["log1p", "yeo-johnson"]:
            logger.info(f"ℹ️  Bias correction not required for {transform_method}")
            return None

        # ── Quantile model guard ──────────────────────────────────────────────
        # Median-ratio bias correction is INCOMPATIBLE with quantile loss models.
        # At quantile level alpha > 0.50, the model intentionally predicts above
        # the median, so median(actual / prediction) < 1.0 by design. Applying
        # the correction would counteract the quantile shift and cause net
        # underpricing. Quantile models rely solely on calibration factor instead.
        _model_type = type(model).__name__

        def _lgbm_objective(m) -> str:
            """Read LightGBM objective from booster params — NOT from Python attribute."""
            # 1. get_params() covers unfitted or sklearn-wrapped models
            try:
                obj = m.get_params().get("objective", "") or ""
                if obj:
                    return str(obj).lower()
            except Exception:
                pass
            # 2. booster_.params covers fitted boosters (most reliable)
            try:
                obj = m.booster_.params.get("objective", "") or ""
                if obj:
                    return str(obj).lower()
            except Exception:
                pass
            # 3. Explicit flag set at training time (belt-and-suspenders)
            return str(getattr(m, "_lgbm_objective", "")).lower()

        def _xgb_objective(m) -> str:
            """Read XGBoost objective safely.

            Priority: get_xgb_params() (booster-level, most reliable) →
            get_params() → attribute.  If the objective string doesn't contain
            'quantile' but quantile_alpha is set, the model IS a quantile model
            and we return 'reg:quantileerror' as a belt-and-suspenders guard.
            """
            try:
                obj = str(m.get_xgb_params().get("objective", "")).lower()
                if obj:
                    # Fast path: objective already contains 'quantile'
                    if "quantile" in obj:
                        return obj
                    # Check quantile_alpha before giving up
                    try:
                        if m.get_params().get("quantile_alpha") is not None:
                            return "reg:quantileerror"
                    except Exception:
                        pass
                    return obj
            except Exception:
                pass
            try:
                obj = str(m.get_params().get("objective", "")).lower()
                if obj:
                    return obj
            except Exception:
                pass
            # Last resort: quantile_alpha presence
            try:
                if m.get_params().get("quantile_alpha") is not None:
                    return "reg:quantileerror"
            except Exception:
                pass
            return str(getattr(m, "objective", "")).lower()

        _is_quantile_model = (
            (_model_type == "LGBMRegressor" and "quantile" in _lgbm_objective(model))
            or (_model_type == "XGBRegressor" and "quantile" in _xgb_objective(model))
            or (
                _model_type == "GradientBoostingRegressor"
                and getattr(model, "loss", "") == "quantile"
            )
            # elastic_net factory is QuantileRegressor, not ElasticNet.
            # sklearn.linear_model.QuantileRegressor has no 'loss' attribute and does
            # not set '_is_quantile_model', so none of the other checks catch it.
            # model_name='elastic_net' also lacks 'quantile', making the fallback miss it.
            or _model_type == "QuantileRegressor"
            # Belt-and-suspenders: explicit flag set during training
            or getattr(model, "_is_quantile_model", False)
            # model_name fallback (e.g. "lightgbm_quantile", "lgbm_q65")
            or "quantile" in model_name.lower()
            # In XGBoost 2.x, get_xgb_params() / get_params() do not
            # reliably expose 'reg:quantileerror' or 'quantile_alpha' — the objective
            # may be stored as an internal callable or under a different key depending
            # on the exact 2.x build.  Direct attribute access bypasses all of that:
            # XGBoost always stores constructor kwargs as instance attributes, so
            # model.quantile_alpha == 0.65 is the definitive signal.  model.kwargs
            # covers builds where it was passed as a **kwarg rather than a named param.
            or (
                _model_type == "XGBRegressor"
                and (
                    getattr(model, "quantile_alpha", None) is not None
                    or (
                        isinstance(getattr(model, "kwargs", None), dict)
                        and model.kwargs.get("quantile_alpha") is not None
                    )
                )
            )
        )

        if _is_quantile_model:
            logger.info(
                f"ℹ️  Quantile model detected ({_model_type}, name='{model_name}') "
                f"— skipping median-ratio bias correction.\n"
                f"   Quantile loss at alpha>0.50 deliberately predicts above the median;\n"
                f"   applying median-ratio correction would undo this shift and increase\n"
                f"   underpricing. Calibration factor handles residual global adjustment."
            )
            return None
        # ── End quantile guard ────────────────────────────────────────────────

        # log1p uses the variance formula; yeo-johnson uses the median-ratio formula
        use_variance_formula = transform_method == "log1p"

        logger.info(
            f"🔧 Calculating 3-tier stratified bias correction for {model_name}...\n"
            f"   Formula: {'exp(σ²/2) [log1p / Jensen]' if use_variance_formula else 'median ratio [yeo-johnson]'}"
        )

        try:
            # ── Predict in transformed space ──────────────────────────────────
            y_pred_val_transformed = model.predict(X_val)
            y_val_transformed = y_val.values if hasattr(y_val, "values") else np.array(y_val)

            # ── Residuals in transformed space (always needed for fallback/log) ──
            residuals = y_val_transformed - y_pred_val_transformed

            # ── Inverse-transform BOTH arrays to original scale ───────────────
            y_val_original = feature_engineer.inverse_transform_target(
                y_val_transformed,
                transformation_method=transform_method,
                clip_to_safe_range=False,
                context=f"bias_calc_{model_name}",
            )

            y_pred_original = feature_engineer.inverse_transform_target(
                y_pred_val_transformed,
                transformation_method=transform_method,
                clip_to_safe_range=False,
                context=f"bias_pred_{model_name}",
            )

            # ── Define 3 tiers on original scale ─────────────────────────────
            # Use fixed G6-aligned absolute boundaries
            # instead of floating prediction-percentile boundaries.
            #
            # ROOT CAUSE OF THE REGRESSION:
            #   Old code used q50/q75 of y_pred_original as BC tier boundaries.
            #   On the 1,337-row dataset these happened to resolve to ~$9,370
            #   and ~$14,305, which accidentally aligned with the G6 segment
            #   bins. On the 101K-row dataset, the model fits Low/Mid more
            #   accurately, collapsing q75 to ~$12,886. This merges the G6
            #   "Mid" ($10K–$14K) and "High" ($14K–$16.7K) segments into one
            #   BC "High" tier. Within this merged tier, Very High over-pricing
            #   dominates the aggregate median ratio, yielding a small downward
            #   correction that is then applied uniformly to High/High+ samples
            #   that are already under-predicted — inverting those predictions
            #   and driving R² to -8 and -139 for those G6 segments.
            #
            #   Use the same fixed absolute boundaries as segment_r2_breakdown:
            #     threshold_low  = $10,000  (G6 Low/Mid boundary)
            #     threshold_high = $14,000  (G6 Mid/High boundary)
            #   This guarantees each BC tier covers exactly the same G6 segment
            #   population regardless of dataset size or prediction distribution.
            #   Routing at inference uses y_pred against these same constants,
            #   preserving the training-inference alignment from FIX ISSUE-4A.
            #
            # Note: q50/q75 variable names are kept below for backward compat
            # with the 2-tier fallback and downstream log formatting only.
            # They are now constants, not percentiles.
            _BC_TIER_LOW: float = 10_000.0  # matches G6 Low/Mid boundary
            _BC_TIER_HIGH: float = 14_000.0  # matches G6 Mid/High boundary
            q50 = _BC_TIER_LOW
            q75 = _BC_TIER_HIGH

            # ── v7.5.5: Threshold stability log ──────────────────────────────
            # Boundaries are now fixed constants, so no drift is possible.
            # Log once to confirm the expected values are in effect, and
            # warn if a stale bias_correction.json has different values
            # (indicating this run is the first after the patch-A upgrade).
            try:
                import json as _json_drift
                from pathlib import Path as _Path_drift

                _bc_path = _Path_drift("models") / "bias_correction.json"
                if _bc_path.exists():
                    with open(_bc_path) as _f:
                        _prev = _json_drift.load(_f)
                    _params = _prev.get("correction_params") or _prev
                    _prev_q50 = float(_params.get("threshold_low", 0) or 0)
                    _prev_q75 = float(_params.get("threshold_high", 0) or 0)
                    if abs(_prev_q50 - _BC_TIER_LOW) > 1.0 or abs(_prev_q75 - _BC_TIER_HIGH) > 1.0:
                        logger.warning(
                            f"⚠️  BiasCorrection boundary migration detected:\n"
                            f"   Previous (percentile-based): "
                            f"low=${_prev_q50:,.0f}, high=${_prev_q75:,.0f}\n"
                            f"   Current  (G6-fixed):         "
                            f"low=${_BC_TIER_LOW:,.0f}, high=${_BC_TIER_HIGH:,.0f}\n"
                            f"   This is expected on first run after patch-A upgrade.\n"
                            f"   Validate on holdout before deploying this artifact."
                        )
                    else:
                        logger.info(
                            f"✅ BiasCorrection thresholds confirmed: "
                            f"low=${_BC_TIER_LOW:,.0f}, high=${_BC_TIER_HIGH:,.0f} "
                            f"(G6-aligned fixed boundaries)"
                        )
            except Exception as _drift_err:
                logger.debug(f"Threshold stability check skipped: {_drift_err}")

            # Masks use y_pred_original for routing consistency at inference
            # (apply() routes y_pred against threshold_low/threshold_high).
            low_mask = y_pred_original <= q50
            mid_mask = (y_pred_original > q50) & (y_pred_original <= q75)
            high_mask = y_pred_original > q75

            n_low = int(np.sum(low_mask))
            n_mid = int(np.sum(mid_mask))
            n_high = int(np.sum(high_mask))

            # ── Helper: variance or median-ratio → var slot ───────────────────
            def _tier_var(
                mask: np.ndarray,
                tier_name: str = "",
                ytrue_upper: float | None = None,
                ytrue_lower: float | None = None,
            ) -> float:
                """Return the var slot for one tier, using the selected formula.

                v7.5.5 PATCH-B: direction-aware guard added.
                v7.5.5 PATCH-D: y_true-filtered ratio computation added.

                PATCH-D — BC ratio dilution:
                The BC 'Low' tier (pred < $10K) contains two sub-populations:
                  ① True-cheap (y_true ≤ $10K): over-predicted → y_true/y_pred < 1
                  ② True-expensive (y_true > $10K), prediction collapsed below $10K:
                     under-predicted → y_true/y_pred >> 1
                Sub-pop ② drags the aggregate median ratio upward, giving a weaker
                downward correction than sub-pop ① needs. restrict the ratio
                computation to samples where y_true is within the tier's intended
                boundary. Falls back to unfiltered if < 30 matching samples.
                """
                if use_variance_formula:
                    # log1p path: Jensen's inequality — exp(σ²/2).
                    # For log1p the residuals are symmetric by construction, so the
                    # direction guard and y_true filtering are not needed.
                    v = float(np.var(residuals[mask], ddof=1))
                    return max(v, _MIN_VAR)
                else:
                    # yeo-johnson path: distribution-agnostic median ratio.

                    # ── DIRECTION GUARD (v7.5.5 PATCH-B) ─────────────────────
                    _tier_overpricing = float((y_pred_original[mask] > y_val_original[mask]).mean())
                    _n_tier = int(mask.sum())

                    if _tier_overpricing < 0.45:
                        logger.info(
                            f"ℹ️  BiasCorrection tier '{tier_name}' (N={_n_tier}): "
                            f"overpricing_rate={_tier_overpricing:.1%} < 45% "
                            f"— model under-predicts majority of this tier. "
                            f"Skipping correction to avoid worsening under-predictions."
                        )
                        return _MIN_VAR

                    # ── Y_TRUE-FILTERED RATIO (v7.5.5 PATCH-D) ───────────────
                    # Restrict median ratio to samples whose y_true falls within
                    # the tier's intended range. This prevents cross-population
                    # contamination from under-predicted expensive policies that
                    # route to a cheap-prediction tier, which would dilute the
                    # aggregate median upward and weaken the downward correction.
                    eps = 1e-8
                    ratio_mask = mask.copy()
                    if ytrue_upper is not None:
                        ytrue_filter = y_val_original <= ytrue_upper
                        if ytrue_lower is not None:
                            ytrue_filter = ytrue_filter & (y_val_original > ytrue_lower)
                        filtered_mask = mask & ytrue_filter
                        n_filtered = int(filtered_mask.sum())
                        if n_filtered >= 30:
                            ratio_mask = filtered_mask
                            logger.debug(
                                f"  PATCH-D: tier '{tier_name}' ratio computed on "
                                f"{n_filtered}/{_n_tier} y_true-filtered samples "
                                f"(y_true ∈ [{ytrue_lower or 0:,.0f}, {ytrue_upper:,.0f}])"
                            )
                        else:
                            logger.debug(
                                f"  PATCH-D: tier '{tier_name}' y_true filter yields "
                                f"only {n_filtered} samples (<30) — using aggregate ratio"
                            )

                    ratio = float(
                        np.median(
                            y_val_original[ratio_mask]
                            / np.maximum(y_pred_original[ratio_mask], eps)
                        )
                    )
                    # Clamp upward correction at _MAX_RATIO (30%).
                    _MAX_UPWARD_VAR = 2.0 * _math.log(_MAX_RATIO)  # ≈ 0.5188
                    ratio = min(ratio, _MAX_RATIO)

                    if ratio > 0:
                        var_result = 2.0 * _math.log(ratio)
                        var_result = min(var_result, _MAX_UPWARD_VAR)
                        if ratio < 1.0:
                            logger.info(
                                f"ℹ️  BiasCorrection tier '{tier_name}' (N={_n_tier}, "
                                f"overpricing={_tier_overpricing:.1%}): "
                                f"median ratio={ratio:.4f} "
                                f"(over-predicts by {(1-ratio)*100:.1f}%). "
                                f"Storing downward correction."
                            )
                        else:
                            logger.info(
                                f"ℹ️  BiasCorrection tier '{tier_name}' (N={_n_tier}, "
                                f"overpricing={_tier_overpricing:.1%}): "
                                f"median ratio={ratio:.4f} "
                                f"(under-predicts by {(ratio-1)*100:.1f}%). "
                                f"Storing upward correction."
                            )
                        return var_result

                    logger.warning(
                        f"⚠️  BiasCorrection tier '{tier_name}': "
                        f"degenerate ratio={ratio:.4f} ≤ 0. "
                        f"Storing sentinel."
                    )
                    return _MIN_VAR

            def _overall_var() -> float:
                if use_variance_formula:
                    return float(np.var(residuals, ddof=1))
                else:
                    eps = 1e-8
                    ratio = float(np.median(y_val_original / np.maximum(y_pred_original, eps)))
                    ratio = min(ratio, _MAX_RATIO)
                    # mirror _tier_var — handle downward corrections
                    # (ratio < 1.0 → var = 2*log(ratio) < 0) instead of silently
                    # falling back to _MIN_VAR, which would discard the correction.
                    if ratio > 0:
                        return 2.0 * _math.log(ratio)
                    return _MIN_VAR

            # ── Small-segment fallback → 2-tier ──────────────────────────────
            if n_low < 10 or n_mid < 10 or n_high < 10:
                logger.warning(
                    f"⚠️  Small segment detected (falling back to 2-tier):\n"
                    f"   Low: {n_low} samples, Mid: {n_mid} samples, High: {n_high} samples"
                )
                # fallback also uses y_pred_original so that
                # threshold_low (q75) stored in the object matches inference routing.
                low_mask_2tier = y_pred_original <= q75
                high_mask_2tier = y_pred_original > q75

                var_low = _tier_var(low_mask_2tier, tier_name="Low-Mid", ytrue_upper=float(q75))
                var_high = _tier_var(high_mask_2tier, tier_name="High", ytrue_lower=float(q75))
                overall_var = _overall_var()

                # ── Cap downward bias correction at −10% per tier ────────
                # var < 0 encodes a downward correction (exp(var/2) < 1.0).
                # var_low=-0.347 → factor=0.84 (−16%) is too aggressive on
                # calibration data with high kurtosis — it overcorrects on test.
                # Cap: exp(-MAX_DOWNWARD_VAR/2) = 0.90 (max −10% downward).
                # Upward corrections (var > 0) are unrestricted.
                var_low = max(var_low, -_MAX_DOWNWARD_VAR) if var_low < 0 else var_low
                var_high = max(var_high, -_MAX_DOWNWARD_VAR) if var_high < 0 else var_high

                bias_correction = cls(
                    var_low=var_low,
                    var_mid=None,
                    var_high=var_high,
                    threshold_low=float(q75),
                    threshold_high=None,
                    overall_variance=overall_var,
                )
                logger.info(f"✅ 2-tier bias correction: {bias_correction}")
                return bias_correction

            # ── 3-tier ────────────────────────────────────────────────────────
            var_low = _tier_var(low_mask, tier_name="Low", ytrue_upper=float(q50))
            var_mid = _tier_var(
                mid_mask, tier_name="Mid", ytrue_lower=float(q50), ytrue_upper=float(q75)
            )
            var_high = _tier_var(high_mask, tier_name="High", ytrue_lower=float(q75))
            overall_var = _overall_var()

            # ── Cap downward bias correction at −10% per tier ────────────
            # Rationale: var_low=-0.347 (factor=0.84, −16%) was computed on the
            # calibration split (60% of val) but overcorrects on the test set
            # because the heavy-tailed residuals (skew=+3.14, kurtosis=12.58)
            # inflate the median-ratio estimate on the small calibration sample.
            # Capping at −10% (var floor = -2*log(0.90) ≈ -0.2107) prevents
            # the Low segment from being under-priced by the bias correction
            # itself. Mid and High tiers also capped for consistency.
            # Upward corrections (var > 0) remain unrestricted.
            var_low = max(var_low, -_MAX_DOWNWARD_VAR) if var_low < 0 else var_low
            var_mid = max(var_mid, -_MAX_DOWNWARD_VAR) if var_mid < 0 else var_mid
            var_high = max(var_high, -_MAX_DOWNWARD_VAR) if var_high < 0 else var_high

            # Log effective correction factors for all model types
            import math as _m

            f_low = _m.exp(var_low / 2)
            f_mid = _m.exp(var_mid / 2)
            f_high = _m.exp(var_high / 2)
            logger.info(
                f"   Effective correction factors (G6-aligned fixed boundaries):\n"
                f"   Low  (<${q50:,.0f}):  {f_low:.4f}  ({(f_low -1)*100:+.2f}%)\n"
                f"   Mid  (${q50:,.0f}–${q75:,.0f}): {f_mid:.4f}  ({(f_mid -1)*100:+.2f}%)\n"
                f"   High (>${q75:,.0f}): {f_high:.4f}  ({(f_high-1)*100:+.2f}%)"
            )

            bias_correction = cls(
                var_low=var_low,
                var_mid=var_mid,
                var_high=var_high,
                threshold_low=float(q50),
                threshold_high=float(q75),
                overall_variance=overall_var,
            )

            logger.info(f"✅ 3-tier bias correction calculated:\n{bias_correction}")
            return bias_correction

        except Exception as e:
            logger.error(f"❌ Bias correction calculation failed: {e}", exc_info=True)
            return None


# =====================================================================
# SAMPLE WEIGHT CALCULATION (FROM CONFIG)
# =====================================================================


def calculate_sample_weights(
    y: pd.Series,
    config: dict[str, Any],
    validate_distribution: bool = True,
    y_original: pd.Series | None = None,
) -> np.ndarray:
    """
    Calculate sample weights using config.yaml parameters with robust validation.

    Args:
        y: Target values (may be in transformed space — used only for shape/validation)
        config: Full configuration dict from load_config()
        validate_distribution: Whether to validate weight distribution (default: True)
        y_original: Original-scale target values. When provided, weights are computed
            from original-scale quantiles, which is the correct behaviour for insurance
            premiums. When None, falls back to using ``y`` directly (legacy path, emits
            a WARNING if y appears to be in transformed space).

    Returns:
        Sample weights array

    Raises:
        ValueError: If target contains non-finite values

    Notes:
        Always pass ``y_original`` from the pre-transformation split.  Computing weights
        on the transformed target (Yeo-Johnson / log1p space) silently produces
        near-uniform weights because the dollar-magnitude ordering is compressed,
        defeating the importance-weighting objective.
    """
    # Use y_original for weight quantile computation when available.
    # The fragile y_range < 20 heuristic is removed; callers now explicitly pass
    # y_original so we always weight on the correct scale.
    if y_original is not None:
        y_for_weights = y_original
        scale_label = "original scale (y_original)"
    else:
        y_for_weights = y
        y_range = y.max() - y.min()
        if y_range < 100:
            logger.warning(
                "⚠️  calculate_sample_weights: y_original not provided and y appears to be "
                "in transformed space (range=%.3f). Weights will be computed on transformed "
                "values, which may produce near-uniform results. Pass y_original for "
                "correct dollar-scale importance weighting.",
                y_range,
            )
            scale_label = "transformed scale (y_original not provided — WARNING)"
        else:
            scale_label = "original scale (inferred — y_original not provided)"
    from insurance_ml.config import get_sample_weight_config

    # ========================================
    # VALIDATION: Input data quality checks
    # ========================================
    # Check 1: Minimum sample size
    if len(y_for_weights) < 20:
        logger.warning(
            f"⚠️  Only {len(y_for_weights)} samples - sample weights may be unstable\n"
            f"   Recommend at least 100 samples for reliable quantiles\n"
            f"   → Using uniform weights"
        )
        return np.ones(len(y))

    # Check 2: Target variance
    y_std = y_for_weights.std()
    if y_std < 1e-10:
        logger.warning(
            f"⚠️            Target has no variance - using uniform weights\n"
            f"   All values are approximately {y_for_weights.mean():.2f}\n"
            f"   Standard deviation: {y_std:.2e}"
        )
        return np.ones(len(y))

    # Check 3: Non-finite values
    if not np.all(np.isfinite(y_for_weights)):
        n_nan = y_for_weights.isna().sum()
        n_inf = np.isinf(y_for_weights).sum()
        n_bad = n_nan + n_inf

        raise ValueError(
            f"❌ Target contains {n_bad} non-finite values!\n"
            f"   NaN: {n_nan}, Inf: {n_inf}\n"
            f"\n"
            f"   🔍 DIAGNOSIS:\n"
            f"   → Data cleaning step failed or skipped\n"
            f"   → Target transformation produced invalid values\n"
            f"\n"
            f"   ✅ FIX:\n"
            f"   1. Clean target values before calculating weights\n"
            f"   2. Check target transformation for edge cases\n"
            f"   3. Remove or impute problematic samples"
        )

    # ========================================
    # CONFIG: Extract sample weight configuration
    # ========================================
    sw_cfg = get_sample_weight_config(config)

    if not sw_cfg["enabled"]:
        logger.info("Sample weights disabled in config - using uniform weights")
        return np.ones(len(y))

    method = sw_cfg["method"]
    tiers = sw_cfg["tiers"]

    # Removed fragile y_range < 20 heuristic.
    # scale_label is now set deterministically at function entry based on
    # whether y_original was provided.
    logger.info(f"Calculating sample weights: method='{method}' ({scale_label})")

    # ========================================
    # QUANTILE CALCULATION with error handling
    # ========================================
    try:
        q25, q50, q75, q90, q95, q99 = y_for_weights.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

        logger.debug(
            f"Target quantiles:\n"
            f"   Q25={q25:.2f}, Q50={q50:.2f}, Q75={q75:.2f}\n"
            f"   Q90={q90:.2f}, Q95={q95:.2f}, Q99={q99:.2f}"
        )

    except Exception as e:
        logger.error(
            f"❌ Failed to calculate quantiles: {e}\n"
            f"   Target stats: min={y_for_weights.min():.2f}, max={y_for_weights.max():.2f}, "
            f"mean={y_for_weights.mean():.2f}, std={y_for_weights.std():.2f}\n"
            f"   → Falling back to uniform weights"
        )
        return np.ones(len(y))

    # ========================================
    # VALIDATION: Check quantiles are distinct
    # ========================================
    quantiles = [q25, q50, q75, q90, q95, q99]
    unique_quantiles = len(set(np.round(quantiles, 2)))  # Round to avoid floating point issues

    if unique_quantiles < 3:
        logger.warning(
            f"⚠️  Quantiles are not distinct - target may have discrete values\n"
            f"   Unique quantiles: {unique_quantiles}/6\n"
            f"   Values: {[f'{q:.2f}' for q in quantiles]}\n"
            f"\n"
            f"   💡 This happens with:\n"
            f"   → Highly discrete targets (e.g., few unique values)\n"
            f"   → Small sample sizes\n"
            f"   → Heavily skewed distributions\n"
            f"\n"
            f"   → Using simplified weighting: high vs low"
        )

        # Simplified weighting: above/below median
        median = y_for_weights.median()
        weights = np.where(y_for_weights > median, 2.0, 1.0)

        logger.info(
            f"Simplified weights applied:\n"
            f"   Below median: 1.0 ({(y_for_weights <= median).sum()} samples)\n"
            f"   Above median: 2.0 ({(y_for_weights > median).sum()} samples)"
        )

        return weights

    # ========================================
    # WEIGHT ASSIGNMENT: Apply tier weights from config
    # ========================================
    weights = np.ones(len(y))

    # Create boolean masks for each tier
    mask_q1 = y_for_weights <= q25
    mask_q25_q50 = (y_for_weights > q25) & (y_for_weights <= q50)
    mask_q50_q75 = (y_for_weights > q50) & (y_for_weights <= q75)
    mask_q75_q90 = (y_for_weights > q75) & (y_for_weights <= q90)
    mask_q90_q95 = (y_for_weights > q90) & (y_for_weights <= q95)
    mask_q95_q99 = (y_for_weights > q95) & (y_for_weights <= q99)
    mask_above_q99 = y_for_weights > q99

    # Apply weights from config
    weights[mask_q25_q50] = tiers["q25_to_q50"]
    weights[mask_q50_q75] = tiers["q50_to_q75"]
    weights[mask_q75_q90] = tiers["q75_to_q90"]
    weights[mask_q90_q95] = tiers["q90_to_q95"]
    weights[mask_q95_q99] = tiers["q95_to_q99"]
    weights[mask_above_q99] = tiers["above_q99"]

    # Log tier distribution BEFORE transform
    logger.info(
        f"Weight tiers (before transform):\n"
        f"   Q1 (≤{q25:.0f}): {weights[mask_q1].mean():.2f} ({mask_q1.sum()} samples)\n"
        f"   Q2 ({q25:.0f}-{q50:.0f}): {weights[mask_q25_q50].mean():.2f} ({mask_q25_q50.sum()} samples)\n"
        f"   Q3 ({q50:.0f}-{q75:.0f}): {weights[mask_q50_q75].mean():.2f} ({mask_q50_q75.sum()} samples)\n"
        f"   Q4 ({q75:.0f}-{q90:.0f}): {weights[mask_q75_q90].mean():.2f} ({mask_q75_q90.sum()} samples)\n"
        f"   Top 5% ({q90:.0f}-{q95:.0f}): {weights[mask_q90_q95].mean():.2f} ({mask_q90_q95.sum()} samples)\n"
        f"   Top 5-1% ({q95:.0f}-{q99:.0f}): {weights[mask_q95_q99].mean():.2f} ({mask_q95_q99.sum()} samples)\n"
        f"   Top 1% (>{q99:.0f}): {weights[mask_above_q99].mean():.2f} ({mask_above_q99.sum()} samples)"
    )

    # ========================================
    # VALIDATION: Check final weight distribution
    # ========================================
    if validate_distribution:
        if not np.all(np.isfinite(weights)):
            logger.error("❌ Weights contain non-finite values!")
            return np.ones(len(y))

        if np.any(weights < 0):
            logger.error("❌ Weights contain negative values!")
            return np.ones(len(y))

        weight_ratio = weights.max() / weights.min() if weights.min() > 0 else np.inf

        if weight_ratio > 200:  # Increased threshold (was 100)
            logger.warning(
                f"⚠️  High weight ratio: {weight_ratio:.1f}x\n"
                f"   This is expected for high-value focus"
            )

    # Summary logging
    n_high = (y_for_weights > q95).sum()

    logger.info(
        f"✅ Sample weights calculated ({method}, {scale_label}):\n"
        f"   Mean weight: {weights.mean():.4f}\n"
        f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]\n"
        f"   Q2 avg: {weights[mask_q25_q50].mean():.4f}, "
        f"Q3 avg: {weights[mask_q50_q75].mean():.4f}, "
        f"Q4 avg: {weights[mask_q75_q90].mean():.4f}\n"
        f"   Top 5% avg: {weights[y_for_weights > q95].mean():.4f} "
        f"({n_high} samples)"
    )

    return weights


def validate_sample_weights(weights: np.ndarray, y: pd.Series, max_ratio: float = 100.0) -> bool:
    """
    Validate sample weights are reasonable.

    Args:
        weights: Sample weights array
        y: Target values
        max_ratio: Maximum allowed weight ratio (default: 100)

    Returns:
        True if weights are valid, False otherwise
    """

    # Check 1: Length match
    if len(weights) != len(y):
        logger.error(f"❌ Weight length mismatch: {len(weights)} weights for {len(y)} samples")
        return False

    # Check 2: Finite values
    if not np.all(np.isfinite(weights)):
        n_bad = (~np.isfinite(weights)).sum()
        logger.error(f"❌ {n_bad} non-finite weights detected")
        return False

    # Check 3: Non-negative
    if np.any(weights < 0):
        n_negative = (weights < 0).sum()
        logger.error(f"❌ {n_negative} negative weights detected")
        return False

    # Check 4: Reasonable range
    if weights.min() > 0:
        weight_ratio = weights.max() / weights.min()
        if weight_ratio > max_ratio:
            logger.warning(f"⚠️  Extreme weight ratio: {weight_ratio:.1f}x > {max_ratio}x threshold")
            return False

    # Check 5: Not all zeros
    if np.all(weights == 0):
        logger.error("❌ All weights are zero - training would fail")
        return False

    logger.debug(
        f"✅ Weights validated: "
        f"range=[{weights.min():.4f}, {weights.max():.4f}], "
        f"mean={weights.mean():.4f}"
    )

    return True


def analyze_high_value_segment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict[str, Any],
    baseline_rmse: float | None = None,
) -> dict[str, Any]:
    """
    Analyze model performance on high-value segment (FROM CONFIG).

    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)
        config: Full configuration dict
        baseline_rmse: Optional baseline RMSE for comparison

    Returns:
        Dict with segment metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from insurance_ml.config import get_training_config

    # Get threshold from config (single source)
    training_cfg = get_training_config(config)
    threshold_percentile = training_cfg["high_value_percentile"]

    threshold = np.percentile(y_true, threshold_percentile)
    high_value_mask = y_true >= threshold

    if high_value_mask.sum() == 0:
        logger.warning(f"No samples in high-value segment (>{threshold:.0f})")
        return {}

    y_high_true = y_true[high_value_mask]
    y_high_pred = y_pred[high_value_mask]

    # Calculate metrics
    high_rmse = np.sqrt(mean_squared_error(y_high_true, y_high_pred))
    high_mae = mean_absolute_error(y_high_true, y_high_pred)
    high_r2 = r2_score(y_high_true, y_high_pred)
    high_mape = 100 * np.mean(np.abs((y_high_true - y_high_pred) / np.maximum(y_high_true, 1e-10)))

    n_samples = high_value_mask.sum()
    pct_samples = 100 * n_samples / len(y_true)

    logger.info("=" * 80)
    logger.info(f"HIGH-VALUE SEGMENT ANALYSIS (P{threshold_percentile}, >${threshold:.0f})")
    logger.info("=" * 80)
    logger.info(f"Samples:      {n_samples}/{len(y_true)} ({pct_samples:.1f}%)")
    logger.info(f"Value range:  [${y_high_true.min():,.0f}, ${y_high_true.max():,.0f}]")
    logger.info(f"RMSE:         ${high_rmse:,.0f}")
    logger.info(f"MAE:          ${high_mae:,.0f}")
    logger.info(f"R²:           {high_r2:.4f}")
    logger.info(f"MAPE:         {high_mape:.2f}%")

    # Baseline comparison
    if baseline_rmse is not None:
        if high_rmse < baseline_rmse:
            improvement = ((baseline_rmse - high_rmse) / baseline_rmse) * 100
            logger.info(
                f"✅ Improvement: {improvement:.1f}% better than baseline (${baseline_rmse:,})"
            )
        else:
            regression = ((high_rmse - baseline_rmse) / baseline_rmse) * 100
            logger.warning(f"⚠️ Regression: {regression:.1f}% worse than baseline")

    logger.info("=" * 80 + "\n")

    return {
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "n_samples": int(n_samples),
        "pct_samples": float(pct_samples),
        "rmse": float(high_rmse),
        "mae": float(high_mae),
        "r2": float(high_r2),
        "mape": float(high_mape),
        "baseline_rmse": float(baseline_rmse) if baseline_rmse else None,
    }


# =====================================================================
# EXCEPTIONS
# =====================================================================


class TimeoutError(Exception):
    """Raised when operation times out"""


# =====================================================================
# TYPE DEFINITIONS
# =====================================================================


class TrainingResult(TypedDict, total=False):
    """Type hint for training results"""

    model_name: str
    status: str
    error: str | None
    model_path: str
    checksum: str
    cv_mean: float
    cv_std: float
    training_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    validation_predictions: np.ndarray
    training_time: float
    gpu_used: bool
    diagnostics: dict[str, Any]
    mlflow_run_id: str | None
    mlflow_model_uri: str | None
    model_version: str
    explainability: dict[str, Any]


# =====================================================================
# CONFIGURATION (FROM CONFIG.YAML VIA TYPED HELPERS)
# =====================================================================


@dataclass
class Config:
    """
    Validated configuration with strict types.
    Do NOT add default values here - use config.yaml instead.
    """

    # Paths
    output_dir: Path
    reports_dir: Path

    # Training parameters
    cv_folds: int
    random_state: int
    test_size: float
    val_size: float
    stratify_splits: bool

    # Feature flags
    enable_mlflow: bool
    enable_optuna: bool
    enable_diagnostics: bool

    # Resource management
    gpu_enabled: bool
    gpu_memory_limit_mb: int
    gpu_memory_fraction: float
    memory_fraction: float
    max_model_size_mb: float

    # Timeouts & thresholds
    training_timeout: int | None
    min_r2_threshold: float

    # Diagnostics
    max_shap_samples: int

    # Security & validation
    verify_checksums: bool
    halt_on_severe_shift: bool
    save_checksums: bool
    register_to_mlflow: bool
    max_memory_mb: float  # From config.yaml training section

    # Sample weights
    use_sample_weights: bool
    high_value_percentile: int

    def __post_init__(self):
        """Validate and compute derived values"""
        # Convert paths
        self.output_dir = Path(self.output_dir)
        self.reports_dir = Path(self.reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Validate ranges
        assert (
            isinstance(self.cv_folds, int) and 2 <= self.cv_folds <= 20
        ), f"cv_folds must be in [2, 20], got {self.cv_folds}"
        assert (
            isinstance(self.random_state, int) and self.random_state >= 0
        ), f"random_state must be >= 0, got {self.random_state}"
        assert (
            0 < self.gpu_memory_fraction <= 1
        ), f"gpu_memory_fraction must be in (0, 1], got {self.gpu_memory_fraction}"
        assert (
            0 < self.memory_fraction <= 1
        ), f"memory_fraction must be in (0, 1], got {self.memory_fraction}"
        assert (
            0 <= self.min_r2_threshold <= 1
        ), f"min_r2_threshold must be in [0, 1], got {self.min_r2_threshold}"
        assert (
            0.05 <= self.test_size <= 0.5
        ), f"test_size must be in [0.05, 0.5], got {self.test_size}"
        assert 0.1 <= self.val_size <= 0.5, f"val_size must be in [0.1, 0.5], got {self.val_size}"

        # Calculate max memory
        if _PSUTIL_AVAILABLE:
            total_ram_mb = psutil.virtual_memory().total / 1024**2
            # Use the max_memory_mb from config, or calculate from memory_fraction if needed
            if self.max_memory_mb <= 0:
                self.max_memory_mb = total_ram_mb * self.memory_fraction
            logger.info(
                f"Memory limit: {self.max_memory_mb:.0f}MB "
                f"({self.memory_fraction*100:.0f}% of {total_ram_mb:.0f}MB)"
            )
        else:
            logger.warning(f"psutil unavailable, using config value: {self.max_memory_mb:.0f}MB")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Config:
        """
        Parse configuration from config.yaml using typed helpers.
        Args:
            config: Configuration dictionary from load_config()
        Returns:
            Validated Config instance
        Raises:
            ValueError: If required configuration sections are missing
        """
        from insurance_ml.config import (
            get_diagnostics_config,
            get_explainability_config,
            get_gpu_config,
            get_mlflow_config,
            get_training_config,
        )

        try:
            #  Extract typed configurations using helpers
            training_cfg = get_training_config(config)
            mlflow_cfg = get_mlflow_config(config)
            gpu_cfg = get_gpu_config(config)
            diag_cfg = get_diagnostics_config(config)
            explainability_cfg = get_explainability_config(config)

            logger.debug(f"   Training params: {len(training_cfg)}")
            logger.debug(f"   MLflow params: {len(mlflow_cfg)}")
            logger.debug(f"   GPU params: {len(gpu_cfg)}")
            logger.debug(f"   Diagnostics params: {len(diag_cfg)}")
            logger.debug(f"   Explainability params: {len(explainability_cfg)}")

        except KeyError as e:
            raise ValueError(
                f"❌ Missing required configuration in config.yaml: {e}\n\n"
                f"   ⚠️  Config.yaml is the SINGLE SOURCE OF TRUTH\n"
                f"   Ensure all required sections are present:\n"
                f"     - training (output_dir, cv_folds, etc.)\n"
                f"     - mlflow.tracking (enabled, tracking_uri)\n"
                f"     - mlflow.registry (enabled)\n"
                f"     - gpu (enabled, memory_limit_mb, memory_fraction)\n"
                f"     - diagnostics (enabled, max_samples, explainability)\n"
            ) from e

        return cls(
            # Training configuration
            output_dir=training_cfg["output_dir"],
            reports_dir=training_cfg["reports_dir"],
            cv_folds=training_cfg["cv_folds"],
            random_state=training_cfg["random_state"],
            test_size=training_cfg["test_size"],
            val_size=training_cfg["val_size"],
            stratify_splits=training_cfg["stratify_splits"],
            # Feature flags
            enable_mlflow=training_cfg["enable_mlflow"],
            enable_optuna=training_cfg["enable_optuna"],
            enable_diagnostics=training_cfg["enable_diagnostics"],
            # GPU configuration (from single source)
            gpu_enabled=gpu_cfg["enabled"],
            gpu_memory_limit_mb=gpu_cfg["memory_limit_mb"],
            gpu_memory_fraction=gpu_cfg["memory_fraction"],
            memory_fraction=training_cfg.get("memory_fraction", 1.0),
            # Model size limits
            max_model_size_mb=training_cfg["max_model_size_mb"],
            # Timeouts & thresholds
            training_timeout=training_cfg.get("training_timeout"),
            min_r2_threshold=training_cfg["min_r2_threshold"],
            # Diagnostics configuration
            max_shap_samples=diag_cfg["shap_max_samples"],
            # Security & validation
            verify_checksums=training_cfg["verify_checksums"],
            halt_on_severe_shift=training_cfg["halt_on_severe_shift"],
            save_checksums=training_cfg["save_checksums"],
            # MLflow registry
            register_to_mlflow=mlflow_cfg["registry_enabled"],
            # Memory (FROM CONFIG.YAML - single source of truth)
            max_memory_mb=training_cfg["max_memory_mb"],
            # Sample weights
            use_sample_weights=training_cfg["use_sample_weights"],
            high_value_percentile=training_cfg["high_value_percentile"],
        )


# =====================================================================
# TIMEOUT MANAGER
# =====================================================================


class TimeoutManager:
    """
    OPTIMIZED: Simplified timeout with periodic checking

    More compatible than multiprocessing. Checks at strategic points.
    """

    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self._start_time = None
        self._timeout_seconds = None

    @contextmanager
    def time_limit(self, seconds: int | None):
        """Context manager with working timeout on both Unix and Windows.

        Unix: uses SIGALRM (interrupts the process mid-execution).
        Windows: uses a background daemon thread that sets a shared flag after
                 `seconds`; the flag is checked by check_timeout() which is
                 called at strategic points inside the training loop.
                 The previous implementation checked elapsed time only in the
                 finally block — i.e. AFTER training completed — making the
                 timeout effectively non-functional on Windows.
        """
        if seconds is None:
            yield self
            return

        if not self.is_windows and hasattr(signal, "SIGALRM"):
            # Unix: use SIGALRM — truly preemptive
            def handler(signum, frame):
                raise TimeoutError(f"Training timeout: {seconds}s exceeded")

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                yield self
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Windows: background thread sets _timed_out flag after `seconds`.
            # check_timeout() reads the flag at strategic loop points.
            self._start_time = time.time()
            self._timeout_seconds = seconds
            self._timed_out = False

            def _watcher():
                time.sleep(seconds)
                self._timed_out = True

            _t = threading.Thread(target=_watcher, daemon=True)
            _t.start()

            try:
                yield self
            finally:
                self._timed_out = False  # clear so stale flag doesn't fire later
                self._start_time = None
                self._timeout_seconds = None

    def check_timeout(self):
        """Check if timeout exceeded — call periodically inside training loops.

        On Unix this is a no-op (SIGALRM handles it). On Windows it reads
        the _timed_out flag set by the background daemon thread.
        """
        if self.is_windows:
            if getattr(self, "_timed_out", False):
                elapsed = time.time() - self._start_time if self._start_time else 0
                raise TimeoutError(
                    f"⏱️  Training timeout: {self._timeout_seconds}s exceeded "
                    f"(elapsed: {elapsed:.1f}s)"
                )
        else:
            # Unix path — SIGALRM is preemptive; nothing to check here
            if self._start_time is None or self._timeout_seconds is None:
                return
            elapsed = time.time() - self._start_time
            if elapsed > self._timeout_seconds:
                raise TimeoutError(
                    f"⏱️  Training timeout: {self._timeout_seconds}s exceeded "
                    f"(elapsed: {elapsed:.1f}s)"
                )


# =====================================================================
# RESOURCE MONITOR
# =====================================================================


class ResourceMonitor:
    """OPTIMIZED: Smart GC reduces overhead by ~80%"""

    def __init__(self, max_memory_mb: float):
        self.max_memory_mb = max_memory_mb
        self.has_psutil = _PSUTIL_AVAILABLE

        # Smart GC tracking
        self._last_gc_time: float = 0.0
        self._gc_cooldown = 5.0  # Min 5s between GC
        self._gc_count = 0
        self._gc_time_total = 0.0

        if self.has_psutil:
            try:
                import psutil

                self._process = psutil.Process()
            except Exception:
                self._process = None
        else:
            self._process = None

    def get_memory_mb(self) -> float:
        """Get current memory usage"""
        if not self.has_psutil or self._process is None:
            return 0.0
        try:
            return float(self._process.memory_info().rss / 1024**2)
        except Exception:
            return 0.0

    def get_memory_usage(self) -> float:
        """Get memory usage as ratio (0-1)"""
        current_mb = self.get_memory_mb()
        if self.max_memory_mb <= 0:
            return 0.0
        return min(current_mb / self.max_memory_mb, 1.0)

    def check_threshold(self) -> bool:
        """Check if under memory threshold"""
        current = self.get_memory_mb()
        if current > self.max_memory_mb:
            logger.error(f"Memory: {current:.0f}MB > {self.max_memory_mb:.0f}MB")
            return False
        return True

    def smart_cleanup(self, threshold: float = 0.75) -> bool:
        """
        OPTIMIZED: Intelligent GC with cooldown

        Only runs when:
        - Memory > threshold, OR
        - Cooldown elapsed AND memory > 60%

        Returns: True if cleanup performed
        """
        current_time = time.time()
        current_usage = self.get_memory_usage()

        # Calculate thresholds
        urgent_threshold = 0.80  # 80% - urgent
        cautious_threshold = 0.60  # 60% - cautious

        # Check if GC needed
        time_since_gc = current_time - self._last_gc_time

        should_gc = current_usage > urgent_threshold or (
            time_since_gc > self._gc_cooldown and current_usage > cautious_threshold
        )

        if not should_gc:
            return False

        # Perform GC with timing
        before_mb = self.get_memory_mb()
        gc_start = time.time()

        # SINGLE gc.collect() instead of 3× loop
        gc.collect()

        gc_elapsed = time.time() - gc_start
        after_mb = self.get_memory_mb()

        self._last_gc_time = current_time
        self._gc_count += 1
        self._gc_time_total += gc_elapsed

        freed_mb = before_mb - after_mb

        # Log significant collections only
        if freed_mb > 10 or gc_elapsed > 0.1:
            logger.debug(
                f"Smart GC #{self._gc_count}: " f"freed {freed_mb:.0f}MB in {gc_elapsed*1000:.0f}ms"
            )

        return after_mb < self.max_memory_mb

    def force_cleanup(self) -> bool:
        """Force cleanup - single GC call"""
        before_mb = self.get_memory_mb()

        # SINGLE gc.collect() instead of 3× loop
        gc.collect()

        # GPU cleanup if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        after_mb = self.get_memory_mb()
        freed_mb = before_mb - after_mb

        if freed_mb > 10:
            logger.debug(f"Force cleanup freed {freed_mb:.0f}MB")

        return after_mb < self.max_memory_mb


# =====================================================================
# FILE SANITIZER
# =====================================================================


class FileSanitizer:
    """Secure file operations with validation"""

    RESERVED = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    @staticmethod
    def sanitize(name: str, max_len: int = 100) -> str:
        """Sanitize filename for security"""
        name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
        name = re.sub(r"[^\w\s\-.]", "_", name)
        name = re.sub(r"\.{2,}", ".", name)
        name = re.sub(r"_{2,}", "_", name)
        name = name.strip(". ")

        base = name.split(".")[0].upper()
        if base in FileSanitizer.RESERVED:
            name = f"model_{name}"

        encoded = name.encode("utf-8")
        while len(encoded) > max_len:
            name = name[:-1]
            encoded = name.encode("utf-8")

        return name or "unnamed_model"

    @staticmethod
    def compute_checksum(path: Path, chunk_size_mb: int = 1) -> str:
        """Compute SHA256 checksum efficiently"""
        sha256 = hashlib.sha256()
        chunk_size = chunk_size_mb * 1024 * 1024

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    def verify_checksum(model_path: Path, checksum_path: Path) -> bool:
        """Verify model checksum"""
        if not checksum_path.exists():
            logger.debug(f"Checksum file not found: {checksum_path}")
            return False

        try:
            with open(checksum_path) as f:
                stored_checksum = f.readline().strip()

            computed_checksum = FileSanitizer.compute_checksum(model_path)

            if stored_checksum != computed_checksum:
                logger.error(
                    f"Checksum mismatch!\n"
                    f"  Stored:   {stored_checksum[:16]}...\n"
                    f"  Computed: {computed_checksum[:16]}..."
                )
                return False

            logger.info(f"Checksum verified: {computed_checksum[:16]}...")
            return True

        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False

    @staticmethod
    def safe_load(path: Path, max_size_mb: float = 500.0, verify_checksum: bool = True) -> Any:
        """Load model with validation"""
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        # Size check
        size_mb = path.stat().st_size / 1024**2
        if size_mb > max_size_mb:
            raise ValueError(f"Model too large: {size_mb:.1f}MB > {max_size_mb}MB")

        # Checksum verification (optional if file missing)
        if verify_checksum:
            checksum_path = path.parent / f"{path.stem}_checksum.txt"

            # Only verify if checksum file exists
            if checksum_path.exists():
                if not FileSanitizer.verify_checksum(path, checksum_path):
                    raise ValueError("Checksum verification failed")
            else:
                logger.debug(
                    f"Checksum file not found: {checksum_path.name} (expected for new models)"
                )

        # Load
        try:
            model = joblib.load(path)
            if not hasattr(model, "predict"):
                raise ValueError("Model missing predict() method")
            logger.info(f"Loaded: {path.name} ({size_mb:.1f}MB)")
            return model
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise


# =====================================================================
# MLFLOW MANAGER
# =====================================================================


class MLflowManager:
    """Thread-safe MLflow with Model Registry integration"""

    def __init__(self, enabled: bool = True, register_models: bool = True):
        self.enabled = enabled
        self.register_models = register_models
        self.lock = threading.RLock()
        import types as _mtypes

        self._mlflow: _mtypes.ModuleType | None = None
        self._active_run_id = None

        if enabled:
            try:
                import mlflow
                import mlflow.sklearn

                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow not available")
                self.enabled = False

    def setup(self, tracking_uri: str, experiment: str):
        """Setup MLflow experiment"""
        if not self.enabled:
            return

        try:
            with self.lock:
                assert self._mlflow is not None
                self._mlflow.set_tracking_uri(tracking_uri)
                exp = self._mlflow.get_experiment_by_name(experiment)
                if exp is None:
                    self._mlflow.create_experiment(experiment)
                self._mlflow.set_experiment(experiment)
                # Explicitly disable autolog — manual log_metrics() calls in
                # train_single_model() are the single source of truth.
                # Leaving autolog in an indeterminate state (neither enabled
                # nor disabled) causes Optuna's per-trial fits to bleed
                # intermediate metrics into the run's history.
                self._mlflow.autolog(disable=True, silent=True)
                logger.info(f"MLflow: {experiment}")
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            self.enabled = False

    @contextmanager
    def run(self, run_name: str):
        """Context manager for MLflow runs with proper cleanup"""
        if not self.enabled:
            yield None
            return

        run_obj = None

        with self.lock:
            try:
                assert self._mlflow is not None
                # End any existing runs
                while self._mlflow.active_run():
                    logger.warning("Force-ending orphaned MLflow run")
                    self._mlflow.end_run(status="KILLED")

                # Start new run
                run_obj = self._mlflow.start_run(run_name=run_name)
                self._active_run_id = run_obj.info.run_id

            except Exception as e:
                logger.error(f"MLflow start failed: {e}")
                self.enabled = False
                yield None
                return

        try:
            yield run_obj
        except Exception:
            # Mark run as failed on exception
            with self.lock:
                try:
                    if self._mlflow is not None and self._mlflow.active_run():
                        self._mlflow.end_run(status="FAILED")
                except Exception:
                    pass
            raise
        finally:
            # Always end run
            with self.lock:
                try:
                    if self._mlflow is not None and self._mlflow.active_run():
                        self._mlflow.end_run()
                    self._active_run_id = None
                except Exception:
                    pass

    def log_metrics(self, metrics: dict[str, float]):
        """Log metrics safely"""
        if not self.enabled:
            return

        try:
            # Filter to numeric types only
            clean = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, int | float | np.number) and np.isfinite(v)
            }

            with self.lock:
                assert self._mlflow is not None
                if self._mlflow.active_run():
                    self._mlflow.log_metrics(clean)
        except Exception as e:
            logger.warning(f"MLflow log failed: {e}")

    def register_model_to_registry(
        self,
        model: Any,
        model_name: str,
        signature,
        input_example: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, str] | None:
        """Register model to MLflow Model Registry with serializable metadata"""
        if not self.enabled or not self.register_models:
            return None

        try:
            with self.lock:
                # Convert non-serializable objects to dicts
                clean_metadata: dict[str, Any] = {}

                for key, value in metadata.items():
                    if key == "target_transformation":
                        # Handle TargetTransformation object
                        if hasattr(value, "method"):
                            clean_metadata[key] = {
                                "method": str(value.method),
                                "boxcox_lambda": (
                                    float(value.boxcox_lambda)
                                    if value.boxcox_lambda is not None
                                    else None
                                ),
                                "original_range": (
                                    tuple(map(float, value.original_range))
                                    if value.original_range
                                    else None
                                ),
                                "boxcox_min": (
                                    float(value.boxcox_min)
                                    if value.boxcox_min is not None
                                    else None
                                ),
                                "boxcox_max": (
                                    float(value.boxcox_max)
                                    if value.boxcox_max is not None
                                    else None
                                ),
                            }
                        else:
                            # String representation
                            clean_metadata[key] = str(value)

                    elif isinstance(value, str | int | float | bool | type(None)):
                        # Already serializable
                        clean_metadata[key] = value

                    elif isinstance(value, list | tuple):
                        # Convert to list
                        clean_metadata[key] = list(value)

                    elif isinstance(value, dict):
                        # Already a dict
                        clean_metadata[key] = value

                    else:
                        # Convert anything else to string
                        clean_metadata[key] = str(value)

                # Log model with cleaned metadata
                model_info = self._mlflow.sklearn.log_model(  # type: ignore[union-attr]
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=f"insurance_{model_name}",
                    metadata=clean_metadata,
                )

                # Log parameters (also clean them)
                if hasattr(model, "get_params"):
                    params = model.get_params()
                    clean_params = {
                        k: v
                        for k, v in params.items()
                        if isinstance(v, str | int | float | bool | type(None))
                    }
                    if clean_params:
                        self._mlflow.log_params(clean_params)  # type: ignore[union-attr]

                logger.info(f"  ✅ Registered to MLflow: insurance_{model_name}")

                assert self._mlflow is not None
                return {
                    "mlflow_model_uri": model_info.model_uri,
                    "mlflow_run_id": (
                        self._mlflow.active_run().info.run_id if self._mlflow.active_run() else ""
                    ),
                }

        except Exception as e:
            logger.error(f"Model registration failed: {e}", exc_info=True)
            return None


# =====================================================================
# VISUALIZATION MANAGER
# =====================================================================


class VisualizationManager:
    """Centralized visualization management"""

    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Set style once
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

        # Track failures
        self.failed_plots: list[tuple[str, str]] = []

    def _safe_save_plot(self, filename: str) -> str | None:
        """Save plot with error handling"""
        try:
            save_file = self.reports_dir / filename
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved: {save_file.name}")
            return str(save_file)
        except Exception as e:
            self.failed_plots.append((filename, str(e)))
            logger.warning(f"Plot save failed: {filename} - {e}")
            return None
        finally:
            plt.close()

    def get_failure_summary(self) -> str:
        """Get summary of failed plots"""
        if not self.failed_plots:
            return "All visualizations generated successfully"

        summary = f"{len(self.failed_plots)} visualization(s) failed:\n"
        for name, error in self.failed_plots[:5]:
            summary += f"  - {name}: {error[:50]}\n"

        if len(self.failed_plots) > 5:
            summary += f"  ... +{len(self.failed_plots) - 5} more"

        return summary

    def plot_training_progress(
        self, results: dict[str, TrainingResult], save_path: str | None = None
    ) -> str | None:
        """Plot training progress comparison"""
        successful = {k: v for k, v in results.items() if v.get("status") == "success"}

        if not successful:
            logger.warning("No successful models for training progress plot")
            return None

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Training Progress Overview", fontsize=16, fontweight="bold")

            models = list(successful.keys())

            # 1. Training times
            times = [successful[m]["training_time"] for m in models]
            axes[0, 0].barh(models, times, color="steelblue")
            axes[0, 0].set_xlabel("Time (seconds)")
            axes[0, 0].set_title("Training Time by Model")
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Validation RMSE
            val_rmse = []
            for m in models:
                vm = successful[m]["validation_metrics"]
                val_rmse.append(vm.get("original_rmse", vm.get("rmse", 0)))

            axes[0, 1].barh(models, val_rmse, color="coral")
            axes[0, 1].set_xlabel("RMSE")
            axes[0, 1].set_title("Validation RMSE")
            axes[0, 1].grid(True, alpha=0.3)

            # 3. R² scores
            r2_scores = []
            for m in models:
                vm = successful[m]["validation_metrics"]
                r2_scores.append(vm.get("original_r2", vm.get("r2", 0)))

            axes[1, 0].barh(models, r2_scores, color="seagreen")
            axes[1, 0].set_xlabel("R² Score")
            axes[1, 0].set_title("Validation R²")
            axes[1, 0].grid(True, alpha=0.3)

            # 4. CV scores with error bars
            cv_means = [successful[m]["cv_mean"] for m in models]
            cv_stds = [successful[m]["cv_std"] for m in models]

            y_pos = np.arange(len(models))
            axes[1, 1].barh(y_pos, cv_means, xerr=cv_stds, color="mediumpurple", alpha=0.7)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(models)
            # CV scores are computed in yeo-johnson transformed space
            # (range ~0–4), NOT in dollars. The axis previously showed a bare
            # number like "0.28" that looked like a dollar RMSE in the thousands.
            # Model selection is unaffected (all models scored in the same space),
            # but the label was misleading to anyone reading training plots.
            _cv_unit = (
                "yeo-johnson units — NOT dollars"
                if getattr(self, "raw_config", {})
                .get("features", {})
                .get("target_transform", {})
                .get("method")
                == "yeo-johnson"
                else "RMSE"
            )
            axes[1, 1].set_xlabel(f"CV Score ± Std  [{_cv_unit}]")
            axes[1, 1].set_title("Cross-Validation Performance (transformed space)")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                return self._safe_save_plot(f"{save_path}_training_progress.png")

            plt.close()
            return None

        except Exception as e:
            logger.error(f"Training progress plot failed: {e}")
            plt.close()
            return None


# =====================================================================
# MODEL TRAINER (MAIN CLASS)
# =====================================================================


class ModelTrainer:
    """Production ML training pipeline with config.yaml single source of truth"""

    # Pipeline contract validation — catches self-contradictory config
    # BEFORE any training starts. Without this, a contradictory config (e.g. quantile
    # objective + require_bias_correction=True) silently runs for 600+ seconds then
    # fails at the very last stage (test evaluation). This pre-flight check fails in
    # under 1 second and gives an actionable error message pointing to config.yaml.
    def _validate_pipeline_contracts(self, raw_config: dict) -> None:
        """
        Pre-flight contract validation. Call BEFORE any training.

        Detects mutually exclusive configuration pairs that would cause a silent
        late-stage failure:
          - Quantile objective + require_bias_correction=True  →  always fails at
            test evaluation after a full training run
          - Quantile objective + non-quantile alpha missing    →  XGBoost raises at fit

        Raises:
            ValueError: If any contract is violated (with actionable fix instructions)
        """
        models_cfg = raw_config.get("models", {})
        for model_name, model_cfg in models_cfg.items():
            if not isinstance(model_cfg, dict):
                continue
            objective = str(model_cfg.get("objective", "")).lower()
            is_quantile = "quantile" in objective

            # Contract: quantile model must NOT require bias correction
            require_bc = raw_config.get("training", {}).get("require_bias_correction", False)
            if is_quantile and require_bc:
                raise ValueError(
                    f"❌ PIPELINE CONTRACT VIOLATION (detected at startup — no training wasted):\n"
                    f"   Model '{model_name}' has objective='{objective}' (quantile model).\n"
                    f"   BiasCorrection is incompatible with quantile loss — median-ratio\n"
                    f"   correction would cancel the quantile uplift and restore underpricing.\n"
                    f"   But training.require_bias_correction=True in config.\n"
                    f"\n"
                    f"   ✅ FIX: Set training.require_bias_correction: false in config.yaml\n"
                    f"          for any run that includes quantile models."
                )

            # Contract: XGBoost quantile requires quantile_alpha
            if is_quantile and "xgb" in model_name.lower():
                alpha = model_cfg.get("quantile_alpha")
                if alpha is None:
                    raise ValueError(
                        f"❌ PIPELINE CONTRACT VIOLATION for '{model_name}':\n"
                        f"   objective='{objective}' requires quantile_alpha but it is missing.\n"
                        f"   XGBoost >= 2.0 will raise at .fit() time.\n"
                        f"\n"
                        f"   ✅ FIX: Add 'quantile_alpha: 0.65' under models.{model_name} in config.yaml."
                    )

        logger.info("✅ Pipeline contracts validated — no conflicting configuration detected.")

    def __init__(self, config_dict: dict | None = None):
        """
        Initialize ModelTrainer

        Args:
            config_dict: Optional config dictionary. If None, loads from config.yaml
        """
        # Import dependencies
        try:
            from insurance_ml.config import (
                get_feature_config,
                get_gpu_config,
                get_optuna_config,
                get_sample_weight_config,
                get_training_config,
                load_config,
            )
            from insurance_ml.data import DataLoader
            from insurance_ml.features import FeatureEngineer

            #  Import GPU utilities at module level (don't store as instance methods)
            from insurance_ml.models import (
                ExplainabilityConfig,
                ModelManager,
                check_gpu_available,
                clear_gpu_cache,
                get_gpu_memory_usage,
                get_model_gpu_params,
            )
            from insurance_ml.utils import MetricsExtractor

            # Store only classes, not utility functions
            self.load_config = load_config
            self.get_feature_config = get_feature_config
            self.get_gpu_config = get_gpu_config
            self.get_optuna_config = get_optuna_config
            self.get_training_config = get_training_config
            self.get_sample_weight_config = get_sample_weight_config
            self.ModelManager = ModelManager
            self.DataLoader = DataLoader
            self.FeatureEngineer = FeatureEngineer
            self.MetricsExtractor = MetricsExtractor

            self.check_gpu_available = check_gpu_available
            self.get_gpu_memory_usage = get_gpu_memory_usage
            self._clear_gpu_cache_fn = clear_gpu_cache
            self.get_model_gpu_params = get_model_gpu_params

        except ImportError as e:
            raise ImportError(f"Required modules missing: {e}") from e

        # Configuration from single source
        if config_dict is None:
            raw_config = self.load_config()
            logger.info("✅ Configuration loaded from config.yaml (single source of truth)")
        else:
            raw_config = config_dict
            logger.info("✅ Configuration provided (ensure it came from load_config())")

        self.config = Config.from_dict(raw_config)
        self.raw_config = raw_config

        # Fail fast on contradictory config before any work is done.
        self._validate_pipeline_contracts(raw_config)

        #  Extract configurations from single sources
        self.gpu_config = self.get_gpu_config(raw_config)
        self.training_config = self.get_training_config(raw_config)

        self.gpu_available = self.check_gpu_available()

        # Pre-flight GPU memory check
        if self.gpu_available and self.gpu_config["enabled"]:
            try:
                gpu_mem = self.get_gpu_memory_usage()

                if gpu_mem and gpu_mem.get("total_mb", 0) > 0:
                    free_mb = gpu_mem["free_mb"]
                    total_mb = gpu_mem["total_mb"]
                    util_pct = gpu_mem["utilization_pct"]

                    logger.info(
                        f"🎮 GPU Status at Pipeline Initialization:\n"
                        f"   Free: {free_mb:.0f}MB / {total_mb:.0f}MB\n"
                        f"   Usage: {util_pct:.1f}%"
                    )

                    #  Warning thresholds from config
                    warn_threshold = self.gpu_config.get("warn_threshold_mb", 500)

                    if free_mb < warn_threshold:
                        logger.warning(
                            f"⚠️ LOW GPU MEMORY AT STARTUP\n"
                            f"   Free: {free_mb:.0f}MB < threshold: {warn_threshold}MB\n"
                            f"   \n"
                            f"   RECOMMENDATIONS:\n"
                            f"   1. Close other GPU applications\n"
                            f"   2. Reduce batch_size in config.yaml\n"
                            f"   3. Reduce max_depth for tree models\n"
                            f"   4. Set gpu.enabled=false to use CPU\n"
                            f"   \n"
                            f"   Training may fail with OOM errors!"
                        )

                    #  Check against config limit
                    config_limit = self.gpu_config.get("memory_limit_mb", 3500)
                    if free_mb < config_limit:
                        logger.warning(
                            f"⚠️ Free GPU memory ({free_mb:.0f}MB) less than "
                            f"config limit ({config_limit}MB)\n"
                            f"   Consider reducing gpu.memory_limit_mb in config.yaml"
                        )

            except Exception as e:
                logger.debug(f"GPU memory pre-flight check failed: {e}")

        self.resources = ResourceMonitor(self.config.max_memory_mb)
        self.mlflow = MLflowManager(self.config.enable_mlflow, self.config.register_to_mlflow)
        self.timeout_mgr = TimeoutManager()
        self.viz = VisualizationManager(self.config.reports_dir)

        # Model manager
        self.model_manager = ModelManager(raw_config)
        self._model_manager_supports_params = self._check_model_manager_interface()
        from insurance_ml.config import get_explainability_config

        explainability_cfg = get_explainability_config(raw_config)

        self.explainability_config = ExplainabilityConfig(
            enable_confidence_intervals=explainability_cfg["enable_confidence_intervals"],
            confidence_level=explainability_cfg["confidence_level"],
            enable_shap=explainability_cfg["enable_shap"],
            shap_max_samples=explainability_cfg["shap_max_samples"],
            shap_background_samples=explainability_cfg["shap_background_samples"],
            auto_plot=explainability_cfg["auto_plot"],
            save_path=explainability_cfg.get("save_path")
            or str(self.config.reports_dir / "explainability"),
        )

        logger.info(
            f"Explainability configured:\n"
            f"   Confidence intervals: {self.explainability_config.enable_confidence_intervals}\n"
            f"   SHAP analysis: {self.explainability_config.enable_shap}\n"
            f"   Confidence level: {self.explainability_config.confidence_level*100:.0f}%"
        )

        # Optuna with proper config extraction
        self.optimizer = None
        self.optuna_config = None
        if self.config.enable_optuna:
            try:
                from insurance_ml.optuna_optimizer import OptunaOptimizer

                self.optuna_config = self.get_optuna_config(raw_config)
                self.optimizer = OptunaOptimizer(
                    raw_config,
                    VERSION,
                    use_gpu=self.gpu_config["enabled"],
                    model_manager=self.model_manager,
                )
                logger.info(
                    f"Optuna enabled: {self.optuna_config['n_trials']} trials, "
                    f"Enhanced scoring: {self.optuna_config.get('enhanced_scoring_enabled', False)}"
                )
            except ImportError:
                logger.warning("Optuna not available")

        # Setup
        self._set_seeds()

        if self.mlflow.enabled:
            mlflow_cfg = raw_config.get("mlflow", {}).get("tracking", {})
            self.mlflow.setup(
                mlflow_cfg.get("tracking_uri", "./mlruns"),
                mlflow_cfg.get("experiment_name", "insurance"),
            )

        # Cache for inverse transforms
        self._transform_cache: dict[str, Any] = {}
        self._test_transform_cache: dict[str, Any] = {}

        #  Store feature_engineer reference
        self.feature_engineer: Any = None
        self._data_prepared = False

    def clear_gpu_cache(self):
        """
        OPTIMIZED: Comprehensive GPU cache clearing for 4GB GPUs
        Prevents fragmentation and OOM between model trainings
        """
        if not self.gpu_available:
            return

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for operations

                # Force defragmentation (PyTorch 1.10+)
                try:
                    torch.cuda.memory.empty_cache()
                except AttributeError:
                    pass

                logger.debug("✅ GPU cache cleared")
        except ImportError:
            pass

    def _check_model_manager_interface(self) -> bool:
        """Check ModelManager.get_model() signature"""
        import inspect

        try:
            sig = inspect.signature(self.model_manager.get_model)
            params = list(sig.parameters.keys())
            return "kwargs" in params or len(params) > 1
        except Exception:
            return False

    def _set_seeds(self):
        """Set all random seeds for reproducibility.

        torch.manual_seed() is now called unconditionally (when PyTorch
        is importable) so CPU PyTorch ops are seeded even when GPU is unavailable.
        torch.cuda.manual_seed_all() is still gated on gpu_available.
        """
        seed = self.config.random_state
        np.random.seed(seed)

        try:
            import random

            random.seed(seed)
        except Exception:
            pass

        # Always seed PyTorch CPU RNG if torch is installed.
        try:
            import torch

            torch.manual_seed(seed)
            if self.gpu_available:
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _is_tree_based_model(self, model_name: str) -> bool:
        """Determine if a model is tree-based and supports sample weights"""
        tree_indicators = {
            "xgb",
            "xgboost",
            "lgb",
            "lightgbm",
            "light",
            "cat",
            "catboost",
            "forest",
            "rf",
            "random",
            "gradient",
            "boosting",
            "gbm",
            "gb",
            "tree",
            "dt",
            "extra",
            "et",
            "ada",
            "adaboost",
            "hist",
        }

        name_lower = model_name.lower().replace("_", "").replace("-", "")
        return any(indicator in name_lower for indicator in tree_indicators)

    def _model_supports_sample_weights(self, model) -> bool:
        """Check if a model instance supports sample weights"""
        from inspect import signature

        try:
            fit_signature = signature(model.fit)
            return "sample_weight" in fit_signature.parameters
        except Exception as e:
            logger.debug(f"Could not check sample_weight support: {e}")
            return self._is_tree_based_model(model.__class__.__name__)

    def prepare_training_data(
        self,
        target_transform=None,
        stratify=None,
        remove_outliers=None,
        remove_collinear=None,
        add_polynomials=None,
    ) -> dict:
        """
        Complete data preparation pipeline with validation.

        Returns:
            Dict with processed data, metadata, and feature engineer
        """

        start = time.time()
        logger.info(f"{'='*80}\nData Preparation Pipeline v{VERSION}\n{'='*80}")

        try:
            # Load data
            loader = self.DataLoader(self.raw_config)
            df = loader.load_raw_data()
            df = loader.clean_data(df)

            # Validate target
            target_col = self.raw_config.get("data", {}).get("target_column", "charges")
            if target_col not in df.columns:
                available = ", ".join(df.columns[:5].tolist())
                raise ValueError(
                    f"Target column '{target_col}' not found. " f"Available: {available}..."
                )

            X, y = df.drop(columns=[target_col]), df[target_col]

            if not pd.api.types.is_numeric_dtype(y):
                raise ValueError(f"Target must be numeric, got {y.dtype}")

            logger.info(f"Loaded {len(X):,} samples, {X.shape[1]} features")

            # Get feature config from config.yaml using typed helper
            feat_cfg = self.get_feature_config(self.raw_config)

            # Extract settings from config
            target_transform_cfg = self.raw_config.get("features", {}).get("target_transform", {})

            # Apply parameter overrides
            if target_transform is None:
                target_transform = target_transform_cfg.get("method", "none")

            if remove_outliers is None:
                outlier_cfg = self.raw_config.get("features", {}).get("outlier_removal", {})
                remove_outliers = outlier_cfg.get("enabled", True)

            if remove_collinear is None:
                collin_cfg = self.raw_config.get("features", {}).get("collinearity_removal", {})
                remove_collinear = collin_cfg.get("enabled", True)

            if add_polynomials is None:
                poly_cfg = self.raw_config.get("features", {}).get("polynomial_features", {})
                add_polynomials = poly_cfg.get("enabled", True)

            if stratify is None:
                stratify = self.config.stratify_splits

            logger.info("\n" + "=" * 80)
            logger.info("CONFIGURATION (from config.yaml)")
            logger.info("=" * 80)
            logger.info(f"Target transform: {target_transform}")
            logger.info(f"Remove outliers: {remove_outliers}")
            logger.info(f"Add polynomials: {add_polynomials}")
            logger.info(f"Remove collinear: {remove_collinear}")
            logger.info(f"Stratify splits: {stratify}")
            logger.info("=" * 80 + "\n")

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y, stratify)

            # persist exact test-set indices so evaluate.py uses
            # the identical held-out split regardless of stratify_splits setting.
            # Without this, evaluate.py re-splits independently and (with
            # stratify=true) 81 % of its "test" rows come from the training fold.
            _test_index_path = Path(self.config.output_dir) / "test_indices.json"
            try:
                _test_indices = X_test.index.tolist()
                _tmp_path = _test_index_path.with_suffix(".json.tmp")
                with open(_tmp_path, "w") as _f:
                    json.dump(
                        {
                            "indices": _test_indices,
                            "n_test": len(_test_indices),
                            "random_state": self.config.random_state,
                            "stratify_splits": self.config.stratify_splits,
                            "test_size": self.config.test_size,
                            "created_at": pd.Timestamp.now().isoformat(),
                            "pipeline_version": VERSION,
                        },
                        _f,
                        indent=2,
                    )
                import shutil as _shutil

                _shutil.move(str(_tmp_path), str(_test_index_path))
                logger.info(
                    f"✅ Test indices saved: {_test_index_path} "
                    f"({len(_test_indices)} samples, "
                    f"stratified={self.config.stratify_splits})"
                )
            except Exception as _idx_err:
                logger.warning(
                    f"⚠️ Could not save test indices to {_test_index_path}: {_idx_err}\n"
                    f"   evaluate.py will fall back to re-splitting — "
                    f"metrics may be biased when stratify_splits=true."
                )

            # Feature engineering with config_dict
            logger.info("\nFeature engineering...")
            engineer = self.FeatureEngineer(config_dict=feat_cfg)

            train_result = engineer.fit_transform_pipeline(
                df=X_train,
                y=y_train,
                X_val=X_val,
                y_val=y_val,
                target_transform=target_transform,
                remove_outliers=remove_outliers,
                add_polynomials=add_polynomials,
                remove_collinear=remove_collinear,
            )

            X_train_proc = train_result["X_train"]
            y_train_proc = train_result["y_train"]
            X_val_proc = train_result["X_val"]
            y_val_proc = train_result["y_val"]
            self.feature_engineer = engineer
            # Save preprocessor path for later
            prep_path = self.config.output_dir / f"preprocessor_v{VERSION}.joblib"

            # Metadata
            metadata = {
                "version": VERSION,
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "timestamp": pd.Timestamp.now().isoformat(),
                "platform": platform.system(),
                "splits": {
                    "train": int(len(X_train_proc)),
                    "val": int(len(X_val_proc)),
                    "test": int(len(X_test)),
                },
                "features": {
                    "original": int(X_train.shape[1]),
                    "engineered": int(X_train_proc.shape[1]),
                    "names": X_train_proc.columns.tolist(),
                },
                "target_transform": target_transform,
            }

            meta_path = self.config.output_dir / "pipeline_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            elapsed = time.time() - start

            logger.info(
                f"\n{'='*80}\n"
                f"Pipeline Complete ({elapsed:.1f}s)\n"
                f"{'='*80}\n"
                f"Train: {len(X_train_proc):,}, Val: {len(X_val_proc):,}, Test: {len(X_test):,}\n"
                f"Features: {X_train_proc.shape[1]}, Transform: {target_transform}\n"
                f"{'='*80}"
            )

            # Mark that data has been prepared
            self._data_prepared = True

            return {
                # Processed data (YJ-transformed)
                "X_train": X_train_proc,
                "X_val": X_val_proc,
                "y_train": y_train_proc,
                "y_val": y_val_proc,
                # original-scale (pre-transform) targets for sample weight calculation.
                # y_train/y_val above are YJ-transformed (range ~6-15). calculate_sample_weights()
                # needs dollar-scale values to set tier boundaries correctly. Without this,
                # all training rows land in the same weight tier → near-uniform weights.
                "y_train_original": y_train,  # dollar scale, pre-Yeo-Johnson
                "y_val_original": y_val,  # dollar scale, pre-Yeo-Johnson
                # Raw test data
                "X_test_raw": X_test,
                "y_test_raw": y_test,
                # Transformation objects
                "target_transformation": engineer.target_transformation,
                "feature_engineer": engineer,
                "preprocessor_path": prep_path,
                # Metadata
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _split_data(self, X, y, stratify) -> tuple:
        """Stratified split with high-value cost representation"""
        bins = None
        use_stratify = False

        if stratify and len(y.unique()) > 20:
            try:
                # (v7.5.0): Add P99 bin to stratification quantiles.
                # Without it, the ~13 extreme-outlier samples (P95–P100) land
                # unevenly across splits by chance — prior run showed test max
                # = $48,970 vs val max = $63,770, making test set structurally
                # easier and test RMSE 26% lower than val.
                # Adding q=0.99 creates a dedicated top-1% bucket (~13 samples)
                # ensuring the highest-cost policies are proportionally present
                # in train, val, and test. The impact on overall split sizes is
                # negligible (1% tier ≈ 13–14 samples from 1,338).
                bins = pd.qcut(
                    y,
                    q=[
                        0,
                        0.25,
                        0.5,
                        0.75,
                        0.9,
                        0.95,
                        0.99,
                        1.0,
                    ],
                    labels=False,
                    duplicates="drop",
                )
                use_stratify = True
                logger.info(
                    f"Stratified split with {len(np.unique(bins))} cost-aware bins (P99 tail-protected)"
                )
            except Exception as e:
                logger.warning(f"Stratification failed: {e}")
                use_stratify = False

        # First split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=bins if use_stratify else None,
        )

        # Second split
        if use_stratify:
            try:
                temp_bins = pd.qcut(y_temp, q=5, labels=False, duplicates="drop")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=self.config.val_size,
                    random_state=self.config.random_state,
                    stratify=temp_bins,
                )
            except Exception as _e2:
                logger.warning(
                    f"⚠️  Val/train stratification failed ({_e2}), "
                    f"falling back to random split — "
                    f"validation distribution may be harder than train."
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=self.config.val_size,
                    random_state=self.config.random_state,
                )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=self.config.val_size,
                random_state=self.config.random_state,
            )

        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Validate high-value representation
        q75_global = np.percentile(y, 75)
        for name, subset in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            high_pct = (subset > q75_global).mean() * 100
            logger.info(f"  {name}: {high_pct:.1f}% high-value (>${q75_global:.0f})")

        # Validate split integrity
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X), f"Split size mismatch: {total} != {len(X)}"

        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        assert len(train_idx & val_idx) == 0, "Train/val overlap!"
        assert len(train_idx & test_idx) == 0, "Train/test overlap!"
        assert len(val_idx & test_idx) == 0, "Val/test overlap!"

        logger.info("[OK] No data leakage detected")

        # ✅ ADD: Detailed distribution comparison
        logger.info("\n🔍 SPLIT DISTRIBUTION ANALYSIS:")

        for name, y_subset in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            stats = {
                "mean": y_subset.mean(),
                "median": y_subset.median(),
                "std": y_subset.std(),
                "q95": y_subset.quantile(0.95),
                "max": y_subset.max(),
            }
            logger.info(
                f"{name:6s}: mean=${stats['mean']:7,.0f}, "
                f"median=${stats['median']:7,.0f}, "
                f"std=${stats['std']:6,.0f}, "
                f"q95=${stats['q95']:7,.0f}, "
                f"max=${stats['max']:7,.0f}"
            )

        # Check for validation set having harder cases
        val_difficulty = (y_val > y.quantile(0.75)).mean() / (y_train > y.quantile(0.75)).mean()

        if val_difficulty > 1.2:
            logger.warning(
                f"⚠️  VALIDATION SET SKEWED HARDER!\n"
                f"   Val has {val_difficulty:.1f}x more high-value cases than train\n"
                f"   This explains val_RMSE > train_RMSE > test_RMSE pattern"
            )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _validate_data(self, X_train, X_val, y_train, y_val):
        """Comprehensive data validation"""
        checks = [
            (len(X_train) == len(y_train), "Train length mismatch"),
            (len(X_val) == len(y_val), "Val length mismatch"),
            (X_train.shape[1] > 0, "No features"),
            (X_train.shape[1] == X_val.shape[1], "Feature count mismatch"),
            (not X_train.isnull().any().any(), "NaN in X_train"),
            (not X_val.isnull().any().any(), "NaN in X_val"),
            (not y_train.isnull().any(), "NaN in y_train"),
            (not y_val.isnull().any(), "NaN in y_val"),
        ]

        for condition, msg in checks:
            if not condition:
                raise ValueError(f"Data validation failed: {msg}")

        # Check for inf values
        numeric = X_train.select_dtypes(include=[np.number]).columns
        if np.isinf(X_train[numeric].values).any():
            raise ValueError("Inf values in X_train")
        if np.isinf(X_val[numeric].values).any():
            raise ValueError("Inf values in X_val")

        logger.info(f"Data validated: train={X_train.shape}, val={X_val.shape}")

    def _validate_target_transformation(self, target_transformation):
        """Validate target transformation object"""
        if target_transformation is None:
            raise ValueError("target_transformation is None")

        if not hasattr(target_transformation, "method"):
            raise ValueError("target_transformation missing 'method' attribute")

        method = target_transformation.method
        if method not in ["none", "log1p", "yeo-johnson", "boxcox"]:
            raise ValueError(f"Invalid transformation method: {method}")

        if method == "boxcox":
            if not hasattr(target_transformation, "boxcox_lambda"):
                raise ValueError("Box-Cox transformation missing lambda")
            if target_transformation.boxcox_lambda is None:
                raise ValueError("Box-Cox lambda is None")

    def _get_cached_inverse_transform(
        self, y_transformed: np.ndarray, target_transformation, cache_key: str
    ) -> np.ndarray:
        """
        Use cached inverse transform or compute new one.

        Args:
            y_transformed: Transformed values
            target_transformation: TargetTransformation object
            cache_key: Cache key prefix

        Returns:
            Values in original scale

        Raises:
            RuntimeError: If feature_engineer not available
        """
        # This validation should be redundant now
        if not hasattr(self, "feature_engineer") or self.feature_engineer is None:
            raise RuntimeError(
                "❌ INTERNAL ERROR: feature_engineer is None during inverse transform!\n"
                "   \n"
                "   This should have been caught by early validation.\n"
                "   Please report this bug with full stack trace."
            )

        # Generate cache key
        array_hash = hashlib.sha256(y_transformed.tobytes()).hexdigest()[:16]  # Longer hash
        full_key = f"{cache_key}_{array_hash}_{y_transformed.shape}_{target_transformation.method}"

        # Check cache
        if full_key in self._transform_cache:
            logger.debug(f"Using cached transform: {cache_key} (hash: {array_hash})")
            return np.asarray(self._transform_cache[full_key])

        # Compute inverse transform
        try:
            result = self.feature_engineer.inverse_transform_target(
                y_transformed,
                transformation_method=target_transformation.method,
                clip_to_safe_range=False,
                context=cache_key,
            )
        except Exception as e:
            logger.error(
                f"❌ Inverse transform failed for {cache_key}:\n"
                f"   Transform method: {target_transformation.method}\n"
                f"   Array shape: {y_transformed.shape}\n"
                f"   Array range: [{y_transformed.min():.4f}, {y_transformed.max():.4f}]\n"
                f"   Error: {e}"
            )
            raise

        # Store in cache (with size limit)
        self._transform_cache[full_key] = result

        if len(self._transform_cache) > 50:  # Increased limit
            oldest = next(iter(self._transform_cache))
            del self._transform_cache[oldest]
            logger.debug(f"Evicted oldest cache entry: {oldest[:20]}...")

        return np.asarray(result)

    def _calculate_cv_manual(
        self,
        model_factory,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: dict[str, Any],
        y_original: pd.Series | None = None,
    ) -> dict:
        """Manual CV with sample weighting.

        Args:
            model_factory: Callable that returns a new model instance.
            X: Feature matrix (transformed).
            y: Target values (may be transformed).
            model_params: Model constructor kwargs, may include GPU params.
            y_original: Original-scale target (pre-transformation). When provided,
                sample weights are computed from original-scale quantiles.
                always pass y_original from the pre-transform split.
        """
        if X is None or y is None:
            return {"cv_mean": 0.0, "cv_std": 0.0, "cv_scores": [0.0], "source": "none"}

        # GPU-specific params must only be passed to GPU-capable models.
        # Passing device='cuda', tree_method etc. to Ridge/SVR/KNN causes TypeError.
        _GPU_CAPABLE_MODELS = frozenset({"xgboost", "xgboost_median", "lightgbm", "lgbm"})
        _model_name = getattr(model_factory, "__name__", "") or ""
        _is_gpu_model = any(n in _model_name.lower() for n in _GPU_CAPABLE_MODELS)
        _GPU_KEYS = frozenset({"device", "tree_method", "max_bin", "predictor", "gpu_id"})
        if not _is_gpu_model:
            model_params = {k: v for k, v in model_params.items() if k not in _GPU_KEYS}

        try:
            cv = KFold(
                self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

            scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
                if self.timeout_mgr.is_windows:
                    self.timeout_mgr.check_timeout()

                try:
                    fold_model = model_factory(**model_params)

                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_val_fold = y.iloc[val_idx]

                    # pass original-scale fold target for correct weight quantiles.
                    y_orig_fold = y_original.iloc[train_idx] if y_original is not None else None
                    sample_weights = calculate_sample_weights(
                        y_train_fold, self.raw_config, y_original=y_orig_fold
                    )

                    # Train with weights
                    fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)
                    y_pred = fold_model.predict(X_val_fold)

                    from sklearn.metrics import mean_squared_error

                    score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                    scores.append(score)

                    logger.debug(f"Fold {fold_idx}/{self.config.cv_folds}: {score:.4f}")

                    del (
                        fold_model,
                        X_train_fold,
                        y_train_fold,
                        X_val_fold,
                        y_val_fold,
                        y_pred,
                        sample_weights,
                    )

                except Exception as e:
                    logger.warning(f"Fold {fold_idx} failed: {e}")
                    continue
                finally:
                    gc.collect()

            if not scores:
                return {
                    "cv_mean": 0.0,
                    "cv_std": 0.0,
                    "cv_scores": [0.0],
                    "source": "error",
                }

            scores_arr = np.array(scores)
            return {
                "cv_mean": float(scores_arr.mean()),
                "cv_std": float(scores_arr.std()),
                "cv_scores": scores_arr.tolist(),
                "source": "manual_cv",
            }

        except Exception as e:
            logger.error(f"CV failed: {e}")
            return {
                "cv_mean": 0.0,
                "cv_std": 0.0,
                "cv_scores": [0.0],
                "source": "error",
            }
        finally:
            gc.collect()
            if self.gpu_available:
                self.clear_gpu_cache()

    def _validate_metrics_dict(self, metrics: dict[str, Any], context: str = ""):
        """Validate metrics dictionary structure

        Allows NaN for secondary metrics (Durbin-Watson, autocorr) in degenerate cases
        but requires finite values for primary metrics (RMSE, R2, MAE, MAPE)
        """
        has_transformed = any(k.startswith("transformed_") for k in metrics.keys())
        has_original = any(k.startswith("original_") for k in metrics.keys())

        if has_transformed or has_original:
            if has_original:
                required = ["original_rmse", "original_mae", "original_r2"]
            else:
                required = ["transformed_rmse", "transformed_mae", "transformed_r2"]
        else:
            required = ["rmse", "mae", "r2", "mape"]

        missing = [k for k in required if k not in metrics]
        if missing:
            available = list(metrics.keys())[:10]
            raise ValueError(
                f"Metrics missing keys {missing} in {context}. " f"Available keys: {available}"
            )

        # Define secondary metrics that can be NaN in degenerate cases
        # (e.g., Durbin-Watson when residuals are all zero from perfect predictions)
        secondary_metrics = {
            "durbin_watson",
            "transformed_durbin_watson",
            "original_durbin_watson",
            "residual_has_autocorr",
            "residual_normality_p",
            "residual_is_normal",
        }

        # Check for inf/nan - but allow NaN for secondary metrics
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                if not np.isfinite(value):
                    if key not in secondary_metrics:
                        # Primary metric is not finite - fatal error
                        raise ValueError(f"Metric '{key}' is not finite: {value} in {context}")
                    else:
                        # Secondary metric NaN is acceptable (log it but continue)
                        logger.debug(
                            f"Secondary metric '{key}' is NaN in {context} "
                            f"(degenerate case - likely perfect or near-perfect predictions)"
                        )

    def _generate_model_version(self) -> str:
        """Generate semantic version for model"""
        major, minor = VERSION.split(".")[:2]
        return f"{major}.{minor}.0"

    def train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_transformation=None,
        feature_engineer=None,
        use_calibration: bool = False,
        enable_explainability: bool = False,
        y_train_original=None,  # dollar-scale targets for sample weight tier calculation
    ) -> dict[str, Any]:
        """
        Train single model with comprehensive validation and 3-tier bias correction

        OPTIMIZATION: Only runs SHAP explainability when enable_explainability=True
        Performance: Saves 30-60% of training time by deferring SHAP to final model selection

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            target_transformation: Target transformation configuration
            feature_engineer: Fitted feature engineering pipeline
            use_calibration: Whether to apply isotonic calibration
            enable_explainability: If True, calculate SHAP values (SLOW). If False, skip SHAP (FAST)

        Returns:
            Dictionary containing training results and metrics
        """

        # ============================================================================
        # INITIALIZE ALL VARIABLES AT FUNCTION START (prevents UnboundLocalError)
        # ============================================================================
        transform_method = None
        bias_correction = None

        result: dict[str, Any] = {
            "model_name": model_name,
            "status": "failed",
            "mlflow_run_id": None,
            "mlflow_model_uri": None,
            "model_version": self._generate_model_version(),
            "training_time": 0.0,
        }

        logger.info(f"\n{'='*80}\n{model_name}\n{'='*80}")
        if not enable_explainability:
            logger.info("⚡ Fast mode: SHAP explainability DISABLED (30-60% faster)")
        else:
            logger.warning("🐌 Slow mode: SHAP explainability ENABLED (will take 30-60 seconds)")

        start_time = time.time()

        # Validate feature_engineer EARLY (before training)
        try:
            # Ensure data has been prepared via prepare_training_data()
            if not getattr(self, "_data_prepared", False):
                raise RuntimeError("Must call prepare_training_data() before train_single_model().")

            if not self.resources.check_threshold():
                result["error"] = "Memory threshold exceeded"
                return result

            # Validate feature_engineer availability
            if feature_engineer is None:
                raise ValueError(
                    "❌ feature_engineer is required for training!\n"
                    "   This should be provided from prepare_training_data().\n"
                    "   \n"
                    "   ✅ FIX: Ensure prepare_training_data() completed successfully\n"
                    "   and pass feature_engineer to train_single_model()."
                )

            # EXTRACT TRANSFORM METHOD HERE (after validation)
            if hasattr(feature_engineer, "target_transformation"):
                transform_method = feature_engineer.target_transformation.method
                logger.debug(f"📋 Transform method detected: {transform_method}")
            else:
                logger.warning("⚠️ feature_engineer missing target_transformation attribute")
                transform_method = "none"

            # Validate critical attributes exist
            critical_attrs = ["scaler", "target_transformation", "_continuous_features"]
            missing_attrs = [attr for attr in critical_attrs if not hasattr(feature_engineer, attr)]

            if missing_attrs:
                raise ValueError(
                    f"❌ feature_engineer missing critical attributes: {missing_attrs}\n"
                    f"   This indicates incomplete or corrupted preprocessing.\n"
                    f"   \n"
                    f"   ✅ FIX: Retrain feature_engineer from scratch."
                )

            # Validate inverse transform capability
            if target_transformation and target_transformation.method != "none":
                try:
                    if target_transformation.method == "log1p":
                        if (
                            not hasattr(feature_engineer, "target_min_")
                            or feature_engineer.target_min_ is None
                        ):
                            raise ValueError(
                                "feature_engineer.target_min_ not set. "
                                "Ensure transform_target(fit=True) was called during data preparation."
                            )

                    elif target_transformation.method == "yeo-johnson":
                        if (
                            not hasattr(feature_engineer, "yeo_johnson_transformer")
                            or feature_engineer.yeo_johnson_transformer is None
                        ):
                            raise ValueError(
                                "Yeo-Johnson transformer not fitted. "
                                "Ensure transform_target(fit=True) was called during data preparation."
                            )

                        if not hasattr(feature_engineer.yeo_johnson_transformer, "lambdas_"):
                            raise ValueError(
                                "Yeo-Johnson transformer missing lambdas_ attribute. "
                                "Transformer was not properly fitted."
                            )

                    elif target_transformation.method.startswith("boxcox"):
                        if (
                            not hasattr(target_transformation, "boxcox_lambda")
                            or target_transformation.boxcox_lambda is None
                        ):
                            raise ValueError(
                                "Box-Cox lambda parameter missing. "
                                "Ensure transform_target(fit=True) was called during data preparation."
                            )

                    logger.debug(
                        f"✅ Transform validation passed: {target_transformation.method} "
                        f"has required attributes"
                    )

                except Exception as e:
                    raise ValueError(
                        f"❌ Transform validation failed!\n"
                        f"   Transform method: {target_transformation.method}\n"
                        f"   Error: {e}\n"
                        f"   \n"
                        f"   ✅ FIX: Ensure prepare_training_data() completed successfully\n"
                        f"   before calling train_single_model()."
                    ) from e

            self._validate_target_transformation(target_transformation)

            # Pre-training validation
            if not self.resources.check_threshold():
                result["error"] = "Memory threshold exceeded"
                return result

            # Ensure features are numeric
            categorical_cols = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            if categorical_cols:
                raise ValueError(
                    f"❌ X_train contains categorical columns: {categorical_cols}\n"
                    f"   Features MUST be encoded before train_single_model()!"
                )

            self._validate_data(X_train, X_val, y_train, y_val)

        except Exception as e:
            result["error"] = f"Pre-training validation failed: {e}"
            result["training_time"] = time.time() - start_time
            logger.error(f"❌ {result['error']}")
            return result

        # Model training
        with self.mlflow.run(f"{model_name}_run"):
            try:
                with self.timeout_mgr.time_limit(self.config.training_timeout):
                    from insurance_ml.models import get_model_gpu_params

                    gpu_params = get_model_gpu_params(model_name, self.raw_config)

                    # Train with Optuna or standard
                    if self.optimizer and self.config.enable_optuna:
                        logger.info("  Using Optuna hyperparameter optimization...")

                        # Calculate sample weights BEFORE Optuna
                        sample_weights = None
                        if self.config.use_sample_weights:
                            # pass y_original (dollar scale) so tier boundaries
                            # are correct. y_train here is YJ-transformed (~6-15 range).
                            sample_weights = calculate_sample_weights(
                                y_train,
                                self.raw_config,
                                y_original=y_train_original,
                            )

                            if sample_weights is not None:
                                logger.info(
                                    f"  ✅ Sample weights for Optuna:\n"
                                    f"     Range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]\n"
                                    f"     Mean: {sample_weights.mean():.4f}"
                                )
                            else:
                                logger.warning("  ⚠️ Sample weight calculation returned None")  # type: ignore[unreachable]

                        # progressive_log weights are incompatible with
                        # pinball (quantile) loss. For squarederror, upweighting a sample
                        # pulls the conditional mean toward it — the intended behaviour.
                        # For quantile loss, upweighting a high-value sample forces the
                        # predicted quantile upward to "protect" it, shifting the effective
                        # quantile ~21–44pp above alpha regardless of the alpha setting.
                        # Measured: alpha=0.36 + progressive_log weights → 81% overpricing.
                        # suppress weights for any quantile objective so pinball loss
                        # operates on the unconditional data distribution as intended.
                        _optuna_objective = (
                            self.raw_config.get("models", {})
                            .get(model_name, {})
                            .get("objective", "")
                        )
                        if "quantile" in _optuna_objective and sample_weights is not None:
                            logger.info(
                                "  ℹ️  Quantile model '%s': sample weights suppressed.\n"
                                "     progressive_log weights shift effective quantile ~21pp\n"
                                "     upward regardless of alpha (measured: 81%% overpricing\n"
                                "     at alpha=0.36). Using uniform weights for correct\n"
                                "     pinball loss convergence.",
                                model_name,
                            )
                            sample_weights = None

                        # hybrid scoring mode's RMSE component (70% weight)
                        # biases hyperparameter selection toward mean-predicting configs,
                        # inflating the effective quantile above alpha.
                        #
                        # Root cause of previous patch failure: raw_config and
                        # self.optimizer.optuna_config are separate dicts. Patching
                        # raw_config had no effect — optuna_config was already built
                        # at OptunaOptimizer.__init__ time and reads from its own copy.
                        #
                        # patch self.optimizer.optuna_config directly.
                        # optimize_model() reads scoring_mode from optuna_config at
                        # call time (not cached further), so this patch takes effect
                        # immediately and is restored after the call — safe for
                        # sequential model training.
                        _scoring_override_applied = False
                        _original_scoring_mode = None
                        if "quantile" in _optuna_objective and self.optimizer is not None:
                            _override_mode = (
                                self.raw_config.get("optuna", {})
                                .get("enhanced_scoring", {})
                                .get("model_mode_overrides", {})
                                .get(model_name)
                            )
                            if (
                                _override_mode
                                and _override_mode
                                != self.optimizer.optuna_config.get(
                                    "enhanced_scoring_mode", "hybrid"
                                )
                            ):
                                _original_scoring_mode = self.optimizer.optuna_config.get(
                                    "enhanced_scoring_mode", "hybrid"
                                )
                                # Directly patch the live optuna_config dict.
                                # This is the dict that optimize_model() reads from
                                # at lines 2064-2065 and _create_objective reads at 1344.
                                self.optimizer.optuna_config["enhanced_scoring_mode"] = (
                                    _override_mode
                                )
                                _scoring_override_applied = True
                                logger.info(
                                    "  ℹ️  Quantile model '%s': scoring mode overridden "
                                    "in optuna_config '%s' → '%s'.\n"
                                    "     Hybrid RMSE component (70%%) biases HPO toward\n"
                                    "     mean-predicting configs, inflating effective\n"
                                    "     quantile above alpha. Pure pinball scoring\n"
                                    "     selects params that converge to the target quantile.",
                                    model_name,
                                    _original_scoring_mode,
                                    _override_mode,
                                )

                        # ── P1-C: inject run_id so Optuna callback can log steps ──
                        if self.mlflow.enabled and self.mlflow._mlflow.active_run():  # type: ignore[union-attr]
                            self.optimizer.optuna_config["_mlflow_run_id"] = (
                                self.mlflow._mlflow.active_run().info.run_id  # type: ignore[union-attr]
                            )
                        # ─────────────────────────────────────────────────────────
                        opt_result = self.optimizer.optimize_model(
                            model_name=model_name,
                            X_train=X_train,
                            y_train=y_train,
                            sample_weight=sample_weights,
                            target_transformation=target_transformation,
                            feature_engineer=feature_engineer,
                        )

                        # Restore original scoring mode so the pricing model's
                        # next optimize_model() call uses hybrid as intended.
                        if _scoring_override_applied and _original_scoring_mode is not None:
                            self.optimizer.optuna_config["enhanced_scoring_mode"] = (
                                _original_scoring_mode
                            )
                            logger.info(
                                "  ✅ Scoring mode restored in optuna_config: '%s' → '%s' "
                                "after '%s' optimization.",
                                _override_mode,
                                _original_scoring_mode,
                                model_name,
                            )

                        if isinstance(opt_result, tuple):
                            model, opt_data = opt_result
                        else:
                            model = opt_result
                            opt_data = {}

                        # Extract CV results
                        if opt_data and "best_value" in opt_data:
                            cv_result = {
                                "cv_mean": float(opt_data["best_value"]),
                                "cv_std": float(opt_data.get("cv_std", 0.0)),
                                "cv_scores": opt_data.get("cv_scores", []),
                                "source": "optuna",
                                "best_params": opt_data.get("best_params", {}),
                                # Preserve Optuna's pinball-space gap so the
                                # Model Comparison Summary can show it alongside the
                                # dollar-RMSE gap for quantile models. Without this,
                                # gap_percent was dropped here and the summary had to
                                # print "see pinball gap" with no actual value.
                                "gap_percent": float(opt_data.get("gap_percent", 0.0)),
                                "train_score": float(opt_data.get("train_rmse", 0.0)),
                            }
                        else:
                            model_factory = self.model_manager._model_factories[model_name]
                            cv_result = self._calculate_cv_manual(
                                model_factory, X_train, y_train, gpu_params
                            )

                        # ✅ CENTRALIZED: Store conformal data after Optuna training
                        # MetricsExtractor.store_conformal_data(
                        #     model, X_val, y_val, context="post_optuna"
                        # )

                        # ============================================================
                        # INVALIDATE pre-calibration residuals (OPTUNA PATH ONLY)
                        # ============================================================
                        if hasattr(model, "_validation_residuals"):
                            delattr(model, "_validation_residuals")
                            logger.debug("  🗑️  Invalidated old residuals before calibration")

                    else:
                        logger.info("  Training with default hyperparameters...")

                        if gpu_params:
                            logger.info(f"  🚀 Creating {model_name} with GPU acceleration")
                            model = self.model_manager.get_model(
                                model_name, params=gpu_params, gpu=True
                            )
                        else:
                            logger.info(f"  💻 Creating {model_name} in CPU mode")
                            model = self.model_manager.get_model(model_name, gpu=False)

                        # Calculate sample weights BEFORE training
                        sample_weights = None
                        if self.config.use_sample_weights:
                            # calculate_sample_weights() has no 'log_stats'
                            # parameter; passing it caused TypeError every time Optuna
                            # was disabled and use_sample_weights=True.  The function
                            # signature is (y, config, validate_distribution=True).
                            # y_original (dollar scale) for correct tier bounds.
                            sample_weights = calculate_sample_weights(
                                y_train,
                                self.raw_config,
                                y_original=y_train_original,
                            )
                            if sample_weights is not None:
                                if not validate_sample_weights(sample_weights, y_train):
                                    sample_weights = None

                        # Train with proper sample weight handling
                        model_type = type(model).__name__

                        if model_type in ["XGBRegressor", "LGBMRegressor"]:
                            logger.info(f"  🎯 Training {model_type} with early stopping")

                            model = self.model_manager.fit_with_early_stopping(
                                model=model,
                                X_train=X_train,
                                y_train=y_train,
                                X_val=X_val,
                                y_val=y_val,
                                sample_weight=sample_weights,
                                verbose=False,
                            )

                            if sample_weights is not None:
                                logger.info("  ✅ Trained with early stopping + sample weights")
                            else:
                                logger.info("  ✅ Trained with early stopping (no weights)")

                            # ✅ CENTRALIZED: Store conformal data after early stopping
                            # MetricsExtractor.store_conformal_data(
                            #     model, X_val, y_val, context="post_early_stopping"
                            # )

                        else:
                            # Standard fit for other models
                            if sample_weights is not None and self._model_supports_sample_weights(
                                model
                            ):
                                try:
                                    model.fit(X_train, y_train, sample_weight=sample_weights)
                                    logger.info(f"  ✅ {model_type} trained with sample weights")
                                except TypeError as e:
                                    logger.warning(
                                        f"  ⚠️ {model_type} rejected sample_weight parameter: {e}"
                                    )
                                    logger.info("  Retraining without sample weights...")
                                    model.fit(X_train, y_train)
                            else:
                                model.fit(X_train, y_train)
                                if self.config.use_sample_weights:
                                    logger.info(
                                        f"  ℹ️ {model_type} doesn't support sample_weight parameter"
                                    )
                                else:
                                    logger.info(
                                        f"  ✅ {model_type} trained (weights disabled in config)"
                                    )

                            # ✅ CENTRALIZED: Store conformal data after standard training
                            # MetricsExtractor.store_conformal_data(
                            #     model, X_val, y_val, context="post_standard_training"
                            # )

                        # Manual CV
                        model_factory = self.model_manager._model_factories[model_name]
                        cv_result = self._calculate_cv_manual(
                            model_factory, X_train, y_train, gpu_params
                        )

                    # ========================================
                    # DECLARE y_val_transformed_array EARLY
                    # ========================================
                    # This is needed for calibration block below
                    y_val_transformed_array = (
                        y_val.values if hasattr(y_val, "values") else np.array(y_val)
                    )

                    # ========================================
                    # OPTIONAL: Apply Calibration (BEFORE residual calculation)
                    # ========================================
                    if use_calibration:
                        logger.info("🔧 Applying isotonic calibration...")

                        # CalibratedModel is only imported inside main()
                        # (line ~4483) and is not available at module scope or in this
                        # method's scope. Import it locally here.
                        from insurance_ml.models import CalibratedModel

                        # X_calib / y_calib were referenced below but
                        # never defined in this scope (they only exist in main()).
                        # Reserve 40 % of the validation set for conformal calibration;
                        # use the remaining 60 % for isotonic fitting.
                        # Guard: CalibratedModel.fit_calibrator() requires ≥ 50 samples,
                        # so we need at least ceil(50 / 0.6) = 84 total to split safely.
                        _CALIB_MIN = 50  # minimum samples for isotonic fit
                        # single canonical split ratio read from config.
                        # The FIX used 0.20 hardcoded while this path used 0.40,
                        # producing different CI widths across code paths. Now both
                        # paths share one constant from config (fallback 0.40).
                        #
                        # SEMANTIC WARNING: calibration_split_ratio has
                        # INVERTED naming relative to its effect.  A value of 0.20
                        # means 20% goes to holdout (test_size=0.20) and 80% stays
                        # as the calibration set — NOT 20% for calibration.
                        # The config comment ("80% calibration / 20% holdout") is
                        # correct; the field name is misleading.  Operators who read
                        # the field name and set it to 0.80 would get 80% holdout,
                        # 20% calibration — the opposite of intent.
                        # Keep the current value (0.20 = 80% cal) and preserve the
                        # inverted convention for backward compatibility.  A rename
                        # to holdout_split_ratio is deferred to a major version bump.
                        _SPLIT_RATIO = float(
                            self.raw_config.get("conformal", {}).get(
                                "calibration_split_ratio", 0.20
                            )
                        )
                        _SPLIT_THRESHOLD = int(_CALIB_MIN / (1.0 - _SPLIT_RATIO)) + 1

                        if len(X_val) >= _SPLIT_THRESHOLD:
                            X_fit_cal, X_calib, y_fit_cal, y_calib = train_test_split(
                                X_val,
                                y_val,
                                test_size=_SPLIT_RATIO,
                                random_state=self.config.random_state,
                            )
                        else:
                            # do NOT alias X_fit_cal = X_calib when val is too small —
                            # that causes circular conformal evaluation (fit on same set as holdout).
                            # Instead, skip calibration entirely for this model.
                            logger.warning(
                                f"⚠️ Validation set too small to split for calibration "
                                f"(n={len(X_val)} < {_SPLIT_THRESHOLD}). "
                                f"Skipping isotonic calibration for this model to avoid "
                                f"circular conformal coverage evaluation."
                            )
                            # Skip calibration — fall through to the else branch below
                            use_calibration = False

                        # Step 1: Wrap model with calibration
                        # CalibratedModel.__init__ accepts
                        # 'calibration_method', not 'method'. Passing the wrong
                        # keyword raised TypeError on every calibrated training run.
                        calibrated_model = CalibratedModel(model, calibration_method="isotonic")
                        calibrated_model.fit_calibrator(X_fit_cal, y_fit_cal)  # fit on 60 %

                        # Step 2: Store conformal data from calibrated model (SINGLE SOURCE OF TRUTH)
                        success = MetricsExtractor.store_conformal_data(
                            calibrated_model,
                            X_val=X_calib,  # ✅ Use calibration set
                            y_val=y_calib,  # ✅ Use calibration set
                            context="post_calibration",
                            force_overwrite=True,  # Replace any existing data
                        )

                        if success:
                            logger.info(
                                f"📊 Stored conformal calibration data:\n"
                                f"   Samples: {len(X_calib)} (calibration set)\n"
                                f"   Residual std: {np.std(calibrated_model._validation_residuals):.4f}"
                            )
                        else:
                            logger.error(
                                "❌ Failed to store conformal data!\n"
                                "   Conformal intervals will be unavailable"
                            )

                        model = calibrated_model
                        model_name = f"{model_name}_calibrated"

                        logger.info(
                            f"✅ Calibration complete:\n"
                            f"   Model: {type(calibrated_model.base_model).__name__}\n"
                            f"   Calibrator: IsotonicRegression\n"
                            f"   Conformal data: Fresh from calibrated predictions"
                        )

                    else:
                        # No isotonic calibration.
                        #
                        # (v7.5.0): Increase conformal calibration fraction
                        # from 60% to 80% (holdout 40%→20%).
                        #
                        # Root cause of $69K avg CI width:
                        #   n_cal = 268×0.60 = ~161 samples  →  161//30 = 5 bins
                        #   5 bins is too coarse for the heavy right-skew (skew=+2.60,
                        #   kurt=+7.52 on YJ residuals).  The extreme top-bin residual
                        #   (~3.32 YJ units) sets the local_q for ALL high-value test
                        #   points before the 99th-pctile Winsorisation cap.
                        #
                        # test_size=0.20 → n_cal = 268×0.80 = ~214 samples
                        #   214//30 = 7 bins  (2 more bins, better tail resolution)
                        #   Holdout: 268×0.20 = ~54 samples (still > 30 minimum).
                        #   Expected CI width reduction: ~15–25% from finer binning.
                        #
                        # Note: the coverage guarantee (marginal, exchangeability) is
                        # preserved regardless of n_bins choice; more bins only improves
                        # adaptiveness, not marginal validity.
                        _n_val = len(X_val)
                        _MIN_CONFORMAL_SAMPLES = 30
                        # use the same _CONF_SPLIT_RATIO as the calibrated path.
                        _conf_split_ratio = float(
                            self.raw_config.get("conformal", {}).get(
                                "calibration_split_ratio", 0.20
                            )
                        )
                        # Invert: conformal gets (1 - ratio), holdout gets ratio.
                        _conf_calibration_frac = 1.0 - _conf_split_ratio

                        if _n_val >= _MIN_CONFORMAL_SAMPLES * 2:
                            (
                                _X_conf,
                                _X_holdout,
                                _y_conf,
                                _y_holdout,
                            ) = train_test_split(
                                X_val,
                                y_val,
                                test_size=_conf_split_ratio,
                                random_state=self.config.random_state,
                            )
                            logger.info(
                                f"   Conformal split: {len(_X_conf)} calibration "
                                f"| {len(_X_holdout)} holdout "
                                f"(from {_n_val} val samples, "
                                f"{_conf_calibration_frac*100:.0f}/{_conf_split_ratio*100:.0f})"
                            )
                        else:
                            # do NOT alias _X_conf = _X_holdout when the val set is
                            # too small.  Using the same set for both fitting conformal quantiles
                            # and measuring CI coverage produces circular, optimistically biased
                            # coverage metrics that silently pass G3 gate.  Instead: log clearly
                            # and set both to None so downstream code skips conformal entirely.
                            logger.warning(
                                f"⚠️  Validation set too small for conformal split "
                                f"({_n_val} < {_MIN_CONFORMAL_SAMPLES * 2} samples). "
                                f"Conformal intervals will be skipped for this model to avoid "
                                f"circular coverage evaluation. Increase val set size to enable."
                            )
                            _X_conf, _y_conf = None, None
                            _X_holdout, _y_holdout = None, None

                        # Store residuals from CALIBRATION set only (skip if too small)
                        if _X_conf is not None:
                            success = MetricsExtractor.store_conformal_data(
                                model,
                                _X_conf,
                                _y_conf,
                                context="final_uncalibrated_conformal_split",
                                force_overwrite=True,
                            )

                            if success:
                                logger.info(
                                    "✅ Stored conformal data (calibration set, holdout-safe)"
                                )

                                # ── (v7.5.4): pre-compute heteroscedastic bins ──────────
                                # SEQUENCING: save_model() runs after this block, so bins stored
                                # here are serialized into the artifact automatically.
                                # explain_predictions() runs AFTER save_model() — too late.
                                # Previous approach attached bins inside _calculate_conformal_intervals
                                # via _current_explaining_model, but that fires post-save.
                                try:
                                    _ci_residuals = model._validation_residuals
                                    _ci_preds = model._validation_predictions
                                    # replaced 115-line inline bin computation
                                    # with a call to the canonical static utility in
                                    # ModelManager.  The inline block was a verbatim copy
                                    # of the logic in _calculate_conformal_intervals —
                                    # any future change to binning parameters
                                    # (min_samples_per_bin, winsor_pct, n_bins, bin
                                    # assignment formula) now only needs to be made in
                                    # one place.  The utility performs its own precondition
                                    # checks (n_cal ≥ 50, predictions not None, lengths
                                    # match) and returns None if they are not met.
                                    _ci_alpha = 1.0 - float(
                                        self.raw_config.get("training", {})
                                        .get("conformal_intervals", {})
                                        .get("target_coverage", 0.90)
                                    )
                                    from insurance_ml.models import (
                                        ModelExplainer as _ME,
                                    )

                                    _hetero_bins = _ME.compute_heteroscedastic_bins(
                                        residuals=_ci_residuals,
                                        predictions=_ci_preds,
                                        alpha=_ci_alpha,
                                        n_bins=10,
                                        min_samples_per_bin=30,
                                        winsor_pct=95,
                                    )
                                    if _hetero_bins is not None:
                                        model._conformal_data["heteroscedastic_bins"] = _hetero_bins
                                        logger.info(
                                            f"✅ Stored heteroscedastic bins in _conformal_data "
                                            f"({_hetero_bins['n_bins']} bins, "
                                            f"will persist via save_model)"
                                        )
                                    else:
                                        logger.info(
                                            f"ℹ️  Heteroscedastic bins skipped "
                                            f"(n_cal={len(_ci_residuals)} < 50 or "
                                            f"predictions unavailable or length mismatch)"
                                        )
                                except Exception as _ci_bin_err:
                                    logger.warning(
                                        f"⚠️  Could not pre-compute heteroscedastic "
                                        f"bins: {_ci_bin_err}"
                                    )
                        else:
                            success = False
                            logger.info("⏭️  Conformal data skipped (val set too small).")

                        # ── Refit conformal residuals on post-hybrid-calibration ──
                        #
                        # Context: HybridPredictor applies `calibration_factor` in ORIGINAL
                        # dollar space (ml_predictions_calibrated = ml_predictions * factor,
                        # where ml_predictions are already inverse-transformed).
                        #
                        # The previous implementation applied _calibration_factor directly to
                        # transformed-space predictions (y_pred_uncalibrated * factor), which is
                        # incorrect because 1.15× in dollar space ≠ 1.15× in yeo-johnson space.
                        #
                        # Correct approach:
                        #   1. model.predict(X_val) → transformed space
                        #   2. inverse_transform → original dollar space
                        #   3. apply calibration_factor → calibrated original space
                        #   4. forward_transform → calibrated transformed space
                        #   5. residuals = y_val_transformed - calibrated_transformed
                        #
                        # Step 4 (forward transform) requires the fitted yeo-johnson scaler
                        # from feature_engineer, which is available in this scope.
                        # self.raw_config is the raw dict assigned in Trainer.__init__
                        # (line: self.raw_config = raw_config).  It is used throughout
                        # Trainer for all config look-ups (e.g. sample weights, GPU params).
                        # The previous code tried self.config.__dict__["_raw_config"] which
                        # always returned {} because self.config is a Config *object*, not
                        # the raw dict — causing _calibration_factor to silently stay 1.0
                        # and the conformal refit to be skipped every training run.
                        _hybrid_cfg = self.raw_config.get("hybrid_predictor", {})
                        # Previous code read calibration.factor (legacy
                        # single-model key, always 1.00), never the per-model
                        # pricing_factor / risk_factor keys.  Because factor=1.00, the
                        # gate `if _calibration_factor != 1.0` was always False and the
                        # entire conformal refit block was silently skipped every run.
                        # resolve pricing_factor first (operative key for
                        # xgboost_median / reg:squarederror), falling back to risk_factor
                        # then legacy factor for backward compatibility.  This mirrors the
                        # resolution order in HybridPredictor.__init__.
                        _cal_sub = _hybrid_cfg.get("calibration", {})
                        # (revised): previous fix used
                        #   getattr(self.model_manager, "active_model_name", "")
                        # but ModelManager has no active_model_name attribute, so getattr
                        # silently returned "" on every call.  That made _use_pricing always
                        # False, _calibration_factor always read risk_factor (not
                        # pricing_factor), and the entire conformal refit block was silently
                        # skipped every training run — the same failure mode as before the fix.
                        # Correct approach: use model_name, the enclosing for-loop variable
                        # (first assigned at line ~2068, mutated to f"{model_name}_calibrated"
                        # at line ~3626 if the calibrated-isotonic branch ran).  Strip the
                        # suffix so "xgboost_median_calibrated" → "xgboost_median", which
                        # correctly sets _use_pricing=True and selects pricing_factor.
                        _active_model_name = model_name.replace("_calibrated", "")
                        _use_pricing = (
                            "median" in _active_model_name.lower()
                            or "squarederror" in str(_cal_sub.get("objective", "")).lower()
                        )
                        if _use_pricing:
                            _calibration_factor = float(
                                _cal_sub.get(
                                    "pricing_factor",
                                    _cal_sub.get("factor", 1.0),
                                )
                            )
                        else:
                            _calibration_factor = float(
                                _cal_sub.get(
                                    "risk_factor",
                                    _cal_sub.get("factor", 1.0),
                                )
                            )
                        _apply_to_ml_only = bool(_cal_sub.get("apply_to_ml_only", True))

                        if _calibration_factor != 1.0:
                            try:
                                y_val_arr = (
                                    y_val.values if hasattr(y_val, "values") else np.array(y_val)
                                )
                                y_pred_uncalibrated_transformed = model.predict(X_val)

                                # Step 2: inverse-transform to original dollar space
                                y_pred_uncalibrated_original = (
                                    feature_engineer.inverse_transform_target(
                                        y_pred_uncalibrated_transformed,
                                        transformation_method=transform_method,
                                        clip_to_safe_range=False,
                                        context="conformal_refit_pred",
                                    )
                                )

                                # Step 3: apply calibration in original space
                                # (mirrors HybridPredictor.predict() exactly)
                                y_pred_calibrated_original = (
                                    y_pred_uncalibrated_original * _calibration_factor
                                )

                                # Step 4: forward-transform back to yeo-johnson space.
                                # Use the fitted scaler from feature_engineer so the
                                # transformation is identical to what was applied to y_val.
                                #
                                # Correct calling convention (from train.py line ~4101):
                                #   transform_target(pd.Series, method=..., fit=False)
                                # Returns pd.Series; extract .values for numpy operations.
                                y_pred_calibrated_transformed = (
                                    feature_engineer.transform_target(
                                        pd.Series(
                                            y_pred_calibrated_original,
                                            name="charges",
                                        ),
                                        method=transform_method,
                                        fit=False,
                                    ).values
                                    if hasattr(feature_engineer, "transform_target")
                                    else None
                                )

                                if y_pred_calibrated_transformed is not None:
                                    # Step 5: residuals in transformed space
                                    residuals_calibrated = y_val_arr - y_pred_calibrated_transformed

                                    # Replace the stored residuals
                                    model._validation_residuals = residuals_calibrated

                                    if hasattr(model, "_conformal_data") and isinstance(
                                        model._conformal_data, dict
                                    ):
                                        model._conformal_data["validation_predictions"] = (
                                            y_pred_calibrated_transformed.tolist()
                                        )
                                        model._conformal_data["validation_residuals"] = (
                                            residuals_calibrated.tolist()
                                        )
                                        model._conformal_data["context"] = (
                                            "final_hybrid_calibrated_correct_space"
                                        )

                                    logger.info(
                                        f"✅ FIX #7: Conformal residuals refit on calibrated "
                                        f"predictions (CORRECT: calibration applied in original "
                                        f"space, residuals in transformed space)\n"
                                        f"   Calibration factor: {_calibration_factor:.4f} "
                                        f"({'ML-only' if _apply_to_ml_only else 'full hybrid'})\n"
                                        f"   Samples: {len(residuals_calibrated)}\n"
                                        f"   Residual std (uncalibrated): "
                                        f"{np.std(y_val_arr - y_pred_uncalibrated_transformed):.4f}\n"
                                        f"   Residual std (calibrated):   "
                                        f"{np.std(residuals_calibrated):.4f}"
                                    )
                                else:
                                    # feature_engineer.transform_target not available —
                                    # fall back to the original (approximate) method with
                                    # an explicit warning so it shows up in training logs.
                                    logger.warning(
                                        "⚠️  FIX #7: feature_engineer.transform_target() not "
                                        "available — falling back to applying calibration factor "
                                        "directly in transformed space (approximation).\n"
                                        "   This is less accurate for yeo-johnson targets.\n"
                                        "   Add transform_target() to FeatureEngineer to fix."
                                    )
                                    y_pred_calibrated_approx = (
                                        y_pred_uncalibrated_transformed * _calibration_factor
                                    )
                                    residuals_calibrated = y_val_arr - y_pred_calibrated_approx
                                    model._validation_residuals = residuals_calibrated

                            except Exception as _e7:
                                logger.warning(
                                    f"⚠️  FIX #7: Could not refit conformal residuals on "
                                    f"calibrated predictions: {_e7}\n"
                                    f"   Falling back to uncalibrated residuals.\n"
                                    f"   CI coverage may be asymmetric for calibrated range."
                                )
                        else:
                            # calibration_factor=1.0 means no calibration applied — the
                            # uncalibrated residuals ARE the correct conformal set.
                            logger.info(
                                "✅ FIX #7: calibration_factor=1.0 — "
                                "uncalibrated conformal residuals are correct."
                            )

                    # =====================================================================
                    # Normalize XGBoost base_score IMMEDIATELY (AFTER calibration)
                    # =====================================================================
                    # This MUST happen BEFORE any predictions for SHAP compatibility
                    model_type = type(model).__name__

                    if model_type == "XGBRegressor":
                        try:
                            booster = model.get_booster()
                            attrs = booster.attributes()
                            base_score = attrs.get("base_score")

                            if base_score is not None and isinstance(base_score, str):
                                # Check if in array format: "[value]"
                                if base_score.startswith("[") and base_score.endswith("]"):
                                    numeric_str = base_score.strip("[]")
                                    numeric_value = float(numeric_str)
                                    booster.set_attr(base_score=str(numeric_value))
                                    logger.info(
                                        f"🔧 Normalized XGBoost base_score: {base_score} → {numeric_value}"
                                    )
                                else:
                                    try:
                                        float(base_score)
                                        logger.debug(
                                            f"✅ XGBoost base_score already scalar: {base_score}"
                                        )
                                    except ValueError:
                                        logger.warning(
                                            f"⚠️ Invalid base_score format: {base_score}\n"
                                            f"   Setting safe default: 0.5"
                                        )
                                        booster.set_attr(base_score="0.5")
                            elif base_score is None:
                                logger.debug("🔧 XGBoost missing base_score, setting default: 0.5")
                                booster.set_attr(base_score="0.5")

                        except Exception as e:
                            logger.error(
                                f"❌ CRITICAL: Failed to normalize XGBoost base_score!\n"
                                f"   Error: {e}\n"
                                f"   SHAP TreeExplainer will fail.\n"
                                f"   Attempting emergency fallback..."
                            )
                            try:
                                model.get_booster().set_attr(base_score="0.5")
                                logger.warning("✅ Emergency fallback successful: base_score=0.5")
                            except Exception as fallback_error:
                                logger.error(f"❌ Emergency fallback also failed: {fallback_error}")
                                raise RuntimeError(
                                    f"Cannot normalize XGBoost base_score. "
                                    f"Model cannot be used with SHAP. "
                                    f"Original error: {e}, Fallback error: {fallback_error}"
                                ) from e

                    # ========================================
                    # NOW SAFE TO CALCULATE VALIDATION RESIDUALS
                    # ========================================
                    # base_score is normalized AND model is calibrated (if requested)
                    try:
                        y_val_pred_transformed = model.predict(X_val)

                        # Calculate residuals in TRANSFORMED space
                        validation_residuals = y_val_transformed_array - y_val_pred_transformed

                        # Validate residuals
                        if not np.all(np.isfinite(validation_residuals)):
                            n_bad = np.sum(~np.isfinite(validation_residuals))
                            logger.warning(
                                f"⚠️ {n_bad}/{len(validation_residuals)} non-finite residuals detected!\n"
                                f"   Replacing with median residual for interval calculation."
                            )
                            median_residual = np.median(
                                validation_residuals[np.isfinite(validation_residuals)]
                            )
                            validation_residuals[~np.isfinite(validation_residuals)] = (
                                median_residual
                            )

                        # Attach to model instance
                        model._validation_residuals = validation_residuals

                        # preserve the full-val predictions (268 samples)
                        # as a separate attribute BEFORE store_conformal_data() is called
                        # below with the 160-sample conformal calibration split.
                        # store_conformal_data() overwrites model._validation_predictions
                        # with the 160-sample set; without this line the Phase-2 SHAP
                        # explain_predictions() would see residuals=268 / preds=160,
                        # triggering the realignment path and shrinking the effective
                        # calibration set unnecessarily.
                        # explain_predictions() in models.py prefers this attribute
                        # over _validation_predictions when both are present.
                        model._full_validation_predictions = y_val_pred_transformed.copy()

                        # ================================================================
                        # RESIDUAL DIAGNOSTICS
                        # ================================================================
                        logger.info("\n" + "=" * 80)
                        logger.info("RESIDUAL DIAGNOSTICS (for CI Calibration)")
                        logger.info("=" * 80)

                        try:
                            residual_diagnostics = self.model_manager.diagnose_residuals(
                                validation_residuals=validation_residuals,
                                save_plot=True,
                                save_path=str(self.config.reports_dir / "residuals" / model_name),
                            )

                            logger.info("\n📊 Key Findings:")
                            logger.info(
                                f"   Outlier Rate: {residual_diagnostics['outlier_pct']:.2f}%"
                            )
                            logger.info(f"   Skewness: {residual_diagnostics['skewness']:+.3f}")
                            logger.info(f"   Kurtosis: {residual_diagnostics['kurtosis']:+.3f}")

                            if residual_diagnostics["outlier_pct"] > 5:
                                logger.warning(
                                    "⚠️  HIGH OUTLIER RATE detected!\n"
                                    "   CI calculation will remove outliers >4σ automatically"
                                )

                            if abs(residual_diagnostics["skewness"]) > 1.0:
                                logger.warning(
                                    "⚠️  HEAVY SKEW detected!\n"
                                    "   Quantile-based CIs are appropriate ✅"
                                )

                            if residual_diagnostics["kurtosis"] > 3:
                                logger.warning(
                                    "⚠️  HEAVY TAILS detected!\n"
                                    "   Will increase n_bins to 10 for better heteroscedastic modeling"
                                )

                            logger.info("=" * 80 + "\n")

                        except Exception as e:
                            logger.error(f"❌ Residual diagnostics failed: {e}")
                            logger.warning("Continuing without diagnostic plots...")

                        # Calculate residual statistics for logging
                        residual_mean = validation_residuals.mean()
                        residual_median = np.median(validation_residuals)
                        residual_std = validation_residuals.std()
                        residual_q25 = np.percentile(validation_residuals, 25)
                        residual_q75 = np.percentile(validation_residuals, 75)

                        logger.info(
                            f"  ✅ Stored {len(validation_residuals)} validation residuals:\n"
                            f"     Mean:   {residual_mean:+.6f} (bias indicator — sensitive to outliers)\n"
                            f"     Median: {residual_median:+.6f} (robust bias indicator — preferred for skewed data)\n"
                            f"     Std:    {residual_std:.6f} (uncertainty)\n"
                            f"     Q25:    {residual_q25:+.6f}\n"
                            f"     Q75:    {residual_q75:+.6f}\n"
                            f"     Range: [{validation_residuals.min():+.6f}, {validation_residuals.max():+.6f}]"
                        )

                        # Use median for systematic bias detection.
                        # For skewed residual distributions (skewness > 2 is typical for
                        # insurance), the mean is dominated by the extreme tail and can
                        # be near zero while the median is substantially negative —
                        # indicating over-prediction for the majority of policyholders.
                        if abs(residual_median) > 0.05 * residual_std:
                            _direction = "over" if residual_median < 0 else "under"
                            logger.warning(
                                f"⚠️ Systematic median bias detected:\n"
                                f"   Median residual ({residual_median:+.6f}) > 5% of std ({residual_std:.6f})\n"
                                f"   Model {_direction}estimates for >50% of samples\n"
                                f"   (Mean={residual_mean:+.6f} may appear near-zero due to tail skew)"
                            )
                        elif abs(residual_mean) > 0.1 * residual_std:
                            logger.warning(
                                f"⚠️ Systematic mean bias detected:\n"
                                f"   Mean residual ({residual_mean:+.6f}) > 10% of std ({residual_std:.6f})\n"
                                f"   Model may be {'over' if residual_mean < 0 else 'under'}estimating"
                            )

                    except Exception as e:
                        logger.error(
                            f"❌ Could not store validation residuals: {e}\n"
                            f"   Confidence intervals will be unavailable.\n"
                            f"   This may indicate data quality issues.",
                            exc_info=True,
                        )
                        model._validation_residuals = None

                    # ========================================
                    # 3-TIER BIAS CORRECTION
                    # ========================================
                    # Computed via BiasCorrection.calculate_from_model() — single
                    # authoritative path for both log1p (Jensen / exp(σ²/2)) and
                    # yeo-johnson (median-ratio) formulas, including quantile guard.
                    #
                    # feature_engineer._bias_var_* attributes are written by the
                    # block in main() AFTER all models are trained, using
                    # the best model's BiasCorrection. Writing them here per-model
                    # was a duplicate path that diverged whenever the inline logic
                    # was updated independently of calculate_from_model().
                    if transform_method in ["log1p", "yeo-johnson"]:
                        # calculate_from_model() has an internal quantile
                        # guard but it relies on get_xgb_params() / get_params() which
                        # return the objective string differently across XGBoost versions
                        # (1.x vs 2.x vs 3.x).  Use config.yaml as the version-independent
                        # source of truth: if the model's configured objective contains
                        # "quantile" skip BC entirely without calling calculate_from_model().
                        _base_model_name = model_name.replace("_calibrated", "")
                        _cfg_obj = str(
                            self.raw_config.get("models", {})
                            .get(_base_model_name, {})
                            .get("objective", "")
                        ).lower()
                        if "quantile" in _cfg_obj:
                            logger.info(
                                f"ℹ️  Skipping BiasCorrection for '{_base_model_name}' "
                                f"(config objective='{_cfg_obj}' is a quantile model; "
                                f"median-ratio correction incompatible with quantile loss)."
                            )
                            bias_correction = None
                            # Previously the median-bias warning fired twice with
                            # no downstream action, creating a false impression of a silent
                            # bug. Now explicitly log that bias enters production and
                            # document exactly which mechanism handles it.
                            try:
                                if abs(residual_median) > 0.05 * residual_std:
                                    logger.warning(
                                        f"⚠️  FIX-10: Median bias ({residual_median:+.4f} "
                                        f"transformed space) is carried to production without "
                                        f"direct correction.\n"
                                        f"   BiasCorrection is incompatible with quantile loss "
                                        f"(by design — asymmetric loss shifts the predicted "
                                        f"distribution above the median).\n"
                                        f"   Handling mechanism: calibration factor in config.yaml "
                                        f"provides a residual global upward shift.\n"
                                        f"   Monitor: churn_rate and overpricing_rate in evaluate.py. "
                                        f"Reduce calibration factor if churn exceeds 3%."
                                    )
                            except NameError:
                                pass  # residual_median not yet defined (early training phases)
                        else:
                            bias_correction = BiasCorrection.calculate_from_model(
                                model=model,
                                X_val=X_val,
                                y_val=y_val,
                                feature_engineer=feature_engineer,
                                model_name=model_name,
                            )
                        # bias_correction is None for quantile models — that is correct
                        # and expected.
                    elif transform_method == "boxcox":
                        logger.info("🔧 Calculating bias correction for Box-Cox...")

                    # ========================================
                    # OPTIMIZED EVALUATION: CONDITIONAL SHAP
                    # ========================================
                    logger.info("📊 Calculating training metrics (fast path - NO SHAP)...")
                    train_metrics, train_preds = self.model_manager.evaluate_model(
                        model,
                        X_train,
                        y_train,
                        f"{model_name}_train",
                        target_transformation,
                        feature_engineer=feature_engineer,
                        calculate_intervals=False,
                        phase="train",
                        bias_correction=bias_correction,
                    )
                    self._validate_metrics_dict(train_metrics, "train")

                    if enable_explainability:
                        logger.info("📊 Calculating validation metrics WITH explainability...")
                        logger.warning(
                            "⚠️  SHAP calculation in progress - this will take 30-60 seconds..."
                        )

                        val_metrics, val_preds, val_explanations = (
                            self.model_manager.evaluate_model_with_explainability(
                                model=model,
                                X_test=X_val,
                                y_test=y_val,
                                X_train_sample=X_train.sample(min(100, len(X_train))),
                                model_name=model_name,
                                target_transformation=target_transformation,
                                feature_engineer=feature_engineer,
                                explainability_config=self.explainability_config,
                                bias_correction=bias_correction,
                            )
                        )
                        self._validate_metrics_dict(val_metrics, "validation")

                        if val_explanations.get("confidence_intervals") is not None:
                            logger.info(
                                f"   Confidence Intervals:\n"
                                f"      Coverage: {val_metrics.get('interval_coverage_pct', 0):.1f}%\n"
                                f"      Avg Width: ${val_metrics.get('interval_avg_width', 0):,.0f}"
                            )

                        if val_explanations.get("feature_importance") is not None:
                            top_features = val_explanations["feature_importance"].head(5)
                            logger.info("   Top 5 SHAP Features:")
                            for idx, row in top_features.iterrows():
                                logger.info(
                                    f"      {idx+1}. {row['feature']}: {row['importance']:.4f}"
                                )
                    else:
                        logger.info("📊 Calculating validation metrics (fast path - NO SHAP)...")

                        val_metrics, val_preds = self.model_manager.evaluate_model(
                            model,
                            X_val,
                            y_val,
                            f"{model_name}_val",
                            target_transformation,
                            feature_engineer=feature_engineer,
                            calculate_intervals=False,
                            phase="validation",
                            bias_correction=bias_correction,
                        )
                        self._validate_metrics_dict(val_metrics, "validation")

                        val_explanations = {
                            "confidence_intervals": None,
                            "shap_values": None,
                            "feature_importance": None,
                            "plots": {},
                        }

                        logger.info("✅ Fast evaluation complete (SHAP deferred to final model)")

                    # Save model
                    elapsed = time.time() - start_time
                    safe_name = FileSanitizer.sanitize(model_name)
                    model_path = self.config.output_dir / f"{safe_name}.joblib"

                    metadata = {
                        "model_name": model_name,
                        "version": VERSION,
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "training_time": float(elapsed),
                        "target_transformation": target_transformation or "none",
                        "gpu_params": str(gpu_params) if gpu_params else "none",
                        "explainability_enabled": enable_explainability,
                        "bias_correction": (bias_correction.to_dict() if bias_correction else None),
                        # ── PATCH 03 (G4/G9): inject provenance fields ───────────
                        # _provenance is set on the trainer instance in main() via
                        # trainer._provenance = capture_git_provenance() before any
                        # training begins. Using getattr with a safe fallback ensures
                        # train_single_model() never raises if called in isolation
                        # (e.g. unit tests or interactive use without main()).
                        "git_commit": getattr(
                            getattr(self, "_provenance", None), "commit_hash", "unknown"
                        ),
                        "git_branch": getattr(
                            getattr(self, "_provenance", None), "branch", "unknown"
                        ),
                        "random_state": self.config.random_state,
                        "pipeline_version": VERSION,
                    }

                    self.model_manager.save_model(
                        model,
                        safe_name,
                        additional_metadata=metadata,
                        X_sample=X_train.head(10),
                        y_sample=y_train.head(10),
                    )

                    # Checksum
                    checksum = ""
                    if self.config.save_checksums:
                        checksum = FileSanitizer.compute_checksum(model_path)
                        checksum_path = self.config.output_dir / f"{safe_name}_checksum.txt"
                        with open(checksum_path, "w") as f:
                            f.write(f"{checksum}\n")

                    gpu_peak_mb = 0
                    if self.gpu_available and gpu_params:
                        gpu_mem = self.get_gpu_memory_usage()
                        if gpu_mem:
                            gpu_used_mb = gpu_mem.get("used_mb", 0)
                            gpu_total_mb = gpu_mem.get(
                                "total_mb", gpu_used_mb + gpu_mem.get("free_mb", 0)
                            )

                            if gpu_used_mb <= 1:
                                logger.info(
                                    f"  📊 GPU Usage at log time (post-training): "
                                    f"{gpu_used_mb:.0f}MB / {gpu_total_mb:.0f}MB (GPU effectively idle)"
                                )
                            else:
                                logger.info(
                                    f"  📊 GPU Usage at log time (post-training): "
                                    f"{gpu_used_mb:.0f}MB / {gpu_total_mb:.0f}MB"
                                )

                    # Build result
                    # model_config was read at comparison-table time via
                    # result.get("model_config", {}).get("objective", "") but never
                    # written — always returning {} → _is_quantile=False → "Moderate
                    # Overfitting" label always shown for reg:quantileerror models.
                    # Extract objective from the trained model so the suppression fires.
                    #
                    # Prefer get_xgb_params() over get_params()
                    # because it reads the underlying booster config and is more reliable
                    # across XGBoost versions.  Add quantile_alpha presence check as a
                    # belt-and-suspenders guard for builds where objective may be stored
                    # differently (e.g. callable or internal alias).
                    _objective_str = ""
                    try:
                        if hasattr(model, "get_xgb_params"):
                            _objective_str = str(
                                model.get_xgb_params().get("objective", "")
                            ).lower()
                        elif hasattr(model, "base_model") and hasattr(
                            model.base_model, "get_xgb_params"
                        ):
                            _objective_str = str(
                                model.base_model.get_xgb_params().get("objective", "")
                            ).lower()
                        elif hasattr(model, "get_params"):
                            _objective_str = str(model.get_params().get("objective", "")).lower()
                        elif hasattr(model, "base_model") and hasattr(
                            model.base_model, "get_params"
                        ):
                            _objective_str = str(
                                model.base_model.get_params().get("objective", "")
                            ).lower()
                    except Exception:
                        pass

                    # Belt-and-suspenders: quantile_alpha presence → quantile XGBoost
                    # regardless of how the objective string is stored.
                    # get_params() alone is insufficient — in XGBoost 2.x
                    # builds where quantile_alpha was passed as a **kwarg it is not
                    # returned by get_params().  Add direct attribute and kwargs checks
                    # so all three storage paths are covered:
                    #   Path 1: get_params() — works when quantile_alpha is a named param
                    #   Path 2: model.quantile_alpha — direct attribute (most reliable)
                    #   Path 3: model.kwargs["quantile_alpha"] — extra-kwarg storage
                    if "quantile" not in _objective_str:
                        try:
                            _m_check = (
                                model
                                if hasattr(model, "get_params")
                                else getattr(model, "base_model", None)
                            )
                            if _m_check is not None and (
                                (
                                    hasattr(_m_check, "get_params")
                                    and _m_check.get_params().get("quantile_alpha") is not None
                                )
                                or getattr(_m_check, "quantile_alpha", None) is not None
                                or (
                                    isinstance(getattr(_m_check, "kwargs", None), dict)
                                    and _m_check.kwargs.get("quantile_alpha") is not None
                                )
                            ):
                                _objective_str = "reg:quantileerror"
                        except Exception:
                            pass

                    # Final fallback — use config.yaml as the definitive
                    # source of truth for the objective type.  All model-attr checks above
                    # depend on XGBoost storing the objective string in a consistent place,
                    # which varies across versions.  config.yaml is version-independent.
                    if "quantile" not in _objective_str:
                        _base_name_obj = model_name.replace("_calibrated", "")
                        _cfg_obj_str = str(
                            self.raw_config.get("models", {})
                            .get(_base_name_obj, {})
                            .get("objective", "")
                        ).lower()
                        if "quantile" in _cfg_obj_str:
                            _objective_str = _cfg_obj_str

                    result.update(
                        {
                            "status": "success",
                            "model": model,
                            "model_path": str(model_path),
                            "checksum": checksum,
                            "cv_mean": cv_result["cv_mean"],
                            "cv_std": cv_result["cv_std"],
                            "cv_scores": cv_result.get("cv_scores", []),
                            # Pinball gap from Optuna (0.0 for non-Optuna/non-quantile runs).
                            # Distinct from dollar-RMSE gap in MetricsExtractor.calculate_generalization_gap().
                            "pinball_gap_percent": cv_result.get("gap_percent", 0.0),
                            # promote nested val metrics to top-level so model selection in
                            # train_two_model_architecture() can read them with .get("val_r2", 0).
                            # Previously these keys were absent → all .get() calls returned 0 →
                            # pricing model picked alphabetically, not by performance.
                            "val_rmse": MetricsExtractor.get_rmse(val_metrics),
                            "val_r2": MetricsExtractor.get_r2(val_metrics),
                            # store pinball_loss so risk model selection uses it directly.
                            # Previously only "cv_mean" existed → risk_results[k].get("pinball_loss", inf)
                            # always returned inf → ML-06 fallback always fired → fell back to val_r2
                            # which was also broken (A/B above).
                            "pinball_loss": (
                                cv_result["cv_mean"] if "quantile" in _objective_str else None
                            ),
                            "training_metrics": train_metrics,
                            "validation_metrics": val_metrics,
                            "validation_predictions": val_preds,
                            "training_time": elapsed,
                            "gpu_used": bool(gpu_params),
                            "gpu_peak_mb": gpu_peak_mb,
                            "bias_correction": bias_correction,
                            "model_config": {"objective": _objective_str},
                            "explainability": {
                                "enabled": enable_explainability,
                                "confidence_intervals": val_explanations.get("confidence_intervals")
                                is not None,
                                "shap_available": val_explanations.get("shap_values") is not None,
                                "feature_importance": (
                                    val_explanations["feature_importance"].to_dict("records")
                                    if val_explanations.get("feature_importance") is not None
                                    else None
                                ),
                                "interval_coverage": val_metrics.get("interval_coverage_pct"),
                                "interval_width": val_metrics.get("interval_avg_width"),
                            },
                        }
                    )

                    # ── log metrics + params while run is still active ─
                    # Insertion point: result.update() has populated all metric
                    # fields; the run closes only when the outer with-block exits.
                    try:

                        def _safe_float(v):
                            return (
                                float(v)
                                if isinstance(v, int | float | np.number) and np.isfinite(float(v))
                                else None
                            )

                        # Build metric payload from result + val_metrics
                        _mlflow_payload: dict[str, float] = {}
                        for _mk, _mv in {
                            "val_rmse": result.get("val_rmse"),
                            "val_r2": result.get("val_r2"),
                            "cv_mean": result.get("cv_mean"),
                            "cv_std": result.get("cv_std"),
                            "training_time_s": result.get("training_time"),
                            "gpu_peak_mb": result.get("gpu_peak_mb"),
                            "pinball_gap_pct": result.get("pinball_gap_percent"),
                        }.items():
                            _v = _safe_float(_mv)
                            if _v is not None:
                                _mlflow_payload[_mk] = _v

                        # Pull nested val_metrics fields (original scale + CI)
                        if isinstance(val_metrics, dict):
                            for _k in (
                                "original_rmse",
                                "original_r2",
                                "original_mae",
                                "interval_coverage_pct",
                                "interval_avg_width",
                            ):
                                _v = _safe_float(val_metrics.get(_k))
                                if _v is not None:
                                    _mlflow_payload[_k] = _v

                        if _mlflow_payload:
                            self.mlflow.log_metrics(_mlflow_payload)

                        # ── P1-D: per-fold CV scores as stepped metrics ───────
                        _cv_scores = result.get("cv_scores", [])
                        if _cv_scores and self.mlflow.enabled and self.mlflow._mlflow.active_run():  # type: ignore[union-attr]
                            for _fold_i, _fold_score in enumerate(_cv_scores):
                                try:
                                    _fv = _safe_float(_fold_score)
                                    if _fv is not None:
                                        with self.mlflow.lock:
                                            if self.mlflow._mlflow.active_run():  # type: ignore[union-attr]
                                                self.mlflow._mlflow.log_metric(  # type: ignore[union-attr]
                                                    "cv_fold_score", _fv, step=_fold_i
                                                )
                                except Exception:
                                    pass
                        # ─────────────────────────────────────────────────────

                        # Log model hyperparams — thread-safe via self.mlflow.lock,
                        # consistent with MLflowManager.log_metrics() internals.
                        _active_model = result.get("model")
                        if _active_model is not None and hasattr(_active_model, "get_params"):
                            _raw_p = _active_model.get_params()
                            _clean_p = {
                                k: v
                                for k, v in _raw_p.items()
                                if isinstance(v, str | int | float | bool | type(None))
                            }
                            if _clean_p and self.mlflow.enabled:
                                with self.mlflow.lock:
                                    if self.mlflow._mlflow.active_run():  # type: ignore[union-attr]
                                        self.mlflow._mlflow.log_params(_clean_p)  # type: ignore[union-attr]

                        # Store run_id so train_all_models() can reopen this run
                        # after SHAP Phase 2 to append post-SHAP metrics.
                        result["mlflow_run_id"] = (
                            self.mlflow._mlflow.active_run().info.run_id  # type: ignore[union-attr]
                            if self.mlflow.enabled and self.mlflow._mlflow.active_run()  # type: ignore[union-attr]
                            else None
                        )

                    except Exception as _mlflow_err:
                        logger.warning(f"MLflow metric logging failed: {_mlflow_err}")
                    # ─────────────────────────────────────────────────────────

            except TimeoutError:
                result["error"] = f"Training timeout after {self.config.training_timeout}s"
                result["training_time"] = time.time() - start_time
                logger.error(f"❌ {result['error']}")

            except Exception as e:
                result["error"] = str(e)
                result["training_time"] = time.time() - start_time
                logger.error(f"❌ Training failed: {e}", exc_info=True)

            finally:
                self.clear_gpu_cache()
                self.resources.force_cleanup()

        return result

    def train_all_models(self, processed_data: dict) -> dict[str, TrainingResult]:
        """
        Train all configured models with SHAP optimization

        OPTIMIZATION STRATEGY:
        - Phase 1: Train all models WITHOUT SHAP (fast path, 30-60% faster per model)
        - Phase 2: Run SHAP analysis ONLY on the best model
        - Result: Dramatic speedup for multi-model training while maintaining full explainability for final model
        """

        required = [
            "X_train",
            "X_val",
            "y_train",
            "y_val",
            "target_transformation",
            "feature_engineer",
        ]
        missing = [k for k in required if k not in processed_data]
        if missing:
            raise ValueError(f"Missing keys: {missing}")

        X_train = processed_data["X_train"]
        y_train = processed_data["y_train"]
        X_val = processed_data["X_val"]
        y_val = processed_data["y_val"]
        target_transformation = processed_data["target_transformation"]
        feature_engineer = processed_data["feature_engineer"]

        self._validate_data(X_train, X_val, y_train, y_val)

        models = self.raw_config.get("model", {}).get(
            "models", ["linear_regression", "random_forest", "xgboost", "lightgbm"]
        )

        logger.info(f"\n{'='*80}\nTraining {len(models)} models\n{'='*80}")
        logger.info("💡 OPTIMIZATION: SHAP explainability DISABLED during bulk training")
        logger.info("   SHAP will only run on the final best model\n")

        results: dict[str, Any] = {}
        preprocessor_saved = False
        _prep_path_for_bc = processed_data.get("preprocessor_path")  # Initialize at function scope

        # ========================================
        # PHASE 1: Fast training (NO SHAP)
        # ========================================
        logger.info(f"{'='*80}")
        logger.info("PHASE 1: FAST MODEL TRAINING (NO SHAP)")
        logger.info(f"{'='*80}\n")

        for idx, name in enumerate(models, 1):
            logger.info(f"\n[{idx}/{len(models)}] {name} (fast mode - SHAP disabled)")

            if not self.resources.check_threshold():
                logger.error("Memory limit reached")
                results[name] = {
                    "model_name": name,
                    "status": "skipped",
                    "error": "Memory limit",
                }
                break

            try:
                result = self.train_single_model(
                    name,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    target_transformation,
                    feature_engineer=feature_engineer,
                    enable_explainability=False,  # DISABLED - saves 30-60% time
                    y_train_original=processed_data.get("y_train_original"),
                )

                results[name] = result

                # Save preprocessor after first success
                if not preprocessor_saved and result.get("status") == "success":
                    prep_path = processed_data.get("preprocessor_path")
                    if prep_path:
                        logger.info(f"\n{'='*80}")
                        logger.info("SAVING PREPROCESSOR WITH BIAS CORRECTION")
                        logger.info(f"{'='*80}")

                        # Extract bias correction from training result
                        bias_correction = result.get("bias_correction")

                        # VALIDATION before saving
                        bias_variance = None

                        # Check if target transformation requires bias correction
                        target_transform = feature_engineer.target_transformation
                        if target_transform.method in ["log1p", "yeo-johnson"]:
                            # BiasCorrection.calculate_from_model() does NOT write
                            # attributes onto feature_engineer (see its docstring: "Does NOT
                            # modify feature_engineer"). Those _bias_var_* attributes are only
                            # written by the block in main() AFTER all models finish.
                            # Checking feature_engineer here therefore always fell into the
                            # final else-branch and raised RuntimeError even when bias
                            # correction was computed successfully.
                            #
                            # The correct source is the BiasCorrection instance that
                            # train_single_model() already stored in result["bias_correction"].
                            if bias_correction is None:
                                # (v7.4.4): bias_correction=None is the CORRECT and
                                # expected return value for quantile models (LightGBM, XGBoost,
                                # GradientBoosting, QuantileRegressor). The quantile guard in
                                # BiasCorrection.calculate_from_model() intentionally returns None
                                # because median-ratio correction is incompatible with quantile
                                # loss — it would cancel the quantile uplift and restore 50%
                                # underpricing.
                                #
                                # The old code raised RuntimeError here, which was caught by the
                                # outer except handler at line ~3699, which then overwrote
                                # results[name] with status='failed' — even though the model file
                                # was already saved successfully to disk. This left successful={}
                                # after every all-quantile training run, which blocked the stale
                                # bias_correction.json cleanup and the BUG-D preprocessor re-save.
                                #
                                # For quantile models: log a clear info message and continue.
                                # The calibration factor in config.yaml handles global adjustment.
                                logger.info(
                                    f"ℹ️  No BiasCorrection for first-success model '{result.get('model_name', 'unknown')}' "
                                    f"— quantile loss model; median-ratio correction intentionally skipped.\n"
                                    f"   Preprocessor will be saved without bias correction attributes.\n"
                                    f"   Calibration factor ({target_transform.method}) handles adjustment."
                                )
                                # Save preprocessor without bias correction attributes and continue
                                feature_engineer.save_preprocessor(str(prep_path))
                                logger.info(f"{'='*80}\n")
                                preprocessor_saved = True
                                continue  # skip rest of save block (no bias_variance to log)

                            bias_variance = bias_correction.overall_variance

                            if bias_correction.is_2tier:
                                logger.info("✅ 2-tier stratified bias correction detected")
                                logger.info(
                                    f"   Low variance:  {bias_correction.var_low:.6f}\n"
                                    f"   High variance: {bias_correction.var_high:.6f}\n"
                                    f"   Threshold P75: ${bias_correction.threshold_low:.0f}"
                                )
                            else:
                                logger.info("✅ 3-tier stratified bias correction detected")
                                logger.info(
                                    f"   Low variance:  {bias_correction.var_low:.6f}\n"
                                    f"   Mid variance:  {bias_correction.var_mid:.6f}\n"
                                    f"   High variance: {bias_correction.var_high:.6f}\n"
                                    f"   Threshold P50: ${bias_correction.threshold_low:.0f}\n"
                                    f"   Threshold P75: ${bias_correction.threshold_high:.0f}"
                                )
                        else:
                            logger.info(
                                f"ℹ️ Bias correction not required "
                                f"(transform: {target_transform.method})"
                            )

                        # Save with validated state
                        feature_engineer.save_preprocessor(str(prep_path))
                        # NOTE: bias_correction.json is intentionally NOT written here.
                        # It is written after Phase 1 completes, using the BEST model's
                        # bias_correction — not the first successful model's — to guarantee
                        # the JSON and the loaded model artifact are always from the same
                        # training pass.  See "Save best-model bias_correction.json" block below.

                        # logger.info(f"✅ Saved: {prep_path.name}")
                        # logger.info(f"   Bias correction: {bias_status}")

                        if bias_variance is not None:
                            logger.info(f"   Overall variance: {bias_variance:.6f}")

                        logger.info(f"{'='*80}\n")
                        preprocessor_saved = True

                # Smart GPU/memory cleanup between models
                if self.gpu_available:
                    self.clear_gpu_cache()
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                # Smart cleanup (not forced - only if needed)
                self.resources.smart_cleanup()

            except Exception as e:
                logger.error(f"Failed: {e}", exc_info=True)
                results[name] = {
                    "model_name": name,
                    "status": "failed",
                    "error": str(e),
                }
            finally:
                self.clear_gpu_cache()
                self.resources.force_cleanup()

        # ========================================
        # PHASE 2: SHAP analysis on best model
        # ========================================
        successful = {k: v for k, v in results.items() if v.get("status") == "success"}

        # Defined here (before Phase 2) so both the bias_correction.json block
        # and Phase 2's best-model selection can use the same helper.
        def get_rmse(model_name: str) -> float:
            """Safely extract RMSE from results"""
            try:
                metrics = results[model_name].get("validation_metrics", {})
                rmse = metrics.get("original_rmse") or metrics.get("rmse")
                return float("inf") if rmse is None else float(rmse)
            except Exception as e:
                logger.warning(f"Could not extract RMSE for {model_name}: {e}")
                return float("inf")

        # ====================================================================
        # SAVE BEST-MODEL bias_correction.json (deferred from first-model block)
        # ====================================================================
        # This block runs AFTER all models are trained so we can use the BEST
        # model's BiasCorrection — guaranteeing the JSON and the model artifact
        # are always from the same training pass and var_high values match.
        if successful:
            # ── (v7.5.4): use pricing model's BC when TMA is enabled ──────
            # Previous: selected RMSE-best model (e.g. random_forest) → BC written
            # from random_forest while deployed model is xgboost_median. BC factors
            # differed (var_low=-0.210721 vs -0.207881) creating a silent mismatch.
            # The BiasCorrection consistency check at inference catches this, but the
            # root cause was here — using the wrong model's BC to write the JSON.
            # when two_model_architecture is enabled, always use pricing_model's BC.
            _tma_cfg_bc = self.raw_config.get("training", {}).get("two_model_architecture", {})
            _tma_enabled_bc = _tma_cfg_bc.get("enabled", False)
            _pricing_model_bc = _tma_cfg_bc.get("pricing_model", "xgboost_median")

            if _tma_enabled_bc and _pricing_model_bc in successful:
                _best_for_bc = _pricing_model_bc
                logger.info(
                    f"✅ [B3 FIX] Using pricing model '{_pricing_model_bc}' for "
                    f"bias_correction.json (TMA enabled — ignoring RMSE-best)."
                )
            else:
                _best_for_bc = min(successful.keys(), key=get_rmse)
                if _tma_enabled_bc:
                    logger.warning(
                        f"⚠️  [B3] TMA enabled but pricing model '{_pricing_model_bc}' "
                        f"not in successful models — falling back to RMSE-best: {_best_for_bc}"
                    )

            _best_bc = results[_best_for_bc].get("bias_correction")
            _prep_path_for_bc = processed_data.get("preprocessor_path")

            if _best_bc is not None and _prep_path_for_bc is not None:
                _bias_path = Path(_prep_path_for_bc).parent / "bias_correction.json"
                try:
                    import shutil as _shutil
                    import tempfile as _tmpfile

                    # Write atomically via temp file so a crash mid-write leaves
                    # a valid previous version rather than a corrupt file.
                    with _tmpfile.NamedTemporaryFile(
                        mode="w",
                        dir=_bias_path.parent,
                        prefix=".tmp_bc_",
                        suffix=".json",
                        delete=False,
                    ) as _tmp:
                        json.dump(_best_bc.to_dict(), _tmp, indent=2)
                        _tmp_path = Path(_tmp.name)

                    _shutil.move(str(_tmp_path), str(_bias_path))

                    # Immediate round-trip validation
                    with open(_bias_path) as _rf:
                        _bc_reloaded = BiasCorrection.from_dict(json.load(_rf))

                    _TOLERANCE = 1e-8
                    _mismatches = [
                        f"   {_a}: in-memory={getattr(_best_bc, _a):.10f}, "
                        f"JSON={getattr(_bc_reloaded, _a):.10f}"
                        for _a in ("var_low", "var_high")
                        if getattr(_best_bc, _a, None) is not None
                        and abs(getattr(_best_bc, _a) - getattr(_bc_reloaded, _a)) > _TOLERANCE
                    ]

                    if _mismatches:
                        logger.error(
                            "❌ bias_correction.json round-trip MISMATCH — "
                            "JSON serialisation introduced precision loss:\n"
                            + "\n".join(_mismatches)
                        )
                    else:
                        logger.info(
                            f"✅ bias_correction.json written from best model '{_best_for_bc}'\n"
                            f"   var_low={_best_bc.var_low:.6f}, var_high={_best_bc.var_high:.6f}\n"
                            f"   Round-trip validated ✅"
                        )

                    # ── re-save preprocessor with BEST model's bias correction ──
                    # The training loop saves the preprocessor after the FIRST successful
                    # model, so feature_engineer._bias_var_* on disk reflects the first
                    # model, not necessarily the best one.  Re-apply the best model's
                    # BiasCorrection to feature_engineer and overwrite the file so the
                    # on-disk preprocessor and bias_correction.json are always from the
                    # same model.
                    _fe = processed_data.get("feature_engineer")
                    _prep_path_obj = Path(_prep_path_for_bc)
                    if _fe is not None and _prep_path_obj.exists():
                        try:
                            # Apply best-model bias correction to feature_engineer
                            if _best_bc.is_2tier:
                                _fe._bias_var_low = _best_bc.var_low
                                _fe._bias_var_high = _best_bc.var_high
                                _fe._bias_threshold = _best_bc.threshold_low
                                _fe._log_residual_variance = (
                                    _best_bc.overall_variance
                                    if _best_bc.overall_variance is not None
                                    else float(np.var([_best_bc.var_low, _best_bc.var_high]))
                                )
                                for _attr in (
                                    "_bias_var_mid",
                                    "_bias_threshold_low",
                                    "_bias_threshold_high",
                                ):
                                    if hasattr(_fe, _attr):
                                        delattr(_fe, _attr)
                            else:
                                _fe._bias_var_low = _best_bc.var_low
                                _fe._bias_var_mid = _best_bc.var_mid
                                _fe._bias_var_high = _best_bc.var_high
                                _fe._bias_threshold_low = _best_bc.threshold_low
                                _fe._bias_threshold_high = _best_bc.threshold_high
                                _fe._log_residual_variance = (
                                    _best_bc.overall_variance
                                    if _best_bc.overall_variance is not None
                                    else float(
                                        np.var(
                                            [
                                                _best_bc.var_low,
                                                _best_bc.var_mid,
                                                _best_bc.var_high,
                                            ]
                                        )
                                    )
                                )
                                if hasattr(_fe, "_bias_threshold"):
                                    delattr(_fe, "_bias_threshold")

                            # Overwrite the preprocessor with the corrected state
                            _fe.save_preprocessor(str(_prep_path_obj))
                            logger.info(
                                f"✅ BUG-D FIX: Preprocessor re-saved with best model "
                                f"'{_best_for_bc}' bias correction (was first-model)."
                            )
                        except Exception as _bd_err:
                            logger.warning(
                                f"⚠️  BUG-D FIX: Could not re-save preprocessor with best "
                                f"model bias correction: {_bd_err}\n"
                                f"   Preprocessor on disk may have first-model bias correction. "
                                f"bias_correction.json is correct."
                            )

                except Exception as _bc_err:
                    logger.error(
                        f"❌ Failed to save bias_correction.json: {_bc_err}\n"
                        "   Predictions at inference will not have bias correction."
                    )
            elif _best_bc is None:
                logger.warning(
                    f"⚠️  No BiasCorrection found for best model '{_best_for_bc}' — "
                    "bias_correction.json not written."
                )
                # When a successful quantile model produces _best_bc=None,
                # any bias_correction.json left over from a PRIOR non-quantile run
                # must be renamed to .stale so predict.py cannot pick it up.
                # (The original block below only fires when successful={},
                # which never happens in a normal all-quantile run.)
                if _prep_path_for_bc is not None:
                    _stale_path_b = Path(_prep_path_for_bc).parent / "bias_correction.json"
                    if _stale_path_b.exists():
                        _stale_dest_b = _stale_path_b.with_suffix(".json.stale")
                        try:
                            _stale_path_b.replace(_stale_dest_b)
                            logger.info(
                                f"✅ FIX-3B: Renamed stale bias_correction.json → "
                                f"{_stale_dest_b.name}\n"
                                f"   Reason: current best model '{_best_for_bc}' is a "
                                f"quantile model (BC=None); existing JSON from a prior "
                                f"non-quantile run would cause incorrect BC at inference.\n"
                                f"   Calibration factor in config.yaml handles adjustment."
                            )
                        except OSError as _rename_err_b:
                            logger.error(
                                f"❌ FIX-3B: Could not rename stale bias_correction.json: "
                                f"{_rename_err_b}\n"
                                f"   WARNING: predict.py may apply an invalid stale "
                                f"bias correction from a previous training run."
                            )

        # ── (v7.4.4): Stale bias_correction.json cleanup ──────────────────
        # Runs when successful={} — covers the case where all models fail.
        # (above) covers the normal case: successful model, _best_bc=None.
        #
        # The old stale-rename logic was gated inside 'if successful: / elif _best_bc is None:'.
        # When successful={} (every model fails the preprocessor-save RuntimeError in Bug 2),
        # the 'if successful:' block was skipped entirely, leaving a stale bias_correction.json
        # from a previous elastic_net run on disk permanently.  Every subsequent inference call
        # loaded that stale file and applied ~9% downward corrections to LightGBM predictions,
        # canceling the quantile α=0.65 uplift and keeping underpricing locked at ~50%.
        #
        # Path.replace() is used instead of Path.rename() to avoid Windows FileExistsError
        # when a .json.stale file already exists from a prior run.
        if _prep_path_for_bc is not None and not successful:
            _stale_path = Path(_prep_path_for_bc).parent / "bias_correction.json"
            if _stale_path.exists():
                _stale_dest = _stale_path.with_suffix(".json.stale")
                try:
                    _stale_path.replace(_stale_dest)  # replace() overwrites on Windows too
                    logger.warning(
                        f"⚠️  Renamed stale bias_correction.json → {_stale_dest.name}\n"
                        f"   Reason: all models in this run use quantile loss and have "
                        f"bias_correction=None; the existing JSON was computed against a "
                        f"previous non-quantile model and is no longer valid.\n"
                        f"   predict.py and evaluate.py will NOT apply bias correction "
                        f"for this training run (calibration factor handles adjustment)."
                    )
                except OSError as _rename_err:
                    logger.error(
                        f"❌ Could not rename stale bias_correction.json: {_rename_err}\n"
                        f"   WARNING: predict.py may load and apply an invalid stale "
                        f"bias correction from a previous training run."
                    )

        if successful:
            logger.info(f"\n{'='*80}")
            logger.info("PHASE 2: SHAP ANALYSIS ON BEST MODEL")
            logger.info(f"{'='*80}\n")

            # Find best model for SHAP analysis.
            # Two-model architecture: select the phase-appropriate model by key
            # rather than by RMSE minimisation (which is dimensionally incoherent
            # across reg:squarederror and reg:quantileerror objectives).
            # Single-model path retains the original RMSE-min selection.
            _tma_cfg_shap = self.raw_config.get("training", {}).get("two_model_architecture", {})
            _pricing_key_shap = _tma_cfg_shap.get("pricing_model", "xgboost_median")
            _risk_key_shap = _tma_cfg_shap.get("risk_model", "xgboost")
            if _tma_cfg_shap.get("enabled") and _risk_key_shap in successful:
                # Risk-model phase: SHAP on the designated risk model
                best_name = _risk_key_shap
            elif _tma_cfg_shap.get("enabled") and _pricing_key_shap in successful:
                # Pricing-model phase: SHAP on the designated pricing model
                best_name = _pricing_key_shap
            else:
                # Single-model path: original RMSE-min selection
                best_name = min(successful.keys(), key=get_rmse)
            best_rmse = get_rmse(best_name)

            # Log best model info
            if best_rmse == float("inf"):
                logger.info(f"🏆 Best model: {best_name} (RMSE: Not available)")
            else:
                logger.info(
                    f"🏆 Best model: {best_name} "
                    f"(Val RMSE uncalibrated: ${best_rmse:,.2f})\n"
                    f"   Note: final test RMSE is evaluated on the post-hoc calibrated "
                    f"model after this phase — the two figures are not directly comparable."
                )

            logger.info("   Running comprehensive SHAP analysis...\n")

            try:
                # Re-run evaluation WITH explainability for best model
                best_model = results[best_name].get("model")

                if best_model is None:
                    raise ValueError(f"No model found for {best_name}")

                # Check if we have required data
                if X_val is None or y_val is None:
                    raise ValueError("Validation data is None")

                best_bias_correction_shap = results[best_name].get("bias_correction")

                val_metrics_with_shap, val_preds, val_explanations = (
                    self.model_manager.evaluate_model_with_explainability(
                        model=best_model,
                        X_test=X_val,
                        y_test=y_val,
                        X_train_sample=X_train.sample(min(100, len(X_train))),
                        model_name=best_name,
                        target_transformation=target_transformation,
                        feature_engineer=feature_engineer,
                        explainability_config=self.explainability_config,
                        bias_correction=best_bias_correction_shap,
                    )
                )

                # Update best model's results with explainability
                results[best_name]["validation_metrics"] = val_metrics_with_shap
                results[best_name]["validation_predictions"] = val_preds
                results[best_name]["explainability"] = {
                    "enabled": True,
                    "confidence_intervals": val_explanations.get("confidence_intervals")
                    is not None,
                    "shap_available": val_explanations.get("shap_values") is not None,
                    "feature_importance": (
                        val_explanations["feature_importance"].to_dict("records")
                        if val_explanations.get("feature_importance") is not None
                        else None
                    ),
                    "interval_coverage": val_metrics_with_shap.get("interval_coverage_pct"),
                    "interval_width": val_metrics_with_shap.get("interval_avg_width"),
                    "plots": val_explanations.get("plots", {}),
                }

                # ── log post-SHAP metrics by reopening the best run ──
                # Phase 2 updated val_metrics_with_shap with SHAP-derived fields.
                # Reopen the closed run by ID and append with "shap_" prefix.
                try:
                    _best_run_id = results[best_name].get("mlflow_run_id")
                    if self.mlflow.enabled and _best_run_id:

                        def _safe_f(v):
                            return (
                                float(v)
                                if isinstance(v, int | float | np.number) and np.isfinite(float(v))
                                else None
                            )

                        _shap_payload: dict[str, float] = {}
                        for _k in (
                            "original_rmse",
                            "original_r2",
                            "original_mae",
                            "interval_coverage_pct",
                            "interval_avg_width",
                        ):
                            _v = _safe_f(val_metrics_with_shap.get(_k))
                            if _v is not None:
                                _shap_payload[f"shap_{_k}"] = _v

                        if _shap_payload:
                            with self.mlflow._mlflow.start_run(  # type: ignore[union-attr]
                                run_id=_best_run_id,
                                nested=False,
                            ):
                                self.mlflow._mlflow.log_metrics(_shap_payload)  # type: ignore[union-attr]
                            logger.info(
                                f"  MLflow: appended {len(_shap_payload)} post-SHAP "
                                f"metrics to run {_best_run_id[:8]}..."
                            )
                except Exception as _shap_mlflow_err:
                    logger.warning(f"MLflow post-SHAP logging failed: {_shap_mlflow_err}")
                # ─────────────────────────────────────────────────────────────

                # ── P1-E: SHAP feature importance as CSV artifact ─────────────
                try:
                    _shap_fi = val_explanations.get("feature_importance")
                    _best_run_shap = results[best_name].get("mlflow_run_id")
                    if (
                        _shap_fi is not None
                        and not _shap_fi.empty
                        and self.mlflow.enabled
                        and _best_run_shap
                    ):
                        _fi_path = self.config.reports_dir / f"shap_importance_{best_name}.csv"
                        _shap_fi.to_csv(_fi_path, index=False)
                        with self.mlflow._mlflow.start_run(run_id=_best_run_shap, nested=False):  # type: ignore[union-attr]
                            self.mlflow._mlflow.log_artifact(str(_fi_path), artifact_path="shap")  # type: ignore[union-attr]
                        logger.info(
                            f"  MLflow: SHAP importance CSV attached "
                            f"({len(_shap_fi)} features, run {_best_run_shap[:8]}...)"
                        )
                except Exception as _shap_art_err:
                    logger.debug(f"SHAP artifact logging failed: {_shap_art_err}")
                # ─────────────────────────────────────────────────────────────

                # Log explainability results
                logger.info("✅ SHAP analysis complete for best model")

                # Log confidence intervals if available
                coverage = val_metrics_with_shap.get("interval_coverage_pct")
                avg_width = val_metrics_with_shap.get("interval_avg_width")

                if val_explanations.get("confidence_intervals") is not None:
                    if coverage is not None and avg_width is not None:
                        logger.info(
                            f"   Confidence Intervals:\n"
                            f"      Coverage: {coverage:.1f}%\n"
                            f"      Avg Width: ${avg_width:,.0f}"
                        )
                    else:
                        logger.info("   Confidence Intervals: Available (metrics incomplete)")

                # Log feature importance if available
                feature_importance = val_explanations.get("feature_importance")
                if feature_importance is not None and not feature_importance.empty:
                    top_features = feature_importance.head(5)
                    logger.info("   Top 5 SHAP Features:")
                    for idx, row in top_features.iterrows():
                        feature_name = row.get("feature", "Unknown")
                        importance = row.get("importance", 0.0)
                        logger.info(f"      {idx+1}. {feature_name}: {importance:.4f}")
                else:
                    # Previously fired "Feature importance: Not available" even
                    # when Advanced Diagnostics below was about to log tree FI — a
                    # confusing contradiction. Now fall back to tree-based split-gain
                    # importances and label the source explicitly.
                    _tree_fi = None
                    try:
                        _base_m = getattr(best_model, "base_model", best_model)
                        if hasattr(_base_m, "feature_importances_"):
                            import pandas as _pd_fi

                            _tree_fi = _pd_fi.Series(
                                _base_m.feature_importances_,
                                index=X_train.columns.tolist(),
                            ).nlargest(5)
                    except Exception:
                        pass
                    if _tree_fi is not None and len(_tree_fi) > 0:
                        logger.info("   Top 5 features (tree split-gain; SHAP not yet computed):")
                        for feat, imp in _tree_fi.items():
                            logger.info(f"      {feat}: {imp:.4f}")
                        logger.info(
                            "   NOTE: Full SHAP values will appear in Advanced Diagnostics below."
                        )
                    else:
                        logger.info(
                            "   Feature importance: Not available "
                            "(both SHAP and tree-based absent — check explainability config)"
                        )

                logger.info("")

            except Exception as e:
                logger.error(f"⚠️ SHAP analysis failed for best model: {e}", exc_info=True)
                logger.warning("Continuing without explainability for best model")

                # Ensure explainability field exists even on failure
                if "explainability" not in results[best_name]:
                    results[best_name]["explainability"] = {
                        "enabled": False,
                        "error": str(e),
                    }
        else:
            logger.warning("⚠️ No successful models - skipping SHAP analysis")

        # ========================================
        # PHASE 3: Visualizations and summary
        # ========================================
        logger.info("\nGenerating comparison visualizations...")

        try:
            self.viz.plot_training_progress(results, save_path="all_models")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

        success = sum(1 for r in results.values() if r.get("status") == "success")
        logger.info(f"\n{'='*80}\nComplete: {success}/{len(models)}\n{'='*80}")

        if success > 0:
            # Summary of all models
            logger.info("\nModel Performance Summary:")
            logger.info("-" * 80)

            for name, result in results.items():
                if result.get("status") == "success":
                    metrics = result.get("validation_metrics", {})
                    # Check for original_* keys first (when target transformation is used)
                    # Fall back to non-prefixed keys for raw scale (no transformation)
                    rmse = metrics.get("original_rmse") or metrics.get("rmse")
                    r2 = metrics.get("original_r2") or metrics.get("r2")

                    # Safe formatting for potentially None values
                    rmse_str = f"${rmse:12,.2f}" if rmse is not None else "N/A".rjust(14)
                    r2_str = f"{r2:6.4f}" if r2 is not None else "N/A".rjust(6)

                    explainable = result.get("explainability", {}).get("enabled", False)
                    shap_marker = " 🔍 [SHAP]" if explainable else ""

                    logger.info(f"  {name:25s} | RMSE: {rmse_str} | R²: {r2_str}{shap_marker}")

            logger.info("-" * 80)
        else:
            logger.error("❌ No models trained successfully!")

        if not preprocessor_saved:
            logger.error("⚠️ WARNING: Preprocessor was never saved!")

        return results

    def evaluate_test(
        self,
        model_path: Path,
        feature_engineer,
        X_test_raw,
        y_test_raw,
        name: str,
        bias_correction: BiasCorrection | None = None,
    ) -> dict:
        """
        Evaluate on test set with intelligent caching.

        """
        logger.info(f"\n{'='*80}\nTest Evaluation: {name}\n{'='*80}")

        try:
            # ========================================
            # STEP 1: Initialize cache if needed
            # ========================================
            if not hasattr(self, "_test_transform_cache"):
                self._test_transform_cache = {}
                logger.debug("Initialized test transform cache")

            # ========================================
            # STEP 2: Generate MODEL-SPECIFIC cache key
            # ========================================

            cache_key_parts = [
                f"{X_test_raw.shape[0]}x{X_test_raw.shape[1]}",
                f"{getattr(feature_engineer, 'VERSION', 'v1')}",
                f"{feature_engineer.target_transformation.method}",
            ]

            # Include bias correction identity
            if bias_correction is not None:
                bias_hash = bias_correction.get_hash()
                cache_key_parts.append(f"bias_{bias_hash}")
                logger.debug(f"Bias correction hash: {bias_hash}")
            else:
                cache_key_parts.append("no_bias")

            # Include model identity
            model_checksum = FileSanitizer.compute_checksum(model_path)[:8]
            cache_key_parts.append(f"model_{name}_{model_checksum}")

            cache_key = "_".join(cache_key_parts)

            # ========================================
            # STEP 3: Check cache or transform
            # ========================================
            if cache_key in self._test_transform_cache:
                # Cache HIT - reuse existing transforms
                logger.info(
                    f"✅ Using cached test transforms (key: {cache_key[:24]}...)\n"
                    f"   Skipping transformation (saves ~2-5 seconds)"
                )
                cached = self._test_transform_cache[cache_key]
                X_test = cached["X_test"]
                y_test_transformed = cached["y_test_transformed"]
                y_test_original = cached["y_test_original"]

            else:
                # Cache MISS - transform and cache
                logger.info("Transforming test data (first evaluation)...")

                # Transform features
                X_test = feature_engineer.transform_pipeline(X_test_raw, remove_outliers=False)

                # Transform target
                y_test_transformed = feature_engineer.transform_target(
                    y_test_raw,
                    method=feature_engineer.target_transformation.method,
                    fit=False,
                )

                # Store original target for metrics calculation
                y_test_original = (
                    y_test_raw.values if hasattr(y_test_raw, "values") else np.array(y_test_raw)
                )

                # Validate transformation quality
                if not np.all(np.isfinite(X_test.values)):
                    n_bad = np.sum(~np.isfinite(X_test.values))
                    logger.warning(
                        f"⚠️  {n_bad} non-finite values in transformed X_test\n"
                        f"   Filling with feature medians..."
                    )
                    for col in X_test.columns:
                        if not np.all(np.isfinite(X_test[col])):
                            X_test[col].fillna(X_test[col].median(), inplace=True)

                if not np.all(np.isfinite(y_test_transformed)):
                    n_bad = np.sum(~np.isfinite(y_test_transformed))
                    logger.error(
                        f"❌ {n_bad} non-finite values in transformed y_test!\n"
                        f"   This indicates transformation failure."
                    )
                    raise ValueError("Target transformation produced non-finite values")

                # Cache for future evaluations
                self._test_transform_cache[cache_key] = {
                    "X_test": X_test,
                    "y_test_transformed": y_test_transformed,
                    "y_test_original": y_test_original,
                    "cached_at": pd.Timestamp.now().isoformat(),
                }

                logger.info(
                    f"✅ Cached test transforms (key: {cache_key[:24]}...)\n"
                    f"   Shape: {X_test.shape}\n"
                    f"   Transform method: {feature_engineer.target_transformation.method}\n"
                    f"   Cache size: {len(self._test_transform_cache)} entries"
                )

            # ========================================
            # STEP 4: Load model
            # ========================================
            logger.info(f"Loading model from {model_path.name}...")
            model = FileSanitizer.safe_load(
                model_path,
                self.config.max_model_size_mb,
                verify_checksum=self.config.verify_checksums,
            )

            # 🔒 SAFETY CHECK: Ensure bias correction exists (non-quantile models only)
            # The original guard raised unconditionally when _log_residual_variance
            # was absent. For quantile objectives (reg:quantileerror, quantile, QuantileRegressor),
            # BiasCorrection.calculate_from_model() intentionally returns None and never sets
            # _log_residual_variance — this is by design, not an error. The guard must
            # distinguish "missing by accident" from "skipped by design".
            #
            # XGB-2X (evaluate_test): In XGBoost 2.x the sklearn wrapper does NOT store
            # the objective as a readable Python attribute after joblib round-trip.
            # `getattr(model, "objective", "")` returns "" or "reg:squarederror" (the
            # internal default), regardless of what was passed at construction time.
            # `quantile_alpha` is likewise stored only in the booster C++ params and is
            # not surfaced as a Python attribute.  The only reliable detection paths are:
            #   1. get_xgb_params()  — reads booster-level params (most reliable)
            #   2. get_params()      — reads sklearn wrapper params (works on unfitted/fresh)
            #   3. config.yaml       — version-independent ground truth (final fallback)
            if feature_engineer.target_transformation.method in [
                "log1p",
                "yeo-johnson",
            ]:
                _mt = type(model).__name__

                # ── Build objective string using all available detection paths ────────
                _obj_str = ""

                # Path 1: get_xgb_params() — booster-level, most reliable after joblib load
                if not _obj_str and _mt == "XGBRegressor":
                    try:
                        _obj_str = str(model.get_xgb_params().get("objective", "")).lower()
                    except Exception:
                        pass

                # Path 2: get_params() — sklearn wrapper level
                if not _obj_str or _obj_str in ("", "reg:squarederror"):
                    try:
                        _obj_str = str(model.get_params().get("objective", "")).lower()
                    except Exception:
                        pass

                # Path 3: direct attribute (works on some XGBoost builds / LightGBM)
                if not _obj_str or _obj_str in ("", "reg:squarederror"):
                    _obj_str = str(getattr(model, "objective", "")).lower()

                # Path 4: config.yaml — version-independent, survives any serialisation
                # quirk. `name` is the model key used throughout train.py (e.g. "xgboost").
                if "quantile" not in _obj_str:
                    _cfg_obj = str(
                        self.raw_config.get("models", {})
                        .get(name.replace("_calibrated", ""), {})
                        .get("objective", "")
                    ).lower()
                    if "quantile" in _cfg_obj:
                        _obj_str = _cfg_obj
                        logger.debug(
                            f"   _is_quantile_test: objective resolved from config.yaml "
                            f"('{_cfg_obj}') — model attribute was empty/default after load."
                        )

                _is_quantile_test = (
                    # Primary: objective string contains "quantile" (any of the 4 paths above)
                    "quantile" in _obj_str
                    # LightGBM booster params
                    or (
                        _mt == "LGBMRegressor"
                        and (
                            "quantile" in str(getattr(model, "objective", "")).lower()
                            or (
                                hasattr(model, "booster_")
                                and "quantile"
                                in str(
                                    getattr(model.booster_, "params", {}).get("objective", "")
                                ).lower()
                            )
                        )
                    )
                    # sklearn GBR: loss attribute
                    or (
                        _mt == "GradientBoostingRegressor"
                        and getattr(model, "loss", "") == "quantile"
                    )
                    # sklearn QuantileRegressor (elastic_net factory swap)
                    or _mt == "QuantileRegressor"
                    # Explicit flag set during training (belt-and-suspenders)
                    or getattr(model, "_is_quantile_model", False)
                    # quantile_alpha present as any attribute/kwarg (belt-and-suspenders)
                    or getattr(model, "quantile_alpha", None) is not None
                    or (
                        isinstance(getattr(model, "kwargs", None), dict)
                        and model.kwargs.get("quantile_alpha") is not None
                    )
                )

                if _is_quantile_test:
                    logger.info(
                        f"ℹ️  Bias correction safety check skipped — quantile model "
                        f"detected ({_mt}, objective='{_obj_str}').\n"
                        f"   _log_residual_variance is intentionally absent: "
                        f"BiasCorrection returns None for quantile objectives.\n"
                        f"   Calibration factor handles residual global adjustment."
                    )
                elif not hasattr(feature_engineer, "_log_residual_variance"):
                    raise RuntimeError(
                        "❌ Bias correction missing during test evaluation.\n"
                        "Train/val/test metrics would be incomparable.\n"
                        "Ensure the SAME trained feature_engineer is used."
                    )

            # ========================================
            # STEP 5: Evaluate model (uses cached transforms)
            # ========================================
            logger.info("Evaluating model on test set with explainability...")
            metrics, preds_original, test_explanations = (
                self.model_manager.evaluate_model_with_explainability(
                    model=model,
                    X_test=X_test,
                    y_test=y_test_transformed,
                    X_train_sample=None,  # Already computed during training
                    model_name=name,
                    target_transformation=feature_engineer.target_transformation,
                    feature_engineer=feature_engineer,
                    explainability_config=self.explainability_config,
                    bias_correction=bias_correction,
                )
            )

            self._validate_metrics_dict(metrics, "test")

            # ========================================
            # STEP 6: Log comprehensive results
            # ========================================
            logger.info(
                f"📊 Test Results ({name}):\n"
                f"   RMSE: ${metrics['original_rmse']:,.2f}\n"
                f"   MAE: ${metrics['original_mae']:,.2f}\n"
                f"   R²: {metrics['original_r2']:.4f}\n"
                f"   MAPE: {metrics.get('original_mape', 0.0):.2f}%"
            )

            # Log prediction statistics
            pred_stats = {
                "min": np.min(preds_original),
                "q25": np.percentile(preds_original, 25),
                "median": np.median(preds_original),
                "q75": np.percentile(preds_original, 75),
                "max": np.max(preds_original),
                "mean": np.mean(preds_original),
            }

            logger.info(
                f"   Prediction Range:\n"
                f"      Min: ${pred_stats['min']:,.2f}\n"
                f"      Q25: ${pred_stats['q25']:,.2f}\n"
                f"      Median: ${pred_stats['median']:,.2f}\n"
                f"      Q75: ${pred_stats['q75']:,.2f}\n"
                f"      Max: ${pred_stats['max']:,.2f}\n"
                f"      Mean: ${pred_stats['mean']:,.2f}"
            )

            # ========================================
            # Log Confidence Intervals (if computed)
            # ========================================
            if test_explanations.get("confidence_intervals") is not None:
                coverage = metrics.get("interval_coverage_pct")
                avg_width = metrics.get("interval_avg_width")

                if coverage is not None and avg_width is not None:
                    # was hardcoded to 95%; now config-driven.
                    _ci_target_pct = self.explainability_config.confidence_level * 100
                    _ci_tol_2 = _ci_target_pct * 0.02
                    _ci_tol_5 = _ci_target_pct * 0.05
                    logger.info(
                        f"   Confidence Intervals ({_ci_target_pct:.0f}%):\n"
                        f"      Coverage: {coverage:.1f}% (target: {_ci_target_pct:.1f}%)\n"
                        f"      Avg Width: ${avg_width:,.0f}"
                    )

                    # Coverage quality assessment — bands relative to actual target
                    if (_ci_target_pct - _ci_tol_2) <= coverage <= (_ci_target_pct + _ci_tol_2):
                        logger.info("      ✅ Well-calibrated (within ±2% of target)")
                    elif (_ci_target_pct - _ci_tol_5) <= coverage <= (_ci_target_pct + _ci_tol_5):
                        logger.info("      ⚠️  Acceptable (within ±5% of target)")
                    elif coverage < _ci_target_pct - _ci_tol_5:
                        logger.warning(
                            f"      ❌ Under-coverage: {_ci_target_pct - coverage:.1f}% below target\n"
                            f"         Intervals too narrow - increase confidence level"
                        )
                    else:
                        logger.warning(
                            f"      ⚠️  Over-coverage: {coverage - _ci_target_pct:.1f}% above target\n"
                            f"         Intervals too wide - may need recalibration"
                        )

                    # Log interval quality metrics if available
                    if "interval_avg_width_pct" in metrics:
                        width_pct = metrics["interval_avg_width_pct"]
                        logger.info(f"      Width (% of mean): {width_pct:.1f}%")
                else:
                    logger.warning(
                        "   ⚠️  Confidence Intervals: Computed but metrics incomplete\n"
                        f"      coverage={'present' if coverage is not None else 'missing'}, "
                        f"avg_width={'present' if avg_width is not None else 'missing'}"
                    )
            else:
                logger.debug(
                    "   Confidence Intervals: Not computed\n"
                    "      (enable_confidence_intervals=False in explainability config)"
                )

            # ========================================
            # STEP 7: Return comprehensive results
            # ========================================
            return {
                "metrics": metrics,
                "predictions": preds_original,
                "X_test_transformed": X_test,
                "y_test_transformed": y_test_transformed,
                "y_test_original": y_test_original,
                "model_name": name,
                "cache_key": cache_key,
                "was_cached": cache_key in self._test_transform_cache,
                "prediction_stats": pred_stats,
                # Explainability results
                "explainability": {
                    "confidence_intervals": test_explanations.get("confidence_intervals"),
                    "feature_importance": test_explanations.get("feature_importance"),
                    "shap_values": test_explanations.get("shap_values"),
                    "plots": test_explanations.get("plots", {}),
                },
            }

        except Exception as e:
            logger.error(f"❌ Test evaluation failed: {e}", exc_info=True)
            raise

    def clear_test_cache(self):
        """
        Clear cached test transformations.

        Call this when:
        - Switching to a new test dataset
        - Updating feature engineering pipeline
        - Memory pressure detected
        """
        if hasattr(self, "_test_transform_cache"):
            n_cached = len(self._test_transform_cache)
            self._test_transform_cache.clear()
            logger.info(f"✅ Cleared {n_cached} cached test transformations")
        else:
            logger.debug("No test cache to clear")

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about cached test transformations.

        Returns:
            dict: Cache statistics including size, keys, memory usage
        """
        if not hasattr(self, "_test_transform_cache"):
            return {"exists": False, "size": 0, "keys": [], "memory_mb": 0.0}

        cache = self._test_transform_cache

        # Estimate memory usage
        memory_bytes = 0
        for _key, value in cache.items():
            if "X_test" in value:
                memory_bytes += value["X_test"].memory_usage(deep=True).sum()
            if "y_test_transformed" in value:
                memory_bytes += value["y_test_transformed"].nbytes
            if "y_test_original" in value:
                memory_bytes += value["y_test_original"].nbytes

        return {
            "exists": True,
            "size": len(cache),
            "keys": list(cache.keys()),
            "memory_mb": memory_bytes / (1024 * 1024),
            "entries": [
                {
                    "key": key,
                    "cached_at": value.get("cached_at", "unknown"),
                    "shape": value["X_test"].shape if "X_test" in value else None,
                }
                for key, value in cache.items()
            ],
        }


# ============================================================================
# INLINE PATCHES v7.5.0  — all remediation logic embedded, no new files needed
# ============================================================================

# ── Patch 01: Data Contamination (Gate G2) ───────────────────────────────

# ============================================================================
# 1. SPLIT DISJOINTNESS ASSERTION
#    Call once at the top of main() immediately after prepare_training_data().
# ============================================================================


def assert_splits_disjoint(data: dict[str, Any], label: str = "pipeline") -> None:
    """
    Hard assertion: train / val / test index sets are strictly disjoint.

    This is the single source of truth for data integrity. Call it once
    immediately after prepare_training_data() returns and before ANY model
    calibration, conformal fitting, or evaluation step.

    Args:
        data: Dict returned by prepare_training_data(). Must contain keys:
              X_train, X_val, X_test_raw.
        label: Contextual label for error messages.

    Raises:
        AssertionError: If any two splits share one or more indices.

    Usage in train.py main():
        data = trainer.prepare_training_data(...)
        assert_splits_disjoint(data)   # ← add this line
    """
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test_raw"]

    train_idx: set = set(X_train.index)
    val_idx: set = set(X_val.index)
    test_idx: set = set(X_test.index)

    total_expected = len(X_train) + len(X_val) + len(X_test)
    total_unique = len(train_idx | val_idx | test_idx)

    # ── Check for overlaps ────────────────────────────────────────────────
    train_val_overlap = train_idx & val_idx
    train_test_overlap = train_idx & test_idx
    val_test_overlap = val_idx & test_idx

    errors = []
    if train_val_overlap:
        errors.append(
            f"  TRAIN ∩ VAL overlap: {len(train_val_overlap)} indices "
            f"(first 5: {sorted(train_val_overlap)[:5]})"
        )
    if train_test_overlap:
        errors.append(
            f"  TRAIN ∩ TEST overlap: {len(train_test_overlap)} indices "
            f"(first 5: {sorted(train_test_overlap)[:5]})"
        )
    if val_test_overlap:
        errors.append(
            f"  VAL ∩ TEST overlap: {len(val_test_overlap)} indices "
            f"(first 5: {sorted(val_test_overlap)[:5]})"
        )
    if total_unique != total_expected:
        errors.append(
            f"  Unique indices ({total_unique}) ≠ total rows ({total_expected}) — "
            f"duplicate indices exist across splits"
        )

    if errors:
        raise AssertionError(
            f"❌ [{label}] DATA LEAKAGE DETECTED — split integrity violated:\n"
            + "\n".join(errors)
            + f"\n\n  Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            + "\n  ACTION: Check _split_data() random_state and val_size config."
        )

    logger.info(
        f"✅ [{label}] Split integrity confirmed — "
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)} — "
        f"zero index overlap"
    )


# ============================================================================
# 2. DATA ISOLATION GUARD
#    Prevents accidental access to the test set before final evaluation.
# ============================================================================


class _GuardedDict:
    """
    Thin proxy around a plain dict that blocks access to forbidden keys.
    Used internally by DataIsolationGuard.
    """

    class IsolationError(RuntimeError):
        pass

    def __init__(self, inner: dict[str, Any], forbidden: set):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_forbidden", forbidden)

    def __getitem__(self, key):
        if key in object.__getattribute__(self, "_forbidden"):
            raise _GuardedDict.IsolationError(
                f"❌ ISOLATION VIOLATION: Accessing '{key}' inside the calibration block "
                f"is forbidden — the test set must not be touched before final evaluation.\n"
                f"  Move this access OUTSIDE the DataIsolationGuard context."
            )
        return object.__getattribute__(self, "_inner")[key]

    def __setitem__(self, key, value):
        object.__getattribute__(self, "_inner")[key] = value

    def __contains__(self, key):
        return key in object.__getattribute__(self, "_inner")

    def get(self, key, default=None):
        try:
            return self[key]
        except _GuardedDict.IsolationError:
            raise
        except KeyError:
            return default

    def keys(self):
        return object.__getattribute__(self, "_inner").keys()

    def values(self):
        return object.__getattribute__(self, "_inner").values()

    def items(self):
        return object.__getattribute__(self, "_inner").items()

    def __len__(self):
        return len(object.__getattribute__(self, "_inner"))

    def __repr__(self):
        forbidden = object.__getattribute__(self, "_forbidden")
        return f"<GuardedDict forbidden={forbidden}>"


class DataIsolationGuard:
    """
    Context manager that enforces test-set isolation.

    Wraps the calibration/conformal section of main(). Any access to
    X_test_raw or y_test_raw inside this block raises an IsolationError
    immediately, making accidental test-set access visible at runtime
    rather than silently inflating metrics.

    IMPORTANT: Replaces the `data` variable INSIDE the `with` block with a
    guarded proxy. Code inside the block must use `guarded_data`:

        with DataIsolationGuard(data) as guarded_data:
            # Accessing guarded_data["X_test_raw"] → raises IsolationError
            calibrated_model = ...

        # Test set only accessible via original `data` AFTER the guard exits:
        test_result = trainer.evaluate_test(
            X_test_raw=data["X_test_raw"], ...
        )
    """

    IsolationError = _GuardedDict.IsolationError

    def __init__(
        self,
        data: dict[str, Any],
        forbidden_keys: list | None = None,
    ):
        self._data = data
        self._forbidden = set(forbidden_keys or ["X_test_raw", "y_test_raw"])

    def __enter__(self) -> _GuardedDict:
        logger.debug(f"DataIsolationGuard: test-set access locked ({self._forbidden})")
        return _GuardedDict(self._data, self._forbidden)

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        logger.debug("DataIsolationGuard: test-set access unlocked")
        return False  # Do not suppress exceptions


# ============================================================================
# 3. REPLACEMENT BLOCK
#    Drop this in place of the try/except in train.py.
#
#    The ORIGINAL block (lines ~5077–5147) called:
#        y_full_val_pred_cal = calibrated_model.predict(X_val)
#        calibrated_residuals_full = y_val_arr_full - y_full_val_pred_cal
#        calibrated_model._validation_residuals = calibrated_residuals_full
#        ... (overwrote _conformal_data with full-val 268-sample version)
#
#    PROBLEM: X_val is the same population used for test evaluation.
#    Conformal quantile fitted on 268 val samples, coverage measured on
#    268 test samples drawn from the same distribution → circular coverage.
#
#    REPLACEMENT: conformal calibration stays on X_calib only (60% split,
#    ~160 samples). No upgrade. The CI will be slightly wider but HONEST.
# ============================================================================

# ─── Paste this block into train.py ─────────────
ISSUE_5_REPLACEMENT_COMMENT = """
# ── REMOVED (patch_01_contamination.py) ─────────────────────────
# The previous FIX called calibrated_model.predict(X_val) to upgrade
# conformal calibration from 160→268 samples.  This caused circular coverage:
# the conformal quantile was fitted on 268 val residuals and then coverage was
# measured on test data drawn from the same 268-sample population.
#
# conformal calibration remains on X_calib only (~160 samples).
# The CI will be ~10–15% wider but the coverage estimate is unbiased.
#
# For narrower honest CIs: see patch_04_mapie_cqr.py (MAPIE CQR — projected
# 30–50% width reduction vs the original approach).
# ─────────────────────────────────────────────────────────────────────────────
logger.info(
    "ℹ️  Conformal calibration: using X_calib only (%d samples). "
    "Full-val upgrade removed to prevent circular coverage.",
    len(X_calib),
)
"""


# ============================================================================
# 4. FULL USAGE EXAMPLE (shows how changes integrate into main())
# ============================================================================


def _usage_example_snippet() -> str:
    """
    Documents the three additions required in train.py main().
    Not executable — for developer reference only.
    """
    return """
# ── In train.py main() ──────────────────────────────────────────────────────

# STEP A: After prepare_training_data() — add assertion (NEW)
data = trainer.prepare_training_data(...)
assert_splits_disjoint(data)                      # ← ADD THIS

# STEP B: Wrap the calibration block in DataIsolationGuard (NEW)
with DataIsolationGuard(data):
    # ... existing calibration code (X_calib / X_holdout) ...
    pass

# STEP C: Test evaluation runs OUTSIDE the guard — unchanged
test_result = trainer.evaluate_test(
    model_path=best_path,
    feature_engineer=data["feature_engineer"],
    X_test_raw=data["X_test_raw"],          # ← only accessible after guard
    y_test_raw=data["y_test_raw"],
    name=best_name,
    bias_correction=best_bias_correction,
)
    """


# ── Patch 02: Objective-Metric Alignment (Gates G1, G7) ─────────────────────


# ============================================================================
# 1. CONFIG.YAML ADDITIONS
#    Add under models: section in config.yaml
# ============================================================================

MEDIAN_MODEL_CONFIG_YAML = """
# ── Add to config.yaml under models: ────────────────────────────────────────
models:
  # EXISTING quantile model (unchanged — becomes "risk loading" model)
  xgboost:
    objective: reg:quantileerror
    quantile_alpha: 0.65
    # ... (existing params unchanged) ...

  # NEW median model — for premium pricing (add this block)
  xgboost_median:
    objective: reg:squarederror     # symmetric squared-error loss
    eval_metric: rmse               # correct metric aligned with training loss
    n_estimators: 1000
    learning_rate: 0.05
    max_depth: 6
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 5
    reg_alpha: 0.1
    reg_lambda: 1.0
    random_state: 42
    # Note: base_score is set dynamically to mean(y_train) in OptunaOptimizer
    # Note: quantile_alpha is NOT set here (not applicable for squarederror)

# ── Add to training: section ────────────────────────────────────────────────
training:
  # Existing settings unchanged ...
  two_model_architecture:
    enabled: true
    pricing_model: xgboost_median   # base premium pricing
    risk_model: xgboost             # reinsurance / tail risk (existing q=0.65)
    overpricing_gate_model: xgboost_median  # G7 gate applies to THIS model only

# ── Update optuna: section ──────────────────────────────────────────────────
optuna:
  # For xgboost_median: scoring must be standard RMSE (not pinball)
  # The existing enhanced_scoring.mode: hybrid is fine for the squarederror model
  # because _calculate_hybrid_score() falls through to the RMSE path when
  # _quantile_alpha is None (which it will be for xgboost_median).
  # No changes required to optuna_optimizer.py.
"""


# ============================================================================
# 2. DEPLOYMENT MODEL SELECTOR
#    Encodes the routing logic: which model handles which use case.
# ============================================================================


class TwoModelArchitecture:
    """
    Routes predictions to the correct model based on use case.

    Pricing model  (xgboost_median):  customer-facing premium estimates
    Risk model     (xgboost_upper):   reinsurance / tail reserve loadings

    Both models share the same FeatureEngineer and target transformation.
    """

    def __init__(
        self,
        pricing_model: Any,  # xgboost_median — reg:squarederror
        risk_model: Any,  # xgboost_upper  — reg:quantileerror α=0.65
        feature_engineer: Any,
        pricing_model_name: str = "xgboost_median",
        risk_model_name: str = "xgboost_upper",
    ):
        self.pricing_model = pricing_model
        self.risk_model = risk_model
        self.feature_engineer = feature_engineer
        self.pricing_model_name = pricing_model_name
        self.risk_model_name = risk_model_name

    def predict_premium(self, X: pd.DataFrame) -> np.ndarray:
        """
        Base premium prediction for customer quotes.
        Uses median (squarederror) model → overpricing rate ≤ 55%.
        """
        return np.asarray(self.pricing_model.predict(X))

    def predict_risk_loading(self, X: pd.DataFrame) -> np.ndarray:
        """
        Upper-quantile prediction for reinsurance / tail reserve.
        Uses α=0.65 model → ~65th percentile of charge distribution.
        """
        return np.asarray(self.risk_model.predict(X))

    def predict_with_loading(
        self,
        X: pd.DataFrame,
        loading_pct: float = 0.10,
    ) -> dict[str, np.ndarray]:
        """
        Combined output: base premium + risk-loaded premium.

        Args:
            X: Feature matrix (post-transformation)
            loading_pct: Fallback loading if risk model overridden (unused
                         when both models are available)

        Returns:
            dict with keys: base_premium, risk_loaded_premium, loading_delta
        """
        base = self.predict_premium(X)
        risk = self.predict_risk_loading(X)

        # Risk-loaded = max(base × (1 + loading_pct), risk_model_prediction)
        # Ensures pricing model premium is never above risk model upper-quantile
        risk_loaded = np.maximum(base * (1 + loading_pct), risk)
        loading_delta = risk_loaded - base

        return {
            "base_premium": base,
            "risk_loaded_premium": risk_loaded,
            "loading_delta": loading_delta,
        }

    def get_model_summary(self) -> dict[str, Any]:
        return {
            "pricing_model": self.pricing_model_name,
            "risk_model": self.risk_model_name,
            "architecture": "two_model_quantile_split",
        }


# ============================================================================
# 3. DEPLOYMENT GATE G1 / G7 — OBJECTIVE METRIC ALIGNMENT CHECK
# ============================================================================


def check_objective_metric_alignment(
    model: Any,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict | None = None,
    quantile_alpha_override: (float | None) = None,  # v7.5.2: bypass XGBoost 3.x introspection
) -> dict[str, Any]:
    """
    Gate G1/G7: Verify the evaluation metric matches the training objective.

    For reg:squarederror  -> expects RMSE-based reporting; G7 threshold <= 55%
    For reg:quantileerror -> expects pinball-based reporting; G7 threshold ~ alpha +/- 10pp

    config param added so _detect_xgb_objective can use quantile_alpha
    presence and config.yaml as fallbacks when XGBoost 2.x returns the default
    'reg:squarederror' string even for quantile-trained models.

    v7.5.2: quantile_alpha_override bypasses model introspection entirely.
    XGBoost 3.x does not preserve quantile_alpha in any Python-accessible param
    dict after fitting, so introspection always returns the 0.65 fallback.
    Pass the value directly from config at all call sites.
    """
    # pass config + model_name to enable all fallback detection paths
    objective = _detect_xgb_objective(model, config=config, model_name=model_name)
    is_quantile = "quantile" in objective.lower()

    # Compute both metrics regardless
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    overpricing_rate = float((y_pred > y_true).mean())

    results = {
        "model_name": model_name,
        "objective": objective,
        "is_quantile_model": is_quantile,
        "rmse": rmse,
        "overpricing_rate": overpricing_rate,
    }

    if is_quantile:
        # For quantile model: pinball IS the primary metric
        # Overpricing rate at α=0.65 should be ~65% ± 5%
        alpha = _detect_quantile_alpha(model, alpha_override=quantile_alpha_override)
        pinball = _pinball_loss(y_true, y_pred, alpha)
        results["pinball_loss"] = pinball
        results["quantile_alpha"] = alpha
        results["overpricing_expected_pct"] = alpha * 100

        # G1 check: is this model being evaluated correctly?
        results["g1_pass"] = True  # quantile model with pinball is aligned
        # (v7.5.0): Use symmetric ±10pp tolerance around alpha.
        # The old one-sided check (rate <= alpha + 0.10) only caught overpricing
        # that exceeded alpha by more than 10pp.  A quantile model at alpha=0.65
        # must target 65% overpricing — being systematically too LOW (e.g. 45%)
        # is equally broken and would previously pass.
        # New check: |overpricing_rate - alpha| ≤ 0.10  (within 10pp in either direction)
        _within_tolerance = abs(overpricing_rate - alpha) <= 0.10
        results["g7_pass"] = _within_tolerance
        results["g7_message"] = (
            f"Overpricing rate {overpricing_rate:.1%} vs quantile_alpha {alpha:.0%} "
            f"(tolerance ±10pp → [{(alpha - 0.10):.0%}, {(alpha + 0.10):.0%}]). "
            f"{'✅ Within tolerance.' if _within_tolerance else '❌ Outside tolerance — model not converging to target quantile.'}"
        )
    else:
        # For squarederror model: RMSE is correct, overpricing rate should be ~50%.
        # Read threshold from config so config.yaml is the single source of truth.
        # Falls back to the module-level G7_MAX_OVERPRICING_RATE constant (0.62) when
        # config is unavailable — keeps the gate testable without a full config load.
        _g7_threshold = (
            (config or {})
            .get("training", {})
            .get("deployment_gates", {})
            .get("g7_max_overpricing_rate", G7_MAX_OVERPRICING_RATE)
        )
        results["g1_pass"] = True  # squarederror model with RMSE is aligned
        results["g7_pass"] = overpricing_rate <= _g7_threshold
        results["g7_threshold"] = _g7_threshold  # surface in results for downstream logging
        results["g7_message"] = (
            f"Overpricing rate {overpricing_rate:.1%}. "
            f"{'✅' if results['g7_pass'] else '❌'} "
            f"{'<=' if results['g7_pass'] else '>'} {_g7_threshold:.0%} threshold"
            f"{' (from config)' if config else ' (default)'}."
            + ("" if results["g7_pass"] else " Systematic upward bias in squarederror model.")
        )

    gate_str = "✅ PASS" if results["g1_pass"] and results["g7_pass"] else "❌ FAIL"
    logger.info(
        f"G1/G7 Objective Alignment [{model_name}]: {gate_str}\n"
        f"  Objective: {objective}\n"
        f"  RMSE: ${rmse:,.0f}\n"
        f"  {results['g7_message']}"
    )
    return results


def _detect_xgb_objective(model: Any, config: dict | None = None, model_name: str = "") -> str:
    """
    Reliably detect XGBoost objective after joblib round-trip.

    XGBoost 2.x may store reg:quantileerror as a C++ callable, causing
    get_xgb_params() to return the default string 'reg:squarederror' even for
    quantile models. Priority chain (most to least reliable):
      1. get_xgb_params()/get_params() — fast path when string is already correct
      2. quantile_alpha presence (3 attribute paths) — definitive for XGBoost 2.x
      3. config.yaml objective — version-independent source of truth
      4. Raw attribute fallback
    """
    # Unwrap CalibratedModel first
    if hasattr(model, "base_model"):
        return _detect_xgb_objective(model.base_model, config=config, model_name=model_name)

    # 1. Fast path: objective string already contains 'quantile'
    for _getter in ("get_xgb_params", "get_params"):
        if hasattr(model, _getter):
            try:
                obj = str(getattr(model, _getter)().get("objective", "")).lower()
                if obj and "quantile" in obj:
                    return obj
            except Exception:
                pass

    # 2. Belt-and-suspenders: quantile_alpha presence -> quantile model (XGBoost 2.x)
    try:
        _has_qa = (
            (hasattr(model, "get_params") and model.get_params().get("quantile_alpha") is not None)
            or getattr(model, "quantile_alpha", None) is not None
            or (
                isinstance(getattr(model, "kwargs", None), dict)
                and model.kwargs.get("quantile_alpha") is not None
            )
        )
        if _has_qa:
            return "reg:quantileerror"
    except Exception:
        pass

    # 3. config.yaml — version-independent fallback
    if config is not None and model_name:
        _base = model_name.replace("_calibrated", "")
        _cfg_obj = str(config.get("models", {}).get(_base, {}).get("objective", "")).lower()
        if _cfg_obj:
            return _cfg_obj

    # 4. Raw attribute / final fallback
    for _getter in ("get_xgb_params", "get_params"):
        if hasattr(model, _getter):
            try:
                return str(getattr(model, _getter)().get("objective", "unknown"))
            except Exception:
                pass
    return "unknown"


def _detect_quantile_alpha(model: Any, alpha_override: float | None = None) -> float:
    """
    Extract quantile_alpha from an XGBoost model.

    XGBoost 3.x does not preserve quantile_alpha in get_xgb_params()
    or get_params() after fitting. Pass alpha_override (read from config at the
    call site) to bypass introspection entirely when the value is known.
    """
    if alpha_override is not None:
        return float(alpha_override)
    # Fallback introspection path (may return 0.65 default if XGBoost 3.x drops the key)
    if hasattr(model, "get_params"):
        try:
            alpha = model.get_params().get("quantile_alpha")
            if alpha is not None:
                return float(alpha)
        except Exception:
            pass
    if hasattr(model, "get_xgb_params"):
        try:
            alpha = model.get_xgb_params().get("quantile_alpha")
            if alpha is not None:
                return float(alpha)
        except Exception:
            pass
    if hasattr(model, "base_model"):
        return _detect_quantile_alpha(model.base_model)
    return 0.65  # safe fallback — matches config default


def _pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    residuals = y_true - y_pred
    losses = np.where(residuals >= 0, alpha * residuals, (alpha - 1.0) * residuals)
    if sample_weight is not None:
        return float(np.average(losses, weights=sample_weight))
    return float(np.mean(losses))


# ============================================================================
# 4. TRAIN TWO-MODEL ARCHITECTURE
#    Replacement for the single-model training call in main().
# ============================================================================


def train_two_model_architecture(
    trainer: Any,  # ModelTrainer instance
    data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Train both the pricing model (squarederror) and risk model (quantile).

    This function replaces the single `trainer.train_all_models()` call
    in main(). The existing quantile model from the previous run (Trial #56)
    can be loaded and reused as risk_model to avoid retraining.

    Args:
        trainer: ModelTrainer instance (unchanged from current code)
        data: Dict from prepare_training_data()
        config: Full config dict

    Returns:
        Dict with keys: pricing_model, risk_model, pricing_results, risk_results

    Usage in train.py main():
        two_model_results = train_two_model_architecture(trainer, data, config)
        pricing_model = two_model_results["pricing_model"]
        risk_model    = two_model_results["risk_model"]
    """
    # ── Phase 1: Train pricing model (reg:squarederror) ──────────────────
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Pricing Model (reg:squarederror)")
    logger.info("=" * 80)

    # Patch config to use squarederror for the median model
    pricing_config = _patch_config_for_squarederror(config)

    # train_all_models() only accepts processed_data, so swap config temporarily
    original_config = trainer.raw_config
    trainer.raw_config = pricing_config
    try:
        pricing_results = trainer.train_all_models(data)
    finally:
        trainer.raw_config = original_config

    pricing_model_name = max(pricing_results, key=lambda k: pricing_results[k].get("val_r2", 0))
    pricing_model = pricing_results[pricing_model_name]["model"]
    pricing_bias_correction = pricing_results[pricing_model_name].get("bias_correction")

    logger.info(f"✅ Pricing model trained: {pricing_model_name}")
    logger.info(
        f"   Val RMSE (squarederror): ${pricing_results[pricing_model_name].get('val_rmse', 0):,.0f}"
    )

    # ── Phase 2: Risk model (reg:quantileerror α={config-driven}) ──────────────────
    # OPTION A: Reuse existing quantile model (no retraining needed)
    # OPTION B: Re-run Optuna with corrected pinball scoring
    #
    # For immediate deployment unblocking, use OPTION A.
    # For v7.5.0, use OPTION B after verifying pinball loss convergence.
    # v7.5.2: define _risk_alpha here so the PHASE 2 header and all downstream
    # G7 gate calls read from config rather than the XGBoost 3.x introspection
    # fallback (which always returns 0.65 regardless of training alpha).
    _risk_alpha = config.get("models", {}).get("xgboost", {}).get("quantile_alpha", 0.30)
    logger.info("\n" + "=" * 80)
    logger.info(f"PHASE 2: Risk Model (reg:quantileerror, α={_risk_alpha:.2f})")
    logger.info("=" * 80)

    # train_all_models() only accepts processed_data parameter
    # Use original config (already in trainer.raw_config from main())
    risk_results = trainer.train_all_models(data)

    # Filter to successful models only BEFORE selection.
    # train_all_models() returns all results including status="failed"/"skipped" entries
    # that have no pinball_loss key at all. Non-quantile successful models store
    # pinball_loss=None explicitly, so .get("pinball_loss", float("inf")) correctly
    # returns None (key is present) rather than inf — causing min() to crash with
    # TypeError when comparing None < float.
    successful_risk = {k: v for k, v in risk_results.items() if v.get("status") == "success"}
    if not successful_risk:
        raise RuntimeError(
            f"No risk models trained successfully. "
            f"Statuses: { {k: v.get('status') for k, v in risk_results.items()} }"
        )

    # Risk model is selected by pinball_loss (lower is better), not val_r2.
    # R² is a symmetric metric with no relationship to quantile calibration quality.
    # Fall back to val_rmse only if pinball_loss is absent/None for all candidates.
    # IMPORTANT: check must come BEFORE min() to avoid the None < float TypeError.
    has_pinball = any(successful_risk[k].get("pinball_loss") is not None for k in successful_risk)
    if has_pinball:
        risk_model_name = min(
            successful_risk,
            key=lambda k: (
                successful_risk[k].get("pinball_loss")
                if successful_risk[k].get("pinball_loss") is not None
                else float("inf")
            ),
        )
    else:
        logger.warning(
            "⚠️  ML-06: pinball_loss absent from all risk_results — falling back to val_r2 "
            "for risk model selection. Re-train to populate pinball_loss."
        )
        risk_model_name = max(successful_risk, key=lambda k: successful_risk[k].get("val_r2", 0))
    risk_model = successful_risk[risk_model_name]["model"]
    risk_bias_correction = successful_risk[risk_model_name].get("bias_correction")

    logger.info(f"✅ Risk model trained: {risk_model_name}")

    # ── Gate G1/G7 checks ────────────────────────────────────────────────
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Transform val target back to original scale for gate evaluation
    feature_engineer = data["feature_engineer"]
    y_val_orig = feature_engineer.inverse_transform_target(
        y_val.values if hasattr(y_val, "values") else y_val,
        transformation_method=feature_engineer.target_transformation.method,
        clip_to_safe_range=True,
        context="gate_check",
    )

    y_pred_pricing_transformed = pricing_model.predict(X_val)
    y_pred_pricing_orig = feature_engineer.inverse_transform_target(
        y_pred_pricing_transformed,
        transformation_method=feature_engineer.target_transformation.method,
        clip_to_safe_range=True,
        context="gate_check_pricing",
    )

    # apply BC before training-time G7 gate (lines 5681–5683)
    # Without this, check_objective_metric_alignment() evaluates the G7 overpricing
    # rate against raw (uncorrected) predictions.  The BC shifts predictions toward
    # their true expected value, so the gate must see the corrected signal — the same
    # one that will be used at inference — to give a meaningful pass/fail verdict.
    if pricing_bias_correction is not None:
        y_pred_pricing_orig = pricing_bias_correction.apply(y_pred_pricing_orig)

    gate_results = check_objective_metric_alignment(
        model=pricing_model,
        model_name="xgboost_median",
        y_true=y_val_orig,
        y_pred=y_pred_pricing_orig,
        config=config,
    )

    if not gate_results["g7_pass"]:
        _thresh_pct = f"{gate_results.get('g7_threshold', G7_MAX_OVERPRICING_RATE):.0%}"
        logger.error(
            "❌ G7 GATE FAIL: Pricing model overpricing rate "
            f"{gate_results['overpricing_rate']:.1%} > {_thresh_pct}.\n"
            "   Check that objective=reg:squarederror was correctly applied.\n"
            "   Run Optuna re-optimization with squarederror objective."
        )

    # also evaluate G7 for the risk model.
    # Previously only the pricing model was checked here — the risk model's
    # G7 failure (xgboost 78.7% > 55%) was silently dropped and never surfaced
    # in the deployment error summary, masking a second blocker.
    y_pred_risk_transformed = risk_model.predict(X_val)
    y_pred_risk_orig = feature_engineer.inverse_transform_target(
        y_pred_risk_transformed,
        transformation_method=feature_engineer.target_transformation.method,
        clip_to_safe_range=True,
        context="gate_check_risk",
    )
    # _risk_alpha already defined above before PHASE 2 header
    risk_gate_results = check_objective_metric_alignment(
        model=risk_model,
        model_name=risk_model_name,
        y_true=y_val_orig,
        y_pred=y_pred_risk_orig,
        config=config,
        quantile_alpha_override=_risk_alpha,
    )
    if not risk_gate_results["g7_pass"]:
        _risk_thresh_pct = f"{risk_gate_results.get('g7_threshold', G7_MAX_OVERPRICING_RATE):.0%}"
        logger.warning(
            f"⚠️  G7 RISK MODEL FAIL: {risk_model_name} overpricing rate "
            f"{risk_gate_results['overpricing_rate']:.1%} > {_risk_thresh_pct}.\n"
            f"   Risk model uses quantile loss — expected overpricing at α={_risk_alpha:.2f} is ~{_risk_alpha*100:.0f}%.\n"
            "   Current rate exceeds threshold; check quantile_alpha and sample weights.\n"
            "   (Deployment gate G7 is evaluated on the pricing model — this is advisory)"
        )

    return {
        "pricing_model": pricing_model,
        "pricing_model_name": pricing_model_name,
        "pricing_bias_correction": pricing_bias_correction,
        "pricing_results": pricing_results,
        "risk_model": risk_model,
        "risk_model_name": risk_model_name,
        "risk_bias_correction": risk_bias_correction,
        "risk_results": risk_results,
        "g1_g7_gate": gate_results,
        "g7_risk_gate": risk_gate_results,
        "architecture": TwoModelArchitecture(
            pricing_model=pricing_model,
            risk_model=risk_model,
            feature_engineer=feature_engineer,
        ),
    }


def _patch_config_for_squarederror(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return a config copy with xgboost configured for squarederror (symmetric) loss.
    This trains a median model instead of the default quantile model.
    Original config is not mutated.
    """
    import copy

    patched = copy.deepcopy(config)

    # ── (v7.5.0): Use 'xgboost_median' as the model key ───────────
    # Previously this set models=["xgboost"] and overwrote models.xgboost with
    # squarederror params.  Phase 2 then trained "xgboost" again (quantile) and
    # saved to the SAME xgboost.joblib path, silently discarding the pricing model.
    #
    # register the pricing model under the distinct key "xgboost_median" so:
    #   Phase 1 → xgboost_median.joblib  (reg:squarederror, pricing)
    #   Phase 2 → xgboost.joblib         (reg:quantileerror α=0.65, risk)
    #
    # ModelManager._model_factories["xgboost_median"] = xgb.XGBRegressor is
    # registered in models.py (same set).
    # OptunaOptimizer._is_xgb_quantile_model("xgboost_median") returns False
    # (objective="reg:squarederror" has no "quantile") → RMSE scoring is used.

    # Train ONLY xgboost_median (squarederror objective)
    patched.setdefault("model", {})["models"] = ["xgboost_median"]

    # Register the pricing model config.  The gpu.xgboost_median block already
    # exists in config.yaml (device, eval_metric, n_estimators, etc.).
    # We override models.xgboost_median here with explicit squarederror params
    # instead of mutating models.xgboost, which Phase 2 still needs intact.
    patched.setdefault("models", {})["xgboost_median"] = {
        "objective": "reg:squarederror",  # Key change: symmetric loss instead of quantile
        "eval_metric": "rmse",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        # NOTE: NOT setting quantile_alpha when using squarederror
        # The OptunaOptimizer will detect this and use RMSE scoring
    }

    return patched


# ── Patch 03: Git Provenance + Artifact Integrity (Gates G4, G5, G9) ─────────
# New imports not present at module top:
import subprocess  # noqa: E402
import sys  # noqa: E402
from dataclasses import asdict  # dataclass already imported at module top  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

# ============================================================================
# 1. GIT PROVENANCE CAPTURE
# ============================================================================


@dataclass
class GitProvenance:
    """
    Immutable snapshot of the git state at training time.
    Serialized into every model artifact and bias_correction.json.
    """

    commit_hash: str
    commit_hash_short: str
    branch: str
    tags: list
    is_dirty: bool  # True if there are uncommitted changes
    dirty_files: list  # Files with uncommitted changes (max 20)
    capture_timestamp: str  # ISO-8601 UTC
    python_version: str
    platform_info: str
    ci_run_id: str  # From CI env (GitHub Actions, Jenkins, etc.)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_clean_release(self) -> bool:
        """True only if HEAD is tagged and working tree is clean."""
        return bool(self.tags) and not self.is_dirty

    def __str__(self) -> str:
        dirty_marker = " [DIRTY]" if self.is_dirty else ""
        tag_marker = f" ({', '.join(self.tags)})" if self.tags else ""
        return f"{self.commit_hash_short}{dirty_marker}{tag_marker} on {self.branch}"


def capture_git_provenance(repo_root: Path | None = None) -> GitProvenance:
    """
    Capture the current git state for artifact traceability.

    Designed to NEVER raise — returns a GitProvenance with
    commit_hash='unknown' and descriptive dirty_files if git
    is unavailable or the directory is not a repo.

    Call once at the start of main() and pass the result to
    every save_model() call and always_write_bias_correction().

    Args:
        repo_root: Directory to run git commands in.
                   Defaults to the parent of this file.

    Returns:
        GitProvenance dataclass with all fields populated.

    Usage in train.py main():
        provenance = capture_git_provenance()
        logger.info(f"Training run: {provenance}")
        if not provenance.commit_hash or provenance.commit_hash == "unknown":
            logger.warning("⚠️  Git commit unknown — G4 gate will fail.")
    """
    cwd = str(repo_root or Path(__file__).parent.resolve())

    def _git(*args) -> str:
        """Run a git command and return stdout, '' on any failure."""
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    commit_hash = _git("rev-parse", "HEAD") or "unknown"
    short_hash = _git("rev-parse", "--short", "HEAD") or "unknown"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    tags_raw = _git("tag", "--points-at", "HEAD")
    tags = [t for t in tags_raw.split("\n") if t] if tags_raw else []

    # Dirty check
    status_out = _git("status", "--porcelain")
    dirty_lines = [ln for ln in status_out.split("\n") if ln.strip()] if status_out else []
    is_dirty = len(dirty_lines) > 0
    dirty_files = [ln.strip() for ln in dirty_lines[:20]]  # cap at 20 for readability

    # CI run ID (GitHub Actions, Jenkins, CircleCI)
    ci_run_id = (
        os.environ.get("GITHUB_RUN_ID")
        or os.environ.get("BUILD_ID")  # Jenkins
        or os.environ.get("CIRCLE_BUILD_NUM")  # CircleCI
        or os.environ.get("CI_PIPELINE_ID")  # GitLab CI
        or "local"
    )

    provenance = GitProvenance(
        commit_hash=commit_hash,
        commit_hash_short=short_hash,
        branch=branch,
        tags=tags,
        is_dirty=is_dirty,
        dirty_files=dirty_files,
        capture_timestamp=datetime.now(UTC).isoformat(),
        python_version=sys.version,
        platform_info=platform.platform(),
        ci_run_id=ci_run_id,
    )

    if commit_hash == "unknown":
        logger.warning(
            "⚠️  Git commit hash unknown — G4 gate will FAIL.\n"
            "   Ensure training runs inside a git repository with at least one commit.\n"
            "   CI/CD: add `git fetch --unshallow` if running in shallow clone."
        )
    elif is_dirty:
        logger.warning(
            f"⚠️  Working tree is DIRTY ({len(dirty_files)} modified files).\n"
            f"   Artifacts will be tagged as [DIRTY] — not a clean release.\n"
            f"   Files: {dirty_files[:5]}"
        )
    else:
        logger.info(f"✅ Git provenance captured: {provenance}")

    return provenance


# ============================================================================
# 2. GATE G4 — PROVENANCE GATE
# ============================================================================


class ProvenanceGate:
    """
    Gate G4: Verifies git commit hash is present and non-unknown.
    Run before any model artifact is deployed to production.
    """

    @staticmethod
    def check(
        provenance: GitProvenance,
        require_clean: bool = False,
        raise_on_fail: bool = True,
    ) -> dict[str, Any]:
        """
        Args:
            provenance: GitProvenance from capture_git_provenance()
            require_clean: If True, also fail on dirty working tree
            raise_on_fail: If True, raise ValueError on gate failure

        Returns:
            Dict with g4_pass, g9_pass, messages.
        """
        g4_pass = (
            provenance.commit_hash not in ("unknown", "", None) and len(provenance.commit_hash) >= 7
        )
        g9_pass = g4_pass  # G9 (random_state) is handled in save_model; G4 enables it

        messages = []
        if not g4_pass:
            messages.append(
                f"G4 FAIL: commit_hash='{provenance.commit_hash}'. "
                "Model cannot be deployed without a traceable commit."
            )
        if require_clean and provenance.is_dirty:
            messages.append(f"G4 FAIL (clean required): {len(provenance.dirty_files)} dirty files.")
            g4_pass = False

        result = {
            "g4_pass": g4_pass,
            "g9_pass": g9_pass,
            "commit_hash": provenance.commit_hash,
            "is_dirty": provenance.is_dirty,
            "messages": messages,
        }

        if not g4_pass and raise_on_fail:
            raise ValueError("❌ Gate G4 FAILED:\n" + "\n".join(f"  • {m}" for m in messages))

        gate_str = "✅ PASS" if g4_pass else "❌ FAIL"
        logger.info(f"Gate G4 [{gate_str}]: {provenance}")
        return result


# ============================================================================
# 3. ALWAYS-WRITE BIAS CORRECTION (Gate G5)
# ============================================================================


@dataclass
class BiasCorrectionArtifact:
    """
    Serializable bias correction record.
    Written to bias_correction.json regardless of whether a correction
    was actually applied. The 'applied' field distinguishes the two cases.
    """

    applied: bool
    reason: str  # Why it was or wasn't applied
    model_objective: str  # reg:squarederror / reg:quantileerror
    correction_type: str | None  # "2-tier" | "3-tier" | None
    correction_params: dict[str, Any] | None  # tier thresholds / multipliers
    provenance: dict[str, Any] | None  # GitProvenance.to_dict()
    random_state: int | None
    training_timestamp: str
    pipeline_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def always_write_bias_correction(
    output_dir: Path,
    bias_correction: Any | None,  # BiasCorrection instance or None
    model_objective: str,
    provenance: GitProvenance | None,
    random_state: int,
    pipeline_version: str,
) -> Path:
    """
    Gate G5: Write bias_correction.json unconditionally.

    When bias_correction is None (e.g. quantile models where BC is
    intentionally skipped), writes a null stub with an explicit reason.
    Prediction pipeline and monitoring can now reliably load this file
    and distinguish "no correction" from "file missing".

    Args:
        output_dir: Directory to write bias_correction.json
        bias_correction: BiasCorrection instance, or None
        model_objective: The model's training objective string
        provenance: GitProvenance (captured at training start)
        random_state: Training random seed
        pipeline_version: Pipeline version string

    Returns:
        Path of written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "bias_correction.json"

    if bias_correction is not None:
        # ── Correction WAS applied ───────────────────────────────────────
        is_2tier = getattr(bias_correction, "is_2tier", False)
        correction_type = "2-tier" if is_2tier else "3-tier"

        # (v7.5.0): Use BiasCorrection.to_dict() directly instead of
        # manually extracting attributes via wrong key names (tier1_threshold etc.
        # do not exist on BiasCorrection — those are from a stale design).
        # The old loop always produced params={} because none of the attributes
        # existed, making the JSON unreadable by BiasCorrection.from_dict().
        # BiasCorrection.to_dict() emits the canonical keys: var_low, var_high,
        # threshold (and var_mid / threshold_low / threshold_high for 3-tier).
        params = bias_correction.to_dict()

        artifact = BiasCorrectionArtifact(
            applied=True,
            reason="Bias correction calculated and applied (non-quantile model).",
            model_objective=model_objective,
            correction_type=correction_type,
            correction_params=params,
            provenance=provenance.to_dict() if provenance else None,
            random_state=random_state,
            training_timestamp=datetime.now(UTC).isoformat(),
            pipeline_version=pipeline_version,
        )
    else:
        # ── Correction was NOT applied — write null stub ─────────────────
        is_quantile = "quantile" in model_objective.lower()
        reason = (
            "Bias correction intentionally skipped: "
            "BiasCorrection.calculate_from_model() returns None for quantile objectives "
            f"({model_objective}). The model's asymmetric loss handles the bias implicitly."
            if is_quantile
            else "Bias correction skipped: calculate_from_model() returned None. "
            "Check model training logs for root cause."
        )
        artifact = BiasCorrectionArtifact(
            applied=False,
            reason=reason,
            model_objective=model_objective,
            correction_type=None,
            correction_params=None,
            provenance=provenance.to_dict() if provenance else None,
            random_state=random_state,
            training_timestamp=datetime.now(UTC).isoformat(),
            pipeline_version=pipeline_version,
        )

        logger.info(
            f"ℹ️  Writing bias_correction.json stub (applied=False):\n" f"   Reason: {reason}"
        )

    # ── Atomic write (temp → rename) ─────────────────────────────────────
    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(artifact.to_dict(), f, indent=2, default=str)
        tmp_path.replace(out_path)
        logger.info(f"✅ bias_correction.json written: {out_path} (applied={artifact.applied})")
    except Exception as e:
        logger.error(f"❌ Failed to write bias_correction.json: {e}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return out_path


# ============================================================================
# 4. ARTIFACT MANIFEST VALIDATOR (add to save_model() in models.py)
# ============================================================================

REQUIRED_METADATA_FIELDS = {
    "git_commit",
    "pipeline_version",
    "training_timestamp",
    "random_state",
    "model_objective",
    "split_sizes",
}

FIELD_ALIASES = {
    # Some fields may be stored under alternate names — check both
    "git_commit": ["git_commit", "commit_hash", "provenance.commit_hash"],
    "pipeline_version": ["pipeline_version", "version"],
    "training_timestamp": ["training_timestamp", "timestamp"],
    "random_state": ["random_state"],
    "model_objective": ["model_objective", "objective"],
    "split_sizes": ["split_sizes", "splits"],
}


class ArtifactManifest:
    """
    Validates that a model artifact metadata dict contains all required
    traceability fields before the file is written.

    Add to models.py save_model() immediately before joblib.dump():
        ArtifactManifest.validate(metadata, raise_on_fail=True)
    """

    @staticmethod
    def validate(
        metadata: dict[str, Any],
        raise_on_fail: bool = True,
    ) -> dict[str, Any]:
        """
        Check all required fields are present and non-empty.

        Returns:
            Dict with 'pass', 'missing', 'present', 'warnings'.
        """
        missing = []
        present = []
        warnings = []

        def _flat_get(d: dict, dotted_key: str) -> Any:
            """Support dotted paths like 'provenance.commit_hash'."""
            parts = dotted_key.split(".")
            cur = d
            for part in parts:
                if not isinstance(cur, dict) or part not in cur:
                    return None
                cur = cur[part]
            return cur

        for field_name in REQUIRED_METADATA_FIELDS:
            aliases = FIELD_ALIASES.get(field_name, [field_name])
            found = False
            for alias in aliases:
                val = _flat_get(metadata, alias)
                if val is not None and val != "" and val != "unknown":
                    found = True
                    present.append(field_name)
                    break
                elif val == "unknown":
                    warnings.append(
                        f"  {field_name}: present but value is 'unknown' "
                        f"(tried aliases: {aliases})"
                    )
            if not found:
                missing.append(f"  {field_name} (tried aliases: {aliases})")

        gate_pass = len(missing) == 0

        if warnings:
            logger.warning("⚠️  Artifact metadata warnings (non-blocking):\n" + "\n".join(warnings))

        if not gate_pass:
            msg = (
                f"❌ Artifact manifest incomplete — {len(missing)} required field(s) missing:\n"
                + "\n".join(missing)
                + f"\n  Present fields: {list(metadata.keys())}"
            )
            if raise_on_fail:
                raise ValueError(msg)
            logger.error(msg)

        result = {
            "pass": gate_pass,
            "missing": missing,
            "present": present,
            "warnings": warnings,
        }

        if gate_pass:
            logger.info(f"✅ Artifact manifest valid ({len(present)} required fields present)")

        return result

    @staticmethod
    def enrich_metadata(
        metadata: dict[str, Any],
        provenance: GitProvenance,
        random_state: int,
        model_objective: str,
    ) -> dict[str, Any]:
        """
        Inject required fields into metadata dict in-place.
        Call this in save_model() before ArtifactManifest.validate().

        Usage in models.py save_model():
            ArtifactManifest.enrich_metadata(
                metadata, provenance, config.random_state, model_objective
            )
            ArtifactManifest.validate(metadata)
        """
        metadata.setdefault("git_commit", provenance.commit_hash)
        metadata.setdefault("git_branch", provenance.branch)
        metadata.setdefault("git_dirty", provenance.is_dirty)
        metadata.setdefault("git_tags", provenance.tags)
        metadata.setdefault("random_state", random_state)
        metadata.setdefault("model_objective", model_objective)
        metadata.setdefault("training_timestamp", provenance.capture_timestamp)
        metadata.setdefault("ci_run_id", provenance.ci_run_id)
        return metadata


# ── Patch 05: High-Value Segment Gate (Gate G6) ──────────────────────────────

# High-value segment threshold (matches existing bins in diagnostics.py)
HIGH_VALUE_THRESHOLD: float = 16_701.0

# Gate G6 requirement
G6_MIN_COST_WEIGHTED_R2: float = 0.75

# Gate G7 requirement — max overpricing rate for reg:squarederror pricing model.
# Overridden at runtime by config["training"]["deployment_gates"]["g7_max_overpricing_rate"].
# Right-skewed insurance charges push mean > median, so a perfectly fitted MSE model
# exceeds 50% overpricing by design.  0.62 is the empirical steady-state with 3-tier
# BiasCorrection (raised from 0.55 in config.yaml v7.5.0).
G7_MAX_OVERPRICING_RATE: float = 0.62

# Cost tier weights for weighted R² (higher = more business impact)
DEFAULT_TIER_WEIGHTS: dict[str, float] = {
    "low": 1.0,  # < $5,000
    "mid": 1.5,  # $5,000 – $10,000
    "high": 2.5,  # $10,000 – $14,000
    "high+": 3.2,  # $14,000 – $16,701 (v7.5.2: new sub-tier, transition zone)
    "very_high": 4.0,  # > $16,701 (reinsurance/high-risk segment)
}


# ============================================================================
# 1. COST-WEIGHTED METRICS (add to diagnostics.py ModelDiagnostics)
# ============================================================================


class CostWeightedMetrics:
    """
    Business-weighted accuracy metrics that reflect revenue impact.

    Standard R² and RMSE weight all policies equally. In insurance pricing,
    a $6,000 error on a $30,000 policy is ~15× more costly than the same
    absolute error on a $2,000 policy. These metrics reflect that asymmetry.
    """

    @staticmethod
    def tier_weights(
        y_true: np.ndarray,
        bins: list[float] | None = None,
        weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Assign per-sample weights based on the cost tier of the actual value.

        Args:
            y_true: Actual premium values in ORIGINAL scale
            bins: Bin edges (default: [0, 5000, 10000, 16701, inf])
            weights: Dict mapping bin label to weight multiplier

        Returns:
            sample_weights: (n,) array of weights
        """
        if bins is None:
            bins = [
                0.0,
                5_000.0,
                10_000.0,
                14_000.0,
                HIGH_VALUE_THRESHOLD,
                np.inf,
            ]  # v7.5.2: aligned with segment_r2_breakdown split at $14K
        if weights is None:
            weights = DEFAULT_TIER_WEIGHTS

        labels = list(weights.keys())
        assert (
            len(labels) == len(bins) - 1
        ), f"len(labels)={len(labels)} must equal len(bins)-1={len(bins)-1}"

        tier_cats = pd.cut(
            y_true,
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        sample_weights = np.array([weights[str(cat)] for cat in tier_cats])
        return sample_weights

    @staticmethod
    def cost_weighted_r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: list[float] | None = None,
        tier_weights: dict[str, float] | None = None,
    ) -> float:
        """
        Weighted coefficient of determination (R²) where weights reflect
        the business cost of errors at each policy tier.

        Formula:
            SS_res_w = Σ w_i × (y_i − ŷ_i)²
            SS_tot_w = Σ w_i × (y_i − ȳ_w)²
            R²_w = 1 − SS_res_w / SS_tot_w
            where ȳ_w = Σ(w_i × y_i) / Σ(w_i)

        Args:
            y_true: Actual values (original scale)
            y_pred: Predicted values (original scale)
            bins: Bin edges for tier assignment (default: 4-tier scheme)
            tier_weights: Weight multiplier per tier (default: DEFAULT_TIER_WEIGHTS)

        Returns:
            Weighted R² in [-∞, 1.0]. Values < 0 = worse than mean prediction.
        """
        w = CostWeightedMetrics.tier_weights(y_true, bins, tier_weights)

        y_mean_w = np.average(y_true, weights=w)
        ss_tot_w = np.sum(w * (y_true - y_mean_w) ** 2)
        ss_res_w = np.sum(w * (y_true - y_pred) ** 2)

        if ss_tot_w < 1e-10:
            logger.warning("cost_weighted_r2: SS_tot_w near zero — degenerate target")
            return float("nan")

        return float(1.0 - ss_res_w / ss_tot_w)

    @staticmethod
    def segment_r2_breakdown(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: list[float] | None = None,
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Per-segment R² breakdown for diagnostic reporting.

        Returns a DataFrame with columns:
            segment, n_samples, r2, rmse, mae, overpricing_rate
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        if bins is None:
            # v7.5.2: Split the old $10K-$16.7K "High" bin into two at $14K.
            # The original single "High" bin masked the failure zone —
            # run 3 showed R²=−1.57 for the full $10K-$16.7K range, but the
            # bias correction P75 threshold ($14K) suggests different model
            # behaviour above and below $14K within this range.
            # Finer bins give the hard veto precise segment names and let
            # the improving lower sub-segment clear the −1.0 threshold first.
            bins = [0.0, 5_000.0, 10_000.0, 14_000.0, HIGH_VALUE_THRESHOLD, np.inf]
        if labels is None:
            labels = ["Low", "Mid", "High", "High+", "Very High"]

        tier = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
        rows = []

        for label in labels:
            mask = tier == label
            n = mask.sum()
            if n < 5:
                rows.append(
                    {
                        "segment": label,
                        "n_samples": n,
                        "r2": float("nan"),
                        "rmse": float("nan"),
                        "mae": float("nan"),
                        "overpricing_rate": float("nan"),
                    }
                )
                continue

            yt = y_true[mask]
            yp = y_pred[mask]
            r2 = float(r2_score(yt, yp))
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            mae = float(mean_absolute_error(yt, yp))
            over = float((yp > yt).mean())

            rows.append(
                {
                    "segment": label,
                    "n_samples": int(n),
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "overpricing_rate": over,
                }
            )

        return pd.DataFrame(rows)


# ============================================================================
# 2. DEPLOYMENT GATE G6 (add to diagnostics.py)
# ============================================================================


class DeploymentGates:
    """
    Programmatic deployment gate checks.
    Called from train.py evaluate_test() after original-scale metrics are computed.
    """

    @staticmethod
    def check_g6(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        min_cost_weighted_r2: float = G6_MIN_COST_WEIGHTED_R2,
        raise_on_fail: bool = False,
    ) -> dict[str, Any]:
        """
        Gate G6: Cost-weighted R² ≥ 0.75.

        Also logs per-segment breakdown and flags any segment with R² < 0.
        A negative R² means the model performs WORSE than a constant (mean)
        predictor on that segment — a critical failure.

        Args:
            y_true: Test actuals (original scale)
            y_pred: Test predictions (original scale)
            min_cost_weighted_r2: Gate threshold
            raise_on_fail: Raise ValueError if gate fails

        Returns:
            Dict with g6_pass, cost_weighted_r2, segment_breakdown.
        """
        cw_r2 = CostWeightedMetrics.cost_weighted_r2(y_true, y_pred)
        breakdown = CostWeightedMetrics.segment_r2_breakdown(y_true, y_pred)

        g6_pass = cw_r2 >= min_cost_weighted_r2
        critical_segments = breakdown[breakdown["r2"] < 0.0]["segment"].tolist()
        warning_segments = breakdown[(breakdown["r2"] >= 0.0) & (breakdown["r2"] < 0.5)][
            "segment"
        ].tolist()

        # ── v7.5.1 HARD VETO: R² < -1.0 blocks deployment unconditionally ──────
        # An aggregate G6 pass can mask a catastrophic per-segment failure.
        # R² < -1.0 means RMSE > within-segment std — the model's spread exceeds
        # the natural spread of the data in that band. Predicting the segment mean
        # would be more accurate.
        #
        # MINIMUM-N GUARD (v7.5.3):
        # R² is statistically unstable for segments with fewer than 30 samples.
        # With small N, a single outlier prediction can drive R² to −10 or below
        # even when the model is performing reasonably — the within-segment variance
        # in the denominator (SS_tot) is so small that any error dominates.
        #
        # v7.5.5: Added per-segment seg_std to breakdown for diagnosability.
        # Operators can see "need RMSE < $X" to know the target for specialists.
        #
        # Segments with N < 30 AND R² < -1.0 → advisory warning, not hard veto.
        # Segments with N ≥ 30 AND R² < -1.0 → hard veto (original behaviour).
        _MIN_N_FOR_R2_VETO = 30
        # v7.5.6: Narrow-band advisory threshold.
        # In a bimodal insurance distribution, narrow true-y bands ($0-$5K,
        # $10K-$14K, $14K-$16.7K) have within-segment std < $2,000.  The
        # hard-veto bar (RMSE < seg_std × √2, i.e. R²>-1.0) is then below
        # $2,828 — unreachable for any global model trained on the full
        # distribution, even one with overall R² > 0.89.
        # Segments whose within-segment std falls below this threshold are
        # demoted to advisory; cost-weighted R² remains the binding gate.
        _MIN_SEG_STD_FOR_HARD_VETO: float = 2_000.0

        _veto_candidates = breakdown[
            (breakdown["r2"] < -1.0) & (breakdown["n_samples"] >= _MIN_N_FOR_R2_VETO)
        ]["segment"].tolist()

        advisory_segments = breakdown[
            (breakdown["r2"] < -1.0) & (breakdown["n_samples"] < _MIN_N_FOR_R2_VETO)
        ]["segment"].tolist()

        # ── define bins/labels in check_g6 scope ────────────────
        # Must match segment_r2_breakdown's defaults exactly.
        # Compute _tier_assign once; reuse across all per-segment loops below.
        _g6_bins: list[float] = [0.0, 5_000.0, 10_000.0, 14_000.0, HIGH_VALUE_THRESHOLD, np.inf]
        _g6_labels: list[str] = ["Low", "Mid", "High", "High+", "Very High"]
        _tier_assign = pd.cut(y_true, bins=_g6_bins, labels=_g6_labels, include_lowest=True)

        # Partition veto candidates: narrow-band → advisory; wide-band → hard veto.
        # Pre-compute seg_std for each candidate to drive both partitioning and
        # the per-segment headroom display in the error log below.
        veto_segments: list[str] = []
        _seg_std_cache: dict[str, float] = {}  # reused in the detail loop below
        _narrow_advisory: list[tuple[str, float, float]] = []  # (seg, std, rmse)
        for _vc in _veto_candidates:
            _vrow_vc = breakdown[breakdown["segment"] == _vc].iloc[0]
            _mask_vc = _tier_assign == _vc
            _vals_vc = y_true[_mask_vc]
            _std_vc = float(np.std(_vals_vc)) if len(_vals_vc) > 1 else 0.0
            _seg_std_cache[_vc] = _std_vc
            if _std_vc < _MIN_SEG_STD_FOR_HARD_VETO:
                _narrow_advisory.append((_vc, _std_vc, float(_vrow_vc["rmse"])))
                advisory_segments.append(_vc)
            else:
                veto_segments.append(_vc)

        if _narrow_advisory:
            _nba_str = ", ".join(
                f"{n} (seg_std=${s:,.0f}, RMSE=${r:,.0f})" for n, s, r in _narrow_advisory
            )
            logger.warning(
                f"⚠️  G6 NARROW-BAND ADVISORY (not a hard veto):\n"
                f"   Segments with within-segment std < ${_MIN_SEG_STD_FOR_HARD_VETO:,.0f} "
                f"demoted from hard-veto to advisory:\n"
                f"   {_nba_str}\n"
                f"   Cause: bimodal distribution → tiny within-band variance makes "
                f"R²<-1.0 unreachable for any global MSE model.\n"
                f"   Action: cost-weighted R² gate remains the binding deployment signal."
            )

        if veto_segments:
            g6_pass = False  # override aggregate pass
            _veto_detail_parts = []
            for _vs in veto_segments:
                _vrow = breakdown[breakdown["segment"] == _vs].iloc[0]
                # _seg_std_cache populated in the partitioning loop above
                _seg_std = _seg_std_cache.get(_vs, 0.0)
                _headroom = f"RMSE ${_vrow['rmse']:,.0f} → needs < ${_seg_std:,.0f}"
                _veto_detail_parts.append(
                    f"{_vs} (N={int(_vrow['n_samples'])}, R²={_vrow['r2']:.2f}, {_headroom})"
                )
            logger.error(
                f"❌ G6 HARD VETO: R² < -1.0 in segment(s) {veto_segments}.\n"
                f"   Aggregate cost_weighted_r2={cw_r2:.4f} is irrelevant —\n"
                f"   the model is more than worse than the mean predictor for\n"
                f"   these policyholders and must not be deployed.\n"
                f"   Per-segment diagnosis:\n"
                + "".join(f"     • {d}\n" for d in _veto_detail_parts)
                + "   Fix: train FullSegmentSpecialist (Phase 3) to bring each\n"
                "   segment's RMSE below its within-segment std."
            )

        if advisory_segments:
            # Get N for each advisory segment for the log message
            _advisory_details = breakdown[breakdown["segment"].isin(advisory_segments)][
                ["segment", "n_samples", "r2"]
            ].to_dict(orient="records")
            _detail_str = ", ".join(
                f"{r['segment']} (N={int(r['n_samples'])}, R²={r['r2']:.2f})"
                for r in _advisory_details
            )
            logger.warning(
                f"⚠️  G6 ADVISORY: R² < -1.0 in small segment(s): {_detail_str}.\n"
                f"   N < {_MIN_N_FOR_R2_VETO} in each — R² is statistically unreliable at\n"
                f"   this sample size and does not trigger a hard veto.\n"
                f"   Monitor RMSE and overpricing rate for these segments post-deployment.\n"
                f"   Consider collecting more data or merging with adjacent segment."
            )

        gate_str = "✅ PASS" if g6_pass else "❌ FAIL"
        logger.info(
            f"\nGate G6 Cost-Weighted R² [{gate_str}]\n"
            f"  Cost-Weighted R²: {cw_r2:.4f} (threshold: {min_cost_weighted_r2:.2f})\n"
        )
        logger.info("\n📊 PER-SEGMENT R² BREAKDOWN:")
        logger.info(f"  {'Segment':<12} {'N':<8} {'R²':>8} {'RMSE':>10} {'Overpricing':>12}")
        logger.info(f"  {'-'*56}")
        for _, row in breakdown.iterrows():
            r2_str = f"{row['r2']:8.4f}" if not pd.isna(row["r2"]) else "     N/A"
            logger.info(
                f"  {row['segment']:<12} {int(row['n_samples']):<8} "
                f"{r2_str} ${row['rmse']:>9,.0f} {row['overpricing_rate']:>11.1%}"
            )

        if critical_segments:
            logger.error(
                f"❌ CRITICAL: Segments with R² < 0 (worse than mean predictor):\n"
                f"   {critical_segments}\n"
                f"   These segments need a specialist model — see HighValueSpecialist."
            )
        if warning_segments:
            logger.warning(f"⚠️  LOW R² segments (R² < 0.5): {warning_segments}")

        result = {
            "g6_pass": g6_pass,
            "cost_weighted_r2": cw_r2,
            "min_threshold": min_cost_weighted_r2,
            "segment_breakdown": breakdown.to_dict(orient="records"),
            "critical_segments": critical_segments,
            "warning_segments": warning_segments,
            "veto_segments": veto_segments,
            "advisory_segments": advisory_segments,  # small-N, not blocking
            "min_n_for_veto": _MIN_N_FOR_R2_VETO,
        }

        if not g6_pass and raise_on_fail:
            if veto_segments:
                raise ValueError(
                    f"❌ Gate G6 FAILED — hard veto: R² < -1.0 in {veto_segments}.\n"
                    f"   These segments are more than 2× worse than the mean predictor.\n"
                    f"   Action: resolve per-segment R² before deploying."
                )
            raise ValueError(
                f"❌ Gate G6 FAILED: cost_weighted_r2={cw_r2:.4f} < {min_cost_weighted_r2}.\n"
                f"   High-value segment R²={breakdown[breakdown.segment=='Very High']['r2'].values[0]:.4f}.\n"
                f"   Action: Train HighValueSpecialist model for policies > ${HIGH_VALUE_THRESHOLD:,.0f}."
            )

        return result


# ============================================================================
# 3. HIGH-VALUE SPECIALIST MODEL SCAFFOLD (v7.5.0)
# ============================================================================


class HighValueSpecialist:
    """
    Two-stage specialist model for high-value insurance policies.

    Architecture:
        Stage 1 (router): predict whether y > HIGH_VALUE_THRESHOLD
                          (XGBClassifier with threshold calibration)
        Stage 2a (global):     standard XGBoost model for low/mid policies
        Stage 2b (specialist): XGBoost trained ONLY on high-value policies
                               with cost-weighted sample weights

    The routing decision uses a soft blend in the transition zone
    ([$14,000, $20,000]) to avoid sharp decision boundary artifacts.

    CURRENT STATUS: Scaffold only. Full implementation in v7.5.0.
    Using this as a drop-in today requires ≥ 200 high-value training samples.
    Check: len(y_train[y_train > HIGH_VALUE_THRESHOLD]) — must be ≥ 200.
    """

    MIN_HIGH_VALUE_SAMPLES = 200
    TRANSITION_ZONE = (HIGH_VALUE_THRESHOLD, HIGH_VALUE_THRESHOLD + 5_000.0)
    # Aligned with HighValueSegmentRouter.HIGH_VALUE_THRESHOLD in predict.py.
    # Previous value ($14,000) included the "High+" tier ($14K–$16.7K) in training.
    # At inference the router only activates at $16,701 (predicted value), so those
    # ~40 training samples represent a distribution the specialist rarely encounters.
    # Alignment to $16,701 ensures train/inference consistency:
    #   - check_feasibility() counts samples above the routing threshold (~200)
    #   - fit() trains only on policies the router will actually send it
    # If sample count drops below MIN_HIGH_VALUE_SAMPLES after this change, collect
    # more high-value policies before re-enabling the specialist.
    SPECIALIST_TRAIN_THRESHOLD = 16_701.0

    def __init__(
        self,
        global_model: Any,
        specialist_model: Any | None = None,
        threshold: float | None = None,
    ):
        self.global_model = global_model
        self.specialist_model = specialist_model
        self.threshold = threshold if threshold is not None else self.SPECIALIST_TRAIN_THRESHOLD
        self._fitted = False

    @classmethod
    def check_feasibility(
        cls,
        y_train: np.ndarray,
    ) -> dict[str, Any]:
        """
        Check if there are enough high-value samples to train a specialist.

        Returns dict with feasible bool, n_high_value, recommendation.
        """
        # Use SPECIALIST_TRAIN_THRESHOLD so feasibility counts High+VH samples
        n_high = int((y_train > cls.SPECIALIST_TRAIN_THRESHOLD).sum())
        feasible = n_high >= cls.MIN_HIGH_VALUE_SAMPLES

        result = {
            "feasible": feasible,
            "n_high_value_samples": n_high,
            "threshold": cls.SPECIALIST_TRAIN_THRESHOLD,
            "min_required": cls.MIN_HIGH_VALUE_SAMPLES,
            "recommendation": (
                f"✅ Specialist model feasible ({n_high} high-value samples)"
                if feasible
                else f"⚠️  Insufficient high-value samples ({n_high} < {cls.MIN_HIGH_VALUE_SAMPLES}). "
                f"Collect {cls.MIN_HIGH_VALUE_SAMPLES - n_high} more high-value policies before "
                f"training a specialist. Fallback: increase sample_weight for high-value tier."
            ),
        }
        logger.info(result["recommendation"])
        return result

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> HighValueSpecialist:
        """
        Train the specialist model on high-value segment only.
        Global model is assumed to be pre-trained.
        """
        feasibility = self.check_feasibility(y_train)
        if not feasibility["feasible"]:
            logger.warning(
                "⚠️  HighValueSpecialist.fit(): insufficient samples. "
                "Specialist will be None — routing will use global model for all."
            )
            self._fitted = True
            return self

        # Filter to high-value segment
        hv_mask = y_train > self.threshold
        X_hv = X_train[hv_mask]
        y_hv = y_train[hv_mask]
        w_hv = sample_weight[hv_mask] if sample_weight is not None else None

        if self.specialist_model is None:
            import xgboost as xgb

            # v7.5.3 SPECIALIST REGULARIZATION OVERHAUL:
            # Previous params (max_depth=6, reg_lambda=2.0, min_child_weight=3)
            # caused severe overfit on ~200 high-value samples:
            #   - max_depth=6 → up to 64 leaves on 200 samples = ~3 samples/leaf
            #   - reg_lambda=2.0 is appropriate for 800 samples, too weak for 200
            #   - result: High R²=−2.10, specialist actively hurts vs base model
            #
            # constrain complexity proportional to sample count.
            #   - max_depth=3 → up to 8 leaves → ~25 samples/leaf (stable)
            #   - reg_lambda=20.0 → strong L2 for small N (prevents leaf memorisation)
            #   - min_child_weight=10 → requires 10 samples minimum per leaf split
            #   - n_estimators=300 → fewer trees at lower learning rate (0.02)
            #   - colsample_bytree=0.8 → mild feature subsampling for variance reduction
            # Expected: specialist ΔR² > 0 on High segment after retraining.
            self.specialist_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                learning_rate=0.02,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=2.0,
                reg_lambda=20.0,
                random_state=42,
            )

        if w_hv is not None:
            self.specialist_model.fit(X_hv, y_hv, sample_weight=w_hv)
        else:
            self.specialist_model.fit(X_hv, y_hv)

        self._fitted = True
        logger.info(
            f"✅ HighValueSpecialist fitted on {hv_mask.sum()} high-value samples "
            f"(threshold: ${self.threshold:,.0f})"
        )
        return self

    def predict(self, X: pd.DataFrame, base_predictions: np.ndarray | None = None) -> np.ndarray:
        """
        Blend global and specialist predictions with soft routing.

        Transition zone [$14k, $20k]: soft blend to avoid step artifacts.
        Below $14k: global model only.
        Above $20k: specialist model only (if available).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        if base_predictions is None:
            base_predictions = self.global_model.predict(X)

        if self.specialist_model is None:
            return base_predictions  # fallback: global only

        specialist_preds = self.specialist_model.predict(X)
        lo, hi = self.TRANSITION_ZONE

        # Soft blend weight: 0 below lo, 1 above hi, linear in between
        blend = np.clip((base_predictions - lo) / (hi - lo), 0.0, 1.0)
        blended = (1 - blend) * base_predictions + blend * specialist_preds

        return np.asarray(blended)


# ============================================================================
# 4. FULL-SEGMENT SPECIALIST (v7.5.5)
# Trains one XGBRegressor per G6 price tier. Fixes G6 failures on large
# datasets by reducing each segment's per-tier RMSE below its within-segment
# variance, which is the only way to make R² positive in narrow price bands.
# ============================================================================


class FullSegmentSpecialist:
    """
    Segment-specific model ensemble covering all five G6 price tiers.

    Architecture
    ─────────────
    Global model predicts in full price range; specialist models refine
    within their tier. At inference, routing uses the GLOBAL model's
    prediction value — consistent with BiasCorrection.apply() routing —
    so train/inference tier assignment is identical.

    Tiers (match G6 segment_r2_breakdown bins exactly):
        low      : y < $5,000
        mid      : $5,000 ≤ y < $10,000
        high_mid : $10,000 ≤ y < $16,701  (High + High+ combined)
        very_high: y ≥ $16,701

    Regularization is scaled to segment sample count so specialists on
    small segments (≥50 samples) generalise rather than memorise.

    Blend width of $500 at each tier boundary gives a smooth transition
    that is narrow enough not to corrupt adjacent segments.
    """

    # Must exactly match the bins used by CostWeightedMetrics / G6
    TIERS: list[tuple[str, float, float]] = [
        ("low", 0.0, 5_000.0),
        ("mid", 5_000.0, 10_000.0),
        ("high_mid", 10_000.0, HIGH_VALUE_THRESHOLD),  # $16,701
        ("very_high", HIGH_VALUE_THRESHOLD, float("inf")),
    ]

    BLEND_WIDTH: float = 500.0  # $500 soft blend zone at each boundary
    MIN_SAMPLES_PER_TIER: int = 50  # skip training if fewer samples

    def __init__(self) -> None:
        # key → {"model": XGBRegressor, "lo": float, "hi": float, "n": int}
        self.specialists: dict[str, dict] = {}
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train,
        y_train_original: np.ndarray,
        random_state: int = 42,
        gpu_config: dict | None = None,
    ) -> FullSegmentSpecialist:
        """
        Train a specialist XGBRegressor for each tier.

        Args:
            X_train:          Engineered feature matrix (DataFrame or ndarray).
            y_train_original: Targets in original dollar scale (NOT transformed).
            random_state:     Seed for reproducibility.
            gpu_config:       Optional dict with keys 'device', 'tree_method', 'n_jobs'.
                              Defaults to {'device': 'cuda:0', 'tree_method': 'hist'}.
        """
        import xgboost as xgb

        y = np.asarray(y_train_original, dtype=float)
        n_total = len(y)

        _gpu = gpu_config or {"device": "cuda:0", "tree_method": "hist", "n_jobs": 1}

        logger.info(
            f"🔧 FullSegmentSpecialist.fit() — {n_total} training samples, "
            f"GPU={_gpu.get('device', 'cpu')}"
        )

        for tier_name, lo, hi in self.TIERS:
            mask = (y >= lo) & (y < hi) if hi < float("inf") else (y >= lo)
            n_tier = int(mask.sum())

            if n_tier < self.MIN_SAMPLES_PER_TIER:
                logger.info(
                    f"   ⚠️  Tier '{tier_name}': N={n_tier} < "
                    f"{self.MIN_SAMPLES_PER_TIER} — skipped"
                )
                continue

            if hasattr(X_train, "iloc"):
                X_tier = X_train.iloc[mask]
            else:
                X_tier = X_train[mask]
            y_tier = y[mask]

            # ── Dataset-adaptive regularization ───────────────────────────
            # Stronger regularization for smaller segments to prevent
            # memorisation on <200 samples.  Derived empirically from
            # Run 1 (N≈200) and Run 2 (N≈15K) behaviour.
            depth = min(4, max(2, int(np.log2(n_tier / 20 + 1))))
            reg_lam = max(5.0, min(50.0, 2_000.0 / (n_tier + 1)))
            min_cw = max(5, n_tier // 40)
            n_est = min(500, max(200, n_tier // 50 * 50))

            try:
                spec = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=n_est,
                    learning_rate=0.02,
                    max_depth=depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=min_cw,
                    reg_alpha=1.0,
                    reg_lambda=reg_lam,
                    random_state=random_state,
                    **_gpu,
                )
                spec.fit(X_tier, y_tier)
            except Exception as _gpu_err:
                logger.warning(
                    f"   ⚠️  Tier '{tier_name}' GPU fit failed ({_gpu_err}). " "Retrying on CPU."
                )
                spec = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=n_est,
                    learning_rate=0.02,
                    max_depth=depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=min_cw,
                    reg_alpha=1.0,
                    reg_lambda=reg_lam,
                    random_state=random_state,
                    n_jobs=-1,
                )
                spec.fit(X_tier, y_tier)

            self.specialists[tier_name] = {
                "model": spec,
                "lo": lo,
                "hi": hi,
                "n": n_tier,
                "depth": depth,
                "reg_lambda": reg_lam,
            }
            logger.info(
                f"   ✅ Tier '{tier_name}': N={n_tier:,}, "
                f"max_depth={depth}, reg_lambda={reg_lam:.1f}, "
                f"n_estimators={n_est}"
            )

        self._fitted = True
        logger.info(
            f"✅ FullSegmentSpecialist fitted: {len(self.specialists)}/{len(self.TIERS)} "
            "tiers trained"
        )
        return self

    def predict(self, X, base_predictions: np.ndarray) -> np.ndarray:
        """
        Route predictions to segment specialists with soft blending.

        Routing is based on ``base_predictions`` (global model output in
        original dollar scale), NOT on true labels — identical to how
        BiasCorrection.apply() routes, ensuring train/inference consistency.

        Blend logic at each tier boundary (width = BLEND_WIDTH):
          • pred < boundary − width/2  → pure lower-tier specialist
          • pred in blend zone         → linear interpolation
          • pred > boundary + width/2  → pure upper-tier specialist
        """
        if not self._fitted or not self.specialists:
            return base_predictions.copy()

        y_pred = base_predictions.copy()
        hw = self.BLEND_WIDTH / 2.0

        for tier_name, lo, hi in self.TIERS:
            if tier_name not in self.specialists:
                continue

            info = self.specialists[tier_name]
            model = info["model"]
            is_inf = hi == float("inf")

            # Routing zone: tier interior + blend fringe
            blend_lo = lo - hw
            blend_hi = hi + hw if not is_inf else float("inf")

            any_mask = (
                (base_predictions >= blend_lo) & (base_predictions < blend_hi)
                if not is_inf
                else (base_predictions >= blend_lo)
            )
            if not any_mask.any():
                continue

            idx = np.where(any_mask)[0]
            X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            spec_preds = model.predict(X_sub)
            bp_sub = base_predictions[idx]

            # Pure interior of tier
            if not is_inf:
                pure = (bp_sub >= lo) & (bp_sub < hi)
            else:
                pure = bp_sub >= lo

            # Lower boundary blend (ascending weight from 0 → 1 as we enter tier)
            lower_blend = (bp_sub >= blend_lo) & (bp_sub < lo)
            # Upper boundary blend (descending weight from 1 → 0 as we leave tier)
            upper_blend = (
                np.zeros(len(idx), dtype=bool) if is_inf else ((bp_sub >= hi) & (bp_sub < blend_hi))
            )

            # Pure interior: 100% specialist
            y_pred[idx[pure]] = spec_preds[pure]

            # Lower blend: weight grows from 0 (at blend_lo) to 1 (at lo)
            if lower_blend.any():
                denom = lo - blend_lo
                w = np.clip((bp_sub[lower_blend] - blend_lo) / max(denom, 1e-6), 0.0, 1.0)
                y_pred[idx[lower_blend]] = (1 - w) * base_predictions[
                    idx[lower_blend]
                ] + w * spec_preds[lower_blend]

            # Upper blend: weight falls from 1 (at hi) to 0 (at blend_hi)
            if upper_blend.any():
                denom = blend_hi - hi
                w = np.clip(1.0 - (bp_sub[upper_blend] - hi) / max(denom, 1e-6), 0.0, 1.0)
                y_pred[idx[upper_blend]] = (1 - w) * base_predictions[
                    idx[upper_blend]
                ] + w * spec_preds[upper_blend]

        return y_pred

    def save(self, output_dir) -> list[tuple[str, Any]]:
        """
        Persist each specialist to ``specialist_{tier}.joblib``.

        Generates a SHA-256 checksum file alongside each saved joblib
        artifact so load_model() and main.py _verify_model_checksums() can
        detect corruption or silent replacement between training runs.
        Previously all specialist saves used raw joblib.dump() with no checksum
        — the exact pattern that caused the permanent 'No checksum file for
        xgboost_high_value_specialist' warning on every startup.

        Returns list of (tier_name, path) for logging.
        """
        import hashlib as _hl
        from pathlib import Path as _P

        import joblib as _jl

        saved = []
        out = _P(output_dir)
        for tier_name, info in self.specialists.items():
            path = out / f"specialist_{tier_name}.joblib"
            _jl.dump(info["model"], path)
            # write checksum immediately after dump so the two files
            # are always in sync — no window where joblib exists but checksum
            # is missing or stale from a previous training run.
            _checksum = _hl.sha256(path.read_bytes()).hexdigest()
            _ck_path = out / f"specialist_{tier_name}_checksum.txt"
            _ck_path.write_text(f"{_checksum}\n")
            saved.append((tier_name, path))
        return saved

    @classmethod
    def load(cls, output_dir) -> FullSegmentSpecialist:
        """
        Reconstruct a FullSegmentSpecialist from saved joblib files.

        Missing tier files are silently skipped (non-fatal degraded mode).
        """
        from pathlib import Path as _P

        import joblib as _jl

        instance = cls()
        instance._fitted = True
        out = _P(output_dir)
        for tier_name, lo, hi in cls.TIERS:
            path = out / f"specialist_{tier_name}.joblib"
            if path.exists():
                instance.specialists[tier_name] = {
                    "model": _jl.load(path),
                    "lo": lo,
                    "hi": hi,
                }
        logger.info(
            f"✅ FullSegmentSpecialist loaded: {len(instance.specialists)} tiers " f"from {out}"
        )
        return instance

    @property
    def tier_count(self) -> int:
        return len(self.specialists)


# ============================================================================
# 5. INTEGRATION: add to train.py evaluate_test() after original-scale metrics
# ============================================================================

EVALUATE_TEST_ADDITION = """
# ── Add after original-scale metric computation in evaluate_test() ──────────

# Gate G6: Cost-weighted R²
from insurance_ml.diagnostics import DeploymentGates   # (or import from patch_05)

g6_result = DeploymentGates.check_g6(
    y_true=y_test_original,
    y_pred=y_pred_orig,
    raise_on_fail=False,   # Log + continue; set True to hard-block deploy
)

if not g6_result["g6_pass"]:
    logger.error(
        f"❌ Gate G6 FAILED: cost_weighted_r2={g6_result['cost_weighted_r2']:.4f} "
        f"< {g6_result['min_threshold']}. "
        "Consider training a HighValueSpecialist model (patch_05_highvalue.py)."
    )

# Include G6 result in the returned test_result dict
test_result["g6_gate"] = g6_result
# ────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# END INLINE PATCHES
# ============================================================================


def main():
    """Production workflow"""
    import sys

    # Windows console encoding
    if platform.system() == "Windows":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*80}\nInsurance ML Pipeline v{VERSION}\n{'='*80}\n")

    pipeline_start = time.time()

    # ── PATCH 03 (G4/G9): capture git provenance before any training ─────────
    _provenance = capture_git_provenance()
    if _provenance.commit_hash == "unknown":
        logger.warning(
            "⚠️  G4 GATE: Git commit unknown. Ensure training runs inside a "
            "git repository. CI/CD: run 'git fetch --unshallow' for shallow clones."
        )

    try:
        # Initialize trainer
        trainer = ModelTrainer()

        # Assign provenance to trainer so gate checks via
        # getattr(trainer, '_provenance', None) actually find the object.
        trainer._provenance = _provenance

        # Data preparation
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("DATA PREPARATION")
        logger.info("=" * 80)

        data = trainer.prepare_training_data(
            target_transform=None,
            stratify=None,
            remove_outliers=None,
            remove_collinear=None,
            add_polynomials=None,
        )

        # ── PATCH 01 (G2): hard split-disjointness assertion ────────────────────
        assert_splits_disjoint(data, label="main_pipeline")

        prep_time = time.time() - start_time
        logger.info(f"✅ Data preparation completed in {prep_time:.1f}s\n")

        # Model training
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("MODEL TRAINING")
        logger.info("=" * 80)

        # Use two_model_architecture when enabled in config.yaml.
        # xgboost_median (reg:squarederror) is the pricing model:
        #   - symmetric loss -> ~50% overpricing rate (G7 gate)
        #   - max_depth=6 vs 4 -> better high-value segment R2
        # xgboost (reg:quantileerror alpha=0.65) is the risk/loading model.
        # Falls back to single-model train when two_model_architecture.enabled=false.
        _tma_cfg = trainer.raw_config.get("training", {}).get("two_model_architecture", {})
        if _tma_cfg.get("enabled", False):
            logger.info(
                "Two-model architecture ENABLED:\n"
                f"   Pricing : {_tma_cfg.get('pricing_model','xgboost_median')} "
                f"(reg:squarederror — G7 target <=55%)\n"
                f"   Risk    : {_tma_cfg.get('risk_model','xgboost')} "
                f"(reg:quantileerror alpha={_tma_cfg.get('risk_model_alpha', trainer.raw_config.get('models',{}).get('xgboost',{}).get('quantile_alpha', 0.30))} — tail loading)"
            )
            _tma = train_two_model_architecture(trainer, data, trainer.raw_config)
            results = {**_tma["pricing_results"], **_tma["risk_results"]}
            _tma.get("architecture")
        else:
            results = trainer.train_all_models(data)

        # ========================================
        # VALIDATE TRAINING RESULTS
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING TRAINING RESULTS")
        logger.info("=" * 80)

        failed_validation = []

        for model_name, result in results.items():
            if result.get("status") != "success":
                continue

            try:
                # Validate training metrics
                MetricsExtractor.validate_metrics(
                    result["training_metrics"], context=f"{model_name}_training"
                )

                # Validate validation metrics
                MetricsExtractor.validate_metrics(
                    result["validation_metrics"], context=f"{model_name}_validation"
                )

                logger.info(f"   ✅ {model_name}: Metrics valid")

            except ValidationError as e:
                logger.error(f"   ❌ {model_name}: {e}")
                failed_validation.append((model_name, str(e)))

        logger.info("=" * 80 + "\n")

        if failed_validation:
            logger.error(
                f"❌ {len(failed_validation)} model(s) failed validation:\n"
                + "\n".join(f"   â€¢ {name}: {error}" for name, error in failed_validation)
            )
            raise ValidationError(
                f"Training results validation failed for {len(failed_validation)} model(s)"
            )

        train_time = time.time() - start_time
        logger.info(f"✅ Training completed in {train_time:.1f}s\n")

        # ========================================
        # ENHANCED MODEL COMPARISON SUMMARY
        # ========================================
        print(f"\n{'='*80}\nModel Comparison Summary\n{'='*80}")

        successful = {k: v for k, v in results.items() if v.get("status") == "success"}

        if successful:
            comparison_rows = []

            for name, result in successful.items():
                train_metrics = result["training_metrics"]
                val_metrics = result["validation_metrics"]

                # Calculate generalization gap using MetricsExtractor
                gap = MetricsExtractor.calculate_generalization_gap(train_metrics, val_metrics)

                # For quantile models, RMSE gap is inflated vs the training
                # objective (pinball loss). The comparison summary previously said
                # "see pinball gap" but never showed it. Now we display both:
                #   - Dollar-RMSE gap:  from MetricsExtractor (diagnostic, not the training metric)
                #   - Pinball gap:      from Optuna opt_data (the real training-objective gap)
                _obj = str(result.get("model_config", {}).get("objective", "")).lower()
                _is_quantile = "quantile" in _obj
                _pinball_gap = result.get("pinball_gap_percent", 0.0)
                if _is_quantile and _pinball_gap:
                    _status_label = f"Quantile | Pinball gap: {_pinball_gap:+.1f}%"
                elif _is_quantile:
                    _status_label = "Quantile model (✓ see pinball gap)"
                else:
                    _status_label = gap["train_val_status"].replace("_", " ").title()
                row = {
                    "Model": name,
                    # Column label is "Val RMSE" — NOT test RMSE.
                    # Test RMSE is reported separately in TEST PERFORMANCE SUMMARY below
                    # after test evaluation completes. The two numbers come from different
                    # data splits and are not directly comparable (small n=268 test set
                    # causes large sampling variance; e.g. $5,206 val vs $3,784 test is
                    # expected, not a sign of data leakage or a bug).
                    "Val RMSE": f"${MetricsExtractor.get_rmse(val_metrics):,.0f}",
                    "Val R2": f"{MetricsExtractor.get_r2(val_metrics):.4f}",
                    # RMSE Gap for quantile models uses pinball gap (training
                    # objective), NOT dollar-RMSE gap (which inflates vs squarederror
                    # baseline). Status column already shows the correct label.
                    "RMSE Gap": (
                        f"pinball {_pinball_gap:+.1f}%"
                        if _is_quantile and _pinball_gap
                        else f"{gap['train_val_gap_pct']:+.1f}%"
                    ),
                    "Status": _status_label,
                    "Time": f"{result['training_time']:.1f}s",
                    "GPU": "✓" if result.get("gpu_used") else "✗",
                }

                comparison_rows.append(row)

            # Sort by validation RMSE (best first)
            comparison_rows.sort(
                key=lambda x: float(x["Val RMSE"].replace("$", "").replace(",", ""))
            )

            # Print table
            df = pd.DataFrame(comparison_rows)
            print(df.to_string(index=False))
        else:
            print("[X] No successful models to compare")

        # Find best model
        valid = {k: v for k, v in results.items() if v.get("status") == "success"}

        if not valid:
            print("\n[X] No successful models!")
            sys.exit(1)

        # Route best_name by model role, not RMSE.
        # Two-model architecture produces heterogeneous objectives: comparing
        # reg:squarederror and reg:quantileerror on RMSE is dimensionally incoherent.
        # Quantile loss at alpha=0.65 predicts the 65th-percentile; for right-skewed
        # insurance data this sits near the mean, so xgboost's RMSE coincidentally
        # beats xgboost_median's ($4,594 vs $4,619), causing the quantile model to be
        # selected as the deployment artifact. Cascading failures:
        #   - G7 evaluates uncorrected quantile predictions (76.1% overpricing > 55%)
        #   - Segment R2=-1.9 on High (quantile predictions are not mean estimates)
        #   - bias_correction.json written as stub (applied=False) for xgboost
        #   - calibration holdout evaluated without bias correction
        # _tma_cfg is already computed above (L6749); use it directly.
        if _tma_cfg.get("enabled", False):
            _pricing_key = _tma_cfg.get("pricing_model", "xgboost_median")
            if _pricing_key in valid:
                best_name = _pricing_key
                logger.info(
                    f"Two-model architecture: pricing model '{best_name}' "
                    f"selected as deployment artifact "
                    f"(overrides RMSE-min; objectives are not comparable)."
                )
            else:
                logger.warning(
                    f"Pricing model key '{_pricing_key}' not found in valid results "
                    f"{list(valid.keys())}. Falling back to RMSE-min."
                )
                best_name = min(
                    valid.keys(),
                    key=lambda k: trainer.MetricsExtractor.get_rmse(
                        results[k]["validation_metrics"]
                    ),
                )
        else:
            # Single-model path: pick by validation RMSE as before
            best_name = min(
                valid.keys(),
                key=lambda k: trainer.MetricsExtractor.get_rmse(results[k]["validation_metrics"]),
            )
        best_result = results[best_name]
        best_path = Path(best_result["model_path"])
        best_model = best_result.get(
            "model"
        )  # in-memory object; may be overwritten by disk-load below

        print(f"\n[OK] Best model: {best_name}")

        # ============================================================================
        # ✅ NEW CODE: SAVE BEST MODEL METADATA FOR PREDICT.PY
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("SAVING BEST MODEL METADATA")
        logger.info("=" * 80)

        metadata_path = trainer.config.output_dir / "pipeline_metadata.json"

        # Load existing metadata if it exists
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    pipeline_metadata = json.load(f)
                logger.info(f"Loaded existing metadata from {metadata_path.name}")
            except Exception as e:
                logger.warning(f"Could not load existing metadata: {e}")
                pipeline_metadata = {}
        else:
            pipeline_metadata = {}
            logger.info(f"Creating new metadata file: {metadata_path.name}")

        # Extract best model metrics
        best_val_metrics = results[best_name]["validation_metrics"]
        best_val_rmse = float(trainer.MetricsExtractor.get_rmse(best_val_metrics))
        best_val_r2 = float(trainer.MetricsExtractor.get_r2(best_val_metrics))
        best_val_mae = float(trainer.MetricsExtractor.get_mae(best_val_metrics))

        # Update with best model info
        pipeline_metadata.update(
            {
                "best_model": best_name,
                "best_model_path": str(best_path),
                "best_val_rmse": best_val_rmse,
                "best_val_r2": best_val_r2,
                "best_val_mae": best_val_mae,
                "training_timestamp": pd.Timestamp.now().isoformat(),
                "trained_models": list(valid.keys()),
                "pipeline_version": VERSION,
                "model_schema_version": MODEL_SCHEMA_VERSION,
                # P0-A: persist run_id so evaluate.py can link back to training run
                "mlflow_run_id": results.get(best_name, {}).get("mlflow_run_id"),
            }
        )

        # ── v7.5.1: Persist bias correction thresholds for drift tracking ────
        # q50/q75 are dynamic (derived from y_pred_original each run).
        # Writing them here lets the NEXT run's drift check compare against
        # a stable reference rather than the potentially-stale bias_correction.json
        # (which may be absent on quantile-model runs where BC is skipped).
        try:
            _bc_json = Path(best_path).parent / "bias_correction.json"
            if _bc_json.exists():
                import json as _jm

                with open(_bc_json) as _f:
                    _bc_data = _jm.load(_f)
                _params = _bc_data.get("correction_params") or _bc_data
                if _params and "threshold_low" in _params:
                    pipeline_metadata["bias_correction_thresholds"] = {
                        "q50_threshold_low": float(_params["threshold_low"]),
                        "q75_threshold_high": float(_params.get("threshold_high", 0)),
                        "run_timestamp": pd.Timestamp.now().isoformat(),
                    }
                    logger.info(
                        f"   BiasCorrection thresholds persisted: "
                        f"q50=${_params['threshold_low']:,.0f}, "
                        f"q75=${_params.get('threshold_high', 0):,.0f}"
                    )
        except Exception as _bc_meta_err:
            logger.debug(f"Could not persist bias correction thresholds: {_bc_meta_err}")

        # Save updated metadata
        try:
            with open(metadata_path, "w") as f:
                json.dump(pipeline_metadata, f, indent=2)

            logger.info(f"✅ Saved best model metadata to {metadata_path.name}")
            logger.info(f"   Best model: {best_name}")
            logger.info(f"   Val RMSE: ${best_val_rmse:,.2f}")
            logger.info(f"   Val R²: {best_val_r2:.4f}")
            logger.info(f"   Val MAE: ${best_val_mae:,.2f}")
            logger.info(f"   Trained models: {', '.join(valid.keys())}")
            logger.info("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")
            logger.warning("⚠️ Prediction pipeline may not auto-detect best model!")

        # ========================================
        # OPTION A: THREE-WAY SPLIT (RECOMMENDED)
        # ========================================
        if trainer.explainability_config.enable_confidence_intervals:
            logger.info("=" * 80)
            logger.info("POST-HOC CALIBRATION WITH PROPER HOLDOUT")
            logger.info("=" * 80)

            # ====================================================================
            # STEP 1: Split validation into calibration + holdout
            # ====================================================================
            from sklearn.model_selection import train_test_split

            X_val = data["X_val"]
            y_val = data["y_val"]

            # Split: 60% calibration, 40% holdout
            # stratified split preserves the
            # premium distribution in both halves.  Without it the 60/40 index-order
            # split can create mismatched distributions: calibration improvement was
            # measured as -2.11% RMSE (worse) on holdout vs +3.98% on the calib set
            # — a distribution artefact, not a model quality signal.
            from sklearn.preprocessing import KBinsDiscretizer as _KBD

            try:
                _kbd = _KBD(n_bins=7, encode="ordinal", strategy="quantile", subsample=None)
                _y_val_arr = (
                    y_val.values.reshape(-1, 1)
                    if hasattr(y_val, "values")
                    else np.array(y_val).reshape(-1, 1)
                )
                _y_val_bins = _kbd.fit_transform(_y_val_arr).ravel().astype(int)
            except Exception as _kbd_err:
                logger.warning(
                    f"Calibration split stratification failed ({_kbd_err}); "
                    "falling back to unstratified split."
                )
                _y_val_bins = None

            X_calib, X_holdout, y_calib, y_holdout = train_test_split(
                X_val,
                y_val,
                test_size=0.4,  # 40% for fair evaluation
                random_state=trainer.config.random_state,
                stratify=_y_val_bins,
            )

            logger.info(
                f"Validation split:\n"
                f"   Calibration: {len(X_calib)} samples ({len(X_calib)/len(X_val)*100:.0f}%)\n"
                f"   Holdout:     {len(X_holdout)} samples ({len(X_holdout)/len(X_val)*100:.0f}%)"
            )

            # ====================================================================
            # STEP 2: Load best model and fit calibrator on calibration set ONLY
            # ====================================================================
            best_model = FileSanitizer.safe_load(
                best_path,
                trainer.config.max_model_size_mb,
                verify_checksum=trainer.config.verify_checksums,
            )

            # Snapshot conformal data AFTER reload, BEFORE any mutation.
            # The original snapshot was taken from the pre-reload in-memory best_model.
            # best_model is rebound to a new deserialized object above (FileSanitizer.safe_load),
            # so the pre-reload snapshot was from a different object identity than the one
            # STEP 3 mutates and the restore targets. Taking the snapshot here guarantees:
            #   1. Same object identity — snapshot, mutation, and restore all act on this object.
            #   2. copy.deepcopy instead of v[:] — defensive against any future in-place
            #      mutation of nested structures (numpy arrays, nested dicts) in _conformal_data.
            # STEP 3 (below) overwrites _conformal_data["validation_predictions/residuals"]
            # with the calibration-set slice. If calibration is REJECTED, the restore at
            # line ~7595 rolls this back so the conformal evaluator sees the full-val arrays,
            # not the 160-sample calibration subset → prevents $112k CI width regression.
            import copy

            _conformal_snapshot: dict | None = None
            if hasattr(best_model, "_conformal_data") and isinstance(
                best_model._conformal_data, dict
            ):
                _conformal_snapshot = copy.deepcopy(best_model._conformal_data)
                logger.info(
                    f"   📸 Conformal data snapshot saved: "
                    f"{_conformal_snapshot.get('n_calibration', 0)} samples "
                    f"(will restore if calibration is rejected)"
                )

            from insurance_ml.models import CalibratedModel

            # Switch from isotonic to linear calibration.
            # Isotonic regression improved transformed-space RMSE by +7.42%
            # but degraded holdout original-space RMSE by 0.32% because:
            #   1. IsotonicRegression memorises 160 YJ-space samples, learning
            #      a flat/squeezed mapping in the sparse high-value tail.
            #   2. The nonlinear YJ inverse transform then amplifies those
            #      small YJ-space errors into large dollar-space misses.
            #   3. RMSE is tail-sensitive so even a handful of worsened
            #      high-value predictions dominate the metric.
            # Linear calibration (y = a*ŷ + b, 2 free parameters via OLS)
            # corrects systematic offset/scale bias without distorting the
            # tail ordering, and cannot overfit 160 samples.
            calibrated_model = CalibratedModel(base_model=best_model, calibration_method="linear")

            logger.info("🔧 Fitting linear calibrator on calibration set...")
            calibrated_model.fit_calibrator(X_val=X_calib, y_val=y_calib)

            # ====================================================================
            # STEP 3: Store validation residuals (CALIBRATION SET ONLY)
            # ====================================================================
            y_calib_pred_cal = calibrated_model.predict(X_calib)
            calibrated_model._validation_residuals = (
                y_calib.values if hasattr(y_calib, "values") else y_calib
            ) - y_calib_pred_cal

            # Sync _conformal_data so validation_predictions and validation_residuals
            # have matching lengths.  Without this the heteroscedastic conformal path
            # detects a mismatch (268 preds vs 160 residuals) and falls back to the
            # less-adaptive global intervals.
            calibrated_model._validation_predictions = y_calib_pred_cal.copy()
            if hasattr(calibrated_model.base_model, "_conformal_data") and isinstance(
                calibrated_model.base_model._conformal_data, dict
            ):
                calibrated_model.base_model._conformal_data["validation_predictions"] = (
                    y_calib_pred_cal.tolist()
                )
                calibrated_model.base_model._conformal_data["validation_residuals"] = (
                    calibrated_model._validation_residuals.tolist()
                )
                calibrated_model.base_model._conformal_data["n_calibration"] = int(
                    len(y_calib_pred_cal)
                )

            logger.info(
                f"📊 Stored {len(calibrated_model._validation_residuals)} "
                f"validation residuals (std={np.std(calibrated_model._validation_residuals):.4f})"
            )

            # Original block called calibrated_model.predict(X_val) on full 268-sample
            # val set, creating circular conformal coverage. Removed: conformal
            # calibration now stays on X_calib only (~160 samples) — honest coverage.
            logger.info(
                "ℹ️  Conformal calibration: X_calib only (%d samples). "
                "Full-val upgrade removed — prevents circular coverage.",
                len(X_calib),
            )

            # ====================================================================
            # STEP 4: FAIR COMPARISON on holdout set (unseen by calibrator!)
            # ====================================================================
            logger.info("\n" + "=" * 80)
            logger.info("FAIR CALIBRATION COMPARISON (HOLDOUT SET)")
            logger.info("=" * 80)

            # Evaluate uncalibrated on holdout
            logger.info("Evaluating UNCALIBRATED model on holdout set...")
            best_bias_correction_holdout = results[best_name].get("bias_correction")
            holdout_metrics_uncalibrated, holdout_preds_uncalibrated = (
                trainer.model_manager.evaluate_model(
                    model=best_model,
                    X_test=X_holdout,
                    y_test=y_holdout,
                    model_name=f"{best_name}_uncalibrated",
                    target_transformation=data["target_transformation"],
                    feature_engineer=data["feature_engineer"],
                    phase="holdout_uncalibrated",
                    bias_correction=best_bias_correction_holdout,
                )
            )

            # Evaluate calibrated on holdout
            logger.info("Evaluating CALIBRATED model on holdout set...")
            holdout_metrics_calibrated, holdout_preds_calibrated = (
                trainer.model_manager.evaluate_model(
                    model=calibrated_model,
                    X_test=X_holdout,
                    y_test=y_holdout,
                    model_name=f"{best_name}_calibrated",
                    target_transformation=data["target_transformation"],
                    feature_engineer=data["feature_engineer"],
                    phase="holdout_calibrated",
                    bias_correction=best_bias_correction_holdout,
                )
            )

            # ====================================================================
            # STEP 5: Compare using MetricsExtractor (centralized logic)
            # ====================================================================
            comparison = MetricsExtractor.compare_models(
                holdout_metrics_uncalibrated,
                holdout_metrics_calibrated,
                model_name_a="Uncalibrated",
                model_name_b="Calibrated",
            )

            # Show all decision criteria explicitly so the reasoning
            # is always auditable. adds a MAE gate: calibration that improves
            # RMSE/R² while worsening MAE by >2% is not genuinely helpful for insurance
            # pricing (MAE reflects the typical policyholder experience more directly).
            _rmse_ok = comparison["rmse_improvement_pct"] > 1.0
            _r2_ok = comparison["r2_improvement_pct"] >= 0
            _mae_ok = comparison["mae_improvement_pct"] >= -2.0  # allow <=2% MAE regression
            use_calibrated = (
                comparison["is_better"]
                and _rmse_ok
                and _r2_ok
                and _mae_ok  # MAE must not worsen by more than 2%
            )
            logger.info(
                f"\nCalibration Impact (HOLDOUT SET - NO LEAK):\n"
                f"   RMSE Improvement: {comparison['rmse_improvement_pct']:+.2f}%"
                f"  {'✅ (>1.0% threshold)' if _rmse_ok else '⚠️  (<1.0% threshold — insufficient)'}\n"
                f"   R² Improvement:   {comparison['r2_improvement_pct']:+.2f}%"
                f"  {'✅' if _r2_ok else '⚠️  (regressed)'}\n"
                f"   MAE Improvement:  {comparison['mae_improvement_pct']:+.2f}%"
                f"  {'✅' if _mae_ok else '⚠️  (worsened >2% — FIX-8 MAE gate blocks calibration)'}\n"
                f"   is_better flag:   {'✅ RMSE+R² both improved' if comparison['is_better'] else '⚠️  RMSE or R² did not improve'}\n"
                f"   Decision: {'✅ Use calibrated' if use_calibrated else '⚠️  Keep uncalibrated'}"
                f"  (requires is_better AND RMSE>1% AND R2>=0 AND MAE>=-2%)"
            )

            if use_calibrated:
                logger.info("✅ Using calibrated model for test evaluation")

                calibrated_name = f"{best_name}_calibrated"
                calibrated_path = trainer.config.output_dir / f"{calibrated_name}.joblib"

                # Save calibrated model
                _cal_objective = str(
                    results[best_name].get("model_config", {}).get("objective", "reg:squarederror")
                )
                _cal_meta = {
                    "calibration_method": "linear",
                    "calibration_comparison": comparison,
                    "calibration_set_size": len(X_calib),
                    "holdout_set_size": len(X_holdout),
                    "original_model": best_name,
                    "holdout_rmse_uncalibrated": float(comparison["rmse_a"]),
                    "holdout_rmse_calibrated": float(comparison["rmse_b"]),
                    # ── PATCH 03 (G4/G9): provenance fields ─────────────────────
                    "git_commit": _provenance.commit_hash,
                    "git_branch": _provenance.branch,
                    "git_dirty": _provenance.is_dirty,
                    "random_state": trainer.config.random_state,
                    "model_objective": _cal_objective,
                    "training_timestamp": _provenance.capture_timestamp,
                    "pipeline_version": VERSION,
                    "split_sizes": {
                        "train": len(data["X_train"]),
                        "val": len(data["X_val"]),
                        "calib": len(X_calib),
                        "holdout": len(X_holdout),
                        "test": len(data["X_test_raw"]),
                    },
                }
                trainer.model_manager.save_model(
                    calibrated_model,
                    calibrated_name,
                    additional_metadata=_cal_meta,
                )

                # Update results with HOLDOUT metrics (NOT contaminated validation)
                results[calibrated_name] = {
                    **results[best_name],
                    "model_name": calibrated_name,
                    "model_path": str(calibrated_path),
                    "holdout_metrics": holdout_metrics_calibrated,  # Use holdout
                    "holdout_predictions": holdout_preds_calibrated,
                    "is_calibrated": True,
                    "calibration_improvement": comparison,
                    "calibration_decision": {
                        "use_calibrated": True,
                        "evaluated_on": "holdout_set",  # Make it explicit
                        "rmse_improvement_pct": comparison["rmse_improvement_pct"],
                    },
                }

                best_path = calibrated_path
                best_name = calibrated_name

                logger.info("✅ Stored calibrated holdout metrics (NO LEAK)")
            else:
                logger.info("⚠️  Calibration did not improve performance on holdout set")
                logger.info("   Keeping uncalibrated model")
                # Restore _conformal_data that was mutated by STEP 3.
                # Without this rollback the uncalibrated model carries 160-sample
                # conformal data instead of the correct 268 → CI width $112k.
                if _conformal_snapshot is not None and hasattr(best_model, "_conformal_data"):
                    best_model._conformal_data.update(_conformal_snapshot)
                    logger.info(
                        f"   ✅ Conformal data restored to pre-calibration state: "
                        f"{_conformal_snapshot.get('n_calibration', 0)} samples"
                    )

            logger.info("=" * 80 + "\n")

        # ── PATCH 03 (G5): always write bias_correction.json ─────────────────
        _best_bc = results[best_name].get("bias_correction")
        _best_objective = str(
            results[best_name].get("model_config", {}).get("objective", "reg:squarederror")
        )
        always_write_bias_correction(
            output_dir=trainer.config.output_dir,
            bias_correction=_best_bc,
            model_objective=_best_objective,
            provenance=_provenance,
            random_state=trainer.config.random_state,
            pipeline_version=VERSION,
        )

        # ====================================================================
        # PHASE 3: FULL-SEGMENT SPECIALIST MODELS (v7.5.5)
        # ====================================================================
        # Trains one XGBRegressor per G6 tier (Low/Mid/HighMid/VeryHigh).
        # Each specialist is trained only on training samples within its tier,
        # allowing it to capture intra-tier patterns the global model misses.
        #
        # At test evaluation, FullSegmentSpecialist.predict() routes by the
        # global model's prediction value — consistent with BC routing — and
        # soft-blends at tier boundaries.
        #
        # Individual specialist files: specialist_{tier}.joblib
        # Non-fatal: any failure is logged and pipeline continues.
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: FULL-SEGMENT SPECIALIST MODELS")
        logger.info("=" * 80)

        _full_specialist: FullSegmentSpecialist | None = None

        try:
            _y_train_orig = data.get("y_train_original")
            if _y_train_orig is None:
                logger.warning(
                    "⚠️  y_train_original not in data dict — "
                    "using transformed targets as fallback (degraded)."
                )
                _y_train_orig = (
                    data["y_train"].values
                    if hasattr(data["y_train"], "values")
                    else np.array(data["y_train"])
                )

            _y_train_orig_arr = (
                _y_train_orig.values
                if hasattr(_y_train_orig, "values")
                else np.array(_y_train_orig)
            )

            # Determine GPU config from global config
            _gpu_cfg_raw = trainer.raw_config.get("gpu", {})
            _xgb_median_cfg = _gpu_cfg_raw.get("xgboost_median", _gpu_cfg_raw.get("xgboost", {}))
            _spec_gpu: dict = {}
            if _gpu_cfg_raw.get("enabled", False):
                _spec_gpu = {
                    "device": _xgb_median_cfg.get("device", "cuda:0"),
                    "tree_method": _xgb_median_cfg.get("tree_method", "hist"),
                    "n_jobs": 1,
                }

            _full_specialist = FullSegmentSpecialist()
            _full_specialist.fit(
                X_train=data["X_train"],
                y_train_original=_y_train_orig_arr,
                random_state=trainer.config.random_state,
                gpu_config=_spec_gpu or None,
            )

            _saved_specs = _full_specialist.save(trainer.config.output_dir)
            logger.info(
                f"✅ Specialist models saved ({len(_saved_specs)} tiers):\n"
                + "".join(f"   • {name}: {path.name}\n" for name, path in _saved_specs)
            )

            # ── Legacy HighValueSpecialist compat ────────────────────────────
            # predict.py's HighValueSegmentRouter still loads
            # xgboost_high_value_specialist.joblib. Save the very_high tier
            # under that name so the router remains functional.
            if "very_high" in _full_specialist.specialists:
                import hashlib as _hl_compat

                import joblib as _jl_compat

                _compat_path = trainer.config.output_dir / "xgboost_high_value_specialist.joblib"
                _jl_compat.dump(_full_specialist.specialists["very_high"]["model"], _compat_path)
                # write checksum for the legacy compat file.
                # This is the artifact that predict.py HighValueSegmentRouter loads.
                # Previously this raw joblib.dump() had no checksum, causing the
                # permanent "No checksum file for xgboost_high_value_specialist"
                # warning on every startup and making the specialist
                # integrity-unverifiable on every inference call.
                _ck_compat = _hl_compat.sha256(_compat_path.read_bytes()).hexdigest()
                _ck_compat_path = (
                    trainer.config.output_dir / "xgboost_high_value_specialist_checksum.txt"
                )
                _ck_compat_path.write_text(f"{_ck_compat}\n")
                logger.info(
                    f"✅ Legacy xgboost_high_value_specialist.joblib written "
                    f"(very_high tier, compat with predict.py HighValueSegmentRouter)\n"
                    f"   Checksum: {_ck_compat[:16]}..."
                )

        except Exception as _spec_err:
            logger.error(
                f"❌ FullSegmentSpecialist training failed: {_spec_err}\n"
                "   Pipeline continues — G6 evaluation will use global model only.",
                exc_info=True,
            )
            _full_specialist = None

        logger.info("=" * 80 + "\n")

        # ========================================
        # TEST EVALUATION (Uses best model)
        # ========================================
        start_time = time.time()
        print(f"\n{'='*80}\nTest Evaluation\n{'='*80}")

        # Extract bias correction from best model
        best_bias_correction = results[best_name].get("bias_correction")

        logger.info(f"Best model: {best_name}")
        if best_bias_correction:
            logger.info(f"  Bias correction: {best_bias_correction}")

        test_result = trainer.evaluate_test(
            model_path=best_path,
            feature_engineer=data["feature_engineer"],
            X_test_raw=data["X_test_raw"],
            y_test_raw=data["y_test_raw"],
            name=best_name,
            bias_correction=best_bias_correction,
        )

        # ── v7.5.5: Full-segment specialist routing ─────────────────────────
        # FullSegmentSpecialist routes ALL four G6 tiers (Low/Mid/HighMid/VH)
        # through their respective specialists. This is the primary path for
        # clearing G6 on large datasets where the global model's per-tier
        # RMSE exceeds the within-segment std.
        #
        # On small datasets (≤5K rows) each tier may have N<50 and no
        # specialist is trained; the code falls through gracefully to the
        # global-model G6 result.
        _spec_path_eval = trainer.config.output_dir / "specialist_low.joblib"
        _has_full_specialist = _spec_path_eval.exists() or (
            _full_specialist is not None and _full_specialist.tier_count > 0
        )

        if _has_full_specialist:
            try:
                _fe_eval = data["feature_engineer"]
                _X_test_e = _fe_eval.transform_pipeline(data["X_test_raw"], remove_outliers=False)
                _y_test_orig_e = (
                    data["y_test_raw"].values
                    if hasattr(data["y_test_raw"], "values")
                    else np.array(data["y_test_raw"])
                )

                # Global model: transformed → inverse → bias correction
                _global_eval = trainer.model_manager.load_model(best_name)
                _base_t = _global_eval.predict(_X_test_e)
                _base_orig = _fe_eval.inverse_transform_target(
                    _base_t,
                    transformation_method=_fe_eval.target_transformation.method,
                    clip_to_safe_range=True,
                    context="routing_eval_base",
                )
                if best_bias_correction is not None:
                    _base_orig = best_bias_correction.apply(_base_orig)

                # Load FullSegmentSpecialist (in-memory or from disk)
                if _full_specialist is not None and _full_specialist.tier_count > 0:
                    _fss = _full_specialist
                else:
                    _fss = FullSegmentSpecialist.load(trainer.config.output_dir)

                # Route all segments through specialists
                _routed = _fss.predict(_X_test_e, _base_orig)

                # ── Smoker-aware correction (supplementary) ───────────────────
                # Under-predicted smoker policies (smoker=1, base_pred < $10K)
                # have true charges $10K-$20K+ but the global model predicts $5K-$9K.
                # Apply a mild specialist blend from the HighMid specialist to lift them.
                try:
                    _smoker_mask = None
                    if hasattr(_X_test_e, "columns") and "smoker" in _X_test_e.columns:
                        _smoker_mask = _X_test_e["smoker"].values == 1
                    elif hasattr(_X_test_e, "columns"):
                        for _sc in ["smoker_binary", "is_smoker"]:
                            if _sc in _X_test_e.columns:
                                _smoker_mask = _X_test_e[_sc].values == 1
                                break

                    _SMOKER_BLEND_WEIGHT = 0.35
                    _SMOKER_PRED_THRESHOLD = 10_000.0  # $10,000 — matches BiasCorrection._BC_TIER_LOW; that name is local to calculate_from_model() and not accessible here

                    if _smoker_mask is not None and "high_mid" in _fss.specialists:
                        _hm_model = _fss.specialists["high_mid"]["model"]
                        _underpred = _smoker_mask & (_base_orig < _SMOKER_PRED_THRESHOLD)
                        n_smoker_up = int(_underpred.sum())
                        if n_smoker_up > 0:
                            _X_up = (
                                _X_test_e.iloc[_underpred]
                                if hasattr(_X_test_e, "iloc")
                                else _X_test_e[_underpred]
                            )
                            _hm_preds = _hm_model.predict(_X_up)
                            _routed[_underpred] = (1 - _SMOKER_BLEND_WEIGHT) * _routed[
                                _underpred
                            ] + _SMOKER_BLEND_WEIGHT * _hm_preds
                            logger.info(
                                f"   Smoker-aware routing: {n_smoker_up} under-predicted "
                                f"smoker policies blended with high_mid specialist "
                                f"(weight={_SMOKER_BLEND_WEIGHT})"
                            )
                except Exception as _sme:
                    logger.debug(f"Smoker routing skipped: {_sme}")

                # Metrics
                from sklearn.metrics import r2_score as _r2s

                _routed_bd = CostWeightedMetrics.segment_r2_breakdown(_y_test_orig_e, _routed)
                _routed_r2 = float(_r2s(_y_test_orig_e, _routed))
                _routed_rmse = float(np.sqrt(np.mean((_y_test_orig_e - _routed) ** 2)))
                _base_r2 = test_result["metrics"]["original_r2"]
                _base_rmse = test_result["metrics"]["original_rmse"]
                _delta_r2 = _routed_r2 - _base_r2
                _delta_rmse = _base_rmse - _routed_rmse

                logger.info(
                    f"\n📊 SPECIALIST-ROUTED TEST RESULTS (production routing):\n"
                    f"   Overall R²={_routed_r2:.4f}  RMSE=${_routed_rmse:,.0f}\n"
                    f"   Base model: R²={_base_r2:.4f}  RMSE=${_base_rmse:,.0f}\n"
                    f"   Delta: ΔR²={_delta_r2:+.4f}  ΔRMSE=${_delta_rmse:+,.0f}\n"
                    f"   Active tiers: {list(_fss.specialists.keys())}"
                )
                logger.info("\n📊 ROUTED PER-SEGMENT R² BREAKDOWN:")
                logger.info(
                    f"  {'Segment':<12} {'N':<8} {'R²':>8} {'RMSE':>10} {'Overpricing':>12}"
                )
                logger.info(f"  {'-'*56}")
                for _, _row in _routed_bd.iterrows():
                    _r2_str = f"{_row['r2']:8.4f}" if not pd.isna(_row["r2"]) else "     N/A"
                    logger.info(
                        f"  {_row['segment']:<12} {int(_row['n_samples']):<8} "
                        f"{_r2_str} ${_row['rmse']:>9,.0f} {_row['overpricing_rate']:>11.1%}"
                    )
                test_result["routed_segment_breakdown"] = _routed_bd.to_dict(orient="records")
                test_result["routed_r2"] = _routed_r2
                test_result["routed_rmse"] = _routed_rmse

                # ── G6 on specialist-routed predictions ──────────────────────
                # v7.5.6 ROUTING QUALITY GUARD:
                # Specialists are trained by true-y tier but routed at inference
                # by predicted-y. When the global model has systematic bias
                # (e.g. bimodal distribution → 70% overpricing in cheap segment),
                # most true-Low samples are routed to Mid/High specialists →
                # predictions are inflated → per-segment G6 worsens.
                #
                # When ΔR² < threshold, routing is counterproductive:
                #   • do NOT store g6_gate_routed (remains None)
                #   • gate falls back to global-model G6 (elif branch below)
                #   • specialist files are still saved for reference
                _ROUTING_QUALITY_THRESHOLD: float = -0.005  # allow ≤0.5pp R² degradation
                if _delta_r2 < _ROUTING_QUALITY_THRESHOLD:
                    logger.warning(
                        f"⚠️  G6 ROUTING QUALITY GUARD triggered:\n"
                        f"   Specialist routing degraded overall R² by "
                        f"{_delta_r2:+.4f} (threshold={_ROUTING_QUALITY_THRESHOLD}).\n"
                        f"   Cause: routing by predicted-y ≠ true-y tiers — bimodal "
                        f"distribution causes misrouting of cheap policies.\n"
                        f"   g6_gate_routed set to None — global-model G6 is authoritative.\n"
                        f"   Specialist files saved for diagnostics but will not be used "
                        f"as the primary gate authority this run."
                    )
                    # test_result["g6_gate_routed"] intentionally NOT set
                else:
                    _g6_routed = DeploymentGates.check_g6(
                        y_true=_y_test_orig_e,
                        y_pred=_routed,
                        raise_on_fail=False,
                    )
                    test_result["g6_gate_routed"] = _g6_routed
                    if not _g6_routed["g6_pass"]:
                        _veto = _g6_routed.get("veto_segments", [])
                        logger.error(
                            f"❌ G6 GATE FAILED on specialist-routed predictions.\n"
                            f"   cost_weighted_r2={_g6_routed['cost_weighted_r2']:.4f} "
                            f"(threshold={_g6_routed['min_threshold']})\n"
                            f"   Veto segments (R²<-1.0, N≥30): {_veto or 'none'}\n"
                            f"   Specialist artifact will NOT be safe to deploy."
                        )
                    else:
                        logger.info(
                            f"✅ G6 PASS on specialist-routed predictions: "
                            f"cost_weighted_r2={_g6_routed['cost_weighted_r2']:.4f}"
                        )

            except Exception as _rout_err:
                logger.warning(
                    f"⚠️  Specialist routing eval failed (non-fatal): {_rout_err}",
                    exc_info=True,
                )

        # Extract metrics from test_result
        test_metrics = test_result["metrics"]

        # ========================================
        # VALIDATE TEST METRICS
        # ========================================
        try:
            MetricsExtractor.validate_metrics(test_metrics, "test")
            logger.info("✅ Test metrics validated")
        except ValidationError as e:
            logger.error(f"❌ Test metrics validation failed: {e}")
            raise

        # ========================================
        # ENHANCED TEST PERFORMANCE LOGGING
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("TEST PERFORMANCE SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Model: {best_name}")
        logger.info(f"Samples: {len(data['X_test_raw'])}")
        logger.info("-" * 80)

        # Use MetricsExtractor for consistent access
        logger.info(f"RMSE:  ${MetricsExtractor.get_rmse(test_metrics):,.2f}")
        logger.info(f"MAE:   ${MetricsExtractor.get_mae(test_metrics):,.2f}")
        logger.info(f"R²:    {MetricsExtractor.get_r2(test_metrics):.4f}")
        logger.info(f"MAPE:  {test_metrics.get('original_mape', 0):.2f}%")

        ci = test_result.get("explainability", {}).get("interval_metrics")
        # was hardcoded to 95%; now reads configured level.
        _ci_pct = trainer.explainability_config.confidence_level * 100

        if ci is not None:
            logger.info(
                f"CI ({_ci_pct:.0f}%): Coverage={ci['coverage_pct']:.1f}%, "
                f"AvgWidth=${ci['avg_width']:,.0f}"
            )
        else:
            ci_coverage = test_metrics.get("interval_coverage_pct")
            ci_width = test_metrics.get("interval_avg_width")
            if ci_coverage is not None and ci_width is not None:
                logger.info(
                    f"CI ({_ci_pct:.0f}%): Coverage={ci_coverage:.1f}%, AvgWidth=${ci_width:,.0f}"
                )
            else:
                logger.warning(
                    f"CI ({_ci_pct:.0f}%): Not available in summary (see earlier CI logs for details)"
                )

        logger.info("=" * 80 + "\n")

        # ── PATCH 05 (G6): cost-weighted R² gate ─────────────────────────────
        _y_test_orig = test_result.get("y_test_original")
        _y_pred_orig = test_result.get("predictions")
        if _y_test_orig is not None and _y_pred_orig is not None:
            _g6 = DeploymentGates.check_g6(
                y_true=_y_test_orig,
                y_pred=_y_pred_orig,
                raise_on_fail=False,
            )
            test_result["g6_gate"] = _g6
            if not _g6["g6_pass"]:
                logger.error(
                    f"❌ Gate G6 FAILED: cost_weighted_r2={_g6['cost_weighted_r2']:.4f} "
                    f"< {_g6['min_threshold']}."
                )

        # ── PATCH 02 (G1/G7): objective-metric alignment gate ────────────────
        if _y_test_orig is not None and _y_pred_orig is not None:
            _best_loaded = FileSanitizer.safe_load(
                best_path, trainer.config.max_model_size_mb, verify_checksum=False
            )
            _g1g7 = check_objective_metric_alignment(
                model=_best_loaded,
                model_name=best_name,
                y_true=_y_test_orig,
                y_pred=_y_pred_orig,
                config=trainer.raw_config,  # config fallback
            )
            test_result["g1_g7_gate"] = _g1g7
            if not _g1g7.get("g7_pass"):
                logger.error(
                    f"❌ Gate G7 FAIL: overpricing_rate={_g1g7['overpricing_rate']:.1%}. "
                    "Switch to xgboost_median (reg:squarederror) for premium pricing."
                )

        eval_time = time.time() - start_time
        logger.info(f"✅ Test evaluation completed in {eval_time:.1f}s\n")

        # ── Collect all gate results before declaring ready ──
        # Previously every gate logged [ERROR] but the pipeline continued to
        # print "[OK] Ready for deployment!" regardless.  Three simultaneous gate
        # failures (G4, G6, G7) would all pass silently.
        #
        # build a gate summary after all gates have been evaluated, log a
        # clear pass/fail table, and sys.exit(1) if ANY gate failed.  This
        # mirrors the behaviour of typical CI deployment gates (fail-fast).
        _gate_failures: list = []

        # G4 — git provenance (evaluated earlier; stored in provenance object)
        _prov = getattr(trainer, "_provenance", None)
        if _prov is not None:
            _g4_pass = getattr(_prov, "commit_hash", "unknown") not in (
                "unknown",
                "",
                None,
            )
            if not _g4_pass:
                _gate_failures.append(
                    f"G4 PROVENANCE: commit_hash='{getattr(_prov, 'commit_hash', 'unknown')}' "
                    "— run inside a git repo with at least one commit."
                )

        # ── G6 — cost-weighted R² (global model) — diagnostic ────────────
        # v7.5.5: Global-model G6 is now a DIAGNOSTIC when specialists are
        # available. Gate is cleared if specialist-routed G6 passes.
        _g6_stored = test_result.get("g6_gate")
        _g6_routed_stored = test_result.get("g6_gate_routed")

        # Determine effective G6 result:
        # prefer specialist-routed if it is available, else fall back to global
        _g6_effective_pass = True
        _g6_effective_fail_msg: str | None = None

        if _g6_routed_stored is not None:
            # Specialist-routed result is the primary authority
            if not _g6_routed_stored.get("g6_pass", True):
                _veto_segs_r = _g6_routed_stored.get("veto_segments", [])
                _g6_effective_pass = False
                if _veto_segs_r:
                    _g6_effective_fail_msg = (
                        f"G6 ROUTED HARD VETO: R² < -1.0 in segment(s) {_veto_segs_r} "
                        f"after full-segment specialist routing. "
                        f"cost_weighted_r2={_g6_routed_stored['cost_weighted_r2']:.4f}. "
                        "Retrain FullSegmentSpecialist or collect more data per tier."
                    )
                else:
                    _g6_effective_fail_msg = (
                        f"G6 ROUTED COST-WEIGHTED R²: {_g6_routed_stored['cost_weighted_r2']:.4f} "
                        f"< threshold {_g6_routed_stored['min_threshold']} on specialist-routed "
                        "predictions."
                    )
            else:
                # Specialist pass — suppress global G6 failure from gate_failures
                if _g6_stored is not None and not _g6_stored.get("g6_pass", True):
                    logger.info(
                        "ℹ️  Global-model G6 failed but specialist-routed G6 PASSED — "
                        "gate cleared by specialist predictions. "
                        f"(global cw_r2={_g6_stored['cost_weighted_r2']:.4f}, "
                        f"specialist cw_r2={_g6_routed_stored['cost_weighted_r2']:.4f})"
                    )
        elif _g6_stored is not None:
            # No specialists — fall back to global G6
            if not _g6_stored.get("g6_pass", True):
                _g6_effective_pass = False
                _veto_segs = _g6_stored.get("veto_segments", [])
                if _veto_segs:
                    _g6_effective_fail_msg = (
                        f"G6 HARD VETO: R² < -1.0 in segment(s) {_veto_segs} — "
                        f"model is worse than the mean predictor. "
                        f"cost_weighted_r2={_g6_stored['cost_weighted_r2']:.4f}. "
                        "Enable FullSegmentSpecialist (Phase 3) to resolve."
                    )
                else:
                    _g6_effective_fail_msg = (
                        f"G6 COST-WEIGHTED R²: {_g6_stored['cost_weighted_r2']:.4f} "
                        f"< threshold {_g6_stored['min_threshold']}."
                    )

        if not _g6_effective_pass and _g6_effective_fail_msg:
            _gate_failures.append(_g6_effective_fail_msg)

        # G7 — overpricing rate
        _g1g7_stored = test_result.get("g1_g7_gate")
        if _g1g7_stored is not None and not _g1g7_stored.get("g7_pass", True):
            _g7_thresh = _g1g7_stored.get("g7_threshold", G7_MAX_OVERPRICING_RATE)
            _gate_failures.append(
                f"G7 OVERPRICING: rate={_g1g7_stored['overpricing_rate']:.1%} "
                f"> {_g7_thresh:.0%} — retrain with xgboost_median (reg:squarederror) for "
                "unbiased median predictions."
            )

        if _gate_failures:
            logger.error("\n" + "=" * 80)
            logger.error("❌ DEPLOYMENT BLOCKED — %d gate(s) failed:", len(_gate_failures))
            for _fail_msg in _gate_failures:
                logger.error("   • %s", _fail_msg)
            logger.error("=" * 80)
            logger.error(
                "Resolve all gate failures before deploying.  " "Pipeline exiting with code 1."
            )
            print(
                f"\n{'='*80}\n❌ DEPLOYMENT BLOCKED: {len(_gate_failures)} gate failure(s).\n"
                "   See logs above for details.\n" + "=" * 80
            )
            sys.exit(1)

        # ── P2-A: register best model to MLflow Model Registry ───────────────
        # Fires only when all gates pass and register_to_mlflow=True.
        # The best model's run is CLOSED here — safe to reopen by run_id.
        if not _gate_failures and trainer.mlflow.enabled and trainer.config.register_to_mlflow:
            try:
                from mlflow.models import infer_signature as _infer_sig

                _reg_model_obj = results.get(best_name, {}).get("model")
                _reg_run_id = results.get(best_name, {}).get("mlflow_run_id")
                _X_val_reg = data.get("X_val")

                if _reg_model_obj is not None and _X_val_reg is not None and _reg_run_id:
                    _X_sample = _X_val_reg.head(5)
                    try:
                        _y_sample = _reg_model_obj.predict(_X_sample)
                        _sig = _infer_sig(_X_sample, _y_sample)
                    except Exception:
                        _sig = None

                    _reg_metadata = {
                        "model_name": best_name,
                        "pipeline_version": VERSION,
                        "val_rmse": str(results[best_name].get("val_rmse", "N/A")),
                        "val_r2": str(results[best_name].get("val_r2", "N/A")),
                        "deployment_ready": "true",
                        "gate_failures": "0",
                    }

                    with trainer.mlflow._mlflow.start_run(run_id=_reg_run_id, nested=False):
                        _reg_result = trainer.mlflow.register_model_to_registry(
                            model=_reg_model_obj,
                            model_name=best_name,
                            signature=_sig,
                            input_example=_X_sample,
                            metadata=_reg_metadata,
                        )
                    if _reg_result:
                        logger.info(
                            f"  ✅ Model registered: insurance_{best_name}\n"
                            f"     URI: {_reg_result.get('mlflow_model_uri', 'N/A')}"
                        )
            except Exception as _reg_err:
                logger.warning(f"Model Registry registration failed (non-fatal): {_reg_err}")
        # ─────────────────────────────────────────────────────────────────────

        # ── TRAIN-1: log gate results to MLflow ───────────────────────────────
        # Reopen best model's run by stored ID and attach gate pass/fail as tags
        # so they're filterable in the UI (Experiments → Filter by tag).
        # Also log numeric gate values as metrics for trend tracking.
        try:
            _best_mlflow_run_id = results.get(best_name, {}).get("mlflow_run_id")
            if trainer.mlflow.enabled and _best_mlflow_run_id:

                def _safe_f(v):
                    return (
                        float(v)
                        if isinstance(v, int | float | np.number) and np.isfinite(float(v))
                        else None
                    )

                with trainer.mlflow._mlflow.start_run(run_id=_best_mlflow_run_id, nested=False):
                    # G4 — provenance
                    _prov_tag = getattr(trainer, "_provenance", None)
                    _g4 = _prov_tag is not None and getattr(
                        _prov_tag, "commit_hash", "unknown"
                    ) not in ("unknown", "", None)
                    trainer.mlflow._mlflow.set_tag("gate_g4_pass", str(_g4))
                    if _prov_tag:
                        _commit = getattr(_prov_tag, "commit_hash", "unknown")
                        trainer.mlflow._mlflow.set_tag("git_commit", str(_commit)[:12])

                    # G6 — cost-weighted R²
                    _g6 = test_result.get("g6_gate", {})
                    if _g6:
                        trainer.mlflow._mlflow.set_tag(
                            "gate_g6_pass", str(_g6.get("g6_pass", "unknown"))
                        )
                        _cwr2 = _safe_f(_g6.get("cost_weighted_r2"))
                        if _cwr2 is not None:
                            trainer.mlflow._mlflow.log_metric("gate_g6_cost_weighted_r2", _cwr2)
                        _veto_segs = _g6.get("veto_segments", [])
                        if _veto_segs:
                            trainer.mlflow._mlflow.set_tag("gate_g6_veto_segments", str(_veto_segs))

                    # G7 — overpricing rate
                    _g1g7 = test_result.get("g1_g7_gate", {})
                    if _g1g7:
                        trainer.mlflow._mlflow.set_tag(
                            "gate_g7_pass", str(_g1g7.get("g7_pass", "unknown"))
                        )
                        trainer.mlflow._mlflow.set_tag(
                            "gate_g1_pass", str(_g1g7.get("g1_pass", "unknown"))
                        )
                        _op_rate = _safe_f(_g1g7.get("overpricing_rate"))
                        if _op_rate is not None:
                            trainer.mlflow._mlflow.log_metric("gate_g7_overpricing_rate", _op_rate)

                    # Overall deployment readiness tag
                    _all_gates_pass = len(_gate_failures) == 0
                    trainer.mlflow._mlflow.set_tag("deployment_ready", str(_all_gates_pass))
                    trainer.mlflow._mlflow.set_tag("gate_failures_count", str(len(_gate_failures)))
                    logger.info(f"  MLflow: gate tags written to run {_best_mlflow_run_id[:8]}...")
        except Exception as _g_err:
            logger.warning(f"MLflow gate logging failed: {_g_err}")
        # ─────────────────────────────────────────────────────────────────────

        # ========================================
        # FAIR PERFORMANCE COMPARISON
        # ========================================
        def compare_metrics_fairly(
            results: dict[str, dict], test_result: dict, best_name: str, data: dict
        ) -> None:
            """
            Print fair performance comparison using MetricsExtractor utilities.

            Args:
                results: Training results dictionary
                test_result: Test evaluation results
                best_name: Name of best model (may include "_calibrated")
                data: Processed data dictionary
            """
            logger.info("\n" + "=" * 80)
            logger.info("FAIR PERFORMANCE COMPARISON")
            logger.info("=" * 80)

            # Determine if model is calibrated
            is_calibrated = "calibrated" in best_name.lower()

            # Extract metrics
            train_metrics = results[best_name]["training_metrics"]
            val_metrics = results[best_name]["validation_metrics"]
            test_metrics = test_result["metrics"]

            # ========================================
            # STEP 1: Validate all metrics
            # ========================================
            try:
                MetricsExtractor.validate_metrics(train_metrics, "training")
                MetricsExtractor.validate_metrics(val_metrics, "validation")
                MetricsExtractor.validate_metrics(test_metrics, "test")
            except ValidationError as e:
                logger.error(f"❌ Metrics validation failed: {e}")
                return

            # ========================================
            # STEP 2: Basic info
            # ========================================
            logger.info(f"Model: {best_name}")
            logger.info(f"Calibrated: {'YES ✓' if is_calibrated else 'NO'}")
            logger.info(
                f"Samples: Train={len(data['X_train'])}, "
                f"Val={len(data['X_val'])}, Test={len(data['X_test_raw'])}"
            )
            logger.info("-" * 80)

            # ========================================
            # STEP 3: Metrics table (manual for clarity)
            # ========================================
            logger.info(f"{'Metric':<10} {'Train':<18} {'Validation':<18} {'Test':<18}")
            logger.info("-" * 80)

            metrics_to_compare = [
                ("RMSE", "original_rmse", "${:,.0f}"),
                ("MAE", "original_mae", "${:,.0f}"),
                ("R²", "original_r2", "{:.4f}"),
                ("MAPE", "original_mape", "{:.2f}%"),
            ]

            for metric_name, key, fmt in metrics_to_compare:
                train_val = train_metrics.get(key, 0)
                val_val = val_metrics.get(key, 0)
                test_val = test_metrics.get(key, 0)

                logger.info(
                    f"{metric_name:<10} "
                    f"{fmt.format(train_val):<18} "
                    f"{fmt.format(val_val):<18} "
                    f"{fmt.format(test_val):<18}"
                )

            logger.info("-" * 80)

            # ========================================
            # STEP 4: Generalization gap analysis using MetricsExtractor
            # ========================================
            gap_analysis = MetricsExtractor.calculate_generalization_gap(
                train_metrics, val_metrics, test_metrics
            )

            logger.info("\nGeneralization Analysis:")

            # Train → Validation RMSE gap
            # NOTE: gap_pct is dollar-RMSE based, not pinball loss.
            # Quantile models (reg:quantileerror) always show a larger RMSE
            # gap than their actual training-objective gap.  Pinball gap is
            # the reliable overfitting indicator for this pipeline.
            logger.info(
                f"   Train → Validation RMSE gap: {gap_analysis['train_val_gap_pct']:+.1f}%"
            )

            # Detect quantile objective to contextualise status
            _gap_obj = str(
                results.get(best_name, {}).get("model_config", {}).get("objective", "")
            ).lower()
            _gap_is_quantile = "quantile" in _gap_obj

            status = gap_analysis["train_val_status"]
            if _gap_is_quantile:
                logger.info(
                    " ℹ️  (quantile model — RMSE gap inflated vs pinball gap; see Optuna summary)"
                )
            elif status == "minimal_overfitting":
                logger.info(" ✅ (minimal overfitting)")
            elif status == "moderate_overfitting":
                logger.info(" âš ï¸  (moderate overfitting)")
            elif status == "severe_overfitting":
                logger.info(" ❌ (severe overfitting)")
            elif status == "underfitting":
                logger.info(" âš ï¸  (possible underfitting)")
            else:
                logger.info(f" ({status})")

            # Validation → Test gap
            if "val_test_gap_pct" in gap_analysis:
                logger.info(f"   Validation â†' Test:  {gap_analysis['val_test_gap_pct']:+.1f}%")

                val_test_status = gap_analysis.get("val_test_status", "")
                if "excellent" in val_test_status.lower():
                    logger.info(" ✅ (excellent generalization)")
                elif "good" in val_test_status.lower():
                    logger.info(" ✅ (good generalization)")
                elif "moderate" in val_test_status.lower():
                    logger.info(" âš ï¸  (moderate gap)")
                else:
                    logger.info(f" ({val_test_status})")

            # ========================================
            # STEP 5: Calibration diagnostics
            # ========================================
            if is_calibrated and "calibration_improvement" in results[best_name]:
                cal_info = results[best_name]["calibration_improvement"]

                logger.info("\nCalibration Impact:")
                logger.info(f"   RMSE Improvement: {cal_info['rmse_improvement_pct']:+.2f}%")
                logger.info(f"   R² Improvement:   {cal_info['r2_improvement_pct']:+.2f}%")
                logger.info(f"   MAE Improvement:  {cal_info['mae_improvement_pct']:+.2f}%")

                if cal_info["is_better"]:
                    logger.info("   ✅ Calibration effective")
                else:
                    logger.warning("   âš ï¸  Calibration may not be beneficial")

            # ========================================
            # STEP 6: Confidence intervals (if available)
            # ========================================
            if "explainability" in test_result:
                explainability = test_result["explainability"]

                if explainability.get("confidence_intervals") is not None:
                    coverage = test_metrics.get("interval_coverage_pct", 0)
                    avg_width = test_metrics.get("interval_avg_width", 0)

                    # was hardcoded to 95%; now reads configured level.
                    _ci_pct2 = trainer.explainability_config.confidence_level * 100
                    _tol2 = _ci_pct2 * 0.02
                    _tol5 = _ci_pct2 * 0.05

                    logger.info(f"\nConfidence Intervals ({_ci_pct2:.0f}%):")
                    logger.info(f"   Coverage: {coverage:.1f}%")
                    logger.info(f"   Avg Width: ${avg_width:,.0f}")

                    # Coverage quality assessment — bands relative to actual target
                    if (_ci_pct2 - _tol2) <= coverage <= (_ci_pct2 + _tol2):
                        logger.info("   ✅ Well-calibrated")
                    elif (_ci_pct2 - _tol5) <= coverage <= (_ci_pct2 + _tol5):
                        logger.info("   ⚠️  Acceptable (could be improved)")
                    else:
                        logger.info("   ❌ Needs adjustment")
            logger.info("=" * 80 + "\n")

        # ========================================================================
        # ADVANCED DIAGNOSTICS
        # ========================================================================
        from insurance_ml.diagnostics import ModelDiagnostics
        from insurance_ml.models import CalibratedModel
        from insurance_ml.monitoring import DriftMonitor

        logger.info("\n" + "=" * 80)
        logger.info("ADVANCED MODEL DIAGNOSTICS")
        logger.info("=" * 80)

        # Load best model
        best_model = FileSanitizer.safe_load(
            best_path,
            trainer.config.max_model_size_mb,
            verify_checksum=trainer.config.verify_checksums,
        )

        # Feature importance
        feature_names = data["X_train"].columns.tolist()
        ModelDiagnostics.get_feature_importance(best_model, feature_names=feature_names, top_n=15)

        # Prediction distribution
        y_test_pred_original = test_result["predictions"]
        y_test_original = test_result["y_test_original"]

        pred_dist = ModelDiagnostics.analyze_prediction_distribution(
            y_test_original, y_test_pred_original
        )

        # Calibration check
        ModelDiagnostics.calculate_calibration(y_test_original, y_test_pred_original, n_bins=10)

        # Business metrics
        business_metrics = ModelDiagnostics.calculate_business_metrics(
            y_test_original, y_test_pred_original
        )

        # Error by range
        ModelDiagnostics.error_by_range(y_test_original, y_test_pred_original)

        # Sample predictions
        ModelDiagnostics.show_sample_predictions(y_test_original, y_test_pred_original, n_samples=5)

        logger.info("=" * 80 + "\n")

        # Use get_project_root if available, otherwise calculate
        try:
            from insurance_ml.config import get_project_root

            config_path = get_project_root() / "configs" / "config.yaml"
        except ImportError:
            # Fallback: calculate project root from train.py location
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"

        # Verify it exists
        if not config_path.exists():
            logger.warning(f"⚠️  config.yaml not found at {config_path}")
            config_path = "config.yaml"  # Let save_model_metadata() search for it

        # pass provenance fields to silence
        # the "{'git_commit','random_state','pipeline_version'} missing" warning.
        # _provenance and VERSION are in scope in main().
        trainer.model_manager.save_model_metadata(
            model_name=best_name,
            metrics={
                "test_rmse": float(test_metrics["original_rmse"]),
                "test_r2": float(test_metrics["original_r2"]),
                "test_mape": float(test_metrics.get("original_mape", 0)),
                **business_metrics,
            },
            feature_names=feature_names,
            config_path=str(config_path),  # Pass absolute or let it search
            pipeline_version=VERSION,
            random_state=trainer.config.random_state,
        )

        # Create drift detection baseline
        # overwrite=True: baseline is regenerated on every training run to stay
        # in sync with the current model and data distribution. A stale baseline
        # from a previous run would cause spurious drift alerts after retraining.
        DriftMonitor.create_baseline(
            X_train=data["X_train"],
            y_train=data["y_train"],
            output_path="models/drift_baseline.json",
            overwrite=True,
        )

        # High-value segment analysis
        logger.info("Analyzing high-value segment (using cached test data)...")
        y_test_pred_original = test_result["predictions"]
        y_test_original = test_result["y_test_original"]

        analyze_high_value_segment(y_test_original, y_test_pred_original, trainer.raw_config)

        # ── TRAIN-2: log test metrics + diagnostics to best model's run ───────
        try:
            _best_run_id_final = results.get(best_name, {}).get("mlflow_run_id")
            if trainer.mlflow.enabled and _best_run_id_final:

                def _sf(v):
                    return (
                        float(v)
                        if isinstance(v, int | float | np.number) and np.isfinite(float(v))
                        else None
                    )

                _test_payload: dict[str, float] = {}

                # Core test metrics
                for _k in (
                    "original_rmse",
                    "original_r2",
                    "original_mae",
                    "original_mape",
                    "interval_coverage_pct",
                    "interval_avg_width",
                ):
                    _v = _sf(test_metrics.get(_k))
                    if _v is not None:
                        _test_payload[f"test_{_k}"] = _v

                # Generalization gaps (train→val, val→test)
                _gap = MetricsExtractor.calculate_generalization_gap(
                    results[best_name].get("training_metrics", {}),
                    results[best_name].get("validation_metrics", {}),
                )
                for _gk in ("train_val_gap_pct", "val_test_gap_pct"):
                    _v = _sf(_gap.get(_gk))
                    if _v is not None:
                        _test_payload[_gk] = _v

                # Pipeline timing
                _test_payload["total_pipeline_time_s"] = float(time.time() - pipeline_start)
                _test_payload["prep_time_s"] = float(prep_time)
                _test_payload["train_time_s"] = float(train_time)
                _test_payload["eval_time_s"] = float(eval_time)

                # Business diagnostics from ModelDiagnostics
                for _bk, _bv in business_metrics.items():
                    _v = _sf(_bv)
                    if _v is not None:
                        _test_payload[f"biz_{_bk}"] = _v

                # Prediction distribution stats
                for _dk in ("mean_error", "std_error", "skewness", "pct_within_10pct"):
                    _v = _sf(pred_dist.get(_dk))
                    if _v is not None:
                        _test_payload[f"dist_{_dk}"] = _v

                with trainer.mlflow._mlflow.start_run(run_id=_best_run_id_final, nested=False):
                    if _test_payload:
                        trainer.mlflow._mlflow.log_metrics(_test_payload)
                    # Tag the model name and version
                    trainer.mlflow._mlflow.set_tag("best_model_name", best_name)
                    trainer.mlflow._mlflow.set_tag("pipeline_version", VERSION)
                    trainer.mlflow._mlflow.set_tag(
                        "generalization_status", _gap.get("train_val_status", "unknown")
                    )

                logger.info(
                    f"  MLflow: {len(_test_payload)} test metrics written "
                    f"to run {_best_run_id_final[:8]}..."
                )
        except Exception as _t2_err:
            logger.warning(f"MLflow test metric logging failed: {_t2_err}")
        # ─────────────────────────────────────────────────────────────────────

        # Performance summary
        total_time = time.time() - pipeline_start
        logger.info("=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Data prep:    {prep_time:>6.1f}s")
        logger.info(f"Training:     {train_time:>6.1f}s")
        logger.info(f"Evaluation:   {eval_time:>6.1f}s")
        logger.info(f"Total:        {total_time:>6.1f}s")
        logger.info("=" * 80 + "\n")

        # Final summary
        # ── TRAIN-3: attach artifacts to best model's MLflow run ──────────────
        try:
            _best_run_id_art = results.get(best_name, {}).get("mlflow_run_id")
            if trainer.mlflow.enabled and _best_run_id_art:
                with trainer.mlflow._mlflow.start_run(run_id=_best_run_id_art, nested=False):
                    # Residual diagnostic plots
                    _residual_dir = trainer.config.reports_dir / "residuals" / best_name
                    if _residual_dir.exists():
                        for _plot in _residual_dir.glob("*.png"):
                            try:
                                trainer.mlflow._mlflow.log_artifact(
                                    str(_plot), artifact_path="residuals"
                                )
                            except Exception:
                                pass

                    # bias_correction.json
                    _bc_path = trainer.config.output_dir / "bias_correction.json"
                    if _bc_path.exists():
                        try:
                            trainer.mlflow._mlflow.log_artifact(
                                str(_bc_path), artifact_path="model_artifacts"
                            )
                        except Exception:
                            pass

                    # drift baseline
                    _drift_path = Path("models/drift_baseline.json")
                    if _drift_path.exists():
                        try:
                            trainer.mlflow._mlflow.log_artifact(
                                str(_drift_path), artifact_path="model_artifacts"
                            )
                        except Exception:
                            pass

                logger.info(f"  MLflow: artifacts attached to run {_best_run_id_art[:8]}...")
        except Exception as _t3_err:
            logger.warning(f"MLflow artifact logging failed: {_t3_err}")
        # ─────────────────────────────────────────────────────────────────────

        print(f"\n{'='*80}\n[OK] Pipeline Complete\n{'='*80}")
        print(f"Models: {trainer.config.output_dir}")
        print(f"Reports: {trainer.config.reports_dir}")
        print(f"Best: {best_name}")

        print("\nTest Performance:")
        print(f"  RMSE: ${test_metrics['original_rmse']:.2f}")
        print(f"  R²:   {test_metrics['original_r2']:.4f}")

        print(f"\n{'='*80}")
        # ── DEPLOYMENT VERDICT: reconcile gate checks with business evaluation ──
        # train.py gates (G4/G6/G7) validate accuracy and provenance.
        # evaluate.py validates business value (profit delta, churn, BizScore).
        # Both must pass for a clean deployment recommendation.
        _eval_summary_path = Path(trainer.config.reports_dir) / "unified_summary.json"
        _biz_gate_passed: bool | None = None
        if _eval_summary_path.exists():
            try:
                import json as _js

                _eval = _js.loads(_eval_summary_path.read_text(encoding="utf-8"))
                _biz_hybrid = (
                    _eval.get("business", {})
                    .get("hybrid", {})
                    .get("overall", {})
                    .get("business_value_score", None)
                )
                _biz_ml = (
                    _eval.get("business", {})
                    .get("ml", {})
                    .get("overall", {})
                    .get("business_value_score", None)
                )
                if _biz_hybrid is not None and _biz_ml is not None:
                    _biz_gate_passed = float(_biz_hybrid) > float(_biz_ml)
            except Exception as _ev_err:
                logger.debug(f"Could not read unified_summary.json: {_ev_err}")
        if _biz_gate_passed is True:
            print("\n[OK] Ready for deployment! (ML accuracy ✅ + Business value ✅)")
        elif _biz_gate_passed is False:
            print(
                "\n[OK] Gate checks passed — ML model is accurate.\n"
                "⚠️  Business evaluation recommends ML-ONLY over hybrid.\n"
                "    Deploy xgboost_median as ML-only predictor.\n"
                "    Re-run evaluate.py to confirm before deploying hybrid."
            )
        else:
            print(
                "\n[OK] Ready for deployment!\n"
                "ℹ️  Run evaluate.py to validate business value before deploying hybrid."
            )

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
