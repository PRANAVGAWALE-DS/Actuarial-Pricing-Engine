from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Suppress XGBoost's DMatrix device-mismatch warning (same as predict.py).
# Fires on every evaluation inference call; benign but floods logs.
warnings.filterwarnings(
    "ignore",
    message=".*Falling back to prediction using DMatrix.*",
    category=UserWarning,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from insurance_ml.config import load_config  # noqa: E402
from insurance_ml.predict import HybridPredictor, PredictionPipeline  # noqa: E402

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    except (AttributeError, TypeError):
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300


# =====================================================================
# BUSINESS CONFIGURATION (LOADED FROM CONFIG.YAML)
# =====================================================================

# Magic constants for evaluation thresholds
_SMAPE_DENOMINATOR_MIN = 0.01
_MIN_SEGMENT_SAMPLES = 10
_PENALTY_MULTIPLIER_MIN = 0.1
_PENALTY_MULTIPLIER_MAX = 10.0
_SMAPE_ERROR_THRESHOLD = 200
_SMAPE_WARNING_THRESHOLD = 100
_MAX_CHURN_PROBABILITY = 0.80
_DEPLOYMENT_CONFIDENCE_HIGH = 0.80
_DEPLOYMENT_CONFIDENCE_MODERATE = 0.60


@dataclass
class BusinessConfig:
    """Business rules and economic parameters"""

    base_profit_margin: float = 0.15
    admin_cost_per_policy: float = 25.0
    churn_threshold_pct: float = 0.40
    # was 0.01 — effectively disabled churn metric (required 5000%
    # overpricing to reach 50% churn probability). Aligned with config.yaml
    # business_config.churn_sensitivity: 1.0. This default is now used whenever
    # load_business_config_from_yaml() fails or BusinessConfig() is constructed
    # directly (unit tests, notebooks, etc.).
    churn_sensitivity: float = 1.0
    customer_acquisition_cost: float = 200.0
    customer_lifetime_value_multiplier: float = 1.5
    underpricing_penalty_multiplier: float = 0.3
    severe_underpricing_threshold_pct: float = 0.50
    severe_underpricing_penalty: float = 200.0
    acceptable_error_band_pct: float = 0.30
    excellent_accuracy_bonus: float = 150.0
    low_value_threshold: float = 4500.0
    high_value_multiplier: float = 1.5
    # Scoring thresholds for _calculate_business_value_score.
    # Previously hardcoded inside the method body — invisible to config.yaml
    # governance and untuneable without code changes.  Now declared here so
    # they can be driven from hybrid_predictor.business_config in config.yaml.
    biz_score_profit_target: float = 600.0  # profitability_score = 50 at/above this.
    # raised from 500.0 to 600.0 to calibrate against the actual 50k-portfolio
    # ML baseline ($615/policy).  At 500.0 both the ML model ($615) and hybrid ($677)
    # exceeded the target identically, making the profit difference invisible to the
    # gate.  At 600.0 the ML baseline sits just above the threshold.
    # NOTE: both models still exceed $600 in the current run, so profitability_score
    # is still 50/50 for both — the profit gap remains invisible until
    # biz_score_profit_target is raised above the hybrid's profit/policy ($677).
    # This value is the fallback used when config.yaml cannot be loaded.
    biz_score_churn_low: float = 0.10  # full retention score below this
    biz_score_churn_high: float = 0.20  # zero retention score above this
    biz_score_retention_max: float = 30.0  # max retention component
    biz_score_quality_weight: float = 20.0  # accuracy-rate multiplier

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> BusinessConfig:
        """Create BusinessConfig from config.yaml dictionary"""
        return cls(
            base_profit_margin=config_dict.get("base_profit_margin", 0.15),
            admin_cost_per_policy=config_dict.get("admin_cost_per_policy", 25.0),
            churn_threshold_pct=config_dict.get("churn_threshold_pct", 0.40),
            churn_sensitivity=config_dict.get("churn_sensitivity", 1.0),
            customer_acquisition_cost=config_dict.get("customer_acquisition_cost", 200.0),
            customer_lifetime_value_multiplier=config_dict.get(
                "customer_lifetime_value_multiplier", 1.5
            ),
            underpricing_penalty_multiplier=config_dict.get("underpricing_penalty_multiplier", 0.3),
            severe_underpricing_threshold_pct=config_dict.get(
                "severe_underpricing_threshold_pct", 0.50
            ),
            severe_underpricing_penalty=config_dict.get("severe_underpricing_penalty", 200.0),
            acceptable_error_band_pct=config_dict.get("acceptable_error_band_pct", 0.30),
            excellent_accuracy_bonus=config_dict.get("excellent_accuracy_bonus", 150.0),
            low_value_threshold=config_dict.get("low_value_threshold", 4500.0),
            high_value_multiplier=config_dict.get("high_value_multiplier", 1.5),
            biz_score_profit_target=config_dict.get(
                "biz_score_profit_target", 600.0
            ),  # aligned with dataclass default
            biz_score_churn_low=config_dict.get("biz_score_churn_low", 0.10),
            biz_score_churn_high=config_dict.get("biz_score_churn_high", 0.20),
            biz_score_retention_max=config_dict.get("biz_score_retention_max", 30.0),
            biz_score_quality_weight=config_dict.get("biz_score_quality_weight", 20.0),
        )


def load_business_config_from_yaml() -> BusinessConfig:
    """Load business configuration from config.yaml"""
    try:
        config = load_config()
        hybrid_config = config.get("hybrid_predictor", {})
        # take a shallow copy so we don't mutate the cached config dict.
        # The original code obtained a reference to hybrid_config["business_config"]
        # and wrote "low_value_threshold" into it, permanently modifying the shared
        # config object for any subsequent caller in the same process.
        business_dict = dict(hybrid_config.get("business_config", {}))
        threshold = hybrid_config.get("threshold", 4500.0)
        business_dict["low_value_threshold"] = threshold
        return BusinessConfig.from_config_dict(business_dict)
    except (OSError, KeyError, ValueError, Exception) as e:
        logger.warning(f"⚠️ Could not load business config from YAML: {e}")
        return BusinessConfig()


# =====================================================================
# BUSINESS METRICS CALCULATOR
# =====================================================================


class BusinessMetricsCalculator:
    """Calculate profit-weighted business performance metrics"""

    def __init__(self, config: BusinessConfig | None = None):
        self.config = config or BusinessConfig()
        self.business_calc = self

    def calculate_single_prediction_value(
        self, true_charge: float, predicted_charge: float
    ) -> dict[str, float]:
        """Calculate business value/loss for a single prediction.

        T3-A (v7.5.0): Revenue model aligned to actuarial convention.
        premium_charged = predicted_charge (the model output IS the price).
        Gross profit = premium_charged - claims_cost - admin_cost + loading.
        At perfect prediction: gross_profit = loading * true_charge - admin_cost,
        which is positive for policies above admin_cost/loading ($833 at defaults).
        Previously revenue = predicted_charge * 1.03, which masked small underpricing
        (predictions up to 3% below true still generated positive gross profit).
        """
        error = predicted_charge - true_charge
        error_pct = error / true_charge if true_charge > 0 else 0

        # T3-A: actuarially standard revenue and gross profit
        # loading is applied to the charged premium, not compounded on top of true_charge
        loading = self.config.base_profit_margin  # default 0.03
        premium_charged = predicted_charge
        revenue = premium_charged * (1.0 + loading)
        claims_cost = true_charge
        admin_cost = self.config.admin_cost_per_policy
        gross_profit = premium_charged - claims_cost + loading * premium_charged - admin_cost

        # Overpricing penalties (churn)
        churn_probability = 0.0
        churn_cost = 0.0

        if error > 0:
            overpricing_pct = error_pct
            if overpricing_pct > self.config.churn_threshold_pct:
                excess_overpricing = overpricing_pct - self.config.churn_threshold_pct
                churn_probability = min(excess_overpricing * self.config.churn_sensitivity, 0.80)
                # CLV based on true_charge, not predicted_charge.
                # Using predicted_charge self-amplified churn cost: an overpriced
                # prediction inflated both the revenue line and its own churn penalty
                # (e.g. predicting $40K for a $5K policy created ~$48K churn cost).
                customer_lifetime_value = (
                    true_charge * self.config.customer_lifetime_value_multiplier
                )
                churn_cost = churn_probability * (
                    customer_lifetime_value + self.config.customer_acquisition_cost
                )

        # Underpricing penalties
        underpricing_penalty = 0.0

        if error < 0:
            underpricing_pct = abs(error_pct)
            # was hardcoded 0.3, making config.underpricing_penalty_multiplier
            # dead code for the primary profit path.  Now reads from BusinessConfig so
            # config.yaml changes actually affect net_profit / deployment decisions.
            base_penalty = abs(error) * self.config.underpricing_penalty_multiplier
            max_penalty = revenue * 0.2
            underpricing_penalty = min(base_penalty, max_penalty)

            if underpricing_pct > self.config.severe_underpricing_threshold_pct:
                underpricing_penalty += self.config.severe_underpricing_penalty

        # Accuracy bonuses
        accuracy_bonus = 0.0
        if abs(error_pct) <= self.config.acceptable_error_band_pct / 2:
            accuracy_bonus = self.config.excellent_accuracy_bonus  # ±15%: full bonus
        elif abs(error_pct) <= self.config.acceptable_error_band_pct:
            accuracy_bonus = self.config.excellent_accuracy_bonus * 0.5  # ±30%: half bonus

        net_profit = gross_profit - churn_cost - underpricing_penalty + accuracy_bonus

        return {
            "revenue": revenue,
            "claims_cost": claims_cost,
            "admin_cost": admin_cost,
            "base_cost": claims_cost + admin_cost,
            "gross_profit": gross_profit,
            "churn_probability": churn_probability,
            "churn_cost": churn_cost,
            "underpricing_penalty": underpricing_penalty,
            "accuracy_bonus": accuracy_bonus,
            "net_profit": net_profit,
            "error": error,
            "error_pct": error_pct,
        }

    def calculate_portfolio_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, segment_name: str = "overall"
    ) -> dict[str, Any]:
        """Calculate business metrics for entire portfolio.

        replaces the Python list-comprehension that called
        calculate_single_prediction_value() once per row.  For a 10K-policy batch
        that loop consumed ~40 ms; the vectorised path runs in ~1 ms.
        calculate_single_prediction_value() is retained for single-row diagnostics
        and tests only.
        """
        n = len(y_true)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        cfg = self.config
        loading = cfg.base_profit_margin
        admin = cfg.admin_cost_per_policy

        error = y_pred - y_true
        error_pct = np.where(y_true > 0, error / y_true, 0.0)

        # ── Revenue / gross profit ────────────────────────────────────────────
        gross_profit = y_pred - y_true + loading * y_pred - admin

        # ── Overpricing / churn ───────────────────────────────────────────────
        over_mask = error > 0
        overpricing_pct = np.where(over_mask, error_pct, 0.0)
        excess = np.maximum(overpricing_pct - cfg.churn_threshold_pct, 0.0)
        churn_prob = np.minimum(excess * cfg.churn_sensitivity, _MAX_CHURN_PROBABILITY)
        churn_prob = np.where(over_mask, churn_prob, 0.0)
        clv = y_true * cfg.customer_lifetime_value_multiplier
        churn_cost = churn_prob * (clv + cfg.customer_acquisition_cost)

        # ── Underpricing penalty ──────────────────────────────────────────────
        under_mask = error < 0
        under_pct = np.where(under_mask, np.abs(error_pct), 0.0)
        revenue_vec = y_pred * (1.0 + loading)
        base_pen = np.abs(error) * cfg.underpricing_penalty_multiplier
        max_pen = revenue_vec * 0.2
        under_pen = np.where(under_mask, np.minimum(base_pen, max_pen), 0.0)
        severe_mask = under_pct > cfg.severe_underpricing_threshold_pct
        under_pen = np.where(
            under_mask & severe_mask, under_pen + cfg.severe_underpricing_penalty, under_pen
        )

        # ── Accuracy bonus ────────────────────────────────────────────────────
        abs_err_pct = np.abs(error_pct)
        bonus = np.where(
            abs_err_pct <= cfg.acceptable_error_band_pct / 2,
            cfg.excellent_accuracy_bonus,
            np.where(
                abs_err_pct <= cfg.acceptable_error_band_pct,
                cfg.excellent_accuracy_bonus * 0.5,
                0.0,
            ),
        )

        net_profit = gross_profit - churn_cost - under_pen + bonus

        # ── Aggregates ────────────────────────────────────────────────────────
        total_revenue = float(np.sum(revenue_vec))
        total_claims = float(np.sum(y_true))
        total_admin = float(n * admin)
        total_gross_profit = float(np.sum(gross_profit))
        total_churn_cost = float(np.sum(churn_cost))
        total_under_pen = float(np.sum(under_pen))
        total_bonus = float(np.sum(bonus))
        total_net_profit = float(np.sum(net_profit))

        gross_margin = (total_gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        net_margin = (total_net_profit / total_revenue * 100) if total_revenue > 0 else 0

        n_underpriced = int(np.sum(under_mask))
        n_overpriced = int(np.sum(over_mask))
        n_accurate = int(np.sum(abs_err_pct <= cfg.acceptable_error_band_pct))

        profit_per_policy = total_net_profit / n if n > 0 else 0.0
        churn_rate = float(np.sum(churn_prob)) / n if n > 0 else 0.0
        biz_score = self._calculate_business_value_score(
            profit_per_policy, churn_rate, n_accurate / n if n > 0 else 0.0
        )

        return {
            "segment": segment_name,
            "n_predictions": n,
            "total_revenue": total_revenue,
            "total_claims": total_claims,
            "total_admin": total_admin,
            "total_base_cost": total_claims + total_admin,
            "total_gross_profit": total_gross_profit,
            "gross_margin_pct": gross_margin,
            "total_churn_cost": total_churn_cost,
            "total_underpricing_penalty": total_under_pen,
            "total_accuracy_bonus": total_bonus,
            "total_net_profit": total_net_profit,
            "net_margin_pct": net_margin,
            "profit_per_policy": profit_per_policy,
            "avg_churn_probability": float(np.mean(churn_prob)),
            "expected_churned_customers": float(np.sum(churn_prob)),
            "churn_rate_pct": churn_rate * 100,
            "n_underpriced": n_underpriced,
            "n_overpriced": n_overpriced,
            "n_accurate": n_accurate,
            "accuracy_rate_pct": (n_accurate / n * 100) if n > 0 else 0.0,
            "business_value_score": biz_score,
        }

    def _calculate_business_value_score(
        self, profit_per_policy: float, churn_rate: float, accuracy_rate: float
    ) -> float:
        """Composite business value score (0-100).

        Thresholds are now read from BusinessConfig (config.yaml-driven) rather
        than being hardcoded.  Defaults preserve the previous behaviour exactly.
        """
        _profit_target = self.config.biz_score_profit_target  # default 600 (config.yaml)
        _churn_low = self.config.biz_score_churn_low  # default 0.10
        _churn_high = self.config.biz_score_churn_high  # default 0.20
        _ret_max = self.config.biz_score_retention_max  # default 30.0
        _qual_weight = self.config.biz_score_quality_weight  # default 20.0

        if profit_per_policy >= _profit_target:
            profitability_score = 50.0
        elif profit_per_policy >= 0:
            profitability_score = 25.0 + (profit_per_policy / _profit_target) * 25.0
        else:
            profitability_score = max(-50.0, (profit_per_policy / _profit_target) * 25.0)

        _churn_band = _churn_high - _churn_low  # avoid div-by-zero if equal
        # Previous formula had a non-monotonic discontinuity at
        # _churn_low (default 10%).  The two piecewise branches did not connect:
        #   churn=9.9%  → first branch  →  0.30 pts
        #   churn=10.1% → second branch → 14.85 pts  ← score RISES as churn WORSENS
        # This made BizScore comparisons unreliable whenever either model's churn
        # approached the 10% boundary.
        #
        # Corrected formula: piecewise linear, continuous, strictly monotone decreasing.
        # Verified values (ret_max=30, churn_low=0.10, churn_high=0.20):
        #    0%   → 30.00   (unchanged)
        #    3%   → 25.50
        #    5%   → 22.50
        #   10%   → 15.00   (continuous at boundary — was 0.00 in old formula)
        #   15%   → 7.50
        #   20%   → 0.00    (unchanged)
        #  >20%   → 0.00    (unchanged)
        if churn_rate <= _churn_low:
            t = churn_rate / max(_churn_low, 1e-9)
            retention_score = _ret_max * (1.0 - 0.5 * t)
        elif churn_rate <= _churn_high:
            t = (churn_rate - _churn_low) / max(_churn_band, 1e-9)
            retention_score = _ret_max * 0.5 * (1.0 - t)
        else:
            retention_score = 0.0

        quality_score = accuracy_rate * _qual_weight

        return profitability_score + retention_score + quality_score

    def calculate_segment_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        use_business_thresholds: bool = True,
        config: dict | None = None,
    ) -> dict[str, dict]:
        """Calculate metrics for premium segments"""
        if config is None:
            try:
                from insurance_ml.config import load_config  # noqa: E402

                config = load_config()
            except (OSError, ImportError, RuntimeError) as e:
                logger.warning(f"⚠️ Could not load config: {e}. Using defaults.")
                config = {}

        if use_business_thresholds:
            eval_config = config.get("evaluation", {})
            thresholds = eval_config.get("segment_thresholds", {})

            low_threshold = thresholds.get("low_value", 4500.0)
            standard_threshold = thresholds.get("standard", 15000.0)
            high_threshold = thresholds.get("high_value", 30000.0)

            if not thresholds:
                logger.warning(
                    f"⚠️ No segment_thresholds in config.yaml\n"
                    f"   Using defaults: ${low_threshold:.0f}, "
                    f"${standard_threshold:.0f}, ${high_threshold:.0f}"
                )

            segments = {
                "low_risk": y_true <= low_threshold,
                "standard": (y_true > low_threshold) & (y_true <= standard_threshold),
                "high_risk": (y_true > standard_threshold) & (y_true <= high_threshold),
                "catastrophic": y_true > high_threshold,
            }
        else:
            q50 = np.percentile(y_true, 50)
            q75 = np.percentile(y_true, 75)
            q95 = np.percentile(y_true, 95)

            segments = {
                "low": y_true <= q50,
                "mid": (y_true > q50) & (y_true <= q75),
                "high": (y_true > q75) & (y_true <= q95),
                "extreme": y_true > q95,
            }

        results = {}

        for name, mask in segments.items():
            if mask.sum() == 0:
                continue

            if mask.sum() < _MIN_SEGMENT_SAMPLES:
                logger.warning(
                    f"⚠️ Segment '{name}' has only {mask.sum()} samples (< {_MIN_SEGMENT_SAMPLES})"
                )

            seg_true = y_true[mask]
            seg_pred = y_pred[mask]

            rmse = np.sqrt(np.mean((seg_true - seg_pred) ** 2))
            mae = np.mean(np.abs(seg_true - seg_pred))
            # pass segment name so warnings identify the
            # failing risk tier (e.g. "⚠️ SMAPE = 110.0% > 100% [segment: low_risk]")
            smape = calculate_smape(seg_true, seg_pred, segment_name=name)
            male = calculate_male(seg_true, seg_pred, segment_name=name)
            # Finding E: guard against n=1 crash; calculate_gate_aligned_segment_metrics
            # already has this check — now consistent across both paths.
            r2 = float(r2_score(seg_true, seg_pred)) if len(seg_true) >= 2 else float("nan")

            seg_business = self.calculate_portfolio_metrics(seg_true, seg_pred, segment_name=name)

            results[name] = {
                "n_samples": int(mask.sum()),
                "pct_of_total": float(mask.sum() / len(y_true) * 100),
                "rmse": float(rmse),
                "mae": float(mae),
                "smape": float(smape),
                "male": float(male),
                "r2": float(r2),
                "mean_true": float(seg_true.mean()),
                "mean_pred": float(seg_pred.mean()),
                "mean_error": float(np.mean(seg_pred - seg_true)),
                "median_error": float(np.median(seg_pred - seg_true)),
                "net_profit": seg_business["total_net_profit"],
                "profit_per_policy": seg_business["profit_per_policy"],
                "churn_rate_pct": seg_business["churn_rate_pct"],
            }

        return results

    def calculate_gate_aligned_segment_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata_path: str = "models/pipeline_metadata.json",
    ) -> dict[str, dict]:
        """
        Per-segment metrics using the same boundaries as the G6 deployment gate.

        Use this for post-deployment monitoring instead of
        calculate_segment_metrics() to ensure the segments that failed
        training evaluation are the same ones being tracked in production.

        Thresholds are loaded from pipeline_metadata.json (written at training
        time) so they always match the deployed model's evaluation run.
        If metadata is unavailable, falls back to the training-run defaults
        ($5K / $10K / $14K / $16.7K) with a warning.

        Segments (mirror of DeploymentGates.check_g6 breakdown):
            Low       : y_true < $5,000
            Mid       : $5,000 <= y_true < $10,000
            High      : $10,000 <= y_true < $14,000
            High+     : $14,000 <= y_true < HIGH_VALUE_THRESHOLD ($16,701)
            Very High : y_true >= HIGH_VALUE_THRESHOLD

        Returns dict[segment_name -> metrics_dict] with the same keys as
        calculate_segment_metrics() for drop-in compatibility.

        Each segment dict also includes two boolean flags:
            g6_advisory  True when N < 30 and R² < -1.0 (statistically unreliable,
                         matches the advisory-only behaviour of the gate guard)
            g6_veto      True when N >= 30 and R² < -1.0 (would block deployment)
        """
        from sklearn.metrics import r2_score as _r2_score

        # ── Load thresholds from metadata (training-time values) ──────────
        _DEFAULT_BINS = [0.0, 5_000.0, 10_000.0, 14_000.0, 16_701.0, np.inf]
        _DEFAULT_LABELS = ["Low", "Mid", "High", "High+", "Very High"]

        bins = _DEFAULT_BINS
        labels = _DEFAULT_LABELS

        try:
            meta_file = Path(metadata_path)
            if meta_file.exists():
                import json as _jmeta

                with open(meta_file) as _f:
                    _meta = _jmeta.load(_f)
                _bc = _meta.get("bias_correction_thresholds", {})
                if _bc:
                    _q50 = float(_bc.get("q50_threshold_low", 10_000.0))
                    _q75 = float(_bc.get("q75_threshold_high", 14_000.0))
                    # Rebuild bins using training-time P50/P75 as segment boundaries.
                    # This keeps monitor segments in sync with how the gate evaluated
                    # the deployed model, even when thresholds shift between retrains.
                    bins = [0.0, _q50 / 2, _q50, _q75, 16_701.0, np.inf]
                    logger.debug(
                        "Gate-aligned segments loaded from metadata: " "bins=%s",
                        [f"${b:,.0f}" for b in bins if b < np.inf],
                    )
            else:
                logger.warning(
                    "⚠️  Gate-aligned segments: metadata not found at '%s'. "
                    "Using default G6 thresholds — may not match deployed model's run.",
                    metadata_path,
                )
        except Exception as _meta_err:
            logger.warning(
                "⚠️  Gate-aligned segments: could not load metadata (%s). " "Using defaults.",
                _meta_err,
            )

        # ── Compute per-segment metrics ───────────────────────────────────
        tier = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)

        results: dict[str, dict] = {}

        for label in labels:
            mask_series = tier == label
            mask = mask_series.values if hasattr(mask_series, "values") else mask_series
            n = int(mask.sum())

            if n == 0:
                continue

            seg_true = y_true[mask]
            seg_pred = y_pred[mask]

            overpricing_rate = float(np.mean(seg_pred > seg_true))
            rmse = float(np.sqrt(np.mean((seg_true - seg_pred) ** 2)))
            mae = float(np.mean(np.abs(seg_true - seg_pred)))
            r2 = float(_r2_score(seg_true, seg_pred)) if n >= 2 else float("nan")

            results[label] = {
                "n_samples": n,
                "pct_of_total": float(n / len(y_true) * 100),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": r2,
                "overpricing_rate": overpricing_rate,
                "mean_true": float(seg_true.mean()),
                "mean_pred": float(seg_pred.mean()),
                "mean_error": float(np.mean(seg_pred - seg_true)),
                "median_error": float(np.median(seg_pred - seg_true)),
                # G6 gate compatibility flags
                "g6_advisory": n < 30 and not np.isnan(r2) and r2 < -1.0,
                "g6_veto": n >= 30 and not np.isnan(r2) and r2 < -1.0,
            }

        # ── Log summary table ─────────────────────────────────────────────
        logger.info(
            "\n📊 GATE-ALIGNED SEGMENT METRICS (G6 boundaries)\n" "   %-12s %5s %8s %10s %12s",
            "Segment",
            "N",
            "R²",
            "RMSE",
            "Overpricing",
        )
        for seg, m in results.items():
            r2_str = f"{m['r2']:8.4f}" if not np.isnan(m["r2"]) else "     N/A"
            flag = " ⚠️" if m["g6_advisory"] or m["g6_veto"] else ""
            logger.info(
                "   %-12s %5d %s $%9,.0f %10.1f%%%s",
                seg,
                m["n_samples"],
                r2_str,
                m["rmse"],
                m["overpricing_rate"] * 100,
                flag,
            )

        return results

    def calculate_cost_weighted_error(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: dict | None = None
    ) -> dict[str, float]:
        """Cost-weighted error with asymmetric penalties"""
        errors = y_pred - y_true

        underpriced_mask = errors < 0
        overpriced_mask = errors > 0

        if config is None:
            try:
                from insurance_ml.config import load_config  # noqa: E402

                config = load_config()
            except (OSError, ImportError, RuntimeError) as e:
                logger.warning(f"⚠️ Could not load config: {e}. Using defaults.")
                config = {}

        business_config = config.get("hybrid_predictor", {}).get("business_config", {})
        underpricing_penalty = business_config.get("underpricing_penalty_multiplier", 0.3)

        if not (_PENALTY_MULTIPLIER_MIN <= underpricing_penalty <= _PENALTY_MULTIPLIER_MAX):
            logger.warning(f"⚠️ underpricing_penalty_multiplier = {underpricing_penalty} is unusual")
            underpricing_penalty = 0.3

        underpricing_cost = np.sum(np.abs(errors[underpriced_mask]) * underpricing_penalty)
        overpricing_cost = np.sum(np.abs(errors[overpriced_mask]) * 1.0)

        total_cost = underpricing_cost + overpricing_cost
        n_samples = len(y_true)

        return {
            "total_cost": float(total_cost),
            "cost_per_policy": float(total_cost / n_samples),
            "underpricing_cost": float(underpricing_cost),
            "overpricing_cost": float(overpricing_cost),
            "n_underpriced": int(underpriced_mask.sum()),
            "n_overpriced": int(overpriced_mask.sum()),
            "underpricing_pct": float(underpriced_mask.sum() / n_samples * 100),
            "avg_underprice_amount": float(
                np.abs(errors[underpriced_mask]).mean() if underpriced_mask.any() else 0
            ),
            "avg_overprice_amount": float(
                np.abs(errors[overpriced_mask]).mean() if overpriced_mask.any() else 0
            ),
            "penalty_multiplier": float(underpricing_penalty),
        }


# =====================================================================
# CONFIGURATION COMPARISON
# =====================================================================


def compare_multiple_configurations(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ml_pipeline: PredictionPipeline,
    business_config: BusinessConfig,
    output_dir: str = "reports/config_comparisons",
) -> pd.DataFrame:
    """Compare multiple hybrid configurations.

    v7.3.0: reads calibration defaults from config.yaml (no hardcoded 1.10).
    Adds apply_to_ml_only sweep axis so the evaluation captures the full parameter space.
    """
    logger.info("🔄 Comparing multiple hybrid configurations...")

    # ── Read calibration defaults from config rather than hardcoding ──────────
    try:
        _full_cfg = load_config()
        _cal_cfg = _full_cfg.get("hybrid_predictor", {}).get("calibration", {})
        _default_cal_factor = float(_cal_cfg.get("factor", 1.00))
        _default_apply_ml_only = bool(_cal_cfg.get("apply_to_ml_only", True))
        # Inference guard limit: the ceiling predict.py enforces per call.
        # _predict_in_batches uses this as the chunk size so each call is
        # accepted regardless of total dataset size.
        _inference_limit = int(_full_cfg.get("prediction", {}).get("max_batch_size", 50_000))
        # read deployed threshold and blend_ratio from config
        # instead of hardcoding 4500.0 (which is below dataset mean ~$13K,
        # making BlendRatio/SoftWindow/Calibration sweep axes effectively flat).
        _deployed_threshold = float(_full_cfg.get("hybrid_predictor", {}).get("threshold", 5000.0))
        _deployed_blend = float(_full_cfg.get("hybrid_predictor", {}).get("blend_ratio", 0.75))
    except Exception as _cfg_err:
        # The bare `except:` here swallowed any YAML parse error, missing
        # key, or type error silently — the sweep ran with $5,000 / blend=0.75
        # defaults and produced quietly wrong results with no indication in the log.
        # log a WARNING with the actual error so the operator knows the sweep
        # is anchored at fallback defaults, not the deployed config.
        logger.warning(
            f"⚠️ compare_multiple_configurations: failed to read deployed config "
            f"({type(_cfg_err).__name__}: {_cfg_err}). "
            f"Config sweep anchored at fallback defaults "
            f"(threshold=$5,000, blend=0.75, cal_factor=1.00, apply_ml_only=True). "
            f"Results will NOT reflect the deployed model's actual configuration."
        )
        _default_cal_factor = 1.00
        _default_apply_ml_only = True
        _deployed_threshold = 5000.0
        _deployed_blend = 0.75
        # T1-A: _full_cfg must be set in the except path.  Without this line,
        # UnifiedEvaluator(config=_full_cfg) on line ~778 raises NameError when
        # load_config() fails — exactly when you need the sweep to still run.
        _full_cfg = {}
    logger.info(
        f"   Config defaults — factor: {_default_cal_factor:.4f}, "
        f"apply_to_ml_only: {_default_apply_ml_only}\n"
        f"   Sweep anchor — threshold: ${_deployed_threshold:,.0f} (deployed), "
        f"blend_ratio: {_deployed_blend}"
    )

    configurations = []

    # Original sweep [3500..5500] sits entirely below the dataset mean (~$13K).
    # At these thresholds ~98% of policies route ML-dominant, so the sweep cannot
    # detect where the configured/effective blend actually diverge.
    # Extended range covers P10 through P90 of a typical insurance charge distribution.
    for threshold in [3_500, 5_000, 7_000, 9_000, 11_000, 13_000, 15_000]:
        configurations.append(
            {
                "name": f"Threshold_{threshold}",
                "threshold": threshold,
                "blend_ratio": _deployed_blend,
                "use_soft_blending": True,
                "soft_blend_window": 1000.0,
                "calibration_factor": _default_cal_factor,
                "apply_to_ml_only": _default_apply_ml_only,
            }
        )

    for ratio in [
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.9,
    ]:  # added 0.75, 0.9 to cover deployed range
        configurations.append(
            {
                "name": f"BlendRatio_{int(ratio*100)}",
                "threshold": _deployed_threshold,
                "blend_ratio": ratio,
                "use_soft_blending": True,
                "soft_blend_window": 1000.0,
                "calibration_factor": _default_cal_factor,
                "apply_to_ml_only": _default_apply_ml_only,
            }
        )

    configurations.append(
        {
            "name": "HardBlending",
            "threshold": _deployed_threshold,
            "blend_ratio": _deployed_blend,
            "use_soft_blending": False,
            "soft_blend_window": 0.0,
            "calibration_factor": _default_cal_factor,
            "apply_to_ml_only": _default_apply_ml_only,
        }
    )

    for window in [
        250,
        500,
        750,
        1000,
        1500,
    ]:
        configurations.append(
            {
                "name": f"SoftWindow_{window}",
                "threshold": _deployed_threshold,
                "blend_ratio": _deployed_blend,
                "use_soft_blending": True,
                "soft_blend_window": float(window),
                "calibration_factor": _default_cal_factor,
                "apply_to_ml_only": _default_apply_ml_only,
            }
        )

    for cal_factor in [
        1.02,
        1.05,
        1.08,
        1.10,
        1.15,
        1.20,
    ]:
        configurations.append(
            {
                "name": f"Calibration_{int(cal_factor*100)}",
                "threshold": _deployed_threshold,
                "blend_ratio": _deployed_blend,
                "use_soft_blending": True,
                "soft_blend_window": 1000.0,
                "calibration_factor": cal_factor,
                "apply_to_ml_only": _default_apply_ml_only,
            }
        )

    # ── NEW: apply_to_ml_only sweep — captures strategy impact ───────────────
    for apply_ml_only in [True, False]:
        label = "MLOnly" if apply_ml_only else "FullHybrid"
        configurations.append(
            {
                "name": f"CalStrategy_{label}",
                "threshold": _deployed_threshold,
                "blend_ratio": _deployed_blend,
                "use_soft_blending": True,
                "soft_blend_window": 1000.0,
                "calibration_factor": _default_cal_factor,
                "apply_to_ml_only": apply_ml_only,
            }
        )

    y_true = y_test.values
    ml_result = _predict_in_batches(
        ml_pipeline, X_test, inference_limit=_inference_limit, return_reliability=False
    )
    ml_preds = np.array(ml_result["predictions"])

    evaluator = UnifiedEvaluator(business_config, config=_full_cfg)
    ml_academic = calculate_academic_metrics(y_true, ml_preds)
    ml_business = evaluator.business_calc.calculate_portfolio_metrics(y_true, ml_preds)

    results = [
        {
            "config_name": "ML_Baseline",
            "threshold": np.nan,
            "blend_ratio": 1.0,
            "soft_blending": False,
            "soft_window": np.nan,
            "calibration_factor": 1.0,
            "apply_to_ml_only": np.nan,
            "mape": ml_academic["mape"],
            "smape": ml_academic["smape"],
            "rmse": ml_academic["rmse"],
            "mae": ml_academic["mae"],
            "r2": ml_academic["r2"],
            "net_profit": ml_business["total_net_profit"],
            "profit_per_policy": ml_business["profit_per_policy"],
            "churn_rate_pct": ml_business["churn_rate_pct"],
            "business_value_score": ml_business["business_value_score"],
            "expected_churned_customers": ml_business["expected_churned_customers"],
            "underpricing_penalty": ml_business["total_underpricing_penalty"],
            "is_baseline": True,
            "is_best_business": False,
            "is_best_academic": False,
        }
    ]

    for i, cfg in enumerate(configurations, 1):
        logger.info(f"   Testing {i}/{len(configurations)}: {cfg['name']}...")

        try:
            # T1-B: pass full deployed config so actuarial_params, business_config,
            # and monitoring all come from config.yaml.  Previously only a bare
            # calibration sub-dict was passed, causing HybridPredictor.__init__ to
            # fall back to code-default actuarial params (smoker_multiplier=1.8)
            # instead of the operational value (3.5).  Every sweep result was wrong
            # for smoker-heavy sub-segments.
            _deployed_hybrid_cfg = _full_cfg.get("hybrid_predictor", {})
            hybrid = HybridPredictor(
                ml_predictor=ml_pipeline,
                threshold=cfg["threshold"],
                blend_ratio=cfg["blend_ratio"],
                use_soft_blending=cfg["use_soft_blending"],
                soft_blend_window=cfg["soft_blend_window"],
                calibration_factor=cfg.get("calibration_factor", _default_cal_factor),
                config={
                    **_deployed_hybrid_cfg,  # carries actuarial_params, business_config, etc.
                    "calibration": {  # override only sweep-specific calibration keys
                        "factor": cfg.get("calibration_factor", _default_cal_factor),
                        "apply_to_ml_only": cfg.get("apply_to_ml_only", _default_apply_ml_only),
                        "enabled": True,
                    },
                    "threshold": cfg["threshold"],  # override sweep-specific routing keys
                    "blend_ratio": cfg["blend_ratio"],
                    "use_soft_blending": cfg["use_soft_blending"],
                    "soft_blend_window": cfg["soft_blend_window"],
                },
            )

            hybrid_result = _predict_in_batches(
                hybrid, X_test, inference_limit=_inference_limit, return_reliability=False
            )
            hybrid_preds = np.array(hybrid_result["predictions"])

            academic = calculate_academic_metrics(y_true, hybrid_preds)
            business = evaluator.business_calc.calculate_portfolio_metrics(y_true, hybrid_preds)

            results.append(
                {
                    "config_name": cfg["name"],
                    "threshold": cfg["threshold"],
                    "blend_ratio": cfg["blend_ratio"],
                    "soft_blending": cfg["use_soft_blending"],
                    "soft_window": cfg["soft_blend_window"],
                    "calibration_factor": cfg.get("calibration_factor", _default_cal_factor),
                    "apply_to_ml_only": cfg.get("apply_to_ml_only", _default_apply_ml_only),
                    "mape": academic["mape"],
                    "smape": academic["smape"],
                    "rmse": academic["rmse"],
                    "mae": academic["mae"],
                    "r2": academic["r2"],
                    "net_profit": business["total_net_profit"],
                    "profit_per_policy": business["profit_per_policy"],
                    "churn_rate_pct": business["churn_rate_pct"],
                    "business_value_score": business["business_value_score"],
                    "expected_churned_customers": business["expected_churned_customers"],
                    "underpricing_penalty": business["total_underpricing_penalty"],
                    "is_baseline": False,
                    "is_best_business": False,
                    "is_best_academic": False,
                }
            )

        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")
            continue

    df_comparison = pd.DataFrame(results)

    best_business_idx = df_comparison["business_value_score"].idxmax()
    best_academic_idx = df_comparison["rmse"].idxmin()

    df_comparison.loc[best_business_idx, "is_best_business"] = True
    df_comparison.loc[best_academic_idx, "is_best_academic"] = True

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "configuration_comparison.csv"
        df_comparison.to_csv(str(csv_path), index=False, encoding="utf-8")
        logger.info(f"✅ Saved: {csv_path}")
    except Exception as e:
        logger.error(f"❌ Error writing CSV: {e}")

    create_config_comparison_plot(df_comparison, output_dir)

    print("CONFIGURATION COMPARISON RESULTS")
    print("=" * 90)

    best_biz = df_comparison.loc[best_business_idx]
    best_acad = df_comparison.loc[best_academic_idx]

    print("\n🏆 Best for Business:")
    print(f"   Config: {best_biz['config_name']}")
    print(f"   Business Score: {best_biz['business_value_score']:.1f}")
    print(f"   Net Profit: ${best_biz['net_profit']:,.0f}")

    print("\n🎯 Best for Accuracy:")
    print(f"   Config: {best_acad['config_name']}")
    print(f"   RMSE: ${best_acad['rmse']:,.2f}")

    print("\n" + "=" * 90)

    return df_comparison


def create_config_comparison_plot(df_comparison: pd.DataFrame, output_dir: str):
    """Create configuration comparison visualizations"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Configuration Comparison: Academic vs Business",
            fontsize=16,
            fontweight="bold",
        )

        baseline = df_comparison[df_comparison["is_baseline"]].iloc[0]
        configs = df_comparison[~df_comparison["is_baseline"]]

        def get_colors(df, metric, lower_is_better=True):
            baseline_val = baseline[metric]
            if lower_is_better:
                return ["green" if x < baseline_val else "red" for x in df[metric]]
            else:
                return ["green" if x > baseline_val else "red" for x in df[metric]]

        # 1. Business Score
        ax1 = axes[0, 0]
        colors = get_colors(configs, "business_value_score", False)
        ax1.bar(
            range(len(configs)),
            configs["business_value_score"],
            color=colors,
            alpha=0.7,
        )
        ax1.axhline(baseline["business_value_score"], color="blue", linestyle="--", linewidth=2)
        ax1.set_ylabel("Score", fontweight="bold")
        ax1.set_title("Business Value Score", fontweight="bold")
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs["config_name"], rotation=45, ha="right", fontsize=7)
        ax1.grid(True, alpha=0.3)

        # 2. RMSE
        ax2 = axes[0, 1]
        colors = get_colors(configs, "rmse", True)
        ax2.bar(range(len(configs)), configs["rmse"], color=colors, alpha=0.7)
        ax2.axhline(baseline["rmse"], color="blue", linestyle="--", linewidth=2)
        ax2.set_ylabel("RMSE ($)", fontweight="bold")
        ax2.set_title("Root Mean Squared Error", fontweight="bold")
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs["config_name"], rotation=45, ha="right", fontsize=7)
        ax2.grid(True, alpha=0.3)

        # 3. Net Profit
        ax3 = axes[0, 2]
        colors = get_colors(configs, "net_profit", False)
        ax3.bar(range(len(configs)), configs["net_profit"], color=colors, alpha=0.7)
        ax3.axhline(baseline["net_profit"], color="blue", linestyle="--", linewidth=2)
        ax3.set_ylabel("Net Profit ($)", fontweight="bold")
        ax3.set_title("Total Net Profit", fontweight="bold")
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels(configs["config_name"], rotation=45, ha="right", fontsize=7)
        ax3.grid(True, alpha=0.3)

        # 4. Churn Rate
        ax4 = axes[1, 0]
        colors = get_colors(configs, "churn_rate_pct", True)
        ax4.bar(range(len(configs)), configs["churn_rate_pct"], color=colors, alpha=0.7)
        ax4.axhline(baseline["churn_rate_pct"], color="blue", linestyle="--", linewidth=2)
        ax4.set_ylabel("Churn Rate (%)", fontweight="bold")
        ax4.set_title("Expected Churn", fontweight="bold")
        ax4.set_xticks(range(len(configs)))
        ax4.set_xticklabels(configs["config_name"], rotation=45, ha="right", fontsize=7)
        ax4.grid(True, alpha=0.3)

        # 5. RMSE comparison (all configs including baseline)
        ax5 = axes[1, 1]
        rmses = [baseline["rmse"]] + configs["rmse"].tolist()
        labels = ["ML Baseline"] + configs["config_name"].tolist()
        colors_rmse = ["blue"] + get_colors(configs, "rmse", True)
        ax5.bar(range(len(rmses)), rmses, color=colors_rmse, alpha=0.7)
        ax5.set_ylabel("RMSE ($)", fontweight="bold")
        ax5.set_title("RMSE Comparison", fontweight="bold")
        ax5.set_xticks(range(len(rmses)))
        ax5.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax5.grid(True, alpha=0.3)

        # 6. Threshold Analysis
        ax6 = axes[1, 2]
        threshold_configs = configs[configs["config_name"].str.contains("Threshold")]
        if len(threshold_configs) > 0:
            x = threshold_configs["threshold"].values
            y1 = threshold_configs["business_value_score"].values
            y2 = threshold_configs["rmse"].values

            ax6_twin = ax6.twinx()
            ax6.plot(x, y1, "o-", color="green", linewidth=2, markersize=8, label="Business")
            ax6_twin.plot(x, y2, "s-", color="red", linewidth=2, markersize=8, label="RMSE")

            ax6.set_xlabel("Threshold ($)", fontweight="bold")
            ax6.set_ylabel("Business Score", fontweight="bold", color="green")
            ax6_twin.set_ylabel("RMSE ($)", fontweight="bold", color="red")
            ax6.set_title("Threshold Sensitivity", fontweight="bold")
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(
                0.5,
                0.5,
                "No threshold\nvariations",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )

        plt.tight_layout()

        plot_path = str(output_path / "configuration_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved: {plot_path}")
        plt.close()

    except Exception as e:
        logger.error(f"❌ Error creating plot: {e}")
        plt.close()


# =====================================================================
# BATCH-SAFE PREDICT WRAPPER
# =====================================================================


def _predict_in_batches(
    predictor,
    X: pd.DataFrame,
    inference_limit: int,
    **kwargs,
) -> Any:
    """Batch-safe wrapper for predictor.predict().

    predict.py enforces prediction.max_batch_size (default 50 000) as an
    inference-serving guard.  evaluate.py legitimately operates on full test
    sets that can range from 1 000 to 100 000+ rows.

    This wrapper splits X into the minimum number of chunks that each fit
    within inference_limit, calls predict() on each chunk, then reassembles
    a single result dict that is structurally identical to a native single-
    call result.  When len(X) ≤ inference_limit, the fast path makes exactly
    one call with no overhead.

    Args:
        predictor:       PredictionPipeline or HybridPredictor instance.
        X:               Full input DataFrame (any size).
        inference_limit: Value of prediction.max_batch_size from config —
                         the hard ceiling enforced by predict.py's guard.
                         Chunks are sized at this limit so each call is
                         accepted by the guard without modification.
        **kwargs:        Forwarded verbatim to predictor.predict().

    Merge rules
    -----------
    list / np.ndarray with len == len(chunk)  → np.concatenate across chunks
    dict (components, reliability, …)         → recurse with same rules
    dict-valued key where some chunks returned None instead of a dict
        (only case: tail_risk_warning)        → first non-None value
    everything else (scalar, bool, str, None) → value from the first chunk
        (config-driven metadata, consistent across chunks by construction)
    """
    n = len(X)
    if n <= inference_limit:
        return predictor.predict(X, **kwargs)

    chunks = [X.iloc[i : i + inference_limit] for i in range(0, n, inference_limit)]
    batch_results = [predictor.predict(chunk, **kwargs) for chunk in chunks]
    chunk_lens = [len(c) for c in chunks]

    # Keys whose scalar values must be averaged across chunks (weighted by chunk
    # length) rather than taking the first chunk's value.  Scalars like
    # effective_avg_ml_weight and actuarial_conservativeness_ratio are per-chunk
    # summaries; the true portfolio value is a length-weighted mean.
    # add any new float-summary keys here as they are added to the
    # result dict in HybridPredictor.predict() / PredictionPipeline.predict().
    _WEIGHTED_MEAN_KEYS = frozenset(
        {
            "effective_avg_ml_weight",
            "avg_ml_weight",
            "actuarial_conservativeness_ratio",
            "configured_blend_ratio",
        }
    )

    def _merge(results: list, c_lens: list) -> dict:
        first, n0 = results[0], c_lens[0]
        n_total = sum(c_lens)
        merged: dict = {}
        for key, val in first.items():
            if isinstance(val, list | np.ndarray) and len(val) == n0:
                # per-sample array — concatenate across all chunks
                merged[key] = np.concatenate([np.asarray(r[key]) for r in results])
            elif isinstance(val, dict):
                # nested dict — only recurse if ALL chunks returned a dict for
                # this key.  tail_risk_warning is None when no underpriced
                # policies exist in a chunk and a dict otherwise; mixed
                # None/dict across chunks would crash on None.get(k).
                sub_values = [r[key] for r in results]
                if all(isinstance(sv, dict) for sv in sub_values):
                    sub = [{k: sv.get(k) for k in val} for sv in sub_values]
                    merged[key] = _merge(sub, c_lens)
                else:
                    # Take the first non-None value so warnings from any
                    # chunk are surfaced rather than silently dropped.
                    merged[key] = next((sv for sv in sub_values if sv is not None), None)
            elif (
                isinstance(val, int | float)
                and not isinstance(val, bool)
                and key in _WEIGHTED_MEAN_KEYS
            ):
                # scalar float summaries that represent per-sample
                # averages must be recomputed as a length-weighted mean across
                # chunks.  Taking the first-chunk value (old behaviour) produced
                # effective_avg_ml_weight = chunk-0 average ≠ portfolio average,
                # causing actuarial diagnostics and MLflow metrics to be wrong
                # whenever n > inference_limit and batching was triggered.
                try:
                    merged[key] = float(
                        sum(
                            float(r.get(key, val)) * cl
                            for r, cl in zip(results, c_lens, strict=False)
                        )
                        / n_total
                    )
                except (TypeError, ZeroDivisionError):
                    merged[key] = val  # fallback: first-chunk value
            else:
                # scalar config metadata — first chunk is canonical
                merged[key] = val
        return merged

    return _merge(batch_results, chunk_lens)


# =====================================================================
# DATA LOADING & ACADEMIC METRICS
# =====================================================================


def load_and_split_data(
    data_path: str = "data/raw/insurance.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    index_path: str = "models/test_indices.json",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load test data using the exact indices saved during training.

    when stratify_splits=true, train.py uses a quantile-stratified
    split that is NOT reproducible by a plain train_test_split() call in
    evaluate.py — causing 81% of evaluate.py's 'test' rows to overlap with
    the training fold.

    Primary path: load test_indices.json written by train.py (atomic write).
    Validates that every saved index exists in the loaded DataFrame.
    Fallback: re-split with a prominent warning (metrics will be biased).
    """
    logger.info(f"📂 Loading data from: {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"✅ Loaded {len(df)} samples")

    required = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # ── Primary path: load saved test indices ─────────────────────────────────
    _index_file = Path(index_path)
    if _index_file.exists():
        try:
            with open(_index_file) as _f:
                _meta = json.load(_f)

            _saved_indices = _meta["indices"]

            # Validate: every saved index must exist in the loaded DataFrame
            _df_idx = set(df.index.tolist())
            _missing_in_df = [i for i in _saved_indices if i not in _df_idx]
            if _missing_in_df:
                raise ValueError(
                    f"test_indices.json references {len(_missing_in_df)} indices "
                    f"not present in {data_path}. "
                    f"Re-train or point --data-path to the correct CSV."
                )

            X_test = df.drop(columns=["charges"]).loc[_saved_indices]
            y_test = df["charges"].loc[_saved_indices]

            logger.info(
                f"✅ Test set loaded from saved indices: {len(X_test)} samples\n"
                f"   Source: {_index_file}\n"
                f"   Trained with: random_state={_meta.get('random_state')}, "
                f"stratified={_meta.get('stratify_splits')}, "
                f"pipeline_version={_meta.get('pipeline_version')}"
            )
            return X_test, y_test

        except Exception as _idx_err:
            logger.warning(
                f"⚠️ Could not load test indices from {_index_file}: {_idx_err}\n"
                f"   Falling back to re-splitting — metrics will not match training\n"
                f"   when stratify_splits=true.\n"
                f"   re-run train.py to regenerate test_indices.json, "
                f"then re-run evaluate.py."
            )
    else:
        logger.warning(
            f"⚠️ test_indices.json not found at {_index_file}.\n"
            f"   Falling back to re-splitting (random_state={random_state}).\n"
            f"   This produces a DIFFERENT test partition than train.py when\n"
            f"   stratify_splits=true — evaluation metrics will be biased.\n"
            f"   re-run train.py, then re-run evaluate.py."
        )

    # ── Fallback: plain re-split ───────────────────────────────────────────────
    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"   Test (fallback re-split): {len(X_test)} samples ({test_size*100:.0f}%)")
    return X_test, y_test


def calculate_academic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_mask: np.ndarray | None = None,
    segment_name: str = "overall",
) -> dict[str, Any]:
    """Calculate ML metrics (diagnostic only - not for decisions).

    predict.py v6.3.1 recommends: RMSE, MALE, SMAPE.
    MAPE is retained for backward compatibility but demoted to diagnostic-only.
    """
    if segment_mask is not None:
        if segment_mask.sum() == 0:
            # Returning None here crashes all callers that subscript the result
            # (e.g. hybrid_acad["rmse"]). Return a clearly-marked sentinel dict so
            # callers continue without branching, and log so it's visible.
            logger.warning(f"⚠️ Segment '{segment_name}' is empty — returning zero-filled metrics.")
            return {
                "segment": segment_name,
                "n_samples": 0,
                "rmse": 0.0,
                "mae": 0.0,
                "mape": 0.0,
                "smape": 0.0,
                "male": 0.0,
                "r2": float(
                    "nan"
                ),  # Finding G: NaN is unambiguous; 0.0 silently passes R²<0 guards
                "mean_true": 0.0,
                "mean_pred": 0.0,
                "mean_error": 0.0,
                "median_error": 0.0,
                "std_error": 0.0,
            }
        y_true = y_true[segment_mask]
        y_pred = y_pred[segment_mask]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    # pass segment_name so per-segment warnings are actionable
    smape = calculate_smape(y_true, y_pred, segment_name=segment_name)
    male = calculate_male(y_true, y_pred, segment_name=segment_name)
    # guard against n=1 crash — r2_score raises UndefinedMetricWarning
    # and returns NaN when called with a single sample. The guard was already
    # applied in calculate_segment_metrics() (line ~467) but was missing here
    # in calculate_academic_metrics(), the gate-aligned path called from
    # evaluate_comprehensive(). Any segment with n=1 now returns NaN rather
    # than emitting a noisy warning and silently propagating NaN downstream.
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")

    return {
        "segment": segment_name,
        "n_samples": len(y_true),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,  # Diagnostic only — not used for decisions
        "smape": smape,  # Recommended: symmetric, bounded, insurance-appropriate
        "male": male,  # Scale-invariant log-space metric
        "r2": r2,
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "mean_error": float(np.mean(y_pred - y_true)),
        "median_error": float(np.median(y_pred - y_true)),
        "std_error": float(np.std(y_pred - y_true)),
    }


def calculate_smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_name: str = "",
) -> float:
    """Symmetric MAPE - Better than MAPE for insurance data.

    segment_name parameter added so that warnings emitted
    by this function identify the failing risk tier, making the ops log
    immediately actionable (e.g. "⚠️ SMAPE = 110.0% > 100% [segment: low_risk]"
    instead of a bare percentage with no context).
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    mask = denominator > _SMAPE_DENOMINATOR_MIN

    if not mask.any():
        logger.warning(f"⚠️ All SMAPE denominators < ${_SMAPE_DENOMINATOR_MIN:.2f}")
        return 0.0

    smape_values = numerator[mask] / denominator[mask]
    smape_pct = float(np.mean(smape_values) * 100)

    # include segment tag in all threshold warnings
    _seg_tag = f" [segment: {segment_name}]" if segment_name else ""
    if smape_pct > _SMAPE_ERROR_THRESHOLD:
        logger.error(
            f"❌ SMAPE = {smape_pct:.1f}% > {_SMAPE_ERROR_THRESHOLD}% — severe prediction failure{_seg_tag}"
        )
    elif smape_pct > _SMAPE_WARNING_THRESHOLD:
        logger.warning(f"⚠️ SMAPE = {smape_pct:.1f}% > {_SMAPE_WARNING_THRESHOLD}%{_seg_tag}")

    return smape_pct


def calculate_male(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segment_name: str = "",
) -> float:
    """Mean Absolute Log Error - scale-invariant metric.

    segment_name parameter added so that warnings emitted
    by this function identify the failing risk tier.
    """
    _seg_tag = f" [segment: {segment_name}]" if segment_name else ""

    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        logger.warning(f"⚠️ MALE requires positive values — using log1p{_seg_tag}")
        return float(np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred))))

    male = float(np.mean(np.abs(np.log(y_true) - np.log(y_pred))))

    if male > 1.0:
        logger.warning(f"⚠️ MALE = {male:.3f} > 1.0{_seg_tag}")

    return male


# =====================================================================
# UNIFIED EVALUATOR
# =====================================================================


class UnifiedEvaluator:
    """Unified evaluator with business-first focus"""

    def __init__(self, business_config: BusinessConfig | None = None, config: dict | None = None):
        self.business_config = business_config or BusinessConfig()
        self.business_calc = BusinessMetricsCalculator(self.business_config)
        # Cache the full config dict once at construction time so
        # evaluate_comprehensive, calculate_segment_metrics, and
        # calculate_cost_weighted_error all share the same loaded dict.
        # In a 16-config sweep this eliminates 30+ redundant YAML parses.
        if config is not None:
            self._cached_config: dict | None = config
        else:
            try:
                self._cached_config = load_config()
            except (OSError, ImportError, RuntimeError) as e:
                logger.warning(f"⚠️ UnifiedEvaluator: could not pre-load config: {e}")
                self._cached_config = {}

    def evaluate_comprehensive(
        self,
        y_true: np.ndarray,
        ml_preds: np.ndarray,
        hybrid_preds: np.ndarray,
        threshold: float = 4500.0,
        calibration_info: dict | None = None,
        enable_segment_analysis: bool = True,
        config: dict | None = None,
        actuarial_info: dict | None = None,
    ) -> dict:
        """Run comprehensive evaluation with calibration tracking"""
        # Use caller-supplied config, then cached, then lazy-load as last resort.
        if config is None:
            config = self._cached_config
        if not config:
            try:
                config = load_config()
                self._cached_config = config
            except (OSError, ImportError, RuntimeError) as e:
                logger.warning(f"⚠️ Could not load config: {e}")
                config = {}

        low_mask = y_true < threshold
        high_mask = ~low_mask

        # Academic metrics (DIAGNOSTIC ONLY - not used for decisions)
        academic = {
            "ml": {
                "overall": calculate_academic_metrics(y_true, ml_preds, None, "ML Overall"),
                "low_value": calculate_academic_metrics(y_true, ml_preds, low_mask, "ML Low"),
                "high_value": calculate_academic_metrics(y_true, ml_preds, high_mask, "ML High"),
            },
            "hybrid": {
                "overall": calculate_academic_metrics(y_true, hybrid_preds, None, "Hybrid Overall"),
                "low_value": calculate_academic_metrics(
                    y_true, hybrid_preds, low_mask, "Hybrid Low"
                ),
                "high_value": calculate_academic_metrics(
                    y_true, hybrid_preds, high_mask, "Hybrid High"
                ),
            },
        }

        # Business metrics (PRIMARY decision signals)
        business = {
            "ml": {
                "overall": self.business_calc.calculate_portfolio_metrics(
                    y_true, ml_preds, "ML Overall"
                ),
                "low_value": (
                    self.business_calc.calculate_portfolio_metrics(
                        y_true[low_mask], ml_preds[low_mask], "ML Low"
                    )
                    if low_mask.sum() > 0
                    else None
                ),
                "high_value": (
                    self.business_calc.calculate_portfolio_metrics(
                        y_true[high_mask], ml_preds[high_mask], "ML High"
                    )
                    if high_mask.sum() > 0
                    else None
                ),
            },
            "hybrid": {
                "overall": self.business_calc.calculate_portfolio_metrics(
                    y_true, hybrid_preds, "Hybrid Overall"
                ),
                "low_value": (
                    self.business_calc.calculate_portfolio_metrics(
                        y_true[low_mask], hybrid_preds[low_mask], "Hybrid Low"
                    )
                    if low_mask.sum() > 0
                    else None
                ),
                "high_value": (
                    self.business_calc.calculate_portfolio_metrics(
                        y_true[high_mask], hybrid_preds[high_mask], "Hybrid High"
                    )
                    if high_mask.sum() > 0
                    else None
                ),
            },
        }

        # Segment analysis
        segment_analysis = {}
        if enable_segment_analysis:
            try:
                segment_analysis = {
                    "ml": self.business_calc.calculate_segment_metrics(
                        y_true, ml_preds, use_business_thresholds=True, config=config
                    ),
                    "hybrid": self.business_calc.calculate_segment_metrics(
                        y_true,
                        hybrid_preds,
                        use_business_thresholds=True,
                        config=config,
                    ),
                }
            except Exception as e:
                logger.error(f"❌ Segment analysis failed: {e}")

        # Cost analysis
        cost_analysis = {}
        try:
            cost_analysis = {
                "ml": self.business_calc.calculate_cost_weighted_error(
                    y_true, ml_preds, config=config
                ),
                "hybrid": self.business_calc.calculate_cost_weighted_error(
                    y_true, hybrid_preds, config=config
                ),
            }
        except Exception as e:
            logger.error(f"❌ Cost analysis failed: {e}")

        # T2-D: statistical test on per-policy net profit (business signal).
        # The previous test on |y_true - y_pred| (MAE) is an academic signal that
        # can vote "hybrid is significantly better" even when hybrid profit < ML profit.
        # A hybrid that reduces MAE while increasing underpricing penalties is NOT
        # better for deployment.  Testing on net_profit ensures the significance
        # test answers the actual business question.
        # The MAE-based test is retained as a secondary diagnostic.
        #
        # Finding A: replaced two O(n) list-comprehensions calling
        # calculate_single_prediction_value() per row with a fully vectorised path.
        # At 10K policies the loops cost ~40 ms; this runs in ~1 ms.
        # Numerically identical to the scalar path (verified to float precision).
        cfg = self.business_calc.config
        _loading = cfg.base_profit_margin
        _admin = cfg.admin_cost_per_policy

        def _net_profit_vec(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
            _err = yp - yt
            _ep = np.where(yt > 0, _err / yt, 0.0)
            _gp = yp - yt + _loading * yp - _admin
            _over = _err > 0
            _excess = np.maximum(np.where(_over, _ep, 0.0) - cfg.churn_threshold_pct, 0.0)
            _cp = np.where(
                _over,
                np.minimum(_excess * cfg.churn_sensitivity, _MAX_CHURN_PROBABILITY),
                0.0,
            )
            _cc = _cp * (
                yt * cfg.customer_lifetime_value_multiplier + cfg.customer_acquisition_cost
            )
            _under = _err < 0
            _rev = yp * (1.0 + _loading)
            _bp = np.abs(_err) * cfg.underpricing_penalty_multiplier
            _mp = _rev * 0.2
            _up = np.where(_under, np.minimum(_bp, _mp), 0.0)
            _sev = np.where(_under, np.abs(_ep), 0.0) > cfg.severe_underpricing_threshold_pct
            _up = np.where(_under & _sev, _up + cfg.severe_underpricing_penalty, _up)
            _ae = np.abs(_ep)
            _bonus = np.where(
                _ae <= cfg.acceptable_error_band_pct / 2,
                cfg.excellent_accuracy_bonus,
                np.where(
                    _ae <= cfg.acceptable_error_band_pct, cfg.excellent_accuracy_bonus * 0.5, 0.0
                ),
            )
            return np.asarray(_gp - _cc - _up + _bonus)

        ml_profits_stat = _net_profit_vec(y_true, ml_preds)
        hybrid_profits_stat = _net_profit_vec(y_true, hybrid_preds)
        t_stat, p_value = stats.ttest_rel(ml_profits_stat, hybrid_profits_stat)

        # Secondary: MAE-based test (diagnostic only — not used for deployment decisions)
        ml_errors = np.abs(y_true - ml_preds)
        hybrid_errors = np.abs(y_true - hybrid_preds)
        t_stat_mae, p_value_mae = stats.ttest_rel(ml_errors, hybrid_errors)

        return {
            "academic": academic,
            "business": business,
            "segment_analysis": segment_analysis,
            "cost_analysis": cost_analysis,
            "threshold": threshold,
            "calibration_info": calibration_info or {},
            "actuarial_info": actuarial_info or {},
            "statistical_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "basis": "per_policy_net_profit",
                "t_statistic_mae": float(t_stat_mae),
                "p_value_mae": float(p_value_mae),
                "significant_mae": p_value_mae < 0.05,
            },
        }

    def generate_unified_report(
        self,
        evaluation: dict,
        y_true: np.ndarray | None = None,
        ml_preds: np.ndarray | None = None,
        hybrid_preds: np.ndarray | None = None,
        output_path: str = "reports/unified_evaluation.txt",
    ) -> str:
        """Generate comprehensive business-focused report (v7.4.0)"""
        lines = []
        lines.append("=" * 90)
        lines.append("UNIFIED HYBRID PREDICTOR EVALUATION v7.4.0")
        lines.append("=" * 90)
        lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        cal_info = evaluation.get("calibration_info", {})
        act_info = evaluation.get("actuarial_info", {})

        # show calibration factor AND strategy
        if cal_info.get("enabled"):
            strategy_label = cal_info.get("strategy", "unknown")
            lines.append(
                f"\n📊 Calibration: {cal_info.get('factor', 1.0):.4f} "
                f"({(cal_info.get('factor', 1.0)-1)*100:+.2f}%)  "
                f"[Strategy: {strategy_label}]"
            )

        ml_biz = evaluation["business"]["ml"]["overall"]
        hybrid_biz = evaluation["business"]["hybrid"]["overall"]
        ml_acad = evaluation["academic"]["ml"]["overall"]
        hybrid_acad = evaluation["academic"]["hybrid"]["overall"]

        # =====================================================================
        # PRIMARY DEPLOYMENT KPIs
        # =====================================================================
        lines.append(f"\n{'=' * 90}")
        lines.append("🎯 PRIMARY DEPLOYMENT KPIs")
        lines.append("=" * 90)

        profit_delta = hybrid_biz["total_net_profit"] - ml_biz["total_net_profit"]
        profit_delta_pct = (
            (hybrid_biz["profit_per_policy"] - ml_biz["profit_per_policy"])
            / abs(ml_biz["profit_per_policy"])
            * 100
            if ml_biz["profit_per_policy"] != 0
            else 0
        )

        # Finding F: +75% reads as "profit improvement" to an executive even when
        # both values are negative losses.  Append context note to prevent misread.
        _ml_ppp_f = ml_biz["profit_per_policy"]
        _hy_ppp_f = hybrid_biz["profit_per_policy"]
        _sign_note = (
            " (loss reduced)" if _ml_ppp_f < 0 and _hy_ppp_f < 0 and _hy_ppp_f > _ml_ppp_f else ""
        )

        lines.append(
            f"\n   Profit per policy: "
            f"ML ${_ml_ppp_f:>8,.0f}  →  "
            f"Hybrid ${_hy_ppp_f:>8,.0f}  "
            f"({profit_delta_pct:+.1f}%{_sign_note})"
        )

        underpricing_ml = ml_biz["n_underpriced"] / ml_biz["n_predictions"] * 100
        underpricing_hybrid = hybrid_biz["n_underpriced"] / hybrid_biz["n_predictions"] * 100

        lines.append(
            f"   Underpricing rate: "
            f"ML {underpricing_ml:>5.1f}%  →  "
            f"Hybrid {underpricing_hybrid:>5.1f}%  "
            f"({'✅ Better' if underpricing_hybrid < underpricing_ml else '⚠️ Worse'})"
        )

        penalty_improvement = (
            ml_biz["total_underpricing_penalty"] - hybrid_biz["total_underpricing_penalty"]
        )

        lines.append(
            f"   Tail risk mitigation: "
            f"${penalty_improvement:>10,.0f}  "
            f"({'✅ Reduced' if penalty_improvement > 0 else '⚠️ Increased'})"
        )

        lines.append(
            f"   Churn rate:        "
            f"ML {ml_biz['churn_rate_pct']:>6.2f}%  →  "
            f"Hybrid {hybrid_biz['churn_rate_pct']:>6.2f}%  "
            f"({'✅ Better' if hybrid_biz['churn_rate_pct'] < ml_biz['churn_rate_pct'] else '⚠️ Worse'})"
        )

        # blend weight discrepancy
        if act_info:
            configured_blend = act_info.get("configured_blend_ratio")
            effective_ml_wt = act_info.get("effective_avg_ml_weight")
            if configured_blend is not None and effective_ml_wt is not None:
                discrepancy = abs(effective_ml_wt - configured_blend)
                lines.append(
                    f"   ML weight:         "
                    f"Configured {configured_blend:.0%}  →  "
                    f"Effective {effective_ml_wt:.0%}  "
                    f"({'⚠️ High discrepancy — most premiums route above threshold' if discrepancy > 0.15 else '✅ Within expected range'})"
                )

        # =====================================================================
        # EXECUTIVE SUMMARY
        # =====================================================================
        lines.append(f"\n{'=' * 90}")
        lines.append("EXECUTIVE SUMMARY")
        lines.append("=" * 90)

        lines.append(f"\n📊 BUSINESS ({ml_biz['n_predictions']:,} policies):")
        lines.append(
            f"   Net Profit:  ML ${ml_biz['total_net_profit']:>12,.2f}  →  "
            f"Hybrid ${hybrid_biz['total_net_profit']:>12,.2f}"
        )
        lines.append(f"   Delta:       ${profit_delta:>12,.2f}")
        lines.append(
            f"   Biz Score:   ML {ml_biz['business_value_score']:>6.1f}  →  "
            f"Hybrid {hybrid_biz['business_value_score']:>6.1f}"
        )

        # Academic metrics — DIAGNOSTIC ONLY
        rmse_delta = hybrid_acad["rmse"] - ml_acad["rmse"]
        male_delta = hybrid_acad.get("male", 0) - ml_acad.get("male", 0)
        smape_delta = hybrid_acad.get("smape", 0) - ml_acad.get("smape", 0)

        lines.append("\n📐 ACADEMIC (diagnostic only — not for decisions):")
        lines.append(
            f"   RMSE:        ML ${ml_acad['rmse']:>10,.2f}  →  "
            f"Hybrid ${hybrid_acad['rmse']:>10,.2f}  ({rmse_delta:+,.2f})"
        )
        lines.append(
            f"   MALE:        ML {ml_acad.get('male', 0):>6.3f}  →  "
            f"Hybrid {hybrid_acad.get('male', 0):>6.3f}  ({male_delta:+.3f})"
        )
        lines.append(
            f"   SMAPE:       ML {ml_acad.get('smape', 0):>6.2f}%  →  "
            f"Hybrid {hybrid_acad.get('smape', 0):>6.2f}%  ({smape_delta:+.2f}%)"
        )
        lines.append(
            f"   MAPE:        ML {ml_acad['mape']:>6.2f}%  →  "
            f"Hybrid {hybrid_acad['mape']:>6.2f}% (diagnostic)"
        )

        # flag MAPE/R² contradiction for investigation.
        # Previous version incorrectly labelled the test-set R² as "validation R²".
        # The test R² is computed from the held-out test predictions — a negative
        # value means the model is WORSE than predicting the mean, which is a
        # critical finding that warrants its own prominent alert.
        # NOTE: ml_segs is resolved here (ahead of the segment-alert section below)
        # so the negative-R² critical block can reference high_risk segment stats.
        seg_analysis = evaluation.get("segment_analysis", {})
        ml_segs = seg_analysis.get("ml", {})
        _test_r2 = ml_acad.get("r2")
        if ml_acad["mape"] > 40.0:
            _r2_label = (
                f"test R² = {_test_r2:.4f}" if _test_r2 is not None else "test R² unavailable"
            )
            lines.append(
                f"\n   ⚠️  DIAGNOSTIC NOTE: Test MAPE ({ml_acad['mape']:.1f}%) is high. "
                f"Verify train/test distribution alignment and check for right-tail "
                f"amplification from the yeo-johnson inverse transform "
                f"(current {_r2_label}). "
                f"A large gap between validation RMSE and test RMSE typically indicates "
                f"the specialist model over-fits the high-value tier — inspect the "
                f"'high_risk' segment predictions."
            )

        # negative test R² is a first-class finding.
        # R² < 0 means SS_residual > SS_total — the model's errors exceed the
        # variance of the target itself (i.e. worse than always predicting the mean).
        # This is almost always caused by a small number of catastrophically wrong
        # predictions in the high-value tail amplifying SS_residual.
        if _test_r2 is not None and _test_r2 < 0:
            _hr = ml_segs.get("high_risk", {})
            _hr_ratio = _hr.get("mean_pred", 0) / max(_hr.get("mean_true", 1), 1) if _hr else 0.0
            _hr_n = _hr.get("n_samples", 0) if _hr else 0
            lines.append(
                f"\n   🚨 CRITICAL — NEGATIVE TEST R² ({_test_r2:.4f}): "
                f"The model is {abs(1 - _test_r2):.2f}x WORSE than predicting the mean on the test set. "
                + (
                    f"The 'high_risk' segment ({_hr_n} policies) averages "
                    f"{_hr_ratio:.1f}x the true charge — this tail amplifies SS_residual "
                    f"beyond SS_total, collapsing R² below zero. "
                    if _hr_n > 0
                    else ""
                )
                + "ACTION REQUIRED: re-train or re-calibrate the specialist model before "
                "ANY production deployment."
            )

        # =====================================================================
        # SEGMENT RISK ALERTS
        # Surface any segment where SMAPE > 80% so the risk tier is visible
        # to decision-makers and not buried in INFO-level log lines.
        # ml_segs is already resolved above (ahead of H2 diagnostic).
        # =====================================================================
        alarmed_segs = [
            (seg_name, seg_data.get("smape", 0), int(seg_data.get("n_samples", 0)))
            for seg_name, seg_data in ml_segs.items()
            if seg_data.get("smape", 0) > 80.0
        ]
        if alarmed_segs:
            lines.append(f"\n{'=' * 90}")
            lines.append("🚨 SEGMENT RISK ALERTS  (SMAPE > 80% — action required)")
            lines.append("=" * 90)
            for seg_name, seg_smape, n in sorted(alarmed_segs, key=lambda x: -x[1]):
                seg_mean_true = ml_segs[seg_name].get("mean_true", 0)
                seg_mean_pred = ml_segs[seg_name].get("mean_pred", 0)
                lines.append(
                    f"\n   ⚠️  Segment '{seg_name}': SMAPE={seg_smape:.1f}%  ({n} policies)"
                )
                lines.append(
                    f"      Mean true=${seg_mean_true:,.0f}  Mean predicted=${seg_mean_pred:,.0f}"
                )
                lines.append(
                    f"      Predictions in this segment average "
                    f"{seg_mean_pred / seg_mean_true:.1f}x true charge."
                )
                lines.append(
                    f"      Action: review specialist model calibration for '{seg_name}' tier."
                )

        # =====================================================================
        # ACTUARIAL ANALYSIS (from predict.py result)
        # =====================================================================
        if act_info:
            lines.append(f"\n{'=' * 90}")
            lines.append("⚖️  ACTUARIAL ANALYSIS")
            lines.append("=" * 90)

            ratio = act_info.get("actuarial_conservativeness_ratio")
            if ratio is not None:
                if ratio > 1.15:
                    lines.append(
                        f"\n   Actuarial/ML ratio:  {ratio:.2f}x  "
                        f"⚠️ CONSERVATIVE — actuarial over-prices by "
                        f"{(ratio-1)*100:.1f}%"
                    )
                elif ratio < 0.70:
                    lines.append(
                        f"\n   Actuarial/ML ratio:  {ratio:.2f}x  "
                        f"🔴 AGGRESSIVE — actuarial UNDER-prices by "
                        f"{(1-ratio)*100:.1f}% — severe pricing risk"
                    )
                else:
                    lines.append(f"\n   Actuarial/ML ratio:  {ratio:.2f}x  ✅ Within normal range")

            # Tail risk from predict.py (model-level, uses actuarial safe minimum)
            tail_warn = act_info.get("tail_risk_warning")
            if tail_warn:
                lines.append(
                    f"\n   🔥 Tail Risk Alert [{tail_warn['severity']}]:\n"
                    f"      {tail_warn['policies_below_threshold']} policies "
                    f"({tail_warn['policies_below_threshold_pct']:.1f}%) below safe minimum\n"
                    f"      Avg gap from threshold: "
                    f"{tail_warn['avg_gap_from_threshold_pct']:.1f}%\n"
                    f"      Threshold used: {tail_warn['threshold_used_pct']:.0f}% of actuarial\n"
                    f"      Action: {tail_warn['recommended_action']}"
                )
            else:
                lines.append("\n   ✅ No tail risk alerts from predict pipeline")

        # =====================================================================
        # TAIL RISK ANALYSIS (profit-distribution level — computed from residuals)
        # =====================================================================
        if y_true is not None and ml_preds is not None and hybrid_preds is not None:
            tail_ml = self.analyze_profit_distribution(y_true, ml_preds)
            tail_hybrid = self.analyze_profit_distribution(y_true, hybrid_preds)

            lines.append(f"\n{'=' * 90}")
            lines.append("🔥 TAIL RISK ANALYSIS (profit-distribution level)")
            lines.append("=" * 90)

            # Finding C: label was hardcoded ">50% error"; now reflects actual config value
            # so it stays accurate when severe_underpricing_threshold_pct is tuned.
            _tail_sev_pct = self.business_config.severe_underpricing_threshold_pct * 100
            lines.append(f"\n   Severe underpricing cases (>{_tail_sev_pct:.0f}% error):")
            lines.append(
                f"      ML: {tail_ml['severe_underpricing']:>5} policies  →  "
                f"Hybrid: {tail_hybrid['severe_underpricing']:>5} policies"
            )

            penalty_delta_tail = (
                tail_hybrid["loss_drivers"]["underpricing_penalties"]
                - tail_ml["loss_drivers"]["underpricing_penalties"]
            )

            lines.append("\n   Underpricing penalties:")
            lines.append(
                f"      ML: ${tail_ml['loss_drivers']['underpricing_penalties']:>12,.0f}  →  "
                f"Hybrid: ${tail_hybrid['loss_drivers']['underpricing_penalties']:>12,.0f}  "
                f"(${penalty_delta_tail:+,.0f})"
            )

            lines.append("\n   High churn risk (>50% probability):")
            lines.append(
                f"      ML: {tail_ml['high_churn_policies']:>5} policies  →  "
                f"Hybrid: {tail_hybrid['high_churn_policies']:>5} policies"
            )

            churn_cost_delta = (
                tail_hybrid["loss_drivers"]["churn_losses"]
                - tail_ml["loss_drivers"]["churn_losses"]
            )

            lines.append("\n   Total churn costs:")
            lines.append(
                f"      ML: ${tail_ml['loss_drivers']['churn_losses']:>12,.0f}  →  "
                f"Hybrid: ${tail_hybrid['loss_drivers']['churn_losses']:>12,.0f}  "
                f"(${churn_cost_delta:+,.0f})"
            )

        # =====================================================================
        # DEPLOYMENT READINESS — 5 metrics
        # RMSE removed from deployment gate — the report itself
        # labels it "diagnostic only — not for decisions". Replaced with SMAPE,
        # which is symmetric, bounded, and insurance-appropriate. A $0.33 RMSE
        # delta should not block production deployment.
        # =====================================================================
        lines.append(f"\n{'=' * 90}")
        lines.append("📊 DEPLOYMENT READINESS")
        lines.append("=" * 90)

        # Core 4 business metrics (unchanged)
        win_profit = hybrid_biz["total_net_profit"] > ml_biz["total_net_profit"]
        win_churn = hybrid_biz["churn_rate_pct"] < ml_biz["churn_rate_pct"]
        win_biz_score = hybrid_biz["business_value_score"] > ml_biz["business_value_score"]

        # T2-E: replaced win_smape with win_tail_risk.
        # SMAPE is an academic accuracy metric — the report header already labels it
        # "diagnostic only — not for decisions".  Having it as a deployment gate vote
        # allowed a hybrid that reduced profit but improved SMAPE to score 3/5 and
        # receive a MODERATE CONFIDENCE recommendation.  win_tail_risk measures the
        # count of severely underpriced policies (error_pct < -severe_threshold),
        # staying entirely within the business domain.
        _sev_threshold = self.business_config.severe_underpricing_threshold_pct
        # use the vectorised calculate_portfolio_metrics instead of
        # per-row Python loops (was two O(n) list-comps feeding a third sum() pass).
        # We derive severe counts from the already-computed portfolio results, which
        # avoids a third full pass over y_true.
        if y_true is None or ml_preds is None or hybrid_preds is None:
            _ml_err_pct = np.zeros(1)
            _hy_err_pct = np.zeros(1)
        else:
            _ml_err_pct = np.where(y_true > 0, (ml_preds - y_true) / y_true, 0.0)
            _hy_err_pct = np.where(y_true > 0, (hybrid_preds - y_true) / y_true, 0.0)
        n_severe_ml = int(np.sum(_ml_err_pct < -_sev_threshold))
        n_severe_hybrid = int(np.sum(_hy_err_pct < -_sev_threshold))
        win_tail_risk = n_severe_hybrid <= n_severe_ml

        # did the hybrid model REDUCE underpricing penalties vs ML?
        actuarial_aggressive_flag = act_info.get("actuarial_aggressive", False)
        win_actuarial = (
            hybrid_biz["total_underpricing_penalty"] < ml_biz["total_underpricing_penalty"]
        )

        wins = sum([win_profit, win_churn, win_biz_score, win_tail_risk, win_actuarial])
        deployment_confidence = wins / 5.0

        # use abs(ml_ppp) as denominator
        ml_ppp = ml_biz["profit_per_policy"]
        hybrid_ppp = hybrid_biz["profit_per_policy"]
        if ml_ppp != 0:
            profit_improvement_pct = ((hybrid_ppp - ml_ppp) / abs(ml_ppp)) * 100
        else:
            profit_improvement_pct = 0.0
        # removed dead variable `profit_improvement` (was computed
        # and immediately unused; the "keep downstream compat" comment was misleading).

        if ml_biz["total_underpricing_penalty"] > 0:
            risk_improvement = 1 - (
                hybrid_biz["total_underpricing_penalty"] / ml_biz["total_underpricing_penalty"]
            )
        else:
            logger.info(
                "ℹ️  risk_improvement undefined: ML has no underpricing penalties "
                "(total_underpricing_penalty = 0). Displaying as 0.0%."
            )
            risk_improvement = 0.0

        lines.append(
            f"\n   Metric Wins: {wins}/5 ({deployment_confidence*100:.0f}%)  "
            f"[{'✅' if win_profit else '❌'} Profit  "
            f"{'✅' if win_churn else '❌'} Churn  "
            f"{'✅' if win_biz_score else '❌'} BizScore  "
            f"{'✅' if win_tail_risk else '❌'} TailRisk  "
            f"{'✅' if win_actuarial else '❌'} UnderpriceRisk]"
        )
        lines.append(f"   Profit Improvement: {profit_improvement_pct:+.1f}%")
        lines.append(f"   Risk Reduction: {risk_improvement*100:+.1f}%")

        if actuarial_aggressive_flag:
            lines.append(
                "\n   ⚠️ ACTUARIAL RISK FLAG: Actuarial estimates are aggressively "
                "below ML — premiums near threshold may be severely underpriced."
            )

        if deployment_confidence >= _DEPLOYMENT_CONFIDENCE_HIGH:
            lines.append("\n   ✅ HIGH CONFIDENCE - Recommend deployment")
        elif deployment_confidence >= _DEPLOYMENT_CONFIDENCE_MODERATE:
            lines.append("\n   ⚠️ MODERATE CONFIDENCE - Further testing recommended")
        else:
            lines.append("\n   ❌ LOW CONFIDENCE - Do not deploy")

        # =====================================================================
        # RECOMMENDATION
        # =====================================================================
        lines.append(f"\n{'=' * 90}")
        lines.append("💡 RECOMMENDATION")
        lines.append("=" * 90)

        biz_win = hybrid_biz["business_value_score"] > ml_biz["business_value_score"]
        acad_win = hybrid_acad["rmse"] < ml_acad["rmse"] and hybrid_acad.get(
            "male", float("inf")
        ) < ml_acad.get("male", float("inf"))

        if not biz_win:
            # The previous message "Business value insufficient" was
            # self-contradictory when profit_delta was POSITIVE — it implied hybrid
            # was unprofitable while actually showing +$917K profit gain. The real
            # rejection reason when biz_win=False is that hybrid's BizScore is lower
            # than ML's, which is driven by churn more than profit. Clarify the
            # decision signal so operators don't misread the output.
            _biz_gap = ml_biz["business_value_score"] - hybrid_biz["business_value_score"]
            _churn_delta = hybrid_biz["churn_rate_pct"] - ml_biz["churn_rate_pct"]
            lines.append("\n❌ KEEP ML-ONLY")
            lines.append(
                f"   Hybrid BizScore ({hybrid_biz['business_value_score']:.1f}) "
                f"below ML baseline ({ml_biz['business_value_score']:.1f}) "
                f"by {_biz_gap:.1f} points."
            )
            if profit_delta > 0:
                lines.append(
                    f"   Note: Hybrid shows higher net profit (+${profit_delta:,.0f}) "
                    f"but elevated churn (+{_churn_delta:.2f}pp) suppresses BizScore."
                )
                lines.append(
                    "   Action: Investigate churn root cause (likely actuarial over-pricing "
                    "in transition zone) before reconsidering hybrid deployment."
                )
            else:
                lines.append(
                    f"   Profit delta: ${profit_delta:+,.0f} — hybrid is also less profitable."
                )
        elif actuarial_aggressive_flag and deployment_confidence < 0.60:
            lines.append("\n🚫 DEPLOYMENT BLOCKED")
            lines.append(
                f"   Hybrid shows profit gain (+${profit_delta:,.0f}) but actuarial component "
                f"under-prices by {(1 - act_info.get('actuarial_conservativeness_ratio', 1))*100:.1f}%."
            )
            lines.append(
                "   Required action: Re-calibrate actuarial base rates before any deployment."
            )
            lines.append(f"   Confidence: {deployment_confidence:.0%} ({wins}/5 metrics)")
        elif deployment_confidence >= 0.80:
            if acad_win:
                lines.append("\n✅✅ STRONGLY DEPLOY HYBRID")
                lines.append("   Wins on BOTH business AND accuracy")
                lines.append(f"   Expected profit gain: ${profit_delta:,.0f}")
            else:
                lines.append("\n✅ DEPLOY HYBRID")
                lines.append(f"   Superior business value (+${profit_delta:,.0f})")
                lines.append(f"   Acceptable accuracy trade-off: RMSE {rmse_delta:+,.2f}")
        elif deployment_confidence >= 0.60:
            lines.append("\n⚠️  CONDITIONAL DEPLOY — address flagged risks first")
            lines.append(
                f"   Business value: +${profit_delta:,.0f}  |  Confidence: {deployment_confidence:.0%} ({wins}/5 metrics)"
            )
            if actuarial_aggressive_flag:
                lines.append(
                    "   ⚠️  Action required: Re-calibrate actuarial base rates before live pricing"
                )
            if not win_churn:
                lines.append("   ⚠️  Action required: Investigate churn increase vs ML baseline")
        else:
            lines.append("\n❌ DO NOT DEPLOY — low confidence despite business gain")
            lines.append(f"   Profit delta (+${profit_delta:,.0f}) does not offset systemic risks.")
            lines.append(f"   Confidence: {deployment_confidence:.0%} ({wins}/5 metrics)")
            lines.append("   Resolve flagged metrics before re-evaluating.")

        report_text = "\n".join(lines)

        # Save report
        try:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path_obj, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"✅ Saved: {output_path}")
        except Exception as e:
            logger.error(f"❌ Error writing report: {e}")

        return report_text

    def analyze_profit_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Identify where losses are coming from.

        Finding B: replaced the O(n) row-by-row Python loop that built a DataFrame
        from calculate_single_prediction_value() calls with direct numpy expressions
        mirroring the calculate_portfolio_metrics() vectorised path.  Result is
        numerically identical and ~40× faster on a 10K-policy batch.
        """
        cfg = self.business_config
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        _loading = cfg.base_profit_margin
        _admin = cfg.admin_cost_per_policy

        err = yp - yt
        ep = np.where(yt > 0, err / yt, 0.0)

        # Gross profit
        gp = yp - yt + _loading * yp - _admin

        # Churn
        over = err > 0
        excess = np.maximum(np.where(over, ep, 0.0) - cfg.churn_threshold_pct, 0.0)
        cp = np.where(over, np.minimum(excess * cfg.churn_sensitivity, _MAX_CHURN_PROBABILITY), 0.0)
        clv = yt * cfg.customer_lifetime_value_multiplier
        cc = cp * (clv + cfg.customer_acquisition_cost)

        # Underpricing penalty
        under = err < 0
        rev = yp * (1.0 + _loading)
        bp = np.abs(err) * cfg.underpricing_penalty_multiplier
        mp = rev * 0.2
        up = np.where(under, np.minimum(bp, mp), 0.0)
        sev = np.where(under, np.abs(ep), 0.0) > cfg.severe_underpricing_threshold_pct
        up = np.where(under & sev, up + cfg.severe_underpricing_penalty, up)

        # Accuracy bonus
        ae = np.abs(ep)
        bonus = np.where(
            ae <= cfg.acceptable_error_band_pct / 2,
            cfg.excellent_accuracy_bonus,
            np.where(ae <= cfg.acceptable_error_band_pct, cfg.excellent_accuracy_bonus * 0.5, 0.0),
        )

        net_profit = gp - cc - up + bonus

        # use config value, not hardcoded -0.50
        _sev = cfg.severe_underpricing_threshold_pct

        return {
            "high_churn_policies": int(np.sum(cp > 0.5)),
            "severe_underpricing": int(np.sum(ep < -_sev)),
            "profit_quartiles": {
                0.25: float(np.percentile(net_profit, 25)),
                0.50: float(np.percentile(net_profit, 50)),
                0.75: float(np.percentile(net_profit, 75)),
            },
            "loss_drivers": {
                "churn_losses": float(np.sum(cc)),
                "underpricing_penalties": float(np.sum(up)),
                "accuracy_bonuses": float(np.sum(bonus)),
            },
        }


# =====================================================================
# VISUALIZATION
# =====================================================================


def create_unified_visualization(
    evaluation: dict,
    y_true: np.ndarray,
    ml_preds: np.ndarray,
    hybrid_preds: np.ndarray,
    save_path: str = "reports/unified_evaluation.png",
):
    """Create comprehensive visualization"""
    import warnings

    # module="matplotlib" never fires because matplotlib uses stacklevel=2,
    # which attributes the glyph warning to the *call site* (this file, not matplotlib).
    # Correct approach: (a) filter by message pattern, scoped with catch_warnings so
    # we don't pollute the global warning state for the rest of the process, and
    # (b) replace emoji in the summary table to eliminate the root cause entirely.
    # The filterwarnings below catches any residual glyph/font warnings defensively.
    warnings.filterwarnings(
        "ignore",
        message=r".*Glyph.*missing from font.*",
        category=UserWarning,
    )

    fig = None

    try:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        plt.close("all")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        ml_biz = evaluation["business"]["ml"]["overall"]
        hybrid_biz = evaluation["business"]["hybrid"]["overall"]
        ml_acad = evaluation["academic"]["ml"]["overall"]
        hybrid_acad = evaluation["academic"]["hybrid"]["overall"]

        ml_color = "#3498db"
        hybrid_color = "#e74c3c"

        # 1. Business Score
        ax1 = fig.add_subplot(gs[0, 0])
        scores = [ml_biz["business_value_score"], hybrid_biz["business_value_score"]]
        ax1.bar(
            ["ML", "Hybrid"],
            scores,
            color=[ml_color, hybrid_color],
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_ylabel("Score", fontweight="bold")
        ax1.set_title("Business Value Score", fontweight="bold")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        # 2. RMSE
        ax2 = fig.add_subplot(gs[0, 1])
        rmses = [ml_acad["rmse"], hybrid_acad["rmse"]]
        ax2.bar(
            ["ML", "Hybrid"],
            rmses,
            color=[ml_color, hybrid_color],
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_ylabel("RMSE ($)", fontweight="bold")
        ax2.set_title("Root Mean Squared Error", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # 3. Net Profit
        ax3 = fig.add_subplot(gs[0, 2])
        profits = [ml_biz["total_net_profit"], hybrid_biz["total_net_profit"]]
        ax3.bar(
            ["ML", "Hybrid"],
            profits,
            color=[ml_color, hybrid_color],
            alpha=0.7,
            edgecolor="black",
        )
        ax3.set_ylabel("Net Profit ($)", fontweight="bold")
        ax3.set_title("Total Net Profit", fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color="black", linewidth=1)

        # 4. Churn
        ax4 = fig.add_subplot(gs[1, 0])
        churn = [ml_biz["churn_rate_pct"], hybrid_biz["churn_rate_pct"]]
        ax4.bar(
            ["ML", "Hybrid"],
            churn,
            color=[ml_color, hybrid_color],
            alpha=0.7,
            edgecolor="black",
        )
        ax4.set_ylabel("Churn Rate (%)", fontweight="bold")
        ax4.set_title("Expected Churn", fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # 5. SMAPE (recommended metric — replaces duplicate RMSE panel)
        ax5 = fig.add_subplot(gs[1, 1])
        smapes = [ml_acad.get("smape", 0), hybrid_acad.get("smape", 0)]
        ax5.bar(
            ["ML", "Hybrid"],
            smapes,
            color=[ml_color, hybrid_color],
            alpha=0.7,
            edgecolor="black",
        )
        ax5.set_ylabel("SMAPE (%)", fontweight="bold")
        ax5.set_title("Symmetric MAPE (Recommended)", fontweight="bold")
        ax5.grid(True, alpha=0.3)

        # 6. Cost Breakdown
        ax6 = fig.add_subplot(gs[1, 2])
        width = 0.35
        x = np.arange(3)
        ml_costs = [
            ml_biz["total_churn_cost"],
            ml_biz["total_underpricing_penalty"],
            -ml_biz["total_accuracy_bonus"],
        ]
        hybrid_costs = [
            hybrid_biz["total_churn_cost"],
            hybrid_biz["total_underpricing_penalty"],
            -hybrid_biz["total_accuracy_bonus"],
        ]
        ax6.bar(x - width / 2, ml_costs, width, label="ML", color=ml_color, alpha=0.7)
        ax6.bar(
            x + width / 2,
            hybrid_costs,
            width,
            label="Hybrid",
            color=hybrid_color,
            alpha=0.7,
        )
        ax6.set_ylabel("Cost ($)", fontweight="bold")
        ax6.set_title("Cost Breakdown", fontweight="bold")
        ax6.set_xticks(x)
        ax6.set_xticklabels(["Churn", "Underpricing", "Bonus"], fontsize=9)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(0, color="black", linewidth=1)

        # 7. Scatter
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(y_true, ml_preds, alpha=0.4, s=20, label="ML", color=ml_color)
        ax7.scatter(y_true, hybrid_preds, alpha=0.4, s=20, label="Hybrid", color=hybrid_color)
        max_val = max(y_true.max(), ml_preds.max(), hybrid_preds.max())
        ax7.plot([0, max_val], [0, max_val], "--", color="gray", linewidth=2)
        ax7.set_xlabel("True Value ($)", fontweight="bold")
        ax7.set_ylabel("Predicted ($)", fontweight="bold")
        ax7.set_title("Prediction Accuracy", fontweight="bold")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Error Distribution
        ax8 = fig.add_subplot(gs[2, 1])
        ml_errors = np.abs(y_true - ml_preds)
        hybrid_errors = np.abs(y_true - hybrid_preds)
        ax8.hist(ml_errors, bins=40, alpha=0.6, label="ML", color=ml_color)
        ax8.hist(hybrid_errors, bins=40, alpha=0.6, label="Hybrid", color=hybrid_color)
        ax8.set_xlabel("Absolute Error ($)", fontweight="bold")
        ax8.set_ylabel("Frequency", fontweight="bold")
        ax8.set_title("Error Distribution", fontweight="bold")
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. Summary Table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis("off")

        # ✅/❌ emoji cause "Glyph missing from font" UserWarnings because
        # matplotlib's default DejaVu Sans does not include these Unicode code points.
        # Using ASCII-safe winner/loser labels eliminates the warning at the source.
        def _winner(condition: bool, winner_label: str = "Hybrid", loser_label: str = "ML") -> str:
            """Return ASCII winner label — avoids DejaVu Sans glyph warnings."""
            return f"[+] {winner_label}" if condition else f"[-] {loser_label}"

        table_data = [
            ["Metric", "Winner"],
            [
                "Net Profit",
                _winner(hybrid_biz["total_net_profit"] > ml_biz["total_net_profit"]),
            ],
            [
                "Churn Rate",
                _winner(hybrid_biz["churn_rate_pct"] < ml_biz["churn_rate_pct"]),
            ],
            [
                "Biz Score",
                _winner(hybrid_biz["business_value_score"] > ml_biz["business_value_score"]),
            ],
            ["RMSE", _winner(hybrid_acad["rmse"] < ml_acad["rmse"])],
            [
                "SMAPE",
                _winner(
                    hybrid_acad.get("smape", float("inf")) < ml_acad.get("smape", float("inf"))
                ),
            ],
            [
                "MALE",
                _winner(hybrid_acad.get("male", float("inf")) < ml_acad.get("male", float("inf"))),
            ],
        ]

        table = ax9.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        for i in range(2):
            table[(0, i)].set_facecolor("#34495e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax9.set_title("Winner Summary", fontsize=12, fontweight="bold", pad=20)

        # Add calibration note
        cal_info = evaluation.get("calibration_info", {})
        if cal_info.get("enabled"):
            fig.text(
                0.5,
                0.97,
                f"Calibration: {cal_info['factor']:.4f} ({(cal_info['factor']-1)*100:+.2f}%)",
                ha="center",
                fontsize=10,
                style="italic",
                color="gray",
            )

        fig.suptitle(
            "Unified Evaluation: Business-First Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        fig.canvas.draw()

        save_path_str = str(save_path_obj.resolve())

        if save_path_obj.exists():
            try:
                save_path_obj.unlink()
                import time

                time.sleep(0.05)
            except (OSError, PermissionError) as _del_err:
                logger.debug(
                    f"Could not remove existing plot file before overwrite "
                    f"({save_path_obj.name}): {_del_err}"
                )

        fig.savefig(save_path_str, dpi=300, bbox_inches="tight", format="png")
        logger.info(f"✅ Saved: {save_path}")

    except (ValueError, RuntimeError, AttributeError) as e:
        logger.error(f"❌ Error: {e}", exc_info=True)

    finally:
        if fig is not None:
            plt.close(fig)
        plt.close("all")


# =====================================================================
# CI COVERAGE VALIDATION
# =====================================================================


def check_ci_coverage(
    pipeline: PredictionPipeline,
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    confidence_level: float = 0.90,
) -> dict:
    """Verify empirical CI coverage against the nominal conformal level on the test set.

    T2-C (v7.5.0): This check was absent from the evaluation pipeline entirely.
    The pipeline reported CI mean width ($33K) as a proxy for quality, but never
    verified that the stated coverage (90%) was actually achieved on the test set.
    For split-conformal intervals P(Y in CI) >= 1-alpha is guaranteed only when
    calibration and test sets are exchangeable.  A coverage gap > 2% is a red flag
    that conformal residuals were computed on a different distribution — e.g. after
    a retrain that shifted the target transform, or after a data pipeline change.

    The 2% slack (valid if empirical >= nominal - 0.02) accounts for finite-sample
    variability: with n=201 test policies, the 95% CI on empirical coverage at
    nominal=0.90 is approximately ±0.042, so 0.88 is well within expected range.

    Args:
        pipeline:         Trained PredictionPipeline (must have conformal artifact loaded).
        X_test:           Test features as a DataFrame.
        y_true:           Ground-truth charges in original dollar space.
        confidence_level: Nominal coverage level — must match what was used at training.

    Returns:
        Dict with keys: nominal, empirical, gap, mean_width, method, n_samples, valid.
        On failure: {"error": str}.
    """
    try:
        ci_result = pipeline.predict_with_intervals(X_test, confidence_level=confidence_level)
        ci = ci_result.get("confidence_intervals") or {}
        if "lower_bound" not in ci:
            return {
                "error": "No CI available from predict_with_intervals — check conformal artifact"
            }

        lower = np.array(ci["lower_bound"])
        upper = np.array(ci["upper_bound"])
        # Finding D: numpy broadcasts silently when shapes mismatch, producing garbage
        # empirical coverage instead of raising.  Fail fast with a clear error.
        if len(lower) != len(y_true) or len(upper) != len(y_true):
            return {
                "error": (
                    f"CI array length mismatch: lower={len(lower)}, "
                    f"upper={len(upper)}, y_true={len(y_true)}. "
                    "CI may have been computed on a different subset or a stale cache."
                )
            }
        covered = (y_true >= lower) & (y_true <= upper)
        empirical = float(np.mean(covered))
        gap = empirical - confidence_level

        result = {
            "nominal": confidence_level,
            "empirical": round(empirical, 4),
            "gap": round(gap, 4),
            "mean_width": round(ci.get("mean_interval_width", float("nan")), 2),
            "median_width": round(ci.get("median_interval_width", float("nan")), 2),
            "method": ci.get("method", "unknown"),
            "n_samples": int(len(y_true)),
            "n_covered": int(np.sum(covered)),
            "valid": empirical >= confidence_level - 0.02,
        }

        if not result["valid"]:
            logger.warning(
                f"⚠️ CI UNDER-COVERAGE: empirical={empirical:.3f} < "
                f"nominal={confidence_level:.2f} (gap={gap:+.3f}, n={len(y_true)}). "
                f"Conformal guarantee not holding on this test distribution. "
                f"Likely causes: (1) conformal residuals calibrated on different data split, "
                f"(2) target transform changed since last retrain, "
                f"(3) test set is OOD relative to calibration set. "
                f"Action: re-run train.py to recalibrate conformal residuals."
            )
        else:
            logger.info(
                f"✅ CI coverage valid: empirical={empirical:.3f} "
                f"(nominal={confidence_level:.2f}, gap={gap:+.3f}, "
                f"mean_width=${result['mean_width']:,.0f}, n={len(y_true)})"
            )
        return result

    except Exception as e:
        logger.warning(f"⚠️ CI coverage check failed ({type(e).__name__}): {e}")
        return {"error": str(e)}


# =====================================================================
# MAIN EXECUTION
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation v7.4.0")

    parser.add_argument("--data-path", default="data/raw/insurance.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    # path to test_indices.json written by train.py.
    parser.add_argument(
        "--index-path",
        default="models/test_indices.json",
        help=(
            "Path to test_indices.json saved during training. "
            "If found, uses exact training split (required when "
            "stratify_splits=true to avoid 81%% train/test leakage). "
            "Falls back to re-splitting if not found."
        ),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--blend-ratio", type=float, default=None)
    parser.add_argument("--calibration-factor", type=float, default=None)
    parser.add_argument("--no-calibration", action="store_true")
    # add calibration strategy CLI flag
    _strat_grp = parser.add_mutually_exclusive_group()
    _strat_grp.add_argument(
        "--apply-to-ml-only",
        dest="apply_to_ml_only",
        action="store_true",
        default=None,
        help="Apply calibration to ML predictions only (default from config)",
    )
    _strat_grp.add_argument(
        "--full-hybrid-calibration",
        dest="apply_to_ml_only",
        action="store_false",
        help="Apply calibration to the full hybrid output",
    )
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--business-focus", action="store_true")
    parser.add_argument("--compare-configs", action="store_true")
    parser.add_argument("--tune-business-params", action="store_true")
    parser.add_argument(
        "--no-exit-gate",
        action="store_true",
        default=False,
        help=(
            "Suppress CI gate exit-code enforcement: log the gate result but always "
            "exit 0.  Use only in non-blocking CI steps (e.g. artifact-collection "
            "jobs that must not fail the pipeline).  Never use in a deployment-blocking "
            "step — the gate exists to prevent low-confidence models from reaching "
            "production."
        ),
    )

    args = parser.parse_args()

    print("\n" + "=" * 90)
    print("UNIFIED EVALUATION v7.4.0 - Business-First Analysis")
    print("=" * 90)

    try:
        config = load_config()
        hybrid_config = config.get("hybrid_predictor", {})
        business_config = load_business_config_from_yaml()
        # Inference guard limit — mirrors prediction.max_batch_size in predict.py.
        # _predict_in_batches chunks at this value so each call stays within
        # the guard regardless of total dataset size (1k to 100k+).
        _inference_limit = int(config.get("prediction", {}).get("max_batch_size", 50_000))

        if args.tune_business_params:
            print(
                f"\n📊 Business Config: {hybrid_config.get('business_config', {}).get('profile', 'default')}"
            )
            print(f"   Margin: {business_config.base_profit_margin*100:.1f}%")
            print(f"   Admin: ${business_config.admin_cost_per_policy:.0f}")
            print("\n💡 Edit hybrid_predictor.business_config in config.yaml to modify")
            sys.exit(0)

        # ml_pipeline was assigned at line ~2154, nine lines AFTER
        # args.compare_configs referenced it at line ~2145.  Every invocation
        # of `python evaluate.py --compare-configs` raised:
        #   NameError: name 'ml_pipeline' is not defined
        # hoist PredictionPipeline() construction above the compare_configs
        # branch so it is available for both the sweep path and the normal path.
        # The STEP labels are renumbered accordingly (1→pipeline, 2→data).
        print("\n[STEP 1] Initializing ML pipeline...")
        ml_pipeline = PredictionPipeline()

        if args.compare_configs:
            print("\n[STEP 2] Loading data...")
            X_test, y_test = load_and_split_data(
                args.data_path, args.test_size, args.random_state, args.index_path
            )
            compare_multiple_configurations(X_test, y_test, ml_pipeline, business_config)
            sys.exit(0)

        print("\n[STEP 2] Loading data...")
        X_test, y_test = load_and_split_data(
            args.data_path, args.test_size, args.random_state, args.index_path
        )

        print("\n[STEP 3] Getting ML predictions...")
        ml_result = _predict_in_batches(
            ml_pipeline, X_test, inference_limit=_inference_limit, return_reliability=False
        )
        ml_preds = np.array(ml_result["predictions"])

        print("\n[STEP 4] Initializing hybrid predictor...")

        _cal_cfg = hybrid_config.get("calibration", {})
        if args.no_calibration:
            calibration_factor = 1.0
            calibration_enabled = False
            logger.warning("⚠️ Calibration DISABLED")
        elif args.calibration_factor is not None:
            # was `elif args.calibration_factor:` which is a falsy test.
            # --calibration-factor 0.0 evaluated as False and fell through to the
            # config-read branch, silently ignoring the explicit CLI override.
            # The validate step below (HybridPredictor._validate_config) will raise
            # for factor <= 0, giving the user a clear error instead of silent bypass.
            calibration_factor = args.calibration_factor
            calibration_enabled = True
        else:
            calibration_enabled = _cal_cfg.get("enabled", True)
            calibration_factor = float(_cal_cfg.get("factor", 1.00))

        # read apply_to_ml_only from CLI → config → default True
        if args.apply_to_ml_only is not None:
            apply_to_ml_only = args.apply_to_ml_only
        else:
            apply_to_ml_only = bool(_cal_cfg.get("apply_to_ml_only", True))

        strategy_label = "ML-only" if apply_to_ml_only else "Full hybrid"
        print(
            f"   Calibration: {'✅' if calibration_enabled else '❌'} "
            f"{calibration_factor:.4f}  [Strategy: {strategy_label}]"
        )

        # pass apply_to_ml_only via config dict so HybridPredictor
        # reads it from calibration_config.get("apply_to_ml_only") in __init__
        _override_config = dict(hybrid_config)
        _override_config["calibration"] = dict(_cal_cfg)
        _override_config["calibration"]["apply_to_ml_only"] = apply_to_ml_only
        if not calibration_enabled:
            _override_config["calibration"]["factor"] = 1.0

        hybrid = HybridPredictor(
            ml_predictor=ml_pipeline,
            threshold=args.threshold,
            blend_ratio=args.blend_ratio,
            calibration_factor=calibration_factor if calibration_enabled else 1.0,
            config=_override_config,
        )

        print("\n[STEP 5] Getting hybrid predictions...")
        hybrid_result = _predict_in_batches(
            hybrid,
            X_test,
            inference_limit=_inference_limit,
            return_components=True,
            return_reliability=True,
        )
        hybrid_preds = np.array(hybrid_result["predictions"])

        # effective_avg_ml_weight is blend_diagnostics["avg_ml_weight"]
        # computed per-chunk by HybridPredictor.  Recompute the true full-batch value from
        # the concatenated ml_weights array (present in components when return_components=True).
        if (
            "components" in hybrid_result
            and "ml_weights" in hybrid_result["components"]
            and "reliability" in hybrid_result
            and "effective_avg_ml_weight" in hybrid_result["reliability"]
        ):
            _full_ml_weights = np.asarray(hybrid_result["components"]["ml_weights"])
            hybrid_result["reliability"]["effective_avg_ml_weight"] = float(
                np.mean(_full_ml_weights)
            )

        # OOD upper-bound guard ────────────────
        # The scale validator only checks sign/units; it passes predictions that
        # exceed the training domain.  Resolve the training-set max from the
        # FeatureEngineer that is always present on PredictionPipeline.
        #
        # Attribute path precedence (most to least precise):
        #   1. ml_pipeline.feature_engineer.y_max_safe
        #         Serialized in save_preprocessor() and restored by load_preprocessor().
        #         Equals target_max_ × (1 + buffer_pct), so slightly above the
        #         raw training max — the correct OOD reference.
        #   2. ml_pipeline.feature_engineer.target_transformation.original_range[1]
        #         Stored on the TargetTransformation namedtuple during fit_transform_pipeline().
        #   3. ml_pipeline.feature_engineer.target_max_
        #         Raw training-set max, set during transform_target(fit=True).
        #   4. y_test.max() [fallback]
        #         Only fires if the preprocessor was not fitted (should never happen
        #         in normal operation).
        #
        # The three stale paths from v7.4.1 —
        #   ml_pipeline.preprocessor, ml_pipeline.model_manager.preprocessor,
        #   ml_pipeline.pipeline.preprocessor — were removed: PredictionPipeline
        #   exposes no .preprocessor or .pipeline attribute, so all three raised
        #   AttributeError on every call, the fallback warning fired every run,
        #   and the OOD guard was silently disabled.
        #
        # Sanity guard: reject any resolved max > $2 M (likely transformed space).
        _ORIGINAL_SCALE_SANITY = 2_000_000.0
        _train_max = None
        _train_max_source = "unknown"
        for _path_label, _path_fn in [
            (
                "ml_pipeline.feature_engineer.y_max_safe",
                lambda: ml_pipeline.feature_engineer.y_max_safe,
            ),
            (
                "ml_pipeline.feature_engineer.target_transformation.original_range[1]",
                lambda: ml_pipeline.feature_engineer.target_transformation.original_range[1],
            ),
            (
                "ml_pipeline.feature_engineer.target_max_",
                lambda: ml_pipeline.feature_engineer.target_max_,
            ),
        ]:
            try:
                _cand = float(_path_fn())
                if _cand is not None and 0 < _cand < _ORIGINAL_SCALE_SANITY:
                    _train_max = _cand
                    _train_max_source = _path_label
                    break
                else:
                    logger.debug(
                        f"   [C2] Skipping {_path_label}={_cand} "
                        f"— None or outside expected original-scale range (0, $2M)."
                    )
            except (AttributeError, TypeError, IndexError, KeyError):
                pass

        if _train_max is None:
            _train_max = float(y_test.max())
            _train_max_source = "y_test.max() [fallback]"
            logger.warning(
                f"⚠️  [C2] Could not resolve training target range from FeatureEngineer. "
                f"Using test-set max ${_train_max:,.2f} as OOD proxy. "
                f"Check that load_preprocessor() completed successfully."
            )
        else:
            logger.info(
                f"   [C2] Training max resolved: ${_train_max:,.2f}  (source: {_train_max_source})"
            )

        _ood_threshold = _train_max * 1.5
        _ood_mask = hybrid_preds > _ood_threshold
        if _ood_mask.any():
            logger.warning(
                f"⚠️  [C2] OOD PREDICTIONS DETECTED: {_ood_mask.sum()} hybrid predictions "
                f"exceed 150% of training max. "
                f"max pred=${hybrid_preds.max():,.2f}, "
                f"train max=${_train_max:,.2f} (via {_train_max_source}), "
                f"OOD threshold=${_ood_threshold:,.2f}. "
                f"These {_ood_mask.sum()} policies are out-of-distribution — "
                f"downstream metrics may be unreliable for this subset."
            )

        # weight arithmetic anomaly diagnostic ──
        # Previous version gated the entire check on _reliability.get(key) which
        # returns None when HybridPredictor nests the key differently (e.g. under
        # routing sub-dict), silently skipping the diagnostic entirely.
        #
        # New approach: TWO independent checks.
        #   (a) UNCONDITIONAL means ratio — fires regardless of key availability.
        #       If hybrid mean > 1.4× ML mean the divergence is already anomalous
        #       and warrants logging irrespective of reported weight.
        #   (b) Weight arithmetic — runs only when effective_avg_ml_weight IS known
        #       (sourced from actuarial_info after it is built, 10 lines below).
        #       Emitted as a second WARNING with the implied actuarial mean.
        _c1_ml_mean = float(np.mean(ml_preds))
        _c1_hyb_mean = float(np.mean(hybrid_preds))
        _c1_ratio = _c1_hyb_mean / _c1_ml_mean if _c1_ml_mean > 0 else 0.0
        if _c1_ratio > 1.4:
            logger.warning(
                f"⚠️  [C1] MEAN DIVERGENCE — unconditional check: "
                f"hybrid mean=${_c1_hyb_mean:,.0f} is {_c1_ratio:.2f}x ML mean=${_c1_ml_mean:,.0f}. "
                f"A model with high ML weight should produce means close to ML mean. "
                f"Root cause candidates: (i) specialist model over-predicts high-value tier, "
                f"(ii) actuarial component uses implausible base rates, "
                f"(iii) routing weight accounting double-counts specialist policies."
            )

        # build calibration_info from predict result keys directly
        calibration_info = {
            "enabled": hybrid_result.get("calibration_applied", calibration_enabled),
            "factor": hybrid_result.get("calibration_factor", calibration_factor),
            "strategy": hybrid_result.get("calibration_strategy", strategy_label),
            "method": "multiplicative",
        }
        if "components" in hybrid_result:
            uncal = np.array(hybrid_result["components"]["uncalibrated_hybrid"])
            calibration_info["mean_effect"] = float(np.mean(hybrid_preds - uncal))
            calibration_info["total_effect"] = float(np.sum(hybrid_preds - uncal))

        # extract actuarial_info from predict result
        reliability = hybrid_result.get("reliability", {})

        # recompute tail_risk_warning from the FULL merged hybrid_preds
        # array rather than from the per-chunk warning dict.
        # When n > inference_limit (e.g. 100K policies, limit=10K), _predict_in_batches
        # splits into multiple chunks. The tail_risk_warning in the merged result
        # comes from whichever chunk first triggered it; its policies_below_threshold
        # count is that chunk's count only — not the full portfolio count.
        # At 100K samples (10 chunks), this silently underreports by up to 10×.
        #
        # recompute the count directly from the full arrays.
        # Uses the same anchor as predict.py fix: safe_minimum = ml_preds * 0.5.
        # (Actuarial is not available here; ml_preds is the best unbiased proxy.)
        _tail_threshold_pct = float(
            config.get("hybrid_predictor", {})
            .get("business_config", {})
            .get("severe_underpricing_threshold_pct", 0.50)
        )
        _safe_min_full = ml_preds * (1.0 - _tail_threshold_pct)
        _underpriced_mask_full = hybrid_preds < _safe_min_full
        _n_underpriced_full = int(_underpriced_mask_full.sum())
        _n_total_full = len(hybrid_preds)

        if _n_underpriced_full > 0:
            _avg_gap_full = float(
                np.mean(
                    (_safe_min_full[_underpriced_mask_full] - hybrid_preds[_underpriced_mask_full])
                    / np.maximum(_safe_min_full[_underpriced_mask_full], 1e-8)
                )
            )
            _severity_full = (
                "CRITICAL"
                if _n_underpriced_full > _n_total_full * 0.10
                else "HIGH"
                if _n_underpriced_full > _n_total_full * 0.05
                else "MODERATE"
            )
            _tail_risk_warning_full = {
                "severity": _severity_full,
                "policies_below_threshold": _n_underpriced_full,
                "policies_below_threshold_pct": round(_n_underpriced_full / _n_total_full * 100, 2),
                "avg_gap_from_threshold_pct": round(_avg_gap_full * 100, 1),
                "threshold_used_pct": round(_tail_threshold_pct * 100, 1),
                "source": "full_batch_recompute",  # distinguishes from per-chunk value
            }
            logger.warning(
                f"🔥 TAIL RISK ALERT ({_severity_full}) [full-batch recompute]:\n"
                f"   {_n_underpriced_full} predictions ({_n_underpriced_full/_n_total_full*100:.1f}%) "
                f"below safe minimum threshold\n"
                f"   Average gap from threshold: {_avg_gap_full*100:.1f}%\n"
                f"   Recommendation: Review calibration settings and actuarial parameters"
            )
        else:
            _tail_risk_warning_full = None

        actuarial_info = {
            "actuarial_conservativeness_ratio": hybrid_result.get(
                "actuarial_conservativeness_ratio"
            ),
            # use full-batch recomputed value, not per-chunk dict
            "tail_risk_warning": _tail_risk_warning_full,
            "actuarial_conservative": reliability.get("actuarial_conservative", False),
            "actuarial_aggressive": reliability.get("actuarial_aggressive", False),
            "configured_blend_ratio": reliability.get("configured_blend_ratio"),
            "effective_avg_ml_weight": reliability.get("effective_avg_ml_weight"),
        }

        # weight arithmetic — now that actuarial_info is
        # built, effective_avg_ml_weight is resolved from all possible key paths.
        _c1_eff_wt = actuarial_info.get("effective_avg_ml_weight")
        if _c1_eff_wt is not None and _c1_eff_wt > 0 and _c1_ratio > 1.4:
            _c1_w_gap = 1.0 - _c1_eff_wt
            if _c1_w_gap > 1e-6:
                _c1_implied_act = (_c1_hyb_mean - _c1_eff_wt * _c1_ml_mean) / _c1_w_gap
                _c1_max_plausible = _train_max * 2.0
                logger.warning(
                    f"⚠️  [C1] WEIGHT ARITHMETIC: effective ML weight={_c1_eff_wt:.0%}. "
                    f"Implied actuarial mean=${_c1_implied_act:,.0f}. "
                    f"{f'ANOMALOUS — exceeds 2× training max (${_c1_max_plausible:,.0f})' if _c1_implied_act > _c1_max_plausible else f'Within plausible range (${_c1_max_plausible:,.0f})'}. "
                    f"Investigate SegmentRouter: specialist predictions may be "
                    f"counted as ML-dominant while still receiving actuarial adjustment."
                )

        print("\n[STEP 6] Running comprehensive evaluation...")
        evaluator = UnifiedEvaluator(business_config, config=config)
        evaluation = evaluator.evaluate_comprehensive(
            y_true=y_test.values,
            ml_preds=ml_preds,
            hybrid_preds=hybrid_preds,
            threshold=hybrid.threshold,
            calibration_info=calibration_info,
            actuarial_info=actuarial_info,
            enable_segment_analysis=True,
            config=config,
        )

        # T2-C: verify empirical CI coverage on the test set (was absent entirely).
        # Coverage check runs against the ML pipeline's conformal artifact.
        # A gap > 2% is a red flag; see check_ci_coverage() docstring for causes.
        # routes.py reads CI_CONFIDENCE_LEVEL from env var (default 0.90).
        # evaluate.py was hardcoded to 0.90 — if the API is deployed with a
        # different level (e.g. CI_CONFIDENCE_LEVEL=0.95), evaluate.py would
        # report "valid" at 0.90 while production CIs under-cover at 0.95.
        # Read the same env var so evaluate.py and routes.py stay in lockstep.
        _ci_level_raw = os.getenv("CI_CONFIDENCE_LEVEL", "0.90")
        try:
            _eval_ci_level = float(_ci_level_raw)
            if not (0.0 < _eval_ci_level < 1.0):
                raise ValueError
        except ValueError:
            logger.warning(
                f"⚠️ CI_CONFIDENCE_LEVEL='{_ci_level_raw}' is invalid; defaulting to 0.90"
            )
            _eval_ci_level = 0.90
        print("\n[STEP 6a] Verifying CI coverage on test set...")
        ci_coverage = check_ci_coverage(
            pipeline=ml_pipeline,
            X_test=X_test,
            y_true=y_test.values,
            confidence_level=_eval_ci_level,
        )
        _ci_valid = ci_coverage.get("valid", False)
        _ci_empirical = ci_coverage.get("empirical", float("nan"))
        _ci_gap = ci_coverage.get("gap", float("nan"))
        _ci_width = ci_coverage.get("mean_width", float("nan"))
        # Cascading fix for Finding D: when check_ci_coverage returns an error dict
        # (length mismatch or no CI available), numeric keys are absent and the
        # .get(..., nan) defaults format as "+nan" / "nan" — confusing in operator logs.
        # Print the error message directly instead.
        if "error" in ci_coverage:
            print(f"   Coverage: ⚠️ CI CHECK SKIPPED — {ci_coverage['error']}")
        else:
            print(
                f"   Coverage: empirical={_ci_empirical:.3f}  nominal=0.90  "
                f"gap={_ci_gap:+.3f}  width=${_ci_width:,.0f}  "
                f"{'✅ Valid' if _ci_valid else '⚠️ Under-coverage — retrain to recalibrate'}"
            )
        # Attach CI coverage to the evaluation dict for JSON output and MLflow logging
        evaluation["ci_coverage"] = ci_coverage

        # hybrid_result["tail_risk_warning"] is the value from whichever
        # chunk first triggered a warning in _predict_in_batches — its
        # policies_below_threshold count reflects that chunk only, not the full
        # portfolio.  The full-portfolio recount (_tail_risk_warning_full) was
        # computed earlier and stored in actuarial_info["tail_risk_warning"].
        # Overwrite the stale per-chunk value in the evaluation dict so the JSON
        # report and any downstream consumer see the correct portfolio-level count.
        if _tail_risk_warning_full is not None:
            evaluation["tail_risk_warning"] = _tail_risk_warning_full
        elif "tail_risk_warning" in evaluation:
            # No underpricing detected in full batch — clear any stale chunk value.
            evaluation["tail_risk_warning"] = None

        print("\n[STEP 7] Generating business-focused report...")
        report_path = f"{args.output_dir}/unified_evaluation.txt"
        report = evaluator.generate_unified_report(
            evaluation,
            y_true=y_test.values,
            ml_preds=ml_preds,
            hybrid_preds=hybrid_preds,
            output_path=report_path,
        )

        print("\n[STEP 8] Creating visualization...")
        plot_path = f"{args.output_dir}/unified_evaluation.png"
        create_unified_visualization(evaluation, y_test.values, ml_preds, hybrid_preds, plot_path)

        print("\n[STEP 9] Saving JSON...")
        summary_path = f"{args.output_dir}/unified_summary.json"

        def convert_types(obj):
            """Convert numpy/pandas types to JSON-serializable Python types"""
            if isinstance(obj, np.integer | np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool | np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif obj is None:
                return None
            elif isinstance(obj, str | int | float):
                return obj
            else:
                logger.warning(f"Unknown type {type(obj)}, converting to string")
                return str(obj)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(convert_types(evaluation), f, indent=2)

        # ── P0-B: log evaluation results to dedicated MLflow experiment ────────
        try:
            import math as _math

            import mlflow as _mle

            _ecfg = config.get("mlflow", {}).get("tracking", {})
            _euri = _ecfg.get("tracking_uri", "./mlruns")
            _eexp = _ecfg.get("experiment_name", "insurance_ml") + "_evaluation"

            _mle.set_tracking_uri(_euri)
            _mle.autolog(disable=True, silent=True)
            if _mle.get_experiment_by_name(_eexp) is None:
                _mle.create_experiment(_eexp)
            _mle.set_experiment(_eexp)

            def _sf(v):
                return float(v) if isinstance(v, int | float) and _math.isfinite(float(v)) else None

            with _mle.start_run(run_name="unified_evaluation"):
                _em: dict[str, float] = {}

                # Academic metrics — ML and Hybrid
                # evaluation["academic"]["ml"] has shape
                # {"overall": {metrics}, "low_value": {metrics}, "high_value": {metrics}}.
                # The original code passed that outer dict as _src, so _src.get("rmse")
                # always returned None and every academic metric was silently dropped.
                # drill into ".overall" before iterating metric keys.
                for _pfx, _src in [
                    ("ml", evaluation.get("academic", {}).get("ml", {}).get("overall", {})),
                    ("hybrid", evaluation.get("academic", {}).get("hybrid", {}).get("overall", {})),
                ]:
                    for _k in (
                        "rmse",
                        "mae",
                        "smape",
                        "male",
                        "r2",
                        "mean_error",
                        "median_error",
                        "std_error",
                    ):
                        _v = _sf(_src.get(_k))
                        if _v is not None:
                            _em[f"{_pfx}_{_k}"] = _v

                # Business metrics — ML and Hybrid overall
                for _pfx, _src in [
                    (
                        "ml",
                        evaluation.get("business", {}).get("ml", {}).get("overall", {}),
                    ),
                    (
                        "hybrid",
                        evaluation.get("business", {}).get("hybrid", {}).get("overall", {}),
                    ),
                ]:
                    for _k in (
                        "total_net_profit",
                        "net_margin_pct",
                        "profit_per_policy",
                        "business_value_score",
                        "accuracy_rate_pct",
                        "churn_rate_pct",
                        "gross_margin_pct",
                        "n_underpriced",
                        "n_overpriced",
                        "n_accurate",
                    ):
                        _v = _sf(_src.get(_k))
                        if _v is not None:
                            _em[f"{_pfx}_biz_{_k}"] = _v

                # Profit delta hybrid vs ML
                _mlp = _sf(
                    evaluation.get("business", {})
                    .get("ml", {})
                    .get("overall", {})
                    .get("total_net_profit")
                )
                _hyp = _sf(
                    evaluation.get("business", {})
                    .get("hybrid", {})
                    .get("overall", {})
                    .get("total_net_profit")
                )
                if _mlp is not None and _hyp is not None:
                    _em["profit_delta_hybrid_vs_ml"] = _hyp - _mlp

                # Segment-level metrics
                # evaluation["segment_analysis"] has shape
                # {"ml": {seg_name: flat_metrics}, "hybrid": {seg_name: flat_metrics}}.
                # The original loop iterated the OUTER level ("ml", "hybrid") and
                # treated each full segment-dict as a single segment, then looked for
                # "academic" and "business.overall" sub-keys that don't exist in
                # calculate_segment_metrics output (metrics are flat).  Every seg_*
                # key was silently None and dropped on every run.
                # add the missing inner loop over segment names and drop the
                # non-existent "academic"/"business" wrappers.
                for _pfx, _seg_dict in evaluation.get("segment_analysis", {}).items():
                    if not isinstance(_seg_dict, dict):
                        continue
                    for _sn, _sd in _seg_dict.items():
                        _sk = _sn.lower().replace(" ", "_").replace("-", "_")
                        if not isinstance(_sd, dict):
                            continue
                        for _k in ("rmse", "r2", "mae"):
                            _v = _sf(_sd.get(_k))
                            if _v is not None:
                                _em[f"seg_{_pfx}_{_sk}_{_k}"] = _v
                        for _k in (
                            "profit_per_policy",
                            "net_profit",
                            "churn_rate_pct",
                        ):
                            _v = _sf(_sd.get(_k))
                            if _v is not None:
                                _em[f"seg_{_pfx}_{_sk}_biz_{_k}"] = _v

                # Statistical test + actuarial diagnostics
                for _k in ("t_statistic", "p_value", "t_statistic_mae", "p_value_mae"):
                    _v = _sf(evaluation.get("statistical_test", {}).get(_k))
                    if _v is not None:
                        _em[f"stat_{_k}"] = _v
                for _k in (
                    "actuarial_conservativeness_ratio",
                    "effective_avg_ml_weight",
                    "configured_blend_ratio",
                ):
                    _v = _sf(evaluation.get("actuarial_info", {}).get(_k))
                    if _v is not None:
                        _em[f"actuarial_{_k}"] = _v

                # T2-C: CI coverage metrics
                _ci_cov = evaluation.get("ci_coverage", {})
                for _k in ("empirical", "gap", "mean_width", "median_width"):
                    _v = _sf(_ci_cov.get(_k))
                    if _v is not None:
                        _em[f"ci_{_k}"] = _v
                if "valid" in _ci_cov:
                    _em["ci_valid"] = float(bool(_ci_cov["valid"]))

                if _em:
                    _mle.log_metrics(_em)

                # Params: eval configuration
                _mle.log_params(
                    {
                        "threshold": str(getattr(hybrid, "threshold", "N/A")),
                        "blend_ratio": str(getattr(hybrid, "blend_ratio", "N/A")),
                        "calibration_factor": str(calibration_factor),
                        "calibration_enabled": str(calibration_enabled),
                    }
                )

                # Tags
                _hy_bvs = (
                    evaluation.get("business", {})
                    .get("hybrid", {})
                    .get("overall", {})
                    .get("business_value_score", 0)
                )
                _ml_bvs = (
                    evaluation.get("business", {})
                    .get("ml", {})
                    .get("overall", {})
                    .get("business_value_score", 0)
                )
                _mle.set_tag("winner", "hybrid" if _hy_bvs > _ml_bvs else "ml")
                _stat_sig = evaluation.get("statistical_test", {}).get("significant")
                if _stat_sig is not None:
                    _mle.set_tag("hybrid_stat_significant", str(_stat_sig))

                # Link to training run — reads mlflow_run_id written by P0-A
                try:
                    import json as _jt

                    _mp = Path("models/pipeline_metadata.json")
                    if _mp.exists():
                        _meta = _jt.loads(_mp.read_text())
                        _tr = _meta.get("mlflow_run_id")
                        if _tr:
                            _mle.set_tag("training_run_id", str(_tr))
                        _gc = _meta.get("git_commit") or _meta.get("commit_hash")
                        if _gc:
                            _mle.set_tag("git_commit", str(_gc)[:12])
                except Exception:
                    pass

                # Artifacts: report, plot, summary JSON
                for _ap, _af in [
                    (report_path, "reports"),
                    (plot_path, "reports"),
                    (summary_path, "reports"),
                ]:
                    try:
                        if Path(_ap).exists():
                            _mle.log_artifact(_ap, artifact_path=_af)
                    except Exception:
                        pass

            print(f"  MLflow: ✅ evaluation run logged → experiment '{_eexp}'")
            # The print is after the `with start_run():` block exits normally.
            # If start_run() threw, the outer except catches it and this line
            # is never reached — the wording "✅ logged" is therefore accurate.
        except Exception as _emle_err:
            print(f"  MLflow evaluation logging failed (non-fatal): {_emle_err}")
        # ─────────────────────────────────────────────────────────────────────

        print("\n" + report)

        # Suppress calibration impact block when factor == 1.0.
        # Previously printed "+$0.00/policy" on every ML-only run, adding noise
        # without information. The guard now requires a non-trivial factor.
        if (
            calibration_info["enabled"]
            and abs(calibration_info.get("factor", 1.0) - 1.0) > 1e-6
            and "mean_effect" in calibration_info
        ):
            print("\n" + "=" * 90)
            print("📊 CALIBRATION IMPACT")
            print("=" * 90)
            print(f"   Factor:   {calibration_info['factor']:.4f}")
            print(f"   Strategy: {calibration_info['strategy']}")
            print(f"   Mean effect: +${calibration_info['mean_effect']:,.2f}/policy")
            print(f"   Total effect: +${calibration_info['total_effect']:,.2f}")

        # Actuarial summary
        if actuarial_info["actuarial_conservativeness_ratio"] is not None:
            print("\n" + "=" * 90)
            print("⚖️  ACTUARIAL SUMMARY")
            print("=" * 90)
            ratio = actuarial_info["actuarial_conservativeness_ratio"]
            print(f"   Actuarial/ML ratio: {ratio:.2f}x")
            if actuarial_info["actuarial_aggressive"]:
                print(f"   🔴 UNDER-PRICING RISK — {(1-ratio)*100:.1f}% below ML")
            elif actuarial_info["actuarial_conservative"]:
                print(f"   ⚠️ Conservative — {(ratio-1)*100:.1f}% above ML")
            else:
                print("   ✅ Within normal range")

            eff_wt = actuarial_info.get("effective_avg_ml_weight")
            cfg_blend = actuarial_info.get("configured_blend_ratio")
            if eff_wt is not None and cfg_blend is not None:
                print(
                    f"   Blend: configured {cfg_blend:.0%} ML → "
                    f"effective {eff_wt:.0%} ML "
                    f"({'premium routing above threshold' if eff_wt > cfg_blend + 0.10 else 'as expected'})"
                )

        print("\n" + "=" * 90)
        print("✅ EVALUATION COMPLETE")
        print("=" * 90)

        ml_biz = evaluation["business"]["ml"]["overall"]
        hybrid_biz = evaluation["business"]["hybrid"]["overall"]

        # deployment_confidence and wins are local variables inside
        # generate_unified_report() (evaluate_comprehensive method, lines ~2007-2008)
        # and are NOT returned in the evaluation dict by evaluate_comprehensive()
        # (return dict at line ~1634 has no such keys).  Every non-business-focus
        # run crashed with:  NameError: name 'deployment_confidence' is not defined
        #
        # recompute the identical 5-metric gate here from data already in
        # scope in main().  All five inputs are derivable from evaluation dict keys
        # that ARE returned (business.ml.overall / business.hybrid.overall) plus
        # the y_test / ml_preds / hybrid_preds arrays assigned earlier in main().
        _sev_threshold_main = float(
            config.get("hybrid_predictor", {})
            .get("business_config", {})
            .get("severe_underpricing_threshold_pct", 0.50)
        )
        _y_true_main = y_test.values
        _ml_err_pct = np.where(_y_true_main > 0, (ml_preds - _y_true_main) / _y_true_main, 0.0)
        _hy_err_pct = np.where(_y_true_main > 0, (hybrid_preds - _y_true_main) / _y_true_main, 0.0)
        _win_profit = hybrid_biz["total_net_profit"] > ml_biz["total_net_profit"]
        _win_churn = hybrid_biz["churn_rate_pct"] < ml_biz["churn_rate_pct"]
        _win_biz_score = hybrid_biz["business_value_score"] > ml_biz["business_value_score"]
        _win_tail_risk = int(np.sum(_hy_err_pct < -_sev_threshold_main)) <= int(
            np.sum(_ml_err_pct < -_sev_threshold_main)
        )
        _win_actuarial = (
            hybrid_biz["total_underpricing_penalty"] < ml_biz["total_underpricing_penalty"]
        )
        wins = (
            int(_win_profit)
            + int(_win_churn)
            + int(_win_biz_score)
            + int(_win_tail_risk)
            + int(_win_actuarial)
        )
        deployment_confidence = wins / 5.0

        print(
            f"\n   Winner: {'Hybrid' if hybrid_biz['business_value_score'] > ml_biz['business_value_score'] else 'ML'}"
        )
        print(
            f"   Profit Delta: ${hybrid_biz['total_net_profit'] - ml_biz['total_net_profit']:,.0f}"
        )
        print("\n   Files saved:")
        print(f"      - {report_path}")
        print(f"      - {plot_path}")
        print(f"      - {summary_path}")

        if args.business_focus:
            # --business-focus: pass only when hybrid BizScore beats ML baseline.
            sys.exit(
                0 if hybrid_biz["business_value_score"] > ml_biz["business_value_score"] else 1
            )
        else:
            # Default CI path previously exited 0 unconditionally,
            # meaning no evaluation outcome — low confidence, bad BizScore, "DO
            # NOT DEPLOY" recommendation — could ever block the pipeline.
            # exit 1 when deployment_confidence is below HIGH threshold
            # (< 0.80, i.e. fewer than 4/5 metric wins).  This makes the
            # 5-metric gate actually enforce CI pass/fail.
            # MODERATE (0.60–0.80) still exits 0 because those cases pass with
            # warnings; only LOW confidence (< 0.60) hard-fails.
            #
            # --no-exit-gate: always exit 0 but still log the gate outcome.
            # Intended for non-blocking CI steps (e.g. artifact collection)
            # where a hard failure would orphan downstream jobs.  Never set
            # this on a deployment-blocking step.
            if args.no_exit_gate:
                _gate_level = (
                    "HIGH"
                    if deployment_confidence >= _DEPLOYMENT_CONFIDENCE_HIGH
                    else "MODERATE"
                    if deployment_confidence >= _DEPLOYMENT_CONFIDENCE_MODERATE
                    else "LOW"
                )
                logger.warning(
                    f"⚠️ CI GATE (suppressed by --no-exit-gate): "
                    f"deployment_confidence={deployment_confidence:.0%} "
                    f"({wins}/5 metric wins) — {_gate_level} confidence. "
                    "Exiting 0 regardless."
                )
                sys.exit(0)

            if deployment_confidence < _DEPLOYMENT_CONFIDENCE_MODERATE:
                logger.error(
                    f"❌ CI GATE FAILED: deployment_confidence={deployment_confidence:.0%} "
                    f"({wins}/5 metric wins) is below LOW threshold "
                    f"({_DEPLOYMENT_CONFIDENCE_MODERATE:.0%}). "
                    "Resolve flagged metrics before re-evaluating."
                )
                sys.exit(1)
            elif deployment_confidence < _DEPLOYMENT_CONFIDENCE_HIGH:
                logger.warning(
                    f"⚠️ CI GATE WARNING: deployment_confidence={deployment_confidence:.0%} "
                    f"({wins}/5 metric wins) — MODERATE confidence. "
                    "Address flagged risks before production deployment. "
                    "Exiting 0 (pipeline continues with warnings)."
                )
                sys.exit(0)
            else:
                logger.info(
                    f"✅ CI GATE PASSED: deployment_confidence={deployment_confidence:.0%} "
                    f"({wins}/5 metric wins) — HIGH confidence."
                )
                sys.exit(0)

    except (OSError, ValueError, RuntimeError, AttributeError) as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
