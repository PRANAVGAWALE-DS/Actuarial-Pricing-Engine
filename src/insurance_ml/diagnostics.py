# src/insurance_ml/diagnostics.py

"""
Advanced model diagnostics and analysis.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelDiagnostics:
    """Advanced diagnostics for trained models."""

    @staticmethod
    def get_feature_importance(model, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
        """
        Extract and rank feature importance.

        Supports:
        - Tree-based models (XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting): uses feature_importances_
        - Linear models (LinearRegression, Ridge, Lasso, QuantileRegressor): uses abs(coefficients) as importance

        Args:
            model: Trained model with feature_importances_ or coef_ attribute
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with features and importance scores
        """
        # Handle wrapped models (e.g., CalibratedModel)
        if hasattr(model, "base_model"):
            logger.debug("Unwrapping CalibratedModel to access base model")
            base_model = model.base_model
        else:
            base_model = model

        importance = None
        model_type = type(base_model).__name__

        # Try tree-based feature importances first
        if hasattr(base_model, "feature_importances_"):
            importance = base_model.feature_importances_
            importance_source = "tree-based feature_importances_"

        # Try linear model coefficients
        elif hasattr(base_model, "coef_"):
            # For linear models, use absolute value of coefficients as importance
            coef = base_model.coef_

            # Handle multidimensional coefficients (shouldn't happen for regression, but be safe)
            if coef.ndim > 1:
                importance = np.abs(coef).mean(axis=0)
            else:
                importance = np.abs(coef)

            importance_source = "linear model coefficients"
        else:
            logger.warning(
                f"⚠️  Model type '{model_type}' does not have feature importance\n"
                f"   Skipping feature importance analysis.\n"
                f"   Supported models:\n"
                f"     • Tree-based: XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting\n"
                f"     • Linear: LinearRegression, Ridge, Lasso, QuantileRegressor"
            )
            return pd.DataFrame(columns=["feature", "importance"])

        # Normalize importance scores to [0, 1] range for interpretability
        importance_normalized = importance / importance.sum()

        df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_normalized}
        ).sort_values("importance", ascending=False)

        logger.info(f"\n📊 TOP {top_n} FEATURES ({importance_source}):")
        for _idx, row in df.head(top_n).iterrows():
            logger.info(f"   {row['feature']:<35} {row['importance']:.6f}")

        return df.head(top_n)

    @staticmethod
    def analyze_prediction_distribution(
        y_true: np.ndarray, y_pred: np.ndarray, bins: list[float] | None = None
    ) -> dict:
        """
        Analyze how predictions are distributed.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            bins: Custom bin edges (default: [0, 5000, 10000, 20000, inf])

        Returns:
            Dictionary with distribution statistics
        """
        if bins is None:
            bins = [0, 5000, 10000, 20000, np.inf]

        pred_bins = pd.cut(y_pred, bins=bins)

        logger.info("\n📊 PREDICTION DISTRIBUTION:")
        dist = pred_bins.value_counts().sort_index()
        for interval, count in dist.items():
            pct = count / len(y_pred) * 100
            logger.info(f"   {interval}: {count} samples ({pct:.1f}%)")

        # Residual analysis
        residuals = y_true - y_pred

        logger.info("\n📊 RESIDUAL ANALYSIS:")
        logger.info(f"   Mean Residual: ${np.mean(residuals):.2f}")
        logger.info(f"   Median Residual: ${np.median(residuals):.2f}")
        logger.info(f"   Std Residual: ${np.std(residuals):.2f}")
        logger.info(f"   95th Percentile |Error|: ${np.percentile(np.abs(residuals), 95):.2f}")

        return {
            "distribution": dist.to_dict(),
            "mean_residual": float(np.mean(residuals)),
            "median_residual": float(np.median(residuals)),
            "p95_error": float(np.percentile(np.abs(residuals), 95)),
        }

    @staticmethod
    def calculate_calibration(
        y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Check if predicted values match actual values across ranges.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            n_bins: Number of bins for calibration

        Returns:
            DataFrame with calibration statistics per bin
        """
        pred_bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
        bin_indices = np.digitize(y_pred, pred_bins)

        calibration_data = []

        logger.info("\n📊 CALIBRATION CHECK:")
        logger.info(
            f"   {'Bin':<5} {'Samples':<10} {'Pred Mean':<12} {'Actual Mean':<12} {'Deviation':<12}"
        )
        logger.info(f"   {'-'*60}")

        for i in range(1, len(pred_bins)):
            mask = bin_indices == i
            if mask.sum() > 5:  # Only consider bins with enough samples
                avg_pred = y_pred[mask].mean()
                avg_actual = y_true[mask].mean()
                deviation = avg_pred - avg_actual

                calibration_data.append(
                    {
                        "bin": i,
                        "n_samples": mask.sum(),
                        "pred_mean": avg_pred,
                        "actual_mean": avg_actual,
                        "deviation": deviation,
                    }
                )

                logger.info(
                    f"   {i:<5} {mask.sum():<10} ${avg_pred:<11.0f} ${avg_actual:<11.0f} ${deviation:<11.0f}"
                )

        df = pd.DataFrame(calibration_data)

        if len(df) > 0:
            logger.info(f"\n   Max |Deviation|: ${df['deviation'].abs().max():.0f}")
            logger.info(f"   Mean Deviation: ${df['deviation'].mean():.0f}")

        return df

    @staticmethod
    def calculate_business_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, thresholds: list[float] | None = None
    ) -> dict:
        """
        Calculate business-relevant metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            thresholds: Percentage error thresholds to evaluate

        Returns:
            Dictionary with business metrics
        """
        if thresholds is None:
            thresholds = [0.05, 0.1, 0.15, 0.2]
        logger.info("\n💼 BUSINESS METRICS:")

        metrics = {}

        # Percentage error thresholds
        pct_errors = np.abs((y_pred - y_true) / y_true)

        for threshold in thresholds:
            within_threshold = (pct_errors <= threshold).mean()
            metrics[f"within_{int(threshold*100)}pct"] = float(within_threshold)
            logger.info(f"   Predictions within ±{int(threshold*100)}%: {within_threshold:.2%}")

        # Absolute errors
        abs_errors = np.abs(y_pred - y_true)
        metrics["mean_abs_error"] = float(abs_errors.mean())
        metrics["median_abs_error"] = float(np.median(abs_errors))
        metrics["max_abs_error"] = float(abs_errors.max())

        logger.info(f"\n   Average Absolute Error: ${abs_errors.mean():.0f}")
        logger.info(f"   Median Absolute Error: ${np.median(abs_errors):.0f}")
        logger.info(f"   Max Absolute Error: ${abs_errors.max():.0f}")

        # Directional bias
        overpriced = (y_pred > y_true).sum()
        underpriced = (y_pred < y_true).sum()

        metrics["overpriced_pct"] = float(overpriced / len(y_true))
        metrics["underpriced_pct"] = float(underpriced / len(y_true))

        logger.info(f"\n   Overpriced: {overpriced}/{len(y_true)} ({overpriced/len(y_true):.1%})")
        logger.info(f"   Underpriced: {underpriced}/{len(y_true)} ({underpriced/len(y_true):.1%})")

        return metrics

    @staticmethod
    def error_by_range(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: list[float] | None = None,
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate error metrics for different prediction ranges.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            bins: Custom bin edges
            labels: Labels for bins

        Returns:
            DataFrame with error metrics per range
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        if bins is None:
            bins = [0, 5000, 10000, 16701, np.inf]
        if labels is None:
            labels = ["Low", "Mid", "High", "Very High"]

        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        df["range"] = pd.cut(y_pred, bins=bins, labels=labels)

        results = []

        logger.info("\n📊 ERROR BY PREDICTION RANGE:")
        logger.info(f"   {'Range':<12} {'Samples':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
        logger.info(f"   {'-'*60}")

        for label in labels:
            mask = df["range"] == label
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(df.loc[mask, "y_true"], df.loc[mask, "y_pred"]))
                mae = mean_absolute_error(df.loc[mask, "y_true"], df.loc[mask, "y_pred"])
                mape = np.mean(
                    np.abs(
                        (df.loc[mask, "y_true"] - df.loc[mask, "y_pred"]) / df.loc[mask, "y_true"]
                    )
                )

                results.append(
                    {
                        "range": label,
                        "n_samples": mask.sum(),
                        "rmse": rmse,
                        "mae": mae,
                        "mape": mape,
                    }
                )

                logger.info(
                    f"   {label:<12} {mask.sum():<10} ${rmse:<9.0f} ${mae:<9.0f} {mape:<9.2%}"
                )

        return pd.DataFrame(results)

    @staticmethod
    def show_sample_predictions(
        y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 5, seed: int = 42
    ) -> pd.DataFrame:
        """
        Show random sample predictions for sanity check.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            n_samples: Number of samples to show
            seed: Random seed

        Returns:
            DataFrame with sample predictions
        """
        np.random.seed(seed)
        indices = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)

        samples = []
        logger.info("\n🔍 SAMPLE PREDICTIONS:")
        logger.info(
            f"   {'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}"
        )
        logger.info(f"   {'-'*60}")

        for idx in indices:
            actual = y_true[idx]
            predicted = y_pred[idx]
            error = actual - predicted
            error_pct = error / actual

            samples.append(
                {
                    "index": idx,
                    "actual": actual,
                    "predicted": predicted,
                    "error": error,
                    "error_pct": error_pct,
                }
            )

            logger.info(
                f"   {idx:<8} ${actual:<11.0f} ${predicted:<11.0f} ${error:<11.0f} {error_pct:<9.1%}"
            )

        return pd.DataFrame(samples)


# ============================================================================
# INLINE PATCH 05 (Gate G6) — CostWeightedMetrics + DeploymentGates
# ============================================================================

HIGH_VALUE_THRESHOLD: float = 16_701.0
G6_MIN_COST_WEIGHTED_R2: float = 0.75
_TIER_WEIGHTS: dict = {"low": 1.0, "mid": 1.5, "high": 2.5, "very_high": 4.0}
# Segment bins intentionally use 4 tiers — the v7.5.2 attempt to add a 5th
# "High+" tier ($14K–$16.7K) created a 12-sample test bucket that triggered
# a G6 hard veto on R²=-28.7 due to insufficient data in that narrow zone,
# not genuine model failure. 4 bins ensure each segment is statistically
# stable on a 268-sample test set (target ≥ 30 samples per segment).


class CostWeightedMetrics:
    """Business-weighted accuracy metrics that reflect revenue impact."""

    @staticmethod
    def tier_weights(
        y_true: np.ndarray, bins: list[float] | None = None, weights: dict | None = None
    ) -> np.ndarray:
        if bins is None:
            bins = [0.0, 5_000.0, 10_000.0, HIGH_VALUE_THRESHOLD, np.inf]  # 4-tier
        if weights is None:
            weights = _TIER_WEIGHTS
        labels = list(weights.keys())
        tier_cats = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
        return np.array([weights[str(cat)] for cat in tier_cats])

    @staticmethod
    def cost_weighted_r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: list[float] | None = None,
        tier_weights: dict | None = None,
    ) -> float:
        """Weighted R² with revenue-proportional tier weights."""
        w = CostWeightedMetrics.tier_weights(y_true, bins, tier_weights)
        y_mean_w = np.average(y_true, weights=w)
        ss_tot_w = np.sum(w * (y_true - y_mean_w) ** 2)
        ss_res_w = np.sum(w * (y_true - y_pred) ** 2)
        if ss_tot_w < 1e-10:
            logger.warning("cost_weighted_r2: SS_tot_w near zero")
            return float("nan")
        return float(1.0 - ss_res_w / ss_tot_w)

    @staticmethod
    def segment_r2_breakdown(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: list[float] | None = None,
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """Per-segment R², RMSE, MAE, and overpricing rate."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        if bins is None:
            bins = [0.0, 5_000.0, 10_000.0, HIGH_VALUE_THRESHOLD, np.inf]  # 4-tier
        if labels is None:
            labels = ["Low", "Mid", "High", "Very High"]
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
            yt, yp = y_true[mask], y_pred[mask]
            rows.append(
                {
                    "segment": label,
                    "n_samples": int(n),
                    "r2": float(r2_score(yt, yp)),
                    "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "overpricing_rate": float((yp > yt).mean()),
                }
            )
        return pd.DataFrame(rows)


class DeploymentGates:
    """Deployment gate checks driven by business-weighted metrics."""

    @staticmethod
    def check_g6(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        min_cost_weighted_r2: float = G6_MIN_COST_WEIGHTED_R2,
        raise_on_fail: bool = False,
    ) -> dict:
        """Gate G6: cost-weighted R² >= 0.75 (high-value policies weighted 4x).

        Hard veto: if any segment has R² < -1.0 the gate FAILS regardless of
        the aggregate cost_weighted_r2.  R² < -1.0 means the model's squared
        error is more than 2× the variance of the target — catastrophically
        worse than predicting the segment mean — and must never pass deployment.
        """
        cw_r2 = CostWeightedMetrics.cost_weighted_r2(y_true, y_pred)
        breakdown = CostWeightedMetrics.segment_r2_breakdown(y_true, y_pred)
        g6_pass = cw_r2 >= min_cost_weighted_r2

        critical = breakdown[breakdown["r2"] < 0.0]["segment"].tolist()
        warnings = breakdown[(breakdown["r2"] >= 0.0) & (breakdown["r2"] < 0.5)]["segment"].tolist()

        # hard veto — R² < -1.0 blocks deployment unconditionally ──
        # An aggregate G6 pass can mask a catastrophic segment (e.g. High R²=-2.016
        # when Very High R²=0.55 compensates in the weighted average). We veto the
        # gate if any individual segment R² drops below -1.0, i.e. the model's
        # prediction error variance exceeds twice the target variance for that band.
        _veto_segments = breakdown[breakdown["r2"] < -1.0]["segment"].tolist()
        if _veto_segments:
            g6_pass = False  # override aggregate pass
            logger.error(
                f"❌ G6 HARD VETO: R² < -1.0 in segment(s) {_veto_segments}. "
                f"Aggregate cost_weighted_r2={cw_r2:.4f} is irrelevant — "
                f"the model is more than 2× worse than the mean predictor for these "
                f"policyholders and must not be deployed. "
                f"Train a specialist model or increase sample_weight for this tier."
            )

        gate_str = "✅ PASS" if g6_pass else "❌ FAIL"
        logger.info(
            f"\nGate G6 [{gate_str}]  cost_weighted_r2={cw_r2:.4f}  threshold={min_cost_weighted_r2:.2f}"
        )
        logger.info(f"  {'Segment':<12} {'N':<7} {'R2':>7} {'RMSE':>10} {'Overpricing':>12}")
        for _, row in breakdown.iterrows():
            r2s = f"{row['r2']:7.4f}" if not pd.isna(row["r2"]) else "    N/A"
            logger.info(
                f"  {row['segment']:<12} {int(row['n_samples']):<7} {r2s} ${row['rmse']:>9,.0f} {row['overpricing_rate']:>11.1%}"
            )

        if critical:
            logger.error(
                f"❌ CRITICAL: R2 < 0 in segments {critical} — model worse than mean predictor"
            )
        if warnings:
            logger.warning(f"⚠️  Low R2 (< 0.5) in segments: {warnings}")

        result = {
            "g6_pass": g6_pass,
            "cost_weighted_r2": cw_r2,
            "min_threshold": min_cost_weighted_r2,
            "segment_breakdown": breakdown.to_dict(orient="records"),
            "critical_segments": critical,
            "warning_segments": warnings,
            "veto_segments": _veto_segments,
        }

        if not g6_pass and raise_on_fail:
            msg = (
                f"Gate G6 FAILED — hard veto: R² < -1.0 in {_veto_segments}"
                if _veto_segments
                else f"Gate G6 FAILED: cost_weighted_r2={cw_r2:.4f} < {min_cost_weighted_r2}"
            )
            raise ValueError(msg)
        return result
