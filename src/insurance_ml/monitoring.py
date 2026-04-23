# src/insurance_ml/monitoring.py

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-classes (typed, serialisable)
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Structured output of a single drift-detection run."""

    drifted_features: list[str] = field(default_factory=list)
    drift_scores: dict[str, float] = field(default_factory=dict)  # z-score or TVD per feature
    n_new: int = 0
    z_threshold: float = 2.0
    tvd_threshold: float = 0.05
    missing_features: list[str] = field(default_factory=list)  # features absent from X_new

    # Convenience -----------------------------------------------------------
    @property
    def has_drift(self) -> bool:
        return len(self.drifted_features) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "drifted_features": self.drifted_features,
            "drift_scores": self.drift_scores,
            "n_new": self.n_new,
            "z_threshold": self.z_threshold,
            "tvd_threshold": self.tvd_threshold,
            "missing_features": self.missing_features,
            "has_drift": self.has_drift,
        }

    def summary(self) -> str:
        lines = [
            f"DriftReport | n_new={self.n_new:,} | "
            f"drifted={len(self.drifted_features)} feature(s)"
        ]
        for feat in self.drifted_features:
            score = self.drift_scores.get(feat, float("nan"))
            lines.append(f"  - {feat}: score={score:.4f}")
        if self.missing_features:
            lines.append(f"  [!] Missing in new data: {self.missing_features}")
        return "\n".join(lines)


@dataclass
class PerformanceSnapshot:
    """One-shot model performance metrics for a labelled batch."""

    n_samples: int
    mae: float
    rmse: float
    mape: float | None  # None if any y_true == 0
    r2: float
    prediction_mean: float
    prediction_std: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _se_of_mean(baseline_std: float, n: int) -> float:
    """Standard error of the sample mean under H0 (same distribution)."""
    return baseline_std / (max(n, 1) ** 0.5)


def _z_score(new_mean: float, baseline_mean: float, se: float) -> float:
    return abs(new_mean - baseline_mean) / (se + 1e-10)


def _tvd(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Total Variation Distance between two empirical distributions."""
    all_cats = set(dist_a) | set(dist_b)
    return sum(abs(dist_a.get(c, 0.0) - dist_b.get(c, 0.0)) for c in all_cats) / 2.0


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------


class DriftMonitor:
    """
    Monitor input-feature drift against a frozen training baseline.

    Usage
    -----
        # Once, after training:
        DriftMonitor.create_baseline(X_train, y_train)

        # Every inference batch:
        report = DriftMonitor.detect_drift(X_batch)
        if report.has_drift:
            alert(report.summary())
    """

    # ------------------------------------------------------------------
    # Baseline creation
    # ------------------------------------------------------------------

    @staticmethod
    def create_baseline(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        output_path: str = "models/drift_baseline.json",
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Snapshot training-set statistics for future drift comparisons.

        Args:
            X_train:      Training features.
            y_train:      Training target (used for target-distribution stats).
            output_path:  Where to persist the baseline JSON.
            overwrite:    If False (default), raise if the file already exists
                          -- prevents silent baseline overwrites in production.

        Returns:
            The baseline dict (also written to output_path).
        """
        out = Path(output_path)
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Baseline already exists at '{output_path}'. " "Pass overwrite=True to replace it."
            )

        baseline: dict[str, Any] = {
            "n_samples": len(X_train),
            "target_stats": {
                "mean": float(y_train.mean()),
                "std": float(y_train.std()),
                "min": float(y_train.min()),
                "max": float(y_train.max()),
                "q25": float(y_train.quantile(0.25)),
                "q50": float(y_train.quantile(0.50)),
                "q75": float(y_train.quantile(0.75)),
            },
            "feature_stats": {},
        }

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            baseline["feature_stats"][col] = {
                "dtype": "numeric",
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std()),
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max()),
                "q25": float(X_train[col].quantile(0.25)),
                "q50": float(X_train[col].quantile(0.50)),
                "q75": float(X_train[col].quantile(0.75)),
            }

        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            dist = X_train[col].value_counts(normalize=True)
            baseline["feature_stats"][col] = {
                "dtype": "categorical",
                "categories": list(X_train[col].unique()),
                "distribution": {k: float(v) for k, v in dist.items()},
            }

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(baseline, fh, indent=2)

        logger.info(
            "\n📊 DRIFT BASELINE CREATED\n"
            f"   Samples  : {baseline['n_samples']:,}\n"
            f"   Features : {len(baseline['feature_stats'])}\n"
            f"   Saved to : {output_path}"
        )
        return baseline

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_drift(
        X_new: pd.DataFrame,
        baseline_path: str = "models/drift_baseline.json",
        z_threshold: float = 2.0,
        tvd_threshold: float = 0.05,
    ) -> DriftReport:
        """
        Detect input drift using sample-size-aware statistical tests.

        Numeric features
        ~~~~~~~~~~~~~~~~
        z = |mu_new - mu_baseline| / (sigma_baseline / sqrt(n_new))

        This is the z-score of the observed new-batch mean under H0 (identical
        distribution). Dividing by the standard error of the mean (not just
        sigma_baseline) makes the test sensitive proportional to sample size:
          - large batch  -> tight CI -> even small shifts are flagged correctly.
          - small batch  -> wide CI  -> only large shifts are flagged (avoids
                                        false positives from sampling noise).

        Categorical features
        ~~~~~~~~~~~~~~~~~~~~
        TVD = 0.5 * sum |p_new(k) - p_baseline(k)|  in [0, 1]
        Threshold of 0.05 flags any category that shifts by more than 5%.

        Args:
            X_new:          Incoming batch to evaluate (>=2 rows recommended).
            baseline_path:  Path to JSON created by create_baseline().
            z_threshold:    Flag numeric feature when z-score exceeds this.
                            2.0 ~= 95% CI; 3.0 ~= 99.7% CI.
            tvd_threshold:  Flag categorical feature when TVD exceeds this.
                            0.05 = 5% shift in the aggregate distribution.

        Returns:
            DriftReport dataclass. Call .to_dict() for JSON serialisation,
            .summary() for a human-readable string, .has_drift for a bool gate.

        Raises:
            FileNotFoundError: If baseline_path does not exist.
        """
        baseline_file = Path(baseline_path)
        if not baseline_file.exists():
            raise FileNotFoundError(
                f"Baseline not found at '{baseline_path}'. " "Run create_baseline() first."
            )

        with open(baseline_file) as fh:
            baseline = json.load(fh)

        n_new = len(X_new)
        report = DriftReport(
            n_new=n_new,
            z_threshold=z_threshold,
            tvd_threshold=tvd_threshold,
        )

        if n_new < 2:
            warnings.warn(
                f"Drift detection requires n_new >= 2; got {n_new}. Returning empty report.",
                RuntimeWarning,
                stacklevel=2,
            )
            return report

        logger.info(
            f"\n🔍 DRIFT DETECTION  n_new={n_new:,}  "
            f"z_threshold={z_threshold}  tvd_threshold={tvd_threshold:.0%}"
        )

        for feature, stats in baseline["feature_stats"].items():
            # Feature absent from the new batch
            if feature not in X_new.columns:
                report.missing_features.append(feature)
                logger.warning(f"   [!]  Feature '{feature}' missing from new data -- skipped")
                continue

            # Numeric
            if stats["dtype"] == "numeric":
                new_mean = float(X_new[feature].mean())
                baseline_mean = float(stats["mean"])
                baseline_std = float(stats["std"])

                se = _se_of_mean(baseline_std, n_new)
                z = _z_score(new_mean, baseline_mean, se)
                report.drift_scores[feature] = round(z, 4)

                if z > z_threshold:
                    report.drifted_features.append(feature)
                    logger.warning(
                        f"   [!]  {feature}: z={z:.2f} > {z_threshold} | "
                        f"baseline_mean={baseline_mean:.3f} -> new_mean={new_mean:.3f} | "
                        f"se={se:.4f}"
                    )
                else:
                    logger.debug(f"   [ok] {feature}: z={z:.2f} (no drift)")

            # Categorical
            elif stats["dtype"] == "categorical":
                new_dist = X_new[feature].value_counts(normalize=True).to_dict()
                baseline_dist = stats["distribution"]

                tvd = _tvd(new_dist, baseline_dist)
                report.drift_scores[feature] = round(tvd, 4)

                if tvd > tvd_threshold:
                    report.drifted_features.append(feature)
                    logger.warning(
                        f"   [!]  {feature}: TVD={tvd:.3f} > {tvd_threshold:.3f} "
                        f"(distribution shifted)"
                    )
                else:
                    logger.debug(f"   [ok] {feature}: TVD={tvd:.3f} (no drift)")

        # Summary log
        n_drifted = len(report.drifted_features)
        if n_drifted == 0:
            logger.info("   [ok] No significant drift detected")
        else:
            logger.warning(f"   [!]  {n_drifted} feature(s) drifted: {report.drifted_features}")

        return report

    # ------------------------------------------------------------------
    # Batch comparison utility (no baseline file required)
    # ------------------------------------------------------------------

    @staticmethod
    def compare_batches(
        X_ref: pd.DataFrame,
        X_new: pd.DataFrame,
        z_threshold: float = 2.0,
        tvd_threshold: float = 0.05,
    ) -> tuple[DriftReport, dict[str, Any]]:
        """
        Compare two live batches without a persisted baseline.

        Useful for A/B traffic comparisons or canary deployments.

        Args:
            X_ref:          Reference batch (acts as the baseline).
            X_new:          Candidate batch to compare against X_ref.
            z_threshold:    See detect_drift().
            tvd_threshold:  See detect_drift().

        Returns:
            (DriftReport, inline_baseline_dict) -- baseline is not written to
            disk; returned so the caller can cache or log it.
        """
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            DriftMonitor.create_baseline(
                X_ref,
                y_train=pd.Series(np.zeros(len(X_ref))),  # placeholder target
                output_path=tmp_path,
                overwrite=True,
            )
            report = DriftMonitor.detect_drift(
                X_new,
                baseline_path=tmp_path,
                z_threshold=z_threshold,
                tvd_threshold=tvd_threshold,
            )
            with open(tmp_path) as fh:
                inline_baseline = json.load(fh)
        finally:
            os.unlink(tmp_path)

        return report, inline_baseline


# ---------------------------------------------------------------------------
# ModelPerformanceMonitor
# ---------------------------------------------------------------------------


class ModelPerformanceMonitor:
    """
    Track regression model performance on labelled production batches.

    Decoupled from DriftMonitor intentionally: performance monitoring requires
    ground-truth labels, which are often not available at inference time.
    Use both classes together when labels arrive with a delay.

    Usage
    -----
        perf = ModelPerformanceMonitor(threshold_mae=500.0)
        snapshot = perf.evaluate(y_true, y_pred, batch_id="2024-Q1")
        if perf.is_degraded:
            alert(perf.degradation_summary())
    """

    def __init__(
        self,
        threshold_mae: float | None = None,
        threshold_rmse: float | None = None,
        threshold_mape: float | None = None,  # e.g. 0.10 = 10%
        history_limit: int = 100,
    ) -> None:
        self.threshold_mae = threshold_mae
        self.threshold_rmse = threshold_rmse
        self.threshold_mape = threshold_mape
        self.history_limit = history_limit
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        batch_id: str | None = None,
    ) -> PerformanceSnapshot:
        """
        Compute regression metrics for one labelled batch.

        Args:
            y_true:    Ground-truth target values.
            y_pred:    Model predictions.
            batch_id:  Optional identifier stored in history (e.g. a date string).

        Returns:
            PerformanceSnapshot with MAE, RMSE, MAPE, R2, and prediction stats.
        """
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        residuals = y_true_arr - y_pred_arr
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        r2 = float(
            1 - np.sum(residuals**2) / (np.sum((y_true_arr - y_true_arr.mean()) ** 2) + 1e-10)
        )

        # MAPE is undefined when y_true contains zeros
        if np.any(y_true_arr == 0):
            mape = None
            logger.debug("MAPE undefined: y_true contains zeros.")
        else:
            mape = float(np.mean(np.abs(residuals / y_true_arr)))

        snapshot = PerformanceSnapshot(
            n_samples=len(y_true_arr),
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2=r2,
            prediction_mean=float(y_pred_arr.mean()),
            prediction_std=float(y_pred_arr.std()),
        )

        record = {"batch_id": batch_id, **snapshot.to_dict()}
        self._history.append(record)
        if len(self._history) > self.history_limit:
            self._history.pop(0)

        self._log_snapshot(snapshot, batch_id)
        return snapshot

    # ------------------------------------------------------------------

    @property
    def is_degraded(self) -> bool:
        """True if the most recent snapshot breaches any configured threshold."""
        if not self._history:
            return False
        latest = self._history[-1]
        if self.threshold_mae is not None and latest["mae"] > self.threshold_mae:
            return True
        if self.threshold_rmse is not None and latest["rmse"] > self.threshold_rmse:
            return True
        if (
            self.threshold_mape is not None
            and latest.get("mape") is not None
            and latest["mape"] > self.threshold_mape
        ):
            return True
        return False

    def degradation_summary(self) -> str:
        """Human-readable summary of threshold breaches."""
        if not self._history:
            return "No evaluation history."
        latest = self._history[-1]
        lines = [f"[!] Performance degradation detected (batch={latest['batch_id']})"]
        if self.threshold_mae and latest["mae"] > self.threshold_mae:
            lines.append(f"  MAE  {latest['mae']:.2f} > threshold {self.threshold_mae:.2f}")
        if self.threshold_rmse and latest["rmse"] > self.threshold_rmse:
            lines.append(f"  RMSE {latest['rmse']:.2f} > threshold {self.threshold_rmse:.2f}")
        if self.threshold_mape and latest.get("mape") and latest["mape"] > self.threshold_mape:
            lines.append(f"  MAPE {latest['mape']:.2%} > threshold {self.threshold_mape:.2%}")
        return "\n".join(lines)

    def history_as_dataframe(self) -> pd.DataFrame:
        """Return the full evaluation history as a tidy DataFrame."""
        return pd.DataFrame(self._history)

    # ------------------------------------------------------------------

    @staticmethod
    def _log_snapshot(snap: PerformanceSnapshot, batch_id: str | None) -> None:
        tag = f"[{batch_id}] " if batch_id else ""
        logger.info(
            f"\n📈 PERFORMANCE SNAPSHOT {tag}\n"
            f"   n={snap.n_samples:,}  "
            f"MAE={snap.mae:.2f}  "
            f"RMSE={snap.rmse:.2f}  "
            f"MAPE={f'{snap.mape:.2%}' if snap.mape is not None else 'n/a'}  "
            f"R2={snap.r2:.4f}"
        )
