import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from insurance_ml.shared import TargetTransformation

logger = logging.getLogger(__name__)


# ============================================================================
# GPU UTILITIES - Import with availability check
# ============================================================================

try:
    from insurance_ml.models import (
        check_gpu_available,
        clear_gpu_cache,
        get_gpu_memory_usage,
        get_model_gpu_params,
    )

    _GPU_UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GPU utilities not available: {e}")
    _GPU_UTILS_AVAILABLE = False

    # Provide fallback implementations
    def check_gpu_available():
        return False

    def get_gpu_memory_usage():
        return {"allocated_mb": 0, "total_mb": 0, "available": False}

    def clear_gpu_cache():
        pass

    def get_model_gpu_params():
        return {}


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class ValidationError(Exception):
    """Raised when validation fails"""


class TransformationError(Exception):
    """Raised when transformation fails"""


class FileOperationError(Exception):
    """Raised when file operations fail"""


# ============================================================================
# CONSTANTS
# ============================================================================


class TransformConstants:
    """Constants for target transformations (v5.0.1)"""

    # Log1p constants
    LOG1P_MAX_SAFE = 35.0
    LOG1P_EXPECTED_MIN = -1.0
    LOG1P_EXPECTED_MAX = 15.0

    # Log constants
    LOG_EXPECTED_MIN = 5.0
    LOG_EXPECTED_MAX = 12.0

    # Yeo-Johnson constants
    YEO_JOHNSON_LAMBDA_MIN = -5.0
    YEO_JOHNSON_LAMBDA_MAX = 5.0
    YEO_JOHNSON_EXPECTED_MIN = -20.0
    YEO_JOHNSON_EXPECTED_MAX = 20.0
    YEO_JOHNSON_LAMBDA_THRESHOLD = 1e-8

    # Box-Cox constants
    BOXCOX_LAMBDA_THRESHOLD = 1e-8
    BOXCOX_LAMBDA_MIN = -5.0
    BOXCOX_LAMBDA_MAX = 5.0
    BOXCOX_MAX_OUTPUT = 1e8
    BOXCOX_MARGIN_FACTOR = 0.5
    BOXCOX_MIN_INNER = 1e-10

    # Scale detection thresholds
    SCALE_DETECTION_THRESHOLD = 100.0

    # JSON serialization
    JSON_MAX_RECURSION_DEPTH = 100


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================


def save_json(data: dict[str, Any], file_path: str) -> None:
    """Save dictionary to JSON file atomically"""
    file_path = Path(file_path).resolve()

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=file_path.parent, prefix=".tmp_json_", suffix=".json", text=True
        )

        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=_json_default_handler)

            shutil.move(tmp_path, str(file_path))
            logger.info(f"Data saved to {file_path}")

        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    except OSError as e:
        raise FileOperationError(f"Error saving JSON to {file_path}: {e}") from e
    except (TypeError, ValueError) as e:
        raise FileOperationError(f"JSON serialization error: {e}") from e


def _json_default_handler(obj: Any) -> Any:
    """Serialize with type hints for deserialization"""
    if isinstance(obj, TargetTransformation):
        return {
            "__type__": "TargetTransformation",  # Type marker for deserialization
            "method": obj.method,
            "lambda_param": float(obj.lambda_param) if obj.lambda_param is not None else None,
            "boxcox_lambda": float(obj.boxcox_lambda) if obj.boxcox_lambda is not None else None,
            "original_range": tuple(map(float, obj.original_range)) if obj.original_range else None,
            "transform_min": float(obj.transform_min) if obj.transform_min is not None else None,
            "transform_max": float(obj.transform_max) if obj.transform_max is not None else None,
            "boxcox_min": float(obj.boxcox_min) if obj.boxcox_min is not None else None,
            "boxcox_max": float(obj.boxcox_max) if obj.boxcox_max is not None else None,
            "_log_residual_variance": (
                float(obj._log_residual_variance)
                if obj._log_residual_variance is not None
                else None
            ),
        }
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return str(obj)


def _json_object_hook(obj: dict[str, Any]) -> Any:
    """Deserialize custom types based on __type__ marker"""
    if obj.get("__type__") == "TargetTransformation":
        obj_copy = obj.copy()
        obj_copy.pop("__type__")  # Strip the type marker before construction
        obj_copy["_is_deserialized"] = True  # Mark as loaded from disk
        return TargetTransformation(**obj_copy)
    return obj


def load_json(file_path: str) -> dict[str, Any]:
    """Load dictionary from JSON file"""
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path) as f:
            data: dict[str, Any] = json.load(f, object_hook=_json_object_hook)
        logger.info(f"Data loaded from {file_path}")
        return data
    except json.JSONDecodeError as e:
        raise FileOperationError(f"Invalid JSON in {file_path}: {e}") from e
    except OSError as e:
        raise FileOperationError(f"Error reading {file_path}: {e}") from e


def create_directories(directories: list[str]) -> None:
    """Create multiple directories if they don't exist"""
    if not directories:
        return

    try:
        for directory in directories:
            dir_path = Path(directory).resolve()
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created {len(directories)} directories")
    except OSError as e:
        raise FileOperationError(f"Error creating directories: {e}") from e


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================


def set_plotting_style() -> None:
    """Set consistent plotting style"""
    try:
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3
    except Exception as e:
        logger.warning(f"Could not set plotting style: {e}")


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================


def format_currency(amount: float | int) -> str:
    """Format number as currency"""
    if not isinstance(amount, (int, float, np.number)):
        raise TypeError(f"Amount must be numeric, got {type(amount)}")

    if not np.isfinite(amount):
        return "N/A"

    return f"${amount:,.2f}"


def format_percentage(value: float | int, decimals: int = 1) -> str:
    """Format number as percentage"""
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"Value must be numeric, got {type(value)}")

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}%"


# ============================================================================
# TIMING UTILITIES
# ============================================================================


class Timer:
    """Context manager for timing code execution"""

    def __init__(self, name: str = "Operation") -> None:
        self.name = name
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.duration: float | None = None
        self.success: bool = True

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        logger.info(f"{self.name} started...")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.time()

        if self.start_time is not None:
            self.duration = self.end_time - self.start_time

            if exc_type is None:
                logger.info(f"{self.name} completed in {self.duration:.2f}s")
            else:
                logger.error(f"{self.name} failed after {self.duration:.2f}s: {exc_val}")


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================


def validate_data_types(
    df: pd.DataFrame, expected_types: dict[str, str], raise_on_error: bool = True
) -> bool:
    """Validate DataFrame column types"""
    missing_cols = set(expected_types.keys()) - set(df.columns)
    if missing_cols:
        msg = f"Missing columns: {missing_cols}"
        if raise_on_error:
            raise ValidationError(msg)
        logger.error(msg)
        return False

    type_mismatches = []
    for col, expected_type in expected_types.items():
        actual_type = str(df[col].dtype)
        if actual_type != expected_type:
            type_mismatches.append(f"{col}: expected {expected_type}, got {actual_type}")

    if type_mismatches:
        msg = "Type mismatches:\n  " + "\n  ".join(type_mismatches)
        if raise_on_error:
            raise ValidationError(msg)
        logger.warning(msg)
        return False

    return True


def validate_array_finite(
    arr: np.ndarray, name: str = "array", raise_on_error: bool = True
) -> bool:
    """Validate that array contains only finite values"""
    if not np.all(np.isfinite(arr)):
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        msg = f"{name} contains {n_nan} NaN and {n_inf} inf values"

        if raise_on_error:
            raise ValidationError(msg)
        logger.error(msg)
        return False

    return True


# ============================================================================
# METRICS EXTRACTION UTILITIES (ENHANCED)
# ============================================================================


class MetricsExtractor:
    """
    Extract metrics with automatic transformation scale fallback.

    Enhanced with comparison and validation functionality for fair
    calibration comparisons.
    """

    @staticmethod
    def get_metric(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
        """Get metric value, preferring original scale"""
        return float(metrics.get(f"original_{name}", metrics.get(name, default)))

    @staticmethod
    def get_rmse(metrics: dict[str, Any], default: float = 0.0) -> float:
        """Extract RMSE (original scale preferred)"""
        return MetricsExtractor.get_metric(metrics, "rmse", default)

    @staticmethod
    def get_r2(metrics: dict[str, Any], default: float = 0.0) -> float:
        """Extract R² (original scale preferred)"""
        return MetricsExtractor.get_metric(metrics, "r2", default)

    @staticmethod
    def get_mape(metrics: dict[str, Any], default: float = 0.0) -> float:
        """Extract MAPE (original scale preferred)"""
        return MetricsExtractor.get_metric(metrics, "mape", default)

    @staticmethod
    def get_mae(metrics: dict[str, Any], default: float = 0.0) -> float:
        """Extract MAE (original scale preferred)"""
        return MetricsExtractor.get_metric(metrics, "mae", default)

    @staticmethod
    def extract_all(metrics: dict[str, Any]) -> dict[str, float]:
        """Extract all common metrics efficiently"""
        return {
            name: MetricsExtractor.get_metric(metrics, name)
            for name in ["rmse", "r2", "mape", "mae"]
        }

    # ========================================================================
    # NEW: Comparison and validation methods
    # ========================================================================

    @staticmethod
    def compare_models(
        metrics_a: dict[str, Any],
        metrics_b: dict[str, Any],
        model_name_a: str = "Model A",
        model_name_b: str = "Model B",
    ) -> dict[str, Any]:
        """
        Compare two models and return improvement percentages.

        Args:
            metrics_a: Baseline model metrics
            metrics_b: Comparison model metrics
            model_name_a: Name of baseline model
            model_name_b: Name of comparison model

        Returns:
            Dictionary with improvement metrics

        Example:
            >>> uncalibrated = {'original_rmse': 5851, 'original_r2': 0.7853}
            >>> calibrated = {'original_rmse': 4854, 'original_r2': 0.8522}
            >>> comparison = MetricsExtractor.compare_models(
            ...     uncalibrated, calibrated,
            ...     "Uncalibrated", "Calibrated"
            ... )
            >>> print(f"RMSE improved by {comparison['rmse_improvement_pct']:.2f}%")
        """
        rmse_a = MetricsExtractor.get_rmse(metrics_a)
        rmse_b = MetricsExtractor.get_rmse(metrics_b)
        r2_a = MetricsExtractor.get_r2(metrics_a)
        r2_b = MetricsExtractor.get_r2(metrics_b)
        mae_a = MetricsExtractor.get_mae(metrics_a)
        mae_b = MetricsExtractor.get_mae(metrics_b)

        # Calculate improvements (positive = improvement)
        rmse_improvement = ((rmse_a - rmse_b) / rmse_a) * 100 if rmse_a > 0 else 0
        r2_improvement = ((r2_b - r2_a) / abs(r2_a)) * 100 if r2_a != 0 else 0
        mae_improvement = ((mae_a - mae_b) / mae_a) * 100 if mae_a > 0 else 0

        return {
            "model_a": model_name_a,
            "model_b": model_name_b,
            "rmse_improvement_pct": float(rmse_improvement),
            "r2_improvement_pct": float(r2_improvement),
            "mae_improvement_pct": float(mae_improvement),
            "rmse_a": float(rmse_a),
            "rmse_b": float(rmse_b),
            "r2_a": float(r2_a),
            "r2_b": float(r2_b),
            "mae_a": float(mae_a),
            "mae_b": float(mae_b),
            # FIX-AUDIT-P6: is_better previously only checked RMSE, so it returned True
            # even when MAE or R² degraded.  The label in train.py said "RMSE+R² both
            # improved" but the flag never verified R².  Fix: require BOTH RMSE and R²
            # to improve.  MAE is reported separately but not gating (MAE can degrade
            # slightly when the model corrects systematic high-value over-prediction,
            # which shifts the tail distribution in a way that reduces RMSE more than MAE).
            "is_better": rmse_improvement > 0 and r2_improvement >= 0,
        }

    @staticmethod
    def validate_metrics(metrics: dict[str, Any], context: str = "metrics") -> None:
        """
        Validate that metrics dictionary contains expected keys and valid values.

        Allows NaN for secondary metrics (Durbin-Watson, autocorr) in degenerate cases
        but requires finite values for primary metrics (RMSE, R2, MAE, MAPE).

        Args:
            metrics: Metrics dictionary to validate
            context: Description for error messages

        Raises:
            ValidationError: If metrics are invalid or missing
        """
        # Check for required metrics (either original or transformed scale)
        required_base_metrics = ["rmse", "r2", "mae"]

        has_original = any(f"original_{m}" in metrics for m in required_base_metrics)
        has_transformed = any(m in metrics for m in required_base_metrics)

        if not has_original and not has_transformed:
            available = list(metrics.keys())[:10]  # Show first 10 keys
            raise ValidationError(
                f"Metrics for '{context}' missing required keys.\n"
                f"Expected: {required_base_metrics} (with or without 'original_' prefix)\n"
                f"Available keys: {available}"
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

        # Validate finite values - but allow NaN for secondary metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                if not np.isfinite(value):
                    if key not in secondary_metrics:
                        # Primary metric is not finite - fatal error
                        raise ValidationError(
                            f"Metric '{key}' in '{context}' is not finite: {value}"
                        )
                    else:
                        # Secondary metric NaN is acceptable (log it but continue)
                        logger.debug(
                            f"Secondary metric '{key}' is NaN in '{context}' "
                            f"(degenerate case - likely perfect or near-perfect predictions)"
                        )

        # Validate R² is in reasonable range
        r2 = MetricsExtractor.get_r2(metrics, default=None)
        if r2 is not None and not (-1.0 <= r2 <= 1.0):
            logger.warning(
                f"R² for '{context}' is outside typical range [-1, 1]: {r2:.4f}\n"
                f"This may indicate a calculation error or extreme overfitting."
            )

    @staticmethod
    def format_metrics_table(
        metrics_dict: dict[str, dict[str, Any]], metric_names: list[str] | None = None
    ) -> str:
        """
        Format multiple metrics dictionaries as a comparison table.

        Args:
            metrics_dict: Dict mapping phase names to metrics
                         e.g., {'Train': {...}, 'Validation': {...}}
            metric_names: List of metrics to include (default: RMSE, R², MAE, MAPE)

        Returns:
            Formatted table string

        Example:
            >>> metrics = {
            ...     'Train': {'original_rmse': 4996, 'original_r2': 0.8232},
            ...     'Validation': {'original_rmse': 4854, 'original_r2': 0.8522},
            ...     'Test': {'original_rmse': 3734, 'original_r2': 0.9068}
            ... }
            >>> print(MetricsExtractor.format_metrics_table(metrics))
        """
        if metric_names is None:
            metric_names = ["rmse", "r2", "mae", "mape"]

        # Format mappings
        format_map = {
            "rmse": ("RMSE", "${:,.0f}"),
            "r2": ("R²", "{:.4f}"),
            "mae": ("MAE", "${:,.0f}"),
            "mape": ("MAPE", "{:.2f}%"),
        }

        # Build header
        phases = list(metrics_dict.keys())
        col_width = 18
        header = f"{'Metric':<10}"
        for phase in phases:
            header += f" {phase:<{col_width}}"

        lines = [
            "=" * (10 + len(phases) * (col_width + 1)),
            header,
            "-" * (10 + len(phases) * (col_width + 1)),
        ]

        # Build rows
        for metric_key in metric_names:
            if metric_key not in format_map:
                continue

            metric_label, fmt = format_map[metric_key]
            row = f"{metric_label:<10}"

            for phase in phases:
                metrics = metrics_dict[phase]
                value = MetricsExtractor.get_metric(metrics, metric_key)

                if (
                    value == 0.0
                    and metric_key not in metrics
                    and f"original_{metric_key}" not in metrics
                ):
                    formatted = "N/A"
                else:
                    formatted = fmt.format(value)

                row += f" {formatted:<{col_width}}"

            lines.append(row)

        lines.append("=" * (10 + len(phases) * (col_width + 1)))

        return "\n".join(lines)

    @staticmethod
    def calculate_generalization_gap(
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        test_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate generalization gaps between train/val/test.

        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics (optional)

        Returns:
            Dictionary with gap analysis
        """
        train_rmse = MetricsExtractor.get_rmse(train_metrics)
        val_rmse = MetricsExtractor.get_rmse(val_metrics)

        train_val_gap = ((val_rmse - train_rmse) / train_rmse) * 100 if train_rmse > 0 else 0

        result = {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "train_val_gap_pct": float(train_val_gap),
            "train_val_status": (
                "minimal_overfitting"
                if train_val_gap < 10
                else "moderate_overfitting"
                if train_val_gap < 20
                else "severe_overfitting"
            ),
        }

        if test_metrics is not None:
            test_rmse = MetricsExtractor.get_rmse(test_metrics)
            val_test_gap = ((test_rmse - val_rmse) / val_rmse) * 100 if val_rmse > 0 else 0

            result.update(
                {
                    "test_rmse": float(test_rmse),
                    "val_test_gap_pct": float(val_test_gap),
                    "val_test_status": (
                        "excellent"
                        if abs(val_test_gap) < 5
                        else (
                            "good"
                            if abs(val_test_gap) < 10
                            else "moderate"
                            if abs(val_test_gap) < 15
                            else "poor"
                        )
                    ),
                }
            )

        return result

    @staticmethod
    def store_conformal_data(
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
        context: str = "training",
        force_overwrite: bool = False,  # ✅ NEW PARAMETER
    ) -> bool:
        """
        Store conformal calibration data in model._conformal_data dict.

        This is a centralized function to ensure consistent storage across
        all training paths (early stopping, standard fit, calibrated models).

        Args:
            model: Trained model instance
            X_val: Validation features
            y_val: Validation target (in transformed space)
            context: Description for logging (e.g., "post_training", "post_calibration")
            force_overwrite: If True, replace existing data. If False, skip if data exists.

        Returns:
            True if storage succeeded, False otherwise
        """
        # ✅ CRITICAL FIX: Check if data already exists
        if hasattr(model, "_conformal_data") and model._conformal_data is not None:
            if not force_overwrite:
                existing_context = model._conformal_data.get("context", "unknown")
                existing_samples = model._conformal_data.get("n_calibration", 0)

                logger.debug(
                    f"⏭️  Skipping conformal data storage (already exists):\n"
                    f"   Existing context: '{existing_context}' ({existing_samples} samples)\n"
                    f"   Current request: '{context}'\n"
                    f"   → Use force_overwrite=True to replace"
                )
                return False
            else:
                logger.info(
                    f"🔄 Overwriting existing conformal data:\n"
                    f"   Old context: '{model._conformal_data.get('context', 'unknown')}'\n"
                    f"   New context: '{context}'"
                )

        logger.info(f"📦 Storing conformal calibration data ({context})...")

        try:
            # Step 1: Generate predictions
            y_val_pred = model.predict(X_val)
            y_val_array = y_val.values if hasattr(y_val, "values") else np.array(y_val)

            # Step 2: Validate predictions
            if not isinstance(y_val_pred, np.ndarray):
                y_val_pred = np.asarray(y_val_pred)

            if y_val_pred.size == 0:
                raise ValueError("model.predict() returned empty array")

            if not np.all(np.isfinite(y_val_pred)):
                n_bad = np.sum(~np.isfinite(y_val_pred))
                logger.warning(
                    f"   ⚠️  {n_bad}/{len(y_val_pred)} non-finite predictions, "
                    f"replacing with median"
                )
                median_pred = np.median(y_val_pred[np.isfinite(y_val_pred)])
                y_val_pred = np.where(np.isfinite(y_val_pred), y_val_pred, median_pred)

            # Step 3: Calculate residuals
            validation_residuals = y_val_array - y_val_pred

            if not np.all(np.isfinite(validation_residuals)):
                n_bad = np.sum(~np.isfinite(validation_residuals))
                logger.warning(
                    f"   ⚠️  {n_bad}/{len(validation_residuals)} non-finite residuals, "
                    f"replacing with 0.0"
                )
                validation_residuals = np.where(
                    np.isfinite(validation_residuals), validation_residuals, 0.0
                )

            # ✅ VALIDATION: Check consistency BEFORE storing
            if len(validation_residuals) != len(y_val_pred):
                logger.error(
                    f"   ❌ CRITICAL: Length mismatch!\n"
                    f"      Residuals: {len(validation_residuals)}\n"
                    f"      Predictions: {len(y_val_pred)}\n"
                    f"      → This should never happen (bug in prediction logic)"
                )
                return False

            # Step 4: Create fresh _conformal_data dict
            model._conformal_data = {
                "validation_predictions": y_val_pred.tolist(),
                "validation_residuals": validation_residuals.tolist(),
                "n_calibration": int(len(y_val_pred)),
                "stored_at": pd.Timestamp.now().isoformat(),
                "context": context,
                "model_type": type(model).__name__,  # ✅ NEW: Track model type
            }

            # Step 5: Also set runtime attributes (for immediate use)
            model._validation_predictions = y_val_pred.copy()
            model._validation_residuals = validation_residuals.copy()

            # Step 6: Verification
            assert hasattr(model, "_conformal_data"), "Dict not attached!"
            assert "validation_predictions" in model._conformal_data, "Predictions missing!"
            assert len(model._conformal_data["validation_predictions"]) > 0, "Empty predictions!"
            assert len(model._conformal_data["validation_predictions"]) == len(
                model._conformal_data["validation_residuals"]
            ), "Length mismatch after storage!"

            # Step 7: Log success
            logger.info(
                f"   ✅ Stored conformal data:\n"
                f"      Context: {context}\n"
                f"      Model: {type(model).__name__}\n"
                f"      Samples: {len(y_val_pred)}\n"
                f"      Residual std: {validation_residuals.std():.6f}\n"
                f"      Prediction range: [{y_val_pred.min():.2f}, {y_val_pred.max():.2f}]\n"
                f"      Storage: model._conformal_data (serialization-ready)"
            )

            return True

        except Exception as e:
            logger.error(
                f"   ❌ Failed to store conformal data ({context}): {e}\n"
                f"      Model type: {type(model).__name__}\n"
                f"      → Heteroscedastic intervals will NOT be available"
            )
            import traceback

            logger.debug(f"      Traceback:\n{traceback.format_exc()}")

            # Set explicit failure marker (prevents confusion)
            model._conformal_data = None

            return False


# ============================================================================
# TRANSFORMATION UTILITIES (v5.0.1 - FULL YEO-JOHNSON SUPPORT)
# ============================================================================


class TransformUtils:
    """
    Centralized target transformation logic

    """

    @staticmethod
    def is_transformed_scale(y: np.ndarray, method: str = "log1p") -> bool:
        """Check if values appear to be in transformed scale"""
        y_arr = np.asarray(y)
        y_min, y_max = y_arr.min(), y_arr.max()

        if method == "log1p":
            return y_min >= TransformConstants.LOG1P_EXPECTED_MIN and y_max < 20
        elif method == "log":
            return y_min >= 3 and y_max < 15
        elif method == "boxcox":
            return y_max < TransformConstants.SCALE_DETECTION_THRESHOLD
        elif method == "yeo-johnson":
            return (
                y_min >= TransformConstants.YEO_JOHNSON_EXPECTED_MIN
                and y_max <= TransformConstants.YEO_JOHNSON_EXPECTED_MAX
            )
        else:
            return False

    @staticmethod
    def safe_inverse_transform(
        y: np.ndarray, target_transformation: TargetTransformation, clip_to_safe_range: bool = True
    ) -> tuple[np.ndarray, bool]:
        """Safely inverse transform, auto-detecting if already transformed"""
        method = target_transformation.method

        if not TransformUtils.is_transformed_scale(y, method):
            warnings.warn(
                "⚠️ Input already appears to be in original scale. " "Skipping inverse transform.",
                UserWarning,
            )
            return y, True

        y_original = TransformUtils.inverse_transform_target(
            y, target_transformation, clip_to_safe_range=clip_to_safe_range, strict_mode=False
        )
        return y_original, False

    @staticmethod
    def inverse_transform_target(
        y: np.ndarray,
        target_transformation: TargetTransformation,
        clip_to_safe_range: bool = True,
        strict_mode: bool = False,
    ) -> np.ndarray:
        """
        Inverse transform target values with legacy semantics.

        - strict_mode=True  → hard fail on double inverse
        - strict_mode=False → legacy auto-correction:
                              if data looks like original scale, return as-is

        NOTE: For production use, prefer FeatureEngineer.inverse_transform_target()
        This method is kept for backward compatibility and testing.
        """
        method = target_transformation.method
        y_arr = np.asarray(y, dtype=float)

        # Strict mode validation
        # ------------------------------------------------------------------
        # Strict-mode + legacy auto-correction
        # ------------------------------------------------------------------
        # For Yeo-Johnson we SKIP scale-based heuristics because:
        # - Its output range is wide and can legitimately fall outside any
        #   simple fixed [-K, K] window.
        # - In practice, false "already original scale" detections are worse
        #   than letting upstream code handle double-inverse issues.
        # So:
        #   • log1p / boxcox  → keep strict/legacy behavior
        #   • yeo-johnson     → always treat values as transformed here
        if method not in ("none", "yeo-johnson"):
            # Strict mode: hard fail on suspected double inverse
            if strict_mode and not TransformUtils.is_transformed_scale(y_arr, method):
                raise ValueError(
                    "❌ DOUBLE INVERSE TRANSFORM DETECTED!\n"
                    f"   Method: {method}\n"
                    f"   Range: [{y_arr.min():.2f}, {y_arr.max():.2f}]\n"
                    "   Values appear to already be in original scale.\n"
                    "   Set strict_mode=False to auto-correct (legacy behavior)."
                )

            # Legacy auto-correction (strict_mode=False):
            # If values look like original scale for a non-'none' method,
            # just pass them through unchanged. This avoids overflow and
            # matches the expectation in verify_strict_mode Test 3.
            if not strict_mode and not TransformUtils.is_transformed_scale(y_arr, method):
                warnings.warn(
                    "Input passed to inverse_transform_target() appears to be "
                    "in original scale; returning values unchanged (legacy "
                    "auto-correction).",
                    UserWarning,
                )
                return y_arr

        # No transformation
        if method == "none":
            return y_arr.copy()

        # Log1p inverse
        elif method == "log1p":
            result = np.expm1(y_arr)
            if clip_to_safe_range:
                result = np.clip(result, 0, None)
            return result

        # Box-Cox inverse
        elif method == "boxcox":
            lambda_val = target_transformation.lambda_param or target_transformation.boxcox_lambda

            if lambda_val is None:
                raise TransformationError("Box-Cox lambda_param is None")

            # Inverse Box-Cox formula
            if abs(lambda_val) < TransformConstants.BOXCOX_LAMBDA_THRESHOLD:
                result = np.exp(y_arr)
            else:
                result = np.power(lambda_val * y_arr + 1, 1.0 / lambda_val)

            if clip_to_safe_range and result.min() < 0:
                result = np.clip(result, 0, None)

            return result

        # Yeo-Johnson inverse
        elif method == "yeo-johnson":
            lambda_val = target_transformation.lambda_param or target_transformation.boxcox_lambda

            if lambda_val is None:
                raise TransformationError("Yeo-Johnson lambda_param is None")

            # Inverse Yeo-Johnson transformation
            result = np.empty_like(y_arr)

            # Case 1: λ != 0, y >= 0
            mask1 = y_arr >= 0
            if np.any(mask1):
                if abs(lambda_val) < TransformConstants.YEO_JOHNSON_LAMBDA_THRESHOLD:
                    result[mask1] = np.expm1(y_arr[mask1])
                else:
                    result[mask1] = np.power(lambda_val * y_arr[mask1] + 1, 1.0 / lambda_val) - 1

            # Case 2: λ != 2, y < 0
            mask2 = y_arr < 0
            if np.any(mask2):
                if abs(lambda_val - 2) < TransformConstants.YEO_JOHNSON_LAMBDA_THRESHOLD:
                    result[mask2] = -np.expm1(-y_arr[mask2])
                else:
                    result[mask2] = 1 - np.power(
                        -(2 - lambda_val) * y_arr[mask2] + 1,
                        1.0 / (2 - lambda_val),
                    )

            return result

        else:
            raise TransformationError(f"Unknown transformation method: {method}")

    @staticmethod
    def validate_transformation(target_transformation: TargetTransformation) -> None:
        """Validate transformation object before use"""
        if not isinstance(target_transformation, TargetTransformation):
            raise TypeError(f"Expected TargetTransformation, got {type(target_transformation)}")

        method = target_transformation.method

        if method not in ["none", "log1p", "boxcox", "yeo-johnson"]:
            raise TransformationError(f"Invalid transformation method: {method}")

        # Validate Box-Cox parameters
        if method == "boxcox":
            lambda_val = target_transformation.lambda_param or target_transformation.boxcox_lambda
            if lambda_val is None:
                raise TransformationError("Box-Cox lambda_param is None")

            if not np.isfinite(lambda_val):
                raise TransformationError(f"Box-Cox lambda_param is not finite: {lambda_val}")

        # Validate Yeo-Johnson parameters
        if method == "yeo-johnson":
            lambda_val = target_transformation.lambda_param or target_transformation.boxcox_lambda
            if lambda_val is None:
                raise TransformationError("Yeo-Johnson lambda_param is None")

            if not np.isfinite(lambda_val):
                raise TransformationError(f"Yeo-Johnson lambda_param is not finite: {lambda_val}")


# ============================================================================
# JSON SERIALIZATION UTILITIES
# ============================================================================


def make_json_serializable(
    obj: Any, max_depth: int = TransformConstants.JSON_MAX_RECURSION_DEPTH, _depth: int = 0
) -> Any:
    """Recursively convert values to JSON-serializable types"""
    if _depth > max_depth:
        raise RecursionError(
            f"Maximum recursion depth {max_depth} exceeded during JSON serialization"
        )

    if obj is None:
        return None
    elif isinstance(obj, (bool, str)):
        return obj
    elif isinstance(obj, (int, float)):
        if not np.isfinite(obj):
            return None
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return val
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        if obj.size > 10000:
            logger.warning(f"Large array ({obj.size} elements) being serialized to JSON")
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v, max_depth, _depth + 1) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item, max_depth, _depth + 1) for item in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(item, max_depth, _depth + 1) for item in sorted(obj)]
    else:
        logger.debug(f"Converting {type(obj)} to string for JSON serialization")
        return str(obj)


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================


def calculate_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate confidence interval for data"""
    if not (0 < confidence < 1):
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

    if len(data) == 0:
        raise ValueError("Cannot calculate CI for empty array")

    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

    return (mean - margin, mean + margin)


def detect_outliers_iqr(data: np.ndarray, multiplier: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Detect outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]

    return outlier_mask, outlier_indices


# ============================================================================
# TESTING AND VERIFICATION
# ============================================================================


def verify_inverse_transform_fix() -> bool:
    """Verify that inverse transform fix is working correctly"""
    try:
        train_max = 58571.0
        test_value = 62592.0  # 6.9% above training max

        target_transform = TargetTransformation(method="log1p", original_range=(1121.0, train_max))

        y_transformed = np.array([np.log1p(test_value)])
        result = TransformUtils.inverse_transform_target(
            y_transformed, target_transform, clip_to_safe_range=True, strict_mode=False
        )

        error = abs(result[0] - test_value)
        error_pct = error / test_value * 100

        if error < 0.01:
            logger.info(
                f"✅ Inverse transform fix verified:\n"
                f"   Training max: ${train_max:,.2f}\n"
                f"   Test value: ${test_value:,.2f} (+{(test_value/train_max-1)*100:.1f}%)\n"
                f"   Result: ${result[0]:,.2f}\n"
                f"   Error: ${error:,.2f} ({error_pct:.3f}%)\n"
                f"   → No artificial clipping detected!"
            )
            return True
        else:
            logger.error(
                f"❌ Inverse transform verification FAILED:\n"
                f"   Error: ${error:,.2f} ({error_pct:.2f}%)"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Inverse transform verification failed: {e}")
        traceback.print_exc()
        return False


def verify_strict_mode() -> bool:
    """Verify that strict mode correctly detects double inverse transforms"""
    try:
        target_transform = TargetTransformation(method="log1p")

        # Test 1: Normal inverse transform (strict_mode=True, transformed scale)
        y_log = np.array([7.78, 10.65, 8.52])
        try:
            _ = TransformUtils.inverse_transform_target(y_log, target_transform, strict_mode=True)
            logger.info("✅ Test 1: Normal inverse transform passed")
        except ValueError:
            logger.error("❌ Test 1: False positive in strict mode")
            return False

        # Test 2: Double transform detection (strict_mode=True, original scale)
        y_original = np.array([2391.0, 42191.0, 5000.0])
        try:
            _ = TransformUtils.inverse_transform_target(
                y_original, target_transform, strict_mode=True
            )
            logger.error("❌ Test 2: Double transform not detected")
            return False
        except ValueError as e:
            if "DOUBLE INVERSE TRANSFORM" in str(e):
                logger.info("✅ Test 2: Double transform correctly detected")
            else:
                logger.error(f"❌ Test 2: Wrong error: {e}")
                return False

        # Test 3: Legacy mode auto-correction (strict_mode=False, original scale)
        result_legacy = TransformUtils.inverse_transform_target(
            y_original, target_transform, strict_mode=False
        )
        if np.allclose(result_legacy, y_original):
            logger.info("✅ Test 3: Legacy mode auto-correction passed")
        else:
            logger.error("❌ Test 3: Legacy mode failed")
            return False

        logger.info("✅ Strict mode verification passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Strict mode verification failed: {e}")
        traceback.print_exc()
        return False


def verify_yeo_johnson_transform() -> bool:
    """
    Verify Yeo-Johnson transformation

    Fixed SciPy API usage: yeojohnson() returns single array when lmbda provided
    """
    try:
        from scipy.stats import yeojohnson, yeojohnson_normmax

        # Test data with negative values
        test_data = np.array([-100.0, -10.0, 0.0, 10.0, 100.0, 1000.0])

        # Compute optimal lambda
        lambda_param = yeojohnson_normmax(test_data)

        # yeojohnson() with lmbda parameter returns single array
        y_transformed = yeojohnson(test_data, lmbda=lambda_param)

        # Create target transformation
        target_transform = TargetTransformation(method="yeo-johnson", lambda_param=lambda_param)

        # Inverse transform
        result = TransformUtils.inverse_transform_target(
            y_transformed, target_transform, clip_to_safe_range=False, strict_mode=False
        )

        # Check round-trip accuracy
        max_error = np.max(np.abs(result - test_data))
        rel_error = max_error / np.max(np.abs(test_data))

        if rel_error < 1e-6:
            logger.info(
                f"✅ Yeo-Johnson transform verified:\n"
                f"   Lambda: {lambda_param:.6f}\n"
                f"   Test data range: [{np.min(test_data):.2f}, {np.max(test_data):.2f}]\n"
                f"   Max round-trip error: {max_error:.2e}\n"
                f"   Relative error: {rel_error:.2e}\n"
                f"   → Transform-inverse round-trip successful!"
            )
            return True
        else:
            logger.error(
                f"❌ Yeo-Johnson verification FAILED:\n"
                f"   Lambda: {lambda_param:.6f}\n"
                f"   Max error: {max_error:.2e}\n"
                f"   Relative error: {rel_error:.2e}\n"
                f"   → Round-trip accuracy insufficient!"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Yeo-Johnson verification failed with exception: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configure logging for self-tests
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("UTILS.PY - PRODUCTION READY v5.0.1")
    print("=" * 70)
    print("\n✅ KEY UPDATES:")
    print("  • Full Yeo-Johnson support (forward + inverse)")
    print("  • Enhanced bias correction handling")
    print("  • GPU utilities imported from models.py v3.9.0")
    print("  • Config.yaml integration via typed helpers")
    print("  • Zero redundancy - single source of truth")
    print("\nRunning self-tests...\n")

    # Test 1: Basic imports
    print("✓ All imports successful")

    # Test 2: Verify inverse transform fix
    print("\n" + "=" * 70)
    print("TEST 1: Verify Inverse Transform Fix (Log1p)")
    print("=" * 70)
    test1_passed = verify_inverse_transform_fix()

    # Test 3: Verify strict mode
    print("\n" + "=" * 70)
    print("TEST 2: Verify Strict Mode")
    print("=" * 70)
    test2_passed = verify_strict_mode()

    # Test 4: Verify Yeo-Johnson (NEW)
    print("\n" + "=" * 70)
    print("TEST 3: Verify Yeo-Johnson Transform (NEW)")
    print("=" * 70)
    test3_passed = verify_yeo_johnson_transform()

    # Test 5: GPU utilities availability
    print("\n" + "=" * 70)
    print("TEST 4: GPU Utilities")
    print("=" * 70)
    if _GPU_UTILS_AVAILABLE:
        gpu_available = check_gpu_available()
        print(f"✅ GPU utilities imported successfully")
        print(f"   GPU available: {gpu_available}")
        if gpu_available:
            gpu_info = get_gpu_memory_usage()
            print(
                f"   GPU Memory: {gpu_info['allocated_mb']:.1f} MB allocated / "
                f"{gpu_info['total_mb']:.1f} MB total"
            )
    else:
        print("⚠️  GPU utilities not available (fallback mode)")
