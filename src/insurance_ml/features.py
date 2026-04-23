"""
Feature Engineering Module

Usage:
    from insurance_ml.config import load_config, get_feature_config

    config = load_config()
    feat_cfg = get_feature_config(config)
    fe = FeatureEngineer(config_dict=feat_cfg)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    StandardScaler,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from insurance_ml.shared import TargetTransformation

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION DATACLASS (NO DEFAULTS - ALL FROM config.yaml)
# ============================================================================


@dataclass
class FeatureEngineeringConfig:
    """
    Feature engineering configuration (v4.3.2)
    Source: config.yaml via config.py::get_feature_config()
    """

    # Categorical Encoding Maps (from config.yaml features.encoding)
    smoker_binary_map: dict[str, int]  # e.g., {"yes": 1, "no": 0}
    smoker_risk_map: dict[str, int]  # e.g., {"yes": 2, "no": 0}

    # Feature Engineering Thresholds (from config.yaml features.engineering)
    variance_threshold: float  # e.g., 0.000001

    # Multicollinearity (from config.yaml features.collinearity_removal)
    correlation_threshold: float  # e.g., 0.90
    vif_threshold: float  # e.g., 5.0
    max_vif_iterations: int  # e.g., 5
    use_optimized_vif: bool  # e.g., true

    # Polynomial Features (from config.yaml features.polynomial_features)
    polynomial_degree: int  # e.g., 2
    max_polynomial_features: int  # e.g., 50

    # Outlier Detection (from config.yaml features.outlier_removal)
    outlier_contamination: float  # e.g., 0.05
    outlier_random_state: int  # e.g., 42

    # Validation Ranges (from config.yaml features)
    bmi_min: float  # e.g., 10.0
    bmi_max: float  # e.g., 100.0
    age_min: float  # e.g., 0.0
    age_max: float  # e.g., 120.0

    # Performance Monitoring (from config.yaml diagnostics.performance)
    enable_performance_logging: bool  # e.g., false
    log_memory_usage: bool  # e.g., false

    # Internal threshold (not user-configurable)
    continuous_feature_threshold: int = 10  # ONLY hardcoded value (internal logic)

    def validate(self) -> None:
        """
        Validate configuration ranges (NO DEFAULTS)

        Raises:
            ValueError: If any parameter is out of valid range
        """
        # Correlation threshold
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError(
                f"❌ correlation_threshold must be in (0, 1], got {self.correlation_threshold}\n"
                f"   (from config.yaml features.collinearity_removal.threshold)"
            )

        # VIF threshold
        if self.vif_threshold <= 0:
            raise ValueError(
                f"❌ vif_threshold must be > 0, got {self.vif_threshold}\n"
                f"   (from config.yaml features.collinearity_removal.vif_threshold)"
            )

        # Polynomial degree
        if self.polynomial_degree < 1:
            raise ValueError(
                f"❌ polynomial_degree must be >= 1, got {self.polynomial_degree}\n"
                f"   (from config.yaml features.polynomial_features.degree)"
            )

        # Outlier contamination
        if not 0 < self.outlier_contamination < 0.5:
            raise ValueError(
                f"❌ outlier_contamination must be in (0, 0.5), got {self.outlier_contamination}\n"
                f"   (from config.yaml features.outlier_removal.contamination)"
            )

        # BMI range
        if self.bmi_min >= self.bmi_max:
            raise ValueError(
                f"❌ bmi_min ({self.bmi_min}) must be < bmi_max ({self.bmi_max})\n"
                f"   (from config.yaml features.bmi_min/bmi_max)"
            )

        # Age range
        if self.age_min >= self.age_max:
            raise ValueError(
                f"❌ age_min ({self.age_min}) must be < age_max ({self.age_max})\n"
                f"   (from config.yaml features.age_min/age_max)"
            )


# ============================================================================
# PIPELINE STATE TRACKING
# ============================================================================


class PipelineState(Enum):
    """Pipeline execution state for validation"""

    INITIALIZED = "initialized"
    FEATURES_CREATED = "features_created"
    IMPUTED = "imputed"
    OUTLIERS_DETECTED = "outliers_detected"
    ENCODED = "encoded"
    SCALED = "scaled"
    COMPLETED = "completed"


# ============================================================================
# BIAS CORRECTION
# ============================================================================


@dataclass
class BiasCorrection:
    """
    Stratified lognormal bias correction (Duan's smearing estimator).

    ML-08 NOTE: A canonical version of this class also exists in train.py.
    Both versions share the same public interface (apply, to_dict, from_dict).
    The train.py version additionally provides calculate_from_model() and get_hash().

    TODO: Consolidate into a single shared module (e.g., insurance_ml/bias_correction.py)
    and import from there in both train.py and features.py to prevent interface drift.

    When a model is trained on a log1p-transformed target, naïve back-transformation
    with expm1() produces predictions that are systematically LOW because:

        E[exp(ŷ)] ≠ exp(E[ŷ])   (Jensen's inequality)

    The unbiased correction multiplies each prediction by exp(σ²/2), where σ² is the
    residual variance in log-space computed on the training fold:

        ŷ_corrected = ŷ_original × exp(σ²/2)

    This class supports a **2-tier** model (low-premium / high-premium with one
    threshold) and an optional **3-tier** extension (low / mid / high with two
    thresholds).  Using separate variances per tier avoids over-correcting cheap
    policies and under-correcting expensive ones.

    Tier routing at inference uses the raw back-transformed prediction as a proxy
    for the ground-truth y (self-referential routing).  Predictions near a tier
    boundary may be misrouted by one tier.  The estimated impact is ~10% of
    policies near the q50/q75 boundaries (~$200–500 systematic error per affected
    policy).  To eliminate: ensure BiasCorrection.calculate_from_model() derives
    tier thresholds and masks from y_pred_original, not y_val_original, so the
    training partition is identical to the inference partition.

    Attributes:
        var_low:        Residual log-space variance for the low-premium tier.
        var_high:       Residual log-space variance for the high-premium tier.
        threshold:      2-tier boundary in original scale (e.g. $15,000).
                        Ignored when var_mid is provided (3-tier mode uses
                        threshold_low / threshold_high instead).
        var_mid:        (Optional) Variance for the mid-tier.  When set, activates
                        3-tier mode.
        threshold_low:  (3-tier) Lower boundary between low and mid tiers.
        threshold_high: (3-tier) Upper boundary between mid and high tiers.

    Example — 2-tier (most common):
        >>> bc = BiasCorrection(var_low=0.04, var_high=0.09, threshold=15000.0)
        >>> corrected = bc.apply(y_pred=preds, y_original=preds)

    Example — construct from JSON artifact written by train.py:
        >>> with open("models/bias_correction.json") as f:
        ...     bc = BiasCorrection.from_dict(json.load(f))
    """

    # ML-08 FIX: Add __version__ for cross-module serialisation validation.
    # When a preprocessor containing this class is deserialised, callers can
    # check that the loaded object's __version__ matches the current one to
    # detect interface mismatches (e.g. missing get_hash / 3-tier attrs).
    __version__ = "1.1"

    var_low: float
    var_high: float
    threshold: float = 15_000.0  # sensible default; always overridden from JSON

    # 3-tier extension (all three must be set together)
    var_mid: float | None = None
    threshold_low: float | None = None
    threshold_high: float | None = None

    def __post_init__(self) -> None:
        # Variance values CAN be negative: var = 2*log(ratio), so when the model
        # over-predicts (ratio < 1.0) var < 0 and exp(var/2) < 1.0 — a valid
        # downward correction. Zero is the only invalid value (sentinel / uninitialised).
        import math as _m

        for _name, _v in [("var_low", self.var_low), ("var_high", self.var_high)]:
            if _v == 0 or not _m.isfinite(_v):
                raise ValueError(f"{_name} must be non-zero and finite, got {_v}")
        three_tier_fields = (self.var_mid, self.threshold_low, self.threshold_high)
        if any(v is not None for v in three_tier_fields):
            if not all(v is not None for v in three_tier_fields):
                raise ValueError(
                    "3-tier mode requires ALL of: var_mid, threshold_low, threshold_high. "
                    f"Got: var_mid={self.var_mid}, threshold_low={self.threshold_low}, "
                    f"threshold_high={self.threshold_high}"
                )
            if self.threshold_low >= self.threshold_high:  # type: ignore[operator]
                raise ValueError(
                    f"threshold_low ({self.threshold_low}) must be < "
                    f"threshold_high ({self.threshold_high})"
                )
            if self.var_mid == 0 or not _m.isfinite(self.var_mid):  # type: ignore[operator]
                raise ValueError(f"var_mid must be non-zero and finite, got {self.var_mid}")
        # In 3-tier mode threshold is a dummy sentinel; routing uses threshold_low/high.
        elif self.threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_2tier(self) -> bool:
        """True when operating in 2-tier mode (var_mid is None)."""
        return self.var_mid is None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise to a plain dict suitable for json.dump().

        Only the fields required for the active tier mode are written,
        keeping the JSON compact and unambiguous.
        """
        d: dict[str, Any] = {
            "var_low": self.var_low,
            "var_high": self.var_high,
            "threshold": self.threshold,
        }
        if not self.is_2tier:
            d["var_mid"] = self.var_mid
            d["threshold_low"] = self.threshold_low
            d["threshold_high"] = self.threshold_high
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BiasCorrection | None:
        """
        Deserialise from a dict produced by to_dict() or from bias_correction.json.

        Handles three formats:

        1. **Canonical 2-tier** (written by ``BiasCorrection.to_dict()``):
              {"var_low": …, "var_high": …, "threshold": …}

        2. **Canonical 3-tier** — ``threshold`` may be absent in JSONs written by
           ModelManager's internal serialiser.  Falls back to ``threshold_low``
           (semantically equivalent; 3-tier routing never uses plain ``threshold``).

        3. **Legacy** — raw FeatureEngineer attrs before BiasCorrection existed:
              {"_bias_var_low": …, "_bias_var_high": …, "_bias_threshold": …}
        """
        # ── BUG-6 FIX (v7.5.0): Unwrap BiasCorrectionArtifact wrapper ──────
        # always_write_bias_correction() wraps the correction inside a
        # BiasCorrectionArtifact dataclass via asdict():
        #   {"applied": bool, "reason": str, "correction_params": {…}, …}
        # The old from_dict expected var_low at the top level; peel the wrapper.
        if "applied" in data:
            if not data["applied"]:
                # Quantile model: bias correction intentionally absent.
                return None
            inner = data.get("correction_params")
            if not inner:
                raise ValueError(
                    "bias_correction.json has applied=True but correction_params "
                    "is empty or null. Re-run train.py to regenerate the artifact."
                )
            data = inner  # fall through to canonical parse below

        # ── Canonical key names ─────────────────────────────────────────────
        if "var_low" in data:
            # Fallback chain: "threshold" → "threshold_low" → 0.0
            # 0.0 is only a safe sentinel for 3-tier objects where __post_init__
            # validation accepts it; 3-tier routing uses threshold_low/threshold_high.
            _threshold = float(
                data["threshold"] if "threshold" in data else data.get("threshold_low", 0.0)
            )
            return cls(
                var_low=float(data["var_low"]),
                var_high=float(data["var_high"]),
                threshold=_threshold,
                var_mid=float(data["var_mid"]) if "var_mid" in data else None,
                threshold_low=float(data["threshold_low"]) if "threshold_low" in data else None,
                threshold_high=float(data["threshold_high"]) if "threshold_high" in data else None,
            )

        # ── Legacy key names ────────────────────────────────────────────────
        if "_bias_var_low" in data:
            return cls(
                var_low=float(data["_bias_var_low"]),
                var_high=float(data["_bias_var_high"]),
                threshold=float(data["_bias_threshold"]),
            )

        raise KeyError(
            "bias_correction.json must contain either:\n"
            "  Canonical: ('var_low', 'var_high', 'threshold')\n"
            "  3-tier:    ('var_low', 'var_high', 'threshold_low', 'threshold_high', 'var_mid')\n"
            "  Legacy:    ('_bias_var_low', '_bias_var_high', '_bias_threshold')\n"
            f"Found keys: {list(data.keys())}"
        )

    # ------------------------------------------------------------------
    # Core correction
    # ------------------------------------------------------------------

    def apply(
        self,
        y_pred: np.ndarray,
        y_original: np.ndarray,
        log_details: bool = False,
    ) -> np.ndarray:
        """
        Apply stratified lognormal smearing correction to point predictions.

        Correction formula per tier:
            y_corrected = y_pred × exp(σ²_tier / 2)

        Args:
            y_pred:      Predictions in original scale (the values being corrected).
            y_original:  Routing signal used to assign tier membership.
                         At inference time pass ``y_pred`` itself (self-referential).
                         During evaluation pass ground-truth y for exact tier routing.
            log_details: Emit DEBUG-level per-tier diagnostics when True.  Gated on
                         ``logger.isEnabledFor(logging.DEBUG)`` in predict.py so this
                         has zero overhead in production INFO mode.

        Returns:
            np.ndarray with the same shape as y_pred, bias-corrected in-place on a copy.
        """
        y_corrected = np.array(y_pred, dtype=float)
        y_routing = np.asarray(y_original, dtype=float)

        if self.is_2tier:
            low_mask = y_routing < self.threshold
            high_mask = ~low_mask

            factor_low = float(np.exp(self.var_low / 2.0))
            factor_high = float(np.exp(self.var_high / 2.0))

            y_corrected[low_mask] *= factor_low
            y_corrected[high_mask] *= factor_high

            if log_details:
                logger.debug(
                    f"BiasCorrection (2-tier, threshold={self.threshold:,.0f}): "
                    f"low  n={int(low_mask.sum())} factor={factor_low:.6f} "
                    f"(var={self.var_low:.6f})  |  "
                    f"high n={int(high_mask.sum())} factor={factor_high:.6f} "
                    f"(var={self.var_high:.6f})"
                )

        else:
            # 3-tier mode
            low_mask = y_routing < self.threshold_low
            mid_mask = (y_routing >= self.threshold_low) & (y_routing < self.threshold_high)
            high_mask = y_routing >= self.threshold_high

            factor_low = float(np.exp(self.var_low / 2.0))
            factor_mid = float(np.exp(self.var_mid / 2.0))  # type: ignore[arg-type]
            factor_high = float(np.exp(self.var_high / 2.0))

            y_corrected[low_mask] *= factor_low
            y_corrected[mid_mask] *= factor_mid
            y_corrected[high_mask] *= factor_high

            if log_details:
                logger.debug(
                    f"BiasCorrection (3-tier, "
                    f"thresholds=[{self.threshold_low:,.0f}, {self.threshold_high:,.0f}]): "
                    f"low n={int(low_mask.sum())} factor={factor_low:.6f} | "
                    f"mid n={int(mid_mask.sum())} factor={factor_mid:.6f} | "
                    f"high n={int(high_mask.sum())} factor={factor_high:.6f}"
                )

        return y_corrected

    def get_hash(self) -> str:
        """Generate deterministic hash for caching and serialisation validation.

        ML-08 FIX: Added to match the train.py BiasCorrection interface.
        Preprocessors serialised with this class can now call get_hash() without
        AttributeError at inference time.
        """
        import hashlib as _hl

        if self.is_2tier:
            state_str = f"2tier_{self.var_low:.10f}_{self.var_high:.10f}_{self.threshold:.4f}"
        else:
            state_str = (
                f"3tier_{self.var_low:.10f}_{self.var_mid:.10f}_{self.var_high:.10f}_"
                f"{self.threshold_low:.4f}_{self.threshold_high:.4f}"
            )
        return _hl.md5(state_str.encode()).hexdigest()[:8]


# ============================================================================
# MAIN FEATURE ENGINEER CLASS (ZERO REDUNDANCY)
# ============================================================================


class FeatureEngineer:
    VERSION = "4.3.2"
    COMPATIBLE_VERSIONS = ["4.3.1", "4.3.2"]
    REQUIRED_CONFIG_VERSION = "6.1.0"  # Minimum config.yaml version

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize FeatureEngineer

        Args:
            config: FeatureEngineeringConfig instance (optional)
            config_dict: Dictionary from get_feature_config() (recommended)

        Raises:
            ValueError: If neither config nor config_dict is provided
            KeyError: If config_dict is missing required keys

        Example:
            >>> from insurance_ml.config import load_config, get_feature_config
            >>> config = load_config()
            >>> feat_cfg = get_feature_config(config)
            >>> fe = FeatureEngineer(config_dict=feat_cfg)
        """
        # config_dict > config > ERROR (no defaults!)
        if config_dict is not None:
            self.config = self._create_config_from_dict(config_dict)
        elif config is not None:
            self.config = config
        else:
            raise ValueError(
                "❌ FeatureEngineer requires configuration!\n\n"
                "   ⚠️  Config.yaml v6.1.0 is the SINGLE SOURCE OF TRUTH\n"
                "   No defaults are provided in Python code.\n\n"
                "   ✅ CORRECT USAGE:\n"
                "     from insurance_ml.config import load_config, get_feature_config\n"
                "     \n"
                "     config = load_config()\n"
                "     feat_cfg = get_feature_config(config)\n"
                "     fe = FeatureEngineer(config_dict=feat_cfg)\n\n"
                f"   📋 Required config.yaml version: >= {self.REQUIRED_CONFIG_VERSION}"
            )

        # Validate config (will fail if missing required fields)
        self.config.validate()

        # Pipeline state tracking
        self._state = PipelineState.INITIALIZED
        self._fit_complete = False

        # Target transformation
        self.target_transformation = TargetTransformation(method="none")
        self.target_min_ = None
        self.target_max_ = None
        self.transformed_min_ = None
        self.transformed_max_ = None

        # Scaling
        self.scaler = StandardScaler()
        self._continuous_features = []

        # Encoding (immutable snapshots)
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.onehot_encoder = OneHotEncoder(
            sparse_output=False, drop="first", handle_unknown="ignore"
        )
        self._feature_names_snapshot: tuple[str, ...] | None = None
        self.most_frequent_values: dict[str, str] = {}

        # Polynomial features
        self.poly_transformer: PolynomialFeatures | None = None
        self._poly_feature_names_snapshot: tuple[str, ...] | None = None
        self._poly_continuous_features_snapshot: tuple[str, ...] | None = None

        # Imputation
        self.demographic_imputer: SimpleImputer | None = None
        self.derived_imputer: SimpleImputer | None = None
        self.categorical_imputer: SimpleImputer | None = None

        # Variance filtering
        self.variance_selector: VarianceThreshold | None = None
        self._variance_feature_names_snapshot: tuple[str, ...] | None = None

        # Multicollinearity tracking
        self._removed_collinear_features_snapshot: tuple[str, ...] = ()

        # Outlier detection
        self.outlier_detector: IsolationForest | None = None
        self.outlier_method: str = "none"
        self._outlier_indices: np.ndarray | None = None

        # Input validation
        self._original_columns: list[str] | None = None
        self._original_y_train_stats: dict | None = None

        # Performance tracking
        self._performance_metrics: dict[str, float] = {}

        # Track bias correction logging to avoid spam
        self._bias_correction_logged = False
        self._bias_application_count = 0

        # Initialize stratified bias correction attributes
        # self._bias_var_low = None
        # self._bias_var_high = None
        # self._bias_threshold = None

        logger.info(
            f"✅ FeatureEngineer v{self.VERSION} initialized from config.yaml v{self.REQUIRED_CONFIG_VERSION}+"
        )
        logger.debug(
            f"   Configuration source: {'config_dict' if config_dict else 'config object'}"
        )

    def _create_config_from_dict(self, config_dict: dict[str, Any]) -> FeatureEngineeringConfig:
        # VALIDATE ALL REQUIRED KEYS
        required_keys = [
            "smoker_binary_map",
            "smoker_risk_map",
            "variance_threshold",
            "correlation_threshold",
            "vif_threshold",
            "max_vif_iterations",
            "use_optimized_vif",
            "polynomial_degree",
            "polynomial_max_features",
            "outlier_contamination",
            "outlier_random_state",
            "bmi_min",
            "bmi_max",
            "age_min",
            "age_max",
            "enable_performance_logging",
            "log_memory_usage",
        ]

        missing_keys = [key for key in required_keys if key not in config_dict]

        if missing_keys:
            raise KeyError(
                f"❌ config_dict missing required keys: {missing_keys}\n\n"
                f"   ⚠️  Config.yaml v{self.REQUIRED_CONFIG_VERSION} is the SINGLE SOURCE OF TRUTH\n"
                f"   get_feature_config() should provide all required keys.\n\n"
                f"   ✅ CHECK YOUR CONFIG.YAML STRUCTURE:\n"
                f"   features:\n"
                f"     encoding:  # ✅ NEW SECTION\n"
                f"       smoker_binary_map: {{yes: 1, no: 0}}\n"
                f"       smoker_risk_map: {{yes: 2, no: 0}}\n"
                f"     engineering:\n"
                f"       variance_threshold: 1e-6\n"
                f"     collinearity_removal:\n"
                f"       threshold: 0.90\n"
                f"       vif_threshold: 5.0\n"
                f"       max_vif_iterations: 5\n"
                f"       use_optimized_vif: true\n"
                f"     polynomial_features:\n"
                f"       degree: 2\n"
                f"       max_features: 50\n"
                f"     outlier_removal:\n"
                f"       contamination: 0.05\n"
                f"       random_state: 42\n"
                f"     bmi_min: 10.0\n"
                f"     bmi_max: 100.0\n"
                f"     age_min: 0.0\n"
                f"     age_max: 120.0\n"
                f"   diagnostics:\n"
                f"     performance:\n"
                f"       enabled: true\n"
                f"       log_memory: true\n"
            )

        # EXTRACT VALUES (NO FALLBACK DEFAULTS!)
        return FeatureEngineeringConfig(
            # Encoding maps (from features.encoding)
            smoker_binary_map=config_dict["smoker_binary_map"],
            smoker_risk_map=config_dict["smoker_risk_map"],
            # Thresholds (from features.engineering)
            variance_threshold=config_dict["variance_threshold"],
            # Multicollinearity (from features.collinearity_removal)
            correlation_threshold=config_dict["correlation_threshold"],
            vif_threshold=config_dict["vif_threshold"],
            max_vif_iterations=config_dict["max_vif_iterations"],
            use_optimized_vif=config_dict["use_optimized_vif"],
            # Polynomial features (from features.polynomial_features)
            polynomial_degree=config_dict["polynomial_degree"],
            max_polynomial_features=config_dict["polynomial_max_features"],
            # Outlier detection (from features.outlier_removal)
            outlier_contamination=config_dict["outlier_contamination"],
            outlier_random_state=config_dict["outlier_random_state"],
            # Validation ranges (from features)
            bmi_min=config_dict["bmi_min"],
            bmi_max=config_dict["bmi_max"],
            age_min=config_dict["age_min"],
            age_max=config_dict["age_max"],
            # Performance (from diagnostics.performance)
            enable_performance_logging=config_dict["enable_performance_logging"],
            log_memory_usage=config_dict["log_memory_usage"],
        )

    # ============================================================
    # CONTEXT MANAGERS
    # ============================================================

    @contextmanager
    def _timed_step(self, step_name: str):
        """
        Context manager for timing pipeline steps

        Uses config.enable_performance_logging and config.log_memory_usage
        """
        if not self.config.enable_performance_logging:
            yield
            return

        start_time = time.perf_counter()
        start_memory = None

        if self.config.log_memory_usage:
            try:
                import psutil

                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024**2  # MB
            except ImportError:
                pass

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self._performance_metrics[step_name] = elapsed

            log_msg = f"⏱️  {step_name}: {elapsed:.2f}s"

            if start_memory is not None:
                try:
                    import psutil

                    process = psutil.Process()
                    end_memory = process.memory_info().rss / 1024**2
                    delta_memory = end_memory - start_memory
                    log_msg += f" | Memory Δ: {delta_memory:+.1f} MB"
                except ImportError:
                    pass

            logger.info(log_msg)

    def _validate_state(self, required_state: PipelineState, operation: str) -> None:
        """Validate pipeline state before operation"""
        state_order = list(PipelineState)
        required_idx = state_order.index(required_state)
        current_idx = state_order.index(self._state)

        if current_idx < required_idx:
            raise RuntimeError(
                f"Cannot perform '{operation}': requires state {required_state.value}, "
                f"but current state is {self._state.value}. "
                f"Please execute pipeline steps in correct order."
            )

    def _update_state(self, new_state: PipelineState) -> None:
        """Update pipeline state.

        After the pipeline reaches COMPLETED (fit_transform_pipeline() finished),
        state is frozen — backward transitions are silently ignored.

        Root cause of the stale-state warning: transform_new_data() calls
        create_features() on an already-fitted instance. create_features() always
        called _update_state(FEATURES_CREATED), regressing COMPLETED → FEATURES_CREATED
        while _fit_complete stayed True. save_preprocessor() then detected the
        contradiction and emitted a warning on every save.

        The fix is here rather than in create_features() so ALL pipeline steps
        (impute, encode, scale, etc.) are protected from backward regression,
        not just create_features().

        Legitimate re-fit: a fresh fit resets _fit_complete=False and _state=INITIALIZED
        in __init__ (or via an explicit reset) before calling fit_transform_pipeline(),
        so the guard never blocks a genuine re-fit.
        """
        state_order = list(PipelineState)
        current_idx = state_order.index(self._state)
        new_idx = state_order.index(new_state)

        if self._fit_complete and new_idx < current_idx:
            # Pipeline is fully fitted — suppress backward transitions silently.
            # These occur when transform_new_data() re-runs create_features() /
            # other steps in transform (fit=False) mode on an already-fitted instance.
            logger.debug(
                f"Pipeline state transition suppressed after COMPLETED: "
                f"{self._state.value} -> {new_state.value} (fit_complete=True, transform mode)"
            )
            return

        logger.debug(f"Pipeline state: {self._state.value} -> {new_state.value}")
        self._state = new_state

    def transform_target(
        self,
        y: pd.Series,
        method: Literal["none", "log1p", "yeo-johnson", "boxcox"] = "none",
        fit: bool = True,
        original_range: tuple[float, float] | None = None,
        add_buffer: bool = True,
        buffer_pct: float = 0.40,
    ) -> pd.Series:
        # Input validation
        if not isinstance(y, pd.Series):
            raise TypeError(f"Target must be pd.Series, got {type(y)}")
        if y.isna().any():
            raise ValueError(f"Target contains {y.isna().sum()} NaN values")
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"Target must be numeric, got dtype {y.dtype}")

        if method == "none":
            if fit:
                self.target_transformation = TargetTransformation(method="none")
            return y

        elif method == "yeo-johnson":
            logger.info("Applying Yeo-Johnson transformation to target")

            if fit:
                self.target_min_ = float(y.min())
                self.target_max_ = float(y.max())

                # Add buffer for unseen values
                if add_buffer:
                    self.y_min_safe = float(self.target_min_ * (1 - buffer_pct))
                    self.y_max_safe = float(self.target_max_ * (1 + buffer_pct))
                    logger.info(
                        f"Added {buffer_pct*100}% buffer: "
                        f"safe range ({self.y_min_safe:.2f}, {self.y_max_safe:.2f})"
                    )

                # Create and fit Yeo-Johnson transformer
                self.yeo_johnson_transformer = PowerTransformer(
                    method="yeo-johnson",
                    standardize=False,  # Keep scale for interpretability
                )

                y_transformed = self.yeo_johnson_transformer.fit_transform(
                    y.values.reshape(-1, 1)
                ).ravel()

                # Extract and store lambda parameter
                lambda_param = float(self.yeo_johnson_transformer.lambdas_[0])

                self.transformed_min_ = float(y_transformed.min())
                self.transformed_max_ = float(y_transformed.max())

                # Calculate safe clipping bounds (3 std from mean)
                self.transformed_mean_ = float(y_transformed.mean())
                self.transformed_std_ = float(y_transformed.std())
                self.clip_lower_ = float(self.transformed_mean_ - 3.0 * self.transformed_std_)
                self.clip_upper_ = float(self.transformed_mean_ + 3.0 * self.transformed_std_)

                y_range = original_range or (self.target_min_, self.target_max_)

                # Create TargetTransformation with lambda_param
                self.target_transformation = TargetTransformation(
                    method="yeo-johnson",
                    lambda_param=lambda_param,
                    original_range=y_range,
                    transform_min=self.transformed_min_,
                    transform_max=self.transformed_max_,
                    _skip_validation=False,
                )

                logger.info(
                    f"Yeo-Johnson transformation complete:\n"
                    f"  Lambda: {lambda_param:.6f}\n"
                    f"  Original range: ({self.target_min_:.2f}, {self.target_max_:.2f})\n"
                    f"  Transformed range: ({self.transformed_min_:.4f}, {self.transformed_max_:.4f})\n"
                    f"  Safe clipping bounds: ({self.clip_lower_:.4f}, {self.clip_upper_:.4f})"
                )
            else:
                if not hasattr(self, "yeo_johnson_transformer"):
                    raise RuntimeError(
                        "Yeo-Johnson transformation not fitted. Call with fit=True first."
                    )
                y_transformed = self.yeo_johnson_transformer.transform(
                    y.values.reshape(-1, 1)
                ).ravel()

                # Defensive check: Ensure lambda_param is available
                if self.target_transformation.lambda_param is None:
                    lambda_param = float(self.yeo_johnson_transformer.lambdas_[0])
                    self.target_transformation.lambda_param = lambda_param
                    logger.warning(
                        f"Lambda parameter was missing, extracted from transformer: {lambda_param:.6f}"
                    )

            return pd.Series(y_transformed, index=y.index, name=y.name)

        elif method == "log1p":
            logger.info("Applying log1p transformation to target")

            if fit:
                self.target_min_ = float(y.min())
                self.target_max_ = float(y.max())

                # Add buffer for unseen values
                if add_buffer:
                    self.y_min_safe = float(self.target_min_ * (1 - buffer_pct))
                    self.y_max_safe = float(self.target_max_ * (1 + buffer_pct))
                    logger.info(
                        f"Added {buffer_pct*100}% buffer: "
                        f"safe range ({self.y_min_safe:.2f}, {self.y_max_safe:.2f})"
                    )

                y_transformed = np.log1p(y)

                self.transformed_min_ = float(y_transformed.min())
                self.transformed_max_ = float(y_transformed.max())

                self.transformed_mean_ = float(y_transformed.mean())
                self.transformed_std_ = float(y_transformed.std())
                self.clip_lower_ = float(self.transformed_mean_ - 3.0 * self.transformed_std_)
                self.clip_upper_ = float(self.transformed_mean_ + 3.0 * self.transformed_std_)

                y_range = original_range or (self.target_min_, self.target_max_)

                self.target_transformation = TargetTransformation(
                    method="log1p", original_range=y_range
                )

                logger.info(
                    f"Stored original target range: ({self.target_min_:.2f}, {self.target_max_:.2f})"
                )
                logger.info(
                    f"Stored transformed range: ({self.transformed_min_:.4f}, {self.transformed_max_:.4f})"
                )
            else:
                y_transformed = np.log1p(y)

            return pd.Series(y_transformed, index=y.index, name=y.name)

        elif method == "boxcox":
            logger.info("Applying Box-Cox transformation to target")

            if (y <= 0).any():
                raise ValueError(
                    f"Box-Cox requires positive values. "
                    f"Found {(y <= 0).sum()} non-positive values in target"
                )

            if fit:
                self.target_min_ = float(y.min())
                self.target_max_ = float(y.max())

                # Add buffer for unseen values
                if add_buffer:
                    self.y_min_safe = float(self.target_min_ * (1 - buffer_pct))
                    self.y_max_safe = float(self.target_max_ * (1 + buffer_pct))
                    logger.info(
                        f"Added {buffer_pct*100}% buffer: "
                        f"safe range ({self.y_min_safe:.2f}, {self.y_max_safe:.2f})"
                    )

                y_transformed, lambda_val = stats.boxcox(y)

                self.transformed_min_ = float(y_transformed.min())
                self.transformed_max_ = float(y_transformed.max())

                logger.info(f"Box-Cox lambda: {lambda_val:.6f}")

                y_range = original_range or (self.target_min_, self.target_max_)

                # ── F-05 FIX: Write canonical lambda_param AND deprecated boxcox_lambda ──
                # lambda_param is the canonical field per shared.py TargetTransformation.
                # boxcox_lambda is the deprecated alias kept for backward-compat with
                # models serialized before this change. Both must be set so that:
                #   - new code reading lambda_param gets a value
                #   - old loaded models reading boxcox_lambda still work
                self.target_transformation = TargetTransformation(
                    method="boxcox",
                    lambda_param=float(lambda_val),  # canonical — use this going forward
                    boxcox_lambda=float(lambda_val),  # deprecated — remove after next full retrain
                    original_range=y_range,
                    boxcox_min=self.transformed_min_,
                    boxcox_max=self.transformed_max_,
                )

                logger.info(
                    f"Stored original target range: ({self.target_min_:.2f}, {self.target_max_:.2f})"
                )
                logger.info(
                    f"Stored transformed range: ({self.transformed_min_:.4f}, {self.transformed_max_:.4f})"
                )
            else:
                if self.target_transformation.boxcox_lambda is None:
                    raise RuntimeError(
                        "Box-Cox transformation not fitted. Call with fit=True first."
                    )
                lambda_val = self.target_transformation.boxcox_lambda
                y_transformed = stats.boxcox(y, lmbda=lambda_val)

            return pd.Series(y_transformed, index=y.index, name=y.name)

        else:
            raise ValueError(
                f"Unknown transformation method: '{method}'. "
                f"Supported: ['none', 'log1p', 'yeo-johnson', 'boxcox']"
            )

    def _log_prediction_diagnostics(
        self, y_original: np.ndarray, context: str = "prediction"
    ) -> None:
        if not hasattr(self, "target_min_") or self.target_min_ is None:
            return

        n_total = len(y_original)

        # Check predictions below training minimum
        below_min = np.sum(y_original < self.target_min_)
        if below_min > 0:
            pct = (below_min / n_total) * 100
            min_pred = np.min(y_original)
            logger.info(
                f"📊 {context.capitalize()}: {below_min}/{n_total} ({pct:.1f}%) "
                f"predictions below training min\n"
                f"   Training min: ${self.target_min_:,.0f}\n"
                f"   Prediction min: ${min_pred:,.0f}"
            )

        # Check predictions above training maximum
        above_max = np.sum(y_original > self.target_max_)
        if above_max > 0:
            pct = (above_max / n_total) * 100
            max_pred = np.max(y_original)
            extrapolation_pct = ((max_pred - self.target_max_) / self.target_max_) * 100

            if extrapolation_pct > 20:
                logger.warning(
                    f"⚠️ {context.capitalize()}: HIGH EXTRAPOLATION DETECTED\n"
                    f"   {above_max}/{n_total} ({pct:.1f}%) predictions exceed training max\n"
                    f"   Training max: ${self.target_max_:,.0f}\n"
                    f"   Prediction max: ${max_pred:,.0f} (+{extrapolation_pct:.1f}%)\n"
                    f"   → Consider retraining with more high-value samples"
                )
            else:
                logger.info(
                    f"✅ {context.capitalize()}: {above_max}/{n_total} ({pct:.1f}%) "
                    f"predictions extend {extrapolation_pct:.1f}% beyond training max"
                )

        # Check safe buffer usage
        if hasattr(self, "y_max_safe") and self.y_max_safe is not None:
            exceeded_buffer = np.sum(y_original > self.y_max_safe)
            if exceeded_buffer > 0:
                pct = (exceeded_buffer / n_total) * 100
                logger.error(
                    f"🚨 {context.capitalize()}: BUFFER EXCEEDED\n"
                    f"   {exceeded_buffer}/{n_total} ({pct:.1f}%) predictions exceed safe buffer\n"
                    f"   Safe max: ${self.y_max_safe:,.0f}\n"
                    f"   Max prediction: ${np.max(y_original):,.0f}\n"
                    f"   ⚠️ This indicates severe extrapolation - model may be unreliable"
                )

        # Summary statistics
        logger.debug(
            f"Prediction summary ({context}):\n"
            f"  Range: [${np.min(y_original):,.0f}, ${np.max(y_original):,.0f}]\n"
            f"  Mean: ${np.mean(y_original):,.0f}\n"
            f"  Median: ${np.median(y_original):,.0f}\n"
            f"  Std: ${np.std(y_original):,.0f}"
        )

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating base features...")

        # STRICT VALIDATION (using config values)
        required_cols = ["age", "bmi", "smoker", "children"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"❌ Missing required columns: {missing_cols}\n"
                f"   Feature engineering requires: {required_cols}"
            )

        df = df.copy()
        original_count = len(df)
        original_index = df.index.copy()

        # Validate smoker values using config encoding
        valid_smoker_values = set(self.config.smoker_binary_map.keys())
        invalid_smoker = ~df["smoker"].isin(valid_smoker_values)
        if invalid_smoker.any():
            invalid_values = df.loc[invalid_smoker, "smoker"].unique()
            raise ValueError(
                f"❌ Invalid smoker values: {invalid_values}\n"
                f"   Expected one of: {valid_smoker_values}\n"
                f"   (from config.yaml features.encoding.smoker_binary_map)"
            )

        # Validate BMI range (from config)
        if df["bmi"].isna().any():
            raise ValueError(f"❌ Found {df['bmi'].isna().sum()} NaN values in BMI")
        # ── BUG 5 FIX ──────────────────────────────────────────────────────
        # Original: (df["bmi"] <= self.config.bmi_min).any()
        #   This rejects BMI == bmi_min (e.g. bmi_min=10.0 would reject BMI=10.0).
        #   predict.py uses strict-less-than for the same check, and the error
        #   message printed the range as "(bmi_min, bmi_max]" (half-open).
        #
        # Fix: use strict-less-than so bmi_min itself is a valid value,
        # making the accepted range [bmi_min, bmi_max] (closed both ends).
        # This is consistent with predict.py and avoids rejecting valid edge cases.
        if (df["bmi"] < self.config.bmi_min).any() or (df["bmi"] > self.config.bmi_max).any():
            raise ValueError(
                f"❌ BMI out of valid range [{self.config.bmi_min}, {self.config.bmi_max}]\n"
                f"   (from config.yaml features.bmi_min/bmi_max)"
            )

        # Validate age range (from config)
        if df["age"].isna().any():
            raise ValueError(f"❌ Found {df['age'].isna().sum()} NaN values in age")
        if (df["age"] < self.config.age_min).any() or (df["age"] > self.config.age_max).any():
            raise ValueError(
                f"❌ Age out of valid range [{self.config.age_min}-{self.config.age_max}]\n"
                f"   (from config.yaml features.age_min/age_max)"
            )

        # Validate children
        if df["children"].isna().any():
            raise ValueError(f"❌ Found {df['children'].isna().sum()} NaN values in children")

        # BMI categories
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 35, float("inf")],
            labels=["underweight", "normal", "overweight", "obese_1", "obese_2"],
        )

        # Age groups
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 65, float("inf")],
            labels=["young", "adult", "middle_aged", "senior", "elderly"],
        )

        # Binary mappings (ONCE - use config encoding throughout)
        smoker_binary = df["smoker"].map(self.config.smoker_binary_map)
        non_smoker = 1 - smoker_binary
        df["smoker"].map(self.config.smoker_risk_map)

        # Risk scores
        bmi_risk = pd.cut(df["bmi"], bins=[0, 25, 30, float("inf")], labels=[0, 1, 2]).astype(int)
        age_risk = pd.cut(df["age"], bins=[0, 40, 60, float("inf")], labels=[0, 1, 2]).astype(int)
        df["unified_risk_score"] = (
            (df["age"] / 64) * 0.2  # Age component
            + (df["bmi"] / 50) * 0.2  # BMI component
            + smoker_binary * 0.4  # Smoking (highest weight)
            + (df["children"] / 5) * 0.1  # Family size
            + ((df["bmi"] > 35).astype(int)) * 0.1  # Extreme obesity
        )

        # Base interactions
        df["smoker_bmi"] = smoker_binary * df["bmi"]
        df["age_bmi"] = df["age"] * df["bmi"]
        df["smoker_age"] = smoker_binary * df["age"]

        # Additional features
        df["family_size"] = df["children"] + 1
        df["bmi_squared"] = df["bmi"] ** 2
        df["age_decade"] = (df["age"] // 10) * 10
        df["is_senior"] = (df["age"] >= 65).astype(int)
        df["high_bmi"] = (df["bmi"] > 30).astype(int)
        df["smoker_senior"] = smoker_binary * df["is_senior"]
        # SMOKER_CHILDREN FIX (Issue D): Without the +1 shift, any smoker with
        # children=0 gets smoker_children=0 — the same value as every non-smoker.
        # This nullifies the smoker signal for ~35% of smokers who have no children
        # and is the primary driver of Sample 22's $17 k miss (actual=$30,064,
        # pred=$13,031).  The shift maps: non-smoker→0, childless smoker→1,
        # smoker+1child→2, etc., preserving separability for all smoker groups.
        df["smoker_children"] = smoker_binary * (df["children"] + 1)
        df["age_squared"] = df["age"] ** 2
        df["bmi_age_interaction"] = (df["bmi"] - 25) * (df["age"] - 40)
        df["is_high_risk"] = ((df["bmi"] > 30) & (smoker_binary == 1) & (df["age"] > 50)).astype(
            int
        )

        # ============================================================
        # OPTIMIZED HIGH-VALUE FEATURES
        # ============================================================

        # ============================================================
        # 1. COMPOUND RISK (Single best version)
        # ============================================================

        # Multiplicative risk (KEEP - best predictor)
        df["compound_risk_multiplicative"] = smoker_binary * (df["bmi"] / 30.0) * (df["age"] / 50.0)

        # ============================================================
        # 2. EXTREME FLAGS (Single consolidated flag)
        # ============================================================

        # Combined extreme risk (replaces 3 redundant flags)
        df["extreme_risk_flag"] = (
            (smoker_binary == 1) & (df["bmi"] > 30) & (df["age"] > 50)
        ).astype(int) * (
            1
            + (df["bmi"] > 35).astype(int)  # Bonus for very obese
            + (df["age"] > 60).astype(int)  # Bonus for elderly
        )

        # ============================================================
        # 3. NON-LINEAR BMI (Single best interaction)
        # ============================================================

        # Quadratic BMI for smokers (KEEP - captures non-linearity)
        df["smoker_bmi_squared"] = smoker_binary * (df["bmi"] ** 2) / 1000.0

        # ============================================================
        # 4. AGE PROGRESSION (Single exponential version)
        # ============================================================

        # Exponential age effect for smokers (KEEP - unique curve)
        df["smoker_age_exponential"] = smoker_binary * np.exp((df["age"] - 40) / 20.0)

        # ============================================================
        # 5. THREE-WAY INTERACTION (Single comprehensive version)
        # ============================================================

        # Age × BMI × Smoker (KEEP - captures full interaction)
        df["age_bmi_smoker_interaction"] = df["age"] * df["bmi"] * smoker_binary / 1000.0

        # ============================================================
        # 6. THRESHOLD FLAGS (Two non-overlapping)
        # ============================================================

        # Senior + obese + smoker (high-risk combo)
        df["senior_obese_smoker"] = (
            (df["age"] > 55) & (df["bmi"] > 30) & (smoker_binary == 1)
        ).astype(int)

        # Very obese senior (non-smoker high-cost proxy)
        df["very_obese_senior"] = (
            (df["bmi"] > 35) & (df["age"] > 55) & (smoker_binary == 0)  # Only for non-smokers
        ).astype(int)

        # ============================================================
        # 7. NON-SMOKER SPECIFIC (Two orthogonal features)
        # ============================================================

        # Compound risk for non-smokers
        # ML-07 FIX: clip each factor to [0, ∞) before multiplying.
        # For age < 25: (age-25)/50 < 0; for BMI < 18: (bmi-18)/35 < 0.
        # A negative × positive product yields a negative "risk", which is
        # semantically invalid and confuses both tree and linear models.
        _age_factor = np.clip((df["age"] - 25) / 50, 0, None)
        _bmi_factor = np.clip((df["bmi"] - 18) / 35, 0, None)
        df["nonsmoker_compound_risk"] = (
            non_smoker * _age_factor * _bmi_factor * (1 + df["children"] * 0.3)
        )

        # Chronic condition proxy (age + BMI for non-smokers)
        df["nonsmoker_chronic_proxy"] = (
            non_smoker
            * ((df["age"] >= 45) & (df["bmi"] > 30)).astype(int)
            * df["age"]
            * df["bmi"]
            / 1000
        )

        # ============================================================
        # 8. AGE-BMI CATEGORIES (Two orthogonal bins)
        # ============================================================

        # BMI severity score (insurance actuarial tiers)
        df["bmi_severity_score"] = pd.cut(
            df["bmi"], bins=[0, 18.5, 25, 30, 35, 40, 100], labels=[1, 2, 4, 7, 11, 16]
        ).astype(int)

        # Age risk tier (decade-based)
        df["age_risk_tier"] = pd.cut(
            df["age"], bins=[0, 30, 40, 50, 60, 70, 100], labels=[1, 2, 4, 7, 11, 15]
        ).astype(int)

        # ============================================================
        # 9. FAMILY RISK (If children exists)
        # ============================================================

        if "children" in df.columns:
            # Family burden multiplier
            df["family_health_burden"] = (
                df["children"]
                * (1 + (df["bmi"] > 30).astype(int) * 0.5)
                * (1 + smoker_binary * 0.3)
            )

        # Additional high-value predictors
        df["chronic_condition_proxy"] = (
            ((df["age"] > 45) & (df["bmi"] > 30)).astype(int) * 3
            + ((df["age"] > 55) & (df["bmi"] > 32)).astype(int) * 5
            + ((df["age"] > 60) & (df["bmi"] > 28)).astype(int) * 4
        )

        df["age_bmi_exp"] = np.exp(0.01 * (df["age"] - 40) * (df["bmi"] - 25)).clip(0, 10)

        df["smoker_adjusted_bmi"] = df["bmi"] * (1 + smoker_binary * 0.5)

        df["hidden_health_risk"] = (
            (df["age"] > 50).astype(int) * (df["bmi"] > 32).astype(int) * non_smoker * 3
        )

        # ============================================================
        # 10. LOW-RISK PROFILES (Features)
        # ============================================================

        # Flag for "minimal risk" profiles
        df["minimal_risk_profile"] = (
            (df["age"] < 30) & (df["bmi"] < 25) & (smoker_binary == 0) & (df["children"] == 0)
        ).astype(int)

        # Age-based cost floor (younger = lower minimum)
        # PA-04 FIX: Read dollar thresholds from config instead of hardcoding.
        # Hardcoded values (1500, 2500, 4000) violate SSOT and drift with inflation.
        # Config path: features.premium_tier.age_cost_floors / age_breakpoints.
        # Falls back to original values if not configured (backward-compatible).
        _tier_cfg = {}
        if hasattr(self, "config") and self.config is not None:
            _raw_cfg = getattr(self.config, "_raw_config", {}) or {}
            _tier_cfg = _raw_cfg.get("features", {}).get("premium_tier", {})
        _age_floors = _tier_cfg.get("age_cost_floors", [1500, 2500, 4000])
        _age_breaks = _tier_cfg.get("age_breakpoints", [30, 40])
        # Defensive defaults in case config lists are shorter than expected
        _f0 = _age_floors[0] if len(_age_floors) > 0 else 1500
        _f1 = _age_floors[1] if len(_age_floors) > 1 else 2500
        _f2 = _age_floors[2] if len(_age_floors) > 2 else 4000
        _b0 = _age_breaks[0] if len(_age_breaks) > 0 else 30
        _b1 = _age_breaks[1] if len(_age_breaks) > 1 else 40
        df["age_cost_floor"] = np.where(
            df["age"] < _b0,
            _f0,
            np.where(df["age"] < _b1, _f1, _f2),
        )

        # Non-smoker youth bonus
        df["youth_nonsmoker_discount"] = (
            ((df["age"] < 35) & (smoker_binary == 0)).astype(int) * (35 - df["age"]) / 35
        )

        # ============================================================
        # HIGH-VALUE PREDICTION ENHANCEMENTS
        # ============================================================

        # ------------------------------------------------------------
        # 1. EXTREME SMOKER–OBESITY–AGE INTERACTION (bounded)
        # ------------------------------------------------------------
        df["extreme_health_burden"] = (
            smoker_binary * np.maximum(df["bmi"] - 32, 0) * np.maximum(df["age"] - 50, 0) / 500
        )

        # ------------------------------------------------------------
        # 2. HIGH-BMI AGE ACCELERATION (non-exponential)
        # ------------------------------------------------------------
        df["high_bmi_age_acceleration"] = (
            (df["bmi"] > 35).astype(int) * smoker_binary * (df["age"] - 45).clip(lower=0) / 20
        )

        # ------------------------------------------------------------
        # 3. QUADRATIC AGE–BMI STRESS (softened)
        # ------------------------------------------------------------
        df["age_bmi_stress"] = (
            smoker_binary
            * ((df["age"] - 40).clip(lower=0) ** 2)
            * ((df["bmi"] - 28).clip(lower=0) ** 2)
        ) / 5000

        # ------------------------------------------------------------
        # 4. SEVERE RISK FLAG (non-redundant)
        # ------------------------------------------------------------
        df["severe_risk_profile"] = (
            (smoker_binary == 1) & (df["bmi"] > 38) & (df["age"] > 58)
        ).astype(int)

        # ------------------------------------------------------------
        # 5. HIGH-VALUE SEVERITY SCORE (bounded composite)
        # ------------------------------------------------------------
        df["high_value_severity"] = (
            (df["age"] / 70) * 0.25
            + (df["bmi"] / 50) * 0.25
            + smoker_binary * 0.35
            + (df["bmi"] > 35).astype(int) * 0.10
            + (df["age"] > 60).astype(int) * 0.05
        )

        # ------------------------------------------------------------
        # 6. EXTREME BMI × SMOKER × MID-AGE BINARY FLAG
        # ------------------------------------------------------------
        # NEW ISSUE B FIX: The model's worst misses (e.g. Sample 22:
        # actual=$30,064, predicted=$12,538) involve smokers with BMI>35
        # aged 45+.  Existing continuous features such as
        # high_bmi_age_acceleration give a graded signal, but XGBoost can
        # benefit from an explicit binary split point at this exact risk
        # boundary.
        #
        # This flag is DISTINCT from the continuous high_bmi_age_acceleration
        # (which scales with age/20) and from severe_risk_profile
        # (which requires bmi>38 AND age>58 — a much stricter, smaller set).
        # Thresholds chosen to align with the P80-P99 cost range:
        #   bmi>35  → Class II+ obesity (actuarial break-point)
        #   age>45  → Mid-life inflection where smoker cost accelerates
        df["smoker_extreme_bmi_age"] = (
            (smoker_binary == 1) & (df["bmi"] > 35) & (df["age"] > 45)
        ).astype(int)

        logger.info("   ✅ Added 6 calibration-safe high-value enhancement features")

        # VALIDATION
        if len(df) != original_count:
            raise ValueError(f"Row count changed: {original_count} -> {len(df)}")
        if not df.index.equals(original_index):
            raise ValueError("Index alignment lost during feature creation")

        self._update_state(PipelineState.FEATURES_CREATED)

        logger.info(f"Created base features. Shape: {df.shape}")
        logger.info(f"   ✅ Added 15 optimized high-value features (non-redundant)")
        logger.info(f"   ✅ Added 3 low-risk profile features for improved predictions")
        logger.info(f"   ✅ Used config.yaml encoding maps for smoker features")

        return df

    def impute_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Domain-aware imputation - ALWAYS creates imputers during fit"""
        if fit:
            self._validate_state(PipelineState.FEATURES_CREATED, "impute_features")

        df_imputed = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        demographic_features = [col for col in numeric_cols if col in ["age", "bmi", "children"]]
        derived_features = [col for col in numeric_cols if col not in demographic_features]

        if fit:
            if demographic_features:
                self.demographic_imputer = SimpleImputer(strategy="median")
                df_imputed[demographic_features] = self.demographic_imputer.fit_transform(
                    df[demographic_features]
                )
                null_count = df[demographic_features].isna().sum().sum()
                logger.debug(
                    f"Fitted demographic imputer on {len(demographic_features)} features ({null_count} nulls)"
                )

            if derived_features:
                self.derived_imputer = SimpleImputer(strategy="mean")
                df_imputed[derived_features] = self.derived_imputer.fit_transform(
                    df[derived_features]
                )
                null_count = df[derived_features].isna().sum().sum()
                logger.debug(
                    f"Fitted derived imputer on {len(derived_features)} features ({null_count} nulls)"
                )

            if categorical_cols:
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
                df_imputed[categorical_cols] = self.categorical_imputer.fit_transform(
                    df[categorical_cols]
                )
                null_count = df[categorical_cols].isna().sum().sum()
                logger.debug(
                    f"Fitted categorical imputer on {len(categorical_cols)} features ({null_count} nulls)"
                )

            self._update_state(PipelineState.IMPUTED)
        else:
            if demographic_features and self.demographic_imputer:
                df_imputed[demographic_features] = self.demographic_imputer.transform(
                    df[demographic_features]
                )
            if derived_features and self.derived_imputer:
                df_imputed[derived_features] = self.derived_imputer.transform(df[derived_features])
            if categorical_cols and self.categorical_imputer:
                df_imputed[categorical_cols] = self.categorical_imputer.transform(
                    df[categorical_cols]
                )

        return df_imputed

    def detect_and_remove_outliers(
        self,
        df: pd.DataFrame,
        y: pd.Series | None = None,
        method: str = "isolation_forest",
        fit: bool = True,
        return_mask: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series | None, np.ndarray | None]:
        if fit:
            self._validate_state(PipelineState.ENCODED, "outlier detection")

        if method == "none":
            self.outlier_method = "none"
            self._update_state(PipelineState.OUTLIERS_DETECTED)
            return df, y, None

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns for outlier detection")
            self._update_state(PipelineState.OUTLIERS_DETECTED)
            return df, y, None

        if fit:
            if method == "isolation_forest":
                self.outlier_detector = IsolationForest(
                    contamination=self.config.outlier_contamination,
                    random_state=self.config.outlier_random_state,
                    n_jobs=-1,
                )
                self._outlier_numeric_features = numeric_cols
                outlier_labels = self.outlier_detector.fit_predict(df[numeric_cols])
                outlier_mask = outlier_labels == -1
                self.outlier_method = method
                self._outlier_indices = df.index[outlier_mask].values

                logger.info(
                    f"IsolationForest detected {outlier_mask.sum()} outliers "
                    f"({outlier_mask.sum()/len(df)*100:.2f}%)"
                )
            else:
                raise ValueError(
                    f"Unknown outlier method: {method}. Supported: ['isolation_forest', 'none']"
                )

            self._update_state(PipelineState.OUTLIERS_DETECTED)
        else:
            if self.outlier_detector is None:
                return df, y, None

            # ── BUG 2 FIX ────────────────────────────────────────────────────
            # Original code: self.outlier_detector.predict(df[numeric_cols])
            #   numeric_cols = df.select_dtypes(include=[np.number]).columns
            #   i.e. derived from the *current* DataFrame.
            #
            # Problem: at fit time the IsolationForest was trained on exactly
            # self._outlier_numeric_features (which reflects the feature set
            # AFTER create_features() but BEFORE encoding).  If the incoming
            # df has gained or lost numeric columns (e.g. via one-hot expansion
            # or variance removal), predict() would receive the wrong number of
            # features and silently produce garbage or raise a ValueError.
            #
            # Fix: use the stored fit-time feature list.  Fall back to
            # numeric_cols only when the stored list is absent (legacy artifact).
            if hasattr(self, "_outlier_numeric_features") and self._outlier_numeric_features:
                _outlier_cols = [c for c in self._outlier_numeric_features if c in df.columns]
                missing_outlier_cols = [
                    c for c in self._outlier_numeric_features if c not in df.columns
                ]
                if missing_outlier_cols:
                    logger.warning(
                        f"⚠️ Outlier detector: {len(missing_outlier_cols)} fit-time features "
                        f"missing from transform input: {missing_outlier_cols}\n"
                        f"   Falling back to available numeric columns."
                    )
                    _outlier_cols = numeric_cols
            else:
                # Legacy: _outlier_numeric_features not stored (pre-v4.3.2 artifact)
                logger.warning(
                    "⚠️ _outlier_numeric_features not stored; using current numeric columns. "
                    "Retrain preprocessor to fix this."
                )
                _outlier_cols = numeric_cols

            outlier_labels = self.outlier_detector.predict(df[_outlier_cols])
            outlier_mask = outlier_labels == -1
            logger.debug(f"Detected {outlier_mask.sum()} outliers in test/validation data")

        df_clean = df[~outlier_mask].copy()
        y_clean = y[~outlier_mask].copy() if y is not None else None

        result_mask = outlier_mask if return_mask else None

        logger.info(f"Removed {outlier_mask.sum()} outliers. Shape: {len(df)} -> {len(df_clean)}")
        return df_clean, y_clean, result_mask

    def remove_low_variance(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Remove zero or low variance features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            logger.debug("No numeric columns for variance filtering")
            return df

        df_result = df.copy()

        if fit:
            self.variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
            try:
                self.variance_selector.fit(df[numeric_cols])
                selected_mask = self.variance_selector.get_support()
                selected_features = [
                    col for col, selected in zip(numeric_cols, selected_mask) if selected
                ]
                self._variance_feature_names_snapshot = tuple(selected_features)

                dropped_features = [
                    col for col, selected in zip(numeric_cols, selected_mask) if not selected
                ]
                if dropped_features:
                    df_result = df_result.drop(columns=dropped_features)
                    logger.info(
                        f"Removed {len(dropped_features)} low-variance features "
                        f"(threshold={self.config.variance_threshold})"
                    )
            except (ValueError, TypeError, LinAlgError) as e:
                logger.error(f"Variance filtering failed: {e}", exc_info=True)
                raise RuntimeError(f"Variance filtering failed: {e}") from e
        else:
            if self.variance_selector is None:
                raise RuntimeError("VarianceThreshold not fitted. Call with fit=True first.")

            dropped_features = [
                col for col in numeric_cols if col not in self._variance_feature_names_snapshot
            ]
            if dropped_features:
                df_result = df_result.drop(columns=dropped_features)
                logger.debug(f"Removed {len(dropped_features)} low-variance features")

        return df_result

    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if fit:
            self._validate_state(PipelineState.IMPUTED, "encoding")

        logger.info("Encoding features...")
        df_encoded = df.copy()
        original_index = df.index.copy()
        original_count = len(df)

        # ========================================
        # STEP 1: Label Encoding for Binary Columns
        # ========================================
        binary_cols = ["sex", "smoker"]

        for col in binary_cols:
            if col not in df.columns:
                continue

            if fit:
                # FIT: Create encoder and store most frequent value
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))

                # Store most frequent for unseen category handling
                most_frequent = df[col].mode()[0]
                self.most_frequent_values[col] = most_frequent
                self.label_encoders[col] = le

                logger.info(
                    f"✓ Fitted LabelEncoder for '{col}': "
                    f"{list(le.classes_)} -> [{df_encoded[col].min()}, {df_encoded[col].max()}]"
                )
            else:
                # TRANSFORM: Handle unseen categories
                if col not in self.label_encoders:
                    raise RuntimeError(
                        f"LabelEncoder for '{col}' not fitted. Call with fit=True first."
                    )

                le = self.label_encoders[col]
                col_values = df[col].astype(str)
                mask = col_values.isin(le.classes_)

                if not mask.all():
                    unseen = col_values[~mask].unique()
                    default_value = self.most_frequent_values.get(col, le.classes_[0])
                    logger.warning(
                        f"⚠ Unseen categories in '{col}': {unseen}. "
                        f"Mapping to '{default_value}'"
                    )
                    # Replace unseen values with default BEFORE encoding
                    col_values = col_values.where(mask, default_value)

                df_encoded[col] = le.transform(col_values)

        # ========================================
        # STEP 2: One-Hot Encoding for Multi-Class Columns
        # ========================================
        cat_cols = ["region", "bmi_category", "age_group"]
        existing_cols = [col for col in cat_cols if col in df.columns]

        if existing_cols:
            if fit:
                # FIT: Create one-hot encoder
                encoded_features = self.onehot_encoder.fit_transform(df[existing_cols].astype(str))
                feature_names = self.onehot_encoder.get_feature_names_out(existing_cols)
                self._feature_names_snapshot = tuple(feature_names)

                logger.info(
                    f"✓ Fitted OneHotEncoder for {existing_cols}: "
                    f"{len(feature_names)} features created"
                )
            else:
                # TRANSFORM: Use fitted encoder (handles unseen via 'ignore')
                if self._feature_names_snapshot is None:
                    raise RuntimeError("OneHotEncoder not fitted. Call with fit=True first.")

                encoded_features = self.onehot_encoder.transform(df[existing_cols].astype(str))
                feature_names = self._feature_names_snapshot

            # Convert to DataFrame and concatenate
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
            df_encoded = pd.concat([df_encoded.drop(columns=existing_cols), encoded_df], axis=1)
            logger.debug(f"  Added {len(feature_names)} one-hot features")

        # ========================================
        # STEP 3: CRITICAL VALIDATION
        # ========================================

        # Check 1: Row count preserved
        if len(df_encoded) != original_count:
            raise ValueError(f"❌ Row count changed: {original_count} -> {len(df_encoded)}")

        # Check 2: Index alignment preserved
        if not df_encoded.index.equals(original_index):
            raise ValueError("❌ Index alignment lost during encoding")

        # Check 3: No NaN values introduced
        if df_encoded.isna().any().any():
            null_cols = df_encoded.columns[df_encoded.isna().any()].tolist()
            raise ValueError(f"❌ NaN values introduced in columns: {null_cols}")

        # Check 4: No categorical columns remain
        remaining_cats = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()

        if remaining_cats:
            raise ValueError(
                f"❌ ENCODING INCOMPLETE: Categorical columns still present: {remaining_cats}\n"
                f"   Expected to encode: {binary_cols + existing_cols}"
            )

        # Check 5: All encoded values are numeric
        non_numeric = df_encoded.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            raise ValueError(f"❌ Non-numeric columns remain: {non_numeric}")

        logger.info(
            f"✅ Encoding complete: {len(binary_cols)} binary + "
            f"{len(existing_cols)} one-hot encoded columns"
        )

        if fit:
            self._update_state(PipelineState.ENCODED)

        return df_encoded

    def _calculate_vif_optimized(self, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
        if X.shape[1] == 0:
            return pd.DataFrame(columns=["Feature", "VIF"])

        vif_values = []
        for i in range(X.shape[1]):
            X_i = X[:, i]
            X_not_i = np.delete(X, i, axis=1)

            if X_not_i.shape[1] == 0:
                vif_values.append(np.inf)
                continue

            try:
                model = LinearRegression(fit_intercept=False)
                model.fit(X_not_i, X_i)
                r_squared = model.score(X_not_i, X_i)

                # Handle numerical edge cases
                if r_squared >= 0.9999:
                    vif = np.inf
                elif r_squared < 0:
                    vif = 1.0  # Negative R² means no correlation
                else:
                    vif = 1.0 / (1.0 - r_squared)

                vif_values.append(vif)
            except (ValueError, LinAlgError) as e:
                logger.warning(f"VIF calculation failed for feature {feature_names[i]}: {e}")
                vif_values.append(np.inf)

        return pd.DataFrame({"Feature": feature_names, "VIF": vif_values}).sort_values(
            "VIF", ascending=False
        )

    def remove_multicollinear_features(
        self, df: pd.DataFrame, y: pd.Series | None = None, fit: bool = True
    ) -> pd.DataFrame:
        # PROTECTED_FEATURES: Never removed by VIF or correlation pruning.
        # Core demographic features are protected by design.
        # smoker_extreme_bmi_age: triple-interaction (smoker & bmi>35 & age>45) —
        #   was consistently removed by VIF due to collinearity with
        #   high_bmi_age_acceleration. Domain knowledge warrants keeping it.
        # smoker_children: 29.3% SHAP importance (run 4 top feature) — must
        #   survive VIF even after the +1 shift changes its correlation profile.
        # ── BUG-4 FIX (v7.5.0) ────────────────────────────────────────────
        # age_severity_interaction and smoker_adjusted_bmi are critical for the
        # high-value segment (R² = -2.15 without them). VIF at threshold=5 was
        # stripping both due to collinearity with nonsmoker_age_bmi_quad and bmi.
        # Domain knowledge overrides: these features encode distinct non-linear
        # risk signals that XGBoost cannot recover from other retained features.
        #   age_severity_interaction  : age × bmi_severity_score — captures the
        #     compounding effect of advancing age + elevated BMI, concentrated in
        #     the high-value boundary ($16k–$48k). Removing it causes the model
        #     to underfit that segment severely (RMSE ↑ 3.2x, R² → negative).
        #   smoker_adjusted_bmi       : bmi × (1 + smoker×0.5) — amplifies BMI
        #     signal for smokers; the amplification is non-redundant with 'bmi'
        #     and 'smoker' individually. Verified necessary for smoker high-value
        #     segment precision (overpricing rate ↓ 12pp when retained).
        PROTECTED_FEATURES = {
            "smoker",
            "age",
            "bmi",
            "sex",
            "smoker_extreme_bmi_age",
            "smoker_children",
            "age_severity_interaction",  # BUG-4 FIX: critical for high-value R²
            "smoker_adjusted_bmi",  # BUG-4 FIX: non-redundant smoker × BMI signal
            "age_bmi_stress",
        }

        df_result = df.copy()

        if fit:
            removed_features: list[str] = []

            # Remove known redundant features
            if "children" in df.columns and "family_size" in df.columns:
                df_result = df_result.drop(columns=["children"])
                removed_features.append("children")
                logger.debug("Removed 'children' (redundant with 'family_size')")

            # Correlation-based removal
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in PROTECTED_FEATURES]

            if len(numeric_cols) > 1:
                corr_matrix = df_result[numeric_cols].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                correlated_pairs = [
                    (column, row, upper_tri.loc[row, column])
                    for column in upper_tri.columns
                    for row in upper_tri.index
                    if upper_tri.loc[row, column] > self.config.correlation_threshold
                ]

                if correlated_pairs:
                    to_drop = set()

                    for col1, col2, corr_value in correlated_pairs:
                        if col1 not in to_drop and col2 not in to_drop:
                            if y is not None:
                                target_corr1 = abs(y.corr(df_result[col1]))
                                target_corr2 = abs(y.corr(df_result[col2]))

                                if target_corr1 >= target_corr2:
                                    to_drop.add(col2)
                                    logger.debug(
                                        f"Corr {corr_value:.3f}: Keeping {col1} "
                                        f"(target_corr={target_corr1:.3f}), "
                                        f"dropping {col2} (target_corr={target_corr2:.3f})"
                                    )
                                else:
                                    to_drop.add(col1)
                                    logger.debug(
                                        f"Corr {corr_value:.3f}: Keeping {col2} "
                                        f"(target_corr={target_corr2:.3f}), "
                                        f"dropping {col1} (target_corr={target_corr1:.3f})"
                                    )
                            else:
                                var1 = df_result[col1].var()
                                var2 = df_result[col2].var()

                                if var1 >= var2:
                                    to_drop.add(col2)
                                else:
                                    to_drop.add(col1)

                    to_drop_list = list(to_drop)
                    if to_drop_list:
                        df_result = df_result.drop(columns=to_drop_list)
                        removed_features.extend(to_drop_list)
                        logger.info(
                            f"Correlation removal: {len(to_drop_list)} features "
                            f"(threshold={self.config.correlation_threshold})"
                        )

            # VIF-based removal
            try:

                def calculate_vif(df_vif: pd.DataFrame) -> pd.DataFrame:
                    """Calculate VIF for continuous base features"""
                    num_feats = df_vif.select_dtypes(include=[np.number])
                    if num_feats.shape[1] <= 1:
                        return pd.DataFrame()

                    onehot_features = (
                        list(self._feature_names_snapshot) if self._feature_names_snapshot else []
                    )
                    poly_features = (
                        list(self._poly_feature_names_snapshot)
                        if self._poly_feature_names_snapshot
                        else []
                    )

                    vif_features = [
                        col
                        for col in num_feats.columns
                        if col not in PROTECTED_FEATURES
                        and col not in onehot_features
                        and col not in poly_features
                        and num_feats[col].nunique() > 2
                    ]

                    if len(vif_features) <= 1:
                        return pd.DataFrame()

                    # Use optimized or standard VIF calculation
                    if self.config.use_optimized_vif:
                        return self._calculate_vif_optimized(
                            num_feats[vif_features].values, vif_features
                        )
                    else:
                        try:
                            vif_data = pd.DataFrame(
                                {
                                    "Feature": vif_features,
                                    "VIF": [
                                        variance_inflation_factor(num_feats[vif_features].values, i)
                                        for i in range(len(vif_features))
                                    ],
                                }
                            )
                            vif_data = vif_data[np.isfinite(vif_data["VIF"])]
                            return vif_data.sort_values("VIF", ascending=False)
                        except (ValueError, LinAlgError) as e:
                            logger.warning(f"VIF calculation numerical error: {e}")
                            return pd.DataFrame()

                vif_removed = []
                for iteration in range(self.config.max_vif_iterations):
                    vif_data = calculate_vif(df_result)

                    if vif_data.empty:
                        logger.debug("VIF calculation complete (no features to analyze)")
                        break

                    max_vif_row = vif_data.iloc[0]
                    max_vif_feature = max_vif_row["Feature"]
                    max_vif_value = max_vif_row["VIF"]

                    if max_vif_value > self.config.vif_threshold:
                        logger.info(
                            f"VIF Iter {iteration+1}: Removing '{max_vif_feature}' "
                            f"(VIF={max_vif_value:.1f})"
                        )
                        df_result = df_result.drop(columns=[max_vif_feature])
                        vif_removed.append(max_vif_feature)
                        removed_features.append(max_vif_feature)
                    else:
                        logger.info(f"VIF converged: Max VIF={max_vif_value:.1f}")
                        break

                if vif_removed:
                    logger.info(f"VIF removal: {len(vif_removed)} features")

            except Exception as e:
                logger.error(f"VIF analysis failed unexpectedly: {e}", exc_info=True)
                raise RuntimeError(f"VIF analysis failed: {e}") from e

            self._removed_collinear_features_snapshot = tuple(removed_features)

        else:
            if not self._removed_collinear_features_snapshot:
                logger.debug("No collinear features to remove")
                return df_result

            features_to_drop = [
                f for f in self._removed_collinear_features_snapshot if f in df_result.columns
            ]
            if features_to_drop:
                df_result = df_result.drop(columns=features_to_drop)
                logger.debug(f"Removed {len(features_to_drop)} collinear features")

        return df_result

    def _get_continuous_features(self, df: pd.DataFrame) -> list[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        onehot_features = (
            list(self._feature_names_snapshot) if self._feature_names_snapshot is not None else []
        )

        if self._poly_continuous_features_snapshot is not None:
            continuous = [
                col for col in self._poly_continuous_features_snapshot if col in numeric_cols
            ]
            logger.debug(
                f"Using stored continuous features (transform mode): {len(continuous)} features"
            )
            return continuous

        continuous = [
            col
            for col in numeric_cols
            if col not in onehot_features
            and df[col].nunique() > self.config.continuous_feature_threshold
        ]
        logger.debug(f"Identified continuous features (fit mode): {len(continuous)} features")
        return continuous

    def add_domain_interactions(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Add insurance-specific domain interactions (post-encoding)"""
        df_with_interactions = df.copy()

        required_features = ["age", "bmi", "smoker"]
        missing = [f for f in required_features if f not in df.columns]

        if missing:
            if fit:
                raise ValueError(
                    f"Cannot create domain interactions - missing required features: {missing}"
                )
            logger.warning(f"Skipping domain interactions - missing features: {missing}")
            return df_with_interactions

        if fit:
            unique_vals = df["smoker"].unique()
            if not set(unique_vals).issubset({0, 1}):
                raise ValueError(
                    f"Smoker must be encoded (0/1) before domain interactions. "
                    f"Got values: {unique_vals}"
                )

        # ============================================================
        # ESSENTIAL INTERACTIONS ONLY
        # ============================================================

        df_with_interactions["smoker_age_bmi"] = df["smoker"] * df["age"] * df["bmi"]
        df_with_interactions["bmi_age_ratio"] = df["bmi"] / (df["age"] + 1)

        # Non-smoker interactions (consolidated)
        non_smoker = 1 - df["smoker"]

        df_with_interactions["non_smoker_high_bmi_age"] = (
            non_smoker * (df["bmi"] > 30).astype(int) * df["age"]
        )

        df_with_interactions["non_smoker_senior"] = non_smoker * (df["age"] >= 65).astype(int)

        # Senior smoker interaction
        df_with_interactions["smoker_senior_interaction"] = df["smoker"] * (df["age"] >= 65).astype(
            int
        )

        # Extreme BMI flag
        df_with_interactions["extreme_bmi"] = (df["bmi"] > 35).astype(int)

        # Multiple children non-smoker (if children exists)
        if "children" in df.columns:
            df_with_interactions["non_smoker_multi_children"] = non_smoker * (
                df["children"] >= 2
            ).astype(int)

            # Catastrophic risk
            df_with_interactions["potential_catastrophic"] = (
                (df["age"] > 40) & (df["bmi"] > 35) & (df["children"] >= 2)
            ).astype(int)

        # ============================================================
        # ADVANCED INTERACTIONS (High-value specific)
        # ============================================================

        # Quadratic age-BMI for non-smokers
        df_with_interactions["nonsmoker_age_bmi_quad"] = (
            non_smoker * ((df["age"] - 40) ** 2) * ((df["bmi"] - 25) ** 2) / 10000  # Normalized
        )

        # Age-severity interaction
        if "bmi_severity_score" in df.columns:
            df_with_interactions["age_severity_interaction"] = df["age"] * df["bmi_severity_score"]

        # Chronic risk amplifier
        if "chronic_condition_proxy" in df.columns:
            df_with_interactions["chronic_risk_amplified"] = df["chronic_condition_proxy"] * (
                1 + (df["age"] > 55).astype(int) * 0.5
            )

        # Non-smoker hidden risk (targets worst predictions)
        df_with_interactions["nonsmoker_hidden_risk"] = (
            non_smoker
            * (df["age"] > 50).astype(int)
            * (df["bmi"] > 32).astype(int)
            * df["age"]
            * df["bmi"]
            / 1000  # Normalized
        )

        # Senior BMI penalty (non-linear)
        df_with_interactions["senior_bmi_penalty"] = (df["age"] > 60).astype(int) * np.maximum(
            df["bmi"] - 25, 0
        ) ** 1.5

        # ============================================================
        # VALIDATION
        # ============================================================

        new_features = [
            "smoker_age_bmi",
            "bmi_age_ratio",
            "non_smoker_high_bmi_age",
            "non_smoker_senior",
            "smoker_senior_interaction",
            "extreme_bmi",
            "nonsmoker_age_bmi_quad",
            "nonsmoker_hidden_risk",
            "senior_bmi_penalty",
        ]

        # Add conditional features
        if "children" in df.columns:
            new_features.extend(["non_smoker_multi_children", "potential_catastrophic"])
        if "bmi_severity_score" in df.columns:
            new_features.append("age_severity_interaction")
        if "chronic_condition_proxy" in df.columns:
            new_features.append("chronic_risk_amplified")

        for feat in new_features:
            if feat in df_with_interactions.columns:
                if df_with_interactions[feat].isna().any():
                    raise ValueError(f"Domain feature '{feat}' contains NaN values")

        if fit:
            n_enhanced = sum(
                [
                    "quad" in f or "hidden" in f or "composite" in f or "penalty" in f
                    for f in new_features
                ]
            )
            logger.info(f"✅ Added domain interactions: {len(new_features)} features")
            logger.info(f"   Includes {n_enhanced} enhanced high-value predictors")

        return df_with_interactions

    def add_polynomial_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df_with_poly = df.copy()
        continuous_features = self._get_continuous_features(df)

        if len(continuous_features) < 2:
            logger.warning(
                f"Need >=2 continuous features for polynomials, found {len(continuous_features)}. "
                f"Skipping polynomial feature generation."
            )
            if fit:
                self._poly_continuous_features_snapshot = tuple()
                self._poly_feature_names_snapshot = tuple(continuous_features)
            return df_with_poly

        if fit:
            self._poly_continuous_features_snapshot = tuple(continuous_features)

            from math import comb

            n = len(continuous_features)
            d = self.config.polynomial_degree
            estimated_features = comb(n + d, d) - 1

            # Respect explicit max features limit from config
            max_poly_features = getattr(self.config, "max_polynomial_features", 200)

            # OT-05 FIX: use interaction_only=True to reduce feature explosion.
            # The previous value (False) generated full polynomial expansions
            # including x², x³ etc., contradicting the inline comment and
            # wasting memory with ~22% more features before the cap is applied.
            use_interaction_only = True

            if estimated_features > max_poly_features:
                logger.warning(
                    f"Polynomial degree={d} would create {estimated_features} features "
                    f"(limit: {max_poly_features}). Trying interaction_only..."
                )

                # Try with interaction_only first
                estimated_interaction = comb(n, 2) + n  # n choose 2 + original features

                if estimated_interaction <= max_poly_features:
                    use_interaction_only = True
                    estimated_features = estimated_interaction
                    logger.info(
                        f"Using interaction_only: creates {estimated_features} features "
                        f"(within limit)"
                    )
                else:
                    # Still too many, reduce degree
                    d = 2
                    estimated_features = comb(n + 2, 2) - 1

                    if estimated_features > max_poly_features:
                        # Use variance-based selection instead of failing
                        logger.warning(
                            f"Even degree=2 creates {estimated_features} features. "
                            f"Using variance-based selection to keep top {max_poly_features}."
                        )
                        use_interaction_only = True

            # Create polynomial transformer
            self.poly_transformer = PolynomialFeatures(
                degree=d, include_bias=False, interaction_only=use_interaction_only
            )

            # Generate all polynomial features
            poly_array = self.poly_transformer.fit_transform(df[continuous_features])
            poly_feature_names = list(
                self.poly_transformer.get_feature_names_out(continuous_features)
            )
            poly_feature_names = [name.replace(" ", "_") for name in poly_feature_names]

            # Smart feature selection if still too many
            new_poly_indices = [
                i for i, name in enumerate(poly_feature_names) if name not in continuous_features
            ]
            new_poly_names = [poly_feature_names[i] for i in new_poly_indices]

            # Calculate allowable new features
            allowable_new = max_poly_features - len(continuous_features)

            if allowable_new <= 0:
                logger.error(
                    f"No budget for polynomial features "
                    f"(max={max_poly_features} <= base={len(continuous_features)}). "
                    f"Skipping."
                )
                self._poly_feature_names_snapshot = tuple(continuous_features)
                return df_with_poly

            # Variance-based selection
            keep_n = min(50, allowable_new)

            if len(new_poly_names) > keep_n:
                logger.info(
                    f"Selecting top {keep_n}/{len(new_poly_names)} polynomial features "
                    f"by variance..."
                )

                try:
                    # Compute variance for each polynomial feature
                    poly_candidate_matrix = poly_array[:, new_poly_indices]
                    poly_variances = np.var(poly_candidate_matrix, axis=0)

                    # Select top features by variance
                    top_rel_idxs = np.argsort(poly_variances)[-keep_n:]
                    selected_global_idxs = [new_poly_indices[i] for i in top_rel_idxs]
                    selected_poly_names = [poly_feature_names[i] for i in selected_global_idxs]

                    logger.info(
                        f"✅ Selected {len(selected_poly_names)} polynomial features "
                        f"(top {keep_n} by variance)"
                    )

                    # Log top 5 selected features
                    top_5_vars = sorted(
                        zip(selected_poly_names, poly_variances[top_rel_idxs]),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]

                    logger.debug("Top 5 polynomial features by variance:")
                    for feat_name, var in top_5_vars:
                        logger.debug(f"  {feat_name}: variance={var:.4f}")

                    chosen_new_poly_names = selected_poly_names

                except Exception as e:
                    logger.error(f"Variance-based selection failed: {e}")
                    # Fallback: just take first keep_n features
                    chosen_new_poly_names = new_poly_names[:keep_n]
                    logger.warning(f"Using first {keep_n} features as fallback")
            else:
                chosen_new_poly_names = new_poly_names
                logger.info(
                    f"✅ Using all {len(chosen_new_poly_names)} polynomial features "
                    f"(within budget)"
                )

            # Store snapshot
            self._poly_feature_names_snapshot = tuple(
                list(continuous_features) + chosen_new_poly_names
            )

            # Add selected features to dataframe
            poly_feature_name_to_idx = {name: i for i, name in enumerate(poly_feature_names)}

            for feat_name in chosen_new_poly_names:
                if feat_name in poly_feature_name_to_idx:
                    idx = poly_feature_name_to_idx[feat_name]
                    df_with_poly[feat_name] = poly_array[:, idx]

            logger.info(
                f"✅ Polynomial features complete: {len(chosen_new_poly_names)} new features "
                f"from {len(continuous_features)} base features (degree={d}, "
                f"interaction_only={use_interaction_only})"
            )

        else:
            # TRANSFORM MODE (unchanged)
            if self.poly_transformer is None:
                logger.debug("No polynomial transformer fitted, skipping")
                return df_with_poly

            if not self._poly_continuous_features_snapshot:
                logger.debug("No continuous features stored, skipping polynomials")
                return df_with_poly

            missing_features = [
                f for f in self._poly_continuous_features_snapshot if f not in df.columns
            ]
            if missing_features:
                raise ValueError(f"Cannot create polynomial features - missing: {missing_features}")

            poly_array = self.poly_transformer.transform(
                df[list(self._poly_continuous_features_snapshot)]
            )

            new_poly_names = [
                name
                for name in self._poly_feature_names_snapshot
                if name not in self._poly_continuous_features_snapshot
            ]

            poly_feature_names = list(
                self.poly_transformer.get_feature_names_out(self._poly_continuous_features_snapshot)
            )
            poly_feature_names = [name.replace(" ", "_") for name in poly_feature_names]
            poly_feature_name_to_idx = {name: i for i, name in enumerate(poly_feature_names)}

            for feat_name in new_poly_names:
                if feat_name in poly_feature_name_to_idx:
                    idx = poly_feature_name_to_idx[feat_name]
                    df_with_poly[feat_name] = poly_array[:, idx]

            logger.debug(f"Created {len(new_poly_names)} polynomial features in transform mode")

        return df_with_poly

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        fit: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Scale continuous features using StandardScaler with robust validation.

        Improvements:
        1. Enhanced feature order validation
        2. Comprehensive sklearn compatibility checks
        3. Better error messages with diagnostics
        4. Defensive programming for transform mode
        """

        if fit:
            self._validate_state(PipelineState.OUTLIERS_DETECTED, "scaling")

        logger.info("Scaling features...")

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            logger.warning("No numerical columns for scaling")
            return X_train, X_test

        # ========================================
        # FIT MODE: Train scaler
        # ========================================
        if fit:
            # Identify continuous features (exclude binary)
            continuous_cols = [col for col in numeric_cols if X_train[col].nunique() > 2]

            # Exclude collinear features if they were removed
            if self._removed_collinear_features_snapshot:
                removed = set(self._removed_collinear_features_snapshot)
                continuous_cols = [col for col in continuous_cols if col not in removed]

            # Store as ORDERED list (not set)
            self._continuous_features = continuous_cols.copy()

            logger.info(f"Identified {len(continuous_cols)} continuous features during fit")
            logger.debug(
                f"Continuous features: {continuous_cols[:5]}{'...' if len(continuous_cols) > 5 else ''}"
            )

            if not continuous_cols:
                logger.info("Only binary features detected, skipping scaling")
                return X_train, X_test

            # ----------------------------------------
            # VALIDATION: Check for NaN/Inf
            # ----------------------------------------
            if X_train[continuous_cols].isna().any().any():
                null_cols = (
                    X_train[continuous_cols].columns[X_train[continuous_cols].isna().any()].tolist()
                )
                raise ValueError(
                    f"❌ Cannot fit scaler with NaN values in columns: {null_cols}\n"
                    f"   Ensure imputation step completed before scaling"
                )

            if np.isinf(X_train[continuous_cols]).any().any():
                inf_cols = (
                    X_train[continuous_cols]
                    .columns[np.isinf(X_train[continuous_cols]).any()]
                    .tolist()
                )
                raise ValueError(
                    f"❌ Cannot fit scaler with infinite values in columns: {inf_cols}\n"
                    f"   Check for division by zero or log of negative values"
                )

            # ----------------------------------------
            # FIT SCALER: Pass DataFrame to preserve feature names
            # ----------------------------------------
            X_train_scaled = X_train.copy()

            # Pass DataFrame (not .values) to sklearn
            self.scaler.fit(X_train[continuous_cols])

            # Validate sklearn stored correct feature names
            if hasattr(self.scaler, "feature_names_in_"):
                stored_names = list(self.scaler.feature_names_in_)

                if stored_names != continuous_cols:
                    raise RuntimeError(
                        f"❌ CRITICAL: Scaler feature name mismatch!\n"
                        f"   Expected order: {continuous_cols}\n"
                        f"   Sklearn stored: {stored_names}\n"
                        f"\n"
                        f"   This indicates a bug in sklearn or feature extraction.\n"
                        f"   Please verify sklearn version >= 1.0"
                    )

                logger.debug(f"✅ Scaler feature names verified: {len(stored_names)} features")
            else:
                logger.warning(
                    "⚠️  sklearn version does not support 'feature_names_in_'.\n"
                    "   Column order validation will be limited.\n"
                    "   Consider upgrading to sklearn >= 1.0"
                )

            # Transform training data
            X_train_scaled[continuous_cols] = self.scaler.transform(X_train[continuous_cols])

            logger.info(
                f"✅ Fitted StandardScaler on {len(continuous_cols)} continuous features\n"
                f"   Mean range: [{self.scaler.mean_.min():.4f}, {self.scaler.mean_.max():.4f}]\n"
                f"   Std range: [{self.scaler.scale_.min():.4f}, {self.scaler.scale_.max():.4f}]"
            )

            self._update_state(PipelineState.SCALED)

        # ========================================
        # TRANSFORM MODE: Apply fitted scaler
        # ========================================
        else:
            # ----------------------------------------
            # VALIDATION: Check scaler was fitted
            # ----------------------------------------
            if not hasattr(self, "_continuous_features"):
                raise RuntimeError(
                    "❌ Continuous features not defined!\n"
                    "   This preprocessor may have been fitted with an older version.\n"
                    "   ✅ FIX: Retrain preprocessor with current version"
                )

            continuous_cols = self._continuous_features

            if not continuous_cols:
                logger.info("No continuous features to scale (binary only)")
                return X_train, X_test

            if not hasattr(self.scaler, "mean_"):
                raise RuntimeError(
                    "❌ StandardScaler not fitted!\n"
                    "   ✅ FIX: Call scale_features() with fit=True first"
                )

            # ----------------------------------------
            # VALIDATION: Check all required features present
            # ----------------------------------------
            missing_features = [col for col in continuous_cols if col not in X_train.columns]

            if missing_features:
                available_features = sorted(X_train.columns.tolist())

                logger.error(
                    f"❌ MISSING FEATURES DETECTED!\n"
                    f"   Available columns ({len(X_train.columns)}): {available_features[:10]}...\n"
                    f"   Expected continuous ({len(continuous_cols)}): {continuous_cols[:10]}...\n"
                    f"   Missing ({len(missing_features)}): {missing_features}"
                )

                raise ValueError(
                    f"❌ Input data is missing required continuous features!\n"
                    f"   Missing: {missing_features}\n"
                    f"\n"
                    f"   🔍 DIAGNOSIS:\n"
                    f"   → Pipeline execution order problem\n"
                    f"   → Features were removed/renamed in earlier step\n"
                    f"   → Train/test feature mismatch\n"
                    f"\n"
                    f"   ✅ FIX:\n"
                    f"   1. Ensure same preprocessing steps applied to test data\n"
                    f"   2. Check feature engineering consistency\n"
                    f"   3. Verify no features dropped between train and test"
                )

            # ----------------------------------------
            # CRITICAL ORDER VALIDATION
            # ----------------------------------------
            if hasattr(self.scaler, "feature_names_in_"):
                expected_order = list(self.scaler.feature_names_in_)

                # Build actual order from stored continuous features
                # (Don't rely on X_train column order - use stored order)
                actual_order = []
                for col in continuous_cols:
                    if col not in X_train.columns:
                        raise ValueError(f"Missing required feature: {col}")
                    actual_order.append(col)

                # Validate EXACT ORDER match
                if expected_order != actual_order:
                    # Find mismatches
                    order_diffs = []
                    for i, (exp, act) in enumerate(zip(expected_order, actual_order)):
                        if exp != act:
                            order_diffs.append(f"Position {i}: expected '{exp}', got '{act}'")

                    raise RuntimeError(
                        f"🚨 COLUMN ORDER MISMATCH DETECTED!\n"
                        f"   This WILL cause catastrophic prediction errors!\n"
                        f"\n"
                        f"   Expected order (from scaler): {expected_order}\n"
                        f"   Actual order (from stored):   {actual_order}\n"
                        f"\n"
                        f"   Mismatches:\n"
                        + "\n".join(f"   - {diff}" for diff in order_diffs[:5])
                        + "\n"
                        f"\n"
                        f"   🔍 DIAGNOSIS:\n"
                        f"   → _continuous_features order doesn't match scaler training\n"
                        f"   → Feature engineering step changed column order\n"
                        f"   → Preprocessor corrupted or version mismatch\n"
                        f"\n"
                        f"   ✅ FIX:\n"
                        f"   1. Retrain preprocessor to reset feature order\n"
                        f"   2. Ensure deterministic feature engineering\n"
                        f"   3. Verify saved preprocessor version matches current code"
                    )

                logger.debug(
                    f"✅ Column order validated: {len(expected_order)} features in correct order"
                )
            else:
                logger.warning(
                    "⚠️  Cannot validate feature order (sklearn < 1.0)\n"
                    "   Proceeding with stored feature order - errors possible if order changed"
                )

            # ----------------------------------------
            # VALIDATION: Check feature count
            # ----------------------------------------
            expected_count = self.scaler.n_features_in_
            actual_count = len(continuous_cols)

            if expected_count != actual_count:
                raise ValueError(
                    f"❌ Feature count mismatch!\n"
                    f"   Scaler was fitted with {expected_count} features\n"
                    f"   Current data has {actual_count} continuous features\n"
                    f"\n"
                    f"   This indicates preprocessing pipeline inconsistency"
                )

            # ----------------------------------------
            # VALIDATION: Check for NaN/Inf in test data
            # ----------------------------------------
            if X_train[continuous_cols].isna().any().any():
                null_cols = (
                    X_train[continuous_cols].columns[X_train[continuous_cols].isna().any()].tolist()
                )
                raise ValueError(
                    f"❌ Input data contains NaN values in columns: {null_cols}\n"
                    f"   Ensure imputation completed before scaling"
                )

            if np.isinf(X_train[continuous_cols]).any().any():
                inf_cols = (
                    X_train[continuous_cols]
                    .columns[np.isinf(X_train[continuous_cols]).any()]
                    .tolist()
                )
                raise ValueError(f"❌ Input data contains infinite values in columns: {inf_cols}")

            # ----------------------------------------
            # Extract features in EXACT ORDER
            # ----------------------------------------
            X_train_scaled = X_train.copy()

            # Use stored order (self._continuous_features)
            # This ensures we pass features to scaler in the EXACT same order as training
            X_to_scale = X_train[continuous_cols]  # ← DataFrame with EXACT order

            logger.debug(
                f"Scaling {len(continuous_cols)} features in order: "
                f"{continuous_cols[:3]}{'...' if len(continuous_cols) > 3 else ''}"
            )

            # Transform with validated ordering
            scaled_array = self.scaler.transform(X_to_scale)

            # Reconstruct DataFrame with same order
            scaled_df = pd.DataFrame(
                scaled_array,
                columns=continuous_cols,  # ← Use SAME order
                index=X_train.index,
            )
            X_train_scaled[continuous_cols] = scaled_df

            logger.info(f"✅ Scaled {len(continuous_cols)} continuous features")

        # ========================================
        # HANDLE TEST SET (if provided)
        # ========================================
        X_test_scaled: pd.DataFrame | None = None

        if X_test is not None:
            X_test_scaled = X_test.copy()

            # Validate test set has required features
            missing_features = [col for col in continuous_cols if col not in X_test.columns]

            if missing_features:
                raise ValueError(
                    f"❌ Test data is missing required continuous features!\n"
                    f"   Missing: {missing_features}\n"
                    f"   Ensure same preprocessing applied to train and test data"
                )

            # Validate no NaN/Inf in test data
            if X_test[continuous_cols].isna().any().any():
                null_cols = (
                    X_test[continuous_cols].columns[X_test[continuous_cols].isna().any()].tolist()
                )
                raise ValueError(
                    f"❌ Test data contains NaN values in columns: {null_cols}\n"
                    f"   Ensure imputation completed before scaling"
                )

            if np.isinf(X_test[continuous_cols]).any().any():
                inf_cols = (
                    X_test[continuous_cols]
                    .columns[np.isinf(X_test[continuous_cols]).any()]
                    .tolist()
                )
                raise ValueError(f"❌ Test data contains infinite values in columns: {inf_cols}")

            # Transform test set using same ordering
            scaled_test_array = self.scaler.transform(X_test[continuous_cols])

            scaled_test_df = pd.DataFrame(
                scaled_test_array,
                columns=continuous_cols,
                index=X_test.index,
            )
            X_test_scaled[continuous_cols] = scaled_test_df

            logger.debug(f"✅ Scaled test set: {len(continuous_cols)} features")

        return X_train_scaled, X_test_scaled

    def inverse_transform_target(
        self,
        y_transformed: np.ndarray,
        transformation_method: str | None = None,
        clip_to_safe_range: bool = True,
        context: str = "prediction",
    ) -> np.ndarray:
        """
        Inverse transform target WITHOUT bias correction.

        ⚠️ IMPORTANT:
        Bias correction MUST be applied separately by the caller.
        This method is now:
        - Stateless
        - Deterministic
        - Cache-safe
        """

        if transformation_method is None:
            transformation_method = self.target_transformation.method

        if transformation_method == "none":
            return y_transformed

        # ========================================
        # LOG1P
        # ========================================
        if transformation_method == "log1p":
            if clip_to_safe_range and hasattr(self, "clip_upper_") and self.clip_upper_ is not None:
                y_transformed = np.clip(y_transformed, self.clip_lower_, self.clip_upper_)

            y_original = np.expm1(y_transformed)

            if clip_to_safe_range and hasattr(self, "y_min_safe") and hasattr(self, "y_max_safe"):
                y_original = np.clip(y_original, self.y_min_safe, self.y_max_safe)

            return y_original

        # ========================================
        # YEO-JOHNSON
        # ========================================
        elif transformation_method == "yeo-johnson":
            if not hasattr(self, "yeo_johnson_transformer"):
                raise RuntimeError("Yeo-Johnson transformer not fitted. Cannot inverse transform.")

            if clip_to_safe_range and hasattr(self, "clip_upper_") and self.clip_upper_ is not None:
                y_transformed = np.clip(y_transformed, self.clip_lower_, self.clip_upper_)

            y_original = self.yeo_johnson_transformer.inverse_transform(
                y_transformed.reshape(-1, 1)
            ).ravel()

            if clip_to_safe_range and hasattr(self, "y_min_safe") and hasattr(self, "y_max_safe"):
                y_original = np.clip(y_original, self.y_min_safe, self.y_max_safe)

            return y_original

        # ========================================
        # BOX-COX
        # ========================================
        elif transformation_method.startswith("boxcox") or transformation_method == "boxcox":
            # ── F-05 FIX: Read canonical field; fall back to deprecated alias ──
            # Allows loading of models serialized before the lambda_param migration.
            lambda_ = getattr(self.target_transformation, "lambda_param", None) or getattr(
                self.target_transformation, "boxcox_lambda", None
            )
            if lambda_ is None:
                raise ValueError(
                    "Box-Cox lambda not found.\n"
                    "Expected: target_transformation.lambda_param or .boxcox_lambda\n"
                    "Fix: retrain the model to populate lambda_param."
                )

            if abs(lambda_) < 1e-10:
                y_original = np.exp(y_transformed)
            else:
                inner = lambda_ * y_transformed + 1
                inner = np.maximum(inner, 1e-10)
                y_original = np.power(inner, 1.0 / lambda_)

            if clip_to_safe_range and hasattr(self, "y_min_safe") and hasattr(self, "y_max_safe"):
                y_original = np.clip(y_original, self.y_min_safe, self.y_max_safe)

            return y_original

        else:
            raise ValueError(f"Unknown transformation method: {transformation_method}")

    def validate_pipeline_complete(self) -> None:
        """Verify the pipeline executed fully and all critical artifacts are present.

        Raises:
            RuntimeError: If the pipeline has not reached COMPLETED state, or if
                          any required artifact is missing or uninitialised.
        """
        if self._state != PipelineState.COMPLETED:
            raise RuntimeError(
                f"Pipeline incomplete: {self._state.value}\n"
                f"  Expected: {PipelineState.COMPLETED.value}\n"
                f"  Call fit_transform_pipeline() before validating."
            )

        # Verify critical artifacts were fitted and stored
        required_artifacts = [
            ("scaler", "StandardScaler"),
            ("onehot_encoder", "OneHotEncoder"),
            ("target_transformation", "TargetTransformation"),
        ]

        missing = [
            f"{attr} ({name})"
            for attr, name in required_artifacts
            if not hasattr(self, attr) or getattr(self, attr) is None
        ]

        if missing:
            raise RuntimeError(
                "Pipeline missing required artifacts:\n"
                + "\n".join(f"  - {item}" for item in missing)
            )

        logger.info(
            f"Pipeline validation passed: state={self._state.value}, "
            f"all {len(required_artifacts)} required artifacts present."
        )

    def fit_transform_pipeline(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        target_transform: Literal["none", "log1p", "yeo-johnson", "boxcox"] = "none",
        remove_outliers: bool = True,
        add_polynomials: bool = True,
        remove_collinear: bool = True,
    ) -> dict[str, pd.DataFrame]:
        logger.info("=" * 80)
        logger.info(f"Starting feature engineering pipeline v{self.VERSION} (FIT mode)")
        logger.info("=" * 80)

        self._original_columns = df.columns.tolist()

        with self._timed_step("Create features"):
            df_features = self.create_features(df)

        with self._timed_step("Impute features"):
            df_imputed = self.impute_features(df_features, fit=True)

        with self._timed_step("Remove low variance (pre-encode)"):
            df_variance_filtered_pre = self.remove_low_variance(df_imputed, fit=True)

        with self._timed_step("Encode features"):
            # ML-03 FIX: encode_features() must run BEFORE detect_and_remove_outliers().
            # IsolationForest requires an all-numeric input matrix.  Running it on
            # pre-encoded data that still contains string or label-encoded ordinal
            # categoricals (smoker, sex, region) corrupts anomaly scores because
            # ordinal integers are treated as continuous distances.
            df_encoded_pre = self.encode_features(df_variance_filtered_pre, fit=True)

        with self._timed_step("Remove outliers"):
            if remove_outliers:
                df_clean, y_clean, outlier_mask = self.detect_and_remove_outliers(
                    df_encoded_pre, y, method="isolation_forest", fit=True, return_mask=True
                )
                logger.info(f"Training set after outlier removal: {len(df_clean)} samples")
            else:
                df_clean, y_clean = df_encoded_pre, y
                self._update_state(PipelineState.OUTLIERS_DETECTED)

        self._original_y_train_stats = {
            "mean": float(y_clean.mean()),
            "std": float(y_clean.std()),
            "min": float(y_clean.min()),
            "max": float(y_clean.max()),
        }

        with self._timed_step("Transform target"):
            y_transformed = self.transform_target(y_clean, method=target_transform, fit=True)

        if target_transform != "none":
            logger.info(
                f"Target transformation: {self.target_transformation.method}, "
                f"original range: {self.target_transformation.original_range}"
            )

        with self._timed_step("Remove low variance (post-encode)"):
            # ML-03 FIX: Run VIF/variance filter on the already-encoded df_clean.
            # The pre-encode step above only did a lightweight pass; this is the
            # definitive variance filter on the fully-numeric feature matrix.
            df_variance_filtered = self.remove_low_variance(df_clean, fit=True)

        # Encoding was already done above (pre-outlier-removal); skip re-encoding.
        df_encoded = df_variance_filtered

        with self._timed_step("Add domain interactions"):
            df_with_domain = self.add_domain_interactions(df_encoded, fit=True)

        with self._timed_step("Add polynomial features"):
            if add_polynomials:
                df_with_poly = self.add_polynomial_features(df_with_domain, fit=True)
            else:
                df_with_poly = df_with_domain
                logger.info("Skipping polynomial features (disabled)")

        with self._timed_step("Remove multicollinear features"):
            if remove_collinear:
                # ML-05 FIX: pass y_clean (original-scale target) as y_original for
                # correlation-based tie-breaking.  Ranking features by correlation with
                # the transformed target can prefer different features than the original
                # scale would, particularly for non-linearly-transformed targets.
                df_no_collinear = self.remove_multicollinear_features(
                    df_with_poly, y=y_clean, fit=True
                )
            else:
                df_no_collinear = df_with_poly
                logger.info("Skipping multicollinearity removal (disabled)")

        with self._timed_step("Scale features"):
            X_train_scaled, _ = self.scale_features(df_no_collinear, fit=True)

        # CALCULATE BIAS CORRECTION FOR LOG TRANSFORM
        if target_transform == "log1p":
            # For bias correction, we need residuals in log space
            # This will be calculated after initial model fit in train.py
            # For now, store a placeholder
            self._log_residual_variance = None
            logger.info(
                "✅ Bias correction setup ready (variance will be calculated after model fit)"
            )

        self._fit_complete = True
        self._update_state(PipelineState.COMPLETED)

        logger.info("=" * 80)
        logger.info(f"Pipeline fit complete. Final shape: {X_train_scaled.shape}")
        logger.info(f"Target transformation: {self.target_transformation.method}")
        if self.config.enable_performance_logging:
            logger.info("\nPerformance Summary:")
            for step, elapsed in self._performance_metrics.items():
                logger.info(f"  {step}: {elapsed:.2f}s")
        logger.info("=" * 80)

        result = {"X_train": X_train_scaled, "y_train": y_transformed}

        if X_val is not None:
            logger.info("\nProcessing validation set...")
            X_val_transformed = self.transform_pipeline(X_val, remove_outliers=False)
            result["X_val"] = X_val_transformed

            if y_val is not None:
                y_val_transformed = self.transform_target(y_val, method=target_transform, fit=False)
                result["y_val"] = y_val_transformed
                logger.info(f"Validation set shape: {X_val_transformed.shape}")

        return result

    def transform_pipeline(self, df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:
        """Apply fitted feature transformations to new data (inference path).

        IMPORTANT — target transformation NOT applied here:
            This method transforms features only.  Callers must separately call
            ``transform_target(fit=False)`` on the target array before computing
            evaluation metrics.  Omitting this produces RMSE/MAE values that are
            orders of magnitude larger than training-time metrics.

            ML-09 FIX: Docstring and pipeline step order updated to match the
            corrected fit_transform_pipeline() (ML-03 fix moved outlier detection
            to after encode_features). The inference path now mirrors training:
            impute → variance filter → encode → outlier removal → downstream steps.

        Returns:
            Transformed feature DataFrame ready for model.predict().

        Raises:
            RuntimeError: If pipeline has not been fitted (fit_transform_pipeline() not called).
        """

        # -- Guard 1: pipeline must be fitted ---------------------------------
        if not self._fit_complete:
            raise RuntimeError("FeatureEngineer not fitted. Call fit_transform_pipeline() first.")

        # -- Guard 2: strict column validation (order + names) ----------------
        if self._original_columns is not None:
            if list(df.columns) != list(self._original_columns):
                missing = set(self._original_columns) - set(df.columns)
                extra = set(df.columns) - set(self._original_columns)
                order_mismatch = set(df.columns) == set(self._original_columns) and list(
                    df.columns
                ) != list(self._original_columns)

                error_msg = "Column mismatch between training and inference!\n"
                if missing:
                    error_msg += f"  Missing columns : {sorted(missing)}\n"
                if extra:
                    error_msg += f"  Extra columns   : {sorted(extra)}\n"
                if order_mismatch:
                    error_msg += (
                        f"  Column order differs:\n"
                        f"    Training : {list(self._original_columns)}\n"
                        f"    Input    : {list(df.columns)}\n"
                    )

                raise ValueError(error_msg)

        # -- Guard 3: warn on unknown categorical values ----------------------
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                unknown_values = set(df[col].unique()) - set(encoder.classes_)
                if unknown_values:
                    logger.warning(
                        f"Column '{col}' has unknown categories: {unknown_values}\n"
                        f"  Known categories     : {list(encoder.classes_)}\n"
                        f"  Fallback (most freq) : {self.most_frequent_values.get(col, 'N/A')}"
                    )

        logger.info("Transforming new data through pipeline...")

        df_features = self.create_features(df)
        df_imputed = self.impute_features(df_features, fit=False)

        # ML-03 FIX (inference path): encode BEFORE outlier removal to match
        # the corrected fit_transform_pipeline() ordering.  IsolationForest was
        # fitted on encoded (numeric-only) data, so it must be applied to encoded
        # data at inference too.
        df_variance_filtered = self.remove_low_variance(df_imputed, fit=False)
        df_encoded = self.encode_features(df_variance_filtered, fit=False)

        if remove_outliers and self.outlier_detector is not None:
            df_clean, _, _ = self.detect_and_remove_outliers(
                df_encoded, method=self.outlier_method, fit=False, return_mask=True
            )
            logger.info(f"Removed outliers from test set: {len(df_encoded)} -> {len(df_clean)}")
        else:
            df_clean = df_encoded

        df_with_domain = self.add_domain_interactions(df_clean, fit=False)

        if self.poly_transformer is not None:
            df_with_poly = self.add_polynomial_features(df_with_domain, fit=False)
        else:
            df_with_poly = df_with_domain

        df_no_collinear = self.remove_multicollinear_features(df_with_poly, fit=False)
        X_scaled, _ = self.scale_features(df_no_collinear, fit=False)

        logger.info(f"Transformation complete. Shape: {X_scaled.shape}")
        return X_scaled

    def transform_pipeline_batched(
        self,
        df: pd.DataFrame,
        batch_size: int = 50000,
        remove_outliers: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if not self._fit_complete:
            raise RuntimeError("Pipeline not fitted. Call fit_transform_pipeline() first.")

        # If small enough, use standard pipeline
        if len(df) <= batch_size:
            logger.info(f"Dataset size ({len(df)}) <= batch_size, using standard pipeline")
            return self.transform_pipeline(df, remove_outliers=remove_outliers)

        logger.info(
            f"Batched transformation: {len(df)} rows in {int(np.ceil(len(df) / batch_size))} batches"
        )

        results = []
        iterator = range(0, len(df), batch_size)

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    iterator,
                    desc="Transforming batches",
                    total=int(np.ceil(len(df) / batch_size)),
                )
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")

        for i in iterator:
            batch = df.iloc[i : i + batch_size]
            transformed_batch = self.transform_pipeline(batch, remove_outliers=remove_outliers)
            results.append(transformed_batch)

        logger.info("Concatenating batches...")
        final_result = pd.concat(results, axis=0, ignore_index=False)
        logger.info(f"Batched transformation complete. Final shape: {final_result.shape}")

        return final_result

    def set_bias_correction(
        self,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        threshold: float | None = None,
        save_json_path: str | None = None,
    ) -> BiasCorrection:
        """
        Compute and store stratified log-space residual variances for bias correction.

        Call this AFTER fitting the model, before calling save_preprocessor():

            model.fit(X_train, y_train_log)
            y_pred_log  = model.predict(X_train)
            y_pred_orig = fe.inverse_transform_target(y_pred_log)
            bc = fe.set_bias_correction(y_train_orig, y_pred_orig)

        The method:
        1. Converts both arrays to log-space via log1p.
        2. Computes the global residual variance (_log_residual_variance) — used by
           predict_with_intervals() for CI width.
        3. Splits residuals at ``threshold`` (default: median of y_train) and computes
           per-tier variances (_bias_var_low, _bias_var_high, _bias_threshold).
        4. Stores all four attributes on self so save_preprocessor() persists them.
        5. Optionally writes bias_correction.json alongside the preprocessor so that
           predict.py can load it independently via BiasCorrection.from_dict().

        Args:
            y_train:      Ground-truth premiums in ORIGINAL scale (not log-transformed).
            y_train_pred: Model predictions in ORIGINAL scale.
            threshold:    Tier split point in original scale.  Defaults to the median
                          of y_train, which is a robust choice for insurance data.
            save_json_path: If provided, writes bias_correction.json to this path.
                            Typically set to the same directory as the preprocessor .joblib.

        Returns:
            BiasCorrection instance (also stored implicitly via the four instance attrs).
        """
        y_train = np.asarray(y_train, dtype=float)
        y_train_pred = np.asarray(y_train_pred, dtype=float)

        if y_train.shape != y_train_pred.shape:
            raise ValueError(
                f"y_train shape {y_train.shape} != y_train_pred shape {y_train_pred.shape}"
            )

        # Clip to avoid log(0) on edge cases
        _eps = 1e-6
        y_train_safe = np.clip(y_train, _eps, None)
        y_pred_safe = np.clip(y_train_pred, _eps, None)

        # Log-space residuals  (log1p keeps parity with inverse_transform_target)
        log_residuals = np.log1p(y_train_safe) - np.log1p(y_pred_safe)

        # Global variance — used by predict_with_intervals() for CI width
        self._log_residual_variance = float(np.var(log_residuals, ddof=1))

        # Tier threshold — default to median for a balanced split
        _threshold = float(threshold) if threshold is not None else float(np.median(y_train))

        low_mask = y_train < _threshold
        high_mask = ~low_mask

        _var_low = (
            float(np.var(log_residuals[low_mask], ddof=1))
            if low_mask.sum() > 1
            else self._log_residual_variance
        )
        _var_high = (
            float(np.var(log_residuals[high_mask], ddof=1))
            if high_mask.sum() > 1
            else self._log_residual_variance
        )

        self._bias_var_low = _var_low
        self._bias_var_high = _var_high
        self._bias_threshold = _threshold

        logger.info(
            f"✅ Bias correction computed:\n"
            f"   Global variance:  {self._log_residual_variance:.6f}\n"
            f"   Threshold:        {_threshold:,.0f}\n"
            f"   var_low  (n={int(low_mask.sum())}):  {_var_low:.6f}  "
            f"→ factor={np.exp(_var_low / 2):.6f}\n"
            f"   var_high (n={int(high_mask.sum())}): {_var_high:.6f}  "
            f"→ factor={np.exp(_var_high / 2):.6f}"
        )

        bc = BiasCorrection(
            var_low=_var_low,
            var_high=_var_high,
            threshold=_threshold,
        )

        # Optionally persist bias_correction.json for predict.py to load
        if save_json_path is not None:
            import json as _json

            _json_path = Path(save_json_path)
            _json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_json_path, "w") as _f:
                _json.dump(bc.to_dict(), _f, indent=2)
            logger.info(f"✅ Saved bias_correction.json → {_json_path}")

        return bc

    def save_preprocessor(self, save_path: str = "models/preprocessor_v4.joblib") -> None:
        """
        Save preprocessing objects with versioning and validation.

        Improvements:
        1. Validates critical attributes before saving
        2. Warns about missing bias correction for log1p
        3. Checks for incomplete preprocessing state
        4. Organizes attributes into logical groups
        """

        # ========================================
        # VALIDATION: Check preprocessing completeness
        # ========================================

        # Guard: heal stale pipeline_state BEFORE serialization.
        #
        # With the _update_state() backward-transition guard in place, this block
        # should NEVER fire during normal operation: transform_new_data() can no
        # longer regress COMPLETED → FEATURES_CREATED after a successful fit.
        #
        # This remains as a last-resort safety net for:
        #   a) Artifacts serialized by older versions of this class (before the
        #      _update_state guard was added) and then loaded + re-saved.
        #   b) Any external code that directly mutates self._state without going
        #      through _update_state().
        #
        # If this warning fires, it means the _update_state guard was bypassed —
        # treat it as a bug signal and investigate the call stack.
        _scaler_fitted = hasattr(self.scaler, "scale_") or hasattr(self.scaler, "mean_")
        if self._fit_complete and _scaler_fitted and self._state != PipelineState.COMPLETED:
            logger.warning(
                f"⚠️  save_preprocessor(): pipeline_state='{self._state.value}' is stale "
                f"(_fit_complete=True, scaler fitted).\n"
                f"   Advancing state → COMPLETED before serialization.\n"
                f"   NOTE: This should not fire with the current _update_state() guard.\n"
                f"   If you see this, _state was mutated outside _update_state() — "
                f"investigate the call stack."
            )
            self._state = PipelineState.COMPLETED

        if not self._fit_complete:
            logger.warning(
                "⚠️  INCOMPLETE PREPROCESSING STATE!\n"
                "   Preprocessing pipeline has not completed fit()\n"
                "   Saved preprocessor may not work correctly on new data"
            )

        # ========================================
        # VALIDATION: Transformation-specific checks
        # ========================================
        transformation_method = self.target_transformation.method

        if transformation_method == "log1p":
            # Bias correction for log1p
            if not hasattr(self, "_log_residual_variance") or self._log_residual_variance is None:
                logger.warning(
                    "⚠️  SAVING WITHOUT BIAS CORRECTION!\n"
                    "   Transformation: log1p\n"
                    "   Impact: Predictions will be systematically LOW by ~5-10%\n"
                    "   ✅ FIX: Call set_bias_correction() before saving:\n"
                    "      preprocessor.set_bias_correction(y_train, y_train_pred)"
                )
            else:
                logger.info(
                    f"✅ Bias correction included: variance={self._log_residual_variance:.6f}"
                )

        elif transformation_method == "yeo-johnson":
            # Check Yeo-Johnson transformer exists
            if not hasattr(self, "yeo_johnson_transformer") or self.yeo_johnson_transformer is None:
                logger.error(
                    "❌ CRITICAL: Yeo-Johnson transformer not found!\n"
                    "   Saved preprocessor will fail on inverse_transform"
                )

        elif transformation_method.startswith("boxcox"):
            # ── BUG 7 FIX: check lambda_param (canonical) OR boxcox_lambda (deprecated) ──
            # post-F05 fix both are written; legacy models only have boxcox_lambda.
            _bc_lambda = getattr(self.target_transformation, "lambda_param", None) or getattr(
                self.target_transformation, "boxcox_lambda", None
            )
            if _bc_lambda is None:
                logger.error(
                    "❌ CRITICAL: Box-Cox lambda not found!\n"
                    "   Neither lambda_param nor boxcox_lambda is set.\n"
                    "   Saved preprocessor will fail on inverse_transform"
                )

        # ========================================
        # VALIDATION: Clipping bounds check
        # ========================================
        if hasattr(self, "clip_upper_") and self.clip_upper_ is not None:
            logger.debug(
                f"Clipping bounds included: "
                f"[{getattr(self, 'clip_lower_', 'N/A'):.4f}, {self.clip_upper_:.4f}]"
            )

        # ========================================
        # BUILD PREPROCESSOR DICTIONARY
        # ========================================
        preprocessor = {
            # Core configuration
            "config": self.config,
            "version": self.VERSION,
            "fit_complete": self._fit_complete,
            "pipeline_state": self._state.value,
            # Feature scaling and encoding
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "most_frequent_values": self.most_frequent_values,
            "onehot_encoder": self.onehot_encoder,
            "feature_names_snapshot": self._feature_names_snapshot,
            # Feature engineering
            "poly_transformer": self.poly_transformer,
            "poly_feature_names_snapshot": self._poly_feature_names_snapshot,
            "poly_continuous_features_snapshot": self._poly_continuous_features_snapshot,
            # Imputation
            "demographic_imputer": self.demographic_imputer,
            "derived_imputer": self.derived_imputer,
            "categorical_imputer": self.categorical_imputer,
            # Feature selection
            "variance_selector": self.variance_selector,
            "variance_feature_names_snapshot": self._variance_feature_names_snapshot,
            "removed_collinear_features_snapshot": self._removed_collinear_features_snapshot,
            # Outlier handling
            "outlier_detector": self.outlier_detector,
            "outlier_method": self.outlier_method,
            "outlier_indices": self._outlier_indices,
            # Target transformation - Core parameters
            "target_transformation": self.target_transformation,
            "target_min_": self.target_min_,
            "target_max_": self.target_max_,
            "transformed_min_": self.transformed_min_,
            "transformed_max_": self.transformed_max_,
            "continuous_features": self._continuous_features,
            # Target transformation - Transformation-specific
            "yeo_johnson_transformer": getattr(self, "yeo_johnson_transformer", None),
            # Target transformation - Clipping bounds
            "clip_lower_": getattr(self, "clip_lower_", None),
            "clip_upper_": getattr(self, "clip_upper_", None),
            "y_min_safe": getattr(self, "y_min_safe", None),
            "y_max_safe": getattr(self, "y_max_safe", None),
            # Bias correction (single source of truth)
            "_log_residual_variance": getattr(self, "_log_residual_variance", None),
            "_bias_var_low": getattr(self, "_bias_var_low", None),
            "_bias_var_high": getattr(self, "_bias_var_high", None),
            "_bias_threshold": getattr(self, "_bias_threshold", None),
            "_yj_correction_factor": getattr(self, "_yj_correction_factor", None),
            "_yj_lambda": getattr(self, "_yj_lambda", None),
            "_yj_residual_variance": getattr(self, "_yj_residual_variance", None),
        }

        # ========================================
        # SAVE TO DISK
        # ========================================
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(preprocessor, save_path)
            file_size = save_path_obj.stat().st_size / (1024 * 1024)  # MB

            logger.info(
                f"Saved preprocessor to {save_path} (v{self.VERSION})\n"
                f"  - Size: {file_size:.2f} MB\n"
                f"  - Transformation: {transformation_method}\n"
                f"  - State: {self._state.value}"
            )

        except Exception as e:
            logger.error(
                f"❌ FAILED to save preprocessor!\n" f"   Path: {save_path}\n" f"   Error: {e}"
            )
            raise

        # ========================================
        # POST-SAVE DIAGNOSTICS (optional)
        # ========================================
        # Check for common missing attributes
        critical_attrs = ["scaler", "label_encoders", "target_transformation"]
        missing_attrs = [attr for attr in critical_attrs if preprocessor.get(attr) is None]

        if missing_attrs:
            logger.warning(f"⚠️  Missing critical attributes in saved preprocessor: {missing_attrs}")

    def load_preprocessor(self, load_path: str = "models/preprocessor_v4.joblib") -> Self:
        """
        Load preprocessing objects with STRICT validation.

        """
        load_path_obj = Path(load_path)

        if not load_path_obj.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {load_path}")

        try:
            preprocessor = joblib.load(load_path)
            file_size = load_path_obj.stat().st_size / (1024 * 1024)  # MB
        except Exception as e:
            logger.error(f"❌ Failed to load preprocessor from {load_path}: {e}")
            raise

        # ========================================
        # CRITICAL VALIDATION: Version check
        # ========================================
        version = preprocessor.get("version", "unknown")
        if version not in self.COMPATIBLE_VERSIONS:
            raise ValueError(
                f"❌ INCOMPATIBLE VERSION: {version}\n"
                f"   Compatible versions: {self.COMPATIBLE_VERSIONS}\n"
                f"   Current version: {self.VERSION}\n"
                f"\n"
                f"   This preprocessor is too old or from a different pipeline.\n"
                f"   You must retrain the preprocessor with the current version."
            )

        # ========================================
        # CRITICAL VALIDATION: Required attributes (except config)
        # ========================================
        required_attrs = {
            "scaler": "Feature scaling",
            "label_encoders": "Categorical encoding",
            "onehot_encoder": "One-hot encoding",
            "target_transformation": "Target transformation",
        }

        missing_critical = []
        for attr, description in required_attrs.items():
            if attr not in preprocessor or preprocessor[attr] is None:
                missing_critical.append(f"{attr} ({description})")

        if missing_critical:
            raise ValueError(
                f"❌ CRITICAL ATTRIBUTES MISSING!\n"
                f"   Missing: {missing_critical}\n"
                f"\n"
                f"   This preprocessor is corrupted or incomplete.\n"
                f"   You must retrain from scratch."
            )

        # ========================================
        # Handle config with proper fallback
        # ========================================
        if "config" in preprocessor and preprocessor["config"] is not None:
            # Config exists in saved preprocessor - use it
            self.config = preprocessor["config"]
            logger.info("✅ Loaded config from saved preprocessor")
        else:
            # No config in preprocessor - use current instance config
            logger.warning(
                "⚠️  LEGACY PREPROCESSOR DETECTED!\n"
                "   This preprocessor was saved without config (pre-v6.1.0).\n"
                "   Using config from current FeatureEngineer instance.\n"
                "   RECOMMENDATION: Re-train to save config in preprocessor."
            )

            # Verify we have a config from __init__
            if not hasattr(self, "config") or self.config is None:
                raise ValueError(
                    "❌ CANNOT LOAD LEGACY PREPROCESSOR!\n"
                    "\n"
                    "   The saved preprocessor doesn't contain config (legacy format),\n"
                    "   and no config was provided when initializing FeatureEngineer.\n"
                    "\n"
                    "   ✅ SOLUTIONS:\n"
                    "   1. RECOMMENDED: Re-train the model to generate new preprocessor:\n"
                    "      python scripts/train_model.py\n"
                    "\n"
                    "   2. OR initialize FeatureEngineer with config before loading:\n"
                    "      from insurance_ml.config import load_config, get_feature_config\n"
                    "      config = load_config()\n"
                    "      feat_cfg = get_feature_config(config)\n"
                    "      fe = FeatureEngineer(config_dict=feat_cfg)\n"
                    "      fe.load_preprocessor('path/to/preprocessor.joblib')\n"
                )

            # Keep the config we already have from __init__
            logger.info("   Using FeatureEngineer instance config (from __init__)")

        # ========================================
        # Restore attributes
        # ========================================
        self.scaler = preprocessor["scaler"]
        self.label_encoders = preprocessor["label_encoders"]
        self.onehot_encoder = preprocessor["onehot_encoder"]
        self._feature_names_snapshot = preprocessor["feature_names_snapshot"]
        self.most_frequent_values = preprocessor.get("most_frequent_values", {})

        # Feature engineering
        self.poly_transformer = preprocessor.get("poly_transformer", None)
        self._poly_feature_names_snapshot = preprocessor.get("poly_feature_names_snapshot", None)
        self._poly_continuous_features_snapshot = preprocessor.get(
            "poly_continuous_features_snapshot", None
        )

        # Imputation
        self.demographic_imputer = preprocessor.get("demographic_imputer", None)
        self.derived_imputer = preprocessor.get("derived_imputer", None)
        self.categorical_imputer = preprocessor.get("categorical_imputer", None)

        # Feature selection
        self.variance_selector = preprocessor.get("variance_selector", None)
        self._variance_feature_names_snapshot = preprocessor.get(
            "variance_feature_names_snapshot", None
        )
        self._removed_collinear_features_snapshot = preprocessor.get(
            "removed_collinear_features_snapshot", ()
        )

        # Outlier handling
        self.outlier_detector = preprocessor.get("outlier_detector", None)
        self.outlier_method = preprocessor.get("outlier_method", "none")
        self._outlier_indices = preprocessor.get("outlier_indices", None)

        # State
        self._fit_complete = preprocessor.get("fit_complete", False)
        state_value = preprocessor.get("pipeline_state", "initialized")
        self._state = PipelineState(state_value)

        # ── FIX 1: State inference ────────────────────────────────────────────
        # Artifacts serialized by older versions of this class (before the
        # _update_state() backward-transition guard was added) may record an
        # incomplete state value even though all fitters are present and the
        # pipeline operates correctly at inference.
        #
        # Artifacts produced by the current version should never reach this
        # branch: _update_state() now blocks COMPLETED → FEATURES_CREATED
        # regressions, so save_preprocessor() always serializes the correct
        # state. If this warning fires on a freshly trained artifact, it means
        # _state was mutated outside _update_state() — investigate.
        #
        # For legacy artifacts: recover silently and emit a WARNING so
        # operators know a retrain will eliminate it.
        _scaler_fitted = self.scaler is not None and hasattr(
            self.scaler, "mean_"
        )  # StandardScaler fit indicator
        if self._fit_complete and _scaler_fitted and self._state != PipelineState.COMPLETED:
            logger.warning(
                f"⚠️  ARTIFACT INTEGRITY: pipeline_state='{self._state.value}' "
                f"in saved preprocessor, but _fit_complete=True and scaler is fitted.\n"
                f"   Legacy artifact (saved before _update_state() backward-transition "
                f"guard was introduced).\n"
                f"   Inferred state → COMPLETED for this session.\n"
                f"   ✅ FIX: Retrain once to produce a clean artifact and eliminate "
                f"this warning."
            )
            self._state = PipelineState.COMPLETED

        # Target transformation
        self.target_transformation = preprocessor.get(
            "target_transformation", TargetTransformation(method="none")
        )
        self.target_min_ = preprocessor.get("target_min_", None)
        self.target_max_ = preprocessor.get("target_max_", None)
        self.transformed_min_ = preprocessor.get("transformed_min_", None)
        self.transformed_max_ = preprocessor.get("transformed_max_", None)

        # ========================================
        # CRITICAL VALIDATION: Continuous features
        # ========================================
        self._continuous_features = preprocessor.get("continuous_features", [])

        if not self._continuous_features and self._fit_complete:
            raise ValueError(
                f"❌ MISSING CONTINUOUS FEATURES!\n"
                f"\n"
                f"   This preprocessor was saved without continuous feature metadata.\n"
                f"   Predictions will FAIL or produce garbage results.\n"
                f"\n"
                f"   📋 DIAGNOSIS:\n"
                f"   → Saved with old version (< v4.3.0)\n"
                f"   → Corrupted during save/load\n"
                f"   → Incompatible serialization format\n"
                f"\n"
                f"   ✅ FIX:\n"
                f"   You MUST retrain the preprocessor with current version {self.VERSION}\n"
                f"   Old preprocessors cannot be migrated - full retrain required."
            )

        # ========================================
        # CRITICAL VALIDATION: Scaler state
        # ========================================
        if not hasattr(self.scaler, "mean_"):
            raise ValueError(
                f"❌ SCALER NOT FITTED!\n"
                f"\n"
                f"   StandardScaler is missing 'mean_' attribute.\n"
                f"   This indicates the scaler was never trained.\n"
                f"\n"
                f"   ✅ FIX: Retrain preprocessor from scratch."
            )

        # Validate scaler feature count matches continuous features
        expected_features = len(self._continuous_features)
        actual_features = self.scaler.n_features_in_

        if expected_features != actual_features:
            raise ValueError(
                f"❌ FEATURE COUNT MISMATCH!\n"
                f"   Continuous features: {expected_features}\n"
                f"   Scaler expects: {actual_features}\n"
                f"\n"
                f"   This preprocessor is corrupted.\n"
                f"   ✅ FIX: Retrain from scratch."
            )

        # Transformation-specific attributes
        self.yeo_johnson_transformer = preprocessor.get("yeo_johnson_transformer", None)
        self.clip_lower_ = preprocessor.get("clip_lower_", None)
        self.clip_upper_ = preprocessor.get("clip_upper_", None)
        self.y_min_safe = preprocessor.get("y_min_safe", None)
        self.y_max_safe = preprocessor.get("y_max_safe", None)
        self._log_residual_variance = preprocessor.get("_log_residual_variance", None)
        self._yj_correction_factor = preprocessor.get("_yj_correction_factor", None)
        self._yj_lambda = preprocessor.get("_yj_lambda", None)
        self._yj_residual_variance = preprocessor.get("_yj_residual_variance", None)

        # Load stratified bias correction
        self._bias_var_low = preprocessor.get("_bias_var_low", None)
        self._bias_var_high = preprocessor.get("_bias_var_high", None)
        self._bias_threshold = preprocessor.get("_bias_threshold", None)

        # ========================================
        # VALIDATION: Transformation consistency
        # ========================================
        transformation_method = self.target_transformation.method

        if transformation_method == "log1p":
            if self._log_residual_variance is None:
                logger.warning(
                    "⚠️  BIAS CORRECTION MISSING!\n"
                    "   Transformation: log1p\n"
                    "   Impact: Predictions will be systematically LOW by ~5-10%\n"
                    "   ✅ FIX: Call set_bias_correction() after loading"
                )

        elif transformation_method == "yeo-johnson":
            if self.yeo_johnson_transformer is None:
                raise ValueError(
                    f"❌ Yeo-Johnson transformer missing!\n"
                    f"   inverse_transform() will fail.\n"
                    f"   ✅ FIX: Retrain preprocessor."
                )

            # Validate lambda exists
            if not hasattr(self.yeo_johnson_transformer, "lambdas_"):
                raise ValueError(
                    f"❌ Yeo-Johnson transformer not fitted (missing lambdas_)!\n"
                    f"   ✅ FIX: Retrain preprocessor."
                )

        elif transformation_method.startswith("boxcox"):
            # ── BUG 7 FIX: check lambda_param (canonical) OR boxcox_lambda (deprecated) ──
            _bc_lambda = getattr(self.target_transformation, "lambda_param", None) or getattr(
                self.target_transformation, "boxcox_lambda", None
            )
            if _bc_lambda is None:
                raise ValueError(
                    f"❌ Box-Cox lambda missing!\n"
                    f"   Neither lambda_param nor boxcox_lambda is set.\n"
                    f"   inverse_transform() will fail.\n"
                    f"   ✅ FIX: Retrain preprocessor."
                )

        # ========================================
        # SUCCESS LOGGING
        # ========================================
        logger.info(
            f"✅ Loaded preprocessor from {load_path} (v{version})\n"
            f"  - Size: {file_size:.2f} MB\n"
            f"  - State: {self._state.value}\n"
            f"  - Transformation: {transformation_method}\n"
            f"  - Continuous features: {len(self._continuous_features)}\n"
            f"  - Fit complete: {self._fit_complete}"
        )

        if self.target_min_ is not None:
            logger.info(f"  - Target range: [{self.target_min_:.2f}, {self.target_max_:.2f}]")

        if self._log_residual_variance is not None:
            # ── FIX 2: Sentinel guard ─────────────────────────────────────────
            # Three distinct cases for _log_residual_variance:
            #
            # Case A — NEGATIVE value (e.g. -0.162753):
            #   var = 2 * log(median_ratio), where ratio = y_true / y_pred < 1
            #   means the model over-predicts on this tier.
            #   exp(var/2) = ratio < 1 → downward correction applied by BiasCorrection.
            #   This is a VALID encoding, NOT a sentinel.
            #   It is unusable for CI std derivation (sqrt of negative is undefined),
            #   but that is handled correctly in predict.py's Priority-2 guard (> 0.01).
            #
            # Case B — NEAR-ZERO POSITIVE value (0 < value ≤ 1e-4):
            #   Written by train.py as _MIN_VAR = 1e-6 when the median ratio ≤ 0
            #   (degenerate training run).  exp(1e-6/2) ≈ 1.0 — functionally absent.
            #   This IS a sentinel.
            #
            # Case C — NORMAL POSITIVE value (value > 1e-4):
            #   Typical bias-correction variance (0.05 – 0.30 for insurance data).
            #   Correction is active and meaningful.
            #
            # The original condition `<= 1e-4` incorrectly classified Case A
            # (all negatives) as sentinels, producing a misleading WARNING and
            # the contradictory log pair seen in the predict.py output.
            _SENTINEL_THRESHOLD = 1e-4
            import math as _m_feat

            if self._log_residual_variance < 0:
                # Case A: legitimate negative downward-correction encoding
                _factor = _m_feat.exp(self._log_residual_variance / 2)
                logger.info(
                    f"ℹ️  _log_residual_variance={self._log_residual_variance:.6f} is negative, "
                    f"encoding a downward yeo-johnson correction "
                    f"(exp(var/2)={_factor:.4f}x — model over-predicts on median). "
                    f"Valid bias-correction value, not a sentinel. "
                    f"Unsuitable for CI std derivation; CI will use conformal residuals."
                )
            elif self._log_residual_variance <= _SENTINEL_THRESHOLD:
                # Case B: near-zero positive sentinel
                logger.warning(
                    f"⚠️  _log_residual_variance={self._log_residual_variance:.6f} "
                    f"is a near-zero sentinel (0 < value ≤ {_SENTINEL_THRESHOLD:.0e}), "
                    f"not a real bias-correction value.\n"
                    f"   This occurs when the yeo-johnson model already over-predicts "
                    f"on the training median.\n"
                    f"   CI computation correctly falls back to conformal residuals.\n"
                    f"   Bias correction: ⚠️  sentinel (functionally absent)"
                )
            else:
                # Case C: normal positive variance — correction is active
                logger.info(f"  - Bias correction: ✅ variance={self._log_residual_variance:.6f}")

        return self

    def get_feature_metadata(self) -> dict:
        """Get comprehensive metadata about feature engineering state"""
        return {
            "version": self.VERSION,
            "pipeline_state": self._state.value,
            "fit_complete": self._fit_complete,
            "config": {
                "correlation_threshold": self.config.correlation_threshold,
                "vif_threshold": self.config.vif_threshold,
                "polynomial_degree": self.config.polynomial_degree,
                "outlier_contamination": self.config.outlier_contamination,
                "use_optimized_vif": self.config.use_optimized_vif,
            },
            "target_transformation": {
                "method": self.target_transformation.method,
                # ── BUG 3 FIX ──────────────────────────────────────────────
                # Original: only returned boxcox_lambda — which is None for
                # yeo-johnson models (those use lambda_param only).  Callers
                # inspecting this metadata to get the lambda would silently
                # receive None for every yeo-johnson model.
                #
                # Fix: return the canonical lambda_param field (set for both
                # boxcox and yeo-johnson post F-05 fix), plus boxcox_lambda for
                # backward compatibility with callers that pre-date the migration.
                "lambda_param": self.target_transformation.lambda_param,
                "boxcox_lambda": self.target_transformation.boxcox_lambda,  # deprecated alias
                "original_range": self.target_transformation.original_range,
            },
            "label_encoders": {
                col: {
                    "classes": le.classes_.tolist(),
                    "most_frequent": self.most_frequent_values.get(col, "unknown"),
                }
                for col, le in self.label_encoders.items()
            },
            "onehot_features": (
                list(self._feature_names_snapshot)
                if self._feature_names_snapshot is not None
                else []
            ),
            "polynomial_features": {
                "enabled": self.poly_transformer is not None,
                "feature_count": (
                    len(self._poly_feature_names_snapshot)
                    if self._poly_feature_names_snapshot
                    else 0
                ),
                "degree": self.poly_transformer.degree if self.poly_transformer else None,
                "base_features": (
                    list(self._poly_continuous_features_snapshot)
                    if self._poly_continuous_features_snapshot
                    else []
                ),
            },
            "imputation": {
                "demographic_fitted": self.demographic_imputer is not None,
                "derived_fitted": self.derived_imputer is not None,
                "categorical_fitted": self.categorical_imputer is not None,
            },
            "variance_filtering": {
                "enabled": self.variance_selector is not None,
                "features_selected": (
                    len(self._variance_feature_names_snapshot)
                    if self._variance_feature_names_snapshot
                    else 0
                ),
            },
            "multicollinearity": {
                "removed_features": list(self._removed_collinear_features_snapshot),
                "count": len(self._removed_collinear_features_snapshot),
            },
            "outlier_detection": {
                "method": self.outlier_method,
                "fitted": self.outlier_detector is not None,
                "training_outliers_removed": (
                    len(self._outlier_indices) if self._outlier_indices is not None else 0
                ),
            },
            "scaler_fitted": hasattr(self.scaler, "mean_"),
            "scaler_features": len(self.scaler.mean_) if hasattr(self.scaler, "mean_") else 0,
            "continuous_features": self._continuous_features,
        }

    def reset(self) -> None:
        """Reset pipeline to initial state (for retraining)"""
        logger.info("Resetting pipeline to initial state")
        self.__init__(config=self.config)

    def save(self, filepath: str, version: str | None = None) -> None:
        version = version or self.VERSION

        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "state": self._state.value if hasattr(self, "_state") else "unknown",
            "continuous_features": getattr(self, "_continuous_features", []),
            "scaler_fitted": hasattr(self.scaler, "mean_"),
            "n_features_in": self.scaler.n_features_in_ if hasattr(self.scaler, "mean_") else None,
            "scaler_feature_names": (
                list(self.scaler.feature_names_in_)
                if hasattr(self.scaler, "feature_names_in_")
                else None
            ),
            "label_encoders_count": len(self.label_encoders),
            "has_variance_selector": self.variance_selector is not None,
            "selected_features_count": (
                len(self._variance_feature_names_snapshot)
                if self._variance_feature_names_snapshot is not None
                else None
            ),
            "performance_metrics": self._performance_metrics,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save preprocessor using save_preprocessor method
        self.save_preprocessor(str(filepath))

        # Save metadata
        meta_path = filepath.with_name(filepath.stem + "_metadata.joblib")
        joblib.dump(metadata, meta_path, compress=3)

        logger.info(f"📦 Saved preprocessor metadata to {meta_path}")
        logger.info(f"  - State: {metadata['state']}")
        logger.info(f"  - Continuous features: {len(metadata['continuous_features'])}")
        logger.info(f"  - Scaler fitted: {metadata['scaler_fitted']}")
        if metadata["n_features_in"]:
            logger.info(f"  - Scaler input features: {metadata['n_features_in']}")

    @classmethod
    def load(cls, filepath: str) -> FeatureEngineer:
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")

        # ── BUG 1 FIX ────────────────────────────────────────────────────────
        # cls() called without arguments raises ValueError("FeatureEngineer
        # requires configuration!") because __init__ has no defaults.
        # load_preprocessor() will restore the full config from the .joblib
        # artifact (including the FeatureEngineeringConfig object stored under
        # the "config" key).  We therefore bootstrap with a _stub_ that bypasses
        # __init__ using object.__new__() — then immediately delegate all state
        # restoration to load_preprocessor(), which validates version and
        # required attrs before returning.
        #
        # WHY NOT pass a real config? load() is a classmethod that intentionally
        # has no config parameter — requiring one would force callers to have
        # a config.yaml path just to load an already-trained artifact, defeating
        # the purpose of the method.  The saved preprocessor is the config's
        # ground truth at this point.
        feature_engineer = object.__new__(cls)
        # Initialise the bare minimum attributes that load_preprocessor()
        # reads before it overwrites them (e.g. COMPATIBLE_VERSIONS, VERSION)
        # by delegating to the class definitions directly.
        feature_engineer.load_preprocessor(str(filepath))

        # Try to load metadata if available
        meta_path = filepath.with_name(filepath.stem + "_metadata.joblib")
        if meta_path.exists():
            try:
                metadata = joblib.load(meta_path)
                version = metadata.get("version", "unknown")
                logger.info(f"Loaded preprocessor from {filepath} (v{version})")
                logger.info(f"  - Created: {metadata.get('created_at', 'unknown')}")
                logger.info(f"  - State: {metadata.get('state', 'unknown')}")
                logger.info(
                    f"  - Continuous features: " f"{len(feature_engineer._continuous_features)}"
                )
                if feature_engineer._continuous_features:
                    logger.debug(f"    {feature_engineer._continuous_features}")
                logger.info(f"  - Scaler fitted: {metadata.get('scaler_fitted', 'unknown')}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

        if (
            not hasattr(feature_engineer, "_continuous_features")
            or not feature_engineer._continuous_features
        ):
            logger.warning(
                "⚠️ WARNING: Loaded preprocessor does not have _continuous_features. "
                "This WILL cause prediction errors. You must either:\n"
                "  1. Run the migration script on this preprocessor, OR\n"
                "  2. Retrain from scratch with the updated code"
            )

        if not hasattr(feature_engineer.scaler, "mean_"):
            logger.warning("Scaler is not fitted!")

        return feature_engineer


if __name__ == "__main__":
    import sys

    print("=" * 80)
    print(f"FEATURE ENGINEER v{FeatureEngineer.VERSION}")
    print("=" * 80)
    print("\n✅ ZERO REDUNDANCY ARCHITECTURE:")
    print("  - ✅ ALL values from config.yaml v6.1.0")
    print("  - ✅ NO hardcoded defaults in Python code")
    print("  - ✅ Uses config.py::get_feature_config() helper")
    print("  - ✅ Strict validation with clear error messages")
    print("  - ✅ Compatible with config.py v5.1.0, data.py v5.0.0")
    print("\n✅ CORRECT USAGE:")
    print("  from insurance_ml.config import load_config, get_feature_config")
    print("  ")
    print("  config = load_config()")
    print("  feat_cfg = get_feature_config(config)  # Extracts ALL parameters")
    print("  fe = FeatureEngineer(config_dict=feat_cfg)")
    print("\n❌ WRONG USAGE:")
    print("  fe = FeatureEngineer()  # Will raise error (no defaults!)")
    print("\n" + "=" * 80)

    # Demo with actual config
    try:
        from insurance_ml.config import get_feature_config, load_config

        config = load_config()
        feat_cfg = get_feature_config(config)

        print(f"\n✅ Config loaded successfully:")
        print(f"  - Config version: {config.get('version', 'unknown')}")
        print(f"  - Features extracted: {len(feat_cfg)} parameters")
        print(f"  - Smoker binary map: {feat_cfg['smoker_binary_map']}")
        print(f"  - Smoker risk map: {feat_cfg['smoker_risk_map']}")
        print(f"  - BMI range: [{feat_cfg['bmi_min']}, {feat_cfg['bmi_max']}]")
        print(f"  - VIF threshold: {feat_cfg['vif_threshold']}")

        fe = FeatureEngineer(config_dict=feat_cfg)
        print(f"\n✅ FeatureEngineer initialized successfully!")
        print(f"  - Version: {fe.VERSION}")
        print(f"  - State: {fe._state.value}")
        print(f"  - Config validated: ✓")
        sys.exit(0)

    except FileNotFoundError as fnf:
        # Give explicit guidance about config location
        expected = "configs/config.yaml"
        print("\n❌ CONFIG FILE NOT FOUND")
        print(f"   load_config() expected a config at: {expected}")
        print("   Make sure your project's config.yaml is present and valid.")
        print(f"   Full error: {fnf}")
        sys.exit(2)
    except Exception as e:
        print("\n❌ ERROR during FeatureEngineer demo:")
        print(f"   {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
