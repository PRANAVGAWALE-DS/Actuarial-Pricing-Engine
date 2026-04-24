"""
Shared pytest fixtures for the insurance_ml test suite.

Design rules:
  - No real model files / actual config.yaml required.
  - minimal_config is the canonical fixture for any test that needs
    a dict that passes _validate_config + validate_single_source_of_truth.
  - config_yaml_path writes that dict to a tmp file so load_config() can be
    exercised end-to-end.
  - All model-artifact-dependent fixtures are marked @pytest.mark.model
    and skipped in fast CI (see pyproject.toml markers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Minimal valid config dict
# ---------------------------------------------------------------------------


def make_minimal_config() -> dict[str, Any]:
    """
    Construct a config dict that satisfies every check inside
    _validate_config() and validate_single_source_of_truth().

    Rules:
      - GPU disabled → no xgboost/lightgbm sub-sections required.
      - outlier_removal disabled → contamination not range-checked.
      - polynomial_features disabled → degree/max_features not range-checked.
      - collinearity_removal disabled → vif_threshold not range-checked.
      - optuna.enhanced_scoring disabled → weight sum not checked.
      - diagnostics.enable_confidence_intervals + enable_shap = False →
        explainability key block not enforced.
      - NO data.validation_size → avoids SSOT violation with training.val_size.
    """
    return {
        "version": "test-1.0",
        "environment": "test",
        "defaults": {
            "random_state": 42,
            "n_jobs": 1,
        },
        "cross_validation": {
            "n_folds": 5,
            "shuffle": True,
            "stratified": False,
            "random_state": 42,
        },
        "gpu": {
            # enabled=False by default.  xgboost/lightgbm sub-sections are
            # included regardless so that if USE_GPU env var is truthy on the
            # developer's machine, _apply_env_overrides flips enabled=True and
            # _validate_config can still find the required sub-sections without
            # crashing.  Having them present when enabled=False is harmless.
            "enabled": False,
            "device_id": 0,
            "memory_limit_mb": 3000,
            "memory_fraction": 0.75,
            "validate_memory": False,
            "monitor_usage": False,
            "log_memory_stats": False,
            "warn_threshold_mb": 2800,
            "fallback_to_cpu": True,
            "cpu_fallback_on_oom": True,
            "retry_with_reduced_params": False,
            "xgboost": {
                "device": "cpu",
                "max_bin": 256,
                "sampling_method": "uniform",
                "grow_policy": "depthwise",
                "max_cached_hist_node": 1024,
                "n_jobs": 1,
            },
            "xgboost_median": {
                "device": "cpu",
                "max_bin": 256,
                "sampling_method": "uniform",
                "grow_policy": "depthwise",
                "max_cached_hist_node": 1024,
                "n_jobs": 1,
            },
            "lightgbm": {
                "device": "cpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
                "max_bin": 255,
                "gpu_use_dp": False,
                "n_jobs": 1,
            },
        },
        "hardware": {
            "gpu_model": "RTX-3050-test",
            "total_vram_mb": 4096,
            "max_safe_vram_mb": 3500,
            "recommended_vram_limit_mb": 3000,
        },
        "data": {
            "raw_path": "data/raw/insurance.csv",
            "target_column": "charges",
            "test_size": 0.2,
            "random_state": 42,
            # NOTE: data.validation_size deliberately absent (SSOT — only
            # training.val_size is the source of truth).
        },
        "features": {
            "engineering": {
                "variance_threshold": 1e-6,
            },
            "target_transform": {
                "method": "none",
            },
            "outlier_removal": {
                "enabled": False,
                "contamination": 0.05,
                "random_state": 42,
            },
            "polynomial_features": {
                "enabled": False,
                "degree": 2,
                "max_features": 50,
            },
            "collinearity_removal": {
                "enabled": False,
                "threshold": 0.90,
                "vif_threshold": 10.0,
                "max_vif_iterations": 5,
                "use_optimized_vif": True,
            },
            "encoding": {
                "smoker_binary_map": {"yes": 1, "no": 0},
                "smoker_risk_map": {"yes": 2, "no": 0},
            },
            "age_min": 0.0,
            "age_max": 120.0,
            "bmi_min": 10.0,
            "bmi_max": 100.0,
            "children_min": 0,
            "children_max": 20,
        },
        "model": {
            "models": ["xgboost"],
            "metric": "rmse",
            "model_path": "models/",
        },
        "models": {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
            }
        },
        "optuna": {
            "n_trials": 20,
            "timeout": 600,
            "n_jobs": 1,
            "random_state": 42,
            "enhanced_scoring": {
                "enabled": False,
            },
        },
        "training": {
            "output_dir": "models/",
            "reports_dir": "reports/",
            "test_size": 0.2,
            "val_size": 0.1,
            "stratify_splits": False,
            "min_r2_threshold": 0.0,
            "enable_mlflow": False,
            "enable_optuna": False,
            "enable_diagnostics": False,
            "training_timeout": 600,
            "max_model_size_mb": 512,
            "max_memory_mb": 4096,
            "verify_checksums": True,
            "save_checksums": True,
            "register_to_mlflow": False,
            "halt_on_severe_shift": False,
            "batch_size": 128,
            "use_sample_weights": False,
            "high_value_percentile": 95,
            "memory_fraction": 0.75,
            "checkpoint": {
                "enabled": False,
                "frequency": 10,
            },
            "memory": {
                "cleanup_frequency": 5,
                "force_gc": False,
                "clear_gpu_cache": False,
            },
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.0,
            },
            "two_model_architecture": {
                "enabled": False,
                "pricing_model": "xgboost",
                "risk_model": "xgboost",
                "overpricing_gate_model": "xgboost",
                "risk_model_alpha": 0.5,
            },
            "deployment_gates": {
                "g6_min_cost_weighted_r2": 0.0,
                "g7_max_overpricing_rate": 1.0,
                "g3_max_width_ratio": 10.0,
            },
            "provenance": {
                "capture_git_hash": False,
                "require_clean_tree": False,
                "random_state_in_artifact": True,
                "always_write_bias_correction": True,
            },
            "conformal_intervals": {
                "method": "heteroscedastic_conformal",
                "fallback": "symmetric_conformal",
                "target_coverage": 0.9,
                "max_width_ratio": 3.0,
                "target_width_ratio": 1.0,
            },
        },
        "diagnostics": {
            "enabled": False,
            "max_samples": 1000,
            "batch_size": 100,
            "rf_tree_batch_size": 10,
            "enable_confidence_intervals": False,
            "confidence_level": 0.90,
            "enable_shap": False,
            "auto_plot": False,
            "performance": {
                "enabled": False,
                "log_memory": False,
                "log_gpu_memory": False,
                "log_training_time": False,
                "log_prediction_time": False,
            },
            "plots": {
                "learning_curves": False,
                "residuals": False,
                "error_distribution": False,
                "calibration": False,
                "feature_importance": False,
                "partial_dependence": False,
                "shap": False,
                "save": False,
                "format": "png",
                "dpi": 100,
            },
            "reports": {"html": False, "pdf": False},
            "display": {
                "top_features_pdp": 5,
                "worst_predictions_show": 10,
                "distribution_shift_top_features": 5,
                "min_sample_for_save": 100,
            },
            "sampling": {
                "shap_max_samples": 100,
                "shap_background_samples": 50,
                "residual_sample_size": 500,
                "autocorr_lag_limit": 10,
                "calibration_bins": 10,
                "learning_curve_points": 5,
                "permutation_importance_repeats": 3,
            },
        },
        "mlflow": {
            "tracking": {
                "enabled": False,
                "tracking_uri": "sqlite:///data/databases/mlflow.db",
                "experiment_name": "test",
                "run_name_prefix": "unit",
            },
            "registry": {
                "enabled": False,
                "model_name": "insurance-ml",
                "register_best_only": True,
            },
            "logging": {
                "level": "INFO",
                "log_metrics": True,
                "log_params": True,
                "log_artifacts": False,
                "log_models": False,
                "log_system_metrics": False,
            },
            "autolog": {
                "sklearn": False,
                "xgboost": False,
                "lightgbm": False,
                "disable": True,
            },
            "log_gpu_metrics": False,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "file": "logs/test.log",
        },
        "high_value_analysis": {
            "enabled": False,
            "threshold_percentile": 95,
            "baseline_model": "linear_regression",
            "compare_against_baseline": False,
            "reporting": {
                "save_report": False,
                "format": "html",
            },
        },
        "overfitting_analysis": {
            "enabled": False,
        },
        "monitoring": {
            "drift_detection_enabled": False,
        },
        "validation": {
            "contamination_min": 0.0,
            "contamination_max": 0.5,
            "vif_min": 1.0,
            "polynomial_degree_min": 1,
            "valid_xgboost_devices": ["cpu", "cuda", "cuda:0"],
            "valid_target_transform_methods": ["none", "log1p", "yeo-johnson"],
            "weight_sum_tolerance": 0.01,
        },
        "sample_weights": {
            "enabled": False,
            "method": "none",
            "tiers": [],
            "transform": "none",
        },
        "prediction": {
            "max_batch_size": 1000,
        },
        "conformal": {
            "calibration_split_ratio": 0.2,
        },
        "hybrid_predictor": {
            "threshold": 4500.0,
            "blend_ratio": 0.70,
            "use_soft_blending": True,
            "soft_blend_window": 500.0,
            "max_actuarial_uplift_ratio": 1.15,
            "calibration": {
                "enabled": True,
                "factor": 1.00,
                "apply_to_ml_only": True,
            },
            "business_config": {
                "base_profit_margin": 0.03,
                "admin_cost_per_policy": 25.0,
                "churn_sensitivity": 1.0,
            },
        },
        "evaluation": {
            "segment_thresholds": {
                "low_value": 4500.0,
                "standard": 15000.0,
                "high_value": 30000.0,
            }
        },
    }


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def minimal_config() -> dict[str, Any]:
    """Minimal config dict that passes all validation checks."""
    return make_minimal_config()


@pytest.fixture()
def mutable_config() -> dict[str, Any]:
    """Fresh copy of minimal_config for tests that mutate it."""
    import copy

    return copy.deepcopy(make_minimal_config())


@pytest.fixture()
def config_yaml_path(tmp_path: Path) -> Path:
    """Write minimal config to a tmp yaml file; returns the path."""
    cfg = make_minimal_config()
    p = tmp_path / "configs" / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    return p


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """
    20-row insurance DataFrame with the raw schema.
    Covers all combinations of smoker / sex / region so segment tests work.
    """
    rng = np.random.default_rng(42)
    n = 20
    regions = ["northeast", "northwest", "southeast", "southwest"]
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, n),
            "sex": rng.choice(["male", "female"], n),
            "bmi": rng.uniform(18.0, 40.0, n).round(1),
            "children": rng.integers(0, 5, n),
            "smoker": rng.choice(["yes", "no"], n),
            "region": rng.choice(regions, n),
            "charges": rng.uniform(1200.0, 50000.0, n).round(2),
        }
    )


@pytest.fixture(scope="session")
def large_sample_df() -> pd.DataFrame:
    """
    200-row DataFrame — enough for segment tests that need >=10 per segment.
    Charges span all four business segments so every mask is non-empty.
    """
    rng = np.random.default_rng(0)
    n = 200
    regions = ["northeast", "northwest", "southeast", "southwest"]
    charges = np.concatenate(
        [
            rng.uniform(500.0, 4500.0, 50),  # low_risk  (< $4,500)
            rng.uniform(4501.0, 15000.0, 80),  # standard
            rng.uniform(15001.0, 30000.0, 40),  # high_risk
            rng.uniform(30001.0, 70000.0, 30),  # catastrophic
        ]
    )
    rng.shuffle(charges)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, n),
            "sex": rng.choice(["male", "female"], n),
            "bmi": rng.uniform(18.0, 45.0, n).round(1),
            "children": rng.integers(0, 5, n),
            "smoker": rng.choice(["yes", "no"], n),
            "region": rng.choice(regions, n),
            "charges": charges.round(2),
        }
    )


# ---------------------------------------------------------------------------
# BiasCorrection fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bc_2tier():
    """Standard 2-tier BiasCorrection instance."""
    from insurance_ml.features import BiasCorrection

    return BiasCorrection(var_low=0.04, var_high=0.09, threshold=15_000.0)


@pytest.fixture(scope="session")
def bc_3tier():
    """Standard 3-tier BiasCorrection instance."""
    from insurance_ml.features import BiasCorrection

    return BiasCorrection(
        var_low=0.04,
        var_high=0.09,
        threshold=0.0,  # sentinel (unused in 3-tier routing)
        var_mid=0.06,
        threshold_low=10_000.0,
        threshold_high=20_000.0,
    )


# ---------------------------------------------------------------------------
# DriftMonitor baseline fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def drift_baseline_path(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Write a drift baseline JSON for sample_df and return its path."""
    from insurance_ml.monitoring import DriftMonitor

    out = tmp_path / "drift_baseline.json"
    DriftMonitor.create_baseline(
        X_train=sample_df.drop(columns=["charges"]),
        y_train=sample_df["charges"],
        output_path=str(out),
        overwrite=True,
    )
    return out


# ---------------------------------------------------------------------------
# Routes / MetricsCollector fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_metrics():
    """Return a new MetricsCollector with sentinel startup_time."""
    from api.routes import MetricsCollector

    return MetricsCollector()
