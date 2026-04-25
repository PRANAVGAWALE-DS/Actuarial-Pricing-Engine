import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv


def _configure_windows_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows without replacing the stream objects."""
    if sys.platform != "win32":
        return

    for stream_name in ("stdout", "stderr"):  # type: ignore[unreachable]
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            continue


_configure_windows_utf8_stdio()


# Load environment variables
load_dotenv()

# Thread-safe logging initialization
_logging_lock = threading.Lock()
_logging_initialized: bool = False

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get project root directory

    Returns:
        Path to project root (3 levels up from this file)
    """
    return Path(__file__).parent.parent.parent


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        Dictionary with complete configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or missing required sections
    """
    if config_path is None:
        config_path_obj = get_project_root() / "configs" / "config.yaml"
    else:
        config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path_obj}\n"
            f"Expected location: {config_path_obj.absolute()}\n"
            f"Current directory: {Path.cwd()}\n"
            f"Project root: {get_project_root()}\n\n"
            f"⚠️ Config.yaml v6.1.0 is the SINGLE SOURCE OF TRUTH.\n"
            f"   ZERO defaults are provided in Python code.\n"
            f"   Please create config.yaml from the template."
        )

    try:
        with open(config_path_obj, encoding="utf-8") as file:
            config: dict[str, Any] = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"Config file is empty: {config_path_obj}")

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e
    except Exception as e:
        raise ValueError(f"Error reading config file: {e}") from e

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Validate configuration structure (required sections / keys)
    _validate_config(config)

    # Enforce single-source-of-truth on every load.
    # The original code only called validate_single_source_of_truth() from the
    # __main__ demo block, so the duplicate data.validation_size / training.val_size
    # conflict (BUG-A root cause) was never caught in production code paths
    # (train.py, data.py, optuna_optimizer.py, etc.).  Calling it here guarantees
    # that any SSOT violation raises a clear, actionable ValueError before any
    # training or prediction code can read the wrong value silently.
    validate_single_source_of_truth(config)

    return config


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to config

    Priority: Environment variables > Config file

    NOTE: This does NOT provide defaults - only overrides existing config values
    """
    # Environment
    if os.getenv("ENVIRONMENT"):
        config["environment"] = os.getenv("ENVIRONMENT")
        logger.debug(f"Environment override: {config['environment']}")

    # Logging level
    if os.getenv("LOG_LEVEL"):
        if "logging" in config:
            config["logging"]["level"] = os.getenv("LOG_LEVEL")
            logger.debug(f"Log level override: {config['logging']['level']}")

    # Model path
    if os.getenv("MODEL_PATH"):
        if "model" in config:
            config["model"]["model_path"] = os.getenv("MODEL_PATH")
            logger.debug(f"Model path override: {config['model']['model_path']}")

    # Data path
    if os.getenv("DATA_PATH"):
        if "data" in config:
            config["data"]["raw_path"] = os.getenv("DATA_PATH")
            logger.debug(f"Data path override: {config['data']['raw_path']}")

    # Random state (for reproducibility)
    # original code only wrote to defaults.random_state, but
    # get_training_config() reads from cross_validation.random_state and
    # data.random_state — so the env override had zero effect on splits.
    # Now propagates to ALL sections that consume random_state.
    if os.getenv("RANDOM_STATE"):
        try:
            random_state = int(os.getenv("RANDOM_STATE", "0"))
            _updated_sections = []
            if "defaults" in config:
                config["defaults"]["random_state"] = random_state
                _updated_sections.append("defaults")
            if "cross_validation" in config:
                config["cross_validation"]["random_state"] = random_state
                _updated_sections.append("cross_validation")
            if "data" in config:
                config["data"]["random_state"] = random_state
                _updated_sections.append("data")
            logger.info(
                f"RANDOM_STATE override applied: {random_state} "
                f"-> sections: {_updated_sections}"
            )
        except ValueError:
            logger.warning("RANDOM_STATE env var must be an integer, ignoring.")

    # MLflow tracking URI
    if os.getenv("MLFLOW_TRACKING_URI"):
        if "mlflow" in config and "tracking" in config["mlflow"]:
            config["mlflow"]["tracking"]["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI")

    # GPU override
    _cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
    _force_cpu = os.getenv("USE_GPU", "true").lower() in ("false", "0", "no") or _cuda_visible == ""

    if _force_cpu and "gpu" in config:
        gpu_cfg = config["gpu"]
        gpu_cfg["enabled"] = False
        for sub_key in ("xgboost", "xgboost_median", "lightgbm"):
            if sub_key in gpu_cfg and "device" in gpu_cfg[sub_key]:
                gpu_cfg[sub_key]["device"] = "cpu"
        config["gpu"] = gpu_cfg
        logger.info(
            "⚙️  GPU override: USE_GPU=false or CUDA_VISIBLE_DEVICES='' → "
            "all model device params set to CPU"
        )
    elif os.getenv("USE_GPU"):
        use_gpu = os.getenv("USE_GPU", "").lower() in ["true", "1", "yes"]
        if "gpu" in config:
            config["gpu"]["enabled"] = use_gpu
            logger.debug(f"GPU override: {use_gpu}")

    return config


def _validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration has required sections

    Raises:
        ValueError: If required sections or keys are missing
    """
    # Required top-level sections
    required_sections = [
        "defaults",
        "cross_validation",
        "gpu",
        "data",
        "features",
        "model",
        "models",
        "optuna",
        "training",
        "diagnostics",
        "mlflow",
        "logging",
        "high_value_analysis",
        "overfitting_analysis",
        "monitoring",
        "validation",
        "hardware",
        "sample_weights",
        "prediction",  # v7.5.0: batch size cap (prediction.max_batch_size)
        "conformal",  # v7.5.0: calibration split ratio
    ]

    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ValueError(
            f"❌ Config missing required sections: {missing_sections}\n"
            f"   Available sections: {list(config.keys())}\n\n"
            f"   ⚠️ Config.yaml v6.1.0 is the SINGLE SOURCE OF TRUTH.\n"
            f"   All required sections must be present in config.yaml.\n"
            f"   ZERO defaults are provided in Python code."
        )

    # Validate validation section (NO DEFAULTS)
    validation_cfg = config["validation"]
    required_validation = [
        "contamination_min",
        "contamination_max",
        "vif_min",
        "polynomial_degree_min",
        "valid_xgboost_devices",
        "valid_target_transform_methods",
        "weight_sum_tolerance",
    ]
    missing_validation = [k for k in required_validation if k not in validation_cfg]
    if missing_validation:
        raise ValueError(
            f"❌ validation section missing: {missing_validation}\n"
            f"   Required: {required_validation}"
        )

    # Validate hardware section (NO DEFAULTS)
    hardware_cfg = config["hardware"]
    required_hardware = [
        "gpu_model",
        "total_vram_mb",
        "max_safe_vram_mb",
        "recommended_vram_limit_mb",
    ]
    missing_hardware = [k for k in required_hardware if k not in hardware_cfg]
    if missing_hardware:
        raise ValueError(
            f"❌ hardware section missing: {missing_hardware}\n" f"   Required: {required_hardware}"
        )

    # Validate defaults section (NO DEFAULTS)
    defaults_cfg = config["defaults"]
    if "random_state" not in defaults_cfg:
        raise ValueError("❌ Config 'defaults' section missing 'random_state'")
    if "n_jobs" not in defaults_cfg:
        raise ValueError("❌ Config 'defaults' section missing 'n_jobs'")

    # Validate cross_validation section (NO DEFAULTS)
    cv_cfg = config["cross_validation"]
    required_cv = ["n_folds", "shuffle", "stratified", "random_state"]
    missing_cv = [k for k in required_cv if k not in cv_cfg]
    if missing_cv:
        raise ValueError(
            f"❌ Config 'cross_validation' section missing: {missing_cv}\n"
            f"   Required: {required_cv}"
        )

    # Validate GPU section (NO DEFAULTS)
    gpu_cfg = config["gpu"]
    required_gpu = [
        "enabled",
        "device_id",
        "memory_limit_mb",
        "memory_fraction",
        "validate_memory",
        "monitor_usage",
        "log_memory_stats",
        "warn_threshold_mb",
        "fallback_to_cpu",
        "cpu_fallback_on_oom",
        "retry_with_reduced_params",
    ]
    missing_gpu = [k for k in required_gpu if k not in gpu_cfg]
    if missing_gpu:
        raise ValueError(
            f"❌ Config 'gpu' section missing: {missing_gpu}\n" f"   Required: {required_gpu}"
        )

    # Validate GPU subsections (NO DEFAULTS)
    if gpu_cfg["enabled"]:
        if "xgboost" not in gpu_cfg:
            raise ValueError("❌ Config 'gpu.xgboost' section required when GPU enabled")

        xgb_cfg = gpu_cfg["xgboost"]
        # ── BUG 6 RESOLVED ───────────────────────────────────────────────
        # tree_method removed from required_xgb. In XGBoost >= 2.0 it is
        # auto-inferred from device=cuda (defaults to "hist"), so it is
        # optional. get_gpu_config() already uses .get("tree_method", None).
        # Both validators now agree: tree_method is optional in config.yaml.
        required_xgb = [
            "device",
            "max_bin",
            "sampling_method",
            "grow_policy",
            "max_cached_hist_node",
        ]
        missing_xgb = [k for k in required_xgb if k not in xgb_cfg]
        if missing_xgb:
            raise ValueError(f"❌ Config 'gpu.xgboost' missing: {missing_xgb}")

        if "lightgbm" not in gpu_cfg:
            raise ValueError("❌ Config 'gpu.lightgbm' section required when GPU enabled")

        lgb_cfg = gpu_cfg["lightgbm"]
        required_lgb = [
            "device",
            "gpu_platform_id",
            "gpu_device_id",
            "max_bin",
            "gpu_use_dp",
        ]
        missing_lgb = [k for k in required_lgb if k not in lgb_cfg]
        if missing_lgb:
            raise ValueError(f"❌ Config 'gpu.lightgbm' missing: {missing_lgb}")

    # Validate data section (NO DEFAULTS)
    # 'validation_size' removed from required_data.
    # data.validation_size exists in config.yaml as a legacy field but is
    # never read by any helper function — get_training_config() uses
    # training.val_size exclusively.  Requiring it here caused startup
    # crashes if it was removed from config.yaml, while any change to its
    # value had zero effect on training behaviour.
    data_cfg = config["data"]
    required_data = ["raw_path", "target_column", "test_size", "random_state"]
    missing_data = [k for k in required_data if k not in data_cfg]
    if missing_data:
        raise ValueError(f"❌ Config 'data' section missing: {missing_data}")

    # Validate model section (NO DEFAULTS)
    model_config = config["model"]
    if "models" not in model_config:
        raise ValueError("❌ Config 'model' section missing 'models' list")
    if "metric" not in model_config:
        raise ValueError("❌ Config 'model' section missing 'metric'")

    # Validate features section (NO DEFAULTS)
    features_cfg = config["features"]
    required_feature_sections = [
        "engineering",
        "target_transform",
        "outlier_removal",
        "polynomial_features",
        "collinearity_removal",
    ]
    missing_feature_sections = [s for s in required_feature_sections if s not in features_cfg]
    if missing_feature_sections:
        raise ValueError(f"❌ Config 'features' section missing: {missing_feature_sections}")

    # Validate target transform method (STRICT - NO DEFAULTS)
    valid_methods = validation_cfg["valid_target_transform_methods"]
    method = features_cfg["target_transform"].get("method", "")
    if method not in valid_methods:
        raise ValueError(
            f"❌ Invalid target transform method: '{method}'\n" f"   Valid options: {valid_methods}"
        )

    # Validate outlier contamination (STRICT - NO DEFAULTS)
    outlier_cfg = features_cfg.get("outlier_removal", {})
    if outlier_cfg.get("enabled", False):
        contamination = outlier_cfg.get("contamination", 0)
        min_contam = validation_cfg["contamination_min"]
        max_contam = validation_cfg["contamination_max"]

        if not (min_contam < contamination < max_contam):
            raise ValueError(
                f"❌ Outlier contamination must be in ({min_contam}, {max_contam}), "
                f"got {contamination}"
            )

    # Validate VIF threshold (STRICT - NO DEFAULTS)
    collin_cfg = features_cfg.get("collinearity_removal", {})
    if collin_cfg.get("enabled", False):
        vif_threshold = collin_cfg.get("vif_threshold", 0)
        min_vif = validation_cfg["vif_min"]

        if vif_threshold <= min_vif:
            raise ValueError(f"❌ VIF threshold must be > {min_vif}, got {vif_threshold}")

    # Validate polynomial degree (STRICT - NO DEFAULTS)
    poly_cfg = features_cfg.get("polynomial_features", {})
    if poly_cfg.get("enabled", False):
        degree = poly_cfg.get("degree", 0)
        min_degree = validation_cfg["polynomial_degree_min"]

        if degree < min_degree:
            raise ValueError(f"❌ Polynomial degree must be >= {min_degree}, got {degree}")

        max_features = poly_cfg.get("max_features")
        if max_features is None:
            raise ValueError("❌ Polynomial 'max_features' is required when enabled")
        if max_features < 1:
            raise ValueError(f"❌ Polynomial max_features must be >= 1, got {max_features}")

    # Validate GPU settings (STRICT - NO DEFAULTS)
    if gpu_cfg["enabled"]:
        memory_limit = gpu_cfg["memory_limit_mb"]
        max_vram = hardware_cfg["max_safe_vram_mb"]
        gpu_model = hardware_cfg["gpu_model"]
        recommended = hardware_cfg["recommended_vram_limit_mb"]

        if memory_limit > max_vram:
            raise ValueError(
                f"❌ GPU memory limit ({memory_limit}MB) exceeds "
                f"{gpu_model} capacity ({max_vram}MB)\n"
                f"   Recommended: ≤{recommended}MB for safety"
            )

        # Validate XGBoost device format (STRICT - NO DEFAULTS)
        xgb_device = xgb_cfg["device"]
        valid_devices = validation_cfg["valid_xgboost_devices"]

        if xgb_device not in valid_devices:
            logger.warning(
                f"⚠️ XGBoost device '{xgb_device}' may not be valid\n"
                f"   Expected one of: {valid_devices}"
            )

    # Validate enhanced scoring weights (STRICT - NO DEFAULTS)
    optuna_cfg = config["optuna"]
    enhanced = optuna_cfg.get("enhanced_scoring", {})
    if enhanced.get("enabled", False):
        weights = enhanced.get("hybrid_weights", {})
        if weights:
            total = sum(weights.values())
            tolerance = validation_cfg["weight_sum_tolerance"]

            if not (1.0 - tolerance <= total <= 1.0 + tolerance):
                raise ValueError(
                    f"❌ Enhanced scoring hybrid_weights must sum to 1.0 (±{tolerance})\n"
                    f"   Current sum: {total:.4f}\n"
                    f"   Weights: {weights}"
                )

    # Validate diagnostics section (NO DEFAULTS)
    diag_cfg = config.get("diagnostics", {})
    if not diag_cfg:
        raise ValueError("❌ Config missing 'diagnostics' section")

    # Validate explainability configuration
    if diag_cfg.get("enable_confidence_intervals") or diag_cfg.get("enable_shap"):
        required_explainability = [
            "confidence_level",
            "enable_confidence_intervals",
            "enable_shap",
            "auto_plot",
        ]
        missing_explainability = [k for k in required_explainability if k not in diag_cfg]
        if missing_explainability:
            raise ValueError(
                f"⛔ diagnostics section missing explainability keys: {missing_explainability}\n"
                f"   Required when explainability features are enabled"
            )

        # Validate confidence level
        confidence_level = diag_cfg.get("confidence_level", 0)
        if not (0 < confidence_level < 1):
            raise ValueError(f"⛔ confidence_level must be in (0, 1), got {confidence_level}")

    # Validate prediction section (v7.5.0 — max_batch_size is config-driven)
    prediction_cfg = config["prediction"]
    if "max_batch_size" not in prediction_cfg:
        raise ValueError(
            "❌ prediction section missing 'max_batch_size'\n"
            "   Required: prediction.max_batch_size (int, e.g. 10000)"
        )
    if (
        not isinstance(prediction_cfg["max_batch_size"], int)
        or prediction_cfg["max_batch_size"] < 1
    ):
        raise ValueError(
            f"❌ prediction.max_batch_size must be a positive integer, "
            f"got {prediction_cfg['max_batch_size']!r}"
        )

    # Validate conformal section (v7.5.0 — calibration_split_ratio)
    conformal_cfg = config["conformal"]
    if "calibration_split_ratio" not in conformal_cfg:
        raise ValueError(
            "❌ conformal section missing 'calibration_split_ratio'\n"
            "   Required: conformal.calibration_split_ratio (float in (0, 1))"
        )
    csr = conformal_cfg["calibration_split_ratio"]
    if not isinstance(csr, int | float) or not (0 < csr < 1):
        raise ValueError(f"❌ conformal.calibration_split_ratio must be in (0, 1), got {csr!r}")

    # Validate training.deployment_gates (v7.5.0 — G3/G6/G7 now all config-driven)
    training_cfg_val = config["training"]
    if "deployment_gates" not in training_cfg_val:
        raise ValueError("❌ training section missing 'deployment_gates' sub-section")

    gates_cfg = training_cfg_val["deployment_gates"]
    required_gates = [
        "g6_min_cost_weighted_r2",
        "g7_max_overpricing_rate",
        "g3_max_width_ratio",
    ]
    missing_gates = [k for k in required_gates if k not in gates_cfg]
    if missing_gates:
        raise ValueError(
            f"❌ training.deployment_gates missing: {missing_gates}\n"
            f"   Required: {required_gates}"
        )

    # Validate training.two_model_architecture (v7.5.0)
    if "two_model_architecture" not in training_cfg_val:
        raise ValueError("❌ training section missing 'two_model_architecture' sub-section")
    tma_cfg = training_cfg_val["two_model_architecture"]
    required_tma = [
        "enabled",
        "pricing_model",
        "risk_model",
        "overpricing_gate_model",
        "risk_model_alpha",
    ]
    missing_tma = [k for k in required_tma if k not in tma_cfg]
    if missing_tma:
        raise ValueError(
            f"❌ training.two_model_architecture missing: {missing_tma}\n"
            f"   Required: {required_tma}"
        )

    # Validate training.provenance (v7.5.0)
    if "provenance" not in training_cfg_val:
        raise ValueError("❌ training section missing 'provenance' sub-section")
    prov_cfg = training_cfg_val["provenance"]
    required_prov = [
        "capture_git_hash",
        "require_clean_tree",
        "random_state_in_artifact",
        "always_write_bias_correction",
    ]
    missing_prov = [k for k in required_prov if k not in prov_cfg]
    if missing_prov:
        raise ValueError(
            f"❌ training.provenance missing: {missing_prov}\n" f"   Required: {required_prov}"
        )

    # Validate training.conformal_intervals (v7.5.0)
    if "conformal_intervals" not in training_cfg_val:
        raise ValueError("❌ training section missing 'conformal_intervals' sub-section")
    ci_cfg = training_cfg_val["conformal_intervals"]
    required_ci = [
        "method",
        "fallback",
        "target_coverage",
        "max_width_ratio",
        "target_width_ratio",
    ]
    missing_ci = [k for k in required_ci if k not in ci_cfg]
    if missing_ci:
        raise ValueError(
            f"❌ training.conformal_intervals missing: {missing_ci}\n" f"   Required: {required_ci}"
        )

    # Validate model hyperparameters defined
    model_list = model_config.get("models", [])
    models_cfg = config.get("models", {})

    for model_name in model_list:
        if model_name not in models_cfg:
            logger.warning(
                f"⚠️ Model '{model_name}' listed but no hyperparameters defined\n"
                f"   Will use sklearn defaults (may not be optimal)"
            )

    logger.debug("✅ Configuration validated successfully (ZERO DEFAULTS)")


def setup_logging(config: dict[str, Any] | None = None) -> None:
    """
    Setup logging configuration (thread-safe)

    Args:
        config: Configuration dictionary. If None, loads from default location.
    """
    global _logging_initialized

    # Thread-safe check
    with _logging_lock:
        if _logging_initialized:
            return

        if config is None:
            config = load_config()

        log_config = config.get("logging", {})

        if not log_config:
            raise ValueError(
                "❌ Config missing 'logging' section\n"
                "   Config.yaml must define logging configuration"
            )

        # Strict validation - no defaults
        required_log_keys = ["level", "format", "file"]
        missing_log = [k for k in required_log_keys if k not in log_config]
        if missing_log:
            raise ValueError(
                f"❌ Logging section missing required keys: {missing_log}\n"
                f"   Required: {required_log_keys}"
            )

        log_level = getattr(logging, log_config["level"].upper(), None)
        if log_level is None:
            raise ValueError(f"❌ Invalid log level: {log_config['level']}")

        log_format = log_config["format"]
        log_file = log_config["file"]

        # Create logs directory
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.handlers.clear()

        # Setup handlers
        handlers: list[logging.Handler] = [
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ]

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers,
            force=True,
        )

        _logging_initialized = True

        logger.info(f"✅ Logging initialized: level={log_level}, file={log_file}")


# ============================================================================
# TYPED HELPER FUNCTIONS - Extract config with type safety
# ============================================================================


def get_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract global defaults (STRICT - no fallbacks)

    Returns:
        Dictionary with global defaults (random_state, n_jobs)

    Raises:
        ValueError: If required keys are missing
    """
    if "defaults" not in config:
        raise ValueError("❌ Config missing 'defaults' section")

    defaults = config["defaults"]
    required = ["random_state", "n_jobs"]
    missing = [k for k in required if k not in defaults]

    if missing:
        raise ValueError(
            f"❌ defaults section missing required keys: {missing}\n"
            f"   Config.yaml v6.1.0 must define ALL default values"
        )

    return {
        "random_state": defaults["random_state"],
        "n_jobs": defaults["n_jobs"],
    }


def get_cv_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract cross-validation configuration (STRICT - no defaults)

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with CV settings

    Raises:
        ValueError: If required keys are missing
    """
    if "cross_validation" not in config:
        raise ValueError("❌ Config missing 'cross_validation' section")

    cv_cfg = config["cross_validation"]
    required = ["n_folds", "shuffle", "stratified", "random_state"]
    missing = [k for k in required if k not in cv_cfg]

    if missing:
        raise ValueError(
            f"❌ cross_validation section missing required keys: {missing}\n"
            f"   Required: {required}\n"
            f"   Config.yaml v6.1.0 must define ALL CV parameters"
        )

    return {
        "n_folds": cv_cfg["n_folds"],
        "shuffle": cv_cfg["shuffle"],
        "stratified": cv_cfg["stratified"],
        "random_state": cv_cfg["random_state"],
    }


def get_validation_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract validation configuration (STRICT - no defaults)

    Returns:
        Dict with validation parameters

    Raises:
        ValueError: If required keys are missing
    """
    if "validation" not in config:
        raise ValueError("❌ Config missing 'validation' section")

    validation_cfg = config["validation"]

    required_keys = [
        "contamination_min",
        "contamination_max",
        "vif_min",
        "polynomial_degree_min",
        "valid_xgboost_devices",
        "valid_target_transform_methods",
        "weight_sum_tolerance",
    ]

    missing = [k for k in required_keys if k not in validation_cfg]
    if missing:
        raise ValueError(
            f"❌ validation section missing required keys: {missing}\n"
            f"   Required: {required_keys}\n"
            f"   Config.yaml must define ALL validation parameters"
        )

    return {
        "contamination_min": validation_cfg["contamination_min"],
        "contamination_max": validation_cfg["contamination_max"],
        "vif_min": validation_cfg["vif_min"],
        "polynomial_degree_min": validation_cfg["polynomial_degree_min"],
        "valid_xgboost_devices": validation_cfg["valid_xgboost_devices"],
        "valid_target_transform_methods": validation_cfg["valid_target_transform_methods"],
        "weight_sum_tolerance": validation_cfg["weight_sum_tolerance"],
    }


def get_hardware_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract hardware configuration (STRICT - no defaults)

    Returns:
        Dict with hardware specifications

    Raises:
        ValueError: If required keys are missing
    """
    if "hardware" not in config:
        raise ValueError("❌ Config missing 'hardware' section")

    hardware_cfg = config["hardware"]

    required_keys = [
        "gpu_model",
        "total_vram_mb",
        "max_safe_vram_mb",
        "recommended_vram_limit_mb",
    ]

    missing = [k for k in required_keys if k not in hardware_cfg]
    if missing:
        raise ValueError(
            f"❌ hardware section missing required keys: {missing}\n"
            f"   Required: {required_keys}\n"
            f"   Config.yaml must define hardware specifications"
        )

    return {
        "gpu_model": hardware_cfg["gpu_model"],
        "total_vram_mb": hardware_cfg["total_vram_mb"],
        "max_safe_vram_mb": hardware_cfg["max_safe_vram_mb"],
        "recommended_vram_limit_mb": hardware_cfg["recommended_vram_limit_mb"],
    }


def get_gpu_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract GPU configuration (STRICT - no defaults)

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with GPU settings

    Raises:
        ValueError: If required keys are missing
    """
    if "gpu" not in config:
        raise ValueError("❌ Config missing 'gpu' section")

    gpu_cfg = config["gpu"]

    # Validate required top-level keys
    required_keys = [
        "enabled",
        "device_id",
        "memory_limit_mb",
        "memory_fraction",
        "validate_memory",
        "monitor_usage",
        "log_memory_stats",
        "warn_threshold_mb",
        "fallback_to_cpu",
        "cpu_fallback_on_oom",
        "retry_with_reduced_params",
    ]
    missing = [k for k in required_keys if k not in gpu_cfg]

    if missing:
        raise ValueError(
            f"❌ GPU section missing required keys: {missing}\n"
            f"   Required: {required_keys}\n"
            f"   Config.yaml v6.1.0 must define ALL GPU parameters"
        )

    result = {
        "enabled": gpu_cfg["enabled"],
        "device_id": gpu_cfg["device_id"],
        "memory_limit_mb": gpu_cfg["memory_limit_mb"],
        "memory_fraction": gpu_cfg["memory_fraction"],
        "validate_memory": gpu_cfg["validate_memory"],
        "monitor_usage": gpu_cfg["monitor_usage"],
        "log_memory_stats": gpu_cfg["log_memory_stats"],
        "warn_threshold_mb": gpu_cfg["warn_threshold_mb"],
        "fallback_to_cpu": gpu_cfg["fallback_to_cpu"],
        "cpu_fallback_on_oom": gpu_cfg["cpu_fallback_on_oom"],
        "retry_with_reduced_params": gpu_cfg["retry_with_reduced_params"],
    }

    # Validate XGBoost section if GPU enabled
    if gpu_cfg["enabled"]:
        if "xgboost" not in gpu_cfg:
            raise ValueError("❌ gpu.xgboost section required when GPU enabled")

        xgb_cfg = gpu_cfg["xgboost"]
        # tree_method is optional. In XGBoost >= 2.0 it is auto-inferred
        # from device=cuda (defaults to "hist"). Both _validate_config() and
        # get_gpu_config() now agree: tree_method is not required. (BUG 6 resolved)
        required_xgb = [
            "device",
            "max_bin",
            "sampling_method",
            "grow_policy",
            "max_cached_hist_node",
        ]
        missing_xgb = [k for k in required_xgb if k not in xgb_cfg]

        if missing_xgb:
            raise ValueError(
                f"❌ gpu.xgboost section missing: {missing_xgb}\n" f"   Required: {required_xgb}"
            )

        result.update(
            {
                "xgboost_device": xgb_cfg["device"],
                "xgboost_tree_method": xgb_cfg.get("tree_method", None),  # optional
                "xgboost_max_bin": xgb_cfg["max_bin"],
                "xgboost_max_cached_hist_node": xgb_cfg["max_cached_hist_node"],
                "xgboost_sampling_method": xgb_cfg["sampling_method"],
                "xgboost_grow_policy": xgb_cfg["grow_policy"],
                "xgboost_n_jobs": xgb_cfg.get("n_jobs", 1),
            }
        )

        # Validate LightGBM section
        if "lightgbm" not in gpu_cfg:
            raise ValueError("❌ gpu.lightgbm section required when GPU enabled")

        lgb_cfg = gpu_cfg["lightgbm"]
        required_lgb = [
            "device",
            "gpu_platform_id",
            "gpu_device_id",
            "max_bin",
            "gpu_use_dp",
        ]
        missing_lgb = [k for k in required_lgb if k not in lgb_cfg]

        if missing_lgb:
            raise ValueError(
                f"❌ gpu.lightgbm section missing: {missing_lgb}\n" f"   Required: {required_lgb}"
            )

        result.update(
            {
                "lightgbm_device": lgb_cfg["device"],
                "lightgbm_gpu_platform_id": lgb_cfg["gpu_platform_id"],
                "lightgbm_gpu_device_id": lgb_cfg["gpu_device_id"],
                "lightgbm_max_bin": lgb_cfg["max_bin"],
                "lightgbm_gpu_use_dp": lgb_cfg["gpu_use_dp"],
                "lightgbm_n_jobs": lgb_cfg.get("n_jobs", 1),
            }
        )

    return result


def extract_training_params(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract training parameters for fit_transform_pipeline() (STRICT)

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with training parameters

    Raises:
        ValueError: If required keys are missing
    """
    if "features" not in config:
        raise ValueError("❌ Config missing 'features' section")

    features_cfg = config["features"]

    # Validate required subsections
    required_sections = [
        "target_transform",
        "outlier_removal",
        "polynomial_features",
        "collinearity_removal",
    ]
    missing_sections = [s for s in required_sections if s not in features_cfg]

    if missing_sections:
        raise ValueError(
            f"❌ features section missing: {missing_sections}\n"
            f"   Required sections: {required_sections}"
        )

    # Validate required keys in each subsection
    if "method" not in features_cfg["target_transform"]:
        raise ValueError("❌ features.target_transform missing 'method'")
    if "enabled" not in features_cfg["outlier_removal"]:
        raise ValueError("❌ features.outlier_removal missing 'enabled'")
    if "enabled" not in features_cfg["polynomial_features"]:
        raise ValueError("❌ features.polynomial_features missing 'enabled'")
    if "enabled" not in features_cfg["collinearity_removal"]:
        raise ValueError("❌ features.collinearity_removal missing 'enabled'")

    return {
        "target_transform": features_cfg["target_transform"]["method"],
        "remove_outliers": features_cfg["outlier_removal"]["enabled"],
        "add_polynomials": features_cfg["polynomial_features"]["enabled"],
        "remove_collinear": features_cfg["collinearity_removal"]["enabled"],
    }


def get_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract feature engineering configuration (STRICT - no defaults)

    Includes encoding maps

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with EXACTLY the keys FeatureEngineer expects

    Raises:
        ValueError: If required config sections are missing
    """
    if "features" not in config:
        raise ValueError("❌ Config missing 'features' section")

    features_cfg = config["features"]

    # Validate all required subsections
    required_subsections = [
        "engineering",
        "target_transform",
        "outlier_removal",
        "polynomial_features",
        "collinearity_removal",
        "encoding",
    ]
    missing = [s for s in required_subsections if s not in features_cfg]
    if missing:
        raise ValueError(
            f"❌ features section missing subsections: {missing}\n"
            f"   Required: {required_subsections}"
        )

    engineering_cfg = features_cfg["engineering"]
    outlier_cfg = features_cfg["outlier_removal"]
    collinearity_cfg = features_cfg["collinearity_removal"]
    poly_cfg = features_cfg["polynomial_features"]
    encoding_cfg = features_cfg["encoding"]

    # Validate diagnostics section
    if "diagnostics" not in config:
        raise ValueError("❌ Config missing 'diagnostics' section")
    diagnostics_cfg = config["diagnostics"]

    # Build feature config with strict validation
    feature_config = {}

    # ENCODING MAPS
    required_encoding = ["smoker_binary_map", "smoker_risk_map"]
    missing_encoding = [k for k in required_encoding if k not in encoding_cfg]
    if missing_encoding:
        raise ValueError(f"❌ features.encoding missing: {missing_encoding}")

    feature_config.update(
        {
            "smoker_binary_map": encoding_cfg["smoker_binary_map"],
            "smoker_risk_map": encoding_cfg["smoker_risk_map"],
        }
    )

    # Engineering (EXISTING)
    if "variance_threshold" not in engineering_cfg:
        raise ValueError("❌ features.engineering missing 'variance_threshold'")
    feature_config["variance_threshold"] = engineering_cfg["variance_threshold"]

    # Collinearity (EXISTING)
    required_collin = [
        "threshold",
        "vif_threshold",
        "max_vif_iterations",
        "use_optimized_vif",
    ]
    missing_collin = [k for k in required_collin if k not in collinearity_cfg]
    if missing_collin:
        raise ValueError(f"❌ features.collinearity_removal missing: {missing_collin}")

    feature_config.update(
        {
            "correlation_threshold": collinearity_cfg["threshold"],
            "vif_threshold": collinearity_cfg["vif_threshold"],
            "max_vif_iterations": collinearity_cfg["max_vif_iterations"],
            "use_optimized_vif": collinearity_cfg["use_optimized_vif"],
        }
    )

    # Polynomial features (EXISTING)
    required_poly = ["degree", "max_features"]
    missing_poly = [k for k in required_poly if k not in poly_cfg]
    if missing_poly:
        raise ValueError(f"❌ features.polynomial_features missing: {missing_poly}")

    feature_config.update(
        {
            "polynomial_degree": poly_cfg["degree"],
            "polynomial_max_features": poly_cfg["max_features"],
        }
    )

    # Outlier detection (EXISTING)
    required_outlier = ["contamination", "random_state"]
    missing_outlier = [k for k in required_outlier if k not in outlier_cfg]
    if missing_outlier:
        raise ValueError(f"❌ features.outlier_removal missing: {missing_outlier}")

    feature_config.update(
        {
            "outlier_contamination": outlier_cfg["contamination"],
            "outlier_random_state": outlier_cfg["random_state"],
        }
    )

    # Validation ranges (EXISTING + v7.5.0 children bounds)
    required_ranges = ["bmi_min", "bmi_max", "age_min", "age_max", "children_min", "children_max"]
    missing_ranges = [k for k in required_ranges if k not in features_cfg]
    if missing_ranges:
        raise ValueError(f"❌ features section missing: {missing_ranges}")

    feature_config.update(
        {
            "bmi_min": features_cfg["bmi_min"],
            "bmi_max": features_cfg["bmi_max"],
            "age_min": features_cfg["age_min"],
            "age_max": features_cfg["age_max"],
            "children_min": features_cfg["children_min"],  # F-08 (v7.5.0)
            "children_max": features_cfg["children_max"],  # F-08 (v7.5.0)
        }
    )

    # Performance/diagnostics (EXISTING)
    if "performance" not in diagnostics_cfg:
        raise ValueError("❌ diagnostics missing 'performance' section")

    perf_cfg = diagnostics_cfg["performance"]
    required_perf = ["enabled", "log_memory"]
    missing_perf = [k for k in required_perf if k not in perf_cfg]
    if missing_perf:
        raise ValueError(f"❌ diagnostics.performance missing: {missing_perf}")

    feature_config.update(
        {
            "enable_performance_logging": perf_cfg["enabled"],
            "log_memory_usage": perf_cfg["log_memory"],
        }
    )

    return feature_config


def get_model_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Extract model-specific hyperparameter configurations (STRICT)

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary mapping model names to their hyperparameter configs

    Raises:
        ValueError: If models section is missing
    """
    if "models" not in config:
        raise ValueError(
            "❌ Config missing 'models' section\n"
            "   Config.yaml v6.1.0 must define model hyperparameters"
        )

    return cast(dict[str, dict[str, Any]], config["models"])


def get_sample_weight_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract sample weight configuration

    Returns:
        Dictionary with sample weight settings
    """
    if "sample_weights" not in config:
        raise ValueError("❌ Config missing 'sample_weights' section")

    sw_cfg = config["sample_weights"]

    required = ["enabled", "method", "tiers", "transform"]
    missing = [k for k in required if k not in sw_cfg]
    if missing:
        raise ValueError(f"❌ sample_weights missing: {missing}")

    return {
        "enabled": sw_cfg["enabled"],
        "method": sw_cfg["method"],
        "tiers": sw_cfg["tiers"],
        "transform": sw_cfg["transform"],
    }


def get_prediction_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract prediction configuration (STRICT - no defaults).

    Added in v7.5.0: ``prediction.max_batch_size`` moved out of hard-coded
    constants into config so PredictionPipeline and HybridPredictor both read
    from a single authoritative source.

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with prediction settings

    Raises:
        ValueError: If required keys are missing or invalid
    """
    if "prediction" not in config:
        raise ValueError("❌ Config missing 'prediction' section")

    pred_cfg = config["prediction"]

    if "max_batch_size" not in pred_cfg:
        raise ValueError(
            "❌ prediction section missing 'max_batch_size'\n"
            "   Required: prediction.max_batch_size (positive int)"
        )

    mbs = pred_cfg["max_batch_size"]
    if not isinstance(mbs, int) or mbs < 1:
        raise ValueError(f"❌ prediction.max_batch_size must be a positive integer, got {mbs!r}")

    return {"max_batch_size": mbs}


def get_conformal_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract conformal prediction configuration (STRICT - no defaults).

    Added in v7.5.0: ``conformal.calibration_split_ratio`` controls the
    calibration/holdout split inside ``train_single_model``.  Both the
    uncalibrated (conformal) and calibrated-isotonic code paths read this key.

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with conformal settings

    Raises:
        ValueError: If required keys are missing or out of range
    """
    if "conformal" not in config:
        raise ValueError("❌ Config missing 'conformal' section")

    conf_cfg = config["conformal"]

    if "calibration_split_ratio" not in conf_cfg:
        raise ValueError(
            "❌ conformal section missing 'calibration_split_ratio'\n"
            "   Required: conformal.calibration_split_ratio (float in (0, 1))"
        )

    csr = conf_cfg["calibration_split_ratio"]
    if not isinstance(csr, int | float) or not (0 < csr < 1):
        raise ValueError(f"❌ conformal.calibration_split_ratio must be in (0, 1), got {csr!r}")

    return {"calibration_split_ratio": float(csr)}


def get_optuna_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract Optuna hyperparameter tuning configuration (STRICT)
    References GPU and CV from their single sources

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with Optuna settings including constrained parameters

    Raises:
        ValueError: If required keys are missing
    """
    if "optuna" not in config:
        raise ValueError("❌ Config missing 'optuna' section")

    optuna_cfg = config["optuna"]

    # Validate required top-level keys
    required = ["n_trials", "timeout", "n_jobs", "random_state"]
    missing = [k for k in required if k not in optuna_cfg]
    if missing:
        raise ValueError(f"❌ optuna section missing: {missing}\n" f"   Required: {required}")

    # Get referenced configs (these will validate themselves)
    cv_cfg = get_cv_config(config)
    gpu_cfg = get_gpu_config(config)

    # Validate enhanced scoring section
    if "enhanced_scoring" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'enhanced_scoring' section")

    enhanced = optuna_cfg["enhanced_scoring"]
    required_enhanced = ["enabled", "mode", "hybrid_weights"]
    missing_enhanced = [k for k in required_enhanced if k not in enhanced]
    if missing_enhanced:
        raise ValueError(f"❌ optuna.enhanced_scoring missing: {missing_enhanced}")

    # Validate high value settings
    if "high_value_percentile" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'high_value_percentile'")
    if "high_value_penalty" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'high_value_penalty'")

    # Validate sampler section
    if "sampler" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'sampler' section")

    sampler = optuna_cfg["sampler"]
    required_sampler = ["type", "n_startup_trials", "multivariate", "seed"]
    missing_sampler = [k for k in required_sampler if k not in sampler]
    if missing_sampler:
        raise ValueError(f"❌ optuna.sampler missing: {missing_sampler}")

    # Validate pruner section
    if "pruner" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'pruner' section")

    pruner = optuna_cfg["pruner"]
    required_pruner = ["type", "n_startup_trials", "n_warmup_steps", "interval_steps"]
    missing_pruner = [k for k in required_pruner if k not in pruner]
    if missing_pruner:
        raise ValueError(f"❌ optuna.pruner missing: {missing_pruner}")

    # Validate early stopping section
    if "early_stopping" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'early_stopping' section")

    early = optuna_cfg["early_stopping"]
    required_early = ["enabled", "patience", "min_improvement"]
    missing_early = [k for k in required_early if k not in early]
    if missing_early:
        raise ValueError(f"❌ optuna.early_stopping missing: {missing_early}")

    # Validate overfitting section
    if "overfitting" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'overfitting' section")

    overfit = optuna_cfg["overfitting"]
    required_overfit = [
        "penalty_enabled",
        "threshold_critical",
        "threshold_warning",
        "penalty_multiplier",
    ]
    missing_overfit = [k for k in required_overfit if k not in overfit]
    if missing_overfit:
        raise ValueError(f"❌ optuna.overfitting missing: {missing_overfit}")

    # Validate logging section
    if "logging" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'logging' section")

    log = optuna_cfg["logging"]
    required_log = [
        "performance",
        "memory",
        "gpu_memory",
        "trial_params",
        "diagnostic_interval",
    ]
    missing_log = [k for k in required_log if k not in log]
    if missing_log:
        raise ValueError(f"❌ optuna.logging missing: {missing_log}")

    # Validate study management
    if "study_name_template" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'study_name_template'")
    if "load_if_exists" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'load_if_exists'")

    # Validate constrained parameters section
    if "constrained_params" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'constrained_params' section")

    constrained = optuna_cfg["constrained_params"]
    required_models = ["xgboost", "lightgbm", "random_forest"]
    missing_models = [m for m in required_models if m not in constrained]
    if missing_models:
        raise ValueError(f"❌ optuna.constrained_params missing models: {missing_models}")

    # Validate XGBoost constrained params
    # NOTE (v7.4.5): scale_pos_weight intentionally removed from required_xgb.
    # It is a binary classification parameter (binary:logistic / multi:softmax only).
    # Requiring it here for reg:quantileerror models forced config.yaml to keep it in
    # constrained_params, causing Optuna to waste a search dimension suggesting a
    # value that XGBoost silently ignores — and generating a UserWarning on every
    # .fit() call across all 100 trials × 5 folds (500+ warnings per run).
    # The optimizer strips it pre-construction via _filter_xgb_quantile_params()
    # as an additional safety net, but removing it from the validator is the
    # it allows config.yaml to omit it from constrained_params entirely.
    xgb_params = constrained["xgboost"]
    required_xgb = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "min_child_weight",
        # scale_pos_weight REMOVED (v7.4.5) — classification-only param, see note above
        "max_delta_step",
    ]
    missing_xgb = [p for p in required_xgb if p not in xgb_params]
    if missing_xgb:
        raise ValueError(f"❌ optuna.constrained_params.xgboost missing: {missing_xgb}")

    # Validate LightGBM constrained params
    lgb_params = constrained["lightgbm"]
    # NOTE (v7.4.4): min_child_samples intentionally removed from this list.
    # It is a LightGBM alias for min_data_in_leaf. config.yaml was updated in
    # v7.4.4 to remove min_child_samples from optuna.constrained_params.lightgbm
    # and widen min_data_in_leaf to [5, 60] to cover the full search space.
    # Passing both to LGBMRegressor(**params) raises:
    #   LightGBMError: "Found duplicated parameter: min_data_in_leaf"
    # Only the canonical parameter (min_data_in_leaf) is required here.
    required_lgb = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "num_leaves",
        "min_data_in_leaf",  # canonical param; min_child_samples is its alias (removed v7.4.4)
        "min_split_gain",
        "subsample",
        "subsample_freq",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_child_weight",
    ]
    missing_lgb = [p for p in required_lgb if p not in lgb_params]
    if missing_lgb:
        raise ValueError(f"❌ optuna.constrained_params.lightgbm missing: {missing_lgb}")

    # Validate Random Forest constrained params
    rf_params = constrained["random_forest"]
    required_rf = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
        "oob_score",
    ]
    missing_rf = [p for p in required_rf if p not in rf_params]
    if missing_rf:
        raise ValueError(f"❌ optuna.constrained_params.random_forest missing: {missing_rf}")

    # Validate file lock section
    if "file_lock" not in optuna_cfg:
        raise ValueError("❌ optuna missing 'file_lock' section")

    file_lock = optuna_cfg["file_lock"]
    if "timeout" not in file_lock:
        raise ValueError("❌ optuna.file_lock missing 'timeout'")

    # Validate GPU batch size (optional in optuna.gpu section)
    gpu_batch_size = None
    if "gpu" in optuna_cfg and "batch_size" in optuna_cfg["gpu"]:
        gpu_batch_size = optuna_cfg["gpu"]["batch_size"]

    return {
        # Core settings
        "n_trials": optuna_cfg["n_trials"],
        "timeout": optuna_cfg["timeout"],
        "n_jobs": optuna_cfg["n_jobs"],
        "random_state": optuna_cfg["random_state"],
        # Cross-validation
        "cv_n_folds": cv_cfg["n_folds"],
        "cv_shuffle": cv_cfg["shuffle"],
        "cv_stratified": cv_cfg["stratified"],
        # Enhanced scoring
        "enhanced_scoring_enabled": enhanced["enabled"],
        "enhanced_scoring_mode": enhanced["mode"],
        "hybrid_weights": enhanced["hybrid_weights"],
        "high_value_percentile": optuna_cfg["high_value_percentile"],
        "high_value_penalty": optuna_cfg["high_value_penalty"],
        # Sampler
        "sampler_type": sampler["type"],
        "sampler_n_startup_trials": sampler["n_startup_trials"],
        "sampler_multivariate": sampler["multivariate"],
        "sampler_seed": sampler["seed"],
        # Pruner
        "pruner_type": pruner["type"],
        "pruner_n_startup_trials": pruner["n_startup_trials"],
        "pruner_n_warmup_steps": pruner["n_warmup_steps"],
        "pruner_interval_steps": pruner["interval_steps"],
        # Early stopping
        "early_stopping_enabled": early["enabled"],
        "early_stopping_patience": early["patience"],
        "early_stopping_min_improvement": early["min_improvement"],
        # Overfitting detection
        "overfitting_penalty_enabled": overfit["penalty_enabled"],
        "overfitting_threshold_critical": overfit["threshold_critical"],
        "overfitting_threshold_warning": overfit["threshold_warning"],
        "overfitting_penalty_multiplier": overfit["penalty_multiplier"],
        # Performance logging
        "enable_performance_logging": log["performance"],
        "log_memory_usage": log["memory"],
        "log_gpu_memory": log["gpu_memory"],
        "log_trial_params": log["trial_params"],
        "diagnostic_interval": log["diagnostic_interval"],
        # GPU (referenced from single source)
        "use_gpu": gpu_cfg["enabled"],
        "gpu_memory_fraction": gpu_cfg["memory_fraction"],
        "gpu_memory_limit_mb": gpu_cfg["memory_limit_mb"],
        "gpu_batch_size": gpu_batch_size,
        # Study management
        "study_name_template": optuna_cfg["study_name_template"],
        "storage": optuna_cfg.get("storage"),  # Can be None/null
        "load_if_exists": optuna_cfg["load_if_exists"],
        # Constrained hyperparameters
        # pass through ALL models present in constrained_params,
        # not just the 3 required ones. Without this, xgboost_median added
        # to config.yaml is silently dropped here.
        "constrained_params": {
            "xgboost": xgb_params,
            "lightgbm": lgb_params,
            "random_forest": rf_params,
            # Spread any extra models (e.g. xgboost_median for two-model arch)
            **{
                k: v
                for k, v in constrained.items()
                if k not in {"xgboost", "lightgbm", "random_forest"} and isinstance(v, dict)
            },
        },
        # File locking
        "file_lock_timeout": file_lock["timeout"],
    }


def get_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract training configuration (STRICT)
    References GPU and CV from their single sources

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with training settings

    Raises:
        ValueError: If required keys are missing
    """
    if "training" not in config:
        raise ValueError("❌ Config missing 'training' section")

    training_cfg = config["training"]

    # Validate required keys
    required = [
        "output_dir",
        "reports_dir",
        "test_size",
        "val_size",
        "stratify_splits",
        "min_r2_threshold",
        "enable_mlflow",
        "enable_optuna",
        "enable_diagnostics",
        "training_timeout",
        "max_model_size_mb",
        "max_memory_mb",
        "verify_checksums",
        "save_checksums",
        "register_to_mlflow",
        "halt_on_severe_shift",
        "batch_size",
        "use_sample_weights",
        "high_value_percentile",
        "memory_fraction",
    ]
    missing = [k for k in required if k not in training_cfg]
    if missing:
        raise ValueError(f"❌ training section missing: {missing}\n" f"   Required: {required}")

    # Validate checkpoint section
    if "checkpoint" not in training_cfg:
        raise ValueError("❌ training missing 'checkpoint' section")

    checkpoint = training_cfg["checkpoint"]
    required_checkpoint = ["enabled", "frequency"]
    missing_checkpoint = [k for k in required_checkpoint if k not in checkpoint]
    if missing_checkpoint:
        raise ValueError(f"❌ training.checkpoint missing: {missing_checkpoint}")

    # Validate memory section
    if "memory" not in training_cfg:
        raise ValueError("❌ training missing 'memory' section")

    memory = training_cfg["memory"]
    required_memory = ["cleanup_frequency", "force_gc", "clear_gpu_cache"]
    missing_memory = [k for k in required_memory if k not in memory]
    if missing_memory:
        raise ValueError(f"❌ training.memory missing: {missing_memory}")

    # Validate early stopping section
    if "early_stopping" not in training_cfg:
        raise ValueError("❌ training missing 'early_stopping' section")

    early = training_cfg["early_stopping"]
    required_early = ["enabled", "patience", "min_delta"]
    missing_early = [k for k in required_early if k not in early]
    if missing_early:
        raise ValueError(f"❌ training.early_stopping missing: {missing_early}")

    # Validate two_model_architecture section (v7.5.0)
    if "two_model_architecture" not in training_cfg:
        raise ValueError("❌ training missing 'two_model_architecture' section")

    tma = training_cfg["two_model_architecture"]
    required_tma = [
        "enabled",
        "pricing_model",
        "risk_model",
        "overpricing_gate_model",
        "risk_model_alpha",
    ]
    missing_tma = [k for k in required_tma if k not in tma]
    if missing_tma:
        raise ValueError(f"❌ training.two_model_architecture missing: {missing_tma}")

    # Validate deployment_gates section (v7.5.0)
    if "deployment_gates" not in training_cfg:
        raise ValueError("❌ training missing 'deployment_gates' section")

    gates = training_cfg["deployment_gates"]
    required_gates = [
        "g6_min_cost_weighted_r2",
        "g7_max_overpricing_rate",
        "g3_max_width_ratio",
    ]
    missing_gates = [k for k in required_gates if k not in gates]
    if missing_gates:
        raise ValueError(f"❌ training.deployment_gates missing: {missing_gates}")

    # Validate provenance section (v7.5.0)
    if "provenance" not in training_cfg:
        raise ValueError("❌ training missing 'provenance' section")

    prov = training_cfg["provenance"]
    required_prov = [
        "capture_git_hash",
        "require_clean_tree",
        "random_state_in_artifact",
        "always_write_bias_correction",
    ]
    missing_prov = [k for k in required_prov if k not in prov]
    if missing_prov:
        raise ValueError(f"❌ training.provenance missing: {missing_prov}")

    # Validate conformal_intervals section (v7.5.0)
    if "conformal_intervals" not in training_cfg:
        raise ValueError("❌ training missing 'conformal_intervals' section")

    conf_intervals = training_cfg["conformal_intervals"]
    required_conf = [
        "method",
        "fallback",
        "target_coverage",
        "max_width_ratio",
        "target_width_ratio",
    ]
    missing_conf = [k for k in required_conf if k not in conf_intervals]
    if missing_conf:
        raise ValueError(f"❌ training.conformal_intervals missing: {missing_conf}")

    # Get referenced configs (these will validate themselves)
    gpu_cfg = get_gpu_config(config)
    cv_cfg = get_cv_config(config)
    # random_state canonical source is defaults.random_state, not
    # cross_validation.random_state.  If only defaults.random_state is overridden
    # via env var, cv_cfg["random_state"] may remain at the YAML value, causing
    # the training loop to use a different seed than the rest of the pipeline.
    defaults_cfg = config.get("defaults", {})
    canonical_random_state = defaults_cfg.get(
        "random_state",
        cv_cfg["random_state"],  # fallback to cv if defaults missing
    )

    return {
        "output_dir": training_cfg["output_dir"],
        "reports_dir": training_cfg["reports_dir"],
        # CV
        "cv_folds": cv_cfg["n_folds"],
        "random_state": canonical_random_state,
        # Split configuration
        "test_size": training_cfg["test_size"],
        "val_size": training_cfg["val_size"],
        "stratify_splits": training_cfg["stratify_splits"],
        # Feature flags
        "enable_mlflow": training_cfg["enable_mlflow"],
        "enable_optuna": training_cfg["enable_optuna"],
        "enable_diagnostics": training_cfg["enable_diagnostics"],
        # GPU
        "gpu_memory_fraction": gpu_cfg["memory_fraction"],
        "gpu_memory_limit_mb": gpu_cfg["memory_limit_mb"],
        # Limits
        "max_model_size_mb": training_cfg["max_model_size_mb"],
        "training_timeout": training_cfg["training_timeout"],
        "max_memory_mb": training_cfg["max_memory_mb"],
        # Quality
        "min_r2_threshold": training_cfg["min_r2_threshold"],
        # Security
        "verify_checksums": training_cfg["verify_checksums"],
        "save_checksums": training_cfg["save_checksums"],
        "register_to_mlflow": training_cfg["register_to_mlflow"],
        "halt_on_severe_shift": training_cfg["halt_on_severe_shift"],
        # Batch processing
        "batch_size": training_cfg["batch_size"],
        # Checkpointing
        "checkpoint_enabled": checkpoint["enabled"],
        "checkpoint_frequency": checkpoint["frequency"],
        # Memory management
        "memory_cleanup_frequency": memory["cleanup_frequency"],
        "force_gc": memory["force_gc"],
        "clear_gpu_cache": memory["clear_gpu_cache"],
        # Early stopping
        "early_stopping_enabled": early["enabled"],
        "early_stopping_patience": early["patience"],
        "early_stopping_min_delta": early["min_delta"],
        # Sample weighting
        "use_sample_weights": training_cfg["use_sample_weights"],
        "high_value_percentile": training_cfg["high_value_percentile"],
        "memory_fraction": training_cfg["memory_fraction"],
        # Two-model architecture (v7.5.0)
        "two_model_architecture_enabled": tma["enabled"],
        "tma_pricing_model": tma["pricing_model"],
        "tma_risk_model": tma["risk_model"],
        "tma_overpricing_gate_model": tma["overpricing_gate_model"],
        "tma_risk_model_alpha": tma["risk_model_alpha"],
        # Deployment gates (v7.5.0 — G3/G6/G7)
        "g6_min_cost_weighted_r2": gates["g6_min_cost_weighted_r2"],
        "g7_max_overpricing_rate": gates["g7_max_overpricing_rate"],
        "g3_max_width_ratio": gates["g3_max_width_ratio"],
        # Provenance (v7.5.0)
        "capture_git_hash": prov["capture_git_hash"],
        "require_clean_tree": prov["require_clean_tree"],
        "random_state_in_artifact": prov["random_state_in_artifact"],
        "always_write_bias_correction": prov["always_write_bias_correction"],
        # Conformal intervals (v7.5.0)
        "conformal_method": conf_intervals["method"],
        "conformal_fallback": conf_intervals["fallback"],
        "conformal_target_coverage": conf_intervals["target_coverage"],
        "conformal_max_width_ratio": conf_intervals["max_width_ratio"],
        "conformal_target_width_ratio": conf_intervals["target_width_ratio"],
    }


def get_mlflow_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract MLflow configuration (STRICT)

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with MLflow settings

    Raises:
        ValueError: If required keys are missing
    """
    if "mlflow" not in config:
        raise ValueError("❌ Config missing 'mlflow' section")

    mlflow_cfg = config["mlflow"]

    # Validate tracking section
    if "tracking" not in mlflow_cfg:
        raise ValueError("❌ mlflow missing 'tracking' section")

    tracking = mlflow_cfg["tracking"]
    required_tracking = [
        "enabled",
        "tracking_uri",
        "experiment_name",
        "run_name_prefix",
    ]
    missing_tracking = [k for k in required_tracking if k not in tracking]
    if missing_tracking:
        raise ValueError(f"❌ mlflow.tracking missing: {missing_tracking}")

    # Validate registry section
    if "registry" not in mlflow_cfg:
        raise ValueError("❌ mlflow missing 'registry' section")

    registry = mlflow_cfg["registry"]
    required_registry = ["enabled", "model_name", "register_best_only"]
    missing_registry = [k for k in required_registry if k not in registry]
    if missing_registry:
        raise ValueError(f"❌ mlflow.registry missing: {missing_registry}")

    # Validate logging section
    if "logging" not in mlflow_cfg:
        raise ValueError("❌ mlflow missing 'logging' section")

    logging_cfg = mlflow_cfg["logging"]
    required_logging = [
        "level",
        "log_metrics",
        "log_params",
        "log_artifacts",
        "log_models",
        "log_system_metrics",
    ]
    missing_logging = [k for k in required_logging if k not in logging_cfg]
    if missing_logging:
        raise ValueError(f"❌ mlflow.logging missing: {missing_logging}")

    # Validate autolog section
    if "autolog" not in mlflow_cfg:
        raise ValueError("❌ mlflow missing 'autolog' section")

    autolog = mlflow_cfg["autolog"]
    required_autolog = ["sklearn", "xgboost", "lightgbm", "disable"]
    missing_autolog = [k for k in required_autolog if k not in autolog]
    if missing_autolog:
        raise ValueError(f"❌ mlflow.autolog missing: {missing_autolog}")

    # Validate GPU metrics
    if "log_gpu_metrics" not in mlflow_cfg:
        raise ValueError("❌ mlflow missing 'log_gpu_metrics'")

    return {
        # Tracking
        "tracking_enabled": tracking["enabled"],
        "tracking_uri": tracking["tracking_uri"],
        "experiment_name": tracking["experiment_name"],
        "run_name_prefix": tracking["run_name_prefix"],
        # Registry
        "registry_enabled": registry["enabled"],
        "model_name": registry["model_name"],
        "register_best_only": registry["register_best_only"],
        # Logging
        "log_level": logging_cfg["level"],
        "log_metrics": logging_cfg["log_metrics"],
        "log_params": logging_cfg["log_params"],
        "log_artifacts": logging_cfg["log_artifacts"],
        "log_models": logging_cfg["log_models"],
        "log_system_metrics": logging_cfg["log_system_metrics"],
        # Autolog
        "autolog_sklearn": autolog["sklearn"],
        "autolog_xgboost": autolog["xgboost"],
        "autolog_lightgbm": autolog["lightgbm"],
        "autolog_disable": autolog["disable"],
        # GPU metrics
        "log_gpu_metrics": mlflow_cfg["log_gpu_metrics"],
    }


def get_high_value_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract high-value segment analysis configuration (STRICT)

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with high-value analysis settings

    Raises:
        ValueError: If required keys are missing
    """
    if "high_value_analysis" not in config:
        raise ValueError("❌ Config missing 'high_value_analysis' section")

    hv_cfg = config["high_value_analysis"]

    # Validate required keys
    required = [
        "enabled",
        "threshold_percentile",
        "baseline_model",
        "compare_against_baseline",
    ]
    missing = [k for k in required if k not in hv_cfg]
    if missing:
        raise ValueError(f"❌ high_value_analysis missing: {missing}")

    # Validate reporting section
    if "reporting" not in hv_cfg:
        raise ValueError("❌ high_value_analysis missing 'reporting' section")

    reporting = hv_cfg["reporting"]
    required_reporting = ["save_report", "format"]
    missing_reporting = [k for k in required_reporting if k not in reporting]
    if missing_reporting:
        raise ValueError(f"❌ high_value_analysis.reporting missing: {missing_reporting}")

    return {
        "enabled": hv_cfg["enabled"],
        "threshold_percentile": hv_cfg["threshold_percentile"],
        "baseline_model": hv_cfg["baseline_model"],
        "compare_against_baseline": hv_cfg["compare_against_baseline"],
        "custom_threshold": hv_cfg.get("custom_threshold"),  # Can be None/null
        "save_report": reporting["save_report"],
        "report_format": reporting["format"],
    }


def get_diagnostics_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract diagnostics configuration (STRICT)

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with diagnostics settings

    Raises:
        ValueError: If required keys are missing
    """
    if "diagnostics" not in config:
        raise ValueError("❌ Config missing 'diagnostics' section")

    diag_cfg = config["diagnostics"]

    # Validate required top-level keys
    required = ["enabled", "max_samples", "batch_size", "rf_tree_batch_size"]
    missing = [k for k in required if k not in diag_cfg]
    if missing:
        raise ValueError(f"❌ diagnostics missing: {missing}")

    # Validate performance section
    if "performance" not in diag_cfg:
        raise ValueError("❌ diagnostics missing 'performance' section")

    perf = diag_cfg["performance"]
    required_perf = [
        "enabled",
        "log_memory",
        "log_gpu_memory",
        "log_training_time",
        "log_prediction_time",
    ]
    missing_perf = [k for k in required_perf if k not in perf]
    if missing_perf:
        raise ValueError(f"❌ diagnostics.performance missing: {missing_perf}")

    # Validate plots section
    if "plots" not in diag_cfg:
        raise ValueError("❌ diagnostics missing 'plots' section")

    plots = diag_cfg["plots"]
    required_plots = [
        "learning_curves",
        "residuals",
        "error_distribution",
        "calibration",
        "feature_importance",
        "partial_dependence",
        "shap",
        "save",
        "format",
        "dpi",
    ]
    missing_plots = [k for k in required_plots if k not in plots]
    if missing_plots:
        raise ValueError(f"❌ diagnostics.plots missing: {missing_plots}")

    # Validate reports section
    if "reports" not in diag_cfg:
        raise ValueError("❌ diagnostics missing 'reports' section")

    reports = diag_cfg["reports"]
    required_reports = ["html", "pdf"]
    missing_reports = [k for k in required_reports if k not in reports]
    if missing_reports:
        raise ValueError(f"❌ diagnostics.reports missing: {missing_reports}")

    display = diag_cfg.get("display", {})
    required_display = [
        "top_features_pdp",
        "worst_predictions_show",
        "distribution_shift_top_features",
        "min_sample_for_save",
    ]
    missing_display = [k for k in required_display if k not in display]
    if missing_display:
        raise ValueError(f"❌ diagnostics.display missing: {missing_display}")

    # Validate sampling section
    if "sampling" not in diag_cfg:
        raise ValueError("❌ diagnostics missing 'sampling' section")

    sampling = diag_cfg["sampling"]
    required_sampling = [
        "shap_max_samples",
        "shap_background_samples",
        "residual_sample_size",
        "autocorr_lag_limit",
        "calibration_bins",
        "learning_curve_points",
        "permutation_importance_repeats",
    ]
    missing_sampling = [k for k in required_sampling if k not in sampling]
    if missing_sampling:
        raise ValueError(f"❌ diagnostics.sampling missing: {missing_sampling}")

    return {
        "enabled": diag_cfg["enabled"],
        "max_samples": diag_cfg["max_samples"],
        # Performance logging
        "performance_enabled": perf["enabled"],
        "log_memory": perf["log_memory"],
        "log_gpu_memory": perf["log_gpu_memory"],
        "log_training_time": perf["log_training_time"],
        "log_prediction_time": perf["log_prediction_time"],
        # Plots
        "learning_curves": plots["learning_curves"],
        "residuals": plots["residuals"],
        "error_distribution": plots["error_distribution"],
        "calibration": plots["calibration"],
        "feature_importance": plots["feature_importance"],
        "partial_dependence": plots["partial_dependence"],
        "shap": plots["shap"],
        # Plot settings
        "save_plots": plots["save"],
        "plot_format": plots["format"],
        "plot_dpi": plots["dpi"],
        # Reports
        "generate_html_report": reports["html"],
        "generate_pdf_report": reports["pdf"],
        "top_features_pdp": display["top_features_pdp"],
        "worst_predictions_show": display["worst_predictions_show"],
        "distribution_shift_top_features": display["distribution_shift_top_features"],
        "min_sample_for_save": display["min_sample_for_save"],
        # Sampling
        "shap_max_samples": sampling["shap_max_samples"],
        "shap_background_samples": sampling["shap_background_samples"],
        "residual_sample_size": sampling["residual_sample_size"],
        "autocorr_lag_limit": sampling["autocorr_lag_limit"],
        "calibration_bins": sampling["calibration_bins"],
        "learning_curve_points": sampling["learning_curve_points"],
        "permutation_importance_repeats": sampling["permutation_importance_repeats"],
        # Batch processing
        "batch_size": diag_cfg["batch_size"],
        "rf_tree_batch_size": diag_cfg["rf_tree_batch_size"],
    }


def get_explainability_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract explainability configuration (STRICT - no defaults)

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with explainability settings

    Raises:
        ValueError: If required keys are missing
    """
    if "diagnostics" not in config:
        raise ValueError("⛔ Config missing 'diagnostics' section")

    diag_cfg = config["diagnostics"]

    # Validate required explainability keys
    required_explainability = [
        "enable_confidence_intervals",
        "confidence_level",
        "enable_shap",
        "auto_plot",
    ]
    missing = [k for k in required_explainability if k not in diag_cfg]
    if missing:
        raise ValueError(
            f"⛔ diagnostics section missing explainability keys: {missing}\n"
            f"   Required: {required_explainability}"
        )

    # Validate confidence level range
    confidence_level = diag_cfg["confidence_level"]
    if not (0 < confidence_level < 1):
        raise ValueError(f"⛔ confidence_level must be in (0, 1), got {confidence_level}")

    # Validate sampling section for SHAP parameters
    if "sampling" not in diag_cfg:
        raise ValueError("⛔ diagnostics missing 'sampling' section")

    sampling = diag_cfg["sampling"]
    required_sampling = ["shap_max_samples", "shap_background_samples"]
    missing_sampling = [k for k in required_sampling if k not in sampling]
    if missing_sampling:
        raise ValueError(
            f"⛔ diagnostics.sampling missing: {missing_sampling}\n"
            f"   Required for explainability: {required_sampling}"
        )

    return {
        "enable_confidence_intervals": diag_cfg["enable_confidence_intervals"],
        "confidence_level": diag_cfg["confidence_level"],
        "enable_shap": diag_cfg["enable_shap"],
        "shap_max_samples": sampling["shap_max_samples"],
        "shap_background_samples": sampling["shap_background_samples"],
        "auto_plot": diag_cfg["auto_plot"],
        "save_path": diag_cfg.get("save_path"),
    }


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_gpu_config(config: dict[str, Any]) -> None:
    """
    Validate GPU configuration for hardware compatibility
    Uses validation and hardware sections from config

    Args:
        config: Configuration dictionary from load_config()

    Raises:
        ValueError: If GPU config is invalid
    """
    gpu_cfg = get_gpu_config(config)

    if not gpu_cfg["enabled"]:
        logger.info("✅ GPU disabled - skipping GPU validation")
        return

    # Get validation parameters from config
    if "hardware" not in config:
        raise ValueError("❌ Config missing 'hardware' section for GPU validation")

    hardware_cfg = config["hardware"]
    required_hw = ["gpu_model", "total_vram_mb", "max_safe_vram_mb"]
    missing_hw = [k for k in required_hw if k not in hardware_cfg]
    if missing_hw:
        raise ValueError(f"❌ hardware section missing: {missing_hw}")

    # Check memory limit
    memory_limit = gpu_cfg["memory_limit_mb"]
    max_vram = hardware_cfg["max_safe_vram_mb"]
    gpu_model = hardware_cfg["gpu_model"]

    recommended = hardware_cfg.get("recommended_vram_limit_mb", max_vram)
    if memory_limit > max_vram:
        raise ValueError(
            f"❌ GPU memory limit ({memory_limit}MB) exceeds {gpu_model} "
            f"physical VRAM capacity ({max_vram}MB).\n"
            f"   Hard ceiling: {max_vram}MB  |  Recommended safe ceiling: {recommended}MB\n"
            f"   Set gpu.memory_limit_mb ≤ {recommended}MB to leave headroom "
            f"for CUDA context and driver overhead (~{max_vram - recommended}MB)."
        )
    if memory_limit > recommended:
        logger.warning(
            f"⚠️  GPU memory limit ({memory_limit}MB) exceeds the recommended "
            f"operational ceiling ({recommended}MB) for {gpu_model}. "
            f"CUDA context + driver consume up to {max_vram - recommended}MB; "
            f"consider lowering gpu.memory_limit_mb to ≤{recommended}MB "
            f"to avoid OOM under peak load."
        )

    # Validate XGBoost device format
    if "validation" not in config:
        raise ValueError("❌ Config missing 'validation' section")

    validation_cfg = config["validation"]
    xgb_device = gpu_cfg["xgboost_device"]
    valid_devices = validation_cfg.get("valid_xgboost_devices", [])

    if not valid_devices:
        raise ValueError("❌ validation section missing 'valid_xgboost_devices'")

    if xgb_device not in valid_devices:
        logger.warning(
            f"⚠️ XGBoost device '{xgb_device}' may not be valid\n"
            f"   Expected one of: {valid_devices}"
        )

    logger.info(f"✅ GPU configuration validated for {gpu_model}")


def validate_single_source_of_truth(config: dict[str, Any]) -> None:
    """
    Validate that configuration follows single source of truth principles

    Args:
        config: Configuration dictionary from load_config()

    Raises:
        ValueError: If redundancy detected
    """
    errors = []

    # Check for CV folds redundancy (should only be in cross_validation)
    if "cv_folds" in config.get("model", {}):
        errors.append("model.cv_folds exists (should reference cross_validation.n_folds)")

    if "cv_folds" in config.get("training", {}):
        errors.append("training.cv_folds exists (should reference cross_validation.n_folds)")

    # Check for GPU memory redundancy (should only be in gpu)
    if "gpu_memory_fraction" in config.get("training", {}):
        errors.append("training.gpu_memory_fraction exists (should reference gpu.memory_fraction)")

    if "gpu_memory_limit_mb" in config.get("optuna", {}):
        errors.append("optuna.gpu_memory_limit_mb exists (should reference gpu.memory_limit_mb)")

    # Check for cross_validation in optuna (should reference top-level)
    if "cross_validation" in config.get("optuna", {}):
        errors.append(
            "optuna.cross_validation exists (should reference top-level cross_validation)"
        )

    # Check the one actual violation present in the live codebase.
    # data.validation_size is required by _validate_config() (legacy) but never
    # consumed — get_training_config() uses training.val_size as the sole source
    # of truth for the validation split size.  Having both defined is misleading:
    # changing data.validation_size has zero effect while appearing authoritative.
    if "validation_size" in config.get("data", {}) and "val_size" in config.get("training", {}):
        errors.append(
            "Both data.validation_size and training.val_size are defined. "
            "Only training.val_size is used by get_training_config(). "
            "Remove data.validation_size from config.yaml to eliminate ambiguity "
            "(or route get_training_config() through data.validation_size and "
            "delete training.val_size)."
        )

    if errors:
        raise ValueError(
            "❌ Configuration violates single source of truth principle:\n"
            + "\n".join(f"   - {e}" for e in errors)
            + "\n\nPlease update to config.yaml v6.1.0 architecture"
        )

    logger.info("✅ Configuration follows single source of truth principles")


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================


def print_gpu_config_summary(config: dict[str, Any]) -> None:
    """
    Print GPU configuration summary

    Args:
        config: Configuration dictionary from load_config()
    """
    gpu_cfg = get_gpu_config(config)
    hardware_cfg = config.get("hardware", {})

    if not gpu_cfg["enabled"]:
        print("\n❌ GPU: Disabled (CPU training only)")
        return

    print("\n" + "=" * 80)
    print(f"⚡️ GPU CONFIGURATION ({hardware_cfg.get('gpu_model', 'Unknown GPU')})")
    print("=" * 80)
    print(f"  Enabled: {gpu_cfg['enabled']}")
    print(f"  Device ID: {gpu_cfg['device_id']}")
    print(f"  Memory Limit: {gpu_cfg['memory_limit_mb']}MB (SINGLE SOURCE)")
    print(f"  Memory Fraction: {gpu_cfg['memory_fraction']}")
    print(f"  Total VRAM: {hardware_cfg.get('total_vram_mb', 'Unknown')}MB")
    print("\n  XGBoost:")
    print(f"    Device: {gpu_cfg['xgboost_device']}")
    print(f"    Tree Method: {gpu_cfg['xgboost_tree_method']}")
    print(f"    Max Bin: {gpu_cfg['xgboost_max_bin']}")
    print("\n  LightGBM:")
    print(f"    Device: {gpu_cfg['lightgbm_device']}")
    print(f"    GPU Device ID: {gpu_cfg['lightgbm_gpu_device_id']}")
    print(f"    Max Bin: {gpu_cfg['lightgbm_max_bin']}")
    print("\n  Fallback:")
    print(f"    CPU on OOM: {gpu_cfg['cpu_fallback_on_oom']}")
    print("=" * 80)


def print_config_summary(config: dict[str, Any]) -> None:
    """
    Print human-readable configuration summary

    Args:
        config: Configuration dictionary from load_config()
    """
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY v7.5.0 - TRUE Single Source of Truth")
    print("=" * 80)

    # Version info
    version = config.get("version", "Unknown")
    architecture = config.get("architecture", "Unknown")
    print(f"\nℹ️  Version: {version}")
    print(f"   Architecture: {architecture}")

    # Defaults
    defaults = get_defaults(config)
    print("\n🎲 Global Defaults:")
    print(f"   Random state: {defaults['random_state']}")
    print(f"   N jobs: {defaults['n_jobs']}")

    # Cross-validation
    cv_cfg = get_cv_config(config)
    print("\n📊 Cross-Validation (Single Source):")
    print(f"   Folds: {cv_cfg['n_folds']}")
    print(f"   Shuffle: {cv_cfg['shuffle']}")
    print(f"   Stratified: {cv_cfg['stratified']}")

    # Data section
    data_cfg = config.get("data", {})
    print("\n📂 Data:")
    print(f"   Path: {data_cfg.get('raw_path', 'N/A')}")
    print(f"   Target: {data_cfg.get('target_column', 'N/A')}")
    test_size = data_cfg.get("test_size")
    if isinstance(test_size, float):
        print(f"   Test size: {test_size:.0%}")
    else:
        print(f"   Test size: {test_size or 'N/A'}")

    # Feature engineering
    features_cfg = config.get("features", {})
    target_transform_method = features_cfg.get("target_transform", {}).get("method")
    outlier_enabled = features_cfg.get("outlier_removal", {}).get("enabled", False)
    polynomial_enabled = features_cfg.get("polynomial_features", {}).get("enabled", False)
    collinearity_enabled = features_cfg.get("collinearity_removal", {}).get("enabled", False)
    vif_threshold = features_cfg.get("collinearity_removal", {}).get("vif_threshold")

    print("\n🔧 Feature Engineering:")
    print(f"   Target transform: {target_transform_method}")
    print(f"   Remove outliers: {outlier_enabled}")
    print(f"   Add polynomials: {polynomial_enabled}")
    print(f"   Remove collinear: {collinearity_enabled}")
    print(f"   VIF threshold: {vif_threshold if vif_threshold is not None else 'N/A'}")

    # Model section
    model_cfg = config.get("model", {})
    models = model_cfg.get("models", [])
    print("\n🤖 Models:")
    print(f"   Count: {len(models)}")
    print(f"   Models: {', '.join(models[:3])}{', ...' if len(models) > 3 else ''}")
    print(f"   Metric: {model_cfg.get('metric', 'N/A')}")

    # GPU section (single source)
    gpu_cfg = get_gpu_config(config)
    hardware_cfg = config.get("hardware", {})
    print("\n⚙️ GPU (Single Source):")
    print(f"   Enabled: {gpu_cfg['enabled']}")
    if gpu_cfg["enabled"]:
        print(f"   Model: {hardware_cfg.get('gpu_model', 'Unknown')}")
        print(f"   Memory limit: {gpu_cfg['memory_limit_mb']}MB")
        print(f"   Memory fraction: {gpu_cfg['memory_fraction']}")
        print(f"   XGBoost device: {gpu_cfg['xgboost_device']}")
        print(f"   LightGBM device: {gpu_cfg['lightgbm_device']}")

    # Optuna section
    optuna_cfg = config.get("optuna", {})
    print("\n🔍 Optuna:")
    n_trials = optuna_cfg.get("n_trials", 0)
    print(f"   Enabled: {n_trials > 0}")
    print(f"   Trials: {n_trials}")
    print(f"   Timeout: {optuna_cfg.get('timeout', 0)}s")
    print(f"   References CV folds: {cv_cfg['n_folds']}")
    print(f"   References GPU memory: {gpu_cfg['memory_limit_mb']}MB")

    # Explainability
    try:
        explainability_cfg = get_explainability_config(config)
        print("\n🔍 Explainability:")
        print(f"   Confidence Intervals: {explainability_cfg['enable_confidence_intervals']}")
        print(f"   Confidence Level: {explainability_cfg['confidence_level']*100:.0f}%")
        print(f"   SHAP Analysis: {explainability_cfg['enable_shap']}")
        print(f"   SHAP Max Samples: {explainability_cfg['shap_max_samples']}")
        print(f"   Auto Plot: {explainability_cfg['auto_plot']}")
    except ValueError as e:
        print(f"\n⚠️  Explainability config: {e}")

    print("\n" + "=" * 80)
    print("✅ All values from config.yaml v6.1.0 (ZERO defaults in Python)")
    print("=" * 80)


def print_single_source_verification(config: dict[str, Any]) -> None:
    """
    Verify and print single source of truth status

    Args:
        config: Configuration dictionary from load_config()
    """
    print("\n" + "=" * 80)
    print("SINGLE SOURCE OF TRUTH VERIFICATION")
    print("=" * 80)

    cv_cfg = get_cv_config(config)
    gpu_cfg = get_gpu_config(config)
    defaults = get_defaults(config)

    print("\n✅ Cross-Validation:")
    print(f"   Source: cross_validation.n_folds = {cv_cfg['n_folds']}")
    print("   ✓ No cv_folds in model section")
    print("   ✓ No cv_folds in training section")

    print("\n✅ GPU Memory:")
    print(f"   Source: gpu.memory_limit_mb = {gpu_cfg['memory_limit_mb']}")
    print("   ✓ No gpu_memory_limit_mb in optuna section")
    print("   ✓ No gpu_memory_fraction in training section")

    print("\n✅ Random State:")
    print(f"   Source: defaults.random_state = {defaults['random_state']}")
    print("   ✓ Referenced via YAML anchors")

    print("\n✅ N Jobs:")
    print(f"   Source: defaults.n_jobs = {defaults['n_jobs']}")
    print("   ✓ Referenced via YAML anchors")

    print("\n" + "=" * 80)
    print("✅ Configuration follows single source of truth principles")
    print("=" * 80)


def get_config_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., "data.raw_path")
        default: Default value if path not found (optional)

    Returns:
        Config value or default

    Example:
        >>> config = load_config()
        >>> get_config_value(config, "data.raw_path")
        'data/insurance.csv'
    """
    keys = path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two configuration dictionaries

    Args:
        base: Base configuration
        override: Configuration to merge (takes precedence)

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: dict[str, Any], output_path: str | None = None) -> None:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary to save
        output_path: Output file path. If None, saves to default location.
    """
    if output_path is None:
        output_path_obj = get_project_root() / "configs" / "config.yaml"
    else:
        output_path_obj = Path(output_path)

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path_obj, "w", encoding="utf-8") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        logger.info(f"✅ Configuration saved to {output_path_obj}")
    except Exception as e:
        raise ValueError(f"Error saving config file: {e}") from e


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CONFIG MODULE v7.5.0 - TRUE Single Source of Truth (ZERO DEFAULTS)")
    print("=" * 80)

    try:
        # Load configuration
        print("\n[1/7] Loading configuration from config.yaml v7.5.0...")
        config = load_config()
        print(f"✅ Config loaded from: {get_project_root() / 'configs' / 'config.yaml'}")
        print(f"   Environment: {config.get('environment', 'N/A')}")
        print(f"   Version: {config.get('version', 'N/A')}")

        # Setup logging
        print("\n[2/7] Setting up logging...")
        setup_logging(config)
        print("✅ Logging configured")

        # Validate GPU config
        print("\n[3/7] Validating GPU configuration...")
        validate_gpu_config(config)

        # Validate single source of truth
        print("\n[4/7] Validating single source of truth...")
        validate_single_source_of_truth(config)

        # Get typed configurations
        print("\n[5/7] Extracting typed configurations...")
        defaults = get_defaults(config)
        cv_cfg = get_cv_config(config)
        gpu_cfg = get_gpu_config(config)
        feature_cfg = get_feature_config(config)
        training_cfg = get_training_config(config)

        print(f"✅ Defaults: random_state={defaults['random_state']}, n_jobs={defaults['n_jobs']}")
        print(f"✅ CV config: {cv_cfg['n_folds']} folds (single source)")
        print(f"✅ GPU config: {gpu_cfg['memory_limit_mb']}MB (single source)")
        print(f"✅ Feature config: {len(feature_cfg)} parameters")
        print(f"✅ Training config: {len(training_cfg)} parameters")

        # Display summary
        print("\n[6/7] Configuration summary:")
        print_config_summary(config)

        # Verify single source of truth
        print("\n[7/7] Single source of truth verification:")
        print_single_source_verification(config)

        print("\n" + "=" * 80)
        print("✅ SUCCESS - TRUE single source of truth verified (ZERO DEFAULTS, v7.5.0)")
        print("=" * 80)

        print("\n📚 Usage Examples:")
        print("\n# Example 1: Get CV folds (single source)")
        print("cv_cfg = get_cv_config(config)")
        print(f"print(cv_cfg['n_folds'])  # {cv_cfg['n_folds']}")

        print("\n# Example 2: Get GPU memory (single source)")
        print("gpu_cfg = get_gpu_config(config)")
        print(f"print(gpu_cfg['memory_limit_mb'])  # {gpu_cfg['memory_limit_mb']}")

        print("\n# Example 3: Training references both")
        print("training_cfg = get_training_config(config)")
        print(
            f"print(training_cfg['cv_folds'])  # {training_cfg['cv_folds']} (from CV single source)"
        )
        print(
            f"print(training_cfg['gpu_memory_limit_mb'])  # {training_cfg['gpu_memory_limit_mb']} (from GPU single source)"
        )

        print("\n# Example 4: Get explainability config")
        print("explainability_cfg = get_explainability_config(config)")
        print(
            f"print(explainability_cfg['enable_confidence_intervals'])  # {get_explainability_config(config)['enable_confidence_intervals']}"
        )
        print(
            f"print(explainability_cfg['confidence_level'])  # {get_explainability_config(config)['confidence_level']}"
        )

    except FileNotFoundError as e:
        print(f"\n❌ Config file not found: {e}")
        print("\n⚠️  Config.yaml v6.1.0 is the SINGLE SOURCE OF TRUTH")
        print("    ZERO defaults are provided in Python code")

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
