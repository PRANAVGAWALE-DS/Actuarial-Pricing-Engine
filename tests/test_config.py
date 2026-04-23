"""
Unit tests for src/insurance_ml/config.py

Coverage:
  - load_config: FileNotFoundError on missing file, success on valid yaml
  - _apply_env_overrides: RANDOM_STATE propagates to all 3 sections
  - _validate_config: missing required section, invalid target_transform method
  - validate_single_source_of_truth: all four violation patterns + clean pass
  - get_defaults / get_cv_config: missing-key raises
  - merge_configs: deep merge, no base mutation
  - get_config_value: dot-notation happy path, missing key with/without default
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from insurance_ml.config import (
    _apply_env_overrides,
    _validate_config,
    extract_training_params,
    get_config_value,
    get_conformal_config,
    get_cv_config,
    get_diagnostics_config,
    get_defaults,
    get_explainability_config,
    get_feature_config,
    get_gpu_config,
    get_hardware_config,
    get_high_value_config,
    get_mlflow_config,
    get_prediction_config,
    get_sample_weight_config,
    get_training_config,
    get_validation_config,
    load_config,
    merge_configs,
    save_config,
    validate_gpu_config,
    validate_single_source_of_truth,
)


# ===========================================================================
# load_config
# ===========================================================================


@pytest.mark.unit
class TestLoadConfig:
    def test_raises_file_not_found_on_missing_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(str(missing))

    def test_raises_value_error_on_empty_yaml(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Config file is empty"):
            load_config(str(empty))

    def test_raises_value_error_on_invalid_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(str(bad))

    def test_loads_valid_minimal_config(self, config_yaml_path: Path) -> None:
        cfg = load_config(str(config_yaml_path))
        assert isinstance(cfg, dict)
        assert "defaults" in cfg
        assert "cross_validation" in cfg

    def test_returned_config_is_not_cached_between_calls(
        self, config_yaml_path: Path
    ) -> None:
        """Each call returns an independent dict — mutations don't bleed."""
        cfg1 = load_config(str(config_yaml_path))
        cfg2 = load_config(str(config_yaml_path))
        cfg1["defaults"]["random_state"] = 9999
        assert cfg2["defaults"]["random_state"] != 9999


# ===========================================================================
# _apply_env_overrides
# ===========================================================================


@pytest.mark.unit
class TestApplyEnvOverrides:
    def test_random_state_propagates_to_defaults(
        self, mutable_config: dict[str, Any], monkeypatch
    ) -> None:
        monkeypatch.setenv("RANDOM_STATE", "123")
        result = _apply_env_overrides(mutable_config)
        assert result["defaults"]["random_state"] == 123

    def test_random_state_propagates_to_cross_validation(
        self, mutable_config: dict[str, Any], monkeypatch
    ) -> None:
        monkeypatch.setenv("RANDOM_STATE", "99")
        result = _apply_env_overrides(mutable_config)
        assert result["cross_validation"]["random_state"] == 99

    def test_random_state_propagates_to_data(
        self, mutable_config: dict[str, Any], monkeypatch
    ) -> None:
        monkeypatch.setenv("RANDOM_STATE", "77")
        result = _apply_env_overrides(mutable_config)
        assert result["data"]["random_state"] == 77

    def test_random_state_all_three_sections_updated_simultaneously(
        self, mutable_config: dict[str, Any], monkeypatch
    ) -> None:
        monkeypatch.setenv("RANDOM_STATE", "55")
        result = _apply_env_overrides(mutable_config)
        assert result["defaults"]["random_state"] == 55
        assert result["cross_validation"]["random_state"] == 55
        assert result["data"]["random_state"] == 55

    def test_invalid_random_state_is_ignored(
        self, mutable_config: dict[str, Any], monkeypatch
    ) -> None:
        monkeypatch.setenv("RANDOM_STATE", "not-an-int")
        original_rs = mutable_config["defaults"]["random_state"]
        result = _apply_env_overrides(mutable_config)
        assert result["defaults"]["random_state"] == original_rs

    def test_environment_override(
        self, mutable_config: dict[str, Any], monkeypatch
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        result = _apply_env_overrides(mutable_config)
        assert result["environment"] == "production"

    def test_use_gpu_false_sets_devices_to_cpu(
        self, mutable_config: dict[str, Any], monkeypatch
    ) -> None:
        # Enable GPU first so there are sub-sections to override.
        mutable_config["gpu"]["enabled"] = True
        mutable_config["gpu"]["xgboost"] = {
            "device": "cuda",
            "max_bin": 256,
            "sampling_method": "gradient_based",
            "grow_policy": "lossguide",
            "max_cached_hist_node": 1024,
        }
        mutable_config["gpu"]["lightgbm"] = {
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "max_bin": 255,
            "gpu_use_dp": False,
        }
        monkeypatch.setenv("USE_GPU", "false")
        result = _apply_env_overrides(mutable_config)
        assert result["gpu"]["enabled"] is False
        assert result["gpu"]["xgboost"]["device"] == "cpu"

    def test_no_override_when_env_vars_absent(
        self, mutable_config: dict[str, Any], monkeypatch
    ) -> None:
        for key in ("RANDOM_STATE", "ENVIRONMENT", "USE_GPU", "MODEL_PATH"):
            monkeypatch.delenv(key, raising=False)
        original = copy.deepcopy(mutable_config)
        result = _apply_env_overrides(mutable_config)
        assert result["defaults"]["random_state"] == original["defaults"]["random_state"]


# ===========================================================================
# _validate_config
# ===========================================================================


@pytest.mark.unit
class TestValidateConfig:
    def test_passes_on_minimal_valid_config(
        self, minimal_config: dict[str, Any]
    ) -> None:
        _validate_config(minimal_config)  # must not raise

    def test_raises_on_missing_defaults(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["defaults"]
        with pytest.raises(ValueError, match="missing required sections"):
            _validate_config(mutable_config)

    def test_raises_on_missing_cross_validation(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["cross_validation"]
        with pytest.raises(ValueError, match="missing required sections"):
            _validate_config(mutable_config)

    def test_raises_on_missing_gpu(self, mutable_config: dict[str, Any]) -> None:
        del mutable_config["gpu"]
        with pytest.raises(ValueError, match="missing required sections"):
            _validate_config(mutable_config)

    def test_raises_on_missing_hardware(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["hardware"]
        with pytest.raises(ValueError, match="missing required sections"):
            _validate_config(mutable_config)

    def test_raises_on_invalid_target_transform_method(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["features"]["target_transform"]["method"] = "invalid_method"
        with pytest.raises(ValueError, match="Invalid target transform method"):
            _validate_config(mutable_config)

    def test_raises_on_missing_defaults_random_state(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["defaults"]["random_state"]
        with pytest.raises(ValueError, match="random_state"):
            _validate_config(mutable_config)

    def test_raises_when_diagnostics_section_empty(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["diagnostics"] = {}
        with pytest.raises(ValueError, match="diagnostics"):
            _validate_config(mutable_config)

    def test_contamination_out_of_range_raises_when_outlier_enabled(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["features"]["outlier_removal"]["enabled"] = True
        mutable_config["features"]["outlier_removal"]["contamination"] = 0.6
        with pytest.raises(ValueError, match="contamination"):
            _validate_config(mutable_config)

    def test_vif_threshold_too_low_raises_when_collinearity_enabled(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["features"]["collinearity_removal"]["enabled"] = True
        mutable_config["features"]["collinearity_removal"]["vif_threshold"] = 0.5
        with pytest.raises(ValueError, match="VIF threshold"):
            _validate_config(mutable_config)


# ===========================================================================
# validate_single_source_of_truth
# ===========================================================================


@pytest.mark.unit
class TestValidateSingleSourceOfTruth:
    def test_passes_on_clean_config(
        self, minimal_config: dict[str, Any]
    ) -> None:
        validate_single_source_of_truth(minimal_config)  # must not raise

    def test_raises_when_model_has_cv_folds(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["model"]["cv_folds"] = 5
        with pytest.raises(ValueError, match="model.cv_folds"):
            validate_single_source_of_truth(mutable_config)

    def test_raises_when_training_has_cv_folds(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["training"]["cv_folds"] = 5
        with pytest.raises(ValueError, match="training.cv_folds"):
            validate_single_source_of_truth(mutable_config)

    def test_raises_when_both_validation_size_keys_present(
        self, mutable_config: dict[str, Any]
    ) -> None:
        # data.validation_size + training.val_size = SSOT violation
        mutable_config["data"]["validation_size"] = 0.1
        with pytest.raises(ValueError, match="data.validation_size"):
            validate_single_source_of_truth(mutable_config)

    def test_raises_when_training_has_gpu_memory_fraction(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["training"]["gpu_memory_fraction"] = 0.75
        with pytest.raises(ValueError, match="training.gpu_memory_fraction"):
            validate_single_source_of_truth(mutable_config)

    def test_raises_when_optuna_has_gpu_memory_limit(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["optuna"]["gpu_memory_limit_mb"] = 3000
        with pytest.raises(ValueError, match="optuna.gpu_memory_limit_mb"):
            validate_single_source_of_truth(mutable_config)

    def test_no_violation_without_data_validation_size(
        self, mutable_config: dict[str, Any]
    ) -> None:
        # training.val_size alone should NOT trigger the check
        assert "validation_size" not in mutable_config["data"]
        validate_single_source_of_truth(mutable_config)  # must not raise


# ===========================================================================
# get_defaults
# ===========================================================================


@pytest.mark.unit
class TestGetDefaults:
    def test_returns_expected_keys(
        self, minimal_config: dict[str, Any]
    ) -> None:
        result = get_defaults(minimal_config)
        assert "random_state" in result
        assert "n_jobs" in result

    def test_raises_when_defaults_section_missing(self) -> None:
        with pytest.raises(ValueError, match="'defaults'"):
            get_defaults({})

    def test_raises_when_random_state_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["defaults"]["random_state"]
        with pytest.raises(ValueError, match="random_state"):
            get_defaults(mutable_config)

    def test_raises_when_n_jobs_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["defaults"]["n_jobs"]
        with pytest.raises(ValueError, match="n_jobs"):
            get_defaults(mutable_config)

    def test_values_match_config(self, minimal_config: dict[str, Any]) -> None:
        result = get_defaults(minimal_config)
        assert result["random_state"] == minimal_config["defaults"]["random_state"]
        assert result["n_jobs"] == minimal_config["defaults"]["n_jobs"]


# ===========================================================================
# get_cv_config
# ===========================================================================


@pytest.mark.unit
class TestGetCvConfig:
    def test_returns_expected_keys(
        self, minimal_config: dict[str, Any]
    ) -> None:
        result = get_cv_config(minimal_config)
        for key in ("n_folds", "shuffle", "stratified", "random_state"):
            assert key in result

    def test_raises_when_section_missing(self) -> None:
        with pytest.raises(ValueError, match="'cross_validation'"):
            get_cv_config({})

    def test_raises_when_n_folds_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["cross_validation"]["n_folds"]
        with pytest.raises(ValueError, match="n_folds"):
            get_cv_config(mutable_config)

    def test_values_match_config(self, minimal_config: dict[str, Any]) -> None:
        result = get_cv_config(minimal_config)
        assert result["n_folds"] == minimal_config["cross_validation"]["n_folds"]


# ===========================================================================
# Additional config accessors
# ===========================================================================


@pytest.mark.unit
class TestGetValidationConfig:
    def test_returns_expected_keys(self, minimal_config: dict[str, Any]) -> None:
        result = get_validation_config(minimal_config)
        assert result["contamination_min"] == 0.0
        assert result["contamination_max"] == 0.5
        assert result["vif_min"] == 1.0
        assert result["polynomial_degree_min"] == 1

    def test_raises_when_validation_missing(self) -> None:
        with pytest.raises(ValueError, match="'validation'"):
            get_validation_config({})


@pytest.mark.unit
class TestGetHardwareConfig:
    def test_returns_expected_keys(self, minimal_config: dict[str, Any]) -> None:
        result = get_hardware_config(minimal_config)
        assert result["gpu_model"] == "RTX-3050-test"
        assert result["max_safe_vram_mb"] == 3500

    def test_raises_when_required_key_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["hardware"]["gpu_model"]
        with pytest.raises(ValueError, match="hardware section missing required keys"):
            get_hardware_config(mutable_config)


@pytest.mark.unit
class TestGetGpuConfig:
    def test_disabled_gpu_returns_top_level_config(
        self, minimal_config: dict[str, Any]
    ) -> None:
        result = get_gpu_config(minimal_config)
        assert result["enabled"] is False
        assert result["memory_limit_mb"] == 3000
        assert "xgboost_device" not in result

    def test_enabled_gpu_returns_backend_specific_keys(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["gpu"]["enabled"] = True
        mutable_config["gpu"]["xgboost"]["device"] = "cuda"
        mutable_config["gpu"]["lightgbm"]["device"] = "gpu"
        result = get_gpu_config(mutable_config)
        assert result["xgboost_device"] == "cuda"
        assert result["lightgbm_device"] == "gpu"
        assert result["xgboost_tree_method"] is None

    def test_enabled_gpu_requires_xgboost_section(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["gpu"]["enabled"] = True
        del mutable_config["gpu"]["xgboost"]
        with pytest.raises(ValueError, match="gpu.xgboost section required"):
            get_gpu_config(mutable_config)


@pytest.mark.unit
class TestExtractTrainingParams:
    def test_returns_feature_flags(self, minimal_config: dict[str, Any]) -> None:
        result = extract_training_params(minimal_config)
        assert result == {
            "target_transform": "none",
            "remove_outliers": False,
            "add_polynomials": False,
            "remove_collinear": False,
        }

    def test_raises_when_target_transform_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["features"]["target_transform"]["method"]
        with pytest.raises(ValueError, match="target_transform missing 'method'"):
            extract_training_params(mutable_config)


@pytest.mark.unit
class TestGetFeatureConfig:
    def test_returns_expected_keys(self, minimal_config: dict[str, Any]) -> None:
        result = get_feature_config(minimal_config)
        assert result["smoker_binary_map"] == {"yes": 1, "no": 0}
        assert result["correlation_threshold"] == 0.90
        assert result["outlier_contamination"] == 0.05
        assert result["children_max"] == 20
        assert result["enable_performance_logging"] is False

    def test_raises_when_encoding_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["features"]["encoding"]
        with pytest.raises(ValueError, match="missing subsections"):
            get_feature_config(mutable_config)


@pytest.mark.unit
class TestGetPredictionConfig:
    def test_returns_expected_keys(self, minimal_config: dict[str, Any]) -> None:
        assert get_prediction_config(minimal_config) == {"max_batch_size": 1000}

    def test_raises_on_non_positive_batch_size(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["prediction"]["max_batch_size"] = 0
        with pytest.raises(ValueError, match="positive integer"):
            get_prediction_config(mutable_config)


@pytest.mark.unit
class TestGetConformalConfig:
    def test_returns_float_ratio(self, minimal_config: dict[str, Any]) -> None:
        result = get_conformal_config(minimal_config)
        assert result == {"calibration_split_ratio": 0.2}

    def test_raises_on_ratio_out_of_range(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["conformal"]["calibration_split_ratio"] = 1.0
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            get_conformal_config(mutable_config)


@pytest.mark.unit
class TestGetSampleWeightConfig:
    def test_returns_expected_keys(self, minimal_config: dict[str, Any]) -> None:
        result = get_sample_weight_config(minimal_config)
        assert result["enabled"] is False
        assert result["method"] == "none"
        assert result["tiers"] == []
        assert result["transform"] == "none"

    def test_raises_when_section_missing(self) -> None:
        with pytest.raises(ValueError, match="'sample_weights'"):
            get_sample_weight_config({})


@pytest.mark.unit
class TestGetTrainingConfig:
    def test_returns_expected_values(self, minimal_config: dict[str, Any]) -> None:
        result = get_training_config(minimal_config)
        assert result["cv_folds"] == 5
        assert result["random_state"] == 42
        assert result["gpu_memory_limit_mb"] == 3000
        assert result["checkpoint_enabled"] is False
        assert result["conformal_method"] == "heteroscedastic_conformal"

    def test_defaults_random_state_is_canonical(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["defaults"]["random_state"] = 1234
        mutable_config["cross_validation"]["random_state"] = 9999
        result = get_training_config(mutable_config)
        assert result["random_state"] == 1234

    def test_raises_when_checkpoint_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["training"]["checkpoint"]
        with pytest.raises(ValueError, match="training missing 'checkpoint'"):
            get_training_config(mutable_config)


@pytest.mark.unit
class TestGetMlflowConfig:
    def test_returns_expected_values(self, minimal_config: dict[str, Any]) -> None:
        result = get_mlflow_config(minimal_config)
        assert result["tracking_enabled"] is False
        assert result["experiment_name"] == "test"
        assert result["registry_enabled"] is False
        assert result["autolog_disable"] is True
        assert result["log_gpu_metrics"] is False

    def test_raises_when_tracking_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["mlflow"]["tracking"]
        with pytest.raises(ValueError, match="mlflow missing 'tracking' section"):
            get_mlflow_config(mutable_config)


@pytest.mark.unit
class TestGetHighValueConfig:
    def test_returns_expected_values(self, minimal_config: dict[str, Any]) -> None:
        result = get_high_value_config(minimal_config)
        assert result["enabled"] is False
        assert result["threshold_percentile"] == 95
        assert result["report_format"] == "html"

    def test_raises_when_reporting_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["high_value_analysis"]["reporting"]
        with pytest.raises(ValueError, match="missing 'reporting' section"):
            get_high_value_config(mutable_config)


@pytest.mark.unit
class TestGetDiagnosticsConfig:
    def test_returns_expected_values(self, minimal_config: dict[str, Any]) -> None:
        result = get_diagnostics_config(minimal_config)
        assert result["enabled"] is False
        assert result["performance_enabled"] is False
        assert result["plot_format"] == "png"
        assert result["generate_html_report"] is False
        assert result["shap_max_samples"] == 100

    def test_raises_when_reports_missing(
        self, mutable_config: dict[str, Any]
    ) -> None:
        del mutable_config["diagnostics"]["reports"]
        with pytest.raises(ValueError, match="diagnostics missing 'reports' section"):
            get_diagnostics_config(mutable_config)


@pytest.mark.unit
class TestGetExplainabilityConfig:
    def test_returns_expected_values(self, minimal_config: dict[str, Any]) -> None:
        result = get_explainability_config(minimal_config)
        assert result["enable_confidence_intervals"] is False
        assert result["confidence_level"] == 0.90
        assert result["enable_shap"] is False
        assert result["shap_background_samples"] == 50

    def test_raises_when_confidence_level_invalid(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["diagnostics"]["confidence_level"] = 1.0
        with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\)"):
            get_explainability_config(mutable_config)


@pytest.mark.unit
class TestValidateGpuConfig:
    def test_disabled_gpu_does_not_raise(self, minimal_config: dict[str, Any]) -> None:
        validate_gpu_config(minimal_config)

    def test_raises_when_memory_limit_exceeds_hardware(
        self, mutable_config: dict[str, Any]
    ) -> None:
        mutable_config["gpu"]["enabled"] = True
        mutable_config["gpu"]["memory_limit_mb"] = 9999
        mutable_config["gpu"]["xgboost"]["device"] = "cuda"
        mutable_config["gpu"]["lightgbm"]["device"] = "gpu"
        with pytest.raises(ValueError, match="GPU memory limit"):
            validate_gpu_config(mutable_config)


@pytest.mark.unit
class TestSaveConfig:
    def test_writes_yaml_to_requested_path(
        self, minimal_config: dict[str, Any], tmp_path: Path
    ) -> None:
        output = tmp_path / "saved" / "config.yaml"
        save_config(minimal_config, str(output))
        loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert loaded["prediction"]["max_batch_size"] == 1000
        assert loaded["conformal"]["calibration_split_ratio"] == 0.2


# ===========================================================================
# merge_configs
# ===========================================================================


@pytest.mark.unit
class TestMergeConfigs:
    def test_override_takes_precedence(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = merge_configs(base, override)
        assert result["b"] == 99
        assert result["a"] == 1

    def test_deep_merge_nested_dicts(self) -> None:
        base = {"section": {"x": 1, "y": 2}}
        override = {"section": {"y": 99, "z": 3}}
        result = merge_configs(base, override)
        assert result["section"]["x"] == 1   # preserved from base
        assert result["section"]["y"] == 99  # overridden
        assert result["section"]["z"] == 3   # added from override

    def test_base_not_mutated(self) -> None:
        base = {"a": {"nested": 1}}
        override = {"a": {"nested": 2}}
        base_copy = copy.deepcopy(base)
        merge_configs(base, override)
        assert base == base_copy

    def test_override_not_mutated(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        override_copy = copy.deepcopy(override)
        merge_configs(base, override)
        assert override == override_copy

    def test_new_key_in_override_is_added(self) -> None:
        base = {"a": 1}
        override = {"new_key": 42}
        result = merge_configs(base, override)
        assert result["new_key"] == 42
        assert result["a"] == 1

    def test_empty_override_returns_copy_of_base(self) -> None:
        base = {"a": 1, "b": {"c": 2}}
        result = merge_configs(base, {})
        assert result == base
        assert result is not base  # must be a new object


# ===========================================================================
# get_config_value
# ===========================================================================


@pytest.mark.unit
class TestGetConfigValue:
    def test_single_level_key(self) -> None:
        cfg = {"environment": "test"}
        assert get_config_value(cfg, "environment") == "test"

    def test_nested_key(self) -> None:
        cfg = {"data": {"raw_path": "data/raw/insurance.csv"}}
        assert get_config_value(cfg, "data.raw_path") == "data/raw/insurance.csv"

    def test_deeply_nested_key(self) -> None:
        cfg = {"level1": {"level2": {"level3": 42}}}
        assert get_config_value(cfg, "level1.level2.level3") == 42

    def test_missing_key_returns_none_by_default(self) -> None:
        cfg = {"a": 1}
        assert get_config_value(cfg, "missing.key") is None

    def test_missing_key_returns_explicit_default(self) -> None:
        cfg = {"a": 1}
        assert get_config_value(cfg, "missing.key", default=99) == 99

    def test_existing_key_ignores_default(self) -> None:
        cfg = {"a": 5}
        assert get_config_value(cfg, "a", default=99) == 5

    def test_returns_zero_not_default_for_zero_value(self) -> None:
        """Zero is a valid config value; must not fall through to default."""
        cfg = {"rate": 0}
        assert get_config_value(cfg, "rate", default=99) == 0

    def test_partial_path_missing_returns_default(self) -> None:
        cfg = {"section": {}}
        assert get_config_value(cfg, "section.missing", default=-1) == -1
