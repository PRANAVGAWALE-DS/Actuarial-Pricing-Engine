"""
Unit tests for src/insurance_ml/data.py

Coverage:
  InsuranceInput (Pydantic model):
    - set_config() required before validation (ML-10 FIX — raises RuntimeError without it)
    - age validated against config bounds, not hardcoded
    - bmi validated against config bounds
    - sex normalised to lowercase, valid values enforced
    - smoker normalised to lowercase, valid values enforced
    - region normalised to lowercase, valid values enforced
    - children: ge=0, le=20 enforced
    - validate_assignment=True: post-construction mutation re-validates

  DataLoader.__init__:
    - missing 'data' section raises ValueError
    - invalid test_size raises ValueError
    - test_size + validation_size >= 0.9 raises ValueError
    - invalid random_state raises ValueError
    - valid minimal config initialises cleanly

  DataLoader.clean_data:
    - removes duplicate rows
    - drops rows with NaN in critical columns
    - raises ValueError if all rows removed

  DataLoader.validate_single_record:
    - valid record returns InsuranceInput
    - invalid record raises (propagated from Pydantic)

  DataLoader.validate_dataframe:
    - all-valid DataFrame returns all rows
    - DataFrame with one invalid row (raise_on_invalid=False) drops the row
    - DataFrame with one invalid row (raise_on_invalid=True) raises ValidationError
    - empty DataFrame returns empty DataFrame

  DataLoader.get_data_summary:
    - returns required keys (shape, columns, missing_values, duplicates)
    - numeric_stats populated for numeric columns
    - categorical_stats populated for object columns

  DataLoader._strict_get_features:
    - raises ValueError if categorical_features missing from config
    - raises ValueError if numerical_features missing from config
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from insurance_ml.data import DataLoader, InsuranceInput

# ===========================================================================
# Helpers
# ===========================================================================


def _make_dataloader_config(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    include_validation_size: bool = False,
    validation_size: float = 0.1,
) -> dict[str, Any]:
    """
    Minimal config dict for DataLoader — includes the keys DataLoader.__init__
    actually validates.  categorical_features and numerical_features are
    required by _strict_get_features() but only when those methods are called.
    """
    cfg = {
        "version": "test-1.0",
        "data": {
            "raw_path": "data/raw/insurance.csv",
            "target_column": "charges",
            "test_size": test_size,
            "random_state": random_state,
        },
        "features": {
            "categorical_features": ["sex", "smoker", "region"],
            "numerical_features": ["age", "bmi", "children"],
            "age_min": 0.0,
            "age_max": 120.0,
            "bmi_min": 10.0,
            "bmi_max": 100.0,
            "engineering": {"variance_threshold": 1e-6},
            "target_transform": {"method": "none"},
            "outlier_removal": {"enabled": False, "contamination": 0.05, "random_state": 42},
            "polynomial_features": {"enabled": False, "degree": 2, "max_features": 50},
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
        },
        "defaults": {"random_state": 42, "n_jobs": 1},
        "cross_validation": {
            "n_folds": 5,
            "shuffle": True,
            "stratified": False,
            "random_state": 42,
        },
        "gpu": {
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
        },
        "hardware": {
            "gpu_model": "RTX-test",
            "total_vram_mb": 4096,
            "max_safe_vram_mb": 3500,
            "recommended_vram_limit_mb": 3000,
        },
        "model": {"models": ["xgboost"], "metric": "rmse", "model_path": "models/"},
        "models": {"xgboost": {}},
        "optuna": {"n_trials": 10, "enhanced_scoring": {"enabled": False}},
        "training": {"val_size": 0.1, "early_stopping_rounds": 10},
        "diagnostics": {
            "enabled": False,
            "max_samples": 100,
            "batch_size": 50,
            "rf_tree_batch_size": 5,
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
                "shap_max_samples": 50,
                "shap_background_samples": 25,
                "residual_sample_size": 100,
                "autocorr_lag_limit": 5,
                "calibration_bins": 5,
                "learning_curve_points": 3,
                "permutation_importance_repeats": 2,
            },
        },
        "mlflow": {
            "tracking": {
                "tracking_uri": "sqlite:///data/databases/mlflow.db",
                "experiment_name": "test",
            },
            "autolog": False,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "file": "logs/test.log",
        },
        "high_value_analysis": {"enabled": False},
        "overfitting_analysis": {"enabled": False},
        "monitoring": {"drift_detection_enabled": False},
        "validation": {
            "contamination_min": 0.0,
            "contamination_max": 0.5,
            "vif_min": 1.0,
            "polynomial_degree_min": 1,
            "valid_xgboost_devices": ["cpu", "cuda", "cuda:0"],
            "valid_target_transform_methods": ["none", "log1p", "yeo-johnson"],
            "weight_sum_tolerance": 0.01,
        },
        "sample_weights": {"enabled": False},
        "hybrid_predictor": {
            "threshold": 4500.0,
            "blend_ratio": 0.70,
            "use_soft_blending": True,
            "soft_blend_window": 500.0,
            "max_actuarial_uplift_ratio": 1.15,
            "calibration": {"enabled": True, "factor": 1.00, "apply_to_ml_only": True},
            "business_config": {},
        },
        "evaluation": {
            "segment_thresholds": {
                "low_value": 4500.0,
                "standard": 15000.0,
                "high_value": 30000.0,
            }
        },
    }
    if include_validation_size:
        cfg["data"]["validation_size"] = validation_size
    return cfg


def _make_loader(overrides: dict | None = None) -> DataLoader:
    cfg = _make_dataloader_config()
    if overrides:
        for k, v in overrides.items():
            cfg["data"][k] = v
    return DataLoader(config=cfg)


def _raw_df() -> pd.DataFrame:
    """Minimal DataFrame matching the insurance schema."""
    return pd.DataFrame(
        {
            "age": [25, 45, 30, 55, 25],
            "sex": ["male", "female", "male", "female", "male"],
            "bmi": [22.5, 30.0, 28.5, 35.0, 22.5],
            "children": [0, 2, 1, 3, 0],
            "smoker": ["no", "yes", "no", "no", "no"],
            "region": [
                "northeast",
                "southwest",
                "southeast",
                "northwest",
                "northeast",
            ],
            "charges": [3200.0, 15000.0, 5000.0, 22000.0, 3200.0],
        }
    )


# ===========================================================================
# InsuranceInput — ML-10 FIX: set_config required
# ===========================================================================


@pytest.mark.unit
class TestInsuranceInputSetConfigRequired:
    def test_raises_runtime_error_without_set_config(self) -> None:
        """
        ML-10 FIX: InsuranceInput must raise RuntimeError when no config
        has been set via set_config().  Previously it silently accepted
        permissive bounds (0-120 age, 10-100 BMI) even without config.
        """
        # Clear any config that may have been set by a previous test
        InsuranceInput._thread_local.config = None

        with pytest.raises(RuntimeError, match="set_config"):
            InsuranceInput(
                age=35, sex="male", bmi=27.5, children=2, smoker="no", region="northeast"
            )

    def test_passes_after_set_config(self) -> None:
        cfg = _make_dataloader_config()
        InsuranceInput.set_config(cfg)
        rec = InsuranceInput(
            age=35, sex="male", bmi=27.5, children=2, smoker="no", region="northeast"
        )
        assert rec.age == 35


# ===========================================================================
# InsuranceInput — field validation (with config set)
# ===========================================================================


@pytest.fixture(autouse=True)
def _set_insurance_input_config():
    """Ensure InsuranceInput has a valid config for every test in this file."""
    InsuranceInput.set_config(_make_dataloader_config())
    yield
    InsuranceInput._thread_local.config = None


@pytest.mark.unit
class TestInsuranceInputAge:
    def test_valid_age(self) -> None:
        rec = InsuranceInput(
            age=40, sex="male", bmi=25.0, children=0, smoker="no", region="northeast"
        )
        assert rec.age == 40

    def test_age_below_config_min_raises(self) -> None:
        # config age_min=0.0; no values below 0 accepted (age is int, 0 is boundary)
        # Testing that the validator fires — use a negative value
        with pytest.raises(ValidationError):
            InsuranceInput(
                age=-1, sex="male", bmi=25.0, children=0, smoker="no", region="northeast"
            )

    def test_age_above_config_max_raises(self) -> None:
        # config age_max=120; 121 should fail
        with pytest.raises(ValidationError):
            InsuranceInput(
                age=121, sex="male", bmi=25.0, children=0, smoker="no", region="northeast"
            )

    def test_age_at_max_passes(self) -> None:
        rec = InsuranceInput(
            age=120, sex="male", bmi=25.0, children=0, smoker="no", region="northeast"
        )
        assert rec.age == 120


@pytest.mark.unit
class TestInsuranceInputBmi:
    def test_valid_bmi(self) -> None:
        rec = InsuranceInput(
            age=35, sex="male", bmi=27.5, children=0, smoker="no", region="northeast"
        )
        assert rec.bmi == 27.5

    def test_bmi_below_config_min_raises(self) -> None:
        with pytest.raises(ValidationError):
            InsuranceInput(age=35, sex="male", bmi=5.0, children=0, smoker="no", region="northeast")

    def test_bmi_above_config_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            InsuranceInput(
                age=35, sex="male", bmi=101.0, children=0, smoker="no", region="northeast"
            )

    def test_bmi_at_max_passes(self) -> None:
        rec = InsuranceInput(
            age=35, sex="male", bmi=100.0, children=0, smoker="no", region="northeast"
        )
        assert rec.bmi == 100.0


@pytest.mark.unit
class TestInsuranceInputSex:
    def test_male_normalised_to_lowercase(self) -> None:
        rec = InsuranceInput(
            age=35, sex="MALE", bmi=25.0, children=0, smoker="no", region="northeast"
        )
        assert rec.sex == "male"

    def test_female_normalised_to_lowercase(self) -> None:
        rec = InsuranceInput(
            age=35, sex="Female", bmi=25.0, children=0, smoker="no", region="northeast"
        )
        assert rec.sex == "female"

    def test_invalid_sex_raises(self) -> None:
        with pytest.raises(ValidationError, match="[Ss]ex"):
            InsuranceInput(
                age=35, sex="unknown", bmi=25.0, children=0, smoker="no", region="northeast"
            )

    def test_whitespace_stripped(self) -> None:
        rec = InsuranceInput(
            age=35, sex="  male  ", bmi=25.0, children=0, smoker="no", region="northeast"
        )
        assert rec.sex == "male"


@pytest.mark.unit
class TestInsuranceInputSmoker:
    def test_yes_normalised(self) -> None:
        rec = InsuranceInput(
            age=35, sex="male", bmi=25.0, children=0, smoker="YES", region="northeast"
        )
        assert rec.smoker == "yes"

    def test_no_normalised(self) -> None:
        rec = InsuranceInput(
            age=35, sex="male", bmi=25.0, children=0, smoker="NO", region="northeast"
        )
        assert rec.smoker == "no"

    def test_invalid_smoker_raises(self) -> None:
        with pytest.raises(ValidationError, match="[Ss]moker"):
            InsuranceInput(
                age=35, sex="male", bmi=25.0, children=0, smoker="maybe", region="northeast"
            )


@pytest.mark.unit
class TestInsuranceInputRegion:
    def test_valid_regions(self) -> None:
        for region in ["northeast", "northwest", "southeast", "southwest"]:
            rec = InsuranceInput(
                age=35, sex="male", bmi=25.0, children=0, smoker="no", region=region
            )
            assert rec.region == region

    def test_region_normalised_to_lowercase(self) -> None:
        rec = InsuranceInput(
            age=35, sex="male", bmi=25.0, children=0, smoker="no", region="NORTHEAST"
        )
        assert rec.region == "northeast"

    def test_invalid_region_raises(self) -> None:
        with pytest.raises(ValidationError, match="[Rr]egion"):
            InsuranceInput(age=35, sex="male", bmi=25.0, children=0, smoker="no", region="midwest")


@pytest.mark.unit
class TestInsuranceInputChildren:
    def test_zero_children_passes(self) -> None:
        rec = InsuranceInput(
            age=35, sex="male", bmi=25.0, children=0, smoker="no", region="northeast"
        )
        assert rec.children == 0

    def test_negative_children_raises(self) -> None:
        with pytest.raises(ValidationError):
            InsuranceInput(
                age=35, sex="male", bmi=25.0, children=-1, smoker="no", region="northeast"
            )

    def test_twenty_children_passes(self) -> None:
        rec = InsuranceInput(
            age=35, sex="male", bmi=25.0, children=20, smoker="no", region="northeast"
        )
        assert rec.children == 20

    def test_twenty_one_children_raises(self) -> None:
        with pytest.raises(ValidationError):
            InsuranceInput(
                age=35, sex="male", bmi=25.0, children=21, smoker="no", region="northeast"
            )


# ===========================================================================
# DataLoader.__init__
# ===========================================================================


@pytest.mark.unit
class TestDataLoaderInit:
    def test_valid_config_initialises(self) -> None:
        loader = DataLoader(config=_make_dataloader_config())
        assert loader is not None

    def test_raises_when_data_section_missing(self) -> None:
        cfg = _make_dataloader_config()
        del cfg["data"]
        with pytest.raises(ValueError, match="'data'"):
            DataLoader(config=cfg)

    def test_raises_when_test_size_is_zero(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            DataLoader(config=_make_dataloader_config(test_size=0.0))

    def test_raises_when_test_size_is_point_five(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            DataLoader(config=_make_dataloader_config(test_size=0.5))

    def test_raises_when_test_size_exceeds_point_five(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            DataLoader(config=_make_dataloader_config(test_size=0.7))

    def test_test_size_0_2_passes(self) -> None:
        loader = DataLoader(config=_make_dataloader_config(test_size=0.2))
        assert loader is not None

    def test_raises_when_random_state_is_negative(self) -> None:
        with pytest.raises(ValueError, match="random_state"):
            DataLoader(config=_make_dataloader_config(random_state=-1))

    def test_raises_when_random_state_is_not_int(self) -> None:
        cfg = _make_dataloader_config()
        cfg["data"]["random_state"] = "forty-two"
        with pytest.raises(ValueError, match="random_state"):
            DataLoader(config=cfg)

    def test_raises_when_total_split_too_large(self) -> None:
        """test_size + validation_size >= 0.9 must raise."""
        cfg = _make_dataloader_config(
            test_size=0.45, include_validation_size=True, validation_size=0.45
        )
        with pytest.raises(ValueError, match="split too large"):
            DataLoader(config=cfg)

    def test_raises_when_target_column_missing_from_data_section(self) -> None:
        cfg = _make_dataloader_config()
        del cfg["data"]["target_column"]
        with pytest.raises(ValueError, match="target_column"):
            DataLoader(config=cfg)

    def test_raises_when_raw_path_missing(self) -> None:
        cfg = _make_dataloader_config()
        del cfg["data"]["raw_path"]
        with pytest.raises(ValueError, match="raw_path"):
            DataLoader(config=cfg)


# ===========================================================================
# DataLoader.clean_data
# ===========================================================================


@pytest.mark.unit
class TestDataLoaderCleanData:
    def _loader(self) -> DataLoader:
        return _make_loader()

    def test_removes_duplicate_rows(self) -> None:
        loader = self._loader()
        df = _raw_df()  # Row 0 and Row 4 are identical
        assert df.duplicated().sum() == 1
        cleaned = loader.clean_data(df)
        assert cleaned.duplicated().sum() == 0

    def test_row_count_reduced_after_dedup(self) -> None:
        loader = self._loader()
        df = _raw_df()
        cleaned = loader.clean_data(df)
        assert len(cleaned) < len(df)

    def test_drops_rows_with_nan_in_critical_columns(self) -> None:
        loader = self._loader()
        df = _raw_df()
        df.loc[1, "bmi"] = float("nan")
        cleaned = loader.clean_data(df)
        assert cleaned["bmi"].isna().sum() == 0

    def test_non_critical_nan_not_dropped(self) -> None:
        """A NaN in a column not in the critical list should NOT remove the row."""
        loader = self._loader()
        df = _raw_df()
        df["extra_column"] = [1.0, float("nan"), 3.0, 4.0, 1.0]
        cleaned = loader.clean_data(df)
        # The extra_column NaN row should still be present (critical cols are fine)
        assert (
            float("nan") in cleaned["extra_column"].values or cleaned["extra_column"].isna().any()
        )

    def test_raises_when_all_rows_removed(self) -> None:
        loader = self._loader()
        df = _raw_df()
        # Put NaN in every row's age
        df["age"] = float("nan")
        with pytest.raises(ValueError, match="All rows were removed"):
            loader.clean_data(df)

    def test_returns_copy_not_original(self) -> None:
        loader = self._loader()
        df = _raw_df()
        original_len = len(df)
        loader.clean_data(df)
        assert len(df) == original_len  # original unchanged


# ===========================================================================
# DataLoader.validate_single_record
# ===========================================================================


@pytest.mark.unit
class TestDataLoaderValidateSingleRecord:
    def _loader(self) -> DataLoader:
        return _make_loader()

    def test_valid_record_returns_insurance_input(self) -> None:
        loader = self._loader()
        record = {
            "age": 35,
            "sex": "male",
            "bmi": 27.5,
            "children": 2,
            "smoker": "no",
            "region": "northeast",
        }
        validated = loader.validate_single_record(record)
        assert isinstance(validated, InsuranceInput)
        assert validated.age == 35

    def test_invalid_sex_raises(self) -> None:
        loader = self._loader()
        with pytest.raises(ValidationError):
            loader.validate_single_record(
                {
                    "age": 35,
                    "sex": "alien",
                    "bmi": 27.5,
                    "children": 0,
                    "smoker": "no",
                    "region": "northeast",
                }
            )

    def test_invalid_region_raises(self) -> None:
        loader = self._loader()
        with pytest.raises(ValidationError):
            loader.validate_single_record(
                {
                    "age": 35,
                    "sex": "male",
                    "bmi": 27.5,
                    "children": 0,
                    "smoker": "no",
                    "region": "mars",
                }
            )

    def test_normalises_values(self) -> None:
        loader = self._loader()
        record = {
            "age": 35,
            "sex": "MALE",
            "bmi": 27.5,
            "children": 0,
            "smoker": "NO",
            "region": "NORTHEAST",
        }
        validated = loader.validate_single_record(record)
        assert validated.sex == "male"
        assert validated.smoker == "no"
        assert validated.region == "northeast"


# ===========================================================================
# DataLoader.validate_dataframe
# ===========================================================================


@pytest.mark.unit
class TestDataLoaderValidateDataframe:
    def _loader(self) -> DataLoader:
        return _make_loader()

    def _valid_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [25, 45, 30],
                "sex": ["male", "female", "male"],
                "bmi": [22.5, 30.0, 28.5],
                "children": [0, 2, 1],
                "smoker": ["no", "yes", "no"],
                "region": ["northeast", "southwest", "southeast"],
            }
        )

    def test_all_valid_returns_all_rows(self) -> None:
        loader = self._loader()
        df = self._valid_df()
        result = loader.validate_dataframe(df)
        assert len(result) == 3

    def test_empty_dataframe_returns_empty(self) -> None:
        loader = self._loader()
        df = pd.DataFrame(columns=["age", "sex", "bmi", "children", "smoker", "region"])
        result = loader.validate_dataframe(df)
        assert len(result) == 0

    def test_one_invalid_row_dropped_when_raise_false(self) -> None:
        loader = self._loader()
        df = self._valid_df()
        # Make row 1 invalid by setting sex to an invalid value
        df.loc[1, "sex"] = "invalid_value"
        result = loader.validate_dataframe(df, raise_on_invalid=False)
        assert len(result) == 2  # only 2 valid rows remain

    def test_one_invalid_row_raises_when_raise_true(self) -> None:
        loader = self._loader()
        df = self._valid_df()
        df.loc[0, "smoker"] = "MAYBE"
        with pytest.raises(ValidationError):
            loader.validate_dataframe(df, raise_on_invalid=True)

    def test_output_dtypes_are_pydantic_output(self) -> None:
        loader = self._loader()
        df = self._valid_df()
        result = loader.validate_dataframe(df)
        # Pydantic normalises sex/smoker/region to lowercase
        assert all(v == v.lower() for v in result["sex"])


# ===========================================================================
# DataLoader.get_data_summary
# ===========================================================================


@pytest.mark.unit
class TestDataLoaderGetDataSummary:
    def _loader(self) -> DataLoader:
        return _make_loader()

    def test_returns_required_keys(self) -> None:
        loader = self._loader()
        df = _raw_df()
        summary = loader.get_data_summary(df)
        for key in ("shape", "columns", "missing_values", "duplicates"):
            assert key in summary, f"Missing key: {key}"

    def test_shape_correct(self) -> None:
        loader = self._loader()
        df = _raw_df()
        summary = loader.get_data_summary(df)
        assert summary["shape"] == df.shape

    def test_duplicates_count_correct(self) -> None:
        loader = self._loader()
        df = _raw_df()
        summary = loader.get_data_summary(df)
        assert summary["duplicates"] == int(df.duplicated().sum())

    def test_missing_values_dict(self) -> None:
        loader = self._loader()
        df = _raw_df()
        df.loc[0, "bmi"] = float("nan")
        summary = loader.get_data_summary(df)
        assert summary["missing_values"]["bmi"] == 1

    def test_numeric_stats_present_for_numeric_cols(self) -> None:
        loader = self._loader()
        df = _raw_df()
        summary = loader.get_data_summary(df)
        assert "numeric_stats" in summary
        assert "age" in summary["numeric_stats"]

    def test_categorical_stats_present_for_object_cols(self) -> None:
        loader = self._loader()
        df = _raw_df()
        summary = loader.get_data_summary(df)
        assert "categorical_stats" in summary
        assert "sex" in summary["categorical_stats"]


# ===========================================================================
# DataLoader._strict_get_features
# ===========================================================================


@pytest.mark.unit
class TestDataLoaderStrictGetFeatures:
    def test_raises_when_categorical_features_missing(self) -> None:
        cfg = _make_dataloader_config()
        del cfg["features"]["categorical_features"]
        loader = DataLoader(config=cfg)
        with pytest.raises(ValueError, match="categorical_features"):
            loader._strict_get_features()

    def test_raises_when_numerical_features_missing(self) -> None:
        cfg = _make_dataloader_config()
        del cfg["features"]["numerical_features"]
        loader = DataLoader(config=cfg)
        with pytest.raises(ValueError, match="numerical_features"):
            loader._strict_get_features()

    def test_raises_when_target_column_missing(self) -> None:
        cfg = _make_dataloader_config()
        del cfg["data"]["target_column"]
        # target_column is required at init; patch after construction
        loader = DataLoader.__new__(DataLoader)
        loader.config = cfg
        loader.data_cfg = cfg["data"]
        with pytest.raises(ValueError, match="target_column"):
            loader._strict_get_features()

    def test_returns_correct_keys(self) -> None:
        loader = DataLoader(config=_make_dataloader_config())
        result = loader._strict_get_features()
        assert "categorical" in result
        assert "numerical" in result
        assert "target" in result

    def test_target_is_charges(self) -> None:
        loader = DataLoader(config=_make_dataloader_config())
        result = loader._strict_get_features()
        assert result["target"] == "charges"
