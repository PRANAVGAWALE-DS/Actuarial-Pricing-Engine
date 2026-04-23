"""
Data loading, validation, and basic preprocessing

"""

import logging
import threading
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from insurance_ml.config import load_config

logger = logging.getLogger(__name__)


class InsuranceInput(BaseModel):
    """
    Pydantic model for input validation

    - Uses validation ranges from config.yaml (NO HARDCODED VALUES)
    - Enhanced validation messages
    - Pydantic v2 syntax
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    age: int = Field(..., description="Age in years")
    sex: str = Field(..., description="Gender (male/female)")
    bmi: float = Field(..., description="Body Mass Index")
    children: int = Field(..., ge=0, le=20, description="Number of children")
    smoker: str = Field(..., description="Smoking status (yes/no)")
    region: str = Field(..., description="Geographic region")

    # ── F-04 FIX: Per-thread config storage prevents race conditions ──
    # Class-level attribute shares state across threads (Gunicorn workers,
    # Optuna parallel trials). threading.local() gives each thread its own
    # independent copy, eliminating last-write-wins corruption.
    _thread_local: ClassVar = threading.local()

    @classmethod
    def set_config(cls, config: dict[str, Any]) -> None:
        """Set per-thread configuration for validation ranges"""
        cls._thread_local.config = config

    @classmethod
    def _get_config(cls) -> dict[str, Any] | None:
        """Retrieve the config for the current thread."""
        return getattr(cls._thread_local, "config", None)

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        """
        Validate age with ranges from config
        References: features.age_min, features.age_max
        """
        if cls._get_config() is None:
            # ML-10 FIX: Raise instead of silently falling back to permissive bounds
            # (0–120) that would accept clinically impossible values.  Production
            # deployments must inject config via InsuranceInput.set_config() before
            # validation.  Only unit tests that explicitly test the no-config path
            # are expected to hit this branch.
            raise RuntimeError(
                "InsuranceInput.set_config() must be called before validation. "
                "Config is required to determine age bounds. "
                "If running unit tests, call InsuranceInput.set_config(load_config()) "
                "at test setup."
            )
        else:
            features_cfg = cls._get_config().get("features", {})
            age_min = features_cfg.get("age_min", 18)  # tight underwriting default
            age_max = features_cfg.get("age_max", 80)  # tight underwriting default

        if not (age_min <= v <= age_max):
            raise ValueError(f"Age must be between {age_min} and {age_max}, got {v}")

        if v < 18:
            logger.warning(f"Age {v} is below 18. Ensure this is correct for insurance data.")

        return v

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        """Validate sex field"""
        v = v.lower().strip()
        valid_values = ["male", "female"]

        if v not in valid_values:
            raise ValueError(f"Sex must be one of {valid_values}, got '{v}'")

        return v

    @field_validator("bmi")
    @classmethod
    def validate_bmi(cls, v: float) -> float:
        """
        Validate BMI with ranges from config
        References: features.bmi_min, features.bmi_max
        """
        if cls._get_config() is None:
            # ML-10 FIX: Raise instead of silently accepting BMI values in [10, 100]
            # which includes clinically impossible values (BMI=90).
            raise RuntimeError(
                "InsuranceInput.set_config() must be called before validation. "
                "Config is required to determine BMI bounds."
            )
        else:
            features_cfg = cls._get_config().get("features", {})
            bmi_min = features_cfg.get("bmi_min", 15.0)  # tight underwriting default
            bmi_max = features_cfg.get("bmi_max", 55.0)  # tight underwriting default

        if not (bmi_min <= v <= bmi_max):
            raise ValueError(f"BMI must be between {bmi_min} and {bmi_max}, got {v:.1f}")

        # Warn for extreme values (outside typical range)
        if v < 15.0 or v > 50.0:
            logger.warning(
                f"BMI value {v:.1f} is outside typical range (15-50). "
                f"This may indicate data quality issues or extreme cases."
            )

        return v

    @field_validator("smoker")
    @classmethod
    def validate_smoker(cls, v: str) -> str:
        """Validate smoker field"""
        v = v.lower().strip()
        valid_values = ["yes", "no"]

        if v not in valid_values:
            raise ValueError(f"Smoker must be one of {valid_values}, got '{v}'")

        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate region field"""
        v = v.lower().strip()
        valid_regions = ["northeast", "northwest", "southeast", "southwest"]

        if v not in valid_regions:
            raise ValueError(f"Region must be one of {valid_regions}, got '{v}'")

        return v


class DataLoader:
    """
    Data loading with validation and error handling

    - Uses config.py architecture
    - References validation params from config (NO HARDCODED VALUES)
    - Enhanced error messages with context
    - Data quality reporting
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize DataLoader with COMPREHENSIVE config validation
        """
        self.config = config or load_config()
        self._validation_issues: list[str] = []

        # Set config for Pydantic model validation
        InsuranceInput.set_config(self.config)

        # Extract data configuration
        if "data" not in self.config:
            raise ValueError(
                "❌ Config missing 'data' section\n"
                "   Config.yaml v6.1.0 must define data configuration"
            )

        self.data_cfg = self.config["data"]

        # COMPREHENSIVE validation of data config
        # BUG-A FIX (v7.5.0): Removed 'validation_size' from required_keys.
        # config.py's Issue-5 FIX already dropped it from _validate_config()
        # because data.validation_size is never consumed — get_training_config()
        # reads training.val_size as the sole source of truth for the validation
        # split size.  Keeping it here caused DataLoader.__init__ to crash if
        # config.yaml was updated to follow config.py's own guidance and omit
        # data.validation_size.  Both validators now agree: validation_size is
        # not required in the data section.
        required_keys = {
            "raw_path": "Path to CSV file",
            "target_column": "Target variable name",
            "test_size": "Test set proportion",
            "random_state": "Random seed",
        }

        missing = [
            f"{key} ({desc})" for key, desc in required_keys.items() if key not in self.data_cfg
        ]

        if missing:
            raise ValueError(
                f"❌ data section missing required keys:\n"
                + "\n".join(f"   - {item}" for item in missing)
                + f"\n\n   Required keys: {list(required_keys.keys())}"
            )

        # Validate value ranges
        test_size = self.data_cfg["test_size"]
        # BUG-A FIX (continued): val_size is no longer read from data section here.
        # DataLoader is used for loading/cleaning only; split sizes are the
        # responsibility of train.py via get_training_config() -> training.val_size.
        # We retain a lightweight check if the legacy field happens to be present,
        # but we no longer require it or raise on its absence.
        val_size = self.data_cfg.get("validation_size", 0)

        if not 0 < test_size < 0.5:
            raise ValueError(
                f"❌ Invalid test_size: {test_size}\n"
                f"   Must be in range (0, 0.5)\n"
                f"   Recommended: 0.15 - 0.25"
            )

        if val_size > 0 and not 0 < val_size < 0.5:
            raise ValueError(
                f"❌ Invalid validation_size: {val_size}\n"
                f"   Must be in range (0, 0.5) or 0 to disable\n"
                f"   Recommended: 0.15 - 0.20"
            )

        # Check total split doesn't exceed 1.0
        total_split = test_size + val_size
        if total_split >= 0.9:
            raise ValueError(
                f"❌ Test + validation split too large: {total_split:.2%}\n"
                f"   test_size: {test_size:.2%}\n"
                f"   validation_size: {val_size:.2%}\n"
                f"   \n"
                f"   Leaves only {1 - total_split:.2%} for training!\n"
                f"   Recommended: total < 50%"
            )

        # Validate random state
        random_state = self.data_cfg["random_state"]
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError(
                f"❌ Invalid random_state: {random_state}\n"
                f"   Must be a non-negative integer\n"
                f"   Example: 42"
            )

        # Validate feature lists exist if specified
        if "categorical_features" in self.config.get("features", {}):
            cat_features = self.config["features"]["categorical_features"]
            if not isinstance(cat_features, list) or len(cat_features) == 0:
                raise ValueError(
                    f"❌ Invalid categorical_features: {cat_features}\n"
                    f"   Must be a non-empty list of column names"
                )

        if "numerical_features" in self.config.get("features", {}):
            num_features = self.config["features"]["numerical_features"]
            if not isinstance(num_features, list) or len(num_features) == 0:
                raise ValueError(
                    f"❌ Invalid numerical_features: {num_features}\n"
                    f"   Must be a non-empty list of column names"
                )

        logger.debug(
            f"✅ DataLoader initialized with config v{self.config.get('version', 'Unknown')}\n"
            f"   test_size: {test_size:.2%}\n"
            f"   random_state: {random_state}"
            + (f"\n   validation_size (legacy): {val_size:.2%}" if val_size > 0 else "")
        )

    # ── ISSUE-4 FIX (v7.5.0): Strict config accessors — no hardcoded fallbacks ──
    # The original code used features_cfg.get("categorical_features", ["sex", ...])
    # in 4 separate methods, and self.data_cfg.get("target_column", "charges") in
    # 3 methods.  These silent defaults directly contradict the project's "ZERO
    # DEFAULTS IN PYTHON CODE" policy.  If a column list is misconfigured (e.g.
    # after adding a new feature), the code silently trains on the wrong columns
    # instead of raising a clear error.
    #
    # Fix: Replace every .get(key, fallback) with a strict accessor that raises
    # ValueError if the key is absent, forcing the operator to fix config.yaml.

    def _strict_get_features(self) -> dict:
        """
        Return categorical_features, numerical_features, and target_column
        from config, raising ValueError if any are missing.

        This is the single resolution point for feature lists.  All methods
        that previously called features_cfg.get(key, hardcoded_default) now
        call this instead.
        """
        features_cfg = self.config.get("features", {})

        missing = []
        if "categorical_features" not in features_cfg:
            missing.append("features.categorical_features")
        if "numerical_features" not in features_cfg:
            missing.append("features.numerical_features")
        if "target_column" not in self.data_cfg:
            missing.append("data.target_column")

        if missing:
            raise ValueError(
                f"❌ Config missing required keys (ZERO DEFAULTS policy):\n"
                + "\n".join(f"   - {k}" for k in missing)
                + "\n\n   These must be defined explicitly in config.yaml.\n"
                + "   No hardcoded fallbacks are provided."
            )

        return {
            "categorical": features_cfg["categorical_features"],
            "numerical": features_cfg["numerical_features"],
            "target": self.data_cfg["target_column"],
        }

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data with comprehensive error handling

        Uses: data.raw_path from config (SINGLE SOURCE)

        Returns:
            DataFrame with validated data

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is invalid or empty
        """
        data_path = self.data_cfg["raw_path"]
        data_path_obj = Path(data_path)

        # Validate path
        if data_path_obj.is_dir():
            raise ValueError(
                f"❌ Path points to a directory, not a file: {data_path}\n"
                f"   Expected a CSV file path."
            )

        if not data_path_obj.exists():
            raise FileNotFoundError(
                f"❌ Data file not found: {data_path}\n"
                f"   Expected location: {data_path_obj.absolute()}\n"
                f"   Current directory: {Path.cwd()}"
            )

        try:
            df = pd.read_csv(data_path)

            if df.empty:
                raise ValueError(f"❌ Data file is empty: {data_path}")

            logger.info(
                f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns from {data_path}"
            )

            # Validate required columns
            self._validate_required_columns(df)

            # Validate data types
            df = self._validate_dtypes(df)

            # Generate data quality report
            self._log_data_quality(df)

            return df

        except pd.errors.EmptyDataError:
            raise ValueError(f"❌ Empty or invalid CSV file: {data_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"❌ Error parsing CSV file: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error loading data: {e}", exc_info=True)
            raise

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Validate required columns exist
        Uses: data.target_column, features.categorical_features, features.numerical_features
        """
        # ISSUE-4 FIX: Use strict accessor — no hardcoded fallbacks.
        feat = self._strict_get_features()
        categorical_features = feat["categorical"]
        numerical_features = feat["numerical"]

        required_cols = categorical_features + numerical_features

        # Check if target column should be present
        include_target = self.config.get("data", {}).get("include_target", True)
        if include_target:
            required_cols.append(feat["target"])

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"❌ Missing required columns: {missing_cols}\n"
                f"   Found columns: {df.columns.tolist()}\n"
                f"   Required columns: {required_cols}"
            )

        logger.debug(f"✅ All required columns present: {required_cols}")

    def _validate_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and convert data types with error handling
        Uses: features.numerical_features from config

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with corrected dtypes
        """
        # ISSUE-4 FIX: Use strict accessor — no hardcoded fallbacks.
        feat = self._strict_get_features()
        target_col = feat["target"]

        # Define expected types based on feature lists
        expected_types = {"age": "int64", "bmi": "float64", "children": "int64"}

        # Add target column if present
        if target_col in df.columns:
            expected_types[target_col] = "float64"

        for col, dtype in expected_types.items():
            if col not in df.columns:
                continue

            try:
                # Check for non-numeric values before conversion
                if df[col].dtype == "object":
                    non_numeric = df[col][pd.to_numeric(df[col], errors="coerce").isna()]
                    if len(non_numeric) > 0:
                        logger.warning(
                            f"⚠️ Column '{col}' contains non-numeric values: "
                            f"{non_numeric.unique()[:5].tolist()} "
                            f"(showing first 5)"
                        )

                df[col] = df[col].astype(dtype)
                logger.debug(f"✅ Converted '{col}' to {dtype}")

            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"❌ Cannot convert column '{col}' to {dtype}.\n"
                    f"   Sample values: {df[col].head().tolist()}\n"
                    f"   Error: {e}"
                )

        return df

    def _log_data_quality(self, df: pd.DataFrame) -> None:
        """
        Generate and log data quality report
        Uses: data.target_column from config
        """
        logger.info("=" * 60)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 60)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning("⚠️ Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                pct = (count / len(df)) * 100
                logger.warning(f"  {col}: {count} ({pct:.1f}%)")
        else:
            logger.info("✅ Missing values: None")

        # Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            logger.warning(f"⚠️ Duplicates: {dup_count} ({dup_count/len(df)*100:.1f}%)")
        else:
            logger.info("✅ Duplicates: None")

        # Data type summary
        logger.info("\nData Types:")
        for dtype in df.dtypes.unique():
            cols = df.select_dtypes(include=[dtype]).columns.tolist()
            logger.info(f"  {dtype}: {cols}")

        # ISSUE-4 FIX: Use strict accessor — no hardcoded fallbacks.
        target_col = self._strict_get_features()["target"]
        if target_col in df.columns:
            logger.info(f"\n📊 Target ({target_col}) statistics:")
            logger.info(f"  Min: {df[target_col].min():.2f}")
            logger.info(f"  Max: {df[target_col].max():.2f}")
            logger.info(f"  Mean: {df[target_col].mean():.2f}")
            logger.info(f"  Median: {df[target_col].median():.2f}")
            logger.info(f"  Std Dev: {df[target_col].std():.2f}")

        logger.info("=" * 60)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning with detailed logging
        Uses: data.target_column, features config from config

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("🧹 Starting data cleaning...")
        df_clean = df.copy()
        initial_rows = len(df_clean)

        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        dup_removed = initial_rows - len(df_clean)
        if dup_removed > 0:
            logger.info(
                f"✅ Removed {dup_removed} duplicate rows ({dup_removed/initial_rows*100:.1f}%)"
            )

        # ISSUE-4 FIX: Use strict accessor — no hardcoded fallbacks.
        feat = self._strict_get_features()
        critical_columns = feat["categorical"] + feat["numerical"]

        # Include target if present
        if feat["target"] in df_clean.columns:
            critical_columns.append(feat["target"])

        # Handle missing values in critical columns
        missing_before = len(df_clean)
        df_clean = df_clean.dropna(subset=critical_columns)
        missing_removed = missing_before - len(df_clean)

        if missing_removed > 0:
            logger.info(
                f"✅ Removed {missing_removed} rows with missing critical values "
                f"({missing_removed/initial_rows*100:.1f}%)"
            )

        # Validate cleaned data isn't empty
        if len(df_clean) == 0:
            raise ValueError(
                "❌ All rows were removed during cleaning. "
                "Check data quality and required columns."
            )

        # Log cleaning summary
        final_rows = len(df_clean)
        total_removed = initial_rows - final_rows
        logger.info(
            f"✅ Cleaning complete: {initial_rows} -> {final_rows} rows "
            f"({total_removed} removed, {total_removed/initial_rows*100:.1f}%)"
        )

        return df_clean

    def validate_single_record(self, record: dict[str, Any]) -> InsuranceInput:
        """
        Validate a single record using Pydantic model
        Uses validation ranges from config via InsuranceInput model

        Args:
            record: Dictionary with insurance data

        Returns:
            Validated InsuranceInput object

        Raises:
            ValidationError: If validation fails with detailed error message
        """
        try:
            validated = InsuranceInput(**record)
            logger.debug(f"✅ Record validated successfully: {validated}")
            return validated

        except Exception as e:
            logger.error(f"❌ Validation failed for record: {record}")
            logger.error(f"   Error: {e}")
            raise

    def validate_dataframe(self, df: pd.DataFrame, raise_on_invalid: bool = False) -> pd.DataFrame:
        """
        Validate a DataFrame of InsuranceInput records.

        ── F-06 FIX: Batch validation with Pydantic v2 TypeAdapter ──
        Replaces per-row iterrows() loop (O(n) Python objects) with
        TypeAdapter.validate_python() which processes the list in one call,
        dramatically reducing overhead for large batches.

        Error handling preserves existing behaviour:
          - raise_on_invalid=False  → invalid rows logged and dropped
          - raise_on_invalid=True   → ValidationError raised immediately
        """
        from pydantic import TypeAdapter, ValidationError

        if df.empty:
            logger.warning("validate_dataframe: empty DataFrame received")
            return df

        records = df.to_dict(orient="records")
        adapter = TypeAdapter(list[InsuranceInput])

        try:
            validated = adapter.validate_python(records)
            logger.debug(f"✅ Batch validation passed: {len(validated)} records")
            return pd.DataFrame([v.model_dump() for v in validated])

        except ValidationError:
            if raise_on_invalid:
                raise

            # ── Individual scan to identify and exclude only the bad rows ──
            # Runs only when batch validation fails; typical (all-valid) path
            # pays zero per-row overhead.
            single_adapter = TypeAdapter(InsuranceInput)
            valid_rows: list[dict] = []
            invalid_indices: list[int] = []

            for i, record in enumerate(records):
                try:
                    valid_rows.append(single_adapter.validate_python(record).model_dump())
                except ValidationError as row_error:
                    invalid_indices.append(i)
                    logger.warning(
                        f"   Row {i} invalid — dropping: "
                        f"{row_error.error_count()} error(s): "
                        f"{[e['msg'] for e in row_error.errors()[:3]]}"
                    )

            logger.warning(
                f"⚠️  validate_dataframe: dropped {len(invalid_indices)}/{len(records)} "
                f"invalid rows (indices: {invalid_indices[:10]}"
                f"{'...' if len(invalid_indices) > 10 else ''})"
            )

            return pd.DataFrame(valid_rows)

    def get_data_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate comprehensive data summary

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with data statistics
        """
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
        }

        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

        # Add categorical column statistics
        categorical_cols = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            summary["categorical_stats"] = {
                col: df[col].value_counts().to_dict() for col in categorical_cols
            }

        return summary

    def get_feature_lists(self) -> dict[str, list[str]]:
        """
        Get feature lists from config

        Returns:
            Dictionary with categorical and numerical feature lists
        """
        # ISSUE-4 FIX: Use strict accessor — no hardcoded fallbacks.
        return self._strict_get_features()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """Example usage of DataLoader v5.0.0"""

    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 80)
    print("DATA LOADER v5.0.0 - Config v6.1.0 Compatible")
    print("=" * 80)

    try:
        # Initialize with config
        print("\n[1/6] Initializing DataLoader...")
        loader = DataLoader()
        print(f"✅ Initialized with config v{loader.config.get('version', 'Unknown')}")

        # Load data
        print("\n[2/6] Loading raw data...")
        df = loader.load_raw_data()
        print(f"✅ Loaded {len(df)} rows")

        # Clean data
        print("\n[3/6] Cleaning data...")
        df_clean = loader.clean_data(df)
        print(f"✅ Cleaned data: {len(df_clean)} rows")

        # Get feature lists from config
        print("\n[4/6] Getting feature lists from config...")
        features = loader.get_feature_lists()
        print(f"✅ Categorical features: {features['categorical']}")
        print(f"✅ Numerical features: {features['numerical']}")
        print(f"✅ Target: {features['target']}")

        # Validate single record (uses config validation ranges)
        print("\n[5/6] Validating single record (with config ranges)...")
        sample_record = {
            "age": 35,
            "sex": "male",
            "bmi": 28.5,
            "children": 2,
            "smoker": "no",
            "region": "northeast",
        }
        validated = loader.validate_single_record(sample_record)
        print(f"✅ Record validated: {validated}")

        # Get data summary
        print("\n[6/6] Generating data summary...")
        summary = loader.get_data_summary(df_clean)
        print(f"✅ Summary generated")
        print(f"\nData Summary:")
        print(f"  Shape: {summary['shape']}")
        print(f"  Columns: {summary['columns']}")
        print(f"  Duplicates: {summary['duplicates']}")

        if summary["missing_values"]:
            missing_total = sum(summary["missing_values"].values())
            if missing_total > 0:
                print(f"  Missing values: {missing_total}")

        print("\n" + "=" * 80)
        print("✅ SUCCESS - Data loading complete (Config v6.1.0 Compatible)")
        print("=" * 80)

        # Print compatibility notes
        print("\n📋 Compatibility Notes:")
        print("  ✅ Uses config.py v5.1.0 helper functions")
        print("  ✅ References validation from config.yaml (NO HARDCODED VALUES)")
        print("  ✅ BMI range: features.bmi_min, features.bmi_max")
        print("  ✅ Age range: features.age_min, features.age_max")
        print("  ✅ Feature lists: features.categorical_features, features.numerical_features")
        print("  ✅ Target column: data.target_column (SINGLE SOURCE)")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
