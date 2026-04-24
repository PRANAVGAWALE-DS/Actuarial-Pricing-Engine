# insurance_ml/shared.py

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class TargetTransformation:
    method: Literal["none", "log1p", "boxcox", "yeo-johnson"] = "none"

    lambda_param: float | None = None
    boxcox_lambda: float | None = None  # Deprecated, use lambda_param

    original_range: tuple[float, float] | None = None

    # Store transformed range for BoxCox/Yeo-Johnson
    transform_min: float | None = None
    transform_max: float | None = None

    # Backward compatibility aliases
    boxcox_min: float | None = None
    boxcox_max: float | None = None

    _skip_validation: bool = False

    _log_residual_variance: float | None = None

    _is_deserialized: bool = False  # Track if loaded from disk

    def __post_init__(self):
        """Validate transformation parameters"""

        valid_methods = ["none", "log1p", "boxcox", "yeo-johnson"]

        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid transformation method: {self.method}. "
                f"Must be one of: {', '.join(repr(m) for m in valid_methods)}"
            )

        # Handle backward compatibility for boxcox_lambda
        if self.boxcox_lambda is not None and self.lambda_param is None:
            self.lambda_param = self.boxcox_lambda
            warnings.warn(
                "boxcox_lambda is deprecated, use lambda_param instead",
                DeprecationWarning,
                stacklevel=2,
            )

        # Strict validation when loaded from disk — lambda_param must be present
        if self._is_deserialized and self.method in ["boxcox", "yeo-johnson"]:
            if self.lambda_param is None:
                raise ValueError(
                    f"Cannot load {self.method} transformation with lambda_param=None\n"
                    f"  This model file is corrupted or was saved incorrectly.\n"
                    f"  Required: Retrain model and save with proper lambda_param.\n"
                    f"  Current values:\n"
                    f"    method: {self.method}\n"
                    f"    lambda_param: {self.lambda_param}\n"
                    f"    transform_min: {self.transform_min}\n"
                    f"    transform_max: {self.transform_max}"
                )

        # During fit, lambda_param is calculated AFTER TargetTransformation is created.
        # This is intentional - do NOT change this!
        if (
            not self._skip_validation
            and not self._is_deserialized
            and self.method in ["boxcox", "yeo-johnson"]
        ):
            if self.lambda_param is None and self.boxcox_lambda is None:
                # This is OK during fit (lambda will be set later)
                pass

            # Sync backward compatibility fields
            if self.transform_min is None and self.boxcox_min is not None:
                self.transform_min = self.boxcox_min
            if self.transform_max is None and self.boxcox_max is not None:
                self.transform_max = self.boxcox_max

            # Warn if min/max not set (but only if lambda is set, indicating fit is complete)
            if self.lambda_param is not None:
                if self.transform_min is None or self.transform_max is None:
                    warnings.warn(
                        f"{self.method} transformation without transform_min/max may cause "
                        f"overflow during inverse transform. Consider retraining preprocessor.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Validate original_range
        if self.original_range is not None:
            if len(self.original_range) != 2:
                raise ValueError("original_range must be a tuple of (min, max)")
            if self.original_range[0] >= self.original_range[1]:
                raise ValueError(f"Invalid range: min >= max ({self.original_range})")

        # Validate bias correction if present
        if self._log_residual_variance is not None:
            if not isinstance(self._log_residual_variance, int | float | np.number):
                raise ValueError(
                    f"_log_residual_variance must be numeric, "
                    f"got {type(self._log_residual_variance)}"
                )
            if not np.isfinite(self._log_residual_variance):
                raise ValueError(
                    f"_log_residual_variance must be finite, " f"got {self._log_residual_variance}"
                )
            # NOTE: Negative values ARE valid.
            # For Yeo-Johnson:  var = 2 * log(median_ratio).
            # When the model over-predicts, median_ratio < 1.0  →  var < 0
            # →  exp(var/2) < 1.0  →  a legitimate DOWNWARD correction.
            # Zero is the only semantically invalid value (uninitialized sentinel).
            # Guard with _is_deserialized so a freshly-constructed object with a
            # zero default can still be caught, but a loaded artifact is not
            # re-validated (mirrors the lambda_param guard above).
            if self._log_residual_variance == 0.0 and not self._is_deserialized:
                warnings.warn(
                    "_log_residual_variance == 0.0 may indicate an uninitialized bias "
                    "correction value. Verify this is intentional (zero → no adjustment).",
                    UserWarning,
                    stacklevel=2,
                )
