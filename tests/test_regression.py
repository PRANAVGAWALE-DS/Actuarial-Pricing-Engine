"""
Regression Pin Registry — insurance_ml pipeline

PURPOSE
-------
This file contains ONE test per known.  Each test:
  1. Documents WHAT the bug was (old behaviour).
  2. Asserts the CURRENT correct behaviour.
  3. Is marked with the bug ID from our audit history for traceability.

If any of these tests start failing, a previously-fixed bug has been
re-introduced.  Fix the source; never weaken or delete these tests.

Bug inventory (chronological):
  BUG-R2-SENTINEL        evaluate.py: n=1 segment r2 returned 0.0 (was sklearn default)
  BUG-N3                 evaluate.py: load_business_config_from_yaml mutated shared config
  BUG-CHURN-SENSITIVITY  evaluate.py: BusinessConfig.churn_sensitivity default was 0.01
  BUG-STARTUP-TIME       routes.py: MetricsCollector._startup_time set to time.time() at
                          construction, making uptime_seconds count from module import
  BUG-BC-TO-DICT         features.py: BiasCorrection.to_dict() used truthiness on numeric
                          fields, dropping small-but-valid values
  BUG-BC-3TIER-PARTIAL   features.py: BiasCorrection accepted partial 3-tier config silently
  BUG-BC-ZERO-VAR        features.py: BiasCorrection accepted var_low=0 (uninitialised sentinel)
  BUG-SSOT-MODEL-CV      config.py: model.cv_folds not caught by validate_single_source_of_truth
  BUG-API-AGE-FLOOR      schemas.py: PredictRequest allowed age=0 (no ge=18 constraint)
  BUG-API-BMI-CEILING    schemas.py: PredictRequest allowed bmi=60 (le=60, should be le=100)
  BUG-DRIFT-MISSING-FEAT monitoring.py: missing features raised KeyError instead of being
                          recorded in report.missing_features
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# BUG-R2-SENTINEL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_r2_sentinel_nan_not_zero_for_single_sample_segment() -> None:
    """
    Old behaviour: r2_score(y_true, y_pred) for n=1 returns 0.0 by default.
    Pipeline was returning that 0.0 directly, misleading callers into thinking
    the model had zero fit on the segment (vs. undefined/unknown fit).

    Fixed: guard `if len(seg_true) >= 2 else float('nan')` added in
    evaluate.py calculate_segment_metrics() and calculate_gate_aligned_*().
    """
    from insurance_ml.evaluate import BusinessMetricsCalculator

    calc = BusinessMetricsCalculator()
    y_true = np.array([100.0, 5000.0, 15000.0, 35000.0])
    y_pred = np.array([110.0, 5100.0, 15100.0, 35100.0])

    # Force exactly one sample in low_risk bin
    results = calc.calculate_segment_metrics(
        y_true=y_true,
        y_pred=y_pred,
        use_business_thresholds=True,
        config={
            "evaluation": {
                "segment_thresholds": {
                    "low_value": 200.0,
                    "standard": 15000.0,
                    "high_value": 30000.0,
                }
            }
        },
    )
    low = results["low_risk"]
    assert low["n_samples"] == 1
    assert math.isnan(low["r2"]), f"BUG-R2-SENTINEL: expected NaN, got {low['r2']}"


# ---------------------------------------------------------------------------
# BUG-N3
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_n3_load_business_config_does_not_mutate_config(minimal_config: dict, monkeypatch) -> None:
    """
    Old behaviour: load_business_config_from_yaml() obtained a reference to
    hybrid_config["business_config"] and wrote "low_value_threshold" into it,
    permanently modifying the shared dict for subsequent callers.

    Fixed in evaluate.py v7.4.0 via: business_dict = dict(hybrid_config.get(...))
    i.e. shallow copy before mutation.
    """
    import copy

    from insurance_ml import config as cfg_module
    from insurance_ml.evaluate import load_business_config_from_yaml

    original_biz = copy.deepcopy(
        minimal_config.get("hybrid_predictor", {}).get("business_config", {})
    )

    monkeypatch.setattr(cfg_module, "load_config", lambda: copy.deepcopy(minimal_config))

    load_business_config_from_yaml()
    load_business_config_from_yaml()  # second call must not see a mutated config

    current_biz = minimal_config.get("hybrid_predictor", {}).get("business_config", {})
    assert current_biz == original_biz, (
        "BUG-N3: load_business_config_from_yaml mutated the shared hybrid_predictor."
        "business_config dict."
    )


# ---------------------------------------------------------------------------
# BUG-CHURN-SENSITIVITY
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_churn_sensitivity_default_is_one() -> None:
    """
    Old behaviour: BusinessConfig.churn_sensitivity defaulted to 0.01, which
    effectively disabled churn (needed ~5000% overpricing to hit 50% churn).
    Fixed to 1.0 in evaluate.py v7.4.0.
    """
    from insurance_ml.evaluate import BusinessConfig

    cfg = BusinessConfig()
    assert (
        cfg.churn_sensitivity == 1.0
    ), f"BUG-CHURN-SENSITIVITY: expected 1.0, got {cfg.churn_sensitivity}"


@pytest.mark.unit
def test_churn_sensitivity_default_in_from_config_dict_is_one() -> None:
    from insurance_ml.evaluate import BusinessConfig

    cfg = BusinessConfig.from_config_dict({})
    assert cfg.churn_sensitivity == 1.0


# ---------------------------------------------------------------------------
# BUG-STARTUP-TIME
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_collector_startup_time_is_sentinel_at_init() -> None:
    """
    Old behaviour: MetricsCollector set _startup_time = time.time() in __init__,
    so uptime_seconds started counting from module import — not from when the API
    was actually ready to serve requests.

    Fixed: _startup_time = float('inf') at construction; main.py calls
    mark_ready() after all startup work completes.
    """
    from api.routes import MetricsCollector

    mc = MetricsCollector()
    assert mc._startup_time == float("inf"), (
        f"BUG-STARTUP-TIME: _startup_time should be inf at construction, " f"got {mc._startup_time}"
    )


@pytest.mark.unit
def test_metrics_collector_uptime_is_zero_before_mark_ready() -> None:
    """uptime_seconds must report 0.0 until mark_ready() is called."""
    from api.routes import MetricsCollector

    mc = MetricsCollector()
    snap = mc.snapshot()
    assert snap["uptime_seconds"] == 0.0, (
        f"BUG-STARTUP-TIME: uptime should be 0.0 before mark_ready(), "
        f"got {snap['uptime_seconds']}"
    )


# ---------------------------------------------------------------------------
# BUG-BC-TO-DICT
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bias_correction_to_dict_preserves_small_positive_values() -> None:
    """
    Old behaviour: to_dict() used Python truthiness on numeric fields.
    Small-but-nonzero values (e.g. threshold=0.01) would be dropped because
    `if self.var_mid:` evaluates to False for values close to 0.

    Fixed: to_dict() now uses `if not self.is_2tier:` (property check).
    """
    from insurance_ml.features import BiasCorrection

    bc = BiasCorrection(var_low=0.0001, var_high=0.0002, threshold=1.0)
    d = bc.to_dict()
    assert d["var_low"] == 0.0001, f"BUG-BC-TO-DICT: var_low dropped (got {d.get('var_low')})"
    assert d["var_high"] == 0.0002
    assert d["threshold"] == 1.0


@pytest.mark.unit
def test_bias_correction_to_dict_3tier_all_fields_preserved() -> None:
    from insurance_ml.features import BiasCorrection

    bc = BiasCorrection(
        var_low=0.04,
        var_high=0.09,
        threshold=0.0,
        var_mid=0.06,
        threshold_low=10000.0,
        threshold_high=20000.0,
    )
    d = bc.to_dict()
    assert "var_mid" in d
    assert d["var_mid"] == 0.06
    assert "threshold_low" in d
    assert "threshold_high" in d


# ---------------------------------------------------------------------------
# BUG-BC-3TIER-PARTIAL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partial_3tier_config_raises_value_error() -> None:
    """
    Old behaviour: BiasCorrection accepted var_mid without threshold_low /
    threshold_high, producing silent misrouting at inference.

    Fixed in __post_init__: all three 3-tier fields must be set together.
    """
    from insurance_ml.features import BiasCorrection

    with pytest.raises(ValueError, match="3-tier"):
        BiasCorrection(
            var_low=0.04,
            var_high=0.09,
            threshold=15000.0,
            var_mid=0.06,
            # threshold_low and threshold_high omitted
        )


# ---------------------------------------------------------------------------
# BUG-BC-ZERO-VAR
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bias_correction_rejects_zero_var_low() -> None:
    """
    var_low=0 is the sentinel for "not yet computed" — it produces a no-op
    correction factor of exp(0/2)=1.0 but silently hides that bias correction
    was never calculated.  __post_init__ must reject it explicitly.
    """
    from insurance_ml.features import BiasCorrection

    with pytest.raises(ValueError, match="var_low"):
        BiasCorrection(var_low=0.0, var_high=0.09, threshold=15000.0)


# ---------------------------------------------------------------------------
# BUG-SSOT-MODEL-CV
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_ssot_raises_when_model_has_cv_folds(
    mutable_config: dict,
) -> None:
    """
    validate_single_source_of_truth must flag model.cv_folds as a violation
    of the single-source-of-truth principle (n_folds lives in cross_validation).
    """
    from insurance_ml.config import validate_single_source_of_truth

    mutable_config["model"]["cv_folds"] = 5
    with pytest.raises(ValueError, match="model.cv_folds"):
        validate_single_source_of_truth(mutable_config)


# ---------------------------------------------------------------------------
# BUG-API-AGE-FLOOR
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_predict_request_rejects_age_below_18() -> None:
    """
    Old behaviour: PredictRequest had no lower bound on age (or ge=0), allowing
    child policies to be submitted.  Fixed: ge=18.
    """
    from pydantic import ValidationError

    from api.schemas import PredictRequest

    with pytest.raises(ValidationError):
        PredictRequest(age=17, sex="male", bmi=25.0, children=0, smoker="no", region="northeast")


@pytest.mark.unit
def test_predict_request_accepts_age_18() -> None:
    """Boundary: age=18 must pass validation (ge=18)."""
    from api.schemas import PredictRequest

    req = PredictRequest(age=18, sex="male", bmi=25.0, children=0, smoker="no", region="northeast")
    assert req.age == 18


# ---------------------------------------------------------------------------
# BUG-API-BMI-CEILING
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_predict_request_accepts_bmi_up_to_100() -> None:
    """
    Old behaviour: PredictRequest had le=60 which was less than the pipeline
    maximum (config.yaml bmi_max=100.0).  Fixed: le=100.
    """
    from api.schemas import PredictRequest

    req = PredictRequest(
        age=35, sex="female", bmi=100.0, children=0, smoker="no", region="northeast"
    )
    assert req.bmi == 100.0


@pytest.mark.unit
def test_predict_request_rejects_bmi_above_100() -> None:
    from pydantic import ValidationError

    from api.schemas import PredictRequest

    with pytest.raises(ValidationError):
        PredictRequest(age=35, sex="female", bmi=100.1, children=0, smoker="no", region="northeast")


# ---------------------------------------------------------------------------
# BUG-DRIFT-MISSING-FEAT
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_drift_detect_records_missing_feature_not_raises(
    tmp_path: Path,
) -> None:
    """
    Old behaviour: detect_drift raised KeyError when a feature present in the
    baseline was absent from X_new.

    Fixed: feature is appended to report.missing_features and skipped.
    """
    from insurance_ml.monitoring import DriftMonitor

    n = 50
    X_train = pd.DataFrame(
        {
            "age": np.random.default_rng(0).integers(18, 65, n).astype(float),
            "bmi": np.random.default_rng(1).uniform(18.0, 40.0, n),
        }
    )
    y = pd.Series(np.ones(n) * 5000.0)
    out = tmp_path / "b.json"
    DriftMonitor.create_baseline(X_train=X_train, y_train=y, output_path=str(out), overwrite=True)

    # New batch is missing 'age'
    X_new = X_train.drop(columns=["age"])
    report = DriftMonitor.detect_drift(X_new=X_new, baseline_path=str(out))

    assert (
        "age" in report.missing_features
    ), "BUG-DRIFT-MISSING-FEAT: 'age' should be in missing_features, not raise KeyError"
