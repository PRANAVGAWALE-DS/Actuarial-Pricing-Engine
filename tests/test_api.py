"""
Unit tests for api/schemas.py and api/routes.py (MetricsCollector, auth, helpers).

No model loading occurs in this file — all tests are pure schema validation,
in-memory state machine tests, and dependency-function unit tests.

Coverage:
  PredictRequest:
    - valid payload passes
    - sex / smoker / region are normalised to lowercase
    - invalid values raise ValidationError
    - age boundary conditions: ge=18, le=120
    - bmi boundary conditions: ge=10.0, le=100.0
    - children boundary conditions: ge=0, le=20

  BatchPredictRequest:
    - empty records list fails (min_length=1)
    - valid single-record list passes
    - large batch (10_000 records) accepted (no upper-bound constraint)

  PredictResponse / BatchPredictResponse:
    - optional CI fields (incl. ci_width_pct) default to None      [FIX F-11]
    - ci_width_pct present when CI fields are supplied              [FIX F-11]
    - ci_available defaults to False in batch response
    - ci_unavailable_reason accepted on BatchPredictResponse        [FIX F-10]

  _hash_input:
    - returns exactly 12 hex characters
    - deterministic: same input → same hash
    - collision-resistant: different inputs → different hashes

  _API_VERSION:
    - is a non-empty string                                         [FIX M-01]

  verify_api_key (FIX F-09 / U-09):
    - no API_KEY + STRICT_AUTH=false → passes (dev mode)
    - no API_KEY + STRICT_AUTH=true  → raises HTTP 503
    - API_KEY set, missing header    → raises HTTP 401, "Missing Authorization header"
    - API_KEY set, wrong token       → raises HTTP 401, "Invalid API key."
    - API_KEY set, correct token     → passes

  verify_metrics_token (FIX U-09):
    - no METRICS_TOKEN               → passes
    - METRICS_TOKEN set, missing header → raises HTTP 401
    - METRICS_TOKEN set, wrong token    → raises HTTP 401
    - METRICS_TOKEN set, correct token  → passes

  MetricsCollector:
    - _startup_time is float('inf') at construction
    - mark_ready() sets startup_time
    - uptime_seconds = 0 before mark_ready()
    - uptime_seconds > 0 after mark_ready()
    - record_prediction(success=True) increments prediction_count
    - record_prediction(success=False) increments both count and error_count
    - record_rejection() increments rejected_count
    - snapshot() error_rate_pct calculation
    - snapshot()["successful"] = total − errors
    - snapshot() latency percentiles (p50 <= p95 <= p99)
    - snapshot() thread-safety: concurrent writes don't corrupt counts
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

# Allow `python test_api.py` from `Pipeline/tests` to resolve local packages.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _path in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import warnings as _warnings_guard  # noqa: E402

from api.schemas import (  # noqa: E402
    VALID_REGIONS,
    VALID_SEX,
    VALID_SMOKER,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

with _warnings_guard.catch_warnings():
    _warnings_guard.simplefilter("ignore", RuntimeWarning)
    import api.routes as _routes_module
    from api.routes import (
        _API_VERSION,
        MetricsCollector,
        _hash_input,
        verify_api_key,
        verify_metrics_token,
    )


# ===========================================================================
# Helpers
# ===========================================================================


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "age": 35,
        "sex": "male",
        "bmi": 27.5,
        "children": 2,
        "smoker": "no",
        "region": "northeast",
    }
    base.update(overrides)
    return base


# ===========================================================================
# PredictRequest — happy path
# ===========================================================================


@pytest.mark.unit
class TestPredictRequestHappyPath:
    def test_valid_payload_constructs(self) -> None:
        req = PredictRequest(**_valid_payload())
        assert req.age == 35
        assert req.bmi == 27.5

    def test_sex_lowercased(self) -> None:
        req = PredictRequest(**_valid_payload(sex="MALE"))
        assert req.sex == "male"

    def test_smoker_lowercased(self) -> None:
        req = PredictRequest(**_valid_payload(smoker="YES"))
        assert req.smoker == "yes"

    def test_region_lowercased(self) -> None:
        req = PredictRequest(**_valid_payload(region="NORTHEAST"))
        assert req.region == "northeast"

    def test_sex_strips_whitespace(self) -> None:
        req = PredictRequest(**_valid_payload(sex="  female  "))
        assert req.sex == "female"

    def test_all_valid_sex_values(self) -> None:
        for s in VALID_SEX:
            PredictRequest(**_valid_payload(sex=s))

    def test_all_valid_smoker_values(self) -> None:
        for s in VALID_SMOKER:
            PredictRequest(**_valid_payload(smoker=s))

    def test_all_valid_regions(self) -> None:
        for r in VALID_REGIONS:
            PredictRequest(**_valid_payload(region=r))


# ===========================================================================
# PredictRequest — age bounds
# ===========================================================================


@pytest.mark.unit
class TestPredictRequestAgeBounds:
    def test_age_18_passes(self) -> None:
        req = PredictRequest(**_valid_payload(age=18))
        assert req.age == 18

    def test_age_17_fails(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(age=17))

    def test_age_0_fails(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(age=0))

    def test_age_120_passes(self) -> None:
        req = PredictRequest(**_valid_payload(age=120))
        assert req.age == 120

    def test_age_121_fails(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(age=121))


# ===========================================================================
# PredictRequest — BMI bounds
# ===========================================================================


@pytest.mark.unit
class TestPredictRequestBmiBounds:
    def test_bmi_10_passes(self) -> None:
        req = PredictRequest(**_valid_payload(bmi=10.0))
        assert req.bmi == 10.0

    def test_bmi_below_10_fails(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(bmi=9.9))

    def test_bmi_100_passes(self) -> None:
        req = PredictRequest(**_valid_payload(bmi=100.0))
        assert req.bmi == 100.0

    def test_bmi_above_100_fails(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(bmi=100.1))

    def test_bmi_60_passes(self) -> None:
        """Ensure old le=60 restriction is NOT present; values up to 100 allowed."""
        req = PredictRequest(**_valid_payload(bmi=60.0))
        assert req.bmi == 60.0

    def test_bmi_80_passes(self) -> None:
        req = PredictRequest(**_valid_payload(bmi=80.0))
        assert req.bmi == 80.0


# ===========================================================================
# PredictRequest — children bounds
# ===========================================================================


@pytest.mark.unit
class TestPredictRequestChildrenBounds:
    def test_children_zero_passes(self) -> None:
        req = PredictRequest(**_valid_payload(children=0))
        assert req.children == 0

    def test_children_negative_fails(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(children=-1))

    def test_children_20_passes(self) -> None:
        req = PredictRequest(**_valid_payload(children=20))
        assert req.children == 20

    def test_children_21_fails(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(children=21))


# ===========================================================================
# PredictRequest — invalid categorical values
# ===========================================================================


@pytest.mark.unit
class TestPredictRequestInvalidCategoricals:
    def test_invalid_sex_raises(self) -> None:
        with pytest.raises(ValidationError, match="sex"):
            PredictRequest(**_valid_payload(sex="unknown"))

    def test_invalid_smoker_raises(self) -> None:
        with pytest.raises(ValidationError, match="smoker"):
            PredictRequest(**_valid_payload(smoker="maybe"))

    def test_invalid_region_raises(self) -> None:
        with pytest.raises(ValidationError, match="region"):
            PredictRequest(**_valid_payload(region="midatlantic"))

    def test_empty_sex_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(sex=""))

    def test_empty_region_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(**_valid_payload(region=""))


# ===========================================================================
# BatchPredictRequest
# ===========================================================================


@pytest.mark.unit
class TestBatchPredictRequest:
    def test_valid_single_record(self) -> None:
        req = BatchPredictRequest(records=[PredictRequest(**_valid_payload())])
        assert len(req.records) == 1

    def test_empty_list_fails(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictRequest(records=[])

    def test_multiple_records_pass(self) -> None:
        records = [PredictRequest(**_valid_payload(age=20 + i)) for i in range(5)]
        req = BatchPredictRequest(records=records)
        assert len(req.records) == 5

    def test_large_batch_accepted(self) -> None:
        """10_000 records should pass; no upper-bound constraint is enforced."""
        records_10k = [PredictRequest(**_valid_payload()) for _ in range(10_000)]
        req = BatchPredictRequest(records=records_10k)
        assert len(req.records) == 10_000


# ===========================================================================
# PredictResponse / BatchPredictResponse schema defaults
# ===========================================================================


@pytest.mark.unit
class TestResponseSchemas:
    def test_predict_response_ci_fields_default_none(self) -> None:
        r = PredictResponse(prediction=12345.0, model_used="xgboost")
        assert r.lower_bound is None
        assert r.upper_bound is None
        assert r.ci_confidence_level is None
        assert r.ci_method is None
        assert r.prediction_id is None
        # FIX F-11: ci_width_pct must also default to None
        assert r.ci_width_pct is None

    def test_predict_response_with_ci(self) -> None:
        r = PredictResponse(
            prediction=12345.0,
            model_used="xgboost",
            lower_bound=9000.0,
            upper_bound=16000.0,
            ci_confidence_level=0.90,
            ci_method="heteroscedastic_conformal",
        )
        assert r.lower_bound == 9000.0
        assert r.ci_method == "heteroscedastic_conformal"

    def test_predict_response_ci_width_pct_accepted(self) -> None:
        """FIX F-11: ci_width_pct is a new optional float field on PredictResponse."""
        r = PredictResponse(
            prediction=12345.0,
            model_used="xgboost",
            lower_bound=9000.0,
            upper_bound=16000.0,
            ci_confidence_level=0.90,
            ci_method="heteroscedastic_conformal",
            ci_width_pct=56.78,
        )
        assert r.ci_width_pct == pytest.approx(56.78)

    def test_batch_predict_response_ci_available_defaults_false(self) -> None:
        resp = BatchPredictResponse(
            results=[],
            total=0,
            successful=0,
            failed=0,
            model_used="xgboost",
        )
        assert resp.ci_available is False

    def test_batch_predict_response_ci_unavailable_reason_accepted(self) -> None:
        """FIX F-10: ci_unavailable_reason is a new optional string field."""
        reason = (
            "Batch endpoint does not compute confidence intervals — "
            "use POST /api/v1/predict for single-record CI."
        )
        resp = BatchPredictResponse(
            results=[],
            total=0,
            successful=0,
            failed=0,
            model_used="xgboost",
            ci_unavailable_reason=reason,
        )
        assert resp.ci_unavailable_reason == reason

    def test_batch_predict_response_ci_unavailable_reason_defaults_none(self) -> None:
        resp = BatchPredictResponse(
            results=[],
            total=0,
            successful=0,
            failed=0,
            model_used="xgboost",
        )
        # Field is optional — must not be required
        assert resp.ci_unavailable_reason is None


# ===========================================================================
# MetricsCollector
# ===========================================================================


@pytest.mark.unit
class TestMetricsCollector:
    def test_startup_time_is_inf_at_construction(self, fresh_metrics: MetricsCollector) -> None:
        assert fresh_metrics._startup_time == float("inf")

    def test_mark_ready_sets_finite_startup_time(self, fresh_metrics: MetricsCollector) -> None:
        fresh_metrics.mark_ready()
        assert fresh_metrics._startup_time != float("inf")
        assert fresh_metrics._startup_time > 0

    def test_uptime_zero_before_mark_ready(self, fresh_metrics: MetricsCollector) -> None:
        snap = fresh_metrics.snapshot()
        assert snap["uptime_seconds"] == 0.0

    def test_uptime_positive_after_mark_ready(self, fresh_metrics: MetricsCollector) -> None:
        fresh_metrics.mark_ready()
        time.sleep(0.11)
        snap = fresh_metrics.snapshot()
        assert snap["uptime_seconds"] > 0.0

    def test_record_prediction_success_increments_count(
        self, fresh_metrics: MetricsCollector
    ) -> None:
        fresh_metrics.record_prediction(elapsed_s=0.1, success=True)
        snap = fresh_metrics.snapshot()
        assert snap["total"] == 1
        assert snap["errors"] == 0

    def test_record_prediction_failure_increments_error_count(
        self, fresh_metrics: MetricsCollector
    ) -> None:
        fresh_metrics.record_prediction(elapsed_s=0.2, success=False)
        snap = fresh_metrics.snapshot()
        assert snap["total"] == 1
        assert snap["errors"] == 1

    def test_error_rate_pct_correct(self, fresh_metrics: MetricsCollector) -> None:
        fresh_metrics.record_prediction(0.1, success=True)
        fresh_metrics.record_prediction(0.1, success=True)
        fresh_metrics.record_prediction(0.1, success=False)
        # 1 error out of 3 total = 33.33%
        snap = fresh_metrics.snapshot()
        assert abs(snap["error_rate_pct"] - 100 / 3) < 0.1

    def test_record_rejection_increments_rejected_count(
        self, fresh_metrics: MetricsCollector
    ) -> None:
        fresh_metrics.record_rejection()
        fresh_metrics.record_rejection()
        snap = fresh_metrics.snapshot()
        assert snap["rejected"] == 2

    def test_snapshot_returns_zero_error_rate_when_no_predictions(
        self, fresh_metrics: MetricsCollector
    ) -> None:
        snap = fresh_metrics.snapshot()
        assert snap["error_rate_pct"] == 0.0

    def test_latency_samples_recorded(self, fresh_metrics: MetricsCollector) -> None:
        for i in range(5):
            fresh_metrics.record_prediction(elapsed_s=(i + 1) * 0.1, success=True)
        snap = fresh_metrics.snapshot()
        assert snap["latency_samples"] == 5

    def test_latency_percentiles_ordered(self, fresh_metrics: MetricsCollector) -> None:
        """p50 <= p95 <= p99 must always hold."""
        for i in range(100):
            fresh_metrics.record_prediction(elapsed_s=i * 0.01, success=True)
        snap = fresh_metrics.snapshot()
        assert snap["p50_ms"] <= snap["p95_ms"] <= snap["p99_ms"]

    def test_mean_ms_is_positive(self, fresh_metrics: MetricsCollector) -> None:
        fresh_metrics.record_prediction(elapsed_s=0.12, success=True)
        snap = fresh_metrics.snapshot()
        assert snap["mean_ms"] > 0

    def test_concurrent_record_predictions_no_data_corruption(
        self, fresh_metrics: MetricsCollector
    ) -> None:
        """
        100 threads each recording 10 predictions → total must be exactly 1000.
        Verifies that the threading.Lock() protects the counter.
        """
        n_threads = 100
        n_per_thread = 10

        def worker():
            for _ in range(n_per_thread):
                fresh_metrics.record_prediction(elapsed_s=0.001, success=True)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = fresh_metrics.snapshot()
        assert snap["total"] == n_threads * n_per_thread

    def test_rolling_window_respects_max_samples(self, fresh_metrics: MetricsCollector) -> None:
        """After MAX_LATENCY_SAMPLES predictions, the deque wraps correctly."""
        limit = MetricsCollector._MAX_LATENCY_SAMPLES
        for _i in range(limit + 50):
            fresh_metrics.record_prediction(elapsed_s=0.001, success=True)
        snap = fresh_metrics.snapshot()
        # latency_samples should be capped at MAX_LATENCY_SAMPLES
        assert snap["latency_samples"] == limit

    def test_snapshot_successful_equals_total_minus_errors(
        self, fresh_metrics: MetricsCollector
    ) -> None:
        """snapshot()['successful'] must equal total − errors at all times."""
        fresh_metrics.record_prediction(0.1, success=True)
        fresh_metrics.record_prediction(0.1, success=True)
        fresh_metrics.record_prediction(0.1, success=False)
        snap = fresh_metrics.snapshot()
        assert snap["successful"] == snap["total"] - snap["errors"]
        assert snap["successful"] == 2


# ===========================================================================
# _hash_input
# ===========================================================================


@pytest.mark.unit
class TestHashInput:
    def test_returns_12_hex_chars(self) -> None:
        result = _hash_input({"age": 30, "sex": "male"})
        assert isinstance(result, str)
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self) -> None:
        data = {"age": 35, "bmi": 27.5, "smoker": "no"}
        assert _hash_input(data) == _hash_input(data)

    def test_different_inputs_produce_different_hashes(self) -> None:
        h1 = _hash_input({"age": 35})
        h2 = _hash_input({"age": 36})
        assert h1 != h2

    def test_key_order_invariant(self) -> None:
        """json.dumps with sort_keys=True means field order doesn't matter."""
        h1 = _hash_input({"age": 35, "bmi": 27.5})
        h2 = _hash_input({"bmi": 27.5, "age": 35})
        assert h1 == h2


# ===========================================================================
# _API_VERSION  (FIX M-01)
# ===========================================================================


@pytest.mark.unit
class TestApiVersion:
    def test_is_non_empty_string(self) -> None:
        assert isinstance(_API_VERSION, str)
        assert len(_API_VERSION) > 0

    def test_matches_expected_version(self) -> None:
        assert _API_VERSION == "1.2.0"

    def test_has_semver_shape(self) -> None:
        """Must be parseable as MAJOR.MINOR.PATCH."""
        parts = _API_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ===========================================================================
# verify_api_key  (FIX F-09 / U-09)
# ===========================================================================

# Minimal stand-in for fastapi.security.HTTPAuthorizationCredentials
_MockCreds = lambda token: SimpleNamespace(credentials=token)  # noqa: E731


@pytest.mark.unit
class TestVerifyApiKey:
    def test_no_api_key_strict_false_allows_all(self) -> None:
        """Dev mode: API_KEY unset + STRICT_AUTH=false → no exception."""
        with (
            patch.object(_routes_module, "_API_KEY", None),
            patch.object(_routes_module, "_STRICT_AUTH", False),
        ):
            verify_api_key(credentials=None)  # must not raise

    def test_no_api_key_strict_true_raises_503(self) -> None:
        """FIX F-09: STRICT_AUTH=true + no API_KEY → 503 (fail-secure)."""
        with (
            patch.object(_routes_module, "_API_KEY", None),
            patch.object(_routes_module, "_STRICT_AUTH", True),
        ):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(credentials=None)
        assert exc_info.value.status_code == 503

    def test_missing_header_raises_401_with_hint(self) -> None:
        """FIX U-09: API_KEY set but no Authorization header → 401, hints at header name."""
        with (
            patch.object(_routes_module, "_API_KEY", "secret-token"),
            patch.object(_routes_module, "_STRICT_AUTH", False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(credentials=None)
        exc = exc_info.value
        assert exc.status_code == 401
        assert "Authorization" in exc.detail

    def test_wrong_token_raises_401(self) -> None:
        """FIX U-09: API_KEY set, wrong bearer token → 401."""
        with (
            patch.object(_routes_module, "_API_KEY", "secret-token"),
            patch.object(_routes_module, "_STRICT_AUTH", False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(credentials=_MockCreds("wrong-token"))
        assert exc_info.value.status_code == 401
        assert "Invalid" in exc_info.value.detail

    def test_correct_token_passes(self) -> None:
        """Valid bearer token → no exception."""
        with (
            patch.object(_routes_module, "_API_KEY", "secret-token"),
            patch.object(_routes_module, "_STRICT_AUTH", False),
        ):
            verify_api_key(credentials=_MockCreds("secret-token"))  # must not raise

    def test_missing_header_and_wrong_token_produce_different_details(self) -> None:
        """Missing header vs wrong token must produce distinct detail strings (log differentiation)."""
        with (
            patch.object(_routes_module, "_API_KEY", "secret-token"),
            patch.object(_routes_module, "_STRICT_AUTH", False),
        ):
            with pytest.raises(HTTPException) as missing_exc:
                verify_api_key(credentials=None)
            with pytest.raises(HTTPException) as wrong_exc:
                verify_api_key(credentials=_MockCreds("bad"))
        assert missing_exc.value.detail != wrong_exc.value.detail


# ===========================================================================
# verify_metrics_token  (FIX U-09)
# ===========================================================================


@pytest.mark.unit
class TestVerifyMetricsToken:
    def test_no_metrics_token_allows_all(self) -> None:
        """METRICS_TOKEN unset → open access (dev/local)."""
        with patch.object(_routes_module, "_METRICS_TOKEN", None):
            verify_metrics_token(credentials=None)  # must not raise

    def test_missing_header_raises_401(self) -> None:
        with patch.object(_routes_module, "_METRICS_TOKEN", "metrics-secret"):
            with pytest.raises(HTTPException) as exc_info:
                verify_metrics_token(credentials=None)
        assert exc_info.value.status_code == 401

    def test_wrong_token_raises_401(self) -> None:
        with patch.object(_routes_module, "_METRICS_TOKEN", "metrics-secret"):
            with pytest.raises(HTTPException) as exc_info:
                verify_metrics_token(credentials=_MockCreds("wrong"))
        assert exc_info.value.status_code == 401

    def test_correct_token_passes(self) -> None:
        with patch.object(_routes_module, "_METRICS_TOKEN", "metrics-secret"):
            verify_metrics_token(credentials=_MockCreds("metrics-secret"))  # must not raise

    def test_missing_vs_wrong_detail_differ(self) -> None:
        """FIX U-09: differentiated log/detail strings for missing vs invalid."""
        with patch.object(_routes_module, "_METRICS_TOKEN", "metrics-secret"):
            with pytest.raises(HTTPException) as missing_exc:
                verify_metrics_token(credentials=None)
            with pytest.raises(HTTPException) as wrong_exc:
                verify_metrics_token(credentials=_MockCreds("bad"))
        assert missing_exc.value.detail != wrong_exc.value.detail


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                "-o",
                "addopts=",
                "-p",
                "no:cov",
                str(Path(__file__).resolve()),
            ]
        )
    )
