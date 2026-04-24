"""
API Schemas - Insurance Cost Predictor
Pydantic v2 request/response models aligned with predict.py v6.3.3

"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

VALID_SEX: frozenset[str] = frozenset({"male", "female"})
VALID_SMOKER: frozenset[str] = frozenset({"yes", "no"})
VALID_REGIONS: frozenset[str] = frozenset({"northeast", "northwest", "southeast", "southwest"})


# =============================================================================
# BATCH SIZE CONSTANT
# =============================================================================

# max batch size as a named constant rather than a bare literal.
# THIS VALUE MUST STAY IN SYNC WITH prediction.max_batch_size in config.yaml.
# If you raise the config limit, update this constant and restart the API.
# A mismatch causes the Pydantic gate (422) and the pipeline gate (ValueError)
# to enforce different limits — one will silently shadow the other.
_MAX_BATCH_SIZE: int = 50_000


class PredictRequest(BaseModel):
    # Bounds aligned with config.yaml features section.
    # age ge=18: policy-holder minimum (adults only at this endpoint).
    # age le=120, bmi le=100, children le=20: match config.yaml maximums exactly.
    age: int = Field(..., ge=18, le=120)
    sex: str
    bmi: float = Field(..., ge=10.0, le=100.0)
    children: int = Field(..., ge=0, le=20)
    smoker: str
    region: str

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        n = v.strip().lower()
        if n not in VALID_SEX:
            raise ValueError(f"sex must be one of {sorted(VALID_SEX)}, got '{v}'")
        return n

    @field_validator("smoker")
    @classmethod
    def validate_smoker(cls, v: str) -> str:
        n = v.strip().lower()
        if n not in VALID_SMOKER:
            raise ValueError(f"smoker must be one of {sorted(VALID_SMOKER)}, got '{v}'")
        return n

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        n = v.strip().lower()
        if n not in VALID_REGIONS:
            raise ValueError(f"region must be one of {sorted(VALID_REGIONS)}, got '{v}'")
        return n

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "sex": "male",
                "bmi": 27.5,
                "children": 2,
                "smoker": "no",
                "region": "northeast",
            }
        }
    }


class BatchPredictRequest(BaseModel):
    # max_length references _MAX_BATCH_SIZE which MUST stay in sync with
    # prediction.max_batch_size in config.yaml.  See constant definition above.
    records: list[PredictRequest] = Field(..., min_length=1, max_length=_MAX_BATCH_SIZE)


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================


class PredictResponse(BaseModel):
    """Single prediction response.

    CI fields (lower_bound … ci_method) are all Optional and default to None.
    Existing callers that only read 'prediction' and 'model_used' are unaffected.
    """

    prediction: float
    model_used: str
    prediction_id: str | None = None
    lower_bound: float | None = Field(None, description="Lower CI bound (USD)")
    upper_bound: float | None = Field(None, description="Upper CI bound (USD)")
    ci_confidence_level: float | None = Field(None, description="e.g. 0.90")
    ci_method: str | None = Field(
        None,
        description="heteroscedastic_conformal | split_conformal | parametric_gaussian_fallback",
    )
    ci_width_pct: float | None = Field(
        None,
        description="CI width as % of point estimate: (upper-lower)/prediction * 100",
    )
    # machine-readable flag so callers don't have to implement their
    # own threshold check on ci_width_pct.  True when ci_width_pct > 80 — at
    # that width the interval spans more than the point estimate itself and
    # provides no actionable pricing signal.  Threshold is conservative; the
    # problematic transition-zone case logged 118.8%.
    ci_unreliable: bool | None = Field(
        None,
        description=(
            "True when ci_width_pct > 80%% — interval too wide for rate-setting. "
            "Downstream systems should route to manual review rather than using the CI."
        ),
    )
    # expose the actuarial/ML ratio so callers can handle high-
    # disagreement cases (e.g. route to manual review when ratio > 1.15).
    # Value is the median(actuarial / ml_calibrated) from HybridPredictor.
    actuarial_ml_ratio: float | None = Field(
        None,
        description=(
            "Median actuarial/ML ratio from HybridPredictor. "
            "> 1.15 → actuarial conservative; < 0.70 → actuarial aggressive."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 12_540.87,
                "model_used": "hybrid_xgboost_median_v6.3.1",
                "prediction_id": "a3f2c1d4-9e2b-4f8c-b1a0-7c5d3e8f9a2b",
                "lower_bound": 9_214.32,
                "upper_bound": 16_803.45,
                "ci_confidence_level": 0.90,
                "ci_method": "heteroscedastic_conformal",
            }
        }
    }


class BatchPredictRecord(BaseModel):
    index: int
    prediction: float | None = None
    model_used: str | None = None
    error: str | None = None
    status: str


class BatchPredictResponse(BaseModel):
    results: list[BatchPredictRecord]
    total: int
    successful: int
    failed: int
    model_used: str
    ci_available: bool = Field(
        False,
        description=(
            "Always False for batch predictions. Confidence intervals are omitted "
            "from batch responses because predict_with_intervals() is O(N)×120ms. "
            "Use POST /api/v1/predict for per-record CI."
        ),
    )
    ci_unavailable_reason: str | None = Field(
        None,
        description="Human-readable explanation of why CI is not available for this response.",
    )


class HealthResponse(BaseModel):
    status: str
    model_name: str
    pipeline_version: str | None = None
    hybrid_version: str | None = None
    detail: str | None = None
    valid_regions: list[str] | None = None
    valid_sex: list[str] | None = None
    valid_smoker: list[str] | None = None


class TargetTransformationInfo(BaseModel):
    method: str
    bias_correction: bool
    bias_correction_variance: float | None = None
    recommended_metrics: list[str] = []
    deprecated_metrics: list[str] = []


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    pipeline_version: str
    target_transformation: TargetTransformationInfo
    feature_count: Any | None = None
    has_feature_importances: bool = False
    has_coefficients: bool = False
    pipeline_state: str | None = None


# =============================================================================
# METRICS RESPONSE  (#6)
# =============================================================================


class PredictionMetrics(BaseModel):
    total: int
    successful: int
    errors: int
    error_rate_pct: float = Field(..., description="Errors as % of total")
    rejected_overload: int = Field(..., description="429s from semaphore cap")
    concurrent_limit: int = Field(..., description="Max concurrent predictions")


class LatencyMetrics(BaseModel):
    """Rolling percentiles over the last 1 000 predictions (ms)."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples: int


class MetricsResponse(BaseModel):
    """
    Operational metrics returned by GET /api/v1/metrics.

    Streamlit admin panel reads these fields to surface alerts:
      predictions.error_rate_pct > 5   → high error rate warning
      latency_ms.p99_ms > 20000        → latency degradation warning
                                       (nominal p99 ≈ 6-10 s at 4 s/prediction;
                                        threshold set at 5× nominal to detect
                                        true degradation, not steady-state load)
      predictions.rejected_overload > 0 → concurrency cap hit
    """

    uptime_seconds: float
    predictions: PredictionMetrics
    latency_ms: LatencyMetrics
    model_name: str
    hybrid_version: str | None = None
    pipeline_version: str | None = None
