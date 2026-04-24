"""
API Routes - Insurance Cost Predictor
Aligned with HybridPredictor v6.3.3 and PredictionPipeline v6.3.3

"""

from __future__ import annotations

import hashlib
import json
import logging
import math as _math
import os
import threading
import time
import uuid
from collections import deque

# Optional Prometheus integration — graceful degradation if not installed.
# Install: pip install prometheus-client
try:
    from fastapi.responses import Response as _FastAPIResponse
    from prometheus_client import (
        CONTENT_TYPE_LATEST as _PROM_CONTENT_TYPE,
    )
    from prometheus_client import (
        Counter as _PCounter,
    )
    from prometheus_client import (
        Histogram as _PHistogram,
    )
    from prometheus_client import (
        generate_latest as _prom_generate_latest,
    )

    _PROM_AVAILABLE = True
    _prom_prediction_counter = _PCounter(
        "insurance_predictions_total",
        "Total prediction requests",
        ["status"],  # "success" | "error"
    )
    _prom_prediction_latency = _PHistogram(
        "insurance_prediction_duration_seconds",
        "Prediction latency in seconds",
        buckets=[0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0],
    )
    _prom_rejected_counter = _PCounter(
        "insurance_predictions_rejected_total",
        "Prediction requests rejected by concurrency semaphore (429)",
    )
except ImportError:
    _PROM_AVAILABLE = False

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.schemas import (
    VALID_REGIONS,
    VALID_SEX,
    VALID_SMOKER,
    BatchPredictRecord,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    LatencyMetrics,
    MetricsResponse,
    ModelInfoResponse,
    PredictionMetrics,
    PredictRequest,
    PredictResponse,
    TargetTransformationInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── API version ───────────────────────────────────────────────────────────────
# Single source of truth for the API version exposed in / (root) and imported
# by main.py.  Previously root() used a hardcoded literal "1.2.0" that was
# decoupled from main._API_VERSION; a version bump required two manual edits.
# main.py now imports _API_VERSION from here instead of defining its own copy.
_API_VERSION: str = "1.2.0"

# ── Auth ──────────────────────────────────────────────────────────────────────
_API_KEY: str | None = os.getenv("API_KEY")
# Separate read-only token for /metrics scraping (Prometheus, Grafana).
# When unset, /metrics is open (acceptable for local/dev).  Set in production.
_METRICS_TOKEN: str | None = os.getenv("METRICS_TOKEN")
_security = HTTPBearer(auto_error=False)

# Strict auth mode — when STRICT_AUTH=true, an unset API_KEY blocks
# all requests rather than allowing them through (fail-secure over fail-open).
# Default false preserves backward-compatible dev behaviour.
# Set STRICT_AUTH=true in any internet-facing deployment.
_STRICT_AUTH: bool = os.getenv("STRICT_AUTH", "false").lower() in ("1", "true", "yes")
if not _API_KEY and not _STRICT_AUTH:
    # NOTE: not using logger here — logging may not be configured at import time.
    # main.py lifespan logs this during startup.
    import warnings as _warnings

    _warnings.warn(
        "API_KEY env var is not set — all prediction endpoints are publicly accessible. "
        "Set API_KEY and STRICT_AUTH=true for production deployments.",
        RuntimeWarning,
        stacklevel=2,
    )

# ── CI confidence level ───────────────────────────────────────────────────────
_CI_CONFIDENCE_LEVEL_RAW = os.getenv("CI_CONFIDENCE_LEVEL", "0.90")
try:
    _CI_CONFIDENCE_LEVEL = float(_CI_CONFIDENCE_LEVEL_RAW)
    if not (0.0 < _CI_CONFIDENCE_LEVEL < 1.0):
        raise ValueError("must be in (0, 1)")
except ValueError as _e:
    # Intentionally not logger.warning here — logging may not be configured yet
    # at import time. main.py logs this during lifespan startup.
    _CI_CONFIDENCE_LEVEL = 0.90

# ── #7: Concurrency semaphore ─────────────────────────────────────────────────
# acquire(blocking=False): reject immediately with 429 rather than queue.
# Queuing hides load; 429 signals the caller to back off.
# Default 10 is safe for workers=1: ~120ms × 10 = ~1.2s saturation, < 15s timeout.
# Lower to 4–5 if GPU OOM appears under load.
_MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT_PREDICTIONS", "10"))
_predict_semaphore = threading.Semaphore(_MAX_CONCURRENT)
# NOTE: do NOT call logger here — logging is not configured at import time.
# main.py lifespan logs max_concurrent at step [3/4].

# ── H-02: Batch per-record fallback row cap ───────────────────────────────────
# When the vectorised predict() path raises, predict_batch falls back to calling
# _predict_single_row() once per row.  Without a cap, a 10,000-row batch whose
# vectorised path fails holds the semaphore slot for up to ~11 hours and writes
# ≥ 10,000 INFO log lines.  Any batch larger than _FALLBACK_ROW_CAP receives a
# 503 instead of the per-record fallback.
# Override via BATCH_FALLBACK_ROW_CAP env var if operational needs differ.
_FALLBACK_ROW_CAP: int = int(os.getenv("BATCH_FALLBACK_ROW_CAP", "500"))


# =============================================================================
# #6 — METRICS COLLECTOR
# =============================================================================


class MetricsCollector:
    """
    Thread-safe rolling metrics for the prediction API.

    _startup_time is set to float('inf') at construction and reset to
    time.time() by main.py lifespan after startup completes. This ensures
    uptime_seconds counts from when the API is ready to serve, not from
    module import time.
    """

    _MAX_LATENCY_SAMPLES = 1_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Set to sentinel — reset by main.py after startup completes.
        self._startup_time: float = float("inf")
        self._prediction_count: int = 0
        self._error_count: int = 0
        self._rejected_count: int = 0
        self._latencies: deque[float] = deque(maxlen=self._MAX_LATENCY_SAMPLES)

    def mark_ready(self) -> None:
        """Called by main.py lifespan once startup is complete."""
        with self._lock:
            self._startup_time = time.time()

    def record_prediction(self, elapsed_s: float, success: bool) -> None:
        with self._lock:
            self._prediction_count += 1
            if not success:
                self._error_count += 1
            self._latencies.append(elapsed_s * 1000.0)

        # Emit Prometheus counters/histogram outside the lock (thread-safe internally)
        if _PROM_AVAILABLE:
            _prom_prediction_counter.labels(status="success" if success else "error").inc()
            _prom_prediction_latency.observe(elapsed_s)

    def record_rejection(self) -> None:
        with self._lock:
            self._rejected_count += 1

        if _PROM_AVAILABLE:
            _prom_rejected_counter.inc()

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            uptime = (now - self._startup_time) if self._startup_time != float("inf") else 0.0
            total = self._prediction_count
            errors = self._error_count
            rejected = self._rejected_count
            latencies = list(self._latencies)

        if latencies:
            arr = np.array(latencies)
            mean_ms = float(np.mean(arr))
            p50_ms = float(np.percentile(arr, 50))  # linear interpolation — no rounding bias
            p95_ms = float(np.percentile(arr, 95))  # correct for small n; was max for n<=20
            p99_ms = float(np.percentile(arr, 99))  # correct for small n; was max for n<=100
        else:
            mean_ms = p50_ms = p95_ms = p99_ms = 0.0

        return {
            "uptime_seconds": round(uptime, 1),
            "total": total,
            "successful": total - errors,
            "errors": errors,
            "error_rate_pct": round(errors / total * 100, 2) if total > 0 else 0.0,
            "rejected": rejected,
            "latency_samples": len(latencies),
            "mean_ms": round(mean_ms, 1),
            "p50_ms": round(p50_ms, 1),
            "p95_ms": round(p95_ms, 1),
            "p99_ms": round(p99_ms, 1),
        }


# Module-level singleton — main.py calls _metrics.mark_ready() at end of lifespan
_metrics = MetricsCollector()


# =============================================================================
# HELPERS
# =============================================================================


def _hash_input(data: dict) -> str:
    """12-char SHA-256 prefix for audit logging without storing PII."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def _compute_ci(
    pipeline,
    input_df: pd.DataFrame,
    prediction_id: str,
    input_hash: str,
) -> tuple[float | None, float | None, float | None, str | None, dict | None]:
    """
    Best-effort CI from predict_with_intervals(). All None on any failure.
    Failure never causes a 500 — point estimate is always returned.

    lower_bound key is always singular — the dead "lower_bounds"
    plural fallback never existed and has been removed.

    Returns the full ci_result dict as 5th element so predict_single
    can pass it as _precomputed_ml_result to HybridPredictor, eliminating the
    redundant second PredictionPipeline.predict() call (was causing 2× latency).
    """
    try:
        ci_result = pipeline.predict_with_intervals(input_df, confidence_level=_CI_CONFIDENCE_LEVEL)
        ci = ci_result.get("confidence_intervals")
        if ci is None:
            logger.warning(f"CI id={prediction_id} hash={input_hash}: no confidence_intervals key")
            return None, None, None, None, None

        # key is "lower_bound" (singular list) — "lower_bounds" never exists.
        lower_raw = ci.get("lower_bound")
        upper_raw = ci.get("upper_bound")
        if lower_raw is None or upper_raw is None:
            logger.warning(
                f"CI id={prediction_id} hash={input_hash}: "
                "lower_bound or upper_bound missing from confidence_intervals"
            )
            return None, None, None, None, None

        lower = float(lower_raw[0] if isinstance(lower_raw, list | tuple) else lower_raw)
        upper = float(upper_raw[0] if isinstance(upper_raw, list | tuple) else upper_raw)
        level = float(ci.get("confidence_level", _CI_CONFIDENCE_LEVEL))
        method = str(ci.get("method", "unknown"))
        return lower, upper, level, method, ci_result

    except Exception as exc:
        logger.warning(
            f"CI id={prediction_id} hash={input_hash}: predict_with_intervals failed: {exc}"
        )
        return None, None, None, None, None


# =============================================================================
# DEPENDENCIES
# =============================================================================


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> None:
    """Bearer-token auth guard.

    - Fail-secure behaviour and differentiated 401 responses.
    - When STRICT_AUTH=true: unset API_KEY blocks all requests (fail-secure).
    - When STRICT_AUTH=false (default): unset API_KEY allows all requests (dev mode).
    - Missing Authorization header vs wrong token produce distinct detail strings
      so log aggregators and alerting systems can differentiate between "client
      forgot the header" (low priority) and "invalid credential" (alert-worthy).
    """
    if not _API_KEY:
        if _STRICT_AUTH:
            # was logger.error — fired on EVERY request at 100+ req/s,
            # flooding logs and masking genuine errors.  Downgraded to logger.warning.
            # This condition is a misconfiguration detectable at startup; log it ONCE
            # there (in main.py / lifespan) rather than per-request.
            logger.warning(
                "auth STRICT_AUTH=true but API_KEY is unset — rejecting all requests. "
                "Set the API_KEY environment variable. "
                "(This misconfiguration should be logged once at startup, not per-request.)"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication not configured on server. Contact the API operator.",
            )
        return  # dev/local mode: open access
    if credentials is None:
        logger.warning("auth MISSING_HEADER — no Authorization header supplied")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Include 'Authorization: Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != _API_KEY:
        logger.warning("auth INVALID_TOKEN — bearer token did not match API_KEY")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_metrics_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> None:
    """Bearer-token auth guard for /metrics. No-op when METRICS_TOKEN is unset.
    Missing header vs wrong token produce distinct log/detail strings.
    """
    if not _METRICS_TOKEN:
        return
    if credentials is None:
        logger.warning("metrics-auth MISSING_HEADER — no Authorization header supplied")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header for metrics endpoint.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != _METRICS_TOKEN:
        logger.warning("metrics-auth INVALID_TOKEN — token did not match METRICS_TOKEN")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid metrics token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_predictor(request: Request):
    """Returns the cached HybridPredictor from app state. Raises 503 if absent."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        startup_error = getattr(request.app.state, "startup_error", "Model not initialised")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction service unavailable: {startup_error}",
        )
    return predictor


def get_pipeline(request: Request):
    """Returns the cached PredictionPipeline from app state. Raises 503 if absent."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML pipeline not initialised",
        )
    return pipeline


# =============================================================================
# PROMETHEUS SCRAPE ENDPOINT  (optional — no-op when prometheus-client absent)
# =============================================================================


@router.get(
    "/metrics/prometheus",
    summary="Prometheus scrape endpoint",
    tags=["System"],
    include_in_schema=_PROM_AVAILABLE,
)
async def prometheus_metrics(
    _auth: None = Depends(verify_metrics_token),
):
    """
    Prometheus text-format metrics for scraping by Prometheus / Grafana.
    Only available when prometheus-client is installed.
    Auth: shares METRICS_TOKEN with /api/v1/metrics.
    """
    if not _PROM_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="prometheus-client not installed. pip install prometheus-client",
        )
    return _FastAPIResponse(content=_prom_generate_latest(), media_type=_PROM_CONTENT_TYPE)


# =============================================================================
# ROOT
# =============================================================================


@router.get(
    "/",
    summary="API welcome and documentation",
    tags=["System"],
)
async def root(request: Request) -> dict:
    """
    Welcome endpoint. Returns API information and available endpoints.

    Visit /docs for interactive Swagger UI documentation.
    Visit /redoc for ReDoc documentation.
    """
    return {
        "message": "Insurance Cost Predictor API",
        "version": _API_VERSION,  # M-01: single source — no longer a hardcoded literal
        "status": "ready",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "metrics": "/api/v1/metrics",
            "single_prediction": "POST /api/v1/predict",
            "batch_predictions": "POST /api/v1/predict/batch",
            "model_info": "GET /api/v1/model/info",
        },
        "description": (
            "Production API for annual insurance premium prediction. "
            "Uses a HybridPredictor that blends ML (XGBoost) with actuarial rules, "
            "bias correction, calibration, and conformal confidence intervals."
        ),
    }


# =============================================================================
# HEALTH
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check(request: Request):
    """Returns service health and categorical schema for Streamlit drift detection."""
    pipeline = getattr(request.app.state, "pipeline", None)
    predictor = getattr(request.app.state, "predictor", None)

    if pipeline is None or predictor is None:
        startup_error = getattr(request.app.state, "startup_error", "Service failed to start")
        # return HTTP 503, not 200, when the model is not loaded.
        # Previously both healthy and unhealthy branches returned HTTP 200 (the
        # FastAPI default for a normal `return`). Docker Compose healthcheck uses
        # `curl -sf` where -f fails on 4xx/5xx — a 200 with body status=unhealthy
        # was invisible to Docker, so a broken startup appeared healthy.
        # Returning a JSONResponse bypasses response_model serialisation and lets
        # us set an explicit status code. The body is intentionally compatible with
        # HealthResponse field names so existing consumers parse cleanly.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_name": "none",
                "detail": startup_error,
            },
        )

    return HealthResponse(
        status="healthy",
        model_name=pipeline.model_name,
        pipeline_version=pipeline.VERSION,
        hybrid_version=predictor.VERSION,
        valid_regions=sorted(VALID_REGIONS),
        valid_sex=sorted(VALID_SEX),
        valid_smoker=sorted(VALID_SMOKER),
    )


# =============================================================================
# METRICS  (#6)
# =============================================================================


@router.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    summary="Operational metrics",
    tags=["System"],
)
async def get_metrics(
    request: Request,
    _auth: None = Depends(verify_metrics_token),
) -> MetricsResponse:
    """
    Rolling prediction counters and latency percentiles.

    Auth: requires METRICS_TOKEN bearer token when env var is set.
    When METRICS_TOKEN is unset (dev/local), endpoint is open.

    Streamlit admin panel surfaces alerts when:
      predictions.error_rate_pct > 5    → high error rate
      latency_ms.p99_ms > 20000         → latency degradation (5× nominal p99)
      predictions.rejected_overload > 0 → concurrency cap hit
    """
    snap = _metrics.snapshot()
    pipeline = getattr(request.app.state, "pipeline", None)
    predictor = getattr(request.app.state, "predictor", None)

    return MetricsResponse(
        uptime_seconds=snap["uptime_seconds"],
        predictions=PredictionMetrics(
            total=snap["total"],
            successful=snap["successful"],
            errors=snap["errors"],
            error_rate_pct=snap["error_rate_pct"],
            rejected_overload=snap["rejected"],
            concurrent_limit=_MAX_CONCURRENT,
        ),
        latency_ms=LatencyMetrics(
            mean_ms=snap["mean_ms"],
            p50_ms=snap["p50_ms"],
            p95_ms=snap["p95_ms"],
            p99_ms=snap["p99_ms"],
            samples=snap["latency_samples"],
        ),
        model_name=pipeline.model_name if pipeline else "unknown",
        hybrid_version=predictor.VERSION if predictor else None,
        pipeline_version=pipeline.VERSION if pipeline else None,
    )


# =============================================================================
# SINGLE PREDICTION
# =============================================================================


@router.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    summary="Predict single insurance cost",
    tags=["Prediction"],
)
def predict_single(
    body: PredictRequest,
    predictor=Depends(get_predictor),
    _auth=Depends(verify_api_key),
) -> PredictResponse:
    """
    Predict annual insurance premium for one individual, with confidence interval.

    I-01: sync def — FastAPI runs in threadpool, preserving event-loop concurrency.
    #7:   Semaphore caps concurrent calls. Returns 429 when at capacity.

    Single inference path.
      predict_with_intervals() is called first to obtain both the ML-only prediction
      and the CI in one pass. The result dict is passed into HybridPredictor via
      _precomputed_ml_result so the ML pipeline is not called a second time.

    CI centred on the hybrid prediction, not the ML-only prediction.
      CI bounds are scaled by (hybrid_pred / ml_pred) after blending, so the
      reported interval always brackets the returned hybrid point estimate —
      even for policies in the actuarial-dominant / transition zone.
    """
    # ── #7: Concurrency guard — must be BEFORE try/finally ───────────────────
    # If acquire fails we raise immediately and never enter the try block,
    # so release() in finally is never reached. Semaphore count stays correct.
    if not _predict_semaphore.acquire(blocking=False):
        _metrics.record_rejection()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Server busy: {_MAX_CONCURRENT} predictions already running. " "Retry in a moment."
            ),
        )

    prediction_id = str(uuid.uuid4())
    input_data = body.model_dump()
    input_hash = _hash_input(input_data)
    # t_start and input_df are now INSIDE the try block so the
    # semaphore is always released by finally even if pd.DataFrame() or
    # time.perf_counter() raise unexpectedly.  The 6-line gap between
    # acquire() and try: has been closed.
    t_start = time.perf_counter()
    success = False

    try:
        input_df = pd.DataFrame([input_data])
        # ── 1. ML prediction + CI in a single inference pass ─────────
        # predict_with_intervals() calls predict() internally and returns the
        # full result dict extended with confidence_intervals.  We reuse this
        # pre-computed ML result when calling HybridPredictor below.
        lower_bound, upper_bound, ci_level, ci_method, ml_ci_result = _compute_ci(
            pipeline=predictor.ml_predictor,
            input_df=input_df,
            prediction_id=prediction_id,
            input_hash=input_hash,
        )
        t_ml = time.perf_counter()

        # ── 2. Hybrid prediction — reuses the ML result, no second inference ──
        # _precomputed_ml_result is the result dict from ml_predictor.predict()
        # (which predict_with_intervals already executed internally).  When
        # provided, HybridPredictor skips the internal ml_predictor.predict()
        # call, cutting total inference time by ~50%.
        # Falls back transparently if HybridPredictor doesn't support the param.
        try:
            hybrid_result = predictor.predict(
                input_df,
                return_reliability=False,
                _precomputed_ml_result=ml_ci_result,
            )
        except TypeError:
            # Older HybridPredictor without _precomputed_ml_result support.
            hybrid_result = predictor.predict(input_df, return_reliability=False)

        prediction = float(hybrid_result["predictions"][0])
        model_used: str = hybrid_result["model_used"]
        t_point = time.perf_counter()

        # ── Guard: non-finite output before Pydantic serialization ────────────
        if not _math.isfinite(
            prediction
        ):  # M-05: uses module-level _math (was redundant inner import)
            raise ValueError(
                f"Model returned non-finite prediction ({prediction!r}) for "
                f"input_hash={input_hash}. "
                "Possible cause: log-space overflow in inverse transformation. "
                "Verify model artifact integrity and input feature range."
            )

        # ── 3. Re-centre CI on the hybrid prediction ─────────────────
        # The CI from predict_with_intervals() is centred on the ML-only
        # prediction.  When the actuarial blend shifts the final prediction
        # (transition / actuarial-dominant policies), the CI must be shifted
        # by the same ratio so that [lower_bound, upper_bound] brackets the
        # reported hybrid prediction value.
        if lower_bound is not None and upper_bound is not None and ml_ci_result is not None:
            ml_pred_raw = ml_ci_result.get("predictions", [None])[0]
            if ml_pred_raw is not None:
                ml_pred_float = float(ml_pred_raw)
                if ml_pred_float > 1e-8:
                    _blend_ratio = prediction / ml_pred_float
                    lower_bound = lower_bound * _blend_ratio
                    upper_bound = upper_bound * _blend_ratio
                    if abs(_blend_ratio - 1.0) > 0.01:
                        logger.debug(
                            f"CI re-centred: blend_ratio={_blend_ratio:.4f} "
                            f"(hybrid=${prediction:,.0f} vs ml=${ml_pred_float:,.0f}) "
                            f"id={prediction_id}"
                        )

        elapsed = time.perf_counter() - t_start
        success = True

        # compute CI width as fraction of prediction — machine-readable.
        # config.yaml notes "CI mean width $11,859 = 92% of mean premium — must be
        # disclosed to downstream consumers."  Callers can now gate on ci_width_pct
        # without computing (upper - lower) / prediction themselves.
        ci_width_pct: float | None = None
        if lower_bound is not None and upper_bound is not None and prediction > 1e-8:
            ci_width_pct = round((upper_bound - lower_bound) / prediction * 100.0, 2)

        # set ci_unreliable flag when width exceeds actionable threshold.
        # 80% chosen conservatively — the logged problem case was 118.8%.
        _CI_UNRELIABLE_THRESHOLD = 80.0
        ci_unreliable: bool | None = (
            (ci_width_pct > _CI_UNRELIABLE_THRESHOLD) if ci_width_pct is not None else None
        )

        # forward actuarial/ML ratio from HybridPredictor result so
        # downstream systems can gate on high-disagreement predictions without
        # parsing server logs.
        actuarial_ml_ratio: float | None = hybrid_result.get("actuarial_conservativeness_ratio")

        ci_str = (
            f"lower=${lower_bound:,.0f} upper=${upper_bound:,.0f} "
            f"width={ci_width_pct:.1f}% method={ci_method}"
            if lower_bound is not None
            else "unavailable"
        )
        logger.info(
            f"predict id={prediction_id} hash={input_hash} model={model_used} "
            f"result=${prediction:,.2f} ci={ci_str} "
            f"t_ml={t_ml - t_start:.3f}s t_hybrid={t_point - t_ml:.3f}s elapsed={elapsed:.3f}s"
        )

        return PredictResponse(
            prediction=prediction,
            model_used=model_used,
            prediction_id=prediction_id,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            ci_confidence_level=ci_level,
            ci_method=ci_method,
            ci_width_pct=ci_width_pct,
            ci_unreliable=ci_unreliable,
            actuarial_ml_ratio=actuarial_ml_ratio,
        )

    except ValueError as exc:
        elapsed = time.perf_counter() - t_start
        logger.warning(
            f"predict validation error id={prediction_id} hash={input_hash} "
            f"elapsed={elapsed:.3f}s error={exc}"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        logger.error(
            f"predict failed id={prediction_id} hash={input_hash} "
            f"elapsed={elapsed:.3f}s error={exc}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check server logs for details.",
        ) from exc

    finally:
        _predict_semaphore.release()
        _metrics.record_prediction(time.perf_counter() - t_start, success)


# =============================================================================
# BATCH PREDICTION
# =============================================================================


def _predict_single_row(predictor, row_data: dict, index: int) -> BatchPredictRecord:
    """Per-record fallback helper. Catches all exceptions so one bad row doesn't abort the rest."""
    try:
        df = pd.DataFrame([row_data])
        result = predictor.predict(df, return_reliability=False)
        return BatchPredictRecord(
            index=index,
            prediction=float(result["predictions"][0]),
            model_used=result["model_used"],
            status="success",
        )
    except Exception as exc:
        return BatchPredictRecord(
            index=index,
            prediction=None,
            model_used=None,
            error=str(exc)[:200],
            status="error",
        )


@router.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch insurance cost predictions",
    tags=["Prediction"],
)
def predict_batch(
    body: BatchPredictRequest,
    predictor=Depends(get_predictor),
    _auth=Depends(verify_api_key),
) -> BatchPredictResponse:
    """
    Batch predictions. CI fields not included — predict_with_intervals() is O(N) × 120ms.

    #7: One semaphore slot per batch regardless of batch size.
        A large batch doesn't starve all concurrent single predictions.
    W01: Vectorised fast path; per-record fallback with partial-success support.
    """
    if not _predict_semaphore.acquire(blocking=False):
        _metrics.record_rejection()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Server busy: {_MAX_CONCURRENT} predictions already running. " "Retry in a moment."
            ),
        )

    records_dicts = [r.model_dump() for r in body.records]
    batch_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()
    success = False

    try:
        input_df = pd.DataFrame(records_dicts)
        result = predictor.predict(input_df, return_reliability=False)
        predictions: list[float] = result["predictions"]
        model_used: str = result["model_used"]

        batch_results = [
            BatchPredictRecord(
                index=i,
                prediction=float(predictions[i]),
                model_used=model_used,
                status="success",
            )
            for i in range(len(predictions))
        ]

        elapsed = time.perf_counter() - t_start
        success = True
        logger.info(
            f"batch id={batch_id} records={len(records_dicts)} model={model_used} "
            f"successful={len(batch_results)} failed=0 path=vectorised elapsed={elapsed:.3f}s"
        )
        return BatchPredictResponse(
            results=batch_results,
            total=len(batch_results),
            successful=len(batch_results),
            failed=0,
            model_used=model_used,
            ci_available=False,
            # expose machine-readable reason so consumers know why CI is absent.
            ci_unavailable_reason=(
                "Batch endpoint does not compute confidence intervals — "
                "predict_with_intervals() is O(N)×120ms. Use POST /api/v1/predict "
                "for single-record CI, or contact the API operator to enable batch CI."
            ),
        )

    except Exception as vec_exc:
        logger.warning(
            f"batch id={batch_id} vectorised failed ({vec_exc!r}), "
            f"falling back to per-record (max {_FALLBACK_ROW_CAP} rows)"
        )

        # Reject batches that exceed the per-record fallback cap.
        # Without this guard a 10K-row batch whose vectorised path fails holds
        # the semaphore slot for ~11 hours and writes ≥10K log lines.
        # Callers should retry with a smaller batch or investigate the
        # vectorised failure via the server logs (vec_exc above).
        if len(records_dicts) > _FALLBACK_ROW_CAP:
            elapsed = time.perf_counter() - t_start
            logger.error(
                f"batch id={batch_id} batch too large for per-record fallback "
                f"({len(records_dicts)} rows > cap {_FALLBACK_ROW_CAP}); "
                f"returning 503. elapsed={elapsed:.3f}s"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Vectorised batch prediction failed and batch size "
                    f"({len(records_dicts)}) exceeds per-record fallback limit "
                    f"({_FALLBACK_ROW_CAP}). Retry with a smaller batch or "
                    f"contact support. Original error: {vec_exc!r}"
                ),
            ) from vec_exc
        batch_results = [
            _predict_single_row(predictor, row, i) for i, row in enumerate(records_dicts)
        ]
        successful = [r for r in batch_results if r.status == "success"]
        failed_records = [r for r in batch_results if r.status == "error"]
        model_used = (
            successful[0].model_used if successful and successful[0].model_used else "unknown"
        )

        elapsed = time.perf_counter() - t_start
        success = len(successful) > 0
        logger.info(
            f"batch id={batch_id} records={len(records_dicts)} "
            f"successful={len(successful)} failed={len(failed_records)} "
            f"path=per-record-fallback elapsed={elapsed:.3f}s"
        )
        if failed_records:
            logger.warning(f"batch id={batch_id} first error: {failed_records[0].error}")

        return BatchPredictResponse(
            results=batch_results,
            total=len(batch_results),
            successful=len(successful),
            failed=len(failed_records),
            model_used=model_used,
            ci_available=False,
            ci_unavailable_reason=(
                "Batch endpoint does not compute confidence intervals — "
                "predict_with_intervals() is O(N)×120ms. Use POST /api/v1/predict "
                "for single-record CI, or contact the API operator to enable batch CI."
            ),
        )

    finally:
        _predict_semaphore.release()
        _metrics.record_prediction(time.perf_counter() - t_start, success)


# =============================================================================
# MODEL INFO
# =============================================================================


@router.get(
    "/api/v1/model/info",
    response_model=ModelInfoResponse,
    summary="Model and pipeline metadata",
    tags=["Model"],
)
async def model_info(
    pipeline=Depends(get_pipeline),
    _auth=Depends(verify_api_key),
) -> ModelInfoResponse:
    """Pipeline version, active model, and target transformation details."""
    try:
        info = pipeline.get_pipeline_info()
        raw_transform = info.get("target_transformation", {})
        transform = TargetTransformationInfo(
            method=raw_transform.get("method", "unknown"),
            bias_correction=bool(raw_transform.get("bias_correction", False)),
            bias_correction_variance=raw_transform.get("bias_correction_variance"),
            recommended_metrics=raw_transform.get("recommended_metrics", []),
            deprecated_metrics=raw_transform.get("deprecated_metrics", []),
        )
        return ModelInfoResponse(
            model_name=info["model_name"],
            model_type=info.get("model_type", "unknown"),
            pipeline_version=info["pipeline_version"],
            target_transformation=transform,
            feature_count=info.get("feature_count"),
            has_feature_importances=info.get("has_feature_importances", False),
            has_coefficients=info.get("has_coefficients", False),
            pipeline_state=info.get("pipeline_state"),
        )
    except Exception as exc:
        logger.error(f"Failed to retrieve model info: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve model information.",
        ) from exc
