"""
API Entry Point - Insurance Cost Predictor
Compatible with predict.py v6.3.1 and config.yaml v7.5.0

Changes vs original:
  W09     TimingMiddleware logs method/path/status/elapsed per request.
  WARMUP  api.warm_up_on_start honoured — primes numba JIT before first user.
          Override via WARMUP_ENABLED env var (set false in CI).
  #6      _metrics.mark_ready() called at end of lifespan so uptime_seconds
          counts from when the API is ready, not from module import time.
          _metrics exposed via app.state.metrics for other components.
  #7      MAX_CONCURRENT_PREDICTIONS and CI_CONFIDENCE_LEVEL logged at startup.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from api.routes import _API_VERSION
# Single-source API version — defined in api/routes.py and imported above.
# Previously defined here AND hardcoded as "1.2.0" in routes.root(); both are
# now sourced from routes._API_VERSION (fix M-01).
_API_VERSION  # re-export reference kept so existing callers of main._API_VERSION work

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_PROJECT_ROOT = Path(__file__).parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# logging.basicConfig MUST be called before importing routes so that the
# named loggers in routes.py have a handler when they first emit messages.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from insurance_ml.config import load_config  # noqa: E402
from insurance_ml.predict import HybridPredictor, PredictionPipeline  # noqa: E402
from api.routes import (
    router,
    _metrics,
    _MAX_CONCURRENT,
    _CI_CONFIDENCE_LEVEL,
    _API_VERSION,       # M-01: single source of truth now lives in routes.py
)  # noqa: E402

# ── Warmup payload ────────────────────────────────────────────────────────────
_WARMUP_ROW = pd.DataFrame(
    [
        {
            "age": 35,
            "sex": "male",
            "bmi": 27.5,
            "children": 2,
            "smoker": "no",
            "region": "northeast",
        }
    ]
)


def _run_warmup(predictor: HybridPredictor) -> None:
    """
    One prediction at startup to prime numba JIT / sklearn caches.
    Non-fatal — logged as WARNING, API continues normally on failure.
    """
    logger.info("[warmup] Priming inference pipeline...")
    t = time.perf_counter()
    try:
        predictor.predict(_WARMUP_ROW, return_reliability=False)
        logger.info(f"[warmup] ✅ Pipeline primed in {time.perf_counter() - t:.3f}s")
    except Exception as exc:
        logger.warning(
            f"[warmup] ⚠️ Failed after {time.perf_counter() - t:.3f}s: {exc}\n"
            "   First real request will pay the cold-start penalty.",
            exc_info=True,
        )


# =============================================================================
# CHECKSUM VERIFICATION
# =============================================================================


def _verify_model_checksums(model_path: str, active_model: str) -> None:
    """
    Verify the SHA-256 checksum of the active model artifact.

    Called after PredictionPipeline is initialised so that `active_model`
    reflects the artifact that was *actually* loaded (dynamic, not hardcoded).

    Behaviour:
      • Checksum file present + match  → logs ✅ INFO, continues.
      • Checksum file absent           → logs ⚠️ WARNING, continues.
                                         (backward-compat: pre-checksum artifacts)
      • Checksum file present + MISMATCH → raises RuntimeError, aborts startup.
      • Model file absent              → logs DEBUG, skips silently.
                                         (specialist models are optional)
    Checksum filename convention: <model_stem>_checksum.txt
    Generate with: python scripts/generate_checksums.py
    """
    base = Path(model_path)
    model_file = f"{active_model}.joblib"
    checksum_file = f"{active_model}_checksum.txt"

    model_path_full = base / model_file
    checksum_path = base / checksum_file

    if not model_path_full.exists():
        logger.debug(
            f"[checksum] Skipping {model_file} — not found at {model_path_full} "
            "(may be loaded by name later or is an optional artifact)"
        )
        return

    if not checksum_path.exists():
        logger.warning(
            f"⚠️  No checksum file for {model_file} "
            f"({checksum_file} not found in {base}). "
            "Run scripts/generate_checksums.py after training to enable verification."
        )
        return

    expected = checksum_path.read_text().strip()
    actual = hashlib.sha256(model_path_full.read_bytes()).hexdigest()

    if actual != expected:
        raise RuntimeError(
            f"Checksum MISMATCH for {model_file}!\n"
            f"  Expected : {expected}\n"
            f"  Actual   : {actual}\n"
            f"  Location : {model_path_full}\n"
            "The artifact may be corrupted or replaced since last training. "
            "Re-run train.py or restore from a known-good backup."
        )

    logger.info(f"✅ Checksum verified: {model_file}")


# =============================================================================
# MIDDLEWARE
# =============================================================================


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        logger.info(
            f"{request.method:<5} {request.url.path:<30} {response.status_code}  {elapsed:.3f}s"
        )
        return response


# =============================================================================
# LIFESPAN
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Insurance Cost Predictor API — starting up")
    logger.info("=" * 60)

    app.state.pipeline = None
    app.state.predictor = None
    app.state.startup_error = None
    app.state.metrics = _metrics  # expose MetricsCollector via app.state

    try:
        # ── 1. Config ─────────────────────────────────────────────────────────
        logger.info("[1/4] Loading configuration...")
        config = load_config()
        app.state.config = config
        logger.info(
            f"       Config v{config.get('version', '?')} | "
            f"env={config.get('environment', '?')}"
        )

        # ── 2. PredictionPipeline ─────────────────────────────────────────────
        logger.info("[2/4] Initialising PredictionPipeline...")
        model_path: str = os.getenv(
            "MODEL_PATH", config.get("model", {}).get("model_path", "models/")
        )
        # Pass preprocessor_path=None to trigger mtime-based auto-resolution
        # inside PredictionPipeline.__init__ → _resolve_preprocessor_path(None).
        # Pass the explicit env var only if the operator has set it.
        preprocessor_path: str | None = os.getenv("PREPROCESSOR_PATH") or None

        pipeline = PredictionPipeline(
            model_dir=model_path, preprocessor_path=preprocessor_path
        )
        app.state.pipeline = pipeline
        logger.info(f"       Active model: {pipeline.model_name}")

        # ── 2.5. Checksum verification ────────────────────────────────────────
        # Called here — after PredictionPipeline is initialised — so
        # `pipeline.model_name` reflects the artifact that was actually loaded
        # (dynamic via _get_best_model()), not a hardcoded guess.
        # `model_path` is already resolved from env + config above, so
        # the directory checked always matches the directory loaded from.
        logger.info("[2.5/4] Verifying model artifact checksum(s)...")
        _verify_model_checksums(model_path, active_model=pipeline.model_name)

        # ── 3. HybridPredictor ────────────────────────────────────────────────
        logger.info("[3/4] Initialising HybridPredictor...")
        predictor = HybridPredictor(ml_predictor=pipeline)
        app.state.predictor = predictor
        logger.info(
            f"       HybridPredictor v{predictor.VERSION} ready | "
            f"threshold=${predictor.threshold:,.0f} | "
            f"calibration={'on' if predictor.calibration_enabled else 'off'}"
        )

        # ── Verify specialist model checksum (if HighValueSegmentRouter active) ─
        # NOTE: _segment_router lives on pipeline (PredictionPipeline), NOT on
        # predictor (HybridPredictor).  The previous code read from predictor,
        # which has no _segment_router attribute — getattr always returned None,
        # so the gate silently skipped the specialist checksum even when the
        # specialist model was loaded and serving live traffic.
        # Fix: read from predictor.ml_predictor (= pipeline).
        _segment_router = getattr(predictor.ml_predictor, "_segment_router", None)
        if _segment_router is not None and getattr(_segment_router, "enabled", False):
            _specialist_name = getattr(
                _segment_router,
                "specialist_model_name",
                getattr(_segment_router, "DEFAULT_SPECIALIST_NAME", None),
            )
            if _specialist_name:
                logger.info(
                    f"       Verifying specialist model checksum: {_specialist_name}"
                )
                _verify_model_checksums(model_path, active_model=_specialist_name)
        else:
            logger.info(
                "       Specialist router not active — skipping specialist checksum"
            )
        # Log #6 / #7 settings now that logging is configured
        _api_workers = int(os.getenv("API_WORKERS", "1"))
        logger.info(
            f"       max_concurrent_predictions={_MAX_CONCURRENT} | "
            f"ci_confidence_level={_CI_CONFIDENCE_LEVEL} | "
            f"workers={_api_workers}"
        )
        if _api_workers > 1:
            _effective_cap = _MAX_CONCURRENT * _api_workers
            logger.warning(
                f"\u26a0\ufe0f  MULTI-WORKER SEMAPHORE MULTIPLICATION:\n"
                f"   workers={_api_workers} x MAX_CONCURRENT_PREDICTIONS={_MAX_CONCURRENT} "
                f"= effective cap {_effective_cap}.\n"
                f"   Set MAX_CONCURRENT_PREDICTIONS={_MAX_CONCURRENT // _api_workers} "
                f"or use a Redis-backed distributed semaphore for true enforcement."
            )

        # ── 4. Warmup ─────────────────────────────────────────────────────────
        _cfg_warmup: bool = config.get("api", {}).get("warm_up_on_start", True)
        _env_raw: str | None = os.getenv("WARMUP_ENABLED")
        _do_warmup: bool = (
            _env_raw.lower() not in ("false", "0", "no")
            if _env_raw is not None
            else _cfg_warmup
        )
        logger.info(
            f"[4/4] Warmup: {'running' if _do_warmup else 'skipped'} "
            f"(config={_cfg_warmup}, env={_env_raw!r})"
        )
        if _do_warmup:
            _run_warmup(predictor)

        # ── Mark metrics start time AFTER all startup work is complete ────────
        # This ensures uptime_seconds in /api/v1/metrics counts from when
        # the API is actually ready to serve, not from module import time.
        _metrics.mark_ready()

        logger.info("=" * 60)
        logger.info("✅ Startup complete — API ready to serve predictions")
        logger.info("=" * 60)

    except FileNotFoundError as exc:
        msg = f"Model files not found: {exc}. Run train.py first."
        logger.critical(f"❌ Startup failed: {msg}")
        app.state.startup_error = msg

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.critical(f"❌ Startup failed: {msg}", exc_info=True)
        app.state.startup_error = msg

    yield

    logger.info("Shutting down Insurance Cost Predictor API...")
    app.state.predictor = None
    app.state.pipeline = None
    logger.info("✅ Shutdown complete")


# =============================================================================
# APP FACTORY
# =============================================================================


def create_app() -> FastAPI:
    app = FastAPI(
        title="Insurance Cost Predictor API",
        description=(
            "Production API for annual insurance premium prediction.\n\n"
            "Uses a HybridPredictor that blends ML (XGBoost) with actuarial rules, "
            "bias correction, calibration, and conformal confidence intervals.\n\n"
            "**Concurrency note:** predict_single and predict_batch are synchronous "
            "(def) routes. FastAPI runs them in a thread pool. A semaphore caps "
            "concurrent predictions; requests beyond the cap receive 429."
        ),
        version=_API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(TimingMiddleware)

    _raw_origins = os.getenv(
        "CORS_ORIGINS",
        (
            "http://localhost:8501,http://127.0.0.1:8501,"
            "http://localhost:8000,http://127.0.0.1:8000"
        ),
    )
    allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.critical(
            f"Unhandled exception on {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected server error occurred."},
        )

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    _host = os.getenv("API_HOST", "127.0.0.1")
    _port = int(os.getenv("API_PORT", "8000"))
    _reload = os.getenv("API_RELOAD", "false").lower() in ("true", "1", "yes")
    _log_level = os.getenv("LOG_LEVEL", "info").lower()
    _workers = int(os.getenv("API_WORKERS", "1"))

    logger.info(
        f"Starting Insurance Cost Predictor API v{_API_VERSION}: "
        f"host={_host} port={_port} reload={_reload} workers={_workers}"
    )
    uvicorn.run(
        "api.main:app",
        host=_host,
        port=_port,
        reload=_reload,
        log_level=_log_level,
        workers=_workers,
    )
