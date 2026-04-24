"""
Inference latency profiler for HybridPredictor.predict()
=========================================================
Run from the project root (Windows PowerShell):

    py311 scripts/profile_inference.py

Produces:
  1. Per-step wall-clock breakdown (printed to stdout)
  2. cProfile cumulative report (top 30 functions by cumtime)
  3. reports/profile_inference.txt — persistent copy of both outputs

Requires: the pipeline to be trained (models/ must exist).

Usage flags:
  --n-runs N     Number of timed prediction runs after warmup (default: 5)
  --no-cprofile  Skip cProfile pass (faster, use when iterating on manual timers)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure src/ is on sys.path (mirrors main.py bootstrap)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Monkey-patch timing hooks onto the hot-path methods BEFORE importing them.
# We inject a thin wrapper around each suspect function so we get per-call
# wall-clock even without line_profiler installed.
# ---------------------------------------------------------------------------

# Record [(label, elapsed_seconds), ...] per predict() call
_TIMINGS: list[tuple[str, float]] = []


def _timed(label: str, fn, *args, **kwargs):
    t = time.perf_counter()
    result = fn(*args, **kwargs)
    _TIMINGS.append((label, time.perf_counter() - t))
    return result


# ---------------------------------------------------------------------------
# Import pipeline — triggers joblib model loads (included in startup timing)
# ---------------------------------------------------------------------------
print("Loading insurance_ml modules...", flush=True)
t_import_start = time.perf_counter()

from insurance_ml.config import load_config  # noqa: E402
from insurance_ml.predict import HybridPredictor, PredictionPipeline  # noqa: E402

t_import_end = time.perf_counter()
print(f"  Module import: {t_import_end - t_import_start:.3f}s\n")

# ---------------------------------------------------------------------------
# Patch PredictionPipeline to capture per-step timings
# ---------------------------------------------------------------------------

_orig_preprocess = PredictionPipeline.preprocess_input
_orig_pp_predict = PredictionPipeline.predict


def _patched_preprocess(self, input_data):
    # 1a. transform_pipeline is the key suspect — time it separately
    df = input_data.copy()
    try:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["children"] = pd.to_numeric(df["children"], errors="coerce")
    except Exception:
        pass
    df = self._validate_categorical_values(df)

    t0 = time.perf_counter()
    processed = self.feature_engineer.transform_pipeline(df)
    _TIMINGS.append(("feature_engineer.transform_pipeline", time.perf_counter() - t0))
    return processed


def _patched_pp_predict(self, input_data, return_reliability=True):
    global _TIMINGS
    _TIMINGS.clear()

    t_total = time.perf_counter()

    # --- preprocess (includes transform_pipeline timing) ---
    t = time.perf_counter()
    processed_input = _patched_preprocess(self, input_data)
    _TIMINGS.append(("preprocess_input (total)", time.perf_counter() - t))

    # --- XGBoost inference ---
    import xgboost as _xgb

    t = time.perf_counter()
    try:
        if hasattr(self.model, "get_booster"):
            _arr = (
                processed_input.values
                if hasattr(processed_input, "values")
                else np.asarray(processed_input)
            )
            _dmat = _xgb.DMatrix(_arr)
            predictions_raw = self.model.get_booster().predict(_dmat)
        else:
            predictions_raw = self.model.predict(processed_input)
    except Exception:
        predictions_raw = self.model.predict(processed_input)
    _TIMINGS.append(("xgboost_inference", time.perf_counter() - t))

    # --- inverse transform ---
    t = time.perf_counter()

    predictions_original = self.feature_engineer.inverse_transform_target(
        predictions_raw,
        transformation_method=self.feature_engineer.target_transformation.method,
        clip_to_safe_range=True,
        context="prediction",
    )
    _TIMINGS.append(("inverse_transform_target", time.perf_counter() - t))

    # --- bias correction ---
    t = time.perf_counter()
    if self._bias_correction is not None:
        predictions_original = self._bias_correction.apply(
            y_pred=predictions_original,
            y_original=predictions_original,
            log_details=False,
        )
    _TIMINGS.append(("bias_correction", time.perf_counter() - t))

    # --- segment routing ---
    t = time.perf_counter()
    _routing_diagnostics = {}
    if self._segment_router.enabled:
        predictions_original, _routing_diagnostics = self._segment_router.route(
            processed_input=processed_input,
            global_preds_original=predictions_original,
            feature_engineer=self.feature_engineer,
        )
    _TIMINGS.append(("segment_router", time.perf_counter() - t))

    # --- statistics dict ---
    t = time.perf_counter()
    _ = {
        "mean": float(np.mean(predictions_original)),
        "median": float(np.median(predictions_original)),
        "min": float(np.min(predictions_original)),
        "max": float(np.max(predictions_original)),
        "std": float(np.std(predictions_original)),
        "q25": float(np.percentile(predictions_original, 25)),
        "q75": float(np.percentile(predictions_original, 75)),
    }
    _TIMINGS.append(("statistics_dict_construction", time.perf_counter() - t))

    _TIMINGS.append(("PredictionPipeline.predict (total)", time.perf_counter() - t_total))

    # Re-run the real predict() to get correct return value
    # (our patched version above returns early — call original for the result)
    return _orig_pp_predict(self, input_data, return_reliability=return_reliability)


PredictionPipeline.preprocess_input = _patched_preprocess  # type: ignore[assignment]
PredictionPipeline.predict = _patched_pp_predict  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Test data (typical single-row prediction — matches Streamlit form defaults)
# ---------------------------------------------------------------------------
TEST_ROW = pd.DataFrame(
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

TEST_ROW_SMOKER = pd.DataFrame(
    [
        {
            "age": 52,
            "sex": "female",
            "bmi": 33.1,
            "children": 0,
            "smoker": "yes",
            "region": "southeast",
        }
    ]
)


# ---------------------------------------------------------------------------
# Pipeline initialisation (mirrors main.py lifespan)
# ---------------------------------------------------------------------------
def init_pipeline():
    import glob
    import os

    config = load_config()
    model_path = os.getenv("MODEL_PATH", config.get("model", {}).get("model_path", "models/"))

    preprocessor_path = os.getenv("PREPROCESSOR_PATH")
    if preprocessor_path is None:
        matches = sorted(glob.glob(f"{model_path}/preprocessor_v*.joblib"), reverse=True)
        preprocessor_path = matches[0] if matches else f"{model_path}/preprocessor_v5.2.0.joblib"

    pipeline = PredictionPipeline(model_dir=model_path, preprocessor_path=preprocessor_path)
    predictor = HybridPredictor(ml_predictor=pipeline)
    return predictor


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def run_timed_predict(predictor, df, label=""):
    global _TIMINGS
    _TIMINGS.clear()
    t_wall = time.perf_counter()
    result = predictor.predict(df, return_reliability=False)
    wall = time.perf_counter() - t_wall

    snapshot = list(_TIMINGS)  # capture before next call clears it
    return wall, result, snapshot


def print_timing_report(wall: float, timings: list, label: str, out=None):
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {label}")
    lines.append(f"{'='*60}")
    lines.append(f"  Total wall-clock:  {wall*1000:.1f} ms")
    lines.append(f"  {'Step':<45} {'ms':>8}  {'%':>6}")
    lines.append(f"  {'-'*45}  {'-'*8}  {'-'*6}")
    for name, elapsed in sorted(timings, key=lambda x: x[1], reverse=True):
        pct = elapsed / wall * 100 if wall > 0 else 0
        lines.append(f"  {name:<45}  {elapsed*1000:>7.1f}  {pct:>5.1f}%")
    text = "\n".join(lines)
    print(text)
    if out is not None:
        out.write(text + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference latency profiler")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--no-cprofile", action="store_true")
    args = parser.parse_args()

    report_path = Path("reports/profile_inference.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(report_path, "w", encoding="utf-8")

    header = (
        "\nINFERENCE LATENCY PROFILER\n"
        "==========================\n"
        f"  n_runs (after warmup): {args.n_runs}\n"
    )
    print(header)
    out.write(header)

    # ── 1. Startup timing ─────────────────────────────────────────────────
    print("Initialising pipeline (includes joblib model loads)...", flush=True)
    t_startup = time.perf_counter()
    predictor = init_pipeline()
    startup_elapsed = time.perf_counter() - t_startup
    msg = f"  Pipeline + predictor init: {startup_elapsed:.3f}s\n"
    print(msg)
    out.write(msg)

    # ── 2. Cold-start (first call — numba JIT / lazy-init penalty) ────────
    print("Running cold-start prediction...", flush=True)
    wall_cold, _, timings_cold = run_timed_predict(predictor, TEST_ROW, "cold")
    print_timing_report(wall_cold, timings_cold, "COLD START (first call)", out)

    # ── 3. Warm runs ──────────────────────────────────────────────────────
    walls = []
    all_timings = []
    for _i in range(args.n_runs):
        w, _, t = run_timed_predict(predictor, TEST_ROW)
        walls.append(w)
        all_timings.append(t)

    # Average timings across warm runs
    avg_timings: dict[str, float] = {}
    for run_timings in all_timings:
        for name, elapsed in run_timings:
            avg_timings[name] = avg_timings.get(name, 0.0) + elapsed / args.n_runs

    print_timing_report(
        sum(walls) / len(walls),
        list(avg_timings.items()),
        f"WARM RUN average ({args.n_runs} runs, non-smoker)",
        out,
    )

    # ── 4. Smoker row (routes through different path in blend logic) ──────
    w_s, _, t_s = run_timed_predict(predictor, TEST_ROW_SMOKER)
    print_timing_report(w_s, t_s, "WARM RUN (smoker, high-premium path)", out)

    # ── 5. cProfile pass ──────────────────────────────────────────────────
    if not args.no_cprofile:
        print("\nRunning cProfile pass (warm call)...", flush=True)
        pr = cProfile.Profile()
        pr.enable()
        predictor.predict(TEST_ROW, return_reliability=False)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(30)
        cprofile_text = (
            "\n\nCPROFILE (top 30 by cumulative time, warm call)\n" + "=" * 60 + "\n" + s.getvalue()
        )
        print(cprofile_text)
        out.write(cprofile_text)

    out.close()
    print(f"\n✅ Full report written to: {report_path}")

    # ── 6. Diagnosis hint ─────────────────────────────────────────────────
    cold_ms = wall_cold * 1000
    warm_ms = sum(walls) / len(walls) * 1000
    jit_penalty = cold_ms - warm_ms

    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print(f"  Cold-start latency:  {cold_ms:.0f} ms")
    print(f"  Warm latency:        {warm_ms:.0f} ms")
    print(f"  JIT/init penalty:    {jit_penalty:.0f} ms  (eliminated by warmup call)")

    if warm_ms > 1000:
        # Find the biggest warm-run contributor
        top = sorted(avg_timings.items(), key=lambda x: x[1], reverse=True)
        bottleneck, bottleneck_ms = top[0][0], top[0][1] * 1000
        print(
            f"\n  ⚠️  Warm latency still > 1s — bottleneck: [{bottleneck}] @ {bottleneck_ms:.0f}ms"
        )
        print("  Recommendation: upload features.py for line-level profiling of that function.")
    else:
        print("\n  ✅ Warm latency < 1s — warmup call in main.py will resolve the UX issue.")

    print("=" * 60)


if __name__ == "__main__":
    main()
