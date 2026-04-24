"""
scripts/train_model.py
======================
CLI entry point for the Insurance ML training pipeline.

Wraps ``insurance_ml.train.main()`` with argparse-driven config overrides so
you can control the full pipeline from the command line without editing
config.yaml between runs.

How it works
------------
1. Parse CLI args.
2. Load config.yaml (``insurance_ml.config.load_config``).
3. Apply CLI overrides to the in-memory config dict only — config.yaml is
   never written.
4. When overrides are present, temporarily monkey-patch
   ``insurance_ml.config.load_config`` so that ``ModelTrainer.__init__``
   (which does ``from insurance_ml.config import load_config`` each time)
   receives the modified dict.  The patch is restored in a ``finally`` block.
5. Call ``insurance_ml.train.main()``, which owns all gate, calibration,
   MLflow, and specialist logic.

Usage examples
--------------
# Full pipeline — all models, MLflow + Optuna HPO (same as running main()):
python scripts/train_model.py

# Fast iteration — single model, no tracking, no HPO:
python scripts/train_model.py --models xgboost_median --no-mlflow --no-hpo

# Two models, HPO, custom config path:
python scripts/train_model.py --models xgboost_median ridge \\
    --config configs/experiment.yaml

# Validate config + data, then exit without training:
python scripts/train_model.py --dry-run

# Debug a failing run at DEBUG verbosity:
python scripts/train_model.py --models ridge --no-hpo --log-level DEBUG

# Disable two-model architecture, train every model individually:
python scripts/train_model.py --no-two-model

# Override output directory and data path:
python scripts/train_model.py --output-dir models/exp_01 --data data/raw/insurance_v2.csv
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``insurance_ml`` is importable when
# the script is run directly (``python scripts/train_model.py``) from any
# working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Valid model names — mirrors the ``model.models`` list in config.yaml.
# Used for argparse ``choices`` validation.
_VALID_MODELS: list[str] = [
    "xgboost_median",
    "xgboost",
    "linear_regression",
    "ridge",
    "lasso",
    "gradient_boosting",
    "svr",
    "knn",
    "lightgbm",
    "elastic_net",
    "random_forest",
]


# ===========================================================================
# CLI PARSER
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_model",
        description="Insurance ML Training Pipeline — CLI entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full run (all models, MLflow + Optuna):
    python scripts/train_model.py

  Fast iteration — one model, no tracking, no HPO:
    python scripts/train_model.py --models xgboost_median --no-mlflow --no-hpo

  Two specific models with a custom config:
    python scripts/train_model.py --models xgboost_median ridge \\
        --config configs/experiment.yaml

  Validate config + data only (no training):
    python scripts/train_model.py --dry-run

  Debug a failing run:
    python scripts/train_model.py --models ridge --no-hpo --log-level DEBUG
        """,
    )

    # ── Config path ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        metavar="PATH",
        help=("Path to config.yaml. " "Default: <project_root>/configs/config.yaml"),
    )

    # ── Model selection ──────────────────────────────────────────────────────
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=_VALID_MODELS,
        metavar="MODEL",
        default=None,
        help=(
            "One or more models to train. "
            f"Choices: {', '.join(_VALID_MODELS)}. "
            "Default: all models listed in config.yaml ``model.models``."
        ),
    )

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_grp = parser.add_mutually_exclusive_group()
    mlflow_grp.add_argument(
        "--no-mlflow",
        action="store_true",
        default=False,
        help="Disable MLflow tracking (overrides training.enable_mlflow).",
    )
    mlflow_grp.add_argument(
        "--mlflow",
        action="store_true",
        default=False,
        help="Force-enable MLflow tracking (overrides training.enable_mlflow).",
    )

    # ── Optuna HPO ───────────────────────────────────────────────────────────
    hpo_grp = parser.add_mutually_exclusive_group()
    hpo_grp.add_argument(
        "--no-hpo",
        action="store_true",
        default=False,
        help="Disable Optuna HPO (overrides training.enable_optuna).",
    )
    hpo_grp.add_argument(
        "--hpo",
        action="store_true",
        default=False,
        help="Force-enable Optuna HPO (overrides training.enable_optuna).",
    )

    # ── Two-model architecture ────────────────────────────────────────────────
    tma_grp = parser.add_mutually_exclusive_group()
    tma_grp.add_argument(
        "--two-model",
        action="store_true",
        default=False,
        help="Force-enable two-model architecture (xgboost_median + xgboost).",
    )
    tma_grp.add_argument(
        "--no-two-model",
        action="store_true",
        default=False,
        help="Disable two-model architecture; train selected models individually.",
    )

    # ── Output directory ─────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        metavar="DIR",
        help="Model output directory (overrides training.output_dir).",
    )

    # ── Data path ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to raw CSV data file (overrides data.raw_path).",
    )

    # ── Random state ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Override random_state in all config sections " "(defaults, cross_validation, data)."
        ),
    )

    # ── Gates ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-gates",
        action="store_true",
        default=False,
        help=(
            "Relax G3/G6/G7 thresholds so gate failures do NOT sys.exit(1). "
            "Gates still run and log results. "
            "⚠️  Dev/debug use only — never use in production runs."
        ),
    )

    # ── Dry run ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and data, then exit without training.",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        metavar="LEVEL",
        help="Logging verbosity (default: INFO).",
    )

    return parser


# ===========================================================================
# CONFIG OVERRIDE
# ===========================================================================


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """
    Apply CLI argument overrides to a loaded config dict.

    All mutations are in-memory only — config.yaml is never written.
    Each override is logged at INFO so the run is self-documenting.

    Args:
        config: Full config dict returned by ``load_config()``.
        args:   Parsed argparse namespace.

    Returns:
        The same dict, mutated in place and returned for chaining.
    """
    # ── Model list ────────────────────────────────────────────────────────────
    if args.models:
        config["model"]["models"] = args.models
        logger.info("  override  model.models → %s", args.models)

    # ── MLflow ────────────────────────────────────────────────────────────────
    if args.no_mlflow:
        config["training"]["enable_mlflow"] = False
        config["training"]["register_to_mlflow"] = False
        logger.info("  override  training.enable_mlflow → false")
        logger.info("  override  training.register_to_mlflow → false")
    elif args.mlflow:
        config["training"]["enable_mlflow"] = True
        logger.info("  override  training.enable_mlflow → true")

    # ── Optuna HPO ────────────────────────────────────────────────────────────
    if args.no_hpo:
        config["training"]["enable_optuna"] = False
        logger.info("  override  training.enable_optuna → false")
    elif args.hpo:
        config["training"]["enable_optuna"] = True
        logger.info("  override  training.enable_optuna → true")

    # ── Two-model architecture ────────────────────────────────────────────────
    if args.two_model:
        config["training"].setdefault("two_model_architecture", {})["enabled"] = True
        logger.info("  override  training.two_model_architecture.enabled → true")
    elif args.no_two_model:
        config["training"].setdefault("two_model_architecture", {})["enabled"] = False
        logger.info("  override  training.two_model_architecture.enabled → false")

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        config["training"]["output_dir"] = str(args.output_dir)
        logger.info("  override  training.output_dir → %s", args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data path ─────────────────────────────────────────────────────────────
    if args.data:
        if not args.data.exists():
            logger.error("  ❌ --data path not found: %s", args.data)
            sys.exit(1)
        config["data"]["raw_path"] = str(args.data)
        logger.info("  override  data.raw_path → %s", args.data)

    # ── Random state ──────────────────────────────────────────────────────────
    if args.random_state is not None:
        rs = args.random_state
        for section in ("defaults", "cross_validation", "data"):
            config.setdefault(section, {})["random_state"] = rs
        logger.info("  override  random_state → %d (all sections)", rs)

    # ── Gates: relax thresholds so the pipeline doesn't sys.exit in dev mode ─
    # G3, G6, and G7 still run and log — we only widen the thresholds so they
    # can never trigger a hard failure.
    if args.no_gates:
        gates = config["training"].setdefault("deployment_gates", {})
        gates["g6_min_cost_weighted_r2"] = -999.0  # effectively disabled
        gates["g7_max_overpricing_rate"] = 1.0  # 100% — can never exceed
        gates["g3_max_width_ratio"] = 999.0  # unbounded — can never exceed
        logger.warning(
            "  ⚠️  --no-gates active: "
            "G3 threshold → 999.0, G6 threshold → -999.0, G7 threshold → 1.0. "
            "Gates will NOT block deployment. For dev/debug only."
        )

    return config


# ===========================================================================
# DRY RUN VALIDATION
# ===========================================================================


def validate_config_and_data(config: dict[str, Any]) -> bool:
    """
    Lightweight pre-flight check for ``--dry-run``.

    Checks required config sections, the data CSV, the target column, and GPU
    availability.  Returns True if everything looks OK, False otherwise.
    """
    import pandas as pd

    ok = True

    # Required top-level sections — must stay in sync with
    # config.py _validate_config() required_sections (v7.5.0).
    required_sections = [
        "data",
        "model",
        "training",
        "features",
        "gpu",
        "cross_validation",
        "defaults",
        "models",
        "optuna",
        "diagnostics",
        "mlflow",
        "logging",
        "high_value_analysis",
        "overfitting_analysis",
        "monitoring",
        "validation",
        "hardware",
        "sample_weights",
        "prediction",  # v7.5.0: batch size cap
        "conformal",  # v7.5.0: calibration split ratio
    ]
    for section in required_sections:
        if section not in config:
            logger.error("  ❌ Missing config section: '%s'", section)
            ok = False
        else:
            logger.info("  ✅ Config section '%s' present", section)

    # Data file
    raw_path = Path(config.get("data", {}).get("raw_path", ""))
    if not raw_path.is_absolute():
        raw_path = _PROJECT_ROOT / raw_path
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        target = config["data"].get("target_column", "charges")
        logger.info(
            "  ✅ Data: %s  (%d rows × %d cols)",
            raw_path.name,
            len(df),
            df.shape[1],
        )
        if target not in df.columns:
            logger.error("  ❌ Target column '%s' not found in data", target)
            ok = False
        else:
            logger.info("  ✅ Target column '%s' present", target)
        # Spot-check for expected feature columns
        expected_features = config.get("features", {}).get("numerical_features", [])
        missing_feats = [f for f in expected_features if f not in df.columns]
        if missing_feats:
            logger.warning("  ⚠️  Numerical features missing from data: %s", missing_feats)
        else:
            logger.info("  ✅ Numerical features present: %s", expected_features)
    else:
        logger.error("  ❌ Data file not found: %s", raw_path)
        ok = False

    # Output directory (informational only)
    output_dir = Path(config.get("training", {}).get("output_dir", "models"))
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    logger.info("  ℹ️  Model output dir: %s", output_dir)

    # GPU availability
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_mb = props.total_memory / 1024**2
            logger.info("  ✅ GPU: %s  (%.0f MB VRAM)", props.name, vram_mb)
            cfg_limit = config.get("gpu", {}).get("memory_limit_mb", 0)
            if vram_mb < cfg_limit:
                logger.warning(
                    "  ⚠️  gpu.memory_limit_mb (%d MB) exceeds available VRAM (%.0f MB)",
                    cfg_limit,
                    vram_mb,
                )
        else:
            logger.warning("  ⚠️  No CUDA GPU detected — " "XGBoost/LightGBM will fall back to CPU")
    except ImportError:
        logger.warning("  ⚠️  PyTorch not installed — cannot verify GPU")

    # MLflow tracking URI (informational only)
    mlflow_enabled = config.get("training", {}).get("enable_mlflow", True)
    logger.info("  ℹ️  MLflow tracking: %s", "enabled" if mlflow_enabled else "disabled")

    # Optuna HPO (informational only)
    hpo_enabled = config.get("training", {}).get("enable_optuna", True)
    logger.info("  ℹ️  Optuna HPO: %s", "enabled" if hpo_enabled else "disabled")

    return ok


# ===========================================================================
# PIPELINE EXECUTION
# ===========================================================================


def _inject_config_and_run(modified_config: dict[str, Any]) -> None:
    """
    Monkey-patch ``insurance_ml.config.load_config`` to return *modified_config*,
    then call ``insurance_ml.train.main()``.

    Why monkey-patching?
    ``ModelTrainer.__init__`` does ``from insurance_ml.config import load_config``
    at construction time (not at module-import time), so replacing the attribute
    on the already-imported module object is picked up cleanly.  The original
    function is restored in the ``finally`` block regardless of success/failure.

    Args:
        modified_config: Config dict with CLI overrides already applied.
    """
    import insurance_ml.config as _cfg_mod

    _original = _cfg_mod.load_config

    def _patched(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        return modified_config

    _cfg_mod.load_config = _patched
    try:
        # Import AFTER patching so that the first ModelTrainer() construction
        # inside main() picks up the patch via its own `from ... import`.
        from insurance_ml.train import main as _train_main  # noqa: PLC0415

        _train_main()
    finally:
        _cfg_mod.load_config = _original


# ===========================================================================
# LOGGING SETUP
# ===========================================================================


def _setup_logging(level: str) -> None:
    """Configure the root logger with a timestamped format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)-5s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Windows: stdout/stderr default to cp1252; reconfigure for UTF-8 so
    # emoji status characters from the pipeline are rendered correctly.
    if platform.system() == "Windows":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass  # Python < 3.7 or non-reconfigurable stream


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Insurance ML — Training CLI")
    print(f"{'='*70}\n")

    wall_start = time.time()

    # ── Load config ───────────────────────────────────────────────────────────
    from insurance_ml.config import load_config, validate_single_source_of_truth

    config_path_arg = str(args.config) if args.config else None
    logger.info(
        "Loading config: %s",
        args.config or "(default) configs/config.yaml",
    )
    try:
        config = load_config(config_path_arg)
    except FileNotFoundError as exc:
        logger.error("❌ Config not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("❌ Invalid config: %s", exc)
        sys.exit(1)

    logger.info("✅ Config loaded (version: %s)", config.get("version", "unknown"))

    # ── Determine whether any override is requested ───────────────────────────
    has_overrides: bool = any(
        [
            args.models,
            args.no_mlflow,
            args.mlflow,
            args.no_hpo,
            args.hpo,
            args.output_dir is not None,
            args.data is not None,
            args.two_model,
            args.no_two_model,
            args.random_state is not None,
            args.no_gates,
        ]
    )

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    if has_overrides:
        logger.info("Applying CLI overrides:")
        config = apply_overrides(config, args)
        # Re-validate single-source-of-truth after mutations.
        # (load_config() already called this on the original; re-run to
        # catch any override that accidentally introduces redundancy.)
        try:
            validate_single_source_of_truth(config)
        except ValueError as exc:
            logger.error("❌ Config consistency check failed after overrides: %s", exc)
            sys.exit(1)
        logger.info("✅ Overrides applied and validated")

    # ── Run summary ───────────────────────────────────────────────────────────
    _models = config.get("model", {}).get("models", _VALID_MODELS)
    _mlflow = config.get("training", {}).get("enable_mlflow", True)
    _hpo = config.get("training", {}).get("enable_optuna", True)
    _tma = config.get("training", {}).get("two_model_architecture", {}).get("enabled", False)
    _out = config.get("training", {}).get("output_dir", "models/")
    _data = config.get("data", {}).get("raw_path", "N/A")

    print(f"  Config     : {args.config or 'configs/config.yaml (default)'}")
    print(f"  Data       : {_data}")
    print(f"  Models     : {_models}")
    print(f"  MLflow     : {'enabled' if _mlflow else 'disabled'}")
    print(f"  Optuna HPO : {'enabled' if _hpo else 'disabled'}")
    print(f"  Two-model  : {'enabled' if _tma else 'disabled'}")
    print(f"  Output     : {_out}")
    if args.no_gates:
        print("  Gates      : ⚠️  DISABLED (dev mode — thresholds relaxed)")
    print()

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"{'='*70}")
        print("  DRY RUN — validating config and data (no training)")
        print(f"{'='*70}\n")

        ok = validate_config_and_data(config)
        elapsed = time.time() - wall_start

        if ok:
            print(f"\n✅ Dry run passed in {elapsed:.1f}s — config and data look good")
            sys.exit(0)
        else:
            print("\n❌ Dry run found issues — fix before training")
            sys.exit(1)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print("  Starting training pipeline …")
    print(f"{'='*70}\n")

    try:
        if has_overrides:
            # Pass modified config to main() via the monkey-patch approach
            # described in _inject_config_and_run().
            _inject_config_and_run(config)
        else:
            # No overrides at all — just call main() directly.
            # This is identical to running ``python -m insurance_ml.train``.
            from insurance_ml.train import main as _train_main  # noqa: PLC0415

            _train_main()

    except SystemExit as exc:
        # Propagate gate failures (sys.exit(1)) from train.main() cleanly.
        sys.exit(exc.code)
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "❌ Pipeline failed: %s",
            exc,
            exc_info=(args.log_level == "DEBUG"),
        )
        sys.exit(1)

    elapsed = time.time() - wall_start
    logger.info("✅ Total wall time: %.1fs", elapsed)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
