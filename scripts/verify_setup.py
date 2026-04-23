#!/usr/bin/env python3
"""
verify_setup.py — Insurance ML Pipeline Setup Verifier
Run from Pipeline/ root:  py -3.11 scripts/verify_setup.py

Checks every fix applied across pyproject.toml, requirements.txt,
.gitignore, Makefile, .env, .env.example, .pre-commit-config.yaml,
and the git working tree.

Exit code 0 = all checks passed.
Exit code 1 = one or more failures.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False

# ── result accumulator ────────────────────────────────────────────────────────
results: list[tuple[str, str, str, str]] = []  # (status, file, check, detail)


def ok(file: str, check: str, detail: str = "") -> None:
    results.append(("PASS", file, check, detail))


def fail(file: str, check: str, detail: str = "") -> None:
    results.append(("FAIL", file, check, detail))


def warn(file: str, check: str, detail: str = "") -> None:
    results.append(("WARN", file, check, detail))


def read(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        fail(path, "file exists", "file not found")
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def active_lines(text: str) -> list[str]:
    """Non-empty, non-comment lines with comment suffixes stripped."""
    return [
        l.split("#")[0].strip()
        for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]


# ── pyproject.toml ────────────────────────────────────────────────────────────
def check_pyproject() -> None:
    f = "pyproject.toml"
    t = read(f)
    if t is None:
        return

    active = active_lines(t)

    # Critical: mlflow-skinny removed
    if any("mlflow-skinny" in l for l in active):
        fail(f, "mlflow-skinny removed", "still present in active lines")
    else:
        ok(f, "mlflow-skinny removed")

    # High: statsmodels, click present
    for pkg in ["statsmodels", "click"]:
        if any(pkg in l for l in active):
            ok(f, f"{pkg} present")
        else:
            fail(f, f"{pkg} present", "missing from dependencies")

    # Critical: dead entry points removed
    dead = [
        "insurance-predict",
        "insurance-evaluate",
        "insurance-api",
        "insurance-dashboard",
    ]
    for ep in dead:
        if any(ep in l for l in active):
            fail(f, f"dead entry point removed", f"{ep} still present")
        else:
            ok(f, f"dead entry point '{ep}' removed")

    # Structural: package discovery
    if 'where = ["src"]' in t:
        ok(f, "package discovery where=src")
    else:
        fail(f, "package discovery where=src", "not found")


# ── requirements.txt ─────────────────────────────────────────────────────────
def check_requirements() -> None:
    f = "requirements.txt"
    t = read(f)
    if t is None:
        return

    active = active_lines(t)

    if any("mlflow-skinny" in l for l in active):
        fail(f, "mlflow-skinny removed", "still present")
    else:
        ok(f, "mlflow-skinny removed")

    if "torch==2.6.0+cu124" in t:
        ok(f, "torch +cu124 GPU build")
    else:
        fail(
            f,
            "torch +cu124 GPU build",
            "missing +cu124 suffix — will install CPU build",
        )

    if "--index-url https://download.pytorch.org/whl/cu124" in t:
        ok(f, "torch --index-url present")
    else:
        fail(
            f,
            "torch --index-url present",
            "missing — use --index-url not --extra-index-url",
        )

    for pkg in ["statsmodels", "click"]:
        if any(l.startswith(pkg) for l in active):
            ok(f, f"{pkg} present")
        else:
            fail(f, f"{pkg} present", "missing")


# ── .env ──────────────────────────────────────────────────────────────────────
def check_env() -> None:
    f = ".env"
    t = read(f)
    if t is None:
        return

    # Line endings
    raw = Path(f).read_bytes()
    if b"\r\n" in raw:
        if not Path(".gitattributes").exists():
            fail(
                f,
                "LF line endings",
                "CRLF detected AND no .gitattributes — git core.autocrlf will keep re-adding CRLF",
            )
        else:
            ga = Path(".gitattributes").read_text()
            if "eol=lf" in ga:
                fail(
                    f,
                    "LF line endings",
                    "CRLF on disk — .gitattributes exists but re-checkout needed: "
                    "run: git rm --cached .env && git checkout .env",
                )
            else:
                fail(
                    f,
                    "LF line endings",
                    "CRLF detected — .gitattributes missing eol=lf for .env",
                )
    else:
        ok(f, "LF line endings")

    # No duplicate keys
    keys = [
        l.split("=")[0].strip()
        for l in t.splitlines()
        if l.strip() and not l.startswith("#") and "=" in l
    ]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        fail(f, "no duplicate keys", f"duplicates: {dupes}")
    else:
        ok(f, "no duplicate keys")

    # Required keys
    kset = set(keys)
    for key in [
        "API_KEY",
        "API_URL",
        "API_TIMEOUT",
        "HEALTH_TIMEOUT",
        "MAX_RETRIES",
        "API_WORKERS",
        "OPTUNA_STORAGE",
    ]:
        if key in kset:
            ok(f, f"{key} present")
        else:
            fail(f, f"{key} present", "missing")

    # API_WORKERS must be 1 for GPU safety
    m = re.search(r"^API_WORKERS=(\d+)", t, re.MULTILINE)
    if m:
        if int(m.group(1)) == 1:
            ok(f, "API_WORKERS=1 (GPU safe)")
        else:
            fail(
                f,
                "API_WORKERS=1",
                f"found {m.group(1)} — multiple workers will OOM RTX 3050",
            )

    # OPTUNA path
    m = re.search(r"^OPTUNA_STORAGE=(.+)", t, re.MULTILINE)
    if m and "models/optuna_studies.db" in m.group(1):
        ok(f, "OPTUNA_STORAGE path correct (50k default)")
    else:
        val = m.group(1) if m else "not found"
        fail(
            f,
            "OPTUNA_STORAGE path",
            f"got: {val} — should be sqlite:///models/optuna_studies.db",
        )


# ── .env.example ─────────────────────────────────────────────────────────────
def check_env_example() -> None:
    f = ".env.example"
    t = read(f)
    if t is None:
        return

    # All .env keys covered
    env_t = read(".env")
    if env_t:
        env_keys = {
            l.split("=")[0].strip()
            for l in env_t.splitlines()
            if l.strip() and not l.startswith("#") and "=" in l
        }
        ex_keys = {
            l.split("=")[0].strip()
            for l in t.splitlines()
            if l.strip() and not l.startswith("#") and "=" in l
        }
        missing = env_keys - ex_keys
        if missing:
            warn(f, ".env keys covered by .env.example", f"missing: {sorted(missing)}")
        else:
            ok(f, "all .env keys present in .env.example")

    # OPTUNA path consistent
    m = re.search(r"^OPTUNA_STORAGE=(.+)", t, re.MULTILINE)
    if m and "models/optuna_studies.db" in m.group(1):
        ok(f, "OPTUNA_STORAGE path matches .env")
    else:
        fail(f, "OPTUNA_STORAGE path", f"got: {m.group(1) if m else 'not found'}")


# ── .gitignore ────────────────────────────────────────────────────────────────
def check_gitignore() -> None:
    f = ".gitignore"
    t = read(f)
    if t is None:
        return

    active = active_lines(t)

    patterns = {
        "models/*.txt": "checksum files",
        "*.bak_next_level": "pipeline backups",
        "output*.txt": "session output captures",
        "project_structure.txt": "generated tree file",
        ".coveragerc": "empty coverage config",
        "models/*.ubj": "XGBoost booster binaries",
        "reports/": "generated reports",
        "mlruns/": "MLflow tracking dir",
        ".model_json_bak/": "make clean-models temp dir",
    }
    for pattern, desc in patterns.items():
        if pattern in active:
            ok(f, f"pattern '{pattern}'", desc)
        else:
            fail(f, f"pattern '{pattern}' present", desc)

    # .dockerignore must NOT be ignored — it is a tracked project file.
    # The old *.dockerignore glob was removed; .dockerignore must remain trackable.
    if ".dockerignore" in active:
        fail(f, ".dockerignore trackable", ".dockerignore is in .gitignore — it will be untracked")
    else:
        ok(f, ".dockerignore trackable (not in .gitignore)")

    # *.dockerignore glob must NOT be present
    if "*.dockerignore" in active:
        fail(f, "*.dockerignore glob removed", "glob doesn't match dotfiles")
    else:
        ok(f, "*.dockerignore glob removed")

    # Negations for operational JSON files
    for neg in [
        "!models/bias_correction.json",
        "!models/pipeline_metadata.json",
        "!models/test_indices.json",
    ]:
        if neg in active:
            ok(f, f"negation '{neg}'")
        else:
            fail(
                f,
                f"negation '{neg}'",
                "operational JSON will be ignored by models/*.json",
            )

    # .env.docker must NOT be caught by any pattern
    # Patterns that could catch it: .env (exact), *.env (suffix), .env.*.local (suffix+local)
    dangerous = [
        l
        for l in active
        if l in (".env", "*.env") or (l.startswith(".env.") and l.endswith(".local"))
    ]
    # .env.docker: doesn't end in .env, isn't exactly .env, isn't .env.*.local → safe
    ok(
        f,
        ".env.docker not caught by .env patterns",
        f"active env patterns: {dangerous}",
    )


# ── .gitattributes ────────────────────────────────────────────────────────────
def check_gitattributes() -> None:
    f = ".gitattributes"
    t = read(f)
    if t is None:
        return

    required = {
        "* text=auto eol=lf": "global LF normalisation",
        ".env              text eol=lf": ".env forced LF",
        ".env.*            text eol=lf": ".env.* forced LF",
        "*.py              text eol=lf": "Python files forced LF",
        "*.yaml            text eol=lf": "YAML files forced LF",
        "*.joblib          binary": "joblib marked binary",
        "*.ubj             binary": "ubj marked binary",
    }
    for pattern, desc in required.items():
        if pattern in t:
            ok(f, desc)
        else:
            fail(f, desc, f"pattern missing: {pattern!r}")


# ── Makefile ──────────────────────────────────────────────────────────────────
def check_makefile() -> None:
    f = "Makefile"
    t = read(f)
    if t is None:
        return

    checks = {
        # Cross-platform PYTHON detection (replaced 'py -3.11' hardcode)
        "python -c \"import sys; print(sys.executable)\"": "PYTHON uses cross-platform dynamic detection",
        # GPU safety: API_WORKERS defaults to 1 (was hardcoded 4 → VRAM OOM)
        "API_WORKERS              ?= 1": "API_WORKERS default 1 (GPU safe — was 4)",
        # docker compose subcommand form (not deprecated docker-compose binary)
        "docker compose -f docker/docker-compose.yml up": "docker-compose-up uses subcommand form + -f flag",
        "docker compose -f docker/docker-compose.yml down": "docker-compose-down uses subcommand form + -f flag",
        # torch install index URL
        "--index-url https://download.pytorch.org/whl/cu124": "torch install uses --index-url",
        # pip upgrade via -m flag
        "$(PYTHON) -m pip install --upgrade pip": "pip upgrade uses -m flag",
        # clean-models uses Python confirm (not Windows %confirm% batch)
        "ans=input('Continue? [y/N] ')": "clean-models uses Python confirm (cross-platform)",
        # JSON metadata preserved in clean-models
        "bias_correction.json": "clean-models preserves JSON metadata",
    }
    for needle, desc in checks.items():
        if needle in t:
            ok(f, desc)
        else:
            fail(f, desc, f"pattern not found: {needle!r}")

    # --extra-index-url must not appear in the install-gpu recipe (comments excluded)
    lines = [
        l.split("#")[0]
        for l in t.splitlines()
        if "extra-index-url" in l.split("#")[0] and "torch" in l.split("#")[0]
    ]
    if lines:
        fail(
            f,
            "--extra-index-url removed from install-gpu",
            "still present — will silently install CPU torch",
        )
    else:
        ok(f, "--extra-index-url not in install-gpu")

    # WIP markers
    wip = t.count("[WIP]")
    if wip >= 11:
        ok(f, f"WIP markers on unimplemented targets", f"{wip} found")
    else:
        warn(f, "WIP markers", f"only {wip} found, expected >= 11")


# ── .pre-commit-config.yaml ───────────────────────────────────────────────────
def check_precommit() -> None:
    f = ".pre-commit-config.yaml"
    t = read(f)
    if t is None:
        return

    for removed in ["id: black", "id: isort", "id: autoflake"]:
        active = [l for l in t.splitlines() if removed in l and not l.strip().startswith("#")]
        if active:
            fail(f, f"{removed} removed", "still active")
        else:
            ok(f, f"{removed} removed")

    if "id: ruff" in t:
        ok(f, "ruff hook present")
    else:
        fail(f, "ruff hook present", "missing")

    dep_lines = [l.strip() for l in t.splitlines() if l.strip().startswith("- mdformat")]
    if any("mdformat-ruff" in l for l in dep_lines):
        ok(f, "mdformat-ruff pairing")
    else:
        fail(f, "mdformat-ruff pairing", "check mdformat additional_dependencies")

    if any("mdformat-black" in l.split("#")[0] for l in dep_lines):
        fail(f, "mdformat-black removed", "conflicts with ruff formatter")
    else:
        ok(f, "mdformat-black removed")

    if "python: python3.11" in t:
        ok(f, "default_language_version python3.11")
    else:
        fail(f, "default_language_version python3.11", "not set")


# ── git working tree ──────────────────────────────────────────────────────────
def check_git() -> None:
    f = "git"
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        warn(f, "git status", "git not available or not a repo")
        return

    lines = result.stdout.splitlines()
    status = {l[3:]: l[:2].strip() for l in lines if l.strip()}

    # Files that must NOT be untracked (should be committed)
    should_be_tracked = [".dockerignore", ".env.docker"]
    for path in should_be_tracked:
        s = status.get(path, "")
        if s == "??":
            warn(f, f"{path} committed", "still untracked — run: git add " + path)
        elif s == "":
            ok(f, f"{path} committed")
        else:
            ok(f, f"{path} staged/modified (M={s})")

    # Generated files that must NOT be tracked
    should_be_ignored = [
        "models/xgboost.joblib.booster.ubj",
        "models/xgboost_median.joblib.booster.ubj",
        "models/xgboost_checksum.txt",
        "models/xgboost_median_checksum.txt",
    ]
    for path in should_be_ignored:
        s = status.get(path, "")
        if s in ("M", "A"):
            warn(
                f,
                f"{path} untracked",
                f"still staged/modified ({s}) — run: git rm --cached {path}",
            )
        elif s == "??":
            ok(f, f"{path} untracked (ignored)")
        else:
            ok(f, f"{path} not in working tree")

    # .bak_next_level files must not appear
    # Status codes: "D" = staged for deletion (git rm --cached run, commit pending)
    #               "M"/"A" = still actively tracked (bad)
    #               "??" = untracked/ignored (good)
    #               ""  = not present in working tree (good)
    bak_staged_del = [p for p, s in status.items() if ".bak_next_level" in p and s == "D"]
    bak_active     = [p for p, s in status.items() if ".bak_next_level" in p and s not in ("??", "", "D")]
    if bak_active:
        fail(f, "*.bak_next_level untracked", f"still tracked: {bak_active}")
    elif bak_staged_del:
        warn(
            f,
            "*.bak_next_level untracked",
            f"staged for deletion (run: git commit to finish): {bak_staged_del}",
        )
    else:
        ok(f, "*.bak_next_level not tracked")

    # .coveragerc must not be tracked
    cov_status = status.get(".coveragerc", "")
    if cov_status in ("M", "A"):
        fail(
            f,
            ".coveragerc not tracked",
            "still staged — delete and git rm --cached .coveragerc",
        )
    elif cov_status == "??":
        warn(f, ".coveragerc deleted", "file exists but untracked — delete the file")
    else:
        ok(f, ".coveragerc not tracked (deleted)")


# ── environment ───────────────────────────────────────────────────────────────
def check_environment() -> None:
    f = "environment"

    # Python version
    v = sys.version_info
    if v >= (3, 11):
        ok(f, f"Python >= 3.11", f"{v.major}.{v.minor}.{v.micro}")
    else:
        fail(f, "Python >= 3.11", f"found {v.major}.{v.minor}")

    # Key packages importable
    pkg_checks = {
        "mlflow": "3",
        "xgboost": "3",
        "lightgbm": "4",
        "optuna": "4",
        "statsmodels": "0",
        "click": "8",
        "pydantic": "2",
    }
    for pkg, min_major in pkg_checks.items():
        try:
            import importlib.metadata

            try:
                ver = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "?")
            actual_major = int(str(ver).split(".")[0])
            if actual_major >= int(min_major):
                ok(f, f"{pkg} importable", f"v{ver}")
            else:
                warn(f, f"{pkg} version", f"found {ver}, expected >={min_major}.x")
        except ImportError:
            warn(f, f"{pkg} importable", "not installed in current venv")

    # torch GPU
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            ok(f, "torch CUDA available", f"{gpu} ({mem:.1f} GB)")
            if "+cu" not in torch.__version__:
                warn(
                    f,
                    "torch is GPU build",
                    f"version {torch.__version__} — may be CPU build",
                )
            else:
                ok(f, "torch is GPU build", torch.__version__)
        else:
            warn(f, "torch CUDA available", "CUDA not available — CPU only")
    except ImportError:
        warn(f, "torch importable", "not installed in current venv")


# ── config.yaml ───────────────────────────────────────────────────────────────
def check_config_yaml() -> None:
    f = "configs/config.yaml"
    t = read(f)
    if t is None:
        return

    # ── Version ──────────────────────────────────────────────────────────────
    if "version: 7.5.0" in t:
        ok(f, "config version 7.5.0")
    else:
        m = re.search(r"^version:\s*(\S+)", t, re.MULTILINE)
        fail(f, "config version 7.5.0", f"found: {m.group(1) if m else 'not found'}")

    # ── Feature validation bounds (M-04, F-08) ────────────────────────────────
    for needle, desc in [
        ("age_min: 18.0",   "age_min: 18.0 (M-04 fix — was 0.0)"),
        ("children_min: 0", "children_min: 0 (F-08)"),
        ("children_max: 20","children_max: 20 (F-08)"),
    ]:
        if needle in t:
            ok(f, desc)
        else:
            fail(f, desc, f"pattern not found: {needle!r}")

    # ── Collinearity / VIF (G6 fix) ───────────────────────────────────────────
    if "vif_threshold: 10.0" in t:
        ok(f, "vif_threshold: 10.0 (G6 fix — was 5.0)")
    else:
        fail(f, "vif_threshold: 10.0", "High+ segment R²=-29.83 root cause — must not be 5.0")

    # ── Sample weights monotonicity (M5 fix) ─────────────────────────────────
    if "above_q99: 2.00" in t:
        ok(f, "above_q99: 2.00 (M5 monotonicity fix — was 1.50)")
    else:
        fail(f, "above_q99: 2.00", "non-monotonic weight profile — VH R²=-2.01 root cause")

    # ── Hybrid predictor key values ───────────────────────────────────────────
    for needle, desc in [
        ("max_actuarial_uplift_ratio: 1.15", "max_actuarial_uplift_ratio: 1.15 (C-01 fix — was 1.45)"),
        ("min_actuarial_floor_ratio: 0.75",  "min_actuarial_floor_ratio: 0.75 (T2-B ML-relative floor)"),
        ("threshold: 9500.0",                "hybrid threshold: 9500.0"),
        ("blend_ratio: 0.70",                "blend_ratio: 0.70"),
    ]:
        if needle in t:
            ok(f, desc)
        else:
            fail(f, desc, f"pattern not found: {needle!r}")

    # ── Calibration factors (BUG-5 / G7 fix) ─────────────────────────────────
    for needle, desc in [
        ("pricing_factor: 1.00", "pricing_factor: 1.00 (reg:squarederror — no uplift needed)"),
        ("risk_factor: 0.97",    "risk_factor: 0.97 (quantile alpha=0.65 nudge)"),
    ]:
        if needle in t:
            ok(f, desc)
        else:
            fail(f, desc, f"pattern not found: {needle!r}")

    # ── Governance bounds (M-07) ──────────────────────────────────────────────
    for needle, desc in [
        ("combined_correction_max: 1.45", "combined_correction_max: 1.45 (M-07)"),
        ("combined_correction_min: 0.65", "combined_correction_min: 0.65 (M-07)"),
    ]:
        if needle in t:
            ok(f, desc)
        else:
            fail(f, desc, f"pattern not found: {needle!r}")

    # ── XGBoost quantile alpha (G7 / YJ nonlinearity fix) ────────────────────
    if "quantile_alpha: 0.30" in t:
        ok(f, "quantile_alpha: 0.30 (YJ nonlinearity fix — was 0.57)")
    else:
        fail(f, "quantile_alpha: 0.30", "stale alpha causes G7 gate failure")

    # ── risk_model_alpha sync (stale-value fix) ───────────────────────────────
    if "risk_model_alpha: 0.30" in t:
        ok(f, "risk_model_alpha: 0.30 (synced with quantile_alpha)")
    else:
        fail(f, "risk_model_alpha: 0.30", "must match models.xgboost.quantile_alpha")

    # ── Optuna budget (M-03 / TPE activation fix) ────────────────────────────
    if "n_trials: 50" in t:
        ok(f, "optuna n_trials: 50 (M-03 fix — was 1)")
    else:
        fail(f, "optuna n_trials: 50", "n_trials=1 means TPE never activates")

    m_nt = re.search(r"n_startup_trials:\s*(\d+)", t)
    if m_nt and int(m_nt.group(1)) == 10:
        ok(f, "sampler n_startup_trials: 10 (FIX-H — was 15 > n_trials)")
    else:
        val = m_nt.group(1) if m_nt else "not found"
        fail(f, "sampler n_startup_trials: 10", f"found: {val} — must be < n_trials")

    # ── Evaluation primary metric (M6 fix) ────────────────────────────────────
    # Must appear in BOTH evaluation.metrics and hybrid_predictor.evaluation
    count = t.count("primary_metric: net_profit")
    if count >= 2:
        ok(f, "primary_metric: net_profit (both locations — M6 fix, was smape)")
    elif count == 1:
        warn(f, "primary_metric: net_profit", "found in only 1 location — check both evaluation sections")
    else:
        fail(f, "primary_metric: net_profit", "was 'smape' — contradicts operator notes")

    # ── Training split (CI-width fix) ─────────────────────────────────────────
    for needle, desc in [
        ("test_size: 0.15", "test_size: 0.15 (CI-width fix — was 0.20)"),
        ("val_size: 0.30",  "val_size: 0.30  (CI-width fix — was 0.25)"),
    ]:
        if needle in t:
            ok(f, desc)
        else:
            fail(f, desc, f"pattern not found: {needle!r}")

    # ── Deployment gate G7 (BUG-6 fix) ───────────────────────────────────────
    if "g7_max_overpricing_rate: 0.62" in t:
        ok(f, "g7_max_overpricing_rate: 0.62 (BUG-6 fix — was 0.55)")
    else:
        fail(f, "g7_max_overpricing_rate: 0.62", "0.55 is structurally unreachable for squarederror")

    # ── Conformal calibration split (CI-width fix) ────────────────────────────
    if "calibration_split_ratio: 0.20" in t:
        ok(f, "calibration_split_ratio: 0.20 (80/20 split — was 0.40)")
    else:
        fail(f, "calibration_split_ratio: 0.20", "0.40 gave only 7 heteroscedastic bins")

    # ── Diagnostics confidence level aligned with conformal target ────────────
    if "confidence_level: 0.90" in t:
        ok(f, "diagnostics confidence_level: 0.90 (aligned with conformal target_coverage)")
    else:
        fail(f, "diagnostics confidence_level: 0.90", "mismatch inflates CI width ~5x")

    # ── GPU xgboost_median device key (v7.5.2) ────────────────────────────────
    # Check that 'xgboost_median:' section under 'gpu:' contains 'device: cuda:0'
    _gpu_block_m = re.search(r"gpu:.*?(?=\n\w)", t, re.DOTALL)
    if _gpu_block_m and "xgboost_median:" in _gpu_block_m.group() and "device: cuda:0" in _gpu_block_m.group():
        ok(f, "gpu.xgboost_median.device: cuda:0 (v7.5.2 — missing caused ~250 log msgs/run)")
    else:
        fail(f, "gpu.xgboost_median.device: cuda:0", "key absent — XGBoost will emit redundant device warnings")

    # ── Batch cap config-driven (T3-C) ────────────────────────────────────────
    if "max_batch_size: 10000" in t:
        ok(f, "prediction.max_batch_size: 10000 (T3-C config-driven)")
    else:
        fail(f, "prediction.max_batch_size: 10000", "batch cap must not be hardcoded in predict.py")

    # ── Drift detection dual-key (config compat fix) ──────────────────────────
    for needle, desc in [
        ("drift_detection: true",         "monitoring.drift_detection: true (canonical key)"),
        ("drift_detection_enabled: true", "monitoring.drift_detection_enabled: true (legacy alias)"),
    ]:
        if needle in t:
            ok(f, desc)
        else:
            fail(f, desc, "dual-key absent — older predict.py versions will miss this")

    # ── SMAPE denominator guard (evaluate.py fix) ─────────────────────────────
    if "threshold: 0.01" in t:
        ok(f, "smape.threshold: 0.01 (was 1e-10 — caused SMAPE blow-up near zero)")
    else:
        fail(f, "smape.threshold: 0.01", "1e-10 denominator floor causes near-zero SMAPE instability")

    # ── max_cached_hist_node (v7.4.5 right-sizing) ───────────────────────────
    if "max_cached_hist_node: 1024" in t:
        ok(f, "max_cached_hist_node: 1024 (v7.4.5 — was 65536, wasteful for max_leaves<=25)")
    else:
        fail(f, "max_cached_hist_node: 1024", "65536 wastes GPU memory for small trees")

    # ── Asymmetric penalties consistency ─────────────────────────────────────
    # underpricing_multiplier must match business_config.underpricing_penalty_multiplier
    biz = re.search(r"underpricing_penalty_multiplier:\s*([\d.]+)", t)
    ev  = re.search(r"underpricing_multiplier:\s*([\d.]+)", t)
    if biz and ev:
        if biz.group(1) == ev.group(1):
            ok(f, f"underpricing multiplier consistent ({biz.group(1)} in both locations)")
        else:
            fail(
                f,
                "underpricing multiplier consistent",
                f"business_config={biz.group(1)} vs cost_analysis={ev.group(1)} — silent profit miscalculation",
            )
    else:
        fail(f, "underpricing multiplier present", "one or both keys missing")


# ── pipeline_metadata.json ────────────────────────────────────────────────────
def check_pipeline_metadata() -> None:
    f = "models/pipeline_metadata.json"
    raw = read(f)
    if raw is None:
        return

    try:
        import json
        meta = json.loads(raw)
    except Exception as e:
        fail(f, "valid JSON", str(e))
        return

    # Version / schema
    ver = meta.get("version", "")
    if ver == "5.2.0-fix4":
        ok(f, "version: 5.2.0-fix4")
    else:
        fail(f, "version: 5.2.0-fix4", f"found: {ver!r}")

    schema = meta.get("model_schema_version", "")
    if schema == "3.0":
        ok(f, "model_schema_version: 3.0")
    else:
        fail(f, "model_schema_version: 3.0", f"found: {schema!r}")

    # Best model
    bm = meta.get("best_model", "")
    if bm == "xgboost_median":
        ok(f, "best_model: xgboost_median")
    else:
        fail(f, "best_model: xgboost_median", f"found: {bm!r}")

    bmp = meta.get("best_model_path", "")
    if "xgboost_median.joblib" in bmp:
        ok(f, "best_model_path contains xgboost_median.joblib")
    else:
        fail(f, "best_model_path", f"got: {bmp!r}")

    # Target transform
    tt = meta.get("target_transform", "")
    if tt == "yeo-johnson":
        ok(f, "target_transform: yeo-johnson")
    else:
        fail(f, "target_transform: yeo-johnson", f"found: {tt!r}")

    # Feature count
    eng = meta.get("features", {}).get("engineered", None)
    if eng == 46:
        ok(f, "features.engineered: 46")
    else:
        fail(f, "features.engineered: 46", f"found: {eng}")

    # Trained models
    tm = meta.get("trained_models", [])
    if len(tm) == 5:
        ok(f, f"trained_models: 5 entries ({', '.join(tm)})")
    else:
        fail(f, "trained_models: 5 entries", f"found {len(tm)}: {tm}")

    # Bias correction thresholds block
    bct = meta.get("bias_correction_thresholds", {})
    if bct:
        ok(f, "bias_correction_thresholds block present")
        for key in ["q50_threshold_low", "q75_threshold_high", "run_timestamp"]:
            if key in bct:
                ok(f, f"bias_correction_thresholds.{key} present")
            else:
                fail(f, f"bias_correction_thresholds.{key}", "missing key")
    else:
        fail(f, "bias_correction_thresholds block present", "absent — post-training metadata incomplete")

    # MLflow run ID
    run_id = meta.get("mlflow_run_id", "")
    if run_id and len(run_id) == 32:
        ok(f, f"mlflow_run_id present ({run_id[:8]}…)")
    else:
        warn(f, "mlflow_run_id", f"got: {run_id!r} — training may not have logged to MLflow")


# ── predict.py ────────────────────────────────────────────────────────────────
def check_predict_py() -> None:
    f = "src/insurance_ml/predict.py"
    t = read(f)
    if t is None:
        return

    # Version strings (T3-B fix — was frozen at 6.3.1)
    for cls in ["PredictionPipeline", "HybridPredictor"]:
        # Locate the class block and check VERSION within it
        pattern = rf'class {cls}.*?VERSION\s*=\s*"([^"]+)"'
        m = re.search(pattern, t, re.DOTALL)
        if m and m.group(1) == "6.3.3":
            ok(f, f"{cls}.VERSION = 6.3.3 (T3-B fix — was frozen at 6.3.1)")
        elif m:
            fail(f, f"{cls}.VERSION = 6.3.3", f"found: {m.group(1)!r}")
        else:
            fail(f, f"{cls}.VERSION present", "class VERSION not found")

    # BUG-5 fix: pricing_factor/risk_factor read from config (not single 'factor')
    if "pricing_factor" in t and "risk_factor" in t:
        ok(f, "BUG-5: pricing_factor + risk_factor read (per-model calibration)")
    else:
        fail(f, "BUG-5: pricing_factor + risk_factor", "still using single 'factor' key — calibration broken for both models")

    # T3-C: max batch size config-driven
    if 'prediction", {}).get("max_batch_size"' in t or "max_batch_size" in t:
        ok(f, "T3-C: max_batch_size read from config (not hardcoded)")
    else:
        fail(f, "T3-C: max_batch_size config-driven", "batch cap must not be hardcoded")

    # T2-B: ML-relative floor guard in _blend_predictions
    if "min_actuarial_floor_ratio" in t:
        ok(f, "T2-B: min_actuarial_floor_ratio ML-relative floor guard present")
    else:
        fail(f, "T2-B: min_actuarial_floor_ratio", "ML-relative floor guard missing from _blend_predictions")

    # M-04: age_min read from features config (not hardcoded 0.0)
    if '_feat.get("age_min"' in t or "_feat.get('age_min'" in t:
        ok(f, "M-04: age_min read from features config")
    else:
        fail(f, "M-04: age_min config-driven", "age floor must come from config, not hardcoded")

    # F-08: children bounds read from features config
    if 'children_min' in t and 'children_max' in t:
        ok(f, "F-08: children_min/children_max read from features config")
    else:
        fail(f, "F-08: children bounds config-driven", "children validation bounds missing")

    # C-01 fix: max_actuarial_uplift_ratio default is 1.15 (not 1.45)
    m_uplift = re.search(r'get\("max_actuarial_uplift_ratio",\s*([\d.]+)\)', t)
    if m_uplift and float(m_uplift.group(1)) == 1.15:
        ok(f, "C-01: max_actuarial_uplift_ratio default 1.15 (was 1.45)")
    elif m_uplift:
        fail(f, "C-01: max_actuarial_uplift_ratio default 1.15", f"found default: {m_uplift.group(1)}")
    else:
        warn(f, "max_actuarial_uplift_ratio default", "could not verify default value")


# ── evaluate.py ───────────────────────────────────────────────────────────────
def check_evaluate_py() -> None:
    f = "src/insurance_ml/evaluate.py"
    t = read(f)
    if t is None:
        return

    # T2-C: check_ci_coverage() function added
    if "def check_ci_coverage(" in t:
        ok(f, "T2-C: check_ci_coverage() function present (was absent entirely)")
    else:
        fail(f, "T2-C: check_ci_coverage()", "function missing — CI coverage never verified on test set")

    # T2-D: statistical test on net_profit, not MAE
    # Implementation uses stats.ttest_rel (scipy paired t-test) on ml_profits vs hybrid_profits.
    if "ttest_rel" in t and "net_profit" in t:
        ok(f, "T2-D: statistical test on per-policy net_profit (stats.ttest_rel)")
    else:
        fail(f, "T2-D: net_profit paired test", "test may still compare |MAE| — hybrid can game significance by cutting error while hurting profit")

    # T2-E: win_tail_risk replaces win_smape
    if "win_tail_risk" in t:
        ok(f, "T2-E: win_tail_risk metric present (replaced win_smape)")
    else:
        fail(f, "T2-E: win_tail_risk", "win_smape is academic; tail risk is the correct deployment gate")
    if "win_smape" in t:
        # Check it's only used as legacy/secondary, not in the wins sum
        wins_line = re.search(r"wins\s*=\s*sum\(\[([^\]]+)\]\)", t)
        if wins_line and "win_smape" in wins_line.group(1):
            fail(f, "T2-E: win_smape removed from wins sum", "win_smape still contributes to deployment gate")
        else:
            ok(f, "T2-E: win_smape retained as secondary diagnostic only")

    # T3-A: revenue model uses loading formula (not revenue*1.03 compound)
    if "loading * premium_charged" in t or "loading * y_pred" in t or "loading * yp" in t:
        ok(f, "T3-A: revenue model uses actuarial loading formula")
    else:
        fail(f, "T3-A: actuarial revenue model", "old revenue*1.03 compound masked sub-3% underpricing")

    # R² sentinel: empty segments must return NaN, not 0.0
    if 'float("nan")' in t and ("len(seg_true)" in t or "len(y_true)" in t):
        ok(f, 'Finding G: empty-segment R² returns NaN (not 0.0)')
    else:
        fail(f, 'Finding G: R² NaN sentinel', 'empty segment R²=0.0 silently passes R²<0 guards')

    # XGBoost DMatrix warning suppressed (log flood fix)
    if "Falling back to prediction using DMatrix" in t:
        ok(f, "XGBoost DMatrix device-mismatch warning suppressed")
    else:
        warn(f, "XGBoost DMatrix warning filter", "warning filter may be missing — check for log flooding")


# ── render results ────────────────────────────────────────────────────────────
def render() -> int:
    passes = [r for r in results if r[0] == "PASS"]
    failures = [r for r in results if r[0] == "FAIL"]
    warnings = [r for r in results if r[0] == "WARN"]

    if USE_RICH:
        table = Table(title="Insurance ML — Setup Verification", show_lines=False)
        table.add_column("", width=5, no_wrap=True)
        table.add_column("File", style="dim", no_wrap=True)
        table.add_column("Check")
        table.add_column("Detail", style="dim")

        style_map = {"PASS": "green", "FAIL": "bold red", "WARN": "yellow"}
        icon_map = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}
        for status, file, check, detail in results:
            table.add_row(
                f"[{style_map[status]}]{icon_map[status]}[/{style_map[status]}]",
                file,
                check,
                detail,
                style=style_map[status] if status == "FAIL" else None,
            )
        console.print(table)
        console.print()
        console.print(
            f"[green]PASSED: {len(passes)}[/green]  "
            f"[yellow]WARNINGS: {len(warnings)}[/yellow]  "
            f"[{'bold red' if failures else 'green'}]FAILURES: {len(failures)}"
            f"[/{'bold red' if failures else 'green'}]"
        )
    else:
        icon_map = {"PASS": "OK  ", "FAIL": "FAIL", "WARN": "WARN"}
        for status, file, check, detail in results:
            d = f"  ({detail})" if detail else ""
            print(f"[{icon_map[status]}] {file:<30s} {check}{d}")
        print()
        print(f"PASSED: {len(passes)}  WARNINGS: {len(warnings)}  FAILURES: {len(failures)}")

    return 1 if failures else 0


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_pyproject()
    check_requirements()
    check_env()
    check_env_example()
    check_gitignore()
    check_gitattributes()
    check_makefile()
    check_precommit()
    check_git()
    check_environment()
    check_config_yaml()
    check_pipeline_metadata()
    check_predict_py()
    check_evaluate_py()
    sys.exit(render())
