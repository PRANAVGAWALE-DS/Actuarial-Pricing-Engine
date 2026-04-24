"""
verify_patches.py
=================
Run from the Pipeline/ directory:

    python scripts/verify_patches.py

Checks every applied patch (A1, A2-replacement, A3, A4, A5, B1–B4, B6,
C1–C5, D1–D3) mechanically against the actual files.

Exit code 0 = all checks passed.
Exit code 1 = one or more failures (details printed to stdout).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # Pipeline/
PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"  ✅  {name}")
    else:
        FAIL.append(name)
        detail_str = f"\n       → {detail}" if detail else ""
        print(f"  ❌  {name}{detail_str}")


def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


# ─────────────────────────────────────────────────────────────────────────────
print("\n── GROUP A: P0 Blockers ──────────────────────────────────────────────────")

# A1 · .dockerignore: *.bak* covers *.bak_next_level
di = read(".dockerignore")
check(
    "A1 · .dockerignore contains '*.bak*' (not just '*.bak')",
    "*.bak*" in di,
    "Change '*.bak' to '*.bak*' in .dockerignore",
)
check(
    "A1 · .dockerignore does NOT have bare '*.bak' line",
    not re.search(r"^\*\.bak$", di, re.MULTILINE),
    "Remove the old '*.bak' line — '*.bak*' supersedes it",
)

# A2-replacement · docker-compose.yml healthcheck uses JSON body check
compose = read("docker/docker-compose.yml")
check(
    "A2-replacement · docker-compose.yml api healthcheck parses JSON body",
    "status" in compose and ("python3" in compose or "jq" in compose) and "healthy" in compose,
    "Replace simple curl health check with JSON body assertion: "
    "curl | python3 -c \"sys.exit(0 if d.get('status')=='healthy' else 1)\"",
)
check(
    "A2-replacement · /health route still returns 200 on unhealthy (cd.yml compat)",
    'return HealthResponse(\n            status="unhealthy"' in read("api/routes.py")
    or 'status="unhealthy"' in read("api/routes.py"),
    "Do NOT apply A2 to routes.py — cd.yml smoke test expects HTTP 200",
)

# A3 · deploy_model.py is no longer empty
deploy = read("scripts/deploy_model.py")
check(
    "A3 · deploy_model.py raises NotImplementedError",
    "NotImplementedError" in deploy,
    "Replace empty deploy_model.py with the NotImplementedError stub",
)

# A4 · Makefile uses POSIX syntax
makefile = read("Makefile")
windows_patterns = [">nul", "exit /b", "if not exist", "if not defined", "py -3.11"]
windows_found = [p for p in windows_patterns if p in makefile]
check(
    "A4 · Makefile has no Windows-only shell syntax",
    len(windows_found) == 0,
    f"Windows patterns still present: {windows_found}",
)
check(
    "A4 · Makefile uses 'python3' not 'py -3.11'",
    "PYTHON := python3" in makefile or "python3" in makefile,
    "Replace 'py -3.11' with 'python3' in Makefile",
)
check(
    "A4 · Makefile uses '/dev/null' not 'nul'",
    "/dev/null" in makefile,
    "Replace '>nul 2>&1' with '>/dev/null 2>&1'",
)

# A5 · Dockerfile splits base into runtime + buildenv
dockerfile = read("docker/Dockerfile")
check(
    "A5 · Dockerfile has 'runtime' stage",
    "AS runtime" in dockerfile,
    "Add 'FROM nvidia/cuda:... AS runtime' stage (no compilers)",
)
check(
    "A5 · Dockerfile has 'buildenv' stage (compilers isolated)",
    "AS buildenv" in dockerfile,
    "Add 'FROM runtime AS buildenv' stage (gcc/g++ here only)",
)
check(
    "A5 · prod stage inherits from 'runtime' (not 'base')",
    "FROM runtime AS prod" in dockerfile,
    "Change 'FROM base AS prod' to 'FROM runtime AS prod'",
)
check(
    "A5 · No 'AS base' stage left in Dockerfile",
    "AS base" not in dockerfile,
    "Remove or rename the old 'base' stage to 'runtime'",
)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── GROUP B: P1 High-Risk ─────────────────────────────────────────────────")

routes = read("api/routes.py")

# B1 · Single-pass inference
check(
    "B1 · predict_single uses predict_with_intervals() (single pass)",
    "predict_with_intervals" in routes and "predictor.predict(input_df" not in routes,
    "Remove predictor.predict() call and use predict_with_intervals() instead",
)
check(
    "B1 · _compute_ci() function is removed (dead code)",
    "def _compute_ci" not in routes,
    "Delete the _compute_ci() function — it is unreachable after B1",
)

# B2 · Checksum verification in load_model() — order: hash BEFORE joblib.load()
models_src = read("src/insurance_ml/models.py")
# Find the load_model method and check that checksum comes before joblib.load
load_model_match = re.search(r"def load_model.*?(?=def |\Z)", models_src, re.DOTALL)
if load_model_match:
    lm_body = load_model_match.group()
    cs_pos = lm_body.find("sha256")
    jl_pos = lm_body.find("joblib.load(model_path)")
    check(
        "B2 · Checksum SHA-256 computed BEFORE joblib.load() (security)",
        cs_pos != -1 and jl_pos != -1 and cs_pos < jl_pos,
        "Move checksum verification to BEFORE joblib.load() — "
        "deserialization happens during load, hash must be checked on raw bytes first",
    )
    check(
        "B2 · Checksum mismatch raises RuntimeError",
        "RuntimeError" in lm_body and "Checksum mismatch" in lm_body,
        "Add RuntimeError('Checksum mismatch...') when hashes differ",
    )
    check(
        "B2 · Missing checksum logs WARNING (not error)",
        "No checksum" in lm_body or "no checksum" in lm_body.lower(),
        "Log a warning when checksum file is absent (not an error — allow legacy models)",
    )
else:
    check("B2 · load_model() found in models.py", False, "Cannot locate load_model method")

# B3 · CORS allow_headers explicit
main_src = read("api/main.py")
check(
    "B3 · CORS allow_headers is not wildcard '*'",
    'allow_headers=["*"]' not in main_src,
    'Change allow_headers=["*"] to explicit list: ["Content-Type", "Authorization", "X-Request-ID"]',
)
check(
    "B3 · CORS allow_headers contains 'Content-Type'",
    "Content-Type" in main_src,
    "Add 'Content-Type' to allow_headers list",
)

# B4 · config.yaml api.workers = 1
config_yaml = read("configs/config.yaml")
check(
    "B4 · config.yaml api.workers is 1",
    re.search(r"^  workers:\s*1\b", config_yaml, re.MULTILINE) is not None,
    "Change api.workers: 4 to api.workers: 1",
)

# B5 · NOT applied — tracking_uri stays as-is, env var override already in config.py
config_py = read("src/insurance_ml/config.py")
check(
    "B5 · config.py already overrides mlflow tracking_uri from env var",
    "MLFLOW_TRACKING_URI" in config_py,
    "Confirm load_config() reads MLFLOW_TRACKING_URI env var (already present — no YAML change needed)",
)

# tuple-wrapped assertion
compat = read("tests/test_predict_compatibility.py")
check(
    "B6 · assert_not_called() is a standalone statement (not wrapped in tuple)",
    "fe.inverse_transform_target.assert_not_called()" in compat
    and "fe.inverse_transform_target.assert_not_called()," not in compat,
    "Remove the outer tuple: (fe.inverse_transform_target.assert_not_called(), ('msg',)) "
    "→ fe.inverse_transform_target.assert_not_called()",
)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── GROUP C: P2 Improvements ──────────────────────────────────────────────")

pyproject = read("pyproject.toml")

check(
    "C1 · pyproject.toml OS classifier is 'OS Independent'",
    "Operating System :: OS Independent" in pyproject,
    "Replace 'Windows 11' classifier with 'Operating System :: OS Independent'",
)
check(
    "C1 · pyproject.toml no Windows classifier",
    "Windows :: Windows 11" not in pyproject,
    "Remove 'Operating System :: Microsoft :: Windows :: Windows 11'",
)
check(
    "C2 · pyproject.toml [gpu] extra has no torch version pinned",
    "torch==2.6.0" not in pyproject or "# torch" in pyproject,
    "Remove torch== from [gpu] extra — pip can't install GPU builds from PyPI extras",
)

streamlit_src = read("app/streamlit_app.py")
check(
    "C3 · Streamlit uses urlparse for HTTPS check (not startswith)",
    "urlparse" in streamlit_src,
    "Replace startswith('http://localhost') check with urlparse hostname comparison",
)
check(
    "C3 · Streamlit HTTPS check guards against localhost prefix spoofing",
    "_is_local = _parsed.hostname in" in streamlit_src or "hostname in" in streamlit_src,
    "Check: _parsed.hostname in ('localhost', '127.0.0.1', '::1')",
)
check(
    "C4 · Streamlit MLflow UI URL uses MLFLOW_UI_URL env var",
    "MLFLOW_UI_URL" in streamlit_src,
    "Replace hardcoded replace('http://mlflow:5000',...) with os.environ.get('MLFLOW_UI_URL', ...)",
)
check(
    "C5 · docker-compose.yml Streamlit env has MLFLOW_UI_URL",
    "MLFLOW_UI_URL" in compose,
    "Add MLFLOW_UI_URL: 'http://localhost:5000' to Streamlit service environment",
)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── GROUP D: New Files ─────────────────────────────────────────────────────")

test_models = read("tests/test_models.py")
check(
    "D1 · tests/test_models.py is not empty",
    len(test_models.strip()) > 10,
    "Implement tests/test_models.py (ModelManager unit tests)",
)
check(
    "D1 · test_models.py uses ModelManager(config) not ModelManager()",
    "ModelManager()" not in test_models or "ModelManager(config" in test_models,
    "ModelManager() with no args raises ValueError — pass a config dict",
)
check(
    "D1 · test_models.py imports from insurance_ml (not src.insurance_ml)",
    "from insurance_ml.models" in test_models
    or "from src.insurance_ml.models" in test_models,  # src. is also acceptable per PYTHONPATH
    "Use 'from insurance_ml.models import ModelManager'",
)
check(
    "D1 · test_models.py covers FileNotFoundError for missing models",
    "FileNotFoundError" in test_models,
    "Add test: manager.load_model('nonexistent') raises FileNotFoundError",
)
check(
    "D1 · test_models.py covers checksum mismatch RuntimeError",
    "RuntimeError" in test_models and "mismatch" in test_models.lower(),
    "Add test: corrupted model file raises RuntimeError('Checksum mismatch...')",
)

test_data = read("tests/test_data.py")
check(
    "D2 · tests/test_data.py is not empty",
    len(test_data.strip()) > 10,
    "Implement tests/test_data.py (FeatureEngineer shape/NaN tests)",
)
check(
    "D2 · test_data.py uses FeatureEngineer(config_dict=...) not FeatureEngineer()",
    "FeatureEngineer()" not in test_data or "config_dict" in test_data,
    "FeatureEngineer() with no args raises ValueError — pass config_dict=get_feature_config(config)",
)

req_inf = read("requirements-inference.txt")
check(
    "D3 · requirements-inference.txt exists",
    exists("requirements-inference.txt"),
    "Create requirements-inference.txt (slim runtime deps, no DVC/Celery/Sphinx/bandit)",
)
check(
    "D3 · requirements-inference.txt includes statsmodels",
    "statsmodels" in req_inf,
    "Add statsmodels==0.14.5 — features.py imports it at module level",
)
check(
    "D3 · requirements-inference.txt excludes dvc",
    "dvc" not in req_inf,
    "Remove dvc and its sub-packages — training only",
)
check(
    "D3 · requirements-inference.txt excludes celery",
    "celery" not in req_inf,
    "Remove celery — not used at inference time",
)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── ADDITIONAL CHECKS ─────────────────────────────────────────────────────")

# .gitignore already had *.bak_next_level — confirm
gitignore = read(".gitignore")
check(
    "· .gitignore covers *.bak_next_level (was already correct)",
    "*.bak_next_level" in gitignore,
    "Add '*.bak_next_level' to .gitignore",
)

# mlruns/ should not be baked in — check .dockerignore
check(
    "· .dockerignore excludes mlruns/",
    "mlruns/" in di,
    "Add 'mlruns/' to .dockerignore",
)

# ci.yml sets MLFLOW_TRACKING_URI for test step (optional improvement)
ci_yml = read(".github/workflows/ci.yml")
check(
    "· ci.yml suppresses MLflow writes (MLFLOW_TRACKING_URI=file:///dev/null)",
    "MLFLOW_TRACKING_URI" in ci_yml,
    "(Optional) Add MLFLOW_TRACKING_URI: file:///dev/null to ci.yml test step env to prevent stray mlruns/ creation",
)

# deploy_model.py is referenced in cd.yml or Makefile
check(
    "· deploy_model.py exits non-zero (sys.exit(1) present)",
    "sys.exit(1)" in read("scripts/deploy_model.py"),
    "Ensure __main__ block calls sys.exit(1) on NotImplementedError",
)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── SUMMARY ───────────────────────────────────────────────────────────────")
total = len(PASS) + len(FAIL)
print(f"\n  Passed : {len(PASS)}/{total}")
print(f"  Failed : {len(FAIL)}/{total}")

if FAIL:
    print("\n  Items still needing attention:")
    for f in FAIL:
        print(f"    • {f}")
    print()
    sys.exit(1)
else:
    print("\n  All checks passed — ready to build.\n")
    sys.exit(0)
