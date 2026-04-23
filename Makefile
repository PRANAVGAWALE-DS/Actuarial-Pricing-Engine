.PHONY: install install-dev install-gpu setup check-python check-venv \
        check-docker check-compose check-gpu gpu-info \
        test test-fast test-unit test-integration test-gpu \
        train train-cpu train-fast train-experiment train-model \
        train-resume train-mixed-precision \
        optimize optimize-model optimize-parallel \
        optuna-studies optuna-best optuna-dashboard optuna-clean optuna-export \
        serve serve-prod serve-gpu streamlit streamlit-gpu \
        docker-build docker-build-dev docker-build-gpu \
        docker-run docker-run-gpu \
        docker-compose-up docker-compose-down docker-clean \
        format lint type-check \
        security-audit security-update bandit-check safety-check \
        data-validate data-profile data-split schema-check \
        model-validate model-compare model-export \
        benchmark benchmark-gpu drift-check \
        docs-build docs-serve docs-check \
        mlflow-ui mlflow-clean mlflow-export \
        pre-commit-install pre-commit-run pre-commit-update \
        env-export deps-check deps-graph deps-update unused-deps \
        profile-code profile-gpu memory-check \
        monitor-gpu tensorboard \
        log-setup log-ci log-train log-test log-benchmark \
        clean clean-cache clean-checkpoints clean-logs \
        clean-models clean-reports clean-all \
        dev-setup dev-setup-gpu \
        ci ci-full quick-test status help

.DEFAULT_GOAL := help

# =============================================================================
# CROSS-PLATFORM PORTABILITY BLOCK
# Pure Python - no shell assumptions.
# Works on Windows (cmd.exe / PowerShell / Git Bash), Linux, and macOS.
# =============================================================================

# PYTHON detection — no $(shell ...) expansion used here because make runs
# $(shell) through cmd.exe on Windows, where >/dev/null and && are invalid.
# Strategy: try 'python' first (correct inside any activated venv on all
# platforms), fall back to 'python3' for bare Linux systems without a venv.
#
# FIX [LOW]: replaced '2>nul' (Windows-only) with an OS-conditional DEVNULL.
# On Linux/macOS, '2>nul' creates a file literally named 'nul' in the CWD.
# $(OS) is set to 'Windows_NT' by GNU make on Windows; empty on Unix.
ifeq ($(OS),Windows_NT)
DEVNULL := nul
else
DEVNULL := /dev/null
endif
PYTHON := $(subst \,/,$(shell python -c "import sys; print(sys.executable)" 2>$(DEVNULL) || python3 -c "import sys; print(sys.executable)" 2>$(DEVNULL)))

# CUDA env prefix — defined as make call-functions via define/endef.
# Usage in recipes:  @$(call RUN_WITH_CUDA,$(PYTHON) scripts/train_model.py)
#
# Why call-functions instead of prefix variables:
# The old prefix pattern (@$(VAR) cmd) needed a "--" separator so Python could
# split its own args from the child command. With shell=True that token reached
# cmd.exe as a literal command and failed on Windows.
#
# The call-function pattern textually substitutes $(1) directly inside the
# Python string at make-expansion time — no separator, no sys.argv needed.
#
# os.system() inherits os.environ changes and delegates to the platform shell
# (cmd.exe on Windows, /bin/sh on Unix) giving correct PATH lookup for
# entry-points like uvicorn and streamlit.
define RUN_WITH_CUDA
$(PYTHON) -c "import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='$(CUDA_VISIBLE_DEVICES)'; sys.exit(os.system('$(strip $(1))'))"
endef

define DISABLE_GPU
$(PYTHON) -c "import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='-1'; sys.exit(os.system('$(strip $(1))'))"
endef

# Filesystem
MKDIR_P = $(PYTHON) -c "import sys,pathlib; \
[pathlib.Path(p).mkdir(parents=True,exist_ok=True) for p in sys.argv[1:]]" --

RM_RF = $(PYTHON) -c "import sys,shutil,pathlib; \
[shutil.rmtree(p,ignore_errors=True) if pathlib.Path(p).is_dir() \
 else pathlib.Path(p).unlink(missing_ok=True) for p in sys.argv[1:]]" --

RM_F = $(PYTHON) -c "import sys,pathlib; \
[pathlib.Path(p).unlink(missing_ok=True) for p in sys.argv[1:]]" --

CP_N = $(PYTHON) -c "import sys,shutil,pathlib; \
src,dst=sys.argv[1],sys.argv[2]; \
shutil.copy2(src,dst) if not pathlib.Path(dst).exists() else None" --

CP = $(PYTHON) -c "import sys,shutil; shutil.copy2(sys.argv[1],sys.argv[2])" --

TOUCH = $(PYTHON) -c "import sys,pathlib; \
[pathlib.Path(p).touch() for p in sys.argv[1:]]" --

FIND_PYC = $(PYTHON) -c "import pathlib; \
[p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

FIND_PCC = $(PYTHON) -c "import pathlib,shutil; \
[shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"

FIND_EGG = $(PYTHON) -c "import pathlib,shutil; \
[shutil.rmtree(p) for p in pathlib.Path('.').rglob('*.egg-info') if p.is_dir()]"

# Variable guards
# FIX: removed the redundant '|| (printf ... && exit 1)' from every call site.
# REQUIRE_VAR already calls sys.exit(1) on failure, making the fallback
# unreachable dead code that also used shell-only printf with ANSI codes.
REQUIRE_VAR = $(PYTHON) -c "import sys; v=sys.argv[1]; \
(print('[ERROR] Variable not set'),sys.exit(1)) if not v else sys.exit(0)" --

# venv check
VENV_CHECK = $(PYTHON) -c \
"import os,sys; \
active=os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_DEFAULT_ENV'); \
sys.exit(0) if active else print('[WARN]  No virtual environment detected')"

# status helpers
COUNT_JOBLIB = $(shell $(PYTHON) -c \
"import glob; print(len(glob.glob('models/**/*.joblib',recursive=True)))")

# NOTE [LOW]: DIR_EXISTS uses $(shell ...) which expands at Makefile parse
# time, not at rule execution time. Results captured here reflect filesystem
# state when make was invoked, not when a target runs. Use only for
# informational status output, never to gate recipe logic.
DIR_EXISTS = $(shell $(PYTHON) -c "import pathlib; print('exists' if pathlib.Path('$(1)').exists() else 'not found')")

# =============================================================================
# CONFIGURATION
# =============================================================================

PIP                      := pip
PROJECT_NAME             := insurance_ml
API_PORT                 := 8000
STREAMLIT_PORT           := 8501
MLFLOW_PORT              := 5000
CUDA_VISIBLE_DEVICES     ?= 0
# FIX [HIGH]: API_WORKERS default is 1, not 4.
# Multiple workers each load the full model stack into VRAM independently.
# On RTX 3050 4GB this causes OOM at startup. Override only for CPU deployments.
# See also: .env, .env.example, and the serve-prod target below.
API_WORKERS              ?= 1

# ANSI escape codes work on Linux/macOS, Windows Terminal, and PowerShell 7+.
# They print as literal text on legacy cmd.exe - acceptable degradation.
RESET  := \033[0m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m

INFO    = @printf "$(GREEN)[INFO]$(RESET)  %s\n"
WARN    = @printf "$(YELLOW)[WARN]$(RESET)  %s\n"
ERROR   = @printf "$(RED)[ERROR]$(RESET) %s\n"
SUCCESS = @printf "$(GREEN)[OK]$(RESET)   %s\n"

# =============================================================================
# GUARDS
# =============================================================================

# FIX: replaced '> /dev/null 2>&1' (Unix path /dev/null does not exist on
# Windows cmd.exe) with a pure Python version check.
check-python:
	@$(PYTHON) -c "import sys; sys.exit(0)" || \
	  (printf "$(RED)[ERROR]$(RESET) Python not found (tried: $(PYTHON))\n" && exit 1)

check-venv:
	@$(VENV_CHECK)

# Docker targets use raw shell constructs intentionally - Docker is only
# available in Unix-like shells or Windows PowerShell/WSL2, never bare cmd.exe.
check-docker:
	@docker --version > /dev/null 2>&1 || \
	  (printf "$(RED)[ERROR]$(RESET) docker not found\n" && exit 1)

check-compose:
	@docker compose version > /dev/null 2>&1 || \
	  (printf "$(RED)[ERROR]$(RESET) docker compose not found\n" && exit 1)

check-gpu: check-python ## Check GPU / CUDA availability
	$(INFO) "Checking GPU availability..."
	@$(PYTHON) -c "\
import torch; \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); \
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); \
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB' \
      if torch.cuda.is_available() else '') \
" || (printf "$(RED)[ERROR]$(RESET) PyTorch not installed\n" && exit 1)
	$(SUCCESS) "GPU check done"

gpu-info: check-python ## Detailed GPU information
	$(INFO) "GPU info..."
	@$(PYTHON) -c "\
import torch; \
[print(f'  [{i}] {torch.cuda.get_device_name(i)} - \
{torch.cuda.get_device_properties(i).total_memory/1024**3:.2f} GB  \
cc={torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}') \
 for i in range(torch.cuda.device_count())] \
if torch.cuda.is_available() else print('  No CUDA GPU detected') \
" || echo "  PyTorch not available"
	@$(PYTHON) -c "\
import subprocess,sys; \
r=subprocess.run(['nvidia-smi'],capture_output=True); \
sys.stdout.buffer.write(r.stdout) if r.returncode==0 \
else print('  nvidia-smi not available') \
"

# =============================================================================
# INSTALL
# =============================================================================

install: check-python ## Install runtime dependencies
	$(INFO) "Installing dependencies..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
	@$(PYTHON) -m pip install -e .
	$(SUCCESS) "Dependencies installed"

# FIX: removed redundant 'pip install -r requirements.txt' from install-dev.
# -e .[dev] via pyproject.toml already pulls runtime deps transitively.
# If your project does NOT declare runtime deps in pyproject.toml, restore it.
install-dev: check-python ## Install with dev extras
	$(INFO) "Installing dev dependencies..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -e ".[dev]"
	$(SUCCESS) "Dev environment ready"

install-gpu: check-python ## Install GPU PyTorch (CUDA 12.4)
	$(INFO) "Installing GPU PyTorch (CUDA 12.4)..."
	@$(PYTHON) -m pip install \
	  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
	  --index-url https://download.pytorch.org/whl/cu124
	$(SUCCESS) "GPU PyTorch installed"
	@$(MAKE) check-gpu

setup: check-python ## Initialise project directories and .env
	$(INFO) "Setting up project structure..."
	@$(PYTHON) scripts/setup_project.py --dev
	@$(MKDIR_P) studies checkpoints cache
	@$(PYTHON) -c "\
import os,shutil; \
shutil.copy('.env.example','.env') if not os.path.exists('.env') else None"
	$(SUCCESS) "Setup complete"

# =============================================================================
# TESTING
# =============================================================================

# FIX: call pytest as 'python -m pytest' throughout so it always uses the venv
# interpreter and avoids the stale system pytest shadowing the venv one.
test: check-python ## Run full test suite with coverage
	$(INFO) "Running tests..."
	@$(PYTHON) -m pytest tests/ -v --cov=insurance_ml --cov-report=html --cov-report=term
	$(SUCCESS) "Tests done"

test-fast: check-python ## Run tests without coverage (fail-fast)
	$(INFO) "Running fast tests..."
	@$(PYTHON) -m pytest tests/ -x -v --tb=short --no-cov
	$(SUCCESS) "Fast tests done"

# FIX [MEDIUM]: was 'pytest tests/unit/' — that subdirectory does not exist.
# Tests are in a flat tests/ layout. Using pytest markers (-m unit / -m integration)
# which are declared in pyproject.toml [tool.pytest.ini_options] markers.
# FIX [RD2]: --no-cov added — pyproject.toml addopts includes --cov=insurance_ml
# globally. Without --no-cov, even marker-filtered runs generate a full coverage
# report, defeating the purpose of running a subset. pytest-cov --no-cov overrides.
test-unit: check-python ## Run tests marked @pytest.mark.unit
	$(INFO) "Running unit tests..."
	@$(PYTHON) -m pytest tests/ -v -m unit --no-cov
	$(SUCCESS) "Unit tests done"

# FIX [MEDIUM]: was 'pytest tests/integration/' — same issue as test-unit above.
# FIX [RD2]: --no-cov added — see test-unit comment above.
test-integration: check-python ## Run tests marked @pytest.mark.integration
	$(INFO) "Running integration tests..."
	@$(PYTHON) -m pytest tests/ -v -m integration --no-cov
	$(SUCCESS) "Integration tests done"

test-gpu: check-python ## Run GPU-marked tests only
	$(INFO) "Running GPU tests..."
	@$(PYTHON) -m pytest tests/ -v -m gpu --cov=insurance_ml --cov-report=term
	$(SUCCESS) "GPU tests done"

# =============================================================================
# TRAINING
# =============================================================================

train: check-python ## Train all models (GPU)
	$(INFO) "Training models (GPU)..."
	@$(call RUN_WITH_CUDA,$(PYTHON) scripts/train_model.py)
	$(SUCCESS) "Training done"

train-cpu: check-python ## Train on CPU only
	$(INFO) "Training on CPU..."
	@$(call DISABLE_GPU,$(PYTHON) scripts/train_model.py)
	$(SUCCESS) "CPU training done"

train-fast: check-python ## Train without HPO (--no-hpo)
	$(INFO) "Fast training (no HPO)..."
	@$(call RUN_WITH_CUDA,$(PYTHON) scripts/train_model.py --no-hpo)
	$(SUCCESS) "Fast training done"

# FIX [HIGH]: was REQUIRE_VAR + silently ignored EXPERIMENT_NAME.
# Now passes EXPERIMENT_NAME to the script via the MLFLOW_EXPERIMENT_NAME
# env var so the value is actually used by MLflow tracking in train.py.
# NOTE: $(PYTHON) is double-quoted inside os.system() to handle Windows paths
# that contain spaces (e.g. C:/Users/My Name/venv/Scripts/python.exe).
train-experiment: check-python ## Train with custom MLflow experiment name  [EXPERIMENT_NAME=<n>]
	@$(REQUIRE_VAR) "$(EXPERIMENT_NAME)"
	$(INFO) "Training experiment: $(EXPERIMENT_NAME)..."
	@$(PYTHON) -c "\
import os,sys; \
os.environ['CUDA_VISIBLE_DEVICES']='$(CUDA_VISIBLE_DEVICES)'; \
os.environ['MLFLOW_EXPERIMENT_NAME']='$(EXPERIMENT_NAME)'; \
sys.exit(os.system('\"$(PYTHON)\" scripts/train_model.py')) \
"
	$(SUCCESS) "Experiment training done"

train-model: check-python ## Train specific model  [MODEL=<n>]
	@$(REQUIRE_VAR) "$(MODEL)"
	$(INFO) "Training model: $(MODEL)..."
	@$(call RUN_WITH_CUDA,$(PYTHON) scripts/train_model.py --models $(MODEL))
	$(SUCCESS) "Model training done"

# FIX [HIGH]: was REQUIRE_VAR + silently ignored CHECKPOINT_PATH — the script
# was called without --resume, making the variable requirement a false contract.
# Target now fails immediately with a clear message until --resume is implemented
# in scripts/train_model.py. Remove the error and restore the call once done.
train-resume: ## [NOT IMPLEMENTED] Resume training from checkpoint  [CHECKPOINT_PATH=<path>]
	$(ERROR) "train-resume requires --resume support in scripts/train_model.py."
	$(ERROR) "Add: parser.add_argument('--resume', ...) and implement checkpoint loading,"
	$(ERROR) "then change this target to: RUN_WITH_CUDA scripts/train_model.py --resume \$$(CHECKPOINT_PATH)"
	@exit 1

# FIX [HIGH]: was a no-op (called plain train_model.py with no extra flag).
# Fails explicitly until --mixed-precision is implemented in the script.
train-mixed-precision: ## [NOT IMPLEMENTED] Train with AMP mixed precision
	$(ERROR) "train-mixed-precision requires --mixed-precision support in scripts/train_model.py."
	$(ERROR) "Add AMP (torch.cuda.amp.autocast) training and the CLI flag, then update this target."
	@exit 1

optimize: check-python ## Run HPO (--hpo flag)
	$(INFO) "Running Optuna optimisation..."
	@$(call RUN_WITH_CUDA,$(PYTHON) scripts/train_model.py --hpo)
	$(SUCCESS) "Optimisation done"

optimize-model: check-python ## Optimise specific model with HPO  [MODEL=<n>]
	@$(REQUIRE_VAR) "$(MODEL)"
	$(INFO) "Optimising model: $(MODEL)..."
	@$(call RUN_WITH_CUDA,$(PYTHON) scripts/train_model.py --models $(MODEL) --hpo)
	$(SUCCESS) "Model optimisation done"

# FIX [HIGH]: was REQUIRE_VAR for N_JOBS (unused) + plain train call.
# Fails explicitly until parallel Optuna is implemented.
# OPTUNA_N_JOBS is already an env var — set it in .env and run 'make optimize'.
optimize-parallel: ## [NOT IMPLEMENTED] Run parallel Optuna HPO  [N_JOBS=<n>]
	$(ERROR) "optimize-parallel: set OPTUNA_N_JOBS in .env and use 'make optimize' for now."
	$(ERROR) "WARNING: multiple parallel jobs on RTX 3050 4GB risks VRAM OOM."
	@exit 1

# =============================================================================
# OPTUNA STUDY MANAGEMENT
# =============================================================================

optuna-studies: check-python ## List all Optuna studies
	$(INFO) "Listing Optuna studies..."
	@$(PYTHON) -c "\
import optuna; \
storage='sqlite:///models/optuna_studies.db'; \
studies=optuna.study.get_all_study_summaries(storage); \
[print(f'  {s.study_name}  trials={s.n_trials}  best={s.best_trial.value if s.best_trial else \"N/A\"}') \
 for s in studies] if studies else print('  No studies found') \
" || echo "  No studies found"

optuna-best: check-python ## Show best params for a study  [STUDY_NAME=<n>]
	@$(REQUIRE_VAR) "$(STUDY_NAME)"
	$(INFO) "Best parameters for $(STUDY_NAME)..."
	@$(PYTHON) -c "\
import optuna; \
study=optuna.load_study(study_name='$(STUDY_NAME)',storage='sqlite:///models/optuna_studies.db'); \
print(f'Best trial : {study.best_trial.number}'); \
print(f'Best value : {study.best_trial.value}'); \
print(f'Best params: {study.best_trial.params}') \
"

optuna-dashboard: check-python ## Start Optuna dashboard on :8081
	$(INFO) "Starting Optuna dashboard on http://127.0.0.1:8081..."
	@$(PYTHON) -c "\
import subprocess,sys; \
r=subprocess.run(['optuna-dashboard','sqlite:///models/optuna_studies.db','--port','8081']); \
sys.exit(r.returncode) \
" || (printf "$(RED)[ERROR]$(RESET) optuna-dashboard not found - install with: pip install optuna-dashboard\n" && exit 1)

optuna-clean: ## Delete Optuna SQLite database
	$(INFO) "Resetting Optuna storage..."
	@$(RM_F) models/optuna_studies.db
	$(SUCCESS) "Optuna storage reset"

optuna-export: check-python ## Export study trials to CSV  [STUDY_NAME=<n>]
	@$(REQUIRE_VAR) "$(STUDY_NAME)"
	$(INFO) "Exporting study $(STUDY_NAME) to CSV..."
	@$(MKDIR_P) reports
	@$(PYTHON) -c "\
import optuna,pandas as pd; \
study=optuna.load_study(study_name='$(STUDY_NAME)',storage='sqlite:///models/optuna_studies.db'); \
df=study.trials_dataframe(); \
df.to_csv('reports/$(STUDY_NAME)_trials.csv',index=False); \
print(f'Exported {len(df)} trials to reports/$(STUDY_NAME)_trials.csv') \
"
	$(SUCCESS) "Study exported"

# =============================================================================
# SERVING
# =============================================================================

# FIX: all entry-point scripts (uvicorn, streamlit, mlflow, tensorboard) are
# now called via 'python -m <module>' so they are resolved through the venv
# interpreter and work on Windows without relying on PATH script lookup.
serve: check-python ## Start API (dev reload)
	$(INFO) "Starting API on :$(API_PORT) (reload)..."
	@$(PYTHON) -m uvicorn api.main:app \
	  --reload \
	  --reload-dir api \
	  --reload-dir src/insurance_ml \
	  --host 127.0.0.1 --port $(API_PORT)

# FIX [CRITICAL]: was hardcoded '--workers 4', causing VRAM OOM on RTX 3050.
# Now reads from $(API_WORKERS) which defaults to 1 (safe for GPU inference).
# Override at invocation: make serve-prod API_WORKERS=2  (CPU-only deployments only)
# NOTE: host is 127.0.0.1 (loopback) for direct local invocation. Inside Docker
# the service is exposed on 0.0.0.0 via docker-compose.yml — do not change this.
serve-prod: check-python ## Start API (production, uses API_WORKERS, default 1)
	$(INFO) "Starting API on :$(API_PORT) (prod, workers=$(API_WORKERS))..."
	@$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port $(API_PORT) --workers $(API_WORKERS)

serve-gpu: check-python ## Start API with GPU env set
	$(INFO) "Starting API on :$(API_PORT) (GPU)..."
	@$(call RUN_WITH_CUDA,$(PYTHON) -m uvicorn api.main:app \
	  --reload \
	  --reload-dir api \
	  --reload-dir src/insurance_ml \
	  --host 127.0.0.1 --port $(API_PORT))

streamlit: check-python ## Start Streamlit dashboard
	$(INFO) "Starting Streamlit on :$(STREAMLIT_PORT)..."
	@$(PYTHON) -m streamlit run app/streamlit_app.py \
	  --server.port $(STREAMLIT_PORT) \
	  --server.address 127.0.0.1

streamlit-gpu: check-python ## Start Streamlit with GPU env set
	$(INFO) "Starting Streamlit on :$(STREAMLIT_PORT) (GPU)..."
	@$(call RUN_WITH_CUDA,$(PYTHON) -m streamlit run app/streamlit_app.py --server.port $(STREAMLIT_PORT) --server.address 127.0.0.1)

mlflow-ui: check-python ## Start MLflow UI
	$(INFO) "Starting MLflow UI on :$(MLFLOW_PORT)..."
	@$(PYTHON) -m mlflow ui --host 127.0.0.1 --port $(MLFLOW_PORT) --workers 1

# FIX [LOW]: tensorboard added to pyproject.toml [dev] extras so this target
# no longer fails with 'No module named tensorboard'.
tensorboard: check-python ## Start TensorBoard on :6006
	$(INFO) "Starting TensorBoard on http://127.0.0.1:6006..."
	@$(PYTHON) -m tensorboard.main --logdir=logs --port=6006 --host=127.0.0.1

monitor-gpu: check-python ## Real-time GPU monitoring (Ctrl+C to stop)
	$(INFO) "Monitoring GPU usage..."
	@$(PYTHON) scripts/monitor_gpu.py

# =============================================================================
# DOCKER
# =============================================================================

# FIX [HIGH]: build context changed from '..' to '.'.
# When running from Pipeline/ (the project root), '..' sends the PARENT
# directory as the Docker build context, bypassing the .dockerignore at
# Pipeline/ and potentially including gigabytes of unrelated files.
# '.' sends Pipeline/ itself — which is what .dockerignore and the Dockerfile
# COPY instructions expect.
docker-build: check-docker ## Build production image
	$(INFO) "Building prod image..."
	@docker build -f docker/Dockerfile --target prod \
	  -t $(PROJECT_NAME):latest .
	$(SUCCESS) "Prod image built"

docker-build-dev: check-docker ## Build dev image
	$(INFO) "Building dev image..."
	@docker build -f docker/Dockerfile --target dev \
	  -t $(PROJECT_NAME):dev .
	$(SUCCESS) "Dev image built"

docker-build-gpu: check-docker ## Build GPU-enabled prod image
	$(INFO) "Building GPU image..."
	@docker build -f docker/Dockerfile --target prod \
	  -t $(PROJECT_NAME):gpu .
	$(SUCCESS) "GPU image built"

docker-run: check-docker ## Smoke-test: run prod container (NO volume mounts, NO env file — not a production start)
	@docker rm -f $(PROJECT_NAME) > /dev/null 2>&1 || true
	$(WARN) "docker-run starts the container WITHOUT model volumes or .env.docker."
	$(WARN) "  /health will return status=unhealthy (models/ not mounted)."
	$(WARN) "  For a production-grade start, use: make docker-compose-up"
	$(INFO) "Starting smoke-test container on :$(API_PORT)..."
	@docker run -p $(API_PORT):$(API_PORT) \
	  --name $(PROJECT_NAME) $(PROJECT_NAME):latest

docker-run-gpu: check-docker ## Run GPU container
	@docker rm -f $(PROJECT_NAME)-gpu > /dev/null 2>&1 || true
	$(INFO) "Running GPU container $(PROJECT_NAME)-gpu..."
	@docker run --gpus all -p $(API_PORT):$(API_PORT) \
	  --name $(PROJECT_NAME)-gpu $(PROJECT_NAME):gpu

docker-compose-up: check-compose ## Start all services
	$(INFO) "Starting services..."
	@docker compose -f docker/docker-compose.yml up -d
	$(SUCCESS) "Services started"

docker-compose-down: check-compose ## Stop all services
	$(INFO) "Stopping services..."
	@docker compose -f docker/docker-compose.yml down
	$(SUCCESS) "Services stopped"

docker-clean: check-docker ## Prune stopped containers and dangling images
	$(INFO) "Cleaning Docker resources..."
	@docker container prune -f > /dev/null 2>&1 || true
	@docker image prune -f > /dev/null 2>&1 || true
	$(SUCCESS) "Docker cleaned"

# =============================================================================
# CODE QUALITY
# =============================================================================

# FIX: all tools called as 'python -m <tool>' for venv-correctness.
format: check-python ## Format with ruff
	$(INFO) "Formatting..."
	@$(PYTHON) -m ruff format src/ api/ app/ tests/ scripts/
	@$(PYTHON) -m ruff check --select I --fix src/ api/ app/ tests/ scripts/
	$(SUCCESS) "Formatted"

lint: check-python ## Lint with ruff
	$(INFO) "Linting..."
	@$(PYTHON) -m ruff check src/ api/ app/ tests/ scripts/
	$(SUCCESS) "Lint passed"

type-check: check-python ## Type-check with mypy
	$(INFO) "Type-checking..."
	@$(PYTHON) -m mypy src/ api/ --ignore-missing-imports
	$(SUCCESS) "Types OK"

# =============================================================================
# SECURITY
# =============================================================================

security-audit: check-python ## pip-audit vulnerability scan
	$(INFO) "Running pip-audit..."
	@$(PYTHON) -m pip_audit
	$(SUCCESS) "Audit done"

security-update: check-python ## Upgrade pip + setuptools to latest
	$(INFO) "Applying security updates..."
	@$(PYTHON) -m pip install --upgrade setuptools pip
	$(SUCCESS) "Security updates applied"

# FIX: bandit was previously invoked twice (once for JSON file, once for
# console) causing double scan time and double exit-code noise.
# Now runs once per format; both are independent invocations but each
# scans once. Total scans: 2 (file + console) vs old 2 - no regression.
# Exit code reflects worst of the two.
bandit-check: check-python ## Bandit security linter (JSON report + console output)
	$(INFO) "Running bandit..."
	@$(MKDIR_P) reports
	@$(PYTHON) -c "\
import subprocess,sys; \
base=[sys.executable,'-m','bandit','-r','src/','api/']; \
r1=subprocess.run(base+['-f','json','-o','reports/bandit_report.json']); \
r2=subprocess.run(base); \
sys.exit(r1.returncode or r2.returncode) \
"
	$(SUCCESS) "Bandit done"

safety-check: check-python ## Safety vulnerability check
	$(INFO) "Running safety..."
	@$(PYTHON) -m safety check
	$(SUCCESS) "Safety done"

# =============================================================================
# DATA VALIDATION & MONITORING
# =============================================================================

data-validate: check-python ## [WIP] Validate data integrity
	$(INFO) "Validating data..."
	@$(PYTHON) scripts/validate_data.py
	$(SUCCESS) "Data validation done"

data-profile: check-python ## [WIP] Generate data profiling report
	$(INFO) "Generating data profile..."
	@$(PYTHON) scripts/profile_data.py
	$(SUCCESS) "Data profiling done"

data-split: check-python ## [WIP] Split data into train/val/test sets
	$(INFO) "Splitting data..."
	@$(PYTHON) scripts/split_data.py
	$(SUCCESS) "Data split done"

schema-check: check-python ## [WIP] Validate data schemas
	$(INFO) "Checking schemas..."
	@$(PYTHON) scripts/schema_validation.py
	$(SUCCESS) "Schema check done"

drift-check: check-python ## [WIP] Check for data/model drift
	$(INFO) "Checking for drift..."
	@$(PYTHON) scripts/drift_detection.py
	$(SUCCESS) "Drift check done"

# =============================================================================
# MODEL VALIDATION
# =============================================================================

model-validate: check-python ## [WIP] Validate model artifacts
	$(INFO) "Validating models..."
	@$(PYTHON) scripts/validate_models.py
	$(SUCCESS) "Model validation done"

model-compare: check-python ## [WIP] Compare trained models
	$(INFO) "Comparing models..."
	@$(PYTHON) scripts/compare_models.py
	$(SUCCESS) "Model comparison done"

model-export: check-python ## [WIP] Export model to ONNX  [MODEL_PATH=<path>]
	@$(REQUIRE_VAR) "$(MODEL_PATH)"
	$(INFO) "Exporting $(MODEL_PATH) to ONNX..."
	@$(PYTHON) scripts/export_model.py --model-path $(MODEL_PATH) --format onnx
	$(SUCCESS) "Model exported"

benchmark: check-python ## [WIP] Run performance benchmarks
	$(INFO) "Running benchmarks..."
	@$(PYTHON) scripts/benchmark.py
	$(SUCCESS) "Benchmarks done"

benchmark-gpu: check-python ## [WIP] Benchmark GPU vs CPU
	$(INFO) "Running GPU vs CPU benchmarks..."
	@$(call RUN_WITH_CUDA,$(PYTHON) scripts/benchmark.py)
	$(SUCCESS) "GPU benchmarks done"

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs-build: check-python ## Build Sphinx documentation
	$(INFO) "Building documentation..."
	@$(MKDIR_P) docs/_build
	@$(PYTHON) -m sphinx docs/ docs/_build/html -b html
	$(SUCCESS) "Documentation built"

# FIX [MEDIUM]: was port 8080, which conflicts with the Docker optuna-dashboard
# service. Changed to 8082 to avoid the collision.
docs-serve: check-python ## Serve documentation on :8082
	$(INFO) "Starting docs server on http://127.0.0.1:8082..."
	@$(PYTHON) -m http.server 8082 --directory docs/_build/html

docs-check: check-python ## Check documentation coverage (80% threshold)
	$(INFO) "Checking documentation coverage..."
	@$(PYTHON) -m interrogate src/ --fail-under=80
	$(SUCCESS) "Docs coverage OK"

# =============================================================================
# MLFLOW
# =============================================================================

mlflow-clean: ## Remove local mlruns directory
	$(INFO) "Cleaning mlruns/..."
	@$(RM_RF) mlruns
	$(SUCCESS) "MLflow runs cleaned"

mlflow-export: check-python ## Export MLflow run artifacts  [EXPERIMENT_ID=<id>]
	@$(REQUIRE_VAR) "$(EXPERIMENT_ID)"
	$(INFO) "Exporting experiment $(EXPERIMENT_ID)..."
	@$(PYTHON) -c "\
import mlflow; \
mlflow.artifacts.download_artifacts(run_id='$(EXPERIMENT_ID)',dst_path='exports/') \
"
	$(SUCCESS) "Experiment exported to exports/"

# =============================================================================
# PRE-COMMIT
# =============================================================================

pre-commit-install: check-python ## Install pre-commit hooks
	$(INFO) "Installing pre-commit hooks..."
	@$(PYTHON) -m pre_commit install
	$(SUCCESS) "Hooks installed"

pre-commit-run: check-python ## Run pre-commit on all files
	$(INFO) "Running pre-commit..."
	@$(PYTHON) -m pre_commit run --all-files
	$(SUCCESS) "Pre-commit done"

pre-commit-update: check-python ## Update pre-commit hook revisions
	$(INFO) "Updating pre-commit hooks..."
	@$(PYTHON) -m pre_commit autoupdate
	$(SUCCESS) "Hooks updated"

# =============================================================================
# ENVIRONMENT & DEPENDENCIES
# =============================================================================

# FIX: 'pip freeze > file' shell redirect replaced with Python file write.
env-export: check-python ## Freeze environment to requirements_frozen.txt
	$(INFO) "Exporting environment..."
	@$(PYTHON) -c "\
import subprocess,sys,pathlib; \
r=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True); \
pathlib.Path('requirements_frozen.txt').write_text(r.stdout); \
print('  Wrote requirements_frozen.txt') \
"
	@$(PYTHON) -c "\
import subprocess,pathlib; \
r=subprocess.run(['conda','env','export'],capture_output=True,text=True); \
(pathlib.Path('environment.yml').write_text(r.stdout), \
 print('  Wrote environment.yml')) if r.returncode==0 \
else print('  (conda not available - skipped)') \
"
	$(SUCCESS) "Environment exported"

deps-update: check-python ## Upgrade all outdated packages
	$(INFO) "Updating dependencies..."
	@$(PYTHON) -c "\
import subprocess,sys; \
r=subprocess.run([sys.executable,'-m','pip','list','--outdated','--format=freeze'], \
    capture_output=True,text=True); \
pkgs=[line.split('==')[0] for line in r.stdout.splitlines() if '==' in line]; \
(subprocess.run([sys.executable,'-m','pip','install','--upgrade']+pkgs), \
 print(f'  Upgraded {len(pkgs)} package(s)')) if pkgs \
else print('  All packages up to date') \
"
	$(SUCCESS) "Dependencies updated"

deps-check: check-python ## Check for dependency conflicts
	$(INFO) "Checking dependencies..."
	@$(PYTHON) -m pip check
	@$(PYTHON) -m pipdeptree --warn fail
	$(SUCCESS) "Dependency check passed"

# FIX: 'pipdeptree ... > file' shell redirect replaced with Python stdout capture.
deps-graph: check-python ## Generate dependency graph PNG
	$(INFO) "Generating dependency graph..."
	@$(MKDIR_P) reports
	@$(PYTHON) -c "\
import subprocess,sys,pathlib; \
r=subprocess.run([sys.executable,'-m','pipdeptree','--graph-output','png'], \
    capture_output=True); \
pathlib.Path('reports/dependencies.png').write_bytes(r.stdout); \
print('  Saved reports/dependencies.png') \
"
	$(SUCCESS) "Dependency graph saved to reports/dependencies.png"

unused-deps: check-python ## List potentially unused dependencies
	$(INFO) "Finding unused dependencies..."
	@$(PYTHON) -m pip_autoremove --list
	$(SUCCESS) "Unused dependency check done"

# =============================================================================
# PROFILING
# =============================================================================

profile-code: check-python ## [WIP] cProfile training script -> reports/profile.stats
	$(INFO) "Profiling code..."
	@$(MKDIR_P) reports
	@$(PYTHON) -m cProfile -o reports/profile.stats scripts/train_model.py
	$(SUCCESS) "Profile saved to reports/profile.stats"

profile-gpu: check-python ## [WIP] GPU utilisation profiling
	$(INFO) "Profiling GPU usage..."
	@$(call RUN_WITH_CUDA,$(PYTHON) scripts/profile_gpu.py)
	$(SUCCESS) "GPU profiling done"

memory-check: check-python ## Memory leak analysis via memory_profiler
	$(INFO) "Running memory analysis..."
	@$(PYTHON) -m memory_profiler scripts/train_model.py
	$(SUCCESS) "Memory analysis done"

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

log-setup: ## Create logs/ directory
	@$(MKDIR_P) logs
	$(SUCCESS) "logs/ ready"

# FIX: log targets previously built a 'log=' variable but never used it -
# output was NOT written to any file. Now captures stdout+stderr and writes
# to the timestamped file while also printing to console (tee behaviour).
log-ci: check-python log-setup ## Run CI pipeline and save to timestamped log
	$(INFO) "Running CI and saving log..."
	@$(PYTHON) -c "\
import subprocess,sys,pathlib; \
from datetime import datetime; \
log=pathlib.Path('logs/ci_'+datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.log'); \
r=subprocess.run(['$(MAKE)','ci'],capture_output=True,text=True); \
out=r.stdout+(r.stderr or ''); \
log.write_text(out); print(out,end=''); \
print(f'  Log saved: {log}'); sys.exit(r.returncode) \
"

log-train: check-python log-setup ## Run training and save to timestamped log
	$(INFO) "Training and saving log..."
	@$(PYTHON) -c "\
import subprocess,sys,pathlib; \
from datetime import datetime; \
log=pathlib.Path('logs/train_'+datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.log'); \
r=subprocess.run(['$(MAKE)','train'],capture_output=True,text=True); \
out=r.stdout+(r.stderr or ''); \
log.write_text(out); print(out,end=''); \
print(f'  Log saved: {log}'); sys.exit(r.returncode) \
"

log-test: check-python log-setup ## Run tests and save to timestamped log
	$(INFO) "Testing and saving log..."
	@$(PYTHON) -c "\
import subprocess,sys,pathlib; \
from datetime import datetime; \
log=pathlib.Path('logs/test_'+datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.log'); \
r=subprocess.run(['$(MAKE)','test'],capture_output=True,text=True); \
out=r.stdout+(r.stderr or ''); \
log.write_text(out); print(out,end=''); \
print(f'  Log saved: {log}'); sys.exit(r.returncode) \
"

log-benchmark: check-python log-setup ## Run GPU benchmark and save to timestamped log
	$(INFO) "Benchmarking and saving log..."
	@$(PYTHON) -c "\
import subprocess,sys,pathlib; \
from datetime import datetime; \
log=pathlib.Path('logs/benchmark_'+datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.log'); \
r=subprocess.run(['$(MAKE)','benchmark-gpu'],capture_output=True,text=True); \
out=r.stdout+(r.stderr or ''); \
log.write_text(out); print(out,end=''); \
print(f'  Log saved: {log}'); sys.exit(r.returncode) \
"

# =============================================================================
# CLEAN
# =============================================================================

clean: ## Remove Python bytecode, caches, coverage artefacts
	$(INFO) "Cleaning..."
	@$(FIND_PYC)
	@$(FIND_PCC)
	@$(RM_RF) .pytest_cache .coverage htmlcov dist build .mypy_cache
	@$(FIND_EGG)
	$(SUCCESS) "Clean done"

clean-cache: ## Remove cache/ directory contents
	$(INFO) "Cleaning cache/..."
	@$(RM_RF) cache
	@$(MKDIR_P) cache
	$(SUCCESS) "Cache cleared"

clean-checkpoints: ## Remove training checkpoints
	$(INFO) "Cleaning checkpoints/..."
	@$(RM_RF) checkpoints
	@$(MKDIR_P) checkpoints
	$(SUCCESS) "Checkpoints cleared"

# FIX: added guard for missing logs/ dir - previously PathLib would throw if
# the directory did not yet exist.
clean-logs: ## Remove log files
	$(INFO) "Cleaning logs/..."
	@$(PYTHON) -c "\
import pathlib; \
[p.unlink() for p in pathlib.Path('logs').glob('*.log') if p.is_file()] \
if pathlib.Path('logs').is_dir() else None"
	$(SUCCESS) "Logs cleared"

# FIX: same guard for missing reports/ dir.
clean-reports: ## Remove generated report files
	$(INFO) "Cleaning reports/..."
	@$(PYTHON) -c "\
import pathlib; \
[p.unlink() for ext in ['*.json','*.png','*.html','*.md','*.csv'] \
 for p in pathlib.Path('reports').glob(ext) if p.is_file()] \
if pathlib.Path('reports').is_dir() else None"
	$(SUCCESS) "Reports cleared"

# FIX: replaced '|| exit 0' which swallowed ALL Python errors (not just the
# intentional user-cancel). Confirmation and deletion are now one Python call;
# cancel path uses sys.exit(0) cleanly without leaving make's error handling
# ambiguous.
clean-models: ## Delete trained model artefacts - preserves JSON metadata
	$(WARN) "This removes all .joblib and checksum files from models/"
	@$(PYTHON) -c "\
import os,shutil,sys; \
ans=input('Continue? [y/N] '); \
sys.exit(0) if ans.lower()!='y' else None; \
tmp='.model_json_bak'; os.makedirs(tmp,exist_ok=True); \
[shutil.copy2(f'models/{f}',f'{tmp}/{f}') \
 for f in ['bias_correction.json','pipeline_metadata.json','test_indices.json'] \
 if os.path.exists(f'models/{f}')]; \
shutil.rmtree('models',ignore_errors=True); os.makedirs('models'); \
[shutil.move(f'{tmp}/{f}',f'models/{f}') \
 for f in os.listdir(tmp) if os.path.exists(f'{tmp}/{f}')]; \
shutil.rmtree(tmp,ignore_errors=True); \
print('Models cleared') \
"
	$(SUCCESS) "Models cleared (JSON metadata preserved)"

# FIX: removed docker-clean from clean-all.
# docker-clean requires Docker installed and running - a silent hard dependency
# on an external daemon is wrong for a routine local filesystem clean.
# Call 'make docker-clean' explicitly when needed.
clean-all: clean clean-logs clean-cache clean-checkpoints clean-reports mlflow-clean optuna-clean ## Full clean (keeps trained models; does not require Docker)
	$(SUCCESS) "Full clean done"

# =============================================================================
# CI & COMBINED TARGETS
# =============================================================================

ci: format lint type-check security-audit bandit-check test ## Full CI pipeline
	$(SUCCESS) "CI passed"

# FIX [HIGH]: removed data-validate and model-validate from ci-full.
# Both targets call scripts that do not exist in scripts/ (validate_data.py,
# validate_models.py, compare_models.py, etc. are WIP stubs). Including them
# caused ci-full to always fail before any tests ran.
# Restore them here once the scripts are implemented.
ci-full: format lint type-check security-audit bandit-check safety-check test ## Extended CI pipeline
	$(SUCCESS) "Full CI passed"

quick-test: format lint test-fast ## Quick dev cycle: format -> lint -> fast tests
	$(SUCCESS) "Quick test cycle done"

dev-setup: setup install-dev ## Full development environment setup
	@$(MAKE) check-venv
	$(SUCCESS) "Dev environment ready"

dev-setup-gpu: setup install-gpu install-dev ## Full GPU development environment setup
	@$(MAKE) check-gpu
	$(SUCCESS) "GPU dev environment ready"

# =============================================================================
# STATUS
# =============================================================================

status: check-python ## Show project status summary
	@$(PYTHON) -c "\
import os,subprocess,sys,glob,pathlib; \
print(); print('=== Project Status ==='); \
\
def run(cmd): \
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True); \
    return r.stdout.strip() if r.returncode==0 else 'N/A'; \
\
v=run('$(PYTHON) --version'); \
print(f'Python:      {v}'); \
venv=os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_DEFAULT_ENV') or 'none'; \
print(f'Venv:        {venv}'); \
print(f'Git branch:  {run(\"git branch --show-current\")}'); \
\
jobs=glob.glob('models/**/*.joblib',recursive=True); \
print(f'models/:     {len(jobs)} .joblib file(s)'); \
\
print(f'mlruns/:     {\"exists\" if pathlib.Path(\"mlruns\").is_dir() else \"not found\"}'); \
print(f'Optuna DB:   {\"exists\" if pathlib.Path(\"models/optuna_studies.db\").is_file() else \"not found\"}'); \
\
try: \
    import torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; \
    print(f'GPU:         {gpu}'); \
except ImportError: \
    print('GPU:         PyTorch not available'); \
\
print('====================='); print() \
"

# =============================================================================
# HELP
# =============================================================================

# FIX: replaced grep+awk (Unix-only tools not available on Windows cmd.exe)
# with a pure Python Makefile parser that extracts target + '##' comment pairs.
help: ## Show available targets
	@$(PYTHON) -c "\
import pathlib,re; \
lines=pathlib.Path('$(firstword $(MAKEFILE_LIST))').read_text(encoding='utf-8').splitlines(); \
print(); print('Usage: make <target>  [VAR=value ...]'); print(); \
rows=[(m.group(1),m.group(2)) for l in lines \
      for m in [re.match(r'^([a-zA-Z_][a-zA-Z0-9_-]*):.*?##\s*(.+)',l)] if m]; \
[print(f'  {t:<30} {d}') for t,d in rows]; \
print(); \
print('Examples:'); \
print('  make train                               # GPU training'); \
print('  make train-experiment EXPERIMENT_NAME=v2_test'); \
print('  make optimize-model   MODEL=xgboost'); \
print('  make optuna-best      STUDY_NAME=pricing_model'); \
print('  make optuna-export    STUDY_NAME=pricing_model'); \
print('  make mlflow-export    EXPERIMENT_ID=abc123'); \
print('  make model-export     MODEL_PATH=models/pricing.joblib'); \
print('  make log-train                           # train + save timestamped log'); \
print('  make ci                                  # full CI pipeline'); \
print('  make status                              # project overview'); \
print() \
"
