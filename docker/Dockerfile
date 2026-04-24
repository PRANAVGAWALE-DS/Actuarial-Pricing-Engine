# =============================================================================
# FILE     : Pipeline/docker/Dockerfile
# PROJECT  : Improved_ML_Insurance
# PYTHON   : 3.11.0  (matches host venv)
# CUDA     : 12.4.1  (matches host driver CUDA 13.0 — backward compatible)
# UBUNTU   : 22.04
#
# STAGES
# ──────
#   base-runtime  CUDA 12.4.1 + Python 3.11 + runtime-only system libs
#                 NO compilers — this is what prod and dev inherit from.
#   base-build    Extends base-runtime with C/C++ compilers + dev headers.
#                 Only builder uses this stage; compilers never reach prod.
#   builder       All pip dependencies compiled into /opt/venv (FROM base-build)
#   prod          Lean production image  ← default (FROM base-runtime)
#   dev           Extends prod with dev tools + editable install (FROM prod)
#   mlflow        Pre-baked MLflow tracking server (FROM python:3.11-slim)
#   optuna        Pre-baked Optuna dashboard       (FROM python:3.11-slim)
#
#   mlflow and optuna are intentionally FROM python:3.11-slim, not from
#   base-runtime. They are CPU-only services with no CUDA dependency; pulling
#   the 7 GB CUDA base image for them would be wasteful. Multi-stage builds
#   allow mixing base images freely — each stage is independent.
#
# Previously all stages inherited from a single 'base' that included
#   build-essential / gcc / g++ / gfortran. Since prod was FROM base, compilers
#   were present in the production image, inflating attack surface and image size.
#   The base is now split:
#     base-runtime  — runtime libs only (prod-safe)
#     base-build    — base-runtime + compilers (builder only)
#   prod is now FROM base-runtime, so compilers are never copied or inherited.
#
# libopenblas-dev, liblapack-dev, wget, git removed from
#   base-runtime. These are build-time dependencies only:
#     - libopenblas-dev / liblapack-dev: C headers needed to compile numpy/scipy
#       wheels from source. Pre-built wheels bundle BLAS routines internally —
#       only the runtime shared libs (libgomp1, libopenblas0) are needed at
#       inference time.
#     - wget / git: no inference-time use. git in particular enables adversarial
#       git clone from inside a compromised container.
#   All four moved to base-build where they are legitimately required.
#
# TZ=Asia/Kolkata removed from ENV. Baking a locale into the
#   image makes it non-portable and leaks developer environment info to any
#   public registry push. Set TZ at runtime via docker-compose environment:
#   or -e TZ=Asia/Kolkata. Default container TZ is UTC, which is correct for
#   production log timestamps.
#
# mlflow and optuna services previously used image: python:3.11-slim
#   with inline bash -c "pip install ... && ..." commands. This ran 25-60s of
#   network-dependent installation on every container start and blocked dependent
#   services. Both are now pre-baked stages in this file.
#
# BUILD CONTEXT
# ─────────────
#   Context = Pipeline/  (one level above this file, i.e. the project root)
#   All COPY paths are relative to Pipeline/
#   .dockerignore at Pipeline/ is applied automatically when context = Pipeline/
#
# USAGE (run from Pipeline/)
# ──────────────────────────
#   docker compose -f docker/docker-compose.yml up --build
#   docker build -f docker/Dockerfile --target prod    -t insurance-ml:prod .
#   docker build -f docker/Dockerfile --target dev     -t insurance-ml:dev  .
#   docker build -f docker/Dockerfile --target mlflow  -t insurance-mlflow .
#   docker build -f docker/Dockerfile --target optuna  -t insurance-optuna .
# =============================================================================


# -----------------------------------------------------------------------------
# STAGE 1 · base-runtime
# Python 3.11 + runtime system libraries ONLY.
# No C compilers, no dev headers, no download utilities — safe for production.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base-runtime

# inline comments removed from ENV continuation blocks.
# Docker BuildKit handles them, but the classic builder (DOCKER_BUILDKIT=0)
# fails to parse '#' tokens inside multi-line ENV declarations.
# Comments now appear as standalone lines above the ENV block they describe.

# Suppress .pyc generation and force stdout/stderr flush for docker logs.
# TZ intentionally absent — set at runtime via environment: or -e TZ=...
# Default container TZ is UTC, which is correct for production log timestamps.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python 3.11 + runtime-only system libraries.
# Compilers, dev headers, wget, and git go in base-build — NOT here.
# libgomp1         : OpenMP runtime — required by LightGBM and XGBoost at
#                    inference time for parallel tree prediction.
# ocl-icd-libopencl1 / clinfo : OpenCL ICD loader — used by LightGBM GPU path.
# curl             : health-check probe (docker-compose healthcheck uses curl).
# ca-certificates  : TLS root certificates for HTTPS calls (mlflow, shap, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3-setuptools \
    libgomp1 \
    ocl-icd-libopencl1 \
    clinfo \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the system-wide default interpreter.
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA_HOME      : PyTorch locates CUDA headers and libraries.
# LD_LIBRARY_PATH: Dynamic linker finds libcuda.so, libcublas.so at runtime.
# NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES:
#   Passed to the NVIDIA Container Runtime to expose the physical GPU.
ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app


# -----------------------------------------------------------------------------
# STAGE 2 · base-build
# Extends base-runtime with C/C++ compilers, Python dev headers, and the
# BLAS/LAPACK dev packages needed to compile numpy/scipy wheels from source.
# Also includes wget and git for any source-fetch build steps.
# Used ONLY by builder — none of these tools reach prod or dev.
# -----------------------------------------------------------------------------
FROM base-runtime AS base-build

# C/C++ compilers to build Python extension wheels from source.
# python3.11-dev       : Python C API headers for extension compilation.
# libopenblas-dev      : BLAS/LAPACK headers for numpy/scipy source builds.
# liblapack-dev        : LAPACK headers (separate package in Ubuntu 22.04).
# opencl-headers       : OpenCL API headers needed at build time by pyopencl.
# wget / git           : source fetch utilities for any pip VCS dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11-dev \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    opencl-headers \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# STAGE 3 · builder
# Installs ALL Python dependencies into /opt/venv for the prod (api) image.
# Inherits from base-build (has compilers). Only /opt/venv is copied forward.
#
# TORCH REMOVAL [SIZE-1]: PyTorch (~4.5 GB) is intentionally absent here.
# All four torch usages in src/insurance_ml/models.py are inside try/except
# ImportError blocks with nvidia-smi / XGBoost fallbacks already in place:
#   • GPU detection  → falls back to XGBoost/LightGBM capability check
#   • Memory stats   → falls back to nvidia-smi (Method 2, line 3537)
#   • Cache clearing → silently returns 0.0 (no-op; XGBoost manages its own
#                      CUDA allocator anyway — torch.cuda.empty_cache() never
#                      freed XGBoost/LightGBM memory to begin with)
#   • VRAM size      → falls back to 0.0 (config validation may warn, non-fatal)
# src/insurance_ml/predict.py has zero torch imports.
#
# GPU inference via XGBoost / LightGBM is unaffected: both link directly
# against the CUDA runtime libs provided by the base-runtime image and do not
# depend on PyTorch's CUDA allocator.
#
# To restore torch in prod (e.g. if future code adds torch tensors to the
# inference path), re-add the pip install block below with:
#   RUN pip install --no-cache-dir \
#       torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
#       --extra-index-url https://download.pytorch.org/whl/cu124
# -----------------------------------------------------------------------------
FROM base-build AS builder

# Isolated virtual environment — copied verbatim into prod/dev.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Inference-only runtime dependencies (torch intentionally absent — see above).
# requirements-inference.txt has no --extra-index-url or -e . entries.
COPY requirements-inference.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# -----------------------------------------------------------------------------
# STAGE 4 · prod  (default target)
# Lean production image.
# FROM base-runtime — NO compilers, NO dev headers, NO build tools.
# Only the pre-compiled /opt/venv from builder is copied in.
# -----------------------------------------------------------------------------
FROM base-runtime AS prod

# Pull in the compiled venv from builder.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the full project source from Pipeline/ (build context = Pipeline/).
# .dockerignore excludes: models/, data/raw|processed|databases|sample/,
# cache/, experiments/, logs/, mlruns/, venv/, notebooks/, reports/.
COPY . /app
WORKDIR /app

# Install only the insurance_ml package code from src/.
# --no-deps: deps are already present from builder step 2 above.
# Without --no-deps, pip re-resolves pyproject.toml core dependencies
# (optuna, seaborn, matplotlib, tqdm) from scratch, defeating the lean image.
RUN pip install --no-cache-dir --no-deps .

# Pre-create runtime directories. Overlaid by bind-mount volumes at compose
# startup, but ensures container works standalone without docker-compose.
RUN mkdir -p logs models data/raw data/processed data/databases mlruns studies

# Drop to non-root user — security best practice for production.
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# embed health check so standalone `docker run` and orchestrators that
# don't use docker-compose (k8s, ECS, etc.) report correct container health.
# start_period=90s: API loads ~16 joblib models at startup.
# Matches docker-compose.yml api healthcheck parameters exactly.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1


# -----------------------------------------------------------------------------
# STAGE 5 · dev
# Extends prod with dev tooling and an editable install.
# Changes to src/insurance_ml/*.py are reflected without rebuild when
# ./src is bind-mounted into /app/src via a docker-compose override.
#
# Runs as root to avoid Windows NTFS -> Linux ext4 UID/GID mismatch on
# bind-mounted volumes (chown failures for non-root inside WSL2).
# -----------------------------------------------------------------------------
FROM prod AS dev

# L3 NOTE: dev stage runs as root to avoid Windows NTFS → Linux ext4 UID/GID
# mismatch on bind-mounted volumes when using WSL2.
# If you are running the dev container on native Linux or macOS, remove this
# USER root directive — the inherited appuser (uid 1001) is safe on ext4.
USER root

# TORCH RESTORE [DEV]: prod intentionally omits torch to keep the production
# image lean. Dev restores it so GPU diagnostics, torch.cuda.* calls in
# models.py, and any exploratory training scripts work without surprises.
# This step is skipped in prod/streamlit-prod builds — it only runs when
# `target: dev` is selected (or when building the dev stage directly).
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# pytest-cov pinned to 6.1.0 — matches pyproject.toml [dev] extras.
RUN pip install --no-cache-dir \
    ruff==0.8.0 \
    mypy==1.18.2 \
    pytest==8.4.2 \
    pytest-cov==6.1.0 \
    ipython \
    "watchdog[watchmedo]"

# Full editable install for dev — reinstalls with all pyproject.toml core deps
# (optuna, seaborn, matplotlib, tqdm, etc.) intentionally absent from prod.
RUN pip install --no-cache-dir -e .

ENV API_RELOAD=true \
    LOG_LEVEL=DEBUG

EXPOSE 8000 8501


# -----------------------------------------------------------------------------
# STAGE 5b · builder-streamlit
# Builds the Python venv for the Streamlit dashboard.
#
# KEY DIFFERENCES from `builder` (api):
#   1. FROM python:3.11-slim — no CUDA, no OpenCL, no GPU runtime libs.
#      Streamlit only calls the FastAPI service over HTTP; it never loads a
#      model file or issues a CUDA call.
#   2. No PyTorch install step — already absent from requirements-inference.txt
#      and not needed by any Streamlit import path.
#
# The resulting /opt/venv is copied into streamlit-prod below.
# Packages like xgboost and lightgbm install their CPU-only wheels when built
# on a non-CUDA base — that is correct; Streamlit has no use for GPU codepaths.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder-streamlit

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-inference.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# -----------------------------------------------------------------------------
# STAGE 5c · streamlit-prod
# Lean Streamlit production image.
#
# SIZE REDUCTION [SIZE-2]:
#   Previous: target: prod  → FROM nvidia/cuda:12.4.1-cudnn-runtime + torch
#             ≈ 16 GB
#   After:    target: streamlit-prod → FROM python:3.11-slim, no CUDA, no torch
#             ≈ 2.5–3 GB
#
# Streamlit's only runtime role is:
#   • Serve the web UI on :8501
#   • Call http://api:8000 for predictions (HTTP, never direct model I/O)
# It does not load joblib artifacts, does not call torch, does not need CUDA.
#
# The bind-mounts defined in docker-compose.yml (models/, data/, mlruns/, …)
# are still present so Streamlit can read config and display metadata if needed,
# but no GPU or CUDA runtime is required for those reads.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS streamlit-prod

# curl: Streamlit healthcheck (_stcore/health endpoint)
# ca-certificates: HTTPS calls to MLflow tracking server
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-streamlit /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# .dockerignore excludes models/, mlruns/, data/raw|processed|databases|sample/,
# cache/, experiments/, logs/, notebooks/, reports/ — same rules apply here.
COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir --no-deps .

# Pre-create runtime directories overlaid by bind-mount volumes at startup.
RUN mkdir -p logs models data/raw data/processed data/databases mlruns studies

RUN useradd --create-home --shell /bin/bash --uid 1001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1



# Pre-baked MLflow tracking server.
# Intentionally FROM python:3.11-slim — this is a CPU-only service.
# Pulling the 7 GB CUDA base image for a tracking server is wasteful.
# Multi-stage builds allow mixing base images freely.
#
# docker-compose.yml targets this stage with:
#   build:
#     context: ..
#     dockerfile: docker/Dockerfile
#     target: mlflow
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS mlflow

# Install curl for the docker-compose healthcheck probe.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Pin to the exact version used in requirements.txt and pyproject.toml.
RUN pip install --no-cache-dir "mlflow[auth]==3.4.0"

# Pre-create paths so the SQLite backend is valid before bind-mount attaches.
RUN mkdir -p /app/data/databases /app/mlruns

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////app/data/databases/mlflow.db", \
     "--default-artifact-root", "/app/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--workers", "1"]


# -----------------------------------------------------------------------------
# STAGE 7 · optuna
# Pre-baked Optuna dashboard.
# Intentionally FROM python:3.11-slim — CPU-only, no CUDA needed.
#
# optuna-dashboard pinned to 0.20.0 to match pyproject.toml
# [dev] extras. The previous docker-compose inline install used 0.17.0,
# a three-minor-version gap that caused schema warnings with optuna==4.5.0.
#
# docker-compose.yml targets this stage with:
#   build:
#     context: ..
#     dockerfile: docker/Dockerfile
#     target: optuna
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS optuna

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Both versions aligned to pyproject.toml [dev] extras.
RUN pip install --no-cache-dir \
    optuna==4.5.0 \
    optuna-dashboard==0.20.0

RUN mkdir -p /app/models

EXPOSE 8080

CMD ["optuna-dashboard", \
     "sqlite:////app/models/optuna_studies.db", \
     "--host", "0.0.0.0", \
     "--port", "8080"]