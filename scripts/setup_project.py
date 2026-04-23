#!/usr/bin/env python3
"""Setup script for the Insurance ML Pipeline.

Idempotent — safe to re-run on an existing project.
Creates only the directories, __init__.py files, and .gitkeep markers
that are missing; never overwrites files that already exist.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)

MIN_PYTHON_VERSION = (3, 11)

# ─── Directories ──────────────────────────────────────────────────────────────
# Derived from project_structure.txt — matches every directory that actually
# exists in the repo.  Subdirs are listed after their parents so mkdir
# parents=True is never required, but we keep it for safety.
PROJECT_DIRECTORIES = [
    # Data
    "data/raw",
    "data/processed",
    "data/sample",
    "data/databases",  # MLflow SQLite DB lives here
    # Source package
    "src/insurance_ml",
    # Models (flat — no baseline/experiments/production subdirs in real project)
    "models",
    # Experiment tracking
    "mlruns",
    "experiments",
    "experiments/experiment_configs",
    # Reports — all sub-directories that exist in the real project
    "reports/figures",
    "reports/calibration",
    "reports/config_comparison",
    "reports/diagnostics",
    "reports/errors",
    "reports/explainability",
    "reports/learning_curves",
    "reports/metadata",
    "reports/pdp",
    "reports/shap",
    "reports/residuals",
    "reports/residuals/elastic_net",
    "reports/residuals/gradient_boosting",
    "reports/residuals/knn",
    "reports/residuals/lasso",
    "reports/residuals/lightgbm",
    "reports/residuals/linear_regression",
    "reports/residuals/random_forest",
    "reports/residuals/ridge",
    "reports/residuals/svr",
    "reports/residuals/xgboost",
    "reports/residuals/xgboost_median",
    # Notebooks (flat — no exploration/experiments subdirs in real project)
    "notebooks",
    # Tests (flat — no unit/integration/fixtures subdirs in real project)
    "tests",
    # Application code
    "api",
    "app",
    "scripts",
    "configs",
    "docker",
    # Runtime scratch dirs (gitignored; also created by Makefile setup target)
    "logs",
    "checkpoints",
    "cache",
]

# ─── .gitkeep markers ─────────────────────────────────────────────────────────
# Directories that must be tracked by git but start empty.
# Taken verbatim from project_structure.txt.
GITKEEP_DIRS = [
    "data/raw",
    "data/processed",
    "reports/figures",
]

# ─── __init__.py files ────────────────────────────────────────────────────────
# Only created if the file does not already exist.
INIT_FILES: list[tuple[str, str]] = [
    ("src/__init__.py", '"""Insurance ML — source root"""'),
    ("src/insurance_ml/__init__.py", '"""Medical Insurance ML Package"""'),
    ("api/__init__.py", '"""Insurance Prediction API Package"""'),
    ("app/__init__.py", '"""Streamlit App Package"""'),
    ("tests/__init__.py", '"""Test Package"""'),
    ("scripts/__init__.py", '"""Utility Scripts Package"""'),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def check_python_version() -> bool:
    version = sys.version_info[:2]
    if version < MIN_PYTHON_VERSION:
        console.print(
            f"[red]Error: Python {'.'.join(map(str, MIN_PYTHON_VERSION))}+ required, "
            f"but found {'.'.join(map(str, version))}[/red]"
        )
        return False
    console.print(f"[green]✓ Python {'.'.join(map(str, version))} detected[/green]")
    return True


# ─── Directory creation ───────────────────────────────────────────────────────


def create_project_structure() -> None:
    """Create all project directories (idempotent)."""
    created = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating directories...", total=len(PROJECT_DIRECTORIES))
        for directory in PROJECT_DIRECTORIES:
            try:
                p = Path(directory)
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)
                    created += 1
                progress.advance(task)
            except OSError as e:
                console.print(f"[red]Failed to create {directory}: {e}[/red]")
                raise

    console.print(
        f"[green]✓ Directories: {created} created, "
        f"{len(PROJECT_DIRECTORIES) - created} already existed[/green]"
    )


def create_gitkeep_files() -> None:
    """Place .gitkeep markers in directories that must be tracked by git."""
    created = 0
    for directory in GITKEEP_DIRS:
        marker = Path(directory) / ".gitkeep"
        if not marker.exists():
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                created += 1
            except OSError as e:
                console.print(f"[red]Failed to create {marker}: {e}[/red]")
                raise
    console.print(
        f"[green]✓ .gitkeep markers: {created} created, "
        f"{len(GITKEEP_DIRS) - created} already existed[/green]"
    )


# ─── Package init files ───────────────────────────────────────────────────────


def create_init_files() -> None:
    """Create __init__.py files only where they are missing."""
    created = 0
    for init_file, content in INIT_FILES:
        file_path = Path(init_file)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if not file_path.exists():
                file_path.write_text(content + "\n", encoding="utf-8")
                created += 1
        except OSError as e:
            console.print(f"[red]Failed to create {init_file}: {e}[/red]")
            raise
    console.print(
        f"[green]✓ __init__.py files: {created} created, "
        f"{len(INIT_FILES) - created} already existed[/green]"
    )


# ─── Config files ─────────────────────────────────────────────────────────────


def create_config_files() -> None:
    """Write config scaffolds only if the target files are absent.

    IMPORTANT: this function never overwrites existing files.
    The project ships .env.example and .pre-commit-config.yaml committed to
    version control; silently clobbering them with stale content would undo
    all the ruff/mlflow fixes already applied.
    """
    # Minimal .env.example scaffold — only written when the file is absent
    # (i.e. fresh clone before the committed version is present).
    env_example_content = (
        "# Environment Configuration — copy to .env and fill in real values\n"
        "ENVIRONMENT=development\n"
        "LOG_LEVEL=INFO\n"
        "LOG_FILE=logs/app.log\n"
        "DATA_PATH=data/raw/insurance.csv\n"
        "PROCESSED_DATA_PATH=data/processed/\n"
        "MODEL_PATH=models/\n"
        "API_HOST=127.0.0.1\n"
        "API_PORT=8000\n"
        "API_WORKERS=1\n"
        "API_RELOAD=false\n"
        "API_KEY=\n"
        "API_URL=http://127.0.0.1:8000\n"
        "API_TIMEOUT=10\n"
        "HEALTH_TIMEOUT=5\n"
        "MAX_RETRIES=2\n"
        "STREAMLIT_HOST=127.0.0.1\n"
        "STREAMLIT_PORT=8501\n"
        "MLFLOW_TRACKING_URI=./mlruns\n"
        "MLFLOW_EXPERIMENT_NAME=insurance_prediction\n"
        "MLFLOW_REGISTRY_MODEL_NAME=insurance_predictor\n"
        "USE_GPU=false\n"
        "GPU_DEVICE_ID=0\n"
        "GPU_MEMORY_LIMIT_MB=3500\n"
        "RANDOM_STATE=42\n"
        "CV_FOLDS=5\n"
        "TEST_SIZE=0.2\n"
        "VALIDATION_SIZE=0.2\n"
        "OPTUNA_ENABLED=true\n"
        "OPTUNA_N_TRIALS=50\n"
        "OPTUNA_TIMEOUT=3600\n"
        "OPTUNA_N_JOBS=1\n"
        "OPTUNA_STUDY_NAME=insurance_optimization\n"
        "OPTUNA_STORAGE=sqlite:///models/optuna_studies.db\n"
        "EARLY_STOPPING_ENABLED=true\n"
        "EARLY_STOPPING_PATIENCE=20\n"
        "CHECKPOINT_ENABLED=true\n"
        "ENABLE_DRIFT_DETECTION=true\n"
        "PREDICTION_LOG=logs/predictions.csv\n"
        "METRICS_LOG=logs/metrics.json\n"
        "TRACK_INFERENCE_TIME=true\n"
        "TRACK_MEMORY_USAGE=true\n"
    )

    created = []
    skipped = []

    env_example = Path(".env.example")
    if not env_example.exists():
        try:
            env_example.write_text(env_example_content, encoding="utf-8")
            created.append(".env.example")
        except OSError as e:
            console.print(f"[red]Error creating .env.example: {e}[/red]")
            raise
    else:
        skipped.append(".env.example")

    # Never auto-create .pre-commit-config.yaml — it is committed to the repo
    # with ruff, bandit, mypy hooks already configured.  Writing a stale
    # black/isort version here would silently regress those settings.
    pre_commit = Path(".pre-commit-config.yaml")
    if pre_commit.exists():
        skipped.append(".pre-commit-config.yaml")
    # (no else — if somehow absent, the user should copy from version control)

    if created:
        console.print(f"[green]✓ Config files created: {', '.join(created)}[/green]")
    if skipped:
        console.print(f"[dim]  Config files already exist (skipped): {', '.join(skipped)}[/dim]")


# ─── Sample data ──────────────────────────────────────────────────────────────


def create_sample_data() -> object | None:
    """Generate sample insurance data — skipped if all three CSVs already exist."""
    sample_files = [
        Path("data/sample/insurance_sample.csv"),
        Path("data/sample/train_sample.csv"),
        Path("data/sample/test_sample.csv"),
    ]
    if all(f.exists() for f in sample_files):
        console.print("[dim]  Sample data already exists — skipped[/dim]")
        return None

    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError:
        console.print(
            "[yellow]Warning: numpy/pandas/sklearn not installed — skipping sample data[/yellow]"
        )
        return None

    try:
        np.random.seed(42)
        n_samples = 2000

        ages = np.random.gamma(2, 20, n_samples).astype(int).clip(18, 80)
        bmis = np.random.lognormal(3.2, 0.3, n_samples).clip(15, 50)
        genders = np.random.choice(["male", "female"], n_samples, p=[0.51, 0.49])

        smoker_probs = np.where(ages < 30, 0.15, np.where(ages < 50, 0.25, 0.20))
        smokers = np.random.binomial(1, smoker_probs, n_samples)

        children_probs = np.where(ages < 25, 0.5, np.where(ages < 45, 2.0, 1.0))
        children = np.random.poisson(children_probs, n_samples).clip(0, 5)

        regions = np.random.choice(
            ["northeast", "northwest", "southeast", "southwest"],
            n_samples,
            p=[0.28, 0.22, 0.32, 0.18],
        )

        df = pd.DataFrame(
            {
                "age": ages,
                "sex": genders,
                "bmi": bmis,
                "children": children,
                "smoker": ["no" if s == 0 else "yes" for s in smokers],
                "region": regions,
            }
        )

        base_cost = 3000
        age_effect = df["age"] * 120 + (df["age"] ** 1.5) * 5
        gender_effect = (df["sex"] == "male") * 500
        bmi_effect = np.where(
            df["bmi"] > 30,
            (df["bmi"] - 30) ** 2 * 100 + (df["bmi"] - 25) * 300,
            np.maximum(0, (df["bmi"] - 25) * 200),
        )
        smoker_effect = (df["smoker"] == "yes") * (
            12000 + df["age"] * 150 + np.where(df["bmi"] > 30, 8000, 0)
        )
        children_effect = df["children"] * 600 + (df["children"] ** 1.5) * 200
        region_multipliers = {
            "northeast": 1.15,
            "northwest": 0.95,
            "southeast": 1.05,
            "southwest": 0.98,
        }
        region_effect = df["region"].map(region_multipliers)

        charges = (
            base_cost + age_effect + gender_effect + bmi_effect + smoker_effect + children_effect
        ) * region_effect

        noise = np.random.normal(0, charges * 0.15)
        df["charges"] = (charges + noise).clip(1200, None)

        Path("data/sample").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/sample/insurance_sample.csv", index=False)

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["smoker"]
        )
        train_df.to_csv("data/sample/train_sample.csv", index=False)
        test_df.to_csv("data/sample/test_sample.csv", index=False)

        console.print(f"[green]✓ Created sample dataset with {len(df)} records[/green]")
        return df

    except Exception as e:
        console.print(f"[red]Error creating sample data: {e}[/red]")
        raise


# ─── Dependency installation ──────────────────────────────────────────────────


def install_dependencies(dev: bool = False) -> bool:
    """Install Python dependencies via the current interpreter.

    Uses sys.executable so the correct venv python is always targeted
    (the Makefile invokes this script via `py -3.11`).
    """
    commands: list[list[str]] = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    ]
    if dev:
        commands.append([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])

    try:
        for cmd in commands:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(result.stdout)
        console.print("[green]✓ Dependencies installed successfully[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error installing dependencies:[/red] {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        console.print("[red]requirements.txt not found[/red]")
        return False


# ─── Next steps ───────────────────────────────────────────────────────────────


def display_next_steps() -> None:
    table = Table(title="Next Steps", show_header=True, header_style="bold cyan")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Command", style="green")
    table.add_column("Description")

    steps = [
        ("1", "cp .env.example .env", "Create local env file and fill in values"),
        ("2", "make install-gpu", "Install GPU torch (CUDA 12.4) — RTX 3050"),
        ("3", "make pre-commit-install", "Install git hooks"),
        ("4", "make train", "Train models (GPU enabled by default)"),
        ("5", "make mlflow-ui", "Inspect runs at http://127.0.0.1:5000"),
        ("6", "make serve", "Start FastAPI server on port 8000"),
    ]
    for step, command, description in steps:
        table.add_row(step, command, description)

    console.print(table)


# ─── Entry point ──────────────────────────────────────────────────────────────


@click.command()
@click.option("--dev", is_flag=True, help="Also install [dev] extras (ruff, pytest, mypy…)")
@click.option("--skip-deps", is_flag=True, help="Skip pip dependency installation entirely")
def main(dev: bool, skip_deps: bool) -> None:
    """Set up the Insurance ML project structure and install dependencies.

    Safe to re-run — all steps are idempotent.
    """
    console.print(
        Panel(
            "[bold blue]Insurance ML Pipeline — Project Setup[/bold blue]\n"
            "[dim]Version 4.1.0 | Python 3.11 | CUDA 12.4 | Windows 11[/dim]",
            title="Setup",
        )
    )

    if not check_python_version():
        sys.exit(1)

    try:
        setup_logging()
        create_project_structure()
        create_gitkeep_files()
        create_init_files()
        create_config_files()
        create_sample_data()

        if not skip_deps and not install_dependencies(dev=dev):
            console.print("[red]Setup completed with dependency errors — check output above[/red]")
            sys.exit(1)

        display_next_steps()
        console.print("[bold green]✓ Project setup complete[/bold green]")

    except Exception as e:
        console.print(f"[red]Setup failed: {e}[/red]")
        logger.exception("Unhandled exception during setup")
        sys.exit(1)


if __name__ == "__main__":
    main()
