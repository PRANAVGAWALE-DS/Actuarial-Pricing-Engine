"""
sweep_g4_params.py
==================
Grid sweep to find XGBoost hyperparameters that reduce the G4 train/val
RMSE gap on CI sample data, and to establish an evidence-based G4 threshold.

Run from project root:
    python scripts/sweep_g4_params.py

Paths, features, seed, and preprocessing mirror ci_model_gate.py exactly.
Nothing is assumed — all constants are taken directly from that file.

Output:
  - Full results table printed to stdout, sorted by gap ascending
  - CSV saved to reports/g4_sweep_results.csv for inspection
  - Prints a "candidate configs" block: rows where test R² >= 0.70
    (the existing G6 floor) AND gap is minimised

After running, pick:
  1. The config row you want to lock in for ci_model_gate.py
  2. The gap value of that row → set as train_val_gap_max in GATE_THRESHOLDS
"""

import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths (identical to ci_model_gate.py) ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_PATH = PROJECT_ROOT / "data" / "sample" / "train_sample.csv"
TEST_PATH  = PROJECT_ROOT / "data" / "sample" / "test_sample.csv"
OUTPUT_CSV = PROJECT_ROOT / "reports" / "g4_sweep_results.csv"

# ── Features (confirmed from config.yaml via ci_model_gate.py) ───────────────
NUMERICAL_FEATURES   = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET_COLUMN        = "charges"

# ── Seed (identical to ci_model_gate.py) ─────────────────────────────────────
SEED = int(os.environ.get("RANDOM_SEED", 42))

# ── Existing G6 floor (from ci_model_gate.py GATE_THRESHOLDS) ────────────────
G6_R2_MIN = 0.70

# ── Sweep grid ────────────────────────────────────────────────────────────────
# These are ranges to search — not assumed final values.
# After seeing the output table, you choose what to keep.
PARAM_GRID = {
    "max_depth":        [3, 4, 5, 6],
    "min_child_weight": [5, 10, 20, 30],
    "reg_lambda":       [1.0, 3.0, 5.0, 10.0],
}

# n_estimators, learning_rate, subsample, colsample_bytree stay fixed —
# they are taken verbatim from ci_model_gate.py's build_pipeline().
FIXED_PARAMS = {
    "n_estimators":    100,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
}


# ── Data loading (identical logic to ci_model_gate.py load_data()) ───────────
def load_data():
    for path in [TRAIN_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Sample data not found: {path}")

    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    expected = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    for name, df in [("train", train), ("test", test)]:
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"{name}_sample.csv missing columns: {missing}")

    X_train = train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_train = train[TARGET_COLUMN]
    X_test  = test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_test  = test[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


# ── Single-run helper ─────────────────────────────────────────────────────────
def run_one(X_train, y_train, X_test, y_test, max_depth, min_child_weight, reg_lambda):
    import xgboost as xgb
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Preprocessing: identical to ci_model_gate.py build_pipeline()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = xgb.XGBRegressor(
        objective        = "reg:squarederror",
        n_estimators     = FIXED_PARAMS["n_estimators"],
        learning_rate    = FIXED_PARAMS["learning_rate"],
        subsample        = FIXED_PARAMS["subsample"],
        colsample_bytree = FIXED_PARAMS["colsample_bytree"],
        max_depth        = max_depth,
        min_child_weight = min_child_weight,
        reg_lambda       = reg_lambda,
        random_state     = SEED,
        tree_method      = "hist",
        device           = "cpu",
        verbosity        = 0,
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_rmse  = float(np.sqrt(mean_squared_error(y_test,  y_pred_test)))
    test_r2    = float(r2_score(y_test, y_pred_test))

    gap = (test_rmse - train_rmse) / train_rmse if train_rmse > 0 else 0.0

    # Overpricing (G7 metric — to confirm it still passes after regularisation)
    pct_error    = (y_pred_test - y_test.values) / y_test.values
    overpriced   = float((pct_error > 0.10).mean())

    return {
        "max_depth":        max_depth,
        "min_child_weight": min_child_weight,
        "reg_lambda":       reg_lambda,
        "train_rmse":       round(train_rmse, 0),
        "test_rmse":        round(test_rmse, 0),
        "gap_pct":          round(gap * 100, 1),   # shown as % for readability
        "gap_raw":          round(gap, 4),          # raw float for threshold decision
        "test_r2":          round(test_r2, 4),
        "overpriced_pct":   round(overpriced * 100, 1),
        "g6_ok":            test_r2 >= G6_R2_MIN,
        "g7_ok":            overpriced <= 0.62,    # from GATE_THRESHOLDS
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("G4 Parameter Sweep — Insurance Pricing CI Gate")
    print(f"Seed: {SEED}  |  Fixed: n_estimators={FIXED_PARAMS['n_estimators']}, "
          f"lr={FIXED_PARAMS['learning_rate']}, "
          f"subsample={FIXED_PARAMS['subsample']}, "
          f"colsample_bytree={FIXED_PARAMS['colsample_bytree']}")
    print("=" * 75)

    print("\nLoading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    # Build all combinations
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print(f"Running {total} combinations "
          f"({len(PARAM_GRID['max_depth'])} depths × "
          f"{len(PARAM_GRID['min_child_weight'])} min_child_weights × "
          f"{len(PARAM_GRID['reg_lambda'])} lambdas)...\n")

    results = []
    t_start = time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        row = run_one(X_train, y_train, X_test, y_test, **params)
        results.append(row)

        # Progress tick every 8 runs
        if i % 8 == 0 or i == total:
            elapsed = time.time() - t_start
            eta     = (elapsed / i) * (total - i)
            print(f"  [{i:>2}/{total}]  "
                  f"depth={params['max_depth']} mcw={params['min_child_weight']:>2} "
                  f"lambda={params['reg_lambda']:>5}  →  "
                  f"gap={row['gap_pct']:>5.1f}%  "
                  f"test_r2={row['test_r2']:.4f}  "
                  f"ETA {eta:.0f}s")

    # ── Sort by gap ascending ─────────────────────────────────────────────────
    results.sort(key=lambda r: r["gap_raw"])

    # ── Print full table ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 95)
    print("FULL RESULTS  (sorted by gap ascending)")
    print("=" * 95)
    header = (
        f"{'depth':>5}  {'mcw':>4}  {'lambda':>7}  "
        f"{'train_rmse':>11}  {'test_rmse':>10}  "
        f"{'gap%':>6}  {'test_r2':>8}  {'overprice%':>11}  "
        f"{'G6_ok':>6}  {'G7_ok':>6}"
    )
    print(header)
    print("-" * 95)
    for r in results:
        row_str = (
            f"{r['max_depth']:>5}  {r['min_child_weight']:>4}  {r['reg_lambda']:>7}  "
            f"${r['train_rmse']:>10,.0f}  ${r['test_rmse']:>9,.0f}  "
            f"{r['gap_pct']:>5.1f}%  {r['test_r2']:>8.4f}  "
            f"{r['overpriced_pct']:>10.1f}%  "
            f"{'YES':>6}  {'YES':>6}"
            if (r["g6_ok"] and r["g7_ok"])
            else
            f"{r['max_depth']:>5}  {r['min_child_weight']:>4}  {r['reg_lambda']:>7}  "
            f"${r['train_rmse']:>10,.0f}  ${r['test_rmse']:>9,.0f}  "
            f"{r['gap_pct']:>5.1f}%  {r['test_r2']:>8.4f}  "
            f"{r['overpriced_pct']:>10.1f}%  "
            f"{'YES' if r['g6_ok'] else 'NO':>6}  {'YES' if r['g7_ok'] else 'NO':>6}"
        )
        print(row_str)

    # ── Candidate block: G6+G7 pass, lowest gap ───────────────────────────────
    candidates = [r for r in results if r["g6_ok"] and r["g7_ok"]]

    print("\n\n" + "=" * 95)
    print(f"CANDIDATES  (G6 ok AND G7 ok, sorted by gap ascending — {len(candidates)} rows)")
    print("=" * 95)
    if candidates:
        print(header)
        print("-" * 95)
        for r in candidates[:10]:   # top 10 — enough to make a decision
            print(
                f"{r['max_depth']:>5}  {r['min_child_weight']:>4}  {r['reg_lambda']:>7}  "
                f"${r['train_rmse']:>10,.0f}  ${r['test_rmse']:>9,.0f}  "
                f"{r['gap_pct']:>5.1f}%  {r['test_r2']:>8.4f}  "
                f"{r['overpriced_pct']:>10.1f}%  "
                f"{'YES':>6}  {'YES':>6}"
            )

        best = candidates[0]
        print(f"""
──────────────────────────────────────────────────────────────────────────────
SUGGESTED NEXT STEP (based on output — confirm before committing):

  Best candidate:
    max_depth        = {best['max_depth']}
    min_child_weight = {best['min_child_weight']}
    reg_lambda       = {best['reg_lambda']}
    gap              = {best['gap_pct']}%  (raw: {best['gap_raw']})
    test_r2          = {best['test_r2']}

  For ci_model_gate.py → GATE_THRESHOLDS["train_val_gap_max"]:
    You need a value >= {best['gap_raw']:.4f} to pass this config.
    Add a small buffer yourself — this script will not pick it for you.
──────────────────────────────────────────────────────────────────────────────""")
    else:
        print("  No candidates found where both G6 and G7 pass.")
        print("  Consider widening the grid (e.g. lower min_child_weight or reg_lambda).")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFull results saved → {OUTPUT_CSV}")
    print(f"Total sweep time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
