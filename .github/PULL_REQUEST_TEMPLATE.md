## Summary
<!-- What does this PR do and why? 1–3 sentences. -->


## Change Type
<!-- Check all that apply -->
- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature / model improvement
- [ ] 🔧 Pipeline / infrastructure change
- [ ] 📊 Experiment / hyperparameter update
- [ ] 📝 Documentation / config update
- [ ] 🧹 Refactor / cleanup (no behaviour change)

---

## Model Performance Delta
<!-- Fill in if this PR touches training, features, preprocessing, or evaluation.
     Compare against the current main branch baseline.
     Use data/sample/test_sample.csv for reproducible comparison. -->

| Metric | Baseline (main) | This PR | Delta |
|---|---|---|---|
| RMSE (test) | | | |
| R² (test) | | | |
| MAE (test) | | | |
| `biz_within_10pct` | | | |
| `biz_overpriced_pct` | | | |
| `pinball_gap_pct` | | | |
| CV Mean ± Std | | | |

**Best model this run:** <!-- e.g. XGBoost (specialist routing) -->

**Train/Val gap status:** <!-- Healthy / Warning / Overfit -->

---

## Deployment Gate Status
<!-- These mirror the production gates. Mark each as ✅ Pass / ❌ Fail / ➖ N/A -->

| Gate | Description | Status |
|---|---|---|
| G1 | RMSE within acceptable threshold | |
| G4 | Train/Val generalisation gap | |
| G6 | Cost-weighted R² | |
| G7 | Overpricing rate | |

**`deployment_ready` tag:** <!-- True / False -->

---

## How Was This Tested?
<!-- Describe what you ran. Check all that apply. -->
- [ ] `make test` — all tests pass, coverage ≥ 70%
- [ ] `make lint` — no ruff errors
- [ ] `make format` — no formatting changes
- [ ] `make type-check` — no mypy errors
- [ ] `make ci` — full local CI pipeline passed
- [ ] Ran `scripts/train_model.py` locally on sample data
- [ ] Verified FastAPI endpoints (`/health`, `/predict`) respond correctly
- [ ] Checked MLflow run for metric anomalies
- [ ] Ran `scripts/ci_model_gate.py` locally — all gates passed

---

## Reproducibility Checklist
- [ ] `random_state=42` (or `RANDOM_SEED` env var) set in all stochastic components
- [ ] No data leakage between train/val/test splits
- [ ] Preprocessor fitted only on train set (`preprocessor_v*.joblib` updated if changed)
- [ ] `models/pipeline_metadata.json` updated if pipeline version bumped
- [ ] New model artifacts include checksum files (`*_checksum.txt`)
- [ ] `.env.example` updated if new environment variables were added
- [ ] `mlruns/` contains the run for this experiment (don't delete)

---

## Breaking Changes
<!-- Does this PR change any of the following? If yes, describe the impact. -->
- [ ] API request/response schema (`api/schemas.py`)
- [ ] Preprocessor contract (feature names, transform order)
- [ ] Model artifact format or filenames
- [ ] Configuration keys in `configs/config.yaml`
- [ ] Docker image base or exposed ports

---

## Notes for Reviewer
<!-- Anything specific to look at, known limitations, or follow-up work. -->