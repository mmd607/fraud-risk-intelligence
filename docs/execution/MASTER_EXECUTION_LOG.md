# Master Execution Log

Actual work performed, in order. Commands run inside `.venv` (Python 3.13.0, Windows 11).
Specification source: `Fraud_Risk_Intelligence_Ultra_Master_Prompts.zip` (README, PROJECT_PLAN,
14 phase prompts, 14 test prompts, quality-gates, dataset-notes, manifest — all present).
Dataset source: local `dataset.zip` containing `creditcard.csv` (150,828,752 bytes).

## Phase 00 — Project Foundation

- **Implementation branch:** `phase/00-foundation` (base: `main` @ `42804b6`, empty root commit)
- **Test branch:** `phase/test-00-foundation` (base: `phase/00-foundation`)
- **Implementation commits:** `d9fac77` feat: add project foundation; `a384535` docs: remove trailing blank line
- **Test commit:** `e376c0f` test: cover foundation contract
- **Files:** `pyproject.toml`, `.gitignore`, `.gitattributes`, `.env.example`, `README.md`,
  `docs/architecture.md`, `.github/workflows/quality.yml`, `src/fraud_risk/{__init__,py.typed}`,
  `src/fraud_risk/config/settings.py`, `src/fraud_risk/core/{exceptions,logging}.py`,
  `tests/conftest.py`, `tests/unit/test_{settings,exceptions,logging}.py`,
  `tests/unit/test_foundation_contract.py` (test branch)
- **Commands and results (test branch):** `python -m pytest` → 52 passed; `ruff check .` → pass;
  `ruff format --check .` → pass; `mypy src` (strict) → no issues in 6 files;
  `git diff --check main..HEAD` → clean.
- **Dataset:** not used in this phase.
- **Limitations:** CI workflow written but not executed (no remote, push not authorized).
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 01 — Dataset Ingestion

- **Implementation branch:** `phase/01-data-ingestion` (base: `phase/test-00-foundation` @ `cb0a7ab`)
- **Test branch:** `phase/test-01-data-ingestion` (base: `phase/01-data-ingestion`)
- **Implementation commit:** `5612d11` feat: add local dataset ingestion with schema checks, metadata, and CLI
- **Test commit:** `b20afee` test: cover data ingestion edge cases and CLI behaviour
- **Files:** `src/fraud_risk/data/{schema,ingestion}.py`, `src/fraud_risk/cli/main.py`,
  `src/fraud_risk/core/exceptions.py` (data errors), settings `dataset_path`, `pyproject.toml`
  (pandas/numpy, `fraud-risk` script), `docs/data.md`, `tests/support/synthetic.py`,
  `tests/unit/test_ingestion.py`, `tests/integration/test_cli_ingest.py`,
  `tests/unit/test_ingestion_edge_cases.py` (test branch)
- **Commands and results (test branch):** `python -m pytest` → 90 passed; `ruff check .` → pass;
  `ruff format --check .` → pass; `mypy src` → no issues in 11 files; `git diff --check` → clean.
- **Test-branch note:** one new test initially failed because the test helper produced `\r\n`
  line endings on Windows (a test bug, not a product defect); fixed in the test before commit.
- **Dataset:** real `creditcard.csv` present locally (Git-ignored). `fraud-risk ingest` verified:
  284,807 rows, 31 columns, Class 0 = 284,315 / 1 = 492 (0.1727 %), 0 missing cells,
  1,081 exact duplicate rows, SHA-256 `76274b69…51a89`.
- **Limitations:** none known for this scope.
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 02 — Data Quality and Validation

- **Implementation branch:** `phase/02-data-validation` (base: `phase/test-01-data-ingestion` @ `c363ad9`)
- **Test branch:** `phase/test-02-data-validation` (base: `phase/02-data-validation`)
- **Implementation commit:** `39eb48f` feat: add data contract, validation issues, and JSON quality report
- **Test commit:** `6deb708`
  (`test: cover data validation branches, issue-code stability, and CLI`)
- **Files:** `src/fraud_risk/data/{contracts,validation,quality_report}.py`, CLI `validate`,
  `DataValidationError`, `docs/data-quality.md`, `tests/unit/test_validation.py`,
  `tests/unit/test_validation_branches.py` (test branch)
- **Commands and results (test branch):** `python -m pytest` → 134 passed; `ruff check .` → pass;
  `ruff format --check .` → pass; `mypy src` → no issues in 14 files; `git diff --check` → clean.
- **Dataset:** `fraud-risk validate` on the real CSV → valid; one `DQ010` warning
  (1,081 exact duplicate rows, 0.3796 %). No nulls, infinities, range violations, label conflicts,
  or ordering issues. Report written to Git-ignored `reports/data_quality.json`.
- **Limitations:** duplicates are reported, not removed (by design); handled in Phase 03 docs.
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 03 — Preprocessing Pipeline

- **Implementation branch:** `phase/03-preprocessing` (base: `phase/test-02-data-validation` @ `1749bcc`)
- **Test branch:** `phase/test-03-preprocessing` (base: `phase/03-preprocessing` @ `561f7b3`)
- **Implementation commits:** `561f7b3` feat: add leakage-safe splitting and configurable preprocessing
  pipeline; `453ec92` fix: stop split CLI from overwriting split metadata with the command payload
- **Test commits:** `e42a427` test: cover splitting/preprocessing leakage and CLI (2 tests expose
  defect); `b707ff7` merge of the implementation fix into the test branch
- **Defect found by the test branch:** `fraud-risk split --output X` wrote split metadata to `X`,
  then the generic CLI writer overwrote `X` with the command payload. Two T03 tests failed
  (`KeyError: 'seed'`). The failing tests were committed as evidence, the fix was made on the
  implementation branch (`453ec92`, `--output` now maps to `metadata_output`), and the fix was
  merged into the test branch (no rebase, no history rewrite). Tests pass afterwards.
- **Files:** `src/fraud_risk/data/{splitting,prepare}.py`, `src/fraud_risk/features/preprocessing.py`,
  CLI `split`, `pyproject.toml` (scikit-learn, joblib, mypy overrides), `docs/validation-strategy.md`,
  `tests/unit/test_splitting_preprocessing.py`, `tests/unit/test_preprocessing_leakage.py` (test branch)
- **Commands and results (test branch, after fix):** `python -m pytest` → 176 passed;
  `ruff check .` → pass; `ruff format --check .` → pass; `mypy src` → no issues in 18 files;
  `git diff --check` → clean.
- **Dataset:** real splits verified (seed 42). Stratified grouped: train 170,894 / 294 frauds,
  validation 56,965 / 102, test 56,948 / 96. Temporal: train 0.211 %, validation 0.100 %,
  test 0.132 % fraud rate. Only metadata written (`reports/split_metadata.json`, Git-ignored).
- **Limitations:** `hour_cyclical` assumes the window starts at midnight (undocumented by source).
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 04 — Baseline Classification

- **Implementation branch:** `phase/04-baseline-model` (base: `phase/test-03-preprocessing` @ `a63bbc2`)
- **Test branch:** `phase/test-04-baseline-model` (base: `phase/04-baseline-model`)
- **Implementation commit:** `cc41fc7` feat: add baseline training with AP-first metrics, validation
  thresholds, and model bundles
- **Test commit:** `765e983` test: cover baseline metrics edge cases, artifact integrity, and leakage guards
- **Files:** `src/fraud_risk/evaluation/metrics.py`, `src/fraud_risk/models/baseline.py`,
  `src/fraud_risk/training/{artifacts,baseline}.py`, CLI `train-baseline`, settings `models_dir`,
  `docs/modeling-baseline.md`, `tests/unit/test_metrics.py`,
  `tests/integration/test_baseline_training.py`, `tests/unit/test_baseline_contract.py` (test branch)
- **Commands and results (test branch):** `python -m pytest` → 224 passed; `ruff check .` → pass;
  `ruff format --check .` → pass; `mypy src` → no issues in 25 files; `git diff --check` → clean.
- **Real-data run:** `fraud-risk train-baseline` (≈17 s). Model id
  `logistic_regression-20261014t045734`. Test partition (56,948 rows, 96 frauds), thresholds from
  validation (max-F1):
  - Dummy prior: AP 0.0017, ROC-AUC 0.500, threshold 0.00172.
  - Logistic regression: **AP 0.6833**, ROC-AUC 0.9621, precision 0.8706, recall 0.7708,
    F1 0.8177 at threshold 0.99999466; confusion `[[56841, 11], [22, 74]]`.
- **Disclosure:** an earlier run selected preprocessing by maximum CV mean (test AP 0.6937).
  The selection rule was changed to a one-standard-error rule because of training-fold CV noise;
  the reported numbers are from the final rule. Documented in `docs/modeling-baseline.md`.
- **Limitations:** single-split estimates with ~100 positives per evaluation partition; scores
  are uncalibrated (class weighting).
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 05 — Advanced Imbalanced-Learning Models

- **Implementation branch:** `phase/05-advanced-models` (base: `phase/test-04-baseline-model` @ `65553e4`)
- **Test branch:** `phase/test-05-advanced-models` (base: `phase/05-advanced-models`)
- **Implementation commit:** `6448e38` feat: add candidate model comparison with training-fold CV selection and optional boosters
- **Test commit:** `aabbbc3` test: cover model comparison determinism, SMOTE isolation, and test-set discipline
- **Files:** `src/fraud_risk/models/candidates.py`, `src/fraud_risk/training/experiments.py`, CLI
  `compare-models`, extras `[boost]` (xgboost) / `[imbalance]` (imbalanced-learn),
  `docs/model-comparison.md`, `tests/integration/test_model_comparison.py`,
  `tests/unit/test_advanced_models_contract.py` + logger-reset fixture in `tests/conftest.py` (test branch)
- **Commands and results (test branch):** `python -m pytest` → 243 passed; `ruff check .` → pass;
  `ruff format --check .` → pass; `mypy src` → no issues in 27 files; `git diff --check` → clean.
- **Test-branch notes:** one new test first failed because synthetic validation and test partitions
  had equal sizes (test logic bug; fixed to compare row indices). A test-only conftest fixture now
  resets package log handlers between tests (removed closed-stream logging noise). No product defect.
- **Real-data run** (`fraud-risk compare-models`, 1 min 57 s, xgboost 3.4.1 and imbalanced-learn
  0.14.2 installed): CV AP — LR 0.7362, RF 0.8253, **HGB 0.8430 (selected)**, XGBoost 0.8407,
  LR+SMOTE 0.7474. Selected HGB on test (threshold 0.9037 from validation): **AP 0.8163**,
  ROC-AUC 0.9696, precision 0.9730, recall 0.7500, F1 0.8471, confusion `[[56850, 2], [24, 72]]`.
  Model id `hist_gradient_boosting-20261014t050441`.
- **Limitations:** HGB and XGBoost are within CV noise of each other; single split; uncalibrated scores.
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 06 — Anomaly Detection Track

- **Implementation branch:** `phase/06-anomaly-detection` (base: `phase/test-05-advanced-models` @ `88695a0`)
- **Test branch:** `phase/test-06-anomaly-detection` (base: `phase/06-anomaly-detection`)
- **Implementation commit:** `e0005d1` feat: add isolation-forest anomaly track with label-isolated fitting and ranking evaluation
- **Test commit:** `e0e047b` test: cover anomaly label isolation, score semantics, determinism, and CLI
- **Files:** `src/fraud_risk/anomaly/{detectors,evaluation}.py`, `src/fraud_risk/training/anomaly.py`,
  CLI `train-anomaly`, `docs/anomaly-detection.md`, `tests/unit/test_anomaly.py`,
  `tests/unit/test_anomaly_contract.py` (test branch)
- **Commands and results (test branch):** `python -m pytest` → 268 passed; `ruff check .` → pass;
  `ruff format --check .` → pass; `mypy src` → no issues in 31 files; `git diff --check` → clean.
- **Real-data run:** Isolation Forest (200 trees, seed 42). Unsupervised (`fit_on=all`, 170,894 rows):
  validation AP 0.2629, **test AP 0.1291**, test ROC-AUC 0.9417, test P@100 0.27 / R@100 0.281.
  Legitimate-only (label-informed): test AP 0.1095. Supervised reference (Phase 05): 0.8163.
- **Limitations:** anomaly ≠ fraud; not combined with supervised scores in this phase.
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 07 — Risk Scoring and Decision Engine

- **Implementation branch:** `phase/07-risk-engine` (base: `phase/test-06-anomaly-detection` @ `6b026a3`)
- **Test branch:** `phase/test-07-risk-engine` (base: `phase/07-risk-engine` @ `4098d22`, fast-forward)
- **Implementation commits:** `2899f2e` feat: add configurable risk policy, decision engine, review queue,
  and illustrative costs; `4098d22` test: construct strict policy through validation in risk engine unit test
- **Process note:** `2899f2e` was committed while one of its own unit tests failed. The pytest gate
  had been piped through `tail`, which masked the exit status. The failure was a test bug
  (`model_copy(update=...)` bypasses pydantic validation). Fixed in `4098d22` before the test
  branch received any commits. From here on, gate exit codes are checked explicitly.
- **Test commit:** `cfe3226` test: cover risk policy precedence, independence, log privacy, costs, and CLI
- **Files:** `src/fraud_risk/policy/config.py`, `src/fraud_risk/risk/{engine,reason_codes,review_queue,costs}.py`,
  `configs/risk_policy.example.json`, CLI `evaluate-policy`, `docs/risk-policy.md`,
  `tests/unit/test_risk_engine.py`, `tests/unit/test_risk_policy_contract.py` (test branch)
- **Commands and results (test branch):** `python -m pytest` → 314 passed (exit 0); `ruff check .` → pass;
  `ruff format --check .` → pass; `mypy src` → no issues in 38 files; `git diff --check` → clean.
- **Real-data run:** example policy (review 0.01, block 0.95; read off validation scores) on the
  Phase 05 HGB bundle. Validation: capture 0.892, block precision 0.957, illustrative cost 1,218 vs
  10,200 approve-all. Test (once): capture 0.823, block precision 0.973, cost 1,822 vs 9,600.
- **Limitations:** costs and thresholds are illustrative; scores uncalibrated.
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 08 — Explainability

- **Implementation branch:** `phase/08-explainability` (base: `phase/test-07-risk-engine` @ `445f657`)
- **Test branch:** `phase/test-08-explainability` (base: `phase/08-explainability` @ `660a455`)
- **Implementation commits:** `660a455` feat: add explainability (permutation importance,
  linear/SHAP/occlusion local explanations); `b83fae8` fix: report SHAP output space per model family
- **Test commits:** `a7ebfcf` test: cover explainability … (RF output-space test exposes defect);
  `f2a0a8b` merge of the implementation fix into the test branch
- **Defect found by the test branch:** SHAP explanations for random forests were labelled
  `output_space="log_odds"`, but TreeExplainer explains RF in probability space. The failing test was
  committed as evidence, fixed on the implementation branch, and merged (no history rewrite).
- **Files:** `src/fraud_risk/explainability/{explainers,report}.py`, CLI `explain` (and the CLI parser
  split into command groups), extra `[explain]` (shap), `docs/explainability.md`,
  `tests/unit/test_explainability.py`, `tests/unit/test_explainability_contract.py` (test branch)
- **Commands and results (test branch, after fix):** `python -m pytest` → 337 passed (exit 0);
  `ruff check .` → pass; `ruff format --check .` → pass; `mypy src` → no issues in 41 files;
  `git diff --check` → clean.
- **Real-data run:** `fraud-risk explain` on the HGB bundle (shap 0.53.0, 23 s): top permutation
  importance V14 0.1648 AP drop, then Amount 0.0069, V10 0.0068, V4 0.0063, V12 0.0062. Local SHAP
  for top-scored validation frauds is dominated by V14 (+9.9 to +10.9 log-odds). LR bundle (linear):
  V14 0.0819, V10 0.0420, V8 0.0350.
- **Limitations:** association, not causation; PCA features have no business meaning; noisy below the top feature.
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 09 — Experiment Tracking and Model Registry

- **Implementation branch:** `phase/09-model-registry` (base: `phase/test-08-explainability` @ `fb655b4`)
- **Test branch:** `phase/test-09-model-registry` (base: `phase/09-model-registry`)
- **Implementation commit:** `a533af6` feat: add local model registry with explicit lifecycle and optional MLflow tracking
- **Test commit:** `a2c93b1` test: cover registry lifecycle safety, metadata validation, offline tracking, and CLI
- **Files:** `src/fraud_risk/registry/{tracking,model_registry}.py`, CLI `registry list|register|transition`,
  settings `tracking_enabled`/`tracking_uri`/`registry_dir`, extra `[tracking]` (mlflow),
  `docs/model-lifecycle.md`, `tests/unit/test_registry.py`, `tests/unit/test_registry_contract.py` (test branch)
- **Commands and results (test branch):** `python -m pytest` → 367 passed, 1 third-party warning
  (MLflow/SQLAlchemy deprecation); `ruff check .` → pass; `ruff format --check .` → pass;
  `mypy src` → no issues in 44 files; `git diff --check` → clean.
- **MLflow:** 3.17.1 installed in the venv (install took ~15 min, network-bound). A real local SQLite
  tracking test passes; both real registry entries were logged to `artifacts/mlflow/mlflow.db`
  (runs `dd8c2dfc…`, `24541fb0…`). Tracking stays disabled by default.
- **Registry state:** v1 `logistic_regression-20261014t045734` (test AP 0.6833) and v2
  `hist_gradient_boosting-20261014t050441` (test AP 0.8163) registered as **candidates**.
  **Neither was approved.** Promotion is left to the user.
- **Limitations:** `code_revision` is captured at registration, not training time (documented).
- **Status:** TESTED → READY_FOR_REVIEW

## Phase 10 — Inference API

- **Implementation branch:** `phase/10-inference-api` (base: `phase/test-09-model-registry` @ `d99dacb`)
- **Test branch:** `phase/test-10-inference-api` (base: `phase/10-inference-api` @ `7bf6a23`)
- **Implementation commits:** `7bf6a23` feat: add FastAPI inference service…; `3abfdea` fix: return the error envelope for routing errors (404/405)
- **Test commits:** `3ba72c4` test: cover API wiring, model selection, log privacy, error contract (exposed defect); `4d86a47` merge of the fix
- **Defect found by the test branch:** 404/405 responses used Starlette's default `{"detail": …}` body instead of the documented error envelope. Fixed on the implementation branch and merged.
- **Commands and results (test branch, after fix):** `python -m pytest` → 399 passed (exit 0, 2 third-party deprecation warnings); `ruff check .` → pass; `mypy src` → no issues in 49 files.
- **Real smoke test:** uvicorn serving the HGB bundle via `FRAUD_RISK_MODEL_BUNDLE_PATH`; the top test fraud was blocked (R003, SHAP top V14/V12/V10), a legit row was approved, and invalid input returned 422. Logs held metadata only.
- **Status:** TESTED → READY_FOR_REVIEW

## RESUME POINT (session stopped: usage limit)

- Phases 00–10 are READY_FOR_REVIEW. **Next action:** from `phase/test-10-inference-api`, create `phase/11-monitoring` and implement Phase 11 (monitoring/drift), then Phases 12 (Streamlit dashboard) and 13 (Docker/CI/integration tests/model card/release checklist).
- Before resuming: check `git status`; the venv is `.venv` (Python 3.13, includes xgboost, imbalanced-learn, shap, mlflow, fastapi). Always check the pytest exit code directly (do not pipe it).
