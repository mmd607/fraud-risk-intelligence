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
