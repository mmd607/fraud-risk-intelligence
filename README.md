# Fraud & Risk Intelligence Platform

An end-to-end, reproducible fraud-risk machine learning system built on the public
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset: validated ingestion, leakage-safe preprocessing, imbalanced classification, a separate
anomaly-detection signal, a configurable risk decision engine, explainability, a local model
registry with explicit promotion, and a FastAPI inference service.

> **Scope disclaimer.** This is an educational / portfolio system. It is **not** a payment
> processor, a certified fraud-control system, financial advice, or suitable for real banking
> decisions. No regulatory compliance is claimed.

## Status

| Area | State |
|---|---|
| Data ingestion, contracts, validation, quality report | Implemented |
| Preprocessing, baseline and advanced models (LR, RF, HGB, XGBoost, SMOTE) | Implemented |
| Isolation Forest anomaly ranking (separate score) | Implemented |
| Risk policy, decision engine, review queue, cost evaluation | Implemented |
| Explainability (permutation importance, linear, SHAP, occlusion) | Implemented |
| Experiment tracking and local model registry | Implemented |
| Inference API (FastAPI, strict schemas, error contract) | Implemented |
| Monitoring and drift signals | Not started |
| Risk operations dashboard | Not started |
| Production hardening and release | Not started |

The work was built phase by phase, each phase with an implementation branch and an independent
test-only branch. Details: [`docs/execution/PHASE_STATUS.md`](docs/execution/PHASE_STATUS.md).

## Results (real data, run `hist_gradient_boosting-20261014t050441`)

Seeded, stratified, duplicate-grouped 60/20/20 split. Average Precision (AP) is the primary
metric because the positive class is about 0.17 % of transactions.

| Candidate | CV AP (mean ± std) | Validation AP |
|---|---|---|
| Logistic regression (C=0.1) | 0.736 ± 0.015 | 0.810 |
| Random forest | 0.825 ± 0.035 | 0.886 |
| **HistGradientBoosting (balanced) — selected** | **0.843 ± 0.038** | 0.895 |
| XGBoost | 0.841 ± 0.031 | 0.896 |
| Logistic regression + SMOTE | 0.747 ± 0.021 | 0.824 |

Selected model on the held-out test partition (56,948 rows, 96 frauds), threshold chosen on
validation only:

| AP | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.816 | 0.970 | 0.973 | 0.750 | 0.847 |

HistGradientBoosting and XGBoost are statistically indistinguishable on this data (difference
0.002 vs fold std ≈ 0.03). With 96 test frauds, one fraud more or less moves recall by about one
point. Scores are rankings, not calibrated probabilities. Full protocol and caveats:
[`docs/model-comparison.md`](docs/model-comparison.md).

## Quick start

Requires Python 3.11+ (developed on 3.13).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional extras: `boost` (XGBoost), `imbalance` (SMOTE), `explain` (SHAP), `tracking` (MLflow),
`api` (FastAPI + Uvicorn).

## Dataset

Download `creditcard.csv` manually from Kaggle and place it at `data/raw/creditcard.csv`
(see [`docs/data.md`](docs/data.md)). The dataset is not included in this repository.

```bash
fraud-risk ingest            # structural checks + aggregate metadata as JSON
fraud-risk validate          # data contract validation -> reports/data_quality.json
fraud-risk split             # deterministic split -> reports/split_metadata.json
fraud-risk train-baseline    # dummy vs logistic regression -> reports/baseline_metrics.json
fraud-risk compare-models    # LR / RF / HGB / XGBoost / SMOTE comparison
fraud-risk train-anomaly     # Isolation Forest anomaly ranking
fraud-risk evaluate-policy --policy configs/risk_policy.example.json --bundle artifacts/models/<id>
fraud-risk explain --bundle artifacts/models/<id>   # global + local explanations (not causal)
fraud-risk registry register --bundle artifacts/models/<id> --actor "<name>"
fraud-risk registry transition --version N --to approved --actor "<name>" --reason "<why>"
```

## Inference API

```bash
pip install -e ".[api,explain]"
FRAUD_RISK_MODEL_BUNDLE_PATH=artifacts/models/<id> uvicorn fraud_risk.api.app:create_app --factory
```

Endpoints, schemas and the error contract: [`docs/api.md`](docs/api.md).

## Configuration

Settings are read from environment variables prefixed with `FRAUD_RISK_` (and from an optional,
Git-ignored `.env` file). See [`.env.example`](.env.example) for every option and its default.
Invalid values fail fast with a `ConfigurationError` that names the variable but never echoes its
value.

## Testing

```bash
python -m pytest
ruff check .
ruff format --check .
mypy src
```

The same checks run in CI ([`.github/workflows/quality.yml`](.github/workflows/quality.yml)).
Last local run: **399 passed** (Python 3.13, Windows 11, 2026-11-19). Tests use synthetic fixtures
and do not need the Kaggle file.

## Architecture

```
src/fraud_risk/
  config/              settings
  core/                domain exceptions, structured logging
  data/                schema, ingestion, contracts, validation, quality report
  features/            leakage-safe sklearn preprocessing
  evaluation/          imbalanced-classification metrics, threshold selection
  models/              model factories (unfitted sklearn pipelines)
  anomaly/             unsupervised anomaly detection (separate score)
  training/            training workflows, model bundles
  policy/, risk/       risk policy config, decision engine, review queue, costs
  explainability/      permutation importance, linear/SHAP/occlusion explanations
  registry/            experiment tracking + local model registry (explicit promotion)
  services/, api/      inference service + FastAPI app
  cli/                 `fraud-risk` command-line entry point
tests/unit/            fast deterministic unit tests
tests/integration/     cross-module tests on synthetic fixtures
docs/                  architecture and design documentation
```

Design overview: [`docs/architecture.md`](docs/architecture.md). Other documents:
[data quality](docs/data-quality.md) · [validation strategy](docs/validation-strategy.md) ·
[baseline](docs/modeling-baseline.md) · [model comparison](docs/model-comparison.md) ·
[anomaly detection](docs/anomaly-detection.md) · [risk policy](docs/risk-policy.md) ·
[explainability](docs/explainability.md) · [model lifecycle](docs/model-lifecycle.md)

## Limitations

- Single public dataset with anonymised PCA features; results do not transfer to real payment traffic.
- Single-split test estimates carry wide uncertainty (96 test frauds).
- No monitoring, drift detection, dashboard or production hardening yet (see Status).
- The API is a local service: single process, in-memory model, no deployment.

## Data and artifacts

The dataset, trained models, MLflow runs, reports and `.env` files are local only and ignored by
Git.

## License

Code: MIT (see [LICENSE](LICENSE)). The dataset is not distributed here; see the Kaggle page for
its terms.

## Author

Mohammad Homaiyan — [@mmd607](https://github.com/mmd607)
