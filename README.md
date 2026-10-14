# Fraud & Risk Intelligence Platform

An end-to-end, reproducible fraud-risk ML engineering project built on the public
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset.

> **Scope disclaimer.** This is an educational / portfolio system. It is **not** a payment
> processor, a certified fraud-control system, financial advice, or suitable for real banking
> decisions. No regulatory compliance is claimed.

## Status

The project is built phase by phase (see `docs/execution/PHASE_STATUS.md`). Each phase has an
implementation branch `phase/NN-slug` and a test-only branch `phase/test-NN-slug`.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quality gates

```bash
python -m pytest
ruff check .
ruff format --check .
mypy src
git diff --check
```

The same checks run in CI (`.github/workflows/quality.yml`).

## Configuration

Settings are read from environment variables prefixed with `FRAUD_RISK_` (and from an optional,
Git-ignored `.env` file). See `.env.example` for every option and its default. Invalid values fail
fast with a `ConfigurationError` that names the variable but never echoes its value.

## Repository layout

```
src/fraud_risk/        reusable package code (typed, py.typed marker)
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
  cli/                 `fraud-risk` command-line entry point
tests/unit/            fast deterministic unit tests
tests/integration/     cross-module tests on synthetic fixtures
docs/                  architecture and design documentation
docs/execution/        phase execution log, status, decisions, blockers
```

See [`docs/architecture.md`](docs/architecture.md) for the design overview.

## Dataset

Download `creditcard.csv` manually from Kaggle and place it at `data/raw/creditcard.csv`
(see [`docs/data.md`](docs/data.md)). Then:

```bash
fraud-risk ingest            # structural checks + aggregate metadata as JSON
fraud-risk validate          # data contract validation -> reports/data_quality.json
fraud-risk split             # deterministic split -> reports/split_metadata.json (metadata only)
fraud-risk train-baseline    # dummy vs logistic regression -> reports/baseline_metrics.json
fraud-risk compare-models    # LR / RF / HGB / XGBoost / SMOTE comparison -> reports/model_comparison.*
fraud-risk train-anomaly     # Isolation Forest anomaly ranking -> reports/anomaly_metrics_all.json
fraud-risk evaluate-policy --policy configs/risk_policy.example.json --bundle artifacts/models/<id>
fraud-risk explain --bundle artifacts/models/<id>   # global + local explanations (not causal)
```

## Data and artifacts

The dataset, trained models, MLflow runs, reports, and `.env` files are local only and ignored by
Git. Never commit them.
