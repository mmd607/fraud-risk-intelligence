# Architecture Overview

## Goals

A reproducible fraud-risk workflow: local ingestion → data contracts → leakage-safe
preprocessing → supervised classification → complementary anomaly detection → configurable risk
decisions → explainability → model lifecycle tracking → inference API → monitoring → operations
dashboard → release hygiene.

## Principles

- **Package first.** All reusable logic lives in `src/fraud_risk/`. Entry points (CLI, API,
  dashboard) are thin layers over package functions.
- **Typed boundaries.** Public interfaces are fully annotated; `mypy --strict` runs on `src/`.
- **Fail loudly, safely.** Deliberate errors derive from `FraudRiskError` and carry a stable
  `code`. Messages are actionable and never contain secrets or raw transaction records.
- **Privacy-safe logging.** `fraud_risk.core.logging` emits structured JSON. Context keys that look
  like secrets or transaction payloads are redacted. Log aggregates and identifiers, never rows.
- **Configuration by environment.** `fraud_risk.config.Settings` (pydantic-settings) provides safe
  local defaults so the package imports without services, data, or secrets.
- **Injected dependencies.** Filesystem paths, trackers, model services, and clocks are passed in,
  so tests run on synthetic fixtures without network access.
- **Local artifacts only.** Data, models, reports, and tracking stores are written to Git-ignored
  directories (`data/`, `artifacts/`, `reports/`).

## Module map

| Module | Responsibility |
|---|---|
| `fraud_risk.config.settings` | `Settings`, `load_settings()`, cached `get_settings()` |
| `fraud_risk.core.exceptions` | `FraudRiskError` hierarchy with stable error codes |
| `fraud_risk.core.logging` | JSON/console formatters, redaction, `configure_logging`, `log_event` |
| `fraud_risk.data.schema` | Column constants; `FEATURE_COLUMNS` excludes the target |
| `fraud_risk.data.ingestion` | `load_transactions()` structural checks + `DatasetMetadata` |
| `fraud_risk.data.contracts` | `DataContract`, `ColumnSpec`, `creditcard_contract()` |
| `fraud_risk.data.validation` | `validate_dataframe()` → issues with stable `DQ0xx` codes and severities |
| `fraud_risk.data.quality_report` | `build_quality_report()` → JSON report of issues + aggregate statistics |
| `fraud_risk.data.prepare` | `load_validated_dataset()` (load + refuse invalid data) |
| `fraud_risk.data.splitting` | Stratified (duplicate-grouped) / temporal splits, `SplitMetadata` |
| `fraud_risk.features.preprocessing` | Unfitted `ColumnTransformer` with per-column strategies |
| `fraud_risk.cli.main` | `fraud-risk` CLI (`ingest`, `validate`, `split`, ...) |

Later phases extend this table as modules are added.

## Error hierarchy

```
FraudRiskError (FRAUD_RISK_ERROR)
├── ConfigurationError (CONFIGURATION_ERROR)
├── DataError (DATA_ERROR)
│   ├── DatasetNotFoundError (DATA_NOT_FOUND)
│   ├── DataReadError (DATA_UNREADABLE)
│   ├── SchemaError (DATA_SCHEMA_INVALID, plus specific DATA_* codes)
│   └── DataValidationError (DATA_VALIDATION_FAILED)
├── ModelError (MODEL_ERROR)
└── DependencyUnavailableError (DEPENDENCY_UNAVAILABLE)
```
