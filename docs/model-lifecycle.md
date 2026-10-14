# Experiment Tracking and Model Lifecycle

## Tracking (`fraud_risk.registry.tracking`)

A narrow `ExperimentTracker` protocol (`start_run(name)` → run with `log_params`, `log_metrics`,
`set_tags`) with three implementations:

| Tracker | When | Needs |
|---|---|---|
| `NullTracker` | Default (`FRAUD_RISK_TRACKING_ENABLED=false`): unit tests, offline use | nothing |
| `InMemoryTracker` | Tests and dry runs | nothing |
| `MlflowTracker` | `FRAUD_RISK_TRACKING_ENABLED=true` | `pip install "fraud-risk[tracking]"` |

The MLflow tracking URI is `FRAUD_RISK_TRACKING_URI`, defaulting to the **local, Git-ignored**
`sqlite:///<repo>/artifacts/mlflow/mlflow.db`. No remote server is needed or contacted unless you
configure one.

What is logged: scalar parameters, finite metrics, string tags, and artifact **references**
(bundle path, model SHA-256, dataset SHA-256, code revision). No binaries are uploaded, and no
data rows. Keys that look like secrets or payloads (`api_key`, `token`, `features`,
`transaction`, …) are dropped by `sanitize_params` / `sanitize_metrics`.

## Registry (`fraud_risk.registry.model_registry.LocalModelRegistry`)

A JSON index at `artifacts/registry/registry.json` (Git-ignored), written atomically
(temp file + replace). A corrupt index is never overwritten.

Each `RegistryEntry` (pydantic-validated) records: `version` (auto-incremented), `model_id`,
`model_name`, `stage`, `bundle_dir`, `model_sha256`, `dataset_sha256`, `data_contract_version`,
`feature_names`, `training_timestamp` (from the manifest), `registered_at`, `code_revision`,
headline validation/test metrics, and the full `stage_history`.

**Registration** verifies the bundle's SHA-256 against its manifest **without unpickling it**,
rejects duplicates, and always creates a `candidate`.

### Lifecycle

```
candidate ──► approved ──► retired
     └──────────────────────► retired
```

- **No automatic promotion.** Every transition is an explicit command with `--actor` and
  `--reason`, appended to `stage_history`.
- At most **one** `approved` version. To switch models, retire the current one first.
- `retired` is terminal.
- "Approved" means "selected to serve in this local demo". It is not a production or regulatory
  approval.

```bash
fraud-risk registry register --bundle artifacts/models/<model_id> --actor "<your name>"
fraud-risk registry list
fraud-risk registry transition --version 2 --to approved --actor "<your name>" --reason "<why>"
fraud-risk registry transition --version 2 --to retired  --actor "<your name>" --reason "<why>"
```

`code_revision` is `git rev-parse HEAD` **at registration time** (suffixed `-dirty` with
uncommitted tracked changes), or `null` outside a Git checkout. It identifies the code that
registered the model, which may differ from the commit that trained it. The manifest's
`library_versions` and `created_at` complement it.

## Current local state (verified)

Two real bundles are registered as **candidates**:

| Version | Model id | Test AP | Stage |
|---|---|---|---|
| 1 | `logistic_regression-20261014t045734` | 0.6833 | candidate |
| 2 | `hist_gradient_boosting-20261014t050441` | 0.8163 | candidate |

Neither has been approved. Approval is deliberately left to the user.

## Storage and cleanup

Everything lives under `artifacts/` (Git-ignored): `artifacts/models/<model_id>/` bundles,
`artifacts/registry/registry.json`, and `artifacts/mlflow/` (SQLite store and run files). To
clean up, retire entries first, then delete bundle directories manually. Never commit any of
these files.
