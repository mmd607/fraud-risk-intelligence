# Data Quality and Validation

## Components

| Module | Role |
|---|---|
| `fraud_risk.data.contracts` | `DataContract` / `ColumnSpec`: expected columns, kinds, ranges, target domain, sanity bounds. `creditcard_contract()` is the default. |
| `fraud_risk.data.validation` | `validate_dataframe(frame, contract)` → `ValidationReport` of `ValidationIssue`s. Never mutates or drops data. |
| `fraud_risk.data.quality_report` | `build_quality_report(...)` → `QualityReport` (validation + per-column aggregates + target counts), `write_json(path)`. |

CLI: `fraud-risk validate [--path CSV] [--report reports/data_quality.json]`. Exit code `1` with
`DATA_VALIDATION_FAILED` when any error-severity issue exists; the report is written either way.

## Severities

- **error** — do not train or evaluate on this data until resolved.
- **warning** — usable, but a human should review.
- **info** — noteworthy facts that are not problems.

## Issue codes (stable; never renumbered)

| Code | Name | Severity | Meaning |
|---|---|---|---|
| DQ001 | EMPTY_DATASET | error | No rows |
| DQ002 | MISSING_COLUMN | error | A contract column is absent |
| DQ003 | UNEXPECTED_COLUMN | warning | Column not in the contract (never used as a feature) |
| DQ004 | NON_NUMERIC_COLUMN | error | Column dtype is not numeric |
| DQ005 | NULL_VALUES | error | Nulls in a non-nullable column (no imputation is done) |
| DQ006 | NON_FINITE_VALUES | error | ±inf values |
| DQ007 | TARGET_OUT_OF_DOMAIN | error | Target values outside {0, 1} |
| DQ008 | SINGLE_CLASS_TARGET | error | Only one class present; supervised training impossible |
| DQ009 | SUSPICIOUS_POSITIVE_RATE | warning | Fraud rate outside the sanity range (default 0.01 %–5 %) |
| DQ010 | DUPLICATE_ROWS | warning (info if ≤ `max_duplicate_fraction`, default 0) | Exact duplicate rows, kept unchanged |
| DQ011 | VALUE_OUT_OF_RANGE | error | Values outside a column's declared range (`Time`, `Amount` ≥ 0) |
| DQ012 | TIME_NOT_SORTED | info | `Time` is not non-decreasing |
| DQ013 | BELOW_MIN_ROWS | error | Fewer rows than `min_rows` |
| DQ014 | CONFLICTING_DUPLICATE_LABELS | warning | Identical feature vectors with different labels |

Messages contain column names and counts only — never cell values or rows.

## Real-dataset validation result (verified locally)

`fraud-risk validate` on the supplied `creditcard.csv` (SHA-256 `76274b69…51a89`): **valid**, with
one finding:

- `DQ010` warning — 1,081 exact duplicate rows (0.3796 %), kept unchanged.

No nulls, infinities, out-of-range values, label conflicts, or ordering issues were found, and
the positive rate (0.1727 %) is within the sanity range.

**Consequence for later phases.** Exact duplicates can land in both training and test splits,
which slightly inflates evaluation. Phase 03 documents how splitting handles this.

## Known dataset limitations

- **Anonymised PCA features.** `V1`–`V28` are PCA components of undisclosed original features.
  They cannot be interpreted as human-readable causes, and explanations over them describe model
  behaviour only.
- **Time span.** `Time` is seconds since the first transaction; the file covers about two days
  (max ≈ 172,792 s). It is not a calendar timestamp, and two days cannot capture seasonality or
  long-term drift.
- **Extreme imbalance.** 492 frauds in 284,807 rows (0.1727 %). Accuracy is meaningless as a
  primary metric (predicting "legitimate" always gives 99.83 %).
- **Provenance.** Published by the Machine Learning Group of ULB (Université Libre de Bruxelles)
  with Worldline; transactions by European cardholders in September 2013. Only the Kaggle
  description documents how the data was collected; labels may contain undetected fraud.
- **Amount scale.** `Amount` is not PCA-transformed and is heavily right-skewed.
