# Data Ingestion

## Source

Kaggle **Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`):
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>. The dataset is distributed under the
Kaggle-listed license (Open Database License, ODbL, at the time of writing — verify on the
dataset page). It is **never** committed to this repository.

## Manual download (no automatic download)

1. Sign in to Kaggle and download the dataset archive from the URL above.
2. Unzip it and place `creditcard.csv` at `data/raw/creditcard.csv` (Git-ignored), **or** set
   `FRAUD_RISK_DATASET_PATH` to its location.
3. Check it:

```bash
fraud-risk ingest                       # uses FRAUD_RISK_DATASET_PATH / data/raw/creditcard.csv
fraud-risk ingest --path other.csv --output reports/ingestion.json
```

The loader requires no network access and no Kaggle credentials.

## Expected schema

| Column | Meaning | Loaded dtype |
|---|---|---|
| `Time` | Seconds elapsed since the first transaction in this dataset version (not a calendar timestamp) | float64 |
| `V1`–`V28` | Anonymised PCA components; no human-readable meaning | float64 |
| `Amount` | Transaction amount | float64 |
| `Class` | Target: `0` legitimate, `1` fraud | int64 |

`fraud_risk.data.schema.FEATURE_COLUMNS` (Time, V1–V28, Amount) is the canonical model-input order.
The target `Class` is never part of it.

## Ingestion checks (`fraud_risk.data.ingestion.load_transactions`)

| Check | Behaviour | Error code |
|---|---|---|
| File exists | `DatasetNotFoundError` with download instructions | `DATA_NOT_FOUND` |
| File readable / UTF-8 / parseable | `DataReadError` | `DATA_UNREADABLE` |
| Header present, file not empty, at least one row | `SchemaError` | `DATA_EMPTY` |
| No duplicate headers | `SchemaError` | `DATA_DUPLICATE_HEADER` |
| No empty header names | `SchemaError` | `DATA_EMPTY_HEADER` |
| All required columns present | `SchemaError` (lists missing names) | `DATA_MISSING_COLUMNS` |
| No extra columns (unless `allow_extra_columns=True`) | `SchemaError` | `DATA_EXTRA_COLUMNS` |
| All required columns numeric | `SchemaError` (column: count, no values) | `DATA_NON_NUMERIC` |
| Target has no missing values | `SchemaError` | `DATA_TARGET_MISSING` |
| Target only contains 0/1 | `SchemaError` | `DATA_TARGET_INVALID` |
| Missing feature values | **Reported** in metadata + warning, not imputed | — |
| Duplicate rows | **Counted** in metadata, not dropped | — |

**Extra columns.** Rejected by default to surface schema drift early. When explicitly allowed,
they are kept in the frame and listed in `metadata.extra_columns`, but they are never used as
model features (features always come from `FEATURE_COLUMNS`).

## Metadata

`DatasetMetadata` holds aggregates only: file name (not full path), SHA-256 fingerprint, size,
row/column counts, column names, target counts and positive rate, per-column missing counts,
duplicate-row count, and warnings. It never contains individual records.

## Observed real-file facts (verified locally)

Measured by running `fraud-risk ingest` on the locally supplied `creditcard.csv`:

| Fact | Value |
|---|---|
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Size | 150,828,752 bytes |
| Rows / columns | 284,807 / 31 |
| Class counts | 0: 284,315 · 1: 492 |
| Positive rate | 0.1727 % |
| Missing cells | 0 |
| Exact duplicate rows | 1,081 (kept; see `docs/data-quality.md` from Phase 02) |
