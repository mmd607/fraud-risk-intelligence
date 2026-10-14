# Validation Strategy and Preprocessing

## Partitioning

`fraud_risk.data.splitting.split_dataset(frame, SplitConfig(...))` makes three disjoint partitions
**before** anything is fitted:

| Partition | Default share | Allowed uses |
|---|---|---|
| train | 60 % | Fitting preprocessing, resampling, models; cross-validation folds for hyperparameter tuning |
| validation | 20 % | Model selection and **decision-threshold selection** |
| test | 20 % | **Final evaluation only.** Never used for fitting, tuning, threshold choice, or feature selection |

### Strategies

- **`stratified` (default).** Random, label-stratified split with seed `FRAUD_RISK_RANDOM_SEED`
  (default 42). With `group_duplicates=True` (default), exact-duplicate rows (all features + label
  identical) form one group and are assigned as a unit. This prevents the 1,081 duplicated real
  rows from sitting in both train and test, which would inflate test scores.
- **`temporal`.** Sorted by `Time`; first 60 % train, next 20 % validation, last 20 % test. Closer to
  deployment (train on the past, score the future), but class ratios differ between partitions.

### Real-data split facts (verified locally, seed 42)

`fraud-risk split --strategy stratified` / `--strategy temporal`:

| Strategy | Train rows / frauds / rate | Validation | Test |
|---|---|---|---|
| stratified (grouped) | 170,894 / 294 / 0.172 % | 56,965 / 102 / 0.179 % | 56,948 / 96 / 0.169 % |
| temporal | 170,884 / 360 / 0.211 % | 56,961 / 57 / 0.100 % | 56,962 / 75 / 0.132 % |

The temporal split shows the fraud rate itself shifts within two days, a reminder that
stratified-split metrics are optimistic about deployment conditions. **The project's headline
evaluation uses the stratified grouped split**; temporal evaluation is a documented sensitivity
check for later phases.

Split metadata (strategy, seed, sizes, per-partition positives and rates, and a SHA-256 fingerprint
of each partition's row index) is written to `reports/split_metadata.json`. **No data is written.**

### Failure modes

`SplitError` (`DATA_SPLIT_INVALID`) is raised when the target is missing, the index is not unique,
there are fewer than 3 examples of a class, sizes make stratification impossible, or any partition
lacks one of the classes. Invalid fractions raise `ConfigurationError`.

## Features vs target

`split_features_target(frame)` returns `X` restricted to `FEATURE_COLUMNS` (Time, V1–V28, Amount)
and `y = Class`. The target can never enter `X`. The preprocessor also uses `remainder="drop"`, so
an accidentally present `Class` column would be dropped, never passed through.

## Preprocessing (`fraud_risk.features.preprocessing`)

`build_preprocessor(PreprocessingConfig(...))` returns an **unfitted** `ColumnTransformer` with
pandas output and stable feature names. It is meant to be the first step of a model `Pipeline`,
so cross-validation refits it inside every training fold.

| Column(s) | Default | Alternatives | Rationale |
|---|---|---|---|
| `V1`–`V28` | standardise | passthrough | Already PCA outputs; scaling matters for regularised linear models, is harmless for trees |
| `Amount` | `log1p` + standardise | robust (median/IQR), standard, passthrough | Non-negative, heavily right-skewed (real file: median 22.0, p99 ≈ 1,018, max 25,691.16) |
| `Time` | standardise | `hour_cyclical` (sin/cos of `Time mod 86,400`), drop | Seconds since the first transaction, not a calendar time; `hour_cyclical` *assumes* the window starts at midnight, which the source does not document |

The defaults are a documented baseline, not a claim of optimality. The baseline phase (04)
compares alternatives with cross-validation on **training folds only**.

## Resampling

No SMOTE or other resampling is applied in this phase. If a later phase adds resampling, it must be
a pipeline step after preprocessing so it only ever sees training folds.

## Leakage checklist

- [x] Split before fitting anything.
- [x] Preprocessing fitted on training data/folds only (it is part of the model pipeline).
- [x] Test partition reserved for final evaluation.
- [x] Thresholds chosen on validation data (Phase 04).
- [x] `Class` excluded from features by construction.
- [x] Exact duplicates cannot straddle partitions (stratified grouped split).
