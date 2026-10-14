# Baseline Modelling

## Command

```bash
fraud-risk train-baseline                       # defaults: 5-fold CV, max-F1 threshold
fraud-risk train-baseline --threshold-strategy min_precision --target-precision 0.9
```

Outputs (all Git-ignored):

- `artifacts/models/<model_id>/model.joblib` + `manifest.json` — logistic-regression bundle with
  configuration, feature names, seed, library versions, dataset SHA-256, split metadata, threshold,
  validation/test metrics, and the model file's SHA-256 (verified before loading).
- `artifacts/models/<model_id>/scores_{validation,test}.csv` — `row_id, label, score` only.
- `reports/baseline_metrics.json` — the full comparison report.

## Protocol

1. **Split once** (`stratified`, duplicate-grouped, seed 42): train 60 % / validation 20 % / test 20 %.
2. **Preprocessing comparison** — logistic regression, stratified 5-fold CV on the **training
   partition only**, scored by AP. Candidates: default (`log1p`+standardise Amount, standardise
   Time and V1–V28), standard-scaled Amount, Time dropped, and Time as cyclical hour. Selection
   uses a **one-standard-error rule**: keep the documented default unless a challenger beats its
   mean CV AP by more than the challenger's standard error. This avoids choosing on CV noise.
3. **Fit** `DummyClassifier(strategy="prior")` and `LogisticRegression(class_weight="balanced",
   C=1.0, lbfgs, max_iter=2000)`, each inside `Pipeline(preprocess → classifier)`, on the training
   partition.
4. **Threshold** chosen per model on the **validation partition** (`max_f1` by default: the
   threshold maximising F1; ties resolved towards fewer alerts).
5. **Final evaluation** on the **test partition**, once, with thresholds frozen.

The test partition is never used for preprocessing choice, hyperparameters, or thresholds.

## Metric definitions

- **Average Precision (primary)** — `sklearn.metrics.average_precision_score`:
  `AP = Σₙ (Rₙ − Rₙ₋₁) Pₙ` over descending score thresholds, no interpolation. A constant scorer
  scores the prevalence.
- **ROC-AUC (secondary)** — reported but not used for selection; it is dominated by the very large
  negative class.
- **Precision / recall / F1 / confusion matrix** — at the stated threshold, positive when
  `score >= threshold`; confusion matrix label order `[0, 1]` → `[[TN, FP], [FN, TP]]`.
- **Precision/recall@k** — among the `k` highest scores (review-capacity view).
- **Accuracy is not reported** (always-legitimate already scores ≈ 99.83 %).

## Real-data results (verified locally)

Dataset SHA-256 `76274b69…51a89`; stratified grouped split, seed 42; test partition 56,948 rows
with 96 frauds (prevalence 0.169 %). Run id `logistic_regression-20261014t045734`.

### Preprocessing comparison (training-fold CV, logistic regression, 5 folds)

| Option | CV AP mean | CV AP std |
|---|---|---|
| **default: Amount log1p+standardise, Time standardise** (selected) | 0.7509 | 0.0579 |
| Amount standardise | 0.7513 | 0.0637 |
| Time dropped | 0.7507 | 0.0571 |
| Time as cyclical hour | 0.7505 | 0.0576 |

All options are within CV noise of one another (spread 0.0008, fold std ≈ 0.06), so the default
is kept.

### Test-set results (thresholds chosen on validation)

| Model | Threshold | AP | ROC-AUC | Precision | Recall | F1 | Confusion `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|---|---|---|
| Dummy (prior) | 0.00172 | 0.0017 | 0.500 | 0.0017 | 1.000 | 0.0034 | `[[0, 56852], [0, 96]]` |
| Logistic regression | 0.99999466 | **0.6833** | 0.9621 | 0.8706 | 0.7708 | 0.8177 | `[[56841, 11], [22, 74]]` |

Logistic regression on the validation partition: AP 0.8099, F1 0.86 at the chosen threshold.

Precision/recall at a fixed review capacity (test, logistic regression):

| k | Precision@k | Recall@k |
|---|---|---|
| 50 | 0.840 | 0.438 |
| 100 | 0.750 | 0.781 |
| 500 | 0.164 | 0.854 |

### Interpretation and caveats

- The learned baseline beats the dummy by a wide margin (AP 0.683 vs 0.0017).
- **Small positive counts → high variance.** With only 96 test and 102 validation frauds, AP moves
  by several points between partitions (validation 0.81 vs test 0.68). Treat single-split numbers
  as estimates, not precise performance.
- **Uncalibrated scores.** `class_weight="balanced"` pushes probabilities towards 0/1; the chosen
  threshold (≈ 0.999995) is therefore extreme. Scores are rankings, not calibrated fraud
  probabilities.
- **Selection-rule disclosure.** The very first run selected preprocessing by the highest mean CV
  AP (it picked standardised Amount, test AP 0.6937). The one-standard-error rule was then adopted
  because of the CV noise visible in the *training-fold* results; it reverted to the default and
  gave the lower test AP reported above. The test set did not motivate the change, but this is
  disclosed because the test result of the first run had been seen.
- These figures come from a public, anonymised 2013 dataset spanning two days; they do not
  represent performance on any real payment system.
