# Advanced Model Comparison

## Command

```bash
pip install -e ".[boost,imbalance]"     # optional: XGBoost and SMOTE candidates
fraud-risk compare-models                 # all available candidates, 3-fold CV
fraud-risk compare-models --candidates logistic_regression hist_gradient_boosting --cv-folds 5
```

Outputs (Git-ignored): `reports/model_comparison.json`, `reports/model_comparison.md`, and a bundle
for the selected model in `artifacts/models/<model_id>/`, with the same manifest, integrity hash,
and score files as the baseline.

## Candidates and justification

| Candidate | Why | Search space | Dependency |
|---|---|---|---|
| `logistic_regression` | Baseline family; interpretable, fast | `C ∈ {0.01, 0.1, 1}`, class-weighted | core |
| `random_forest` | Non-linear interactions, scale-free, strong default | 200 trees, `balanced_subsample`, `min_samples_leaf ∈ {1, 5}` | core |
| `hist_gradient_boosting` | Strong tabular learner with no extra dependency | 300 iters, `class_weight ∈ {None, balanced}`, `learning_rate ∈ {0.05, 0.1}` | core |
| `xgboost` | Widely used boosting reference | 400 trees, depth 5, `scale_pos_weight ∈ {1, neg/pos}` | optional `[boost]` |
| `logistic_regression_smote` | Resampling vs class weighting | SMOTE (ratio 0.1) inside the pipeline, `C ∈ {0.1, 1}` | optional `[imbalance]` |

**Optional dependencies.** A candidate whose package cannot be imported is recorded as
`skipped` with the reason, and the comparison continues with the rest. There is no fake fallback.

## Protocol (leakage-safe)

1. Same seeded, stratified, duplicate-grouped split as the baseline (60/20/20, seed 42).
2. `GridSearchCV`, stratified 3-fold, AP scoring, on the **training partition only**.
   Preprocessing and SMOTE are pipeline steps, so they are refitted inside each training fold.
   SMOTE never sees validation or test rows.
3. Each tuned candidate is refitted on the full training partition. On **validation** it gets AP,
   a max-F1 threshold, precision at recall 0.5/0.7/0.8/0.9, and runtime/size.
4. **Selection objective:** highest mean CV AP on training folds. This is preferred to validation
   AP because the training folds contain about 3× more frauds (294 vs 102), so it is less noisy.
5. Only the **selected** model is scored on the **test** partition, once. Other candidates never
   see test data, which removes the temptation to pick by test score.

Determinism: all estimators use `random_state = seed`. Results are reproducible on the same
machine and library versions; multithreaded tree builders can differ in the last decimals across
platforms or versions.

## Real-data results (verified locally, run `hist_gradient_boosting-20261014t050441`)

Wall time 1 min 57 s on a 20-thread Windows machine.

| Candidate | CV AP (mean ± std) | Val AP | Val precision @ recall 0.8 | Val F1 @ thr | Search s | Refit s | Size KB |
|---|---|---|---|---|---|---|---|
| logistic_regression (C=0.1) | 0.7362 ± 0.0154 | 0.8095 | 0.884 | 0.860 | 3.8 | 0.3 | 4 |
| random_forest (leaf=1) | 0.8253 ± 0.0351 | 0.8861 | 0.956 | 0.902 | 36.5 | 11.8 | 5,129 |
| **hist_gradient_boosting** (balanced, lr 0.1) — selected | **0.8430 ± 0.0375** | 0.8946 | 0.988 | 0.913 | 26.6 | 12.9 | 1,109 |
| xgboost (scale_pos_weight = 580.3) | 0.8407 ± 0.0306 | 0.8955 | 0.977 | 0.900 | 9.4 | 2.9 | 779 |
| logistic_regression_smote (C=0.1) | 0.7474 ± 0.0212 | 0.8235 | 0.832 | 0.828 | 1.7 | 0.4 | 74 |

### Selected model on the held-out test partition (56,948 rows, 96 frauds)

Threshold 0.9037 (max-F1 on validation):

| AP | ROC-AUC | Precision | Recall | F1 | Confusion `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|---|
| **0.8163** | 0.9696 | 0.9730 | 0.7500 | 0.8471 | `[[56850, 2], [24, 72]]` |

Review capacity (test): top-50 precision 0.98 / recall 0.51; top-100 precision 0.78 / recall 0.81;
top-500 precision 0.164 / recall 0.854.

Phase 04's logistic-regression baseline scored test AP 0.6833 on the same partition.

### Interpretation

- Tree ensembles beat linear models clearly (CV AP ≈ 0.83–0.84 vs ≈ 0.74).
- HistGradientBoosting and XGBoost are **statistically indistinguishable** (difference 0.002 vs
  fold std ≈ 0.03). HGB won the stated objective; XGBoost is about 3× faster to tune and smaller.
  Either is a defensible choice.
- SMOTE improved logistic regression's CV AP only slightly (0.747 vs 0.736) and lowered precision
  at high recall. Class weighting is simpler, with no synthetic samples, so it remains the default
  approach.
- With 96 test frauds, one fraud more or less moves recall by about 1 point. Single-split
  estimates carry wide uncertainty.
- Scores are not calibrated probabilities (class weighting shifts them). They are used as
  rankings and with thresholds.
