# Explainability

> **Association, not causation.** Explanations describe how a *model* maps inputs to its score.
> They do not prove why a transaction is fraudulent. `V1`–`V28` are anonymised PCA components
> with no business meaning, so "V14 drove this score" says something about the model, not about
> a real-world cause.

## Methods (`fraud_risk.explainability.explainers`)

| Scope | Method | Applies to | Output space | Additive? |
|---|---|---|---|---|
| Global | Permutation importance (drop in AP when a raw input feature is shuffled; full pipeline) | any `Pipeline(preprocess → classifier)` | AP drop | — |
| Local | `linear`: `coef_j × x_j` on transformed features, plus the intercept | `LogisticRegression` | log-odds | yes (exact) |
| Local | `shap_tree`: SHAP `TreeExplainer` on the preprocessed row | HGB, RF, ExtraTrees, GB, DecisionTree, XGBoost; needs optional `shap` (`pip install "fraud-risk[explain]"`) | log-odds (raw margin) | yes |
| Local | `occlusion` (fallback): `f(x) − f(x with feature j := background median)` | any model with `predict_proba` | probability | **no** |

`resolve_method(model, "auto")` picks `linear` for logistic regression, `shap_tree` for supported
trees when `shap` is installed, and `occlusion` otherwise. An explicit request that does not fit
the model raises `UnsupportedModelError`. Requesting SHAP without the package raises
`DependencyUnavailableError` with the install command. Attributions are never fabricated.

**Alignment with inference.** Every method runs the fitted pipeline's own `preprocess` step.
Linear/SHAP contributions are named by the preprocessor's transformed feature names, which match
the manifest's `transformed_feature_names`. Permutation importance and occlusion use the raw
`FEATURE_COLUMNS`.

**Typed outputs** (`GlobalExplanation`, `LocalExplanation`) carry `ModelInfo` (model id, name,
model SHA-256), the method, the output space, the additivity flag, and the causality caveat. They
contain contributions only, never raw feature values, and nothing is logged.

## Report (`fraud-risk explain --bundle artifacts/models/<id>`)

1. Re-derives the split from the manifest and checks the validation index fingerprint against
   the manifest. On a mismatch it refuses (`EXPLANATION_FAILED`). The CLI also checks the dataset
   SHA-256.
2. Global permutation importance on the **validation** partition (AP scoring, 3 repeats, seeded).
3. Local explanations for the 5 highest-scored validation rows, identified by `row_id`, with the
   top 8 contributions. Occlusion uses a seeded 2,000-row training background.
4. Writes `reports/explanations_<model_id>.json` (Git-ignored). The test partition is not used.

## Real-data results (verified locally)

**HistGradientBoosting** (`hist_gradient_boosting-20261014t050441`, local method `shap_tree`,
23 s). Permutation importance (AP drop, mean ± std over 3 repeats):

| Feature | AP drop |
|---|---|
| V14 | 0.1648 ± 0.0081 |
| Amount | 0.0069 ± 0.0037 |
| V10 | 0.0068 ± 0.0002 |
| V4 | 0.0063 ± 0.0067 |
| V12 | 0.0062 ± 0.0011 |
| V3 | 0.0054 ± 0.0009 |

Local SHAP (log-odds, base value −13.81) for the three top-scored validation rows, all labelled
fraud: the largest contributions came from V14 (+9.9 to +10.9), V12 (≈ +3.4), and V10 (≈ +2.9).

**Logistic regression** (`logistic_regression-20261014t045734`, local method `linear`): top AP
drops were V14 0.0819, V10 0.0420, V8 0.0350, V4 0.0179, V12 0.0121.

Both models lean most on V14. Under permutation, many features show small or noisy AP drops
(std comparable to the mean, e.g. V4 for HGB), so the ordering below the top feature is
unstable.

## Limitations

- Permutation importance underestimates features that are correlated with others (the model can
  substitute one for another). PCA components are uncorrelated by construction, but `Time` and
  `Amount` are not PCA outputs.
- SHAP values explain the boosted model's raw margin, not calibrated probabilities.
- Occlusion depends on the chosen reference (background median) and is not additive.
- About 100 validation frauds make importance estimates noisy.
