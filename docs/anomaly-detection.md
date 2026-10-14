# Anomaly Detection Track

## Purpose

A **complementary, unsupervised signal**: "how unusual is this transaction compared with past
traffic?" It is not a replacement for the supervised classifier and is not combined with it in
this phase. Any future combination must be an explicit, documented design (see the Phase 07 risk
policy).

## Components

| Module | Role |
|---|---|
| `fraud_risk.anomaly.detectors` | `AnomalyDetector` protocol; `IsolationForestDetector` (preprocessor + `IsolationForest`) |
| `fraud_risk.anomaly.evaluation` | `evaluate_anomaly_ranking()`: AP, ROC-AUC, precision/recall at review budgets |
| `fraud_risk.training.anomaly` | Workflow: split, fit, score held-out partitions, persist bundle and report |

```bash
fraud-risk train-anomaly                  # fully unsupervised (fit on all training features)
fraud-risk train-anomaly --fit-on legitimate
```

Outputs (Git-ignored): `artifacts/models/isolation_forest_<mode>-<timestamp>/` (bundle + scores),
`reports/anomaly_metrics_<mode>.json`.

## Score semantics

`score(X)` returns `-IsolationForest.score_samples(X)`: **higher = more anomalous**. It is
**not** a probability, is not calibrated, and its scale is specific to the detector. It is used
only as a ranking. It must not be averaged with, compared to, or substituted for the supervised
fraud score. `contamination` is left at `"auto"` and the detector's own binary `predict` is never
used. Operating points are expressed as **review budgets** (top-k).

## Setup and label isolation

- Same seeded split as the supervised track.
- **`fit_on=all` (default, fully unsupervised):** fitted on all training-partition features. No
  labels are used; the training frauds are just unlabelled points.
- **`fit_on=legitimate` (semi-supervised novelty detection):** fitted only on training rows
  labelled legitimate. This **uses training labels** (reported as `uses_training_labels: true`).
- The detector rejects any frame containing `Class`, so labels cannot leak into fitting.
- Hyperparameters are fixed a priori (200 trees, 256 samples per tree, seed 42), not tuned on
  labels. Validation/test labels are used only for evaluation.

## Real-data results (verified locally)

Stratified grouped split, seed 42. Fit time ≈ 0.5 s.

| Mode | Partition | AP | ROC-AUC | P@100 | R@100 | P@1000 | R@1000 |
|---|---|---|---|---|---|---|---|
| all (unsupervised) | validation | 0.2629 | 0.9518 | 0.33 | 0.324 | 0.079 | 0.775 |
| all (unsupervised) | **test** | **0.1291** | 0.9417 | 0.27 | 0.281 | 0.065 | 0.677 |
| legitimate-only | validation | 0.2267 | 0.9527 | 0.30 | 0.294 | 0.084 | 0.824 |
| legitimate-only | test | 0.1095 | 0.9437 | 0.21 | 0.219 | 0.071 | 0.740 |

Reference points on the same test partition: prevalence 0.0017 (random ranking) and the selected
supervised model's AP 0.8163 (Phase 05).

## When it is useful, and its limits

- **Useful:** no labels are needed. It can surface novel or emerging patterns the supervised model
  never saw, and it gives a second, independent view for review prioritisation and for monitoring
  input drift.
- **Limits:** "unusual" is not "fraudulent". Many rare, legitimate transactions rank high, and
  frauds that resemble normal traffic rank low. On this dataset it ranks frauds about 76× better
  than random (AP 0.129 vs 0.0017), yet far below the supervised model. ROC-AUC (≈ 0.94) looks
  strong only because of the huge negative class. AP and top-k precision show the real workload.
  Validation-to-test AP differs substantially (0.26 vs 0.13) with ~100 frauds per partition.
- Fitting on legitimate rows only did not help here (lower AP), and it introduces label
  dependence.
