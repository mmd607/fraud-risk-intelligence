# Risk Scoring and Decision Policy

> **Illustrative only.** The actions `approve`, `review`, and `block` are labels. The engine
> executes nothing, contacts no payment system, and moves no money. Thresholds and costs in this
> repository are demo configuration, not banking policy.

## Separation of concerns

| Concept | Source | Meaning |
|---|---|---|
| **Score** | Supervised model (`predict_proba[:, 1]`) | Ranking score in `[0, 1]`; not calibrated |
| **Anomaly score** | Phase 06 detector | "How unusual"; separate scale; never mixed into the score |
| **Band** | Policy thresholds | `low` / `medium` / `high` / `unknown` |
| **Action** | Policy | `approve` / `review` / `block` (labels only) |
| **Reason codes** | Engine | Why the action was chosen (stable `R00x` codes) |

The policy (`fraud_risk.policy`) and engine (`fraud_risk.risk`) never import training code. They
operate on scores only.

## Policy configuration (`fraud_risk.policy.config.RiskPolicy`)

JSON, validated by pydantic (`extra="forbid"`), loaded with `load_policy(path)`. See
`configs/risk_policy.example.json`.

| Field | Constraint | Meaning |
|---|---|---|
| `version` | 1–64 chars | Recorded on every decision |
| `review_threshold` | `[0, 1]`, `< block_threshold` | Lower edge of `medium` / `review` |
| `block_threshold` | `[0, 1]` | Lower edge of `high` / `block` |
| `missing_score_action` | `review` (default) or `block` | Fail-safe for missing/invalid scores |
| `anomaly_escalation` | `{enabled: false, score_threshold > 0}` | Optional approve→review escalation |
| `review_capacity` | int ≥ 1 or null | Analyst capacity used for queue sizing |

Invalid policies raise `ConfigurationError` naming the offending field.

## Decision precedence

1. Missing, non-finite, or out-of-range score → `missing_score_action`, band `unknown`, `R004`.
2. `score >= block_threshold` → `block`, band `high`, `R003`.
3. `score >= review_threshold` → `review`, band `medium`, `R002`.
4. Otherwise `approve`, band `low`, `R001`. With escalation enabled and
   `anomaly_score >= escalation.score_threshold`, the action becomes `review` and `R005` is added.
   Escalation never blocks and never lowers an action.

Thresholds are inclusive (`>=`). Each `RiskDecision` records the score, anomaly score, band,
action, reason codes, policy version, both thresholds, model version, and an optional reference.

## Reason codes

| Code | Name | Description |
|---|---|---|
| R001 | SCORE_BELOW_REVIEW_THRESHOLD | Model score is below the review threshold |
| R002 | SCORE_AT_OR_ABOVE_REVIEW_THRESHOLD | Score in `[review, block)` |
| R003 | SCORE_AT_OR_ABOVE_BLOCK_THRESHOLD | Score `>= block` |
| R004 | SCORE_MISSING_OR_INVALID | Missing/non-finite/out-of-range score; fail-safe applied |
| R005 | ANOMALY_SCORE_ESCALATION | Anomaly score triggered approve→review escalation |

## Review queue

`build_review_queue(references, scores, capacity)` ranks by descending score (ties keep input
order), drops unscorable items (they go through the fail-safe instead), and cuts at capacity.
`capacity_tradeoff(y, scores, capacities)` reports precision, recall, frauds found, and the score
cut-off for each capacity.

## Expected-cost analysis (illustrative)

`evaluate_policy(y, scores, policy, CostAssumptions(...))` counts actions per true label and
computes:

`cost = missed_frauds × missed_fraud_cost + blocked_legit × false_block_cost + reviewed × review_cost`

where reviewed frauds are caught with probability `review_detection_rate` (default 1.0).
**Defaults (100 / 10 / 2 units) are made-up round numbers** for relative comparison of policies.
They are not real losses.

## Logging

Each decision logs a `risk_decision` event with action, band, reason codes, policy version, and
model version. Scores, features, and payloads are never logged.

## Example run (verified locally)

```bash
fraud-risk evaluate-policy --policy configs/risk_policy.example.json \
  --bundle artifacts/models/hist_gradient_boosting-20261014t050441 --partition validation
```

The example thresholds (review 0.01, block 0.95) were read off the **validation** scores of the
Phase 05 model, where scores are strongly polarised: 131 validation rows score ≥ 0.01, and 91 of
the 102 frauds are among them. The test partition was evaluated once afterwards.

| Partition | Approved (legit / fraud) | Reviewed (legit / fraud) | Blocked (legit / fraud) | Fraud capture | Block precision | Illustrative cost | Approve-all cost |
|---|---|---|---|---|---|---|---|
| validation | 56,823 / 11 | 36 / 3 | 4 / 88 | 0.892 | 0.957 | 1,218 | 10,200 |
| test | 56,807 / 17 | 43 / 8 | 2 / 71 | 0.823 | 0.973 | 1,822 | 9,600 |

Capacity trade-off on test: top-100 precision 0.78 / recall 0.81; top-150 precision 0.53 /
recall 0.82.
