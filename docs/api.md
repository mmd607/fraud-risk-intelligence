# Inference API

> Educational scoring service. `approve` / `review` / `block` are illustrative labels: nothing is
> executed, no payment system is contacted, and there are no external side effects.

## Run locally

```bash
pip install -e ".[api,explain]"
# Serve an explicit bundle...
FRAUD_RISK_MODEL_BUNDLE_PATH=artifacts/models/<model_id> \
  uvicorn fraud_risk.api.app:create_app --factory --host 127.0.0.1 --port 8000
# ...or approve a registry version and omit the path (see docs/model-lifecycle.md).
```

OpenAPI UI: `http://127.0.0.1:8000/docs`; schema: `/openapi.json`.

**Model selection** (explicit, never automatic): `FRAUD_RISK_MODEL_BUNDLE_PATH`, otherwise the
registry's single **approved** version, otherwise no model (`/ready` → 503). The bundle's
SHA-256 is verified before loading. The risk policy comes from `FRAUD_RISK_RISK_POLICY_PATH`
(default `configs/risk_policy.example.json`).

## Endpoints

| Method | Path | Auth* | Purpose |
|---|---|---|---|
| GET | `/health` | no | **Liveness**: the process is up (does not check the model) |
| GET | `/ready` | no | **Readiness**: `200 {"status":"ready"}` when a model is loaded, else `503` |
| GET | `/v1/model` | yes | Model id, name, SHA-256, policy version, feature names (no paths) |
| POST | `/v1/predict[?explain=true]` | yes | Score one transaction (optional local explanation) |
| POST | `/v1/predict/batch` | yes | Score up to `FRAUD_RISK_API_MAX_BATCH_SIZE` (default 500) transactions |

\* When `FRAUD_RISK_API_KEY` is set, `/v1/*` requires the `X-API-Key` header (constant-time
comparison). When it is unset (the local default), `/v1/*` is open. Do not expose an unkeyed
service beyond localhost.

### Request (`POST /v1/predict`)

```json
{"transaction": {"Time": 406.0, "V1": -2.31, "...": "V2-V28", "Amount": 0.0}, "reference": "tx-001"}
```

`TransactionFeatures` is strict: all 30 fields are required, unknown fields are rejected, strings
are not coerced to numbers, values must be finite, and `Time`/`Amount` must be ≥ 0.
`reference` is optional (`^[A-Za-z0-9._:-]{1,64}$`) and echoed back, never logged.

### Response

```json
{
  "request_id": "2c68c47b…",
  "model_version": "hist_gradient_boosting-20261014t050441",
  "policy_version": "demo-2026-10-14",
  "result": {
    "reference": "tx-001",
    "score": 0.9998,
    "risk_band": "high",
    "action": "block",
    "reason_codes": ["R003"],
    "explanation": {"available": true, "method": "shap_tree", "output_space": "log_odds",
                    "top_contributions": [{"feature": "V14", "contribution": 9.9}], "caveat": "..."}
  }
}
```

`score` is a ranking score, not a calibrated probability. `explanation` is `null` unless
`?explain=true`. Batch mode returns no explanations.

## Error contract

All errors use one envelope, and the `X-Request-ID` header matches `request_id`:

```json
{"request_id": "…", "error": {"code": "REQUEST_INVALID", "message": "Request validation failed.",
 "details": [{"field": "body.transaction.Time", "message": "Input should be greater than or equal to 0"}]}}
```

| Status | Code | When |
|---|---|---|
| 401 | `UNAUTHORIZED` | API key configured and missing/wrong |
| 413 | `BATCH_TOO_LARGE` | Batch above the configured limit |
| 422 | `REQUEST_INVALID` | Schema/validation failure (field + message only; inputs never echoed) |
| 503 | `MODEL_NOT_READY` | No model loaded |
| 500 | `INTERNAL_ERROR` or a domain code | Unexpected failure; generic message, no stack trace |

## Logging and request ids

A valid incoming `X-Request-ID` (`^[A-Za-z0-9._-]{1,64}$`) is reused; otherwise a UUID is
generated. Each request logs `request_completed` with request id, method, path, status, and
duration. Each decision logs `risk_decision` with action, band, reason codes, and policy/model
versions. **Payloads, feature values, scores, and references are never logged.** This was checked
on the smoke-test log below.

## Verified smoke test (local, real model)

uvicorn on `127.0.0.1:8765` with `FRAUD_RISK_MODEL_BUNDLE_PATH` set to the Phase 05 HGB bundle:
`/ready` → 200; `/v1/model` → model id and SHA-256 `f58b75d3…`. The highest-scored **test** fraud
→ score 1.0, `block`, `R003`, SHAP top features V14, V12, V10. The lowest-scored test row (legit)
→ `approve`, `R001`. A malformed request → 422 with field-level messages. The log contained only
metadata events.

## Testing

`tests/integration/test_api.py` uses FastAPI's `TestClient` with an injected
`FakeInferenceService` (`tests/support/fakes.py`). No trained model is needed. (Starlette emits a
third-party deprecation warning about `httpx` in its test client; it does not affect behaviour.)

## Limitations

- No authentication beyond an optional static API key, no TLS termination, and no rate limiting.
  This is a local demo, not an internet-facing service.
- Single-process, in-memory model; changing the model requires a restart.
