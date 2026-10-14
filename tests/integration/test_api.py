from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_risk.api.app import create_app
from fraud_risk.config.settings import load_settings
from tests.support.fakes import FakeInferenceService, valid_transaction

pytestmark = pytest.mark.integration


def _client(service: FakeInferenceService | None = None, **settings: object) -> TestClient:
    return TestClient(
        create_app(service or FakeInferenceService(), load_settings(**settings)),
        raise_server_exceptions=False,
    )


def test_health_and_ready() -> None:
    with _client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["model_loaded"] is True


def test_not_ready_service() -> None:
    with _client(FakeInferenceService(ready=False)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/v1/predict", json={"transaction": valid_transaction()})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_NOT_READY"


@pytest.mark.parametrize(
    ("v1", "action", "band", "code"),
    [
        (0.1, "approve", "low", "R001"),
        (0.5, "review", "medium", "R002"),
        (0.9, "block", "high", "R003"),
    ],
)
def test_predict_returns_decision(v1: float, action: str, band: str, code: str) -> None:
    with _client() as client:
        response = client.post(
            "/v1/predict", json={"transaction": valid_transaction(V1=v1), "reference": "tx-1"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "fake-model-1"
    assert body["policy_version"] == "test-policy"
    assert body["result"] == {
        "reference": "tx-1",
        "score": v1,
        "risk_band": band,
        "action": action,
        "reason_codes": [code],
        "explanation": None,
    }
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_request_id_is_propagated_or_generated() -> None:
    with _client() as client:
        echoed = client.get("/health", headers={"X-Request-ID": "abc-123"})
        generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert echoed.headers["X-Request-ID"] == "abc-123"
    assert generated.headers["X-Request-ID"] != "bad id with spaces"


@pytest.mark.parametrize(
    "payload",
    [
        {"transaction": valid_transaction(Amount=-1.0)},
        {"transaction": {**valid_transaction(), "extra_field": 1.0}},
        {"transaction": {k: v for k, v in valid_transaction().items() if k != "V7"}},
        {"transaction": {**valid_transaction(), "V3": "1.5"}},
        {"transaction": valid_transaction(), "reference": "has spaces!"},
        {"unexpected": True},
    ],
)
def test_invalid_input_is_controlled_422(payload: dict[str, object]) -> None:
    with _client() as client:
        response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "REQUEST_INVALID"
    assert "has spaces!" not in response.text  # offending values are never echoed
    for detail in body["error"]["details"]:
        assert set(detail) == {"field", "message"}


def test_non_finite_values_rejected() -> None:
    with _client() as client:
        response = client.post(
            "/v1/predict",
            content='{"transaction": '
            + str(valid_transaction()).replace("'", '"').replace('"V2": 0.0', '"V2": NaN')
            + "}",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422


def test_unexpected_failure_hides_details() -> None:
    with _client(FakeInferenceService(fail=True)) as client:
        response = client.post("/v1/predict", json={"transaction": valid_transaction()})
    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "INTERNAL_ERROR",
        "message": "Internal server error.",
        "details": None,
    }
    assert "4111" not in response.text
    assert "Traceback" not in response.text


def test_batch_prediction_and_limit() -> None:
    items = [{"transaction": valid_transaction(V1=v)} for v in (0.1, 0.95)]
    with _client() as client:
        response = client.post("/v1/predict/batch", json={"items": items})
    assert response.status_code == 200
    assert [r["action"] for r in response.json()["results"]] == ["approve", "block"]
    with _client(api_max_batch_size=1) as client:
        too_big = client.post("/v1/predict/batch", json={"items": items})
    assert too_big.status_code == 413
    assert too_big.json()["error"]["code"] == "BATCH_TOO_LARGE"


def test_api_key_enforced_when_configured() -> None:
    with _client(api_key="s3cret-key-value") as client:
        assert client.get("/health").status_code == 200
        denied = client.post("/v1/predict", json={"transaction": valid_transaction()})
        assert denied.status_code == 401
        assert "s3cret" not in denied.text
        allowed = client.post(
            "/v1/predict",
            json={"transaction": valid_transaction()},
            headers={"X-API-Key": "s3cret-key-value"},
        )
        assert allowed.status_code == 200


def test_model_endpoint_has_no_paths() -> None:
    with _client() as client:
        body = client.get("/v1/model").json()
    assert body["model_id"] == "fake-model-1"
    assert "bundle" not in str(body).lower()
    assert "artifacts" not in str(body)


def test_openapi_schema_documents_contract() -> None:
    with _client() as client:
        schema = client.get("/openapi.json").json()
    assert "/v1/predict" in schema["paths"]
    features = schema["components"]["schemas"]["TransactionFeatures"]
    assert features["additionalProperties"] is False
    assert {"Time", "Amount", "V1", "V28"} <= set(features["required"])
