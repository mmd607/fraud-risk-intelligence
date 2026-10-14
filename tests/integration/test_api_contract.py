"""Test-phase T10: real service wiring, model selection, privacy of logs, and error contract."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_risk.api.app import create_app
from fraud_risk.config.settings import load_settings
from fraud_risk.core.logging import configure_logging
from fraud_risk.data.splitting import SplitConfig
from fraud_risk.registry.model_registry import LocalModelRegistry, ModelStage
from fraud_risk.services.inference import (
    ModelNotReadyError,
    ModelService,
    build_model_service,
    resolve_bundle_path,
)
from fraud_risk.training.artifacts import ModelArtifactError
from fraud_risk.training.baseline import BaselineConfig, run_baseline_training
from tests.support.fakes import TEST_POLICY, FakeInferenceService, valid_transaction
from tests.support.synthetic import make_transactions

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "configs" / "risk_policy.example.json"


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("api-model")
    result = run_baseline_training(
        make_transactions(1_500, 45, seed=23, signal=3.0),
        dataset_sha256=None,
        config=BaselineConfig(split=SplitConfig(seed=2), cv_folds=2),
        models_dir=root / "models",
        reports_dir=root / "reports",
    )
    return result.bundle_dir


def _client(**settings: object) -> TestClient:
    return TestClient(create_app(None, load_settings(**settings)), raise_server_exceptions=False)


def test_app_builds_real_service_from_explicit_bundle(bundle_dir: Path) -> None:
    with _client(model_bundle_path=bundle_dir, risk_policy_path=POLICY_PATH) as client:
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/v1/predict?explain=true", json={"transaction": valid_transaction(V1=3.0)}
        )
        model = client.get("/v1/model").json()
    assert response.status_code == 200
    result = response.json()["result"]
    assert 0.0 <= result["score"] <= 1.0
    assert result["explanation"]["available"] is True
    assert result["explanation"]["method"] == "linear"
    assert len(result["explanation"]["top_contributions"]) == 5
    assert model["model_id"] == bundle_dir.name
    assert len(model["feature_names"]) == 30


def test_app_without_any_model_is_not_ready(tmp_path: Path) -> None:
    with _client(artifacts_dir=tmp_path, risk_policy_path=POLICY_PATH) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json() == {"status": "not_ready", "model_loaded": False}


def test_approved_registry_model_is_served(bundle_dir: Path, tmp_path: Path) -> None:
    settings = load_settings(artifacts_dir=tmp_path, risk_policy_path=POLICY_PATH)
    registry = LocalModelRegistry(settings.registry_dir)
    registry.register(bundle_dir, actor="tester")
    assert resolve_bundle_path(settings) is None  # candidates are never served
    registry.transition(1, ModelStage.APPROVED, actor="tester", reason="api test")
    assert resolve_bundle_path(settings) == bundle_dir.resolve()
    assert build_model_service(settings).ready


def test_explicit_path_takes_precedence(bundle_dir: Path, tmp_path: Path) -> None:
    settings = load_settings(artifacts_dir=tmp_path, model_bundle_path=Path("explicit/bundle"))
    assert resolve_bundle_path(settings) == Path("explicit/bundle")


def test_tampered_bundle_fails_startup(bundle_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / bundle_dir.name
    copy.mkdir()
    for item in bundle_dir.iterdir():
        (copy / item.name).write_bytes(item.read_bytes())
    (copy / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ModelArtifactError):
        build_model_service(load_settings(model_bundle_path=copy, risk_policy_path=POLICY_PATH))


def test_model_service_not_ready_raises() -> None:
    service = ModelService(None, TEST_POLICY)
    assert not service.ready
    with pytest.raises(ModelNotReadyError):
        service.describe()


def test_logs_contain_no_payload_values() -> None:
    stream = io.StringIO()
    app = create_app(FakeInferenceService(), load_settings())
    with TestClient(app) as client:
        configure_logging("INFO", stream=stream)  # after lifespan configured logging
        client.post(
            "/v1/predict",
            json={"transaction": valid_transaction(V1=0.5, Amount=987.65), "reference": "ref-xyz"},
        )
    output = stream.getvalue()
    events = [json.loads(line)["event"] for line in output.strip().splitlines()]
    assert "request_completed" in events
    assert "987.65" not in output
    assert "ref-xyz" not in output


@pytest.mark.parametrize(("method", "path"), [("GET", "/v1/predict"), ("GET", "/does-not-exist")])
def test_routing_errors_use_error_envelope(method: str, path: str) -> None:
    with TestClient(create_app(FakeInferenceService(), load_settings())) as client:
        response = client.request(method, path)
    assert response.status_code in {404, 405}
    body: dict[str, Any] = response.json()
    assert set(body) == {"request_id", "error"}
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_wrong_api_key_rejected() -> None:
    app = create_app(FakeInferenceService(), load_settings(api_key="right-key"))
    with TestClient(app) as client:
        response = client.get("/v1/model", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_empty_batch_and_non_json_body_rejected() -> None:
    with TestClient(create_app(FakeInferenceService(), load_settings())) as client:
        assert client.post("/v1/predict/batch", json={"items": []}).status_code == 422
        response = client.post(
            "/v1/predict", content="not json", headers={"Content-Type": "text/plain"}
        )
    assert response.status_code == 422


def test_batch_references_round_trip() -> None:
    items = [
        {"transaction": valid_transaction(V1=0.4), "reference": "a-1"},
        {"transaction": valid_transaction(V1=0.0)},
    ]
    with TestClient(create_app(FakeInferenceService(), load_settings())) as client:
        results = client.post("/v1/predict/batch", json={"items": items}).json()["results"]
    assert [r["reference"] for r in results] == ["a-1", None]
    assert [r["explanation"] for r in results] == [None, None]


def test_integer_json_numbers_are_accepted() -> None:
    payload = {k: int(v) for k, v in valid_transaction().items()}
    with TestClient(create_app(FakeInferenceService(), load_settings())) as client:
        assert client.post("/v1/predict", json={"transaction": payload}).status_code == 200


def test_predictions_match_direct_model_scores(bundle_dir: Path) -> None:
    from fraud_risk.training.artifacts import load_model_bundle

    bundle = load_model_bundle(bundle_dir)
    tx = valid_transaction(V1=1.2, V2=-0.7, Amount=12.5)
    expected = float(bundle.model.predict_proba(pd.DataFrame([tx]))[0, 1])
    with _client(model_bundle_path=bundle_dir, risk_policy_path=POLICY_PATH) as client:
        score = client.post("/v1/predict", json={"transaction": tx}).json()["result"]["score"]
    assert score == pytest.approx(expected)
