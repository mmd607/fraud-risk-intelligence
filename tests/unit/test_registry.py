from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier

from fraud_risk.core.exceptions import DependencyUnavailableError
from fraud_risk.registry import tracking as tracking_module
from fraud_risk.registry.model_registry import (
    LocalModelRegistry,
    ModelStage,
    RegistryError,
    track_entry,
)
from fraud_risk.registry.tracking import (
    InMemoryTracker,
    MlflowTracker,
    NullTracker,
    sanitize_metrics,
    sanitize_params,
)
from fraud_risk.training.artifacts import ModelArtifactError, ModelManifest, save_model_bundle


def _bundle(root: Path, model_id: str) -> Path:
    manifest = ModelManifest(
        model_id=model_id,
        model_name="dummy",
        created_at="2026-01-01T00:00:00+00:00",
        seed=1,
        target="Class",
        feature_names=["Time", "Amount"],
        transformed_feature_names=["Time", "Amount"],
        preprocessing={},
        model_params={},
        threshold={"threshold": 0.5},
        metrics={"test": {"average_precision": 0.5, "roc_auc": 0.9}, "validation": {}},
        dataset_sha256="abc",
        split={},
        library_versions={},
    )
    return save_model_bundle(DummyClassifier(), manifest, root)


@pytest.fixture
def registry(tmp_path: Path) -> LocalModelRegistry:
    return LocalModelRegistry(tmp_path / "registry")


def test_register_creates_candidate_with_metadata(
    registry: LocalModelRegistry, tmp_path: Path
) -> None:
    entry = registry.register(_bundle(tmp_path, "model-a"), actor="alice", code_revision="abc123")
    assert entry.version == 1
    assert entry.stage is ModelStage.CANDIDATE
    assert entry.code_revision == "abc123"
    assert entry.metrics["test"]["average_precision"] == 0.5
    assert entry.stage_history[0].actor == "alice"
    assert registry.approved() is None


def test_versions_increment_and_duplicates_rejected(
    registry: LocalModelRegistry, tmp_path: Path
) -> None:
    registry.register(_bundle(tmp_path, "model-a"), actor="a")
    assert registry.register(_bundle(tmp_path, "model-b"), actor="a").version == 2
    with pytest.raises(RegistryError, match="already registered"):
        registry.register(tmp_path / "model-a", actor="a")


def test_tampered_bundle_cannot_be_registered(registry: LocalModelRegistry, tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "model-a")
    (bundle / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ModelArtifactError):
        registry.register(bundle, actor="a")


def test_lifecycle_requires_explicit_transitions(
    registry: LocalModelRegistry, tmp_path: Path
) -> None:
    registry.register(_bundle(tmp_path, "model-a"), actor="a")
    registry.register(_bundle(tmp_path, "model-b"), actor="a")
    approved = registry.transition(1, ModelStage.APPROVED, actor="bob", reason="best AP")
    assert approved.stage is ModelStage.APPROVED
    with pytest.raises(RegistryError, match="already approved"):
        registry.transition(2, ModelStage.APPROVED, actor="bob", reason="x")
    registry.transition(1, ModelStage.RETIRED, actor="bob", reason="replace")
    registry.transition(2, ModelStage.APPROVED, actor="bob", reason="replacement")
    assert registry.approved().version == 2  # type: ignore[union-attr]
    history = registry.get(1).stage_history
    assert [h.to_stage for h in history] == [
        ModelStage.CANDIDATE,
        ModelStage.APPROVED,
        ModelStage.RETIRED,
    ]


def test_illegal_transitions(registry: LocalModelRegistry, tmp_path: Path) -> None:
    registry.register(_bundle(tmp_path, "model-a"), actor="a")
    registry.transition(1, ModelStage.RETIRED, actor="a", reason="old")
    with pytest.raises(RegistryError, match="not allowed"):
        registry.transition(1, ModelStage.APPROVED, actor="a", reason="revive")
    with pytest.raises(RegistryError, match="actor"):
        registry.transition(1, ModelStage.RETIRED, actor=" ", reason="x")
    with pytest.raises(RegistryError, match="not registered"):
        registry.transition(9, ModelStage.RETIRED, actor="a", reason="x")


def test_corrupt_registry_file_is_not_overwritten(
    registry: LocalModelRegistry, tmp_path: Path
) -> None:
    registry.root.mkdir(parents=True)
    registry.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RegistryError, match="invalid"):
        registry.register(_bundle(tmp_path, "model-a"), actor="a")
    assert registry.path.read_text(encoding="utf-8") == "{not json"


def test_registry_file_is_valid_json(registry: LocalModelRegistry, tmp_path: Path) -> None:
    registry.register(_bundle(tmp_path, "model-a"), actor="a")
    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    assert payload["entries"][0]["model_id"] == "model-a"


def test_sanitizers_drop_sensitive_and_non_scalar_values() -> None:
    params = sanitize_params({"C": 0.1, "api_key": "x", "grid": [1, 2], "note": None})
    assert params == {"C": 0.1, "note": "None"}
    metrics = sanitize_metrics({"ap": 0.8, "bad": float("nan"), "flag": True, "token_count": 3})
    assert metrics == {"ap": 0.8}


def test_null_tracker_records_nothing() -> None:
    tracker = NullTracker()
    with tracker.start_run("x") as run:
        run.log_params({"a": 1})
    assert tracker.enabled is False


def test_track_entry_with_in_memory_tracker(registry: LocalModelRegistry, tmp_path: Path) -> None:
    entry = registry.register(_bundle(tmp_path, "model-a"), actor="a")
    tracker = InMemoryTracker()
    run_id = track_entry(tracker, entry)
    run = tracker.runs[0]
    assert run_id == run.run_id
    assert run.metrics["test_average_precision"] == 0.5
    assert run.tags["stage"] == "candidate"
    assert run.params["feature_count"] == 2


def test_mlflow_tracker_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking_module.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(DependencyUnavailableError, match=r"fraud-risk\[tracking\]"):
        MlflowTracker("sqlite:///x.db")


@pytest.mark.skipif(
    tracking_module.importlib.util.find_spec("mlflow") is None, reason="mlflow not installed"
)
def test_mlflow_tracker_logs_to_local_sqlite(registry: LocalModelRegistry, tmp_path: Path) -> None:
    import mlflow

    entry = registry.register(_bundle(tmp_path, "model-a"), actor="a")
    uri = f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').as_posix()}"
    run_id = track_entry(MlflowTracker(uri, experiment="unit-test"), entry)
    run = mlflow.MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.data.metrics["test_average_precision"] == 0.5
    assert run.data.tags["model_id"] == "model-a"
    assert "api_key" not in run.data.params
