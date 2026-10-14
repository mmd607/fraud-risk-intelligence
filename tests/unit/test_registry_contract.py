"""Test-phase T09: metadata validation, lifecycle safety, offline tracking, and registry CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier

from fraud_risk.cli.main import build_tracker, main
from fraud_risk.config.settings import load_settings
from fraud_risk.registry import model_registry as registry_module
from fraud_risk.registry.model_registry import (
    ALLOWED_TRANSITIONS,
    LocalModelRegistry,
    ModelStage,
    RegistryEntry,
    RegistryError,
    current_code_revision,
)
from fraud_risk.registry.tracking import NullTracker, default_tracking_uri
from fraud_risk.training.artifacts import ModelManifest, save_model_bundle


def _bundle(root: Path, model_id: str = "model-x") -> Path:
    manifest = ModelManifest(
        model_id=model_id,
        model_name="dummy",
        created_at="2026-02-02T00:00:00+00:00",
        seed=3,
        target="Class",
        feature_names=["Time"],
        transformed_feature_names=["Time"],
        preprocessing={},
        model_params={},
        threshold={},
        metrics={"validation": {"average_precision": 0.4}},
        dataset_sha256=None,
        split={},
        library_versions={},
    )
    return save_model_bundle(DummyClassifier(), manifest, root)


def test_transition_table_has_no_path_back_to_candidate() -> None:
    for targets in ALLOWED_TRANSITIONS.values():
        assert ModelStage.CANDIDATE not in targets
    assert ALLOWED_TRANSITIONS[ModelStage.RETIRED] == frozenset()


def test_registration_never_promotes(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "r")
    for i in range(3):
        registry.register(_bundle(tmp_path, f"model-{i}"), actor="a")
    assert {e.stage for e in registry.entries()} == {ModelStage.CANDIDATE}
    assert registry.approved() is None


def test_candidate_can_be_retired_directly(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "r")
    registry.register(_bundle(tmp_path), actor="a")
    assert registry.transition(1, ModelStage.RETIRED, actor="a", reason="x").stage is (
        ModelStage.RETIRED
    )


def test_approved_cannot_be_reapproved(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "r")
    registry.register(_bundle(tmp_path), actor="a")
    registry.transition(1, ModelStage.APPROVED, actor="a", reason="x")
    with pytest.raises(RegistryError, match="not allowed"):
        registry.transition(1, ModelStage.APPROVED, actor="a", reason="again")


def test_registration_requires_actor(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "r")
    with pytest.raises(Exception, match="actor"):
        registry.register(_bundle(tmp_path), actor="  ")


def test_missing_model_file(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "model.joblib").unlink()
    with pytest.raises(Exception, match="incomplete"):
        LocalModelRegistry(tmp_path / "r").register(bundle, actor="a")


def test_get_unknown_version(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        LocalModelRegistry(tmp_path / "r").get(1)


def test_entry_metadata_is_validated() -> None:
    with pytest.raises(ValidationError):
        RegistryEntry(
            version=0,
            model_id="x",
            model_name="x",
            stage="candidate",  # type: ignore[arg-type]
            bundle_dir="x",
            model_sha256="short",
            dataset_sha256=None,
            data_contract_version="1.0.0",
            feature_names=[],
            training_timestamp="t",
            registered_at="t",
            code_revision=None,
            metrics={},
            stage_history=[],
        )


def test_entries_record_contract_version_and_training_timestamp(tmp_path: Path) -> None:
    entry = LocalModelRegistry(tmp_path / "r").register(_bundle(tmp_path), actor="a")
    assert entry.data_contract_version == "1.0.0"
    assert entry.training_timestamp == "2026-02-02T00:00:00+00:00"
    assert entry.metrics == {"validation": {"average_precision": 0.4}}


def test_registry_does_not_unpickle_on_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("joblib.load must not be called during registration")

    monkeypatch.setattr("joblib.load", forbidden)
    LocalModelRegistry(tmp_path / "r").register(_bundle(tmp_path), actor="a")


def test_code_revision_outside_git_is_none(tmp_path: Path) -> None:
    assert current_code_revision(tmp_path) is None


def test_code_revision_without_git_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module, "which", lambda _name: None)
    assert current_code_revision(tmp_path) is None


def test_tracking_disabled_by_default() -> None:
    assert isinstance(build_tracker(load_settings()), NullTracker)


def test_default_tracking_uri_is_local_sqlite(tmp_path: Path) -> None:
    uri = default_tracking_uri(tmp_path / "artifacts")
    assert uri.startswith("sqlite:///")
    assert uri.endswith("artifacts/mlflow/mlflow.db")


def test_cli_registry_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FRAUD_RISK_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    bundle = _bundle(tmp_path / "models")
    assert main(["registry", "register", "--bundle", str(bundle), "--actor", "tester"]) == 0
    registered = json.loads(capsys.readouterr().out)
    assert registered["entry"]["stage"] == "candidate"
    assert registered["tracking_run"] is None
    args = ["registry", "transition", "--version", "1", "--to", "approved"]
    assert main([*args, "--actor", "tester", "--reason", "demo"]) == 0
    capsys.readouterr()
    assert main(["registry", "list"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["approved_version"] == 1


def test_cli_registry_rejects_promotion_to_candidate(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["registry", "transition", "--version", "1", "--to", "candidate", "--actor", "a",
              "--reason", "x"])  # fmt: skip
    capsys.readouterr()


def test_cli_registry_requires_actor(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["registry", "register", "--bundle", "x"])
    capsys.readouterr()


def test_cli_registry_error_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FRAUD_RISK_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    args = ["registry", "transition", "--version", "5", "--to", "retired"]
    assert main([*args, "--actor", "a", "--reason", "x"]) == 1
    assert "REGISTRY_ERROR" in capsys.readouterr().err
