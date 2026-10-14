"""Local-first model registry with an explicit, auditable lifecycle.

The registry is a JSON index (``artifacts/registry/registry.json``, Git-ignored) of model
bundles. Each entry records version, model id, stage, schema/contract version, feature list,
training timestamp, code revision, dataset fingerprint, model hash, and headline metrics.

Lifecycle
---------
``candidate`` → ``approved`` → ``retired`` (and ``candidate`` → ``retired``).

* Registration always creates a ``candidate``. Nothing is ever promoted automatically.
* Every transition is an explicit operator action that needs an ``actor`` and a ``reason``, and
  it is appended to the entry's ``stage_history``.
* At most one model can be ``approved``. Approving another requires retiring the current one
  first, which keeps the operator in control of which model serves.
* "Approved" means approved for this local demo's inference service. It is not production
  approval of any kind.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from shutil import which
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fraud_risk.core.exceptions import ConfigurationError, ModelError
from fraud_risk.data.contracts import CONTRACT_VERSION
from fraud_risk.data.ingestion import file_sha256
from fraud_risk.registry.tracking import ExperimentTracker
from fraud_risk.training.artifacts import MODEL_FILENAME, ModelArtifactError, read_manifest

REGISTRY_FILENAME = "registry.json"
REGISTRY_SCHEMA_VERSION = "1.0.0"


class RegistryError(ModelError):
    """Registry operation is invalid (unknown version, illegal transition, ...)."""

    code = "REGISTRY_ERROR"


class ModelStage(StrEnum):
    """Lifecycle stages."""

    CANDIDATE = "candidate"
    APPROVED = "approved"
    RETIRED = "retired"


ALLOWED_TRANSITIONS: dict[ModelStage, frozenset[ModelStage]] = {
    ModelStage.CANDIDATE: frozenset({ModelStage.APPROVED, ModelStage.RETIRED}),
    ModelStage.APPROVED: frozenset({ModelStage.RETIRED}),
    ModelStage.RETIRED: frozenset(),
}


class StageTransition(BaseModel):
    """One audited stage change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_stage: ModelStage | None
    to_stage: ModelStage
    at: str
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=500)


class RegistryEntry(BaseModel):
    """Validated metadata of one registered model version."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    model_id: str
    model_name: str
    stage: ModelStage
    bundle_dir: str
    model_sha256: str = Field(min_length=64, max_length=64)
    dataset_sha256: str | None
    data_contract_version: str
    feature_names: list[str] = Field(min_length=1)
    training_timestamp: str
    registered_at: str
    code_revision: str | None
    metrics: dict[str, Any]
    stage_history: list[StageTransition]


class RegistryIndex(BaseModel):
    """The whole registry file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = REGISTRY_SCHEMA_VERSION
    entries: list[RegistryEntry] = Field(default_factory=list)


def current_code_revision(repo_dir: Path) -> str | None:
    """``git rev-parse HEAD`` (suffixed ``-dirty`` when uncommitted changes exist), else None."""
    git = which("git")
    if git is None:
        return None
    try:
        head = subprocess.run(  # noqa: S603 - fixed executable and arguments
            [git, "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if status else head


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _headline_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for partition in ("validation", "test"):
        values = metrics.get(partition)
        if isinstance(values, dict):
            summary[partition] = {
                key: values.get(key)
                for key in (
                    "average_precision",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                    "threshold",
                )
                if key in values
            }
    return summary


class LocalModelRegistry:
    """JSON-file registry. Writes are atomic (temp file + replace)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / REGISTRY_FILENAME

    def _load(self) -> RegistryIndex:
        if not self.path.is_file():
            return RegistryIndex()
        try:
            return RegistryIndex.model_validate_json(self.path.read_text(encoding="utf-8"))
        except ValidationError:
            msg = f"Registry file '{self.path}' is invalid; refusing to modify it."
            raise RegistryError(msg) from None

    def _save(self, index: RegistryIndex) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        text = json.dumps(index.model_dump(mode="json"), indent=2) + "\n"
        handle, temp_name = tempfile.mkstemp(dir=self.root, prefix=".registry-", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(temp_name).replace(self.path)

    def entries(self) -> list[RegistryEntry]:
        """All entries, oldest first."""
        return list(self._load().entries)

    def get(self, version: int) -> RegistryEntry:
        """Entry by version number."""
        for entry in self._load().entries:
            if entry.version == version:
                return entry
        msg = f"Model version {version} is not registered."
        raise RegistryError(msg)

    def approved(self) -> RegistryEntry | None:
        """The approved entry, if any."""
        return next((e for e in self._load().entries if e.stage is ModelStage.APPROVED), None)

    def register(
        self, bundle_dir: Path, *, actor: str, code_revision: str | None = None
    ) -> RegistryEntry:
        """Register a bundle as a new ``candidate`` after an integrity check (no unpickling)."""
        if not actor.strip():
            msg = "actor is required to register a model."
            raise ConfigurationError(msg)
        manifest = read_manifest(bundle_dir)
        model_path = bundle_dir / MODEL_FILENAME
        if not model_path.is_file() or manifest.model_sha256 is None:
            msg = f"Bundle '{bundle_dir.name}' is incomplete."
            raise ModelArtifactError(msg)
        if file_sha256(model_path) != manifest.model_sha256:
            msg = f"Integrity check failed for bundle '{bundle_dir.name}'."
            raise ModelArtifactError(msg)
        index = self._load()
        if any(e.model_id == manifest.model_id for e in index.entries):
            msg = f"Model '{manifest.model_id}' is already registered."
            raise RegistryError(msg)
        now = _now()
        entry = RegistryEntry(
            version=max((e.version for e in index.entries), default=0) + 1,
            model_id=manifest.model_id,
            model_name=manifest.model_name,
            stage=ModelStage.CANDIDATE,
            bundle_dir=str(bundle_dir.resolve()),
            model_sha256=manifest.model_sha256,
            dataset_sha256=manifest.dataset_sha256,
            data_contract_version=CONTRACT_VERSION,
            feature_names=manifest.feature_names,
            training_timestamp=manifest.created_at,
            registered_at=now,
            code_revision=code_revision,
            metrics=_headline_metrics(manifest.metrics),
            stage_history=[
                StageTransition(
                    from_stage=None,
                    to_stage=ModelStage.CANDIDATE,
                    at=now,
                    actor=actor,
                    reason="registered",
                )
            ],
        )
        index.entries.append(entry)
        self._save(index)
        return entry

    def transition(
        self, version: int, to_stage: ModelStage, *, actor: str, reason: str
    ) -> RegistryEntry:
        """Explicitly move a version to ``to_stage`` (see module docstring for rules)."""
        if not actor.strip() or not reason.strip():
            msg = "Stage transitions require a non-empty actor and reason."
            raise RegistryError(msg)
        index = self._load()
        entry = next((e for e in index.entries if e.version == version), None)
        if entry is None:
            msg = f"Model version {version} is not registered."
            raise RegistryError(msg)
        if to_stage not in ALLOWED_TRANSITIONS[entry.stage]:
            msg = f"Transition {entry.stage.value} -> {to_stage.value} is not allowed."
            raise RegistryError(msg)
        if to_stage is ModelStage.APPROVED:
            current = next((e for e in index.entries if e.stage is ModelStage.APPROVED), None)
            if current is not None:
                msg = (
                    f"Version {current.version} is already approved; retire it explicitly "
                    "before approving another version."
                )
                raise RegistryError(msg)
        entry.stage_history.append(
            StageTransition(
                from_stage=entry.stage, to_stage=to_stage, at=_now(), actor=actor, reason=reason
            )
        )
        entry.stage = to_stage
        self._save(index)
        return entry


def track_entry(tracker: ExperimentTracker, entry: RegistryEntry) -> str:
    """Log a registry entry's metadata (no binaries, no data) to an experiment tracker."""
    with tracker.start_run(entry.model_id) as run:
        run.log_params(
            {
                "model_name": entry.model_name,
                "registry_version": entry.version,
                "data_contract_version": entry.data_contract_version,
                "feature_count": len(entry.feature_names),
            }
        )
        for partition, values in entry.metrics.items():
            if isinstance(values, dict):
                run.log_metrics({f"{partition}_{key}": value for key, value in values.items()})
        run.set_tags(
            {
                "model_id": entry.model_id,
                "stage": entry.stage.value,
                "model_sha256": entry.model_sha256,
                "dataset_sha256": entry.dataset_sha256,
                "code_revision": entry.code_revision,
                "artifact_reference": entry.bundle_dir,
            }
        )
        return run.run_id
