"""Local model artifact persistence with integrity checks.

A *model bundle* is a directory containing ``model.joblib`` and ``manifest.json``. The manifest
records configuration, feature names, seeds, library versions, dataset fingerprint, split
metadata, threshold, metrics, and the SHA-256 of the model file.

Security: joblib files are pickles and can execute code when loaded. :func:`load_model_bundle`
verifies the recorded SHA-256 *before* unpickling and refuses mismatches. Only load bundles you
created yourself. Bundles live under the Git-ignored ``artifacts/`` directory.
"""

from __future__ import annotations

import json
import platform
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fraud_risk.core.exceptions import ModelError
from fraud_risk.data.ingestion import file_sha256

MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0.0"
_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,127}$")
_TRACKED_PACKAGES = ("fraud-risk", "scikit-learn", "numpy", "pandas", "joblib", "xgboost")


class ModelArtifactError(ModelError):
    """A model bundle is missing, corrupt, tampered with, or inconsistent."""

    code = "MODEL_ARTIFACT_INVALID"


class ModelManifest(BaseModel):
    """Metadata stored next to every saved model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = MANIFEST_SCHEMA_VERSION
    model_id: str
    model_name: str
    created_at: str
    seed: int
    target: str
    feature_names: list[str] = Field(min_length=1)
    transformed_feature_names: list[str] = Field(min_length=1)
    preprocessing: dict[str, str]
    model_params: dict[str, Any]
    threshold: dict[str, Any]
    metrics: dict[str, Any]
    dataset_sha256: str | None
    split: dict[str, Any]
    library_versions: dict[str, str]
    model_sha256: str | None = None
    notes: str = ""


def library_versions() -> dict[str, str]:
    """Installed versions of relevant packages plus the Python version."""
    versions = {"python": platform.python_version()}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def new_model_id(model_name: str, *, now: datetime | None = None) -> str:
    """Build an id such as ``logistic_regression-20260101t120000``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dt%H%M%S")
    return f"{model_name}-{stamp}".lower()


def utc_now_iso() -> str:
    """Current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A loaded model and its manifest."""

    model: Any
    manifest: ModelManifest
    directory: Path


def save_model_bundle(model: object, manifest: ModelManifest, root: Path) -> Path:
    """Persist ``model`` and ``manifest`` under ``root / manifest.model_id`` and return the dir.

    Refuses to overwrite an existing bundle directory.
    """
    if not _MODEL_ID_PATTERN.fullmatch(manifest.model_id):
        msg = f"Invalid model_id '{manifest.model_id}' (lowercase letters, digits, '._-')."
        raise ModelArtifactError(msg)
    directory = root / manifest.model_id
    if directory.exists():
        msg = f"Model bundle '{manifest.model_id}' already exists; refusing to overwrite."
        raise ModelArtifactError(msg)
    directory.mkdir(parents=True)
    model_path = directory / MODEL_FILENAME
    joblib.dump(model, model_path)
    final = manifest.model_copy(update={"model_sha256": file_sha256(model_path)})
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(final.model_dump(), indent=2) + "\n", encoding="utf-8"
    )
    return directory


def read_manifest(directory: Path) -> ModelManifest:
    """Read and validate a bundle manifest without loading the model."""
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        msg = f"No manifest found in model bundle '{directory.name}'."
        raise ModelArtifactError(msg)
    try:
        return ModelManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        msg = f"Manifest of '{directory.name}' is invalid ({error.error_count()} problems)."
        raise ModelArtifactError(msg) from None


def load_model_bundle(directory: Path) -> ModelBundle:
    """Load a bundle after verifying the model file's SHA-256 against the manifest."""
    manifest = read_manifest(directory)
    model_path = directory / MODEL_FILENAME
    if not model_path.is_file():
        msg = f"Model file missing in bundle '{directory.name}'."
        raise ModelArtifactError(msg)
    if manifest.model_sha256 is None or file_sha256(model_path) != manifest.model_sha256:
        msg = f"Integrity check failed for bundle '{directory.name}'; refusing to load."
        raise ModelArtifactError(msg)
    model = joblib.load(model_path)
    return ModelBundle(model=model, manifest=manifest, directory=directory)
