"""Narrow experiment-tracking abstraction with a no-op default and an optional MLflow backend.

* :class:`NullTracker` — default. Records nothing, needs nothing: unit tests and offline use.
* :class:`InMemoryTracker` — keeps runs in memory, for tests and dry runs.
* :class:`MlflowTracker` — optional (``pip install "fraud-risk[tracking]"``). The tracking URI
  is configurable and defaults to a local, Git-ignored SQLite file under ``artifacts/mlflow``.

Only scalar parameters/metrics, tags, and artifact *references* (paths and hashes, as tags) are
logged. Binaries are not uploaded, and keys that look like secrets or transaction payloads are
dropped (see :mod:`fraud_risk.core.logging`).
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fraud_risk.core.exceptions import DependencyUnavailableError
from fraud_risk.core.logging import SENSITIVE_KEY_MARKERS

Scalar = str | int | float | bool


def _is_safe_key(key: str) -> bool:
    lowered = key.lower()
    return not any(marker in lowered for marker in SENSITIVE_KEY_MARKERS)


def sanitize_params(params: Mapping[str, object]) -> dict[str, Scalar]:
    """Keep scalar values under safe keys; stringify ``None``; drop everything else."""
    clean: dict[str, Scalar] = {}
    for key, value in params.items():
        if not _is_safe_key(key):
            continue
        if value is None:
            clean[key] = "None"
        elif isinstance(value, str | int | float | bool):
            clean[key] = value
    return clean


def sanitize_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Keep finite numeric values under safe keys."""
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if (
            _is_safe_key(key)
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            clean[key] = float(value)
    return clean


class TrackingRun(Protocol):
    """Handle of an active run."""

    run_id: str

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log scalar parameters."""
        ...

    def log_metrics(self, metrics: Mapping[str, object]) -> None:
        """Log numeric metrics."""
        ...

    def set_tags(self, tags: Mapping[str, object]) -> None:
        """Set string tags."""
        ...


class ExperimentTracker(Protocol):
    """Factory of tracking runs."""

    enabled: bool

    def start_run(self, name: str) -> AbstractContextManager[TrackingRun]:
        """Context manager yielding a :class:`TrackingRun`."""
        ...


@dataclass
class RecordedRun:
    """A run captured in memory (also used as the null run)."""

    run_id: str
    name: str
    params: dict[str, Scalar] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    record: bool = True

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log scalar parameters."""
        if self.record:
            self.params.update(sanitize_params(params))

    def log_metrics(self, metrics: Mapping[str, object]) -> None:
        """Log numeric metrics."""
        if self.record:
            self.metrics.update(sanitize_metrics(metrics))

    def set_tags(self, tags: Mapping[str, object]) -> None:
        """Set string tags."""
        if self.record:
            self.tags.update({k: str(v) for k, v in sanitize_params(tags).items()})


class NullTracker:
    """Tracking disabled: runs accept calls and discard them."""

    enabled = False

    @contextmanager
    def start_run(self, name: str) -> Iterator[TrackingRun]:
        """Yield a run that records nothing."""
        yield RecordedRun(run_id="disabled", name=name, record=False)


class InMemoryTracker:
    """Keeps runs in memory (tests, dry runs)."""

    enabled = True

    def __init__(self) -> None:
        self.runs: list[RecordedRun] = []

    @contextmanager
    def start_run(self, name: str) -> Iterator[TrackingRun]:
        """Yield an in-memory run and keep it after the block ends."""
        run = RecordedRun(run_id=f"mem-{len(self.runs) + 1}", name=name)
        self.runs.append(run)
        yield run


class _MlflowRun:
    def __init__(self, client: object, run_id: str) -> None:
        self._client = client
        self.run_id = run_id

    def log_params(self, params: Mapping[str, object]) -> None:
        import mlflow  # noqa: PLC0415 - optional dependency

        if clean := sanitize_params(params):
            mlflow.log_params(clean)

    def log_metrics(self, metrics: Mapping[str, object]) -> None:
        import mlflow  # noqa: PLC0415

        if clean := sanitize_metrics(metrics):
            mlflow.log_metrics(clean)

    def set_tags(self, tags: Mapping[str, object]) -> None:
        import mlflow  # noqa: PLC0415

        if clean := sanitize_params(tags):
            mlflow.set_tags({k: str(v) for k, v in clean.items()})


def default_tracking_uri(artifacts_dir: Path) -> str:
    """Local SQLite tracking store under the (Git-ignored) artifacts directory."""
    db_path = (artifacts_dir / "mlflow" / "mlflow.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


class MlflowTracker:
    """MLflow-backed tracker (optional dependency)."""

    enabled = True

    def __init__(self, tracking_uri: str, experiment: str = "fraud-risk") -> None:
        if importlib.util.find_spec("mlflow") is None:
            raise DependencyUnavailableError(
                "mlflow", extra="tracking", purpose="experiment tracking"
            )
        if tracking_uri.startswith("sqlite:///"):
            Path(tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = tracking_uri
        self.experiment = experiment

    @contextmanager
    def start_run(self, name: str) -> Iterator[TrackingRun]:
        """Start an MLflow run in the configured experiment."""
        import mlflow  # noqa: PLC0415 - optional dependency

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        with mlflow.start_run(run_name=name) as active:
            yield _MlflowRun(mlflow, active.info.run_id)
