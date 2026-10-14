"""Inference service: scores validated feature rows and applies the risk policy.

The service is the seam between HTTP and the ML code. The API depends on the
:class:`InferenceService` protocol, so tests can inject a deterministic fake instead of a trained
model.

Model selection (explicit, never automatic):

1. ``FRAUD_RISK_MODEL_BUNDLE_PATH`` if set, otherwise
2. the registry's single **approved** version, otherwise
3. no model, in which case the service is not ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from fraud_risk.config.settings import Settings
from fraud_risk.core.exceptions import FraudRiskError, ModelError
from fraud_risk.core.logging import get_logger, log_event
from fraud_risk.data.schema import FEATURE_COLUMNS
from fraud_risk.explainability.explainers import (
    ExplainabilityError,
    LocalMethod,
    ModelInfo,
    explain_local,
    resolve_method,
)
from fraud_risk.policy.config import RiskPolicy, load_policy
from fraud_risk.registry.model_registry import LocalModelRegistry
from fraud_risk.risk.engine import RiskDecision, RiskEngine
from fraud_risk.training.artifacts import ModelBundle, load_model_bundle

_LOGGER = get_logger(__name__)


class ModelNotReadyError(ModelError):
    """No model is loaded; the service cannot score."""

    code = "MODEL_NOT_READY"


@dataclass(frozen=True, slots=True)
class ModelDescription:
    """Public model metadata (no filesystem paths)."""

    model_id: str
    model_name: str
    model_sha256: str | None
    policy_version: str
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoredTransaction:
    """Score, decision, and optional explanation for one transaction."""

    decision: RiskDecision
    explanation: dict[str, object] | None = None


class InferenceService(Protocol):
    """What the API needs from a model service."""

    @property
    def ready(self) -> bool:
        """True when a model is loaded and can score."""
        ...

    def describe(self) -> ModelDescription:
        """Public metadata of the loaded model."""
        ...

    def predict(
        self,
        features: pd.DataFrame,
        *,
        references: list[str | None],
        explain: bool = False,
    ) -> list[ScoredTransaction]:
        """Score rows of ``FEATURE_COLUMNS`` and decide."""
        ...


class ModelService:
    """Concrete service backed by a verified model bundle."""

    def __init__(self, bundle: ModelBundle | None, policy: RiskPolicy) -> None:
        self._bundle = bundle
        self._engine = RiskEngine(policy)
        self._policy = policy

    @property
    def ready(self) -> bool:
        """True when a bundle is loaded."""
        return self._bundle is not None

    def _require_bundle(self) -> ModelBundle:
        if self._bundle is None:
            msg = "No model is loaded. Configure FRAUD_RISK_MODEL_BUNDLE_PATH or approve a model."
            raise ModelNotReadyError(msg)
        return self._bundle

    def describe(self) -> ModelDescription:
        """Public metadata of the loaded model."""
        manifest = self._require_bundle().manifest
        return ModelDescription(
            model_id=manifest.model_id,
            model_name=manifest.model_name,
            model_sha256=manifest.model_sha256,
            policy_version=self._policy.version,
            feature_names=tuple(manifest.feature_names),
        )

    def predict(
        self,
        features: pd.DataFrame,
        *,
        references: list[str | None],
        explain: bool = False,
    ) -> list[ScoredTransaction]:
        """Score rows, apply the policy, and optionally explain each prediction."""
        bundle = self._require_bundle()
        x = features.loc[:, list(FEATURE_COLUMNS)]
        scores = np.asarray(bundle.model.predict_proba(x)[:, 1], dtype="float64")
        model_id = bundle.manifest.model_id
        results = []
        for position, score in enumerate(scores):
            decision = self._engine.decide(
                float(score), model_version=model_id, reference=references[position]
            )
            explanation = self._explain(bundle, x.iloc[[position]]) if explain else None
            results.append(ScoredTransaction(decision=decision, explanation=explanation))
        return results

    @staticmethod
    def _explain(bundle: ModelBundle, row: pd.DataFrame) -> dict[str, object]:
        manifest = bundle.manifest
        info = ModelInfo(manifest.model_id, manifest.model_name, manifest.model_sha256)
        try:
            method = resolve_method(bundle.model, LocalMethod.AUTO)
            if method is LocalMethod.OCCLUSION:
                return {"available": False, "reason": "no background sample for occlusion"}
            explanation = explain_local(bundle.model, row, model_info=info, method=method)
        except (ExplainabilityError, FraudRiskError) as error:
            return {"available": False, "reason": error.code}
        payload = explanation.to_dict(top_k=5)
        return {
            "available": True,
            "method": payload["method"],
            "output_space": payload["output_space"],
            "top_contributions": payload["contributions"],
            "caveat": payload["caveat"],
        }


def resolve_bundle_path(settings: Settings) -> Path | None:
    """Explicit bundle path, else the registry's approved version, else None."""
    if settings.model_bundle_path is not None:
        return settings.model_bundle_path
    approved = LocalModelRegistry(settings.registry_dir).approved()
    return Path(approved.bundle_dir) if approved is not None else None


def build_model_service(settings: Settings) -> ModelService:
    """Build the service from settings. A missing model yields a not-ready service."""
    policy = load_policy(settings.risk_policy_path)
    path = resolve_bundle_path(settings)
    bundle = load_model_bundle(path) if path is not None else None
    log_event(
        _LOGGER,
        "model_service_initialised",
        ready=bundle is not None,
        model_id=bundle.manifest.model_id if bundle else None,
        policy_version=policy.version,
    )
    return ModelService(bundle, policy)
