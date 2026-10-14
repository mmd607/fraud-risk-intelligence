"""Deterministic test doubles."""

from __future__ import annotations

import pandas as pd

from fraud_risk.policy.config import RiskPolicy
from fraud_risk.risk.engine import RiskEngine
from fraud_risk.services.inference import ModelDescription, ScoredTransaction

TEST_POLICY = RiskPolicy(version="test-policy", review_threshold=0.3, block_threshold=0.8)


def valid_transaction(**overrides: float) -> dict[str, float]:
    """A schema-valid synthetic transaction payload."""
    payload: dict[str, float] = {"Time": 10.0, "Amount": 25.0}
    payload.update({f"V{i}": 0.0 for i in range(1, 29)})
    payload.update(overrides)
    return payload


class FakeInferenceService:
    """Scores = clip(V1, 0, 1); lets tests choose the band via V1."""

    def __init__(self, *, ready: bool = True, fail: bool = False) -> None:
        self._ready = ready
        self._fail = fail
        self._engine = RiskEngine(TEST_POLICY, log_decisions=False)
        self.calls = 0

    @property
    def ready(self) -> bool:
        return self._ready

    def describe(self) -> ModelDescription:
        return ModelDescription("fake-model-1", "fake", None, TEST_POLICY.version, ("Time",))

    def predict(
        self, features: pd.DataFrame, *, references: list[str | None], explain: bool = False
    ) -> list[ScoredTransaction]:
        self.calls += 1
        if self._fail:
            raise RuntimeError("boom with secret detail 4111")
        scores = features["V1"].clip(0, 1).tolist()
        explanation = {"available": False, "reason": "fake"} if explain else None
        return [
            ScoredTransaction(
                self._engine.decide(s, model_version="fake-model-1", reference=r), explanation
            )
            for s, r in zip(scores, references, strict=True)
        ]
