"""Test-phase T07: precedence, boundaries, independence, privacy, costs, and CLI (risk)."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import pandas as pd
import pytest

import fraud_risk
from fraud_risk.cli.main import main
from fraud_risk.core.logging import configure_logging
from fraud_risk.policy.config import AnomalyEscalation, RiskPolicy, policy_to_json
from fraud_risk.risk.costs import CostAssumptions, evaluate_policy
from fraud_risk.risk.engine import Action, RiskBand, RiskEngine
from fraud_risk.risk.reason_codes import ReasonCode
from fraud_risk.risk.review_queue import build_review_queue, capacity_tradeoff

PACKAGE_ROOT = Path(fraud_risk.__file__).parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _policy(**overrides: object) -> RiskPolicy:
    base: dict[str, object] = {"version": "v1", "review_threshold": 0.2, "block_threshold": 0.7}
    return RiskPolicy(**{**base, **overrides})  # type: ignore[arg-type]


def _imports(module_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in module_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
            elif isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("package", ["risk", "policy"])
def test_policy_layer_is_independent_of_training(package: str) -> None:
    imports = _imports(PACKAGE_ROOT / package)
    forbidden = [
        name
        for name in imports
        if name.startswith(("fraud_risk.training", "fraud_risk.models", "sklearn", "joblib"))
    ]
    assert forbidden == []


EXPECTED_REASON_CODES = {
    "SCORE_BELOW_REVIEW_THRESHOLD": "R001",
    "SCORE_AT_OR_ABOVE_REVIEW_THRESHOLD": "R002",
    "SCORE_AT_OR_ABOVE_BLOCK_THRESHOLD": "R003",
    "SCORE_MISSING_OR_INVALID": "R004",
    "ANOMALY_SCORE_ESCALATION": "R005",
}


def test_reason_codes_are_stable_and_documented() -> None:
    assert {c.name: c.value for c in ReasonCode} == EXPECTED_REASON_CODES
    doc = (REPO_ROOT / "docs" / "risk-policy.md").read_text(encoding="utf-8")
    for name, value in EXPECTED_REASON_CODES.items():
        assert value in doc
        assert name in doc


def test_every_decision_has_reason_codes() -> None:
    engine = RiskEngine(_policy(), log_decisions=False)
    for score in [None, 0.0, 0.1, 0.2, 0.5, 0.7, 1.0, float("nan")]:
        assert engine.decide(score).reason_codes


def test_extreme_but_valid_thresholds() -> None:
    engine = RiskEngine(_policy(review_threshold=0.0, block_threshold=1.0), log_decisions=False)
    assert engine.decide(0.0).action is Action.REVIEW
    assert engine.decide(0.999999).action is Action.REVIEW
    assert engine.decide(1.0).action is Action.BLOCK


def test_missing_score_precedence_beats_escalation() -> None:
    policy = _policy(anomaly_escalation=AnomalyEscalation(enabled=True, score_threshold=0.1))
    decision = RiskEngine(policy, log_decisions=False).decide(None, anomaly_score=0.9)
    assert decision.reason_codes == (ReasonCode.SCORE_MISSING_OR_INVALID,)


def test_escalation_threshold_is_inclusive_and_ignores_non_finite() -> None:
    policy = _policy(anomaly_escalation=AnomalyEscalation(enabled=True, score_threshold=0.5))
    engine = RiskEngine(policy, log_decisions=False)
    assert engine.decide(0.05, anomaly_score=0.5).action is Action.REVIEW
    assert engine.decide(0.05, anomaly_score=float("nan")).action is Action.APPROVE
    assert engine.decide(0.05, anomaly_score=None).action is Action.APPROVE


def test_escalation_does_not_change_band() -> None:
    policy = _policy(anomaly_escalation=AnomalyEscalation(enabled=True, score_threshold=0.5))
    decision = RiskEngine(policy, log_decisions=False).decide(0.05, anomaly_score=0.9)
    assert decision.band is RiskBand.LOW
    assert decision.action is Action.REVIEW


def test_escalation_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic ValidationError
        AnomalyEscalation(enabled=True, score_threshold=0.0)


def test_policy_rejects_unknown_fields_and_bad_capacity() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        _policy(auto_refund=True)
    with pytest.raises(ValueError):  # noqa: PT011
        _policy(review_capacity=0)
    with pytest.raises(ValueError):  # noqa: PT011
        _policy(missing_score_action="approve")


def test_policy_round_trip_json(tmp_path: Path) -> None:
    from fraud_risk.policy.config import load_policy

    path = tmp_path / "p.json"
    path.write_text(policy_to_json(_policy()), encoding="utf-8")
    assert load_policy(path) == _policy()


def test_decision_log_contains_metadata_only() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    RiskEngine(_policy()).decide(0.5123456, model_version="m9", reference="tx-secret-77")
    record = json.loads(stream.getvalue())
    assert record["event"] == "risk_decision"
    assert record["action"] == "review"
    assert record["reason_codes"] == ["R002"]
    assert "0.5123456" not in stream.getvalue()
    assert "tx-secret-77" not in stream.getvalue()


def test_batch_with_anomaly_scores_and_references() -> None:
    policy = _policy(anomaly_escalation=AnomalyEscalation(enabled=True, score_threshold=0.5))
    decisions = RiskEngine(policy, log_decisions=False).decide_batch(
        [0.1, 0.1, 0.9], anomaly_scores=[0.9, None, 0.0], references=["a", "b", "c"]
    )
    assert [d.action for d in decisions] == [Action.REVIEW, Action.APPROVE, Action.BLOCK]
    assert [d.reference for d in decisions] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="anomaly_scores"):
        RiskEngine(policy).decide_batch([0.1], anomaly_scores=[0.1, 0.2])


def test_cost_with_partial_review_detection() -> None:
    outcome = evaluate_policy(
        [1, 1], [0.5, 0.5], _policy(), CostAssumptions(review_detection_rate=0.5)
    )
    # two reviewed frauds: one caught on average, one missed -> 1*100 + 2*2
    assert outcome.total_cost == pytest.approx(104.0)
    assert outcome.fraud_capture_rate == pytest.approx(0.5)


def test_zero_costs_and_no_frauds() -> None:
    zero = CostAssumptions(missed_fraud_cost=0, false_block_cost=0, review_cost=0)
    outcome = evaluate_policy([0, 0], [0.9, 0.1], _policy(), zero)
    assert outcome.total_cost == 0.0
    assert outcome.fraud_capture_rate == 0.0
    assert outcome.block_precision == 0.0


def test_cost_assumptions_validated() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        CostAssumptions(review_detection_rate=1.5)
    with pytest.raises(ValueError):  # noqa: PT011
        CostAssumptions(missed_fraud_cost=-1)


def test_evaluate_policy_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equally long"):
        evaluate_policy([0, 1], [0.5], _policy())


def test_review_queue_edge_cases() -> None:
    assert [i.reference for i in build_review_queue(["a", "b"], [float("nan"), float("inf")])] == []
    assert build_review_queue(["a"], [float("nan")]) == []
    assert len(build_review_queue(["a", "b"], [0.1, 0.2], capacity=10)) == 2
    with pytest.raises(ValueError, match="same length"):
        build_review_queue(["a"], [0.1, 0.2])


def test_capacity_tradeoff_caps_at_population() -> None:
    points = capacity_tradeoff([0, 1], [0.2, 0.9], [5])
    assert points[0].capacity == 2
    assert points[0].recall == 1.0


# --- CLI ---------------------------------------------------------------------------------------


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    pd.DataFrame(
        {"row_id": range(6), "label": [0, 0, 1, 1, 0, 1], "score": [0.1, 0.3, 0.9, 0.6, 0.05, 0.0]}
    ).to_csv(bundle / "scores_validation.csv", index=False)
    return bundle


def test_cli_evaluate_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy_path = tmp_path / "p.json"
    policy_path.write_text(policy_to_json(_policy()), encoding="utf-8")
    out = tmp_path / "out.json"
    code = main(
        [
            "evaluate-policy",
            "--policy", str(policy_path),
            "--bundle", str(_bundle(tmp_path)),
            "--output", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"]["blocked_fraud"] == 1
    assert payload["outcome"]["n_transactions"] == 6
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_cli_evaluate_policy_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"version": "x", "review_threshold": 0.9, "block_threshold": 0.1}')
    assert main(["evaluate-policy", "--policy", str(bad), "--bundle", str(tmp_path)]) == 1
    assert "CONFIGURATION_ERROR" in capsys.readouterr().err
    good = tmp_path / "good.json"
    good.write_text(policy_to_json(_policy()), encoding="utf-8")
    args = ["evaluate-policy", "--policy", str(good), "--bundle", str(_bundle(tmp_path))]
    assert main([*args, "--partition", "test"]) == 1
    assert "No test scores" in capsys.readouterr().err
