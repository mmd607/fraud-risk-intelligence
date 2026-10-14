"""Test-phase T06: label isolation, score semantics, determinism, and CLI for anomaly detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from fraud_risk.anomaly.detectors import IsolationForestConfig
from fraud_risk.anomaly.evaluation import evaluate_anomaly_ranking
from fraud_risk.cli.main import main
from fraud_risk.data.schema import TARGET_COLUMN
from fraud_risk.data.splitting import SplitConfig, split_dataset
from fraud_risk.evaluation.metrics import MetricsInputError
from fraud_risk.training.anomaly import AnomalyConfig, FitOn, run_anomaly_training
from fraud_risk.training.artifacts import load_model_bundle
from tests.support.synthetic import make_transactions, write_csv


def _config(fit_on: FitOn = FitOn.ALL, seed: int = 3) -> AnomalyConfig:
    return AnomalyConfig(
        split=SplitConfig(seed=seed),
        detector=IsolationForestConfig(n_estimators=40, seed=seed),
        fit_on=fit_on,
        review_budgets=(5, 20),
    )


def _run(tmp_path: Path, frame: pd.DataFrame, config: AnomalyConfig) -> Any:
    return run_anomaly_training(
        frame,
        dataset_sha256=None,
        config=config,
        models_dir=tmp_path / "m",
        reports_dir=tmp_path / "r",
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_transactions(1_500, 45, seed=13, signal=4.0)


def test_label_never_reaches_isolation_forest(
    tmp_path: Path, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real_fit = IsolationForest.fit

    def spy(self: IsolationForest, x: Any, y: Any = None, **kwargs: Any) -> Any:
        seen.append((list(getattr(x, "columns", [])), y))
        return real_fit(self, x, y, **kwargs)

    monkeypatch.setattr(IsolationForest, "fit", spy)
    _run(tmp_path, frame, _config())
    assert seen
    for columns, target in seen:
        assert TARGET_COLUMN not in columns
        assert target is None


def test_unsupervised_mode_fits_on_whole_training_partition(
    tmp_path: Path, frame: pd.DataFrame
) -> None:
    result = _run(tmp_path, frame, _config(FitOn.ALL))
    expected = split_dataset(frame, SplitConfig(seed=3)).metadata.train.rows
    assert result.training_rows == expected


def test_legitimate_mode_excludes_training_frauds(tmp_path: Path, frame: pd.DataFrame) -> None:
    meta = split_dataset(frame, SplitConfig(seed=3)).metadata
    result = _run(tmp_path, frame, _config(FitOn.LEGITIMATE))
    assert result.training_rows == meta.train.rows - meta.train.positives


def test_workflow_is_deterministic(tmp_path: Path, frame: pd.DataFrame) -> None:
    a = _run(tmp_path / "a", frame, _config())
    b = _run(tmp_path / "b", frame, _config())
    assert a.test.to_dict() == b.test.to_dict()


def test_anomaly_scores_are_not_probabilities(tmp_path: Path, frame: pd.DataFrame) -> None:
    result = _run(tmp_path, frame, _config())
    scores = pd.read_csv(result.bundle_dir / "scores_test.csv")["score"]
    bundle = load_model_bundle(result.bundle_dir)
    assert bundle.manifest.threshold["strategy"] == "none"
    assert "Not a probability" in bundle.manifest.notes
    # IsolationForest negated score_samples lies in (0, 1] but is not calibrated; ensure it is
    # used as-is (no clipping/rescaling to look like a probability) by checking it is positive.
    assert (scores > 0).all()


def test_report_flags_label_usage(tmp_path: Path, frame: pd.DataFrame) -> None:
    for mode in FitOn:
        result = _run(tmp_path / mode.value, frame, _config(mode))
        payload = json.loads(result.report_path.read_text(encoding="utf-8"))
        assert payload["uses_training_labels"] is (mode is FitOn.LEGITIMATE)
        assert payload["test"]["score_semantics"].startswith("anomaly score")


def test_budget_larger_than_partition_is_capped() -> None:
    report = evaluate_anomaly_ranking([0, 1, 0], [0.1, 0.9, 0.2], (10,))
    assert report.budgets[0]["k"] == 3
    assert report.budgets[0]["recall_at_k"] == 1.0


def test_evaluation_rejects_mismatched_lengths() -> None:
    with pytest.raises(MetricsInputError):
        evaluate_anomaly_ranking([0, 1], [0.1], (1,))


def test_evaluation_rejects_non_finite_scores() -> None:
    with pytest.raises(MetricsInputError):
        evaluate_anomaly_ranking([0, 1], [0.1, np.inf], (1,))


def test_cli_train_anomaly(
    tmp_path: Path,
    frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv = write_csv(frame, tmp_path / "d.csv")
    monkeypatch.setenv("FRAUD_RISK_ARTIFACTS_DIR", str(tmp_path / "a"))
    monkeypatch.setenv("FRAUD_RISK_REPORTS_DIR", str(tmp_path / "r"))
    assert main(["train-anomaly", "--path", str(csv), "--fit-on", "legitimate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fit_on"] == "legitimate"
    assert (tmp_path / "r" / "anomaly_metrics_legitimate.json").is_file()


def test_cli_train_anomaly_missing_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train-anomaly", "--path", str(tmp_path / "none.csv")]) == 1
    assert "DATA_NOT_FOUND" in capsys.readouterr().err
