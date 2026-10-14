from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_risk.anomaly.detectors import (
    AnomalyDetector,
    AnomalyInputError,
    DetectorNotFittedError,
    IsolationForestConfig,
    IsolationForestDetector,
)
from fraud_risk.anomaly.evaluation import evaluate_anomaly_ranking
from fraud_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_risk.data.splitting import SplitConfig
from fraud_risk.training.anomaly import AnomalyConfig, FitOn, run_anomaly_training
from tests.support.synthetic import make_transactions


@pytest.fixture
def features() -> pd.DataFrame:
    return make_transactions(800, 20, seed=2, signal=4.0).drop(columns=[TARGET_COLUMN])


def _detector(seed: int = 0) -> IsolationForestDetector:
    return IsolationForestDetector(IsolationForestConfig(n_estimators=50, seed=seed))


def test_detector_satisfies_protocol() -> None:
    assert isinstance(_detector(), AnomalyDetector)


def test_scores_shape_and_finiteness(features: pd.DataFrame) -> None:
    scores = _detector().fit(features).score(features)
    assert scores.shape == (len(features),)
    assert np.all(np.isfinite(scores))


def test_scores_are_deterministic(features: pd.DataFrame) -> None:
    a = _detector(3).fit(features).score(features)
    b = _detector(3).fit(features).score(features)
    np.testing.assert_array_equal(a, b)


def test_extreme_point_scores_higher(features: pd.DataFrame) -> None:
    detector = _detector().fit(features)
    outlier = features.iloc[[0]].copy()
    outlier[list(FEATURE_COLUMNS[1:29])] = 25.0
    assert detector.score(outlier)[0] > np.percentile(detector.score(features), 99)


def test_target_column_rejected(features: pd.DataFrame) -> None:
    with pytest.raises(AnomalyInputError, match="target"):
        _detector().fit(features.assign(Class=0))


@pytest.mark.parametrize("bad", ["missing", "nan", "empty", "array"])
def test_invalid_input(features: pd.DataFrame, bad: str) -> None:
    data: object
    if bad == "missing":
        data = features.drop(columns=["V2"])
    elif bad == "nan":
        data = features.assign(V2=np.nan)
    elif bad == "empty":
        data = features.iloc[0:0]
    else:
        data = features.to_numpy()
    with pytest.raises(AnomalyInputError):
        _detector().fit(data)  # type: ignore[arg-type]


def test_score_before_fit(features: pd.DataFrame) -> None:
    with pytest.raises(DetectorNotFittedError):
        _detector().score(features)


def test_ranking_evaluation() -> None:
    report = evaluate_anomaly_ranking([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8], (1, 2))
    assert report.average_precision == 1.0
    assert report.budgets[1]["recall_at_k"] == 1.0
    assert "not a probability" in str(report.to_dict()["score_semantics"])


def test_ranking_evaluation_single_class_is_undefined() -> None:
    report = evaluate_anomaly_ranking([0, 0], [0.1, 0.2], (1,))
    assert report.average_precision is None


@pytest.mark.parametrize("fit_on", list(FitOn))
def test_training_workflow(tmp_path: Path, fit_on: FitOn) -> None:
    frame = make_transactions(2_000, 60, seed=5, signal=4.0)
    result = run_anomaly_training(
        frame,
        dataset_sha256=None,
        config=AnomalyConfig(
            split=SplitConfig(seed=1),
            detector=IsolationForestConfig(n_estimators=50),
            fit_on=fit_on,
            review_budgets=(10, 50),
        ),
        models_dir=tmp_path / "m",
        reports_dir=tmp_path / "r",
    )
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["uses_training_labels"] is (fit_on is FitOn.LEGITIMATE)
    train_rows = result.training_rows
    if fit_on is FitOn.LEGITIMATE:
        assert train_rows < 1_200
    assert result.test.average_precision is not None
    assert (result.bundle_dir / "scores_test.csv").is_file()
