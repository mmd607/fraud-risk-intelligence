"""Test-phase T05: determinism, resampling isolation, test-set discipline, and fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_risk.core.exceptions import ConfigurationError, ModelError
from fraud_risk.data.splitting import SplitConfig, split_dataset
from fraud_risk.features.preprocessing import PreprocessingConfig
from fraud_risk.models import candidates as candidates_module
from fraud_risk.models.baseline import build_pipeline
from fraud_risk.models.candidates import CANDIDATES, CandidateSpec
from fraud_risk.training import experiments as experiments_module
from fraud_risk.training.experiments import (
    CandidateResult,
    ExperimentConfig,
    model_complexity,
    precision_at_recall_levels,
    run_model_comparison,
)
from tests.support.synthetic import make_transactions


def _run(tmp_path: Path, candidates: tuple[str, ...], seed: int = 5) -> Any:
    return run_model_comparison(
        make_transactions(1_500, 45, seed=seed, signal=2.5),
        dataset_sha256=None,
        config=ExperimentConfig(
            split=SplitConfig(seed=seed), candidates=candidates, cv_folds=2, n_jobs=1
        ),
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )


def test_comparison_is_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path / "a", ("logistic_regression", "random_forest"))
    second = _run(tmp_path / "b", ("logistic_regression", "random_forest"))
    assert [c.cv_ap_mean for c in first.candidates] == [c.cv_ap_mean for c in second.candidates]
    assert first.selected == second.selected
    assert first.test == second.test


def test_only_selected_model_touches_test_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = make_transactions(1_500, 45, seed=6, signal=2.5)
    test_index = set(split_dataset(frame, SplitConfig(seed=6)).test.index)
    calls: list[bool] = []
    real = experiments_module.positive_scores

    def spy(model: Any, features: Any) -> Any:
        calls.append(bool(set(features.index) & test_index))
        return real(model, features)

    monkeypatch.setattr(experiments_module, "positive_scores", spy)
    run_model_comparison(
        frame,
        dataset_sha256=None,
        config=ExperimentConfig(
            split=SplitConfig(seed=6),
            candidates=("logistic_regression", "hist_gradient_boosting"),
            cv_folds=2,
            n_jobs=1,
        ),
        models_dir=tmp_path / "m",
        reports_dir=tmp_path / "r",
    )
    assert calls.count(True) == 1  # exactly one scoring pass touches test rows


@pytest.mark.skipif(not CANDIDATES["logistic_regression_smote"].is_available(), reason="imblearn")
def test_smote_resamples_training_folds_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imblearn.over_sampling import SMOTE

    seen: list[int] = []
    real = SMOTE.fit_resample

    def spy(self: Any, x: Any, y: Any) -> Any:
        seen.append(len(y))
        return real(self, x, y)

    monkeypatch.setattr(SMOTE, "fit_resample", spy)
    frame = make_transactions(1_500, 45, seed=7, signal=2.5)
    train_rows = split_dataset(frame, SplitConfig(seed=7)).metadata.train.rows
    run_model_comparison(
        frame,
        dataset_sha256=None,
        config=ExperimentConfig(
            split=SplitConfig(seed=7),
            candidates=("logistic_regression_smote",),
            cv_folds=2,
            n_jobs=1,
        ),
        models_dir=tmp_path / "m",
        reports_dir=tmp_path / "r",
    )
    assert seen, "SMOTE was never applied"
    assert max(seen) <= train_rows  # never more rows than the training partition


def test_all_candidates_skipped_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ghost = CandidateSpec(
        "logistic_regression", "", CANDIDATES["logistic_regression"].factory, requires="no_pkg_q"
    )
    monkeypatch.setitem(experiments_module.CANDIDATES, "logistic_regression", ghost)
    with pytest.raises(ModelError, match="skipped"):
        _run(tmp_path, ("logistic_regression",))


def test_skipped_candidate_recorded_with_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ghost = CandidateSpec("xgboost", "", CANDIDATES["xgboost"].factory, requires="no_pkg_q")
    monkeypatch.setitem(experiments_module.CANDIDATES, "xgboost", ghost)
    result = _run(tmp_path, ("logistic_regression", "xgboost"))
    skipped = next(c for c in result.candidates if c.name == "xgboost")
    assert skipped.status == "skipped"
    assert "no_pkg_q" in skipped.reason
    assert "skipped" in result.comparison_table()
    assert result.selected == "logistic_regression"


@pytest.mark.parametrize("kwargs", [{"candidates": ()}, {"cv_folds": 1}, {"candidates": ("svm",)}])
def test_invalid_experiment_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        ExperimentConfig(**kwargs)


def test_precision_at_recall_levels() -> None:
    labels = np.array([1, 1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    result = precision_at_recall_levels(labels, scores, (0.5, 1.0))
    assert result["0.5"] == 1.0
    assert result["1.0"] == pytest.approx(0.75)


def test_model_complexity_for_linear_and_forest() -> None:
    frame = make_transactions(400, 20, seed=1)
    x, y = frame.drop(columns=["Class"]), frame["Class"]
    linear = build_pipeline(LogisticRegression(max_iter=300)).fit(x, y)
    assert model_complexity(linear)["n_coefficients"] == 30
    forest = CANDIDATES["random_forest"].build(PreprocessingConfig(), 0, 1.0)
    forest.set_params(classifier__n_estimators=5).fit(x, y)
    info = model_complexity(forest)
    assert info["n_trees"] == 5
    assert info["total_tree_nodes"] >= 5


def test_every_candidate_pipeline_starts_with_preprocessor() -> None:
    for spec in CANDIDATES.values():
        if spec.is_available():
            pipeline = spec.build(PreprocessingConfig(), 0, 10.0)
            assert pipeline.steps[0][0] == "preprocess"
            assert pipeline.steps[-1][0] == "classifier"


def test_only_smote_candidate_declares_resampling() -> None:
    flagged = {name for name, spec in candidates_module.CANDIDATES.items() if spec.uses_resampling}
    assert flagged == {"logistic_regression_smote"}


def test_candidate_result_serialises() -> None:
    payload = CandidateResult(name="x", status="skipped", reason="r").to_dict()
    assert payload["status"] == "skipped"
    assert payload["cv_ap_mean"] is None
