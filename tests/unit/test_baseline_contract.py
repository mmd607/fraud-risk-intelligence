"""Test-phase T04: metric edge cases, artifact integrity, selection rules, and leakage guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier

from fraud_risk.cli.main import main
from fraud_risk.data.splitting import SplitConfig
from fraud_risk.evaluation import metrics as metrics_module
from fraud_risk.evaluation.metrics import (
    MetricsInputError,
    ThresholdStrategy,
    compute_classification_metrics,
    select_threshold,
)
from fraud_risk.features.preprocessing import AmountStrategy, PreprocessingConfig
from fraud_risk.training import baseline as baseline_module
from fraud_risk.training.artifacts import (
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    ModelArtifactError,
    ModelManifest,
    load_model_bundle,
    new_model_id,
    read_manifest,
    save_model_bundle,
)
from fraud_risk.training.baseline import (
    BaselineConfig,
    PreprocessingComparison,
    run_baseline_training,
    select_preprocessing,
)
from tests.support.synthetic import make_transactions, write_csv


def _manifest(model_id: str = "dummy-model-1") -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        model_name="dummy",
        created_at="2026-01-01T00:00:00+00:00",
        seed=1,
        target="Class",
        feature_names=["Time"],
        transformed_feature_names=["Time"],
        preprocessing={},
        model_params={},
        threshold={},
        metrics={},
        dataset_sha256=None,
        split={},
        library_versions={},
    )


# --- metrics ---------------------------------------------------------------------------------


def test_perfect_ranking_has_ap_one() -> None:
    metrics = compute_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert metrics.average_precision == 1.0
    assert metrics.roc_auc == 1.0


def test_constant_scores_have_ap_equal_prevalence() -> None:
    labels = np.array([0] * 95 + [1] * 5)
    metrics = compute_classification_metrics(labels, np.full(100, 0.3), threshold=0.5)
    assert metrics.average_precision == pytest.approx(0.05)
    assert metrics.roc_auc == pytest.approx(0.5)


def test_nothing_flagged_gives_zero_precision_and_recall() -> None:
    metrics = compute_classification_metrics([0, 1, 1], [0.1, 0.2, 0.3], threshold=0.99)
    assert metrics.flagged == 0
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
    assert metrics.confusion.as_matrix() == [[1, 0], [2, 0]]


def test_threshold_zero_flags_everything() -> None:
    metrics = compute_classification_metrics([0, 1, 0], [0.0, 0.5, 0.2], threshold=0.0)
    assert metrics.flagged == 3
    assert metrics.recall == 1.0


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_non_finite_threshold_rejected(threshold: float) -> None:
    with pytest.raises(MetricsInputError):
        compute_classification_metrics([0, 1], [0.1, 0.9], threshold=threshold)


def test_only_positives_has_undefined_ranking_metrics() -> None:
    metrics = compute_classification_metrics([1, 1], [0.1, 0.9], threshold=0.5)
    assert metrics.average_precision is None
    assert metrics.recall == 0.5


def test_metrics_dict_has_explicit_label_order() -> None:
    payload = compute_classification_metrics([0, 1], [0.1, 0.9], threshold=0.5).to_dict()
    assert payload["confusion_matrix_labels"] == [0, 1]
    assert payload["confusion_matrix"] == [[1, 0], [0, 1]]
    assert "accuracy" not in payload


def test_boolean_labels_are_accepted() -> None:
    metrics = compute_classification_metrics(np.array([False, True]), [0.1, 0.9], threshold=0.5)
    assert metrics.n_positives == 1


def test_max_f1_tie_prefers_higher_threshold() -> None:
    # second example: thresholds 0.1 and 0.9 both give F1 = 2/3; the higher one must win
    selection = select_threshold([1, 0, 1], [0.9, 0.8, 0.6])
    assert selection.validation_f1 == pytest.approx(0.8)
    labels = [1, 0, 0, 1]
    scores = [0.9, 0.7, 0.6, 0.1]
    assert select_threshold(labels, scores).threshold == pytest.approx(0.9)


def test_min_precision_falls_back_to_best_precision_when_unreachable() -> None:
    selection = select_threshold(
        [0, 1, 0, 1],
        [0.9, 0.8, 0.7, 0.6],
        strategy=ThresholdStrategy.MIN_PRECISION,
        target_precision=1.0,
    )
    assert selection.validation_precision < 1.0
    assert selection.target_precision == 1.0


def test_min_precision_rejects_invalid_target() -> None:
    with pytest.raises(MetricsInputError):
        select_threshold(
            [0, 1], [0.1, 0.9], strategy=ThresholdStrategy.MIN_PRECISION, target_precision=0
        )


# --- artifacts -------------------------------------------------------------------------------


def test_bundle_round_trip(tmp_path: Path) -> None:
    model = DummyClassifier().fit([[0], [1]], [0, 1])
    directory = save_model_bundle(model, _manifest(), tmp_path)
    bundle = load_model_bundle(directory)
    assert bundle.manifest.model_sha256 is not None
    assert bundle.model.predict([[0]]).shape == (1,)


def test_bundle_refuses_overwrite(tmp_path: Path) -> None:
    save_model_bundle(DummyClassifier(), _manifest(), tmp_path)
    with pytest.raises(ModelArtifactError, match="refusing to overwrite"):
        save_model_bundle(DummyClassifier(), _manifest(), tmp_path)


@pytest.mark.parametrize("bad_id", ["../escape", "UPPER", "a", "has space"])
def test_bundle_rejects_unsafe_model_ids(tmp_path: Path, bad_id: str) -> None:
    with pytest.raises(ModelArtifactError):
        save_model_bundle(DummyClassifier(), _manifest(bad_id), tmp_path)


def test_missing_manifest_and_model_file(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="manifest"):
        read_manifest(tmp_path)
    directory = save_model_bundle(DummyClassifier(), _manifest(), tmp_path)
    (directory / MODEL_FILENAME).unlink()
    with pytest.raises(ModelArtifactError, match="missing"):
        load_model_bundle(directory)


def test_invalid_manifest_is_rejected_without_echoing(tmp_path: Path) -> None:
    directory = save_model_bundle(DummyClassifier(), _manifest(), tmp_path)
    payload = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload["unexpected"] = "secret-ish"
    (directory / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelArtifactError) as excinfo:
        read_manifest(directory)
    assert "secret-ish" not in str(excinfo.value)


def test_manifest_without_hash_is_not_loaded(tmp_path: Path) -> None:
    directory = save_model_bundle(DummyClassifier(), _manifest(), tmp_path)
    payload = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload["model_sha256"] = None
    (directory / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="Integrity"):
        load_model_bundle(directory)


def test_manifest_requires_feature_names() -> None:
    with pytest.raises(ValidationError):
        ModelManifest(**{**_manifest().model_dump(), "feature_names": []})


def test_new_model_id_format() -> None:
    from datetime import UTC, datetime

    assert new_model_id("LR", now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == (
        "lr-20260102t030405"
    )


# --- selection rule ----------------------------------------------------------------------------


def _comparison(mean: float, std: float, config: PreprocessingConfig) -> PreprocessingComparison:
    return PreprocessingComparison(config, mean, std, (mean,) * 4)


def test_one_standard_error_rule_keeps_default_within_noise() -> None:
    default = _comparison(0.70, 0.05, PreprocessingConfig())
    challenger = _comparison(0.71, 0.05, PreprocessingConfig(amount=AmountStrategy.STANDARD))
    assert select_preprocessing((default, challenger)) == PreprocessingConfig()


def test_one_standard_error_rule_switches_on_clear_win() -> None:
    default = _comparison(0.60, 0.01, PreprocessingConfig())
    better = PreprocessingConfig(amount=AmountStrategy.STANDARD)
    assert select_preprocessing((default, _comparison(0.75, 0.01, better))) == better


def test_selection_requires_candidates() -> None:
    with pytest.raises(ValueError, match="candidate"):
        select_preprocessing(())


# --- leakage guards ----------------------------------------------------------------------------


def test_threshold_and_cv_never_see_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = make_transactions(2_000, 60, seed=8)
    seen: dict[str, Any] = {"threshold_sizes": [], "cv_sizes": []}
    real_select = metrics_module.select_threshold
    real_compare = baseline_module.compare_preprocessing

    def spy_select(y: Any, scores: Any, **kwargs: Any) -> Any:
        seen["threshold_sizes"].append(len(y))
        return real_select(y, scores, **kwargs)

    def spy_compare(x_train: Any, y_train: Any, *args: Any, **kwargs: Any) -> Any:
        seen["cv_sizes"].append(len(x_train))
        return real_compare(x_train, y_train, *args, **kwargs)

    monkeypatch.setattr(baseline_module, "select_threshold", spy_select)
    monkeypatch.setattr(baseline_module, "compare_preprocessing", spy_compare)
    result = run_baseline_training(
        frame,
        dataset_sha256=None,
        config=BaselineConfig(split=SplitConfig(seed=2), cv_folds=2),
        models_dir=tmp_path / "m",
        reports_dir=tmp_path / "r",
    )
    validation_rows = result.split["validation"]["rows"]  # type: ignore[index]
    train_rows = result.split["train"]["rows"]  # type: ignore[index]
    assert seen["threshold_sizes"] == [validation_rows, validation_rows]
    assert seen["cv_sizes"] == [train_rows]


def test_report_contains_no_feature_values(tmp_path: Path) -> None:
    frame = make_transactions(1_500, 45, seed=9)
    result = run_baseline_training(
        frame,
        dataset_sha256=None,
        config=BaselineConfig(split=SplitConfig(seed=3), cv_folds=2),
        models_dir=tmp_path / "m",
        reports_dir=tmp_path / "r",
    )
    text = result.report_path.read_text(encoding="utf-8")
    assert f"{frame.loc[10, 'V5']!r}" not in text
    scores_header = (result.bundle_dir / "scores_test.csv").read_text("utf-8").splitlines()[0]
    assert scores_header == "row_id,label,score"


# --- CLI -------------------------------------------------------------------------------------


def test_cli_train_baseline_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = write_csv(make_transactions(2_000, 60, seed=10), tmp_path / "d.csv")
    monkeypatch.setenv("FRAUD_RISK_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FRAUD_RISK_REPORTS_DIR", str(tmp_path / "reports"))
    assert main(["train-baseline", "--path", str(csv), "--cv-folds", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["models"]) == {"dummy_prior", "logistic_regression"}
    assert (tmp_path / "reports" / "baseline_metrics.json").is_file()
    assert any((tmp_path / "artifacts" / "models").iterdir())


def test_cli_train_baseline_fails_clearly_without_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["train-baseline", "--path", str(tmp_path / "missing.csv")]) == 1
    assert "DATA_NOT_FOUND" in capsys.readouterr().err


def test_cli_train_baseline_refuses_invalid_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_transactions(500, 20)
    frame.loc[0, "Amount"] = -3.0
    csv = write_csv(frame, tmp_path / "bad.csv")
    assert main(["train-baseline", "--path", str(csv)]) == 1
    assert "DATA_VALIDATION_FAILED" in capsys.readouterr().err
