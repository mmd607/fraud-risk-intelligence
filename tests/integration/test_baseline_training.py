from __future__ import annotations

import json

import pytest

from fraud_risk.data.splitting import SplitConfig
from fraud_risk.training.artifacts import ModelArtifactError, load_model_bundle
from fraud_risk.training.baseline import BaselineConfig, run_baseline_training
from tests.support.synthetic import make_transactions

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> object:
    root = tmp_path_factory.mktemp("baseline")
    frame = make_transactions(3_000, 90, seed=4, signal=2.5)
    return run_baseline_training(
        frame,
        dataset_sha256="synthetic",
        config=BaselineConfig(split=SplitConfig(seed=1), cv_folds=3),
        models_dir=root / "models",
        reports_dir=root / "reports",
    )


def test_learned_model_beats_dummy_on_synthetic_signal(result: object) -> None:
    # Synthetic data: this asserts behaviour (ranking works), not real-world performance.
    lr = result.evaluation("logistic_regression")  # type: ignore[attr-defined]
    dummy = result.evaluation("dummy_prior")  # type: ignore[attr-defined]
    assert lr.test["average_precision"] > dummy.test["average_precision"]
    assert dummy.test["roc_auc"] == pytest.approx(0.5)


def test_report_and_bundle_written(result: object) -> None:
    report = json.loads(result.report_path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    assert {m["name"] for m in report["models"]} == {"dummy_prior", "logistic_regression"}
    assert report["dataset_sha256"] == "synthetic"
    assert "average_precision" in report["metric_definitions"]
    bundle = load_model_bundle(result.bundle_dir)  # type: ignore[attr-defined]
    assert "Class" not in bundle.manifest.feature_names
    assert bundle.manifest.threshold["strategy"] == "max_f1"
    assert (result.bundle_dir / "scores_test.csv").is_file()  # type: ignore[attr-defined]


def test_tampered_bundle_is_refused(result: object) -> None:
    model_file = result.bundle_dir / "model.joblib"  # type: ignore[attr-defined]
    original = model_file.read_bytes()
    try:
        model_file.write_bytes(original + b"tamper")
        with pytest.raises(ModelArtifactError):
            load_model_bundle(result.bundle_dir)  # type: ignore[attr-defined]
    finally:
        model_file.write_bytes(original)
