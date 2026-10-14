"""Test-phase T08: alignment, model coverage, failure handling, report integrity, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from fraud_risk.cli.main import main
from fraud_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_risk.data.splitting import SplitConfig
from fraud_risk.explainability import explainers
from fraud_risk.explainability.explainers import (
    ExplainabilityError,
    FeatureContribution,
    LocalExplanation,
    LocalMethod,
    ModelInfo,
    explain_local,
    global_permutation_importance,
)
from fraud_risk.explainability.report import build_explanation_report, split_config_from_manifest
from fraud_risk.models.baseline import build_pipeline
from fraud_risk.training.artifacts import load_model_bundle
from fraud_risk.training.baseline import BaselineConfig, run_baseline_training
from tests.support.synthetic import make_transactions, write_csv

INFO = ModelInfo("m-2", "contract", "abc")
needs_shap = pytest.mark.skipif(not explainers.shap_available(), reason="shap not installed")


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    frame = make_transactions(1_000, 50, seed=17, signal=3.0)
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("explain")
    frame = make_transactions(2_000, 60, seed=19, signal=3.0)
    result = run_baseline_training(
        frame,
        dataset_sha256="synthetic",
        config=BaselineConfig(split=SplitConfig(seed=5), cv_folds=2),
        models_dir=root / "models",
        reports_dir=root / "reports",
    )
    return {"frame": frame, "bundle_dir": result.bundle_dir, "root": root}


@needs_shap
def test_shap_random_forest_returns_fraud_class(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)).fit(*data)
    explanation = explain_local(model, data[0].iloc[[2]], model_info=INFO)
    assert explanation.method is LocalMethod.SHAP_TREE
    assert len(explanation.contributions) == len(FEATURE_COLUMNS)
    total = explanation.base_value + sum(c.contribution for c in explanation.contributions)  # type: ignore[operator]
    assert total == pytest.approx(float(model.predict_proba(data[0].iloc[[2]])[0, 1]), abs=1e-6)
    assert explanation.output_space == "probability"  # RF SHAP is in probability space


@needs_shap
def test_shap_boosting_output_space_is_log_odds(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = build_pipeline(HistGradientBoostingClassifier(max_iter=20)).fit(*data)
    explanation = explain_local(model, data[0].iloc[[2]], model_info=INFO)
    assert explanation.output_space == "log_odds"


@needs_shap
def test_shap_xgboost_is_supported(data: tuple[pd.DataFrame, pd.Series]) -> None:
    xgboost = pytest.importorskip("xgboost")
    model = build_pipeline(xgboost.XGBClassifier(n_estimators=20, random_state=0)).fit(*data)
    assert explainers.resolve_method(model) is LocalMethod.SHAP_TREE
    explanation = explain_local(model, data[0].iloc[[1]], model_info=INFO)
    assert explanation.additive


def test_forced_occlusion_on_tree_model(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = build_pipeline(HistGradientBoostingClassifier(max_iter=20)).fit(*data)
    explanation = explain_local(
        model, data[0].iloc[[0]], model_info=INFO, method=LocalMethod.OCCLUSION, background=data[0]
    )
    assert explanation.output_space == "probability"
    assert explanation.base_value is None
    assert explanation.notes


def test_explanations_carry_model_metadata(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = build_pipeline(HistGradientBoostingClassifier(max_iter=20)).fit(*data)
    explanation = explain_local(model, data[0].iloc[[0]], model_info=INFO, background=data[0])
    assert explanation.model == INFO
    assert explanation.to_dict()["model"]["model_sha256"] == "abc"


def test_top_orders_by_absolute_contribution() -> None:
    explanation = LocalExplanation(
        model=INFO,
        method=LocalMethod.LINEAR,
        output_space="log_odds",
        score=0.5,
        base_value=0.0,
        additive=True,
        contributions=(
            FeatureContribution("a", 0.1),
            FeatureContribution("b", -2.0),
            FeatureContribution("c", 1.0),
        ),
    )
    assert [c.feature for c in explanation.top(2)] == ["b", "c"]


def test_global_importance_requires_positives(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = build_pipeline(HistGradientBoostingClassifier(max_iter=10)).fit(*data)
    with pytest.raises(ExplainabilityError, match="positive"):
        global_permutation_importance(
            model, data[0], pd.Series([0] * len(data[0])), model_info=INFO
        )


def test_non_dataframe_input_rejected(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = build_pipeline(HistGradientBoostingClassifier(max_iter=10)).fit(*data)
    with pytest.raises(ExplainabilityError, match="DataFrame"):
        explain_local(model, data[0].iloc[[0]].to_numpy(), model_info=INFO)  # type: ignore[arg-type]


def test_report_matches_manifest_and_contains_no_values(trained: dict[str, Any]) -> None:
    bundle = load_model_bundle(trained["bundle_dir"])
    report = build_explanation_report(
        bundle, trained["frame"], reports_dir=trained["root"] / "out", n_local=2, n_repeats=1
    )
    payload = json.loads(report.path.read_text(encoding="utf-8"))
    assert payload["partition"] == "validation"
    assert payload["local_method"] == "linear"
    assert len(payload["local_examples"]) == 2
    first = payload["local_examples"][0]
    names = {c["feature"] for c in first["contributions"]}
    assert names <= set(bundle.manifest.transformed_feature_names)
    row = trained["frame"].loc[first["row_id"]]
    assert f"{row['V3']!r}" not in report.path.read_text(encoding="utf-8")


def test_report_refuses_different_data(trained: dict[str, Any]) -> None:
    bundle = load_model_bundle(trained["bundle_dir"])
    other = make_transactions(2_000, 60, seed=99, signal=3.0)
    with pytest.raises(ExplainabilityError, match="does not match"):
        build_explanation_report(bundle, other, reports_dir=trained["root"] / "x", n_repeats=1)


def test_split_config_from_incomplete_manifest() -> None:
    with pytest.raises(ExplainabilityError):
        split_config_from_manifest({"seed": 1})


def test_cli_explain(
    trained: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = write_csv(trained["frame"], trained["root"] / "data.csv")
    monkeypatch.setenv("FRAUD_RISK_REPORTS_DIR", str(trained["root"] / "cli_reports"))
    # the bundle manifest records dataset_sha256="synthetic" -> CLI must refuse other data
    code = main(["explain", "--bundle", str(trained["bundle_dir"]), "--path", str(csv)])
    assert code == 1
    assert "differs" in capsys.readouterr().err


def test_cli_explain_missing_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = write_csv(make_transactions(300, 10), tmp_path / "d.csv")
    assert main(["explain", "--bundle", str(tmp_path / "nope"), "--path", str(csv)]) == 1
    assert "MODEL_ARTIFACT_INVALID" in capsys.readouterr().err
