from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from fraud_risk.core.exceptions import DependencyUnavailableError
from fraud_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_risk.explainability import explainers
from fraud_risk.explainability.explainers import (
    ExplainabilityError,
    LocalMethod,
    ModelInfo,
    UnsupportedModelError,
    explain_local,
    global_permutation_importance,
    resolve_method,
)
from fraud_risk.features.preprocessing import transformed_feature_names
from fraud_risk.models.baseline import build_pipeline
from tests.support.synthetic import make_transactions

INFO = ModelInfo("m-1", "test-model")


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    frame = make_transactions(1_200, 60, seed=3, signal=3.0)
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


def _fit(classifier: Any, data: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    return build_pipeline(classifier).fit(*data)


def test_linear_contributions_are_exact(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = _fit(LogisticRegression(max_iter=500), data)
    row = data[0].iloc[[7]]
    explanation = explain_local(model, row, model_info=INFO)
    assert explanation.method is LocalMethod.LINEAR
    total = explanation.base_value + sum(c.contribution for c in explanation.contributions)  # type: ignore[operator]
    assert total == pytest.approx(float(model.decision_function(row)[0]))
    names = tuple(c.feature for c in explanation.contributions)
    assert names == transformed_feature_names(model.named_steps["preprocess"])


@pytest.mark.skipif(not explainers.shap_available(), reason="shap not installed")
def test_shap_tree_is_additive_in_log_odds(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = _fit(HistGradientBoostingClassifier(max_iter=30, random_state=0), data)
    row = data[0].iloc[[3]]
    explanation = explain_local(model, row, model_info=INFO)
    assert explanation.method is LocalMethod.SHAP_TREE
    total = explanation.base_value + sum(c.contribution for c in explanation.contributions)  # type: ignore[operator]
    assert total == pytest.approx(float(model.decision_function(row)[0]), abs=1e-6)


def test_occlusion_fallback_is_deterministic(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = _fit(KNeighborsClassifier(n_neighbors=5), data)
    row = data[0].iloc[[0]]
    first = explain_local(model, row, model_info=INFO, background=data[0])
    second = explain_local(model, row, model_info=INFO, background=data[0])
    assert first.method is LocalMethod.OCCLUSION
    assert not first.additive
    assert first == second
    assert tuple(c.feature for c in first.contributions) == FEATURE_COLUMNS


def test_occlusion_requires_background(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = _fit(KNeighborsClassifier(), data)
    with pytest.raises(ExplainabilityError, match="background"):
        explain_local(model, data[0].iloc[[0]], model_info=INFO)


def test_unsupported_models() -> None:
    with pytest.raises(UnsupportedModelError):
        resolve_method(LogisticRegression())
    with pytest.raises(UnsupportedModelError):
        resolve_method(Pipeline([("clf", LogisticRegression())]))


def test_method_mismatch(data: tuple[pd.DataFrame, pd.Series]) -> None:
    tree = _fit(HistGradientBoostingClassifier(max_iter=10), data)
    with pytest.raises(UnsupportedModelError):
        resolve_method(tree, LocalMethod.LINEAR)
    linear = _fit(LogisticRegression(max_iter=300), data)
    with pytest.raises(UnsupportedModelError):
        resolve_method(linear, LocalMethod.SHAP_TREE)


def test_missing_shap_is_actionable(
    data: tuple[pd.DataFrame, pd.Series], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(explainers, "shap_available", lambda: False)
    tree = _fit(HistGradientBoostingClassifier(max_iter=10), data)
    with pytest.raises(DependencyUnavailableError, match=r"fraud-risk\[explain\]"):
        resolve_method(tree, LocalMethod.SHAP_TREE)
    assert resolve_method(tree) is LocalMethod.OCCLUSION


def test_global_importance(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = _fit(LogisticRegression(max_iter=500), data)
    result = global_permutation_importance(model, data[0], data[1], model_info=INFO, n_repeats=2)
    assert {i.feature for i in result.importances} == set(FEATURE_COLUMNS)
    assert result.importances[0].importance_mean >= result.importances[-1].importance_mean
    assert result.model.model_id == "m-1"
    assert "not causes" in result.caveat


def test_input_validation(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = _fit(LogisticRegression(max_iter=300), data)
    with pytest.raises(ExplainabilityError, match="exactly one row"):
        explain_local(model, data[0].iloc[:2], model_info=INFO)
    with pytest.raises(ExplainabilityError, match="finite"):
        explain_local(model, data[0].iloc[[0]].assign(V1=np.nan), model_info=INFO)
    with pytest.raises(ExplainabilityError, match="Missing"):
        explain_local(model, data[0].iloc[[0]].drop(columns=["V2"]), model_info=INFO)


def test_serialisation_has_no_feature_values(data: tuple[pd.DataFrame, pd.Series]) -> None:
    model = _fit(LogisticRegression(max_iter=300), data)
    row = data[0].iloc[[5]]
    payload = explain_local(model, row, model_info=INFO).to_dict(top_k=3)
    assert len(payload["contributions"]) == 3
    assert f"{row['V1'].iloc[0]!r}" not in str(payload)
