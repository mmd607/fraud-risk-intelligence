from __future__ import annotations

import json

import pytest

from fraud_risk.core.exceptions import ConfigurationError, DependencyUnavailableError
from fraud_risk.data.splitting import SplitConfig
from fraud_risk.features.preprocessing import PreprocessingConfig
from fraud_risk.models.candidates import CANDIDATES, CandidateSpec, resolve_grid
from fraud_risk.training.artifacts import load_model_bundle
from fraud_risk.training.experiments import (
    ComparisonResult,
    ExperimentConfig,
    run_model_comparison,
)
from tests.support.synthetic import make_transactions

pytestmark = pytest.mark.integration

FAST = ("logistic_regression", "hist_gradient_boosting")


@pytest.fixture(scope="module")
def comparison(tmp_path_factory: pytest.TempPathFactory) -> ComparisonResult:
    root = tmp_path_factory.mktemp("compare")
    return run_model_comparison(
        make_transactions(2_500, 75, seed=12, signal=2.5),
        dataset_sha256="synthetic",
        config=ExperimentConfig(split=SplitConfig(seed=4), candidates=FAST, cv_folds=2, n_jobs=1),
        models_dir=root / "models",
        reports_dir=root / "reports",
    )


def test_selection_follows_cv_objective(comparison: ComparisonResult) -> None:
    ok = [c for c in comparison.candidates if c.status == "ok"]
    assert {c.name for c in ok} == set(FAST)
    best = max(ok, key=lambda c: c.cv_ap_mean or 0.0)
    assert comparison.selected == best.name


def test_candidates_record_runtime_and_complexity(comparison: ComparisonResult) -> None:
    for candidate in comparison.candidates:
        assert candidate.search_seconds is not None
        assert candidate.complexity["serialized_kb"] > 0
        assert "0.8" in candidate.precision_at_recall
        assert candidate.threshold["strategy"] == "max_f1"


def test_reports_and_bundle(comparison: ComparisonResult) -> None:
    payload = json.loads(comparison.report_path.read_text(encoding="utf-8"))
    assert payload["selected"] == comparison.selected
    assert "selected_test" in payload
    assert all("test" not in c for c in payload["candidates"])
    table = (comparison.report_path.parent / "model_comparison.md").read_text(encoding="utf-8")
    assert "(selected)" in table
    bundle = load_model_bundle(comparison.bundle_dir)
    assert bundle.manifest.model_name == comparison.selected
    assert "test" in bundle.manifest.metrics


def test_unknown_candidate_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ExperimentConfig(candidates=("nope",))


def test_missing_optional_dependency_is_reported() -> None:
    spec = CandidateSpec(
        "ghost", "", CANDIDATES["logistic_regression"].factory, requires="ghost_pkg_x"
    )
    assert not spec.is_available()
    with pytest.raises(DependencyUnavailableError, match="ghost_pkg_x"):
        spec.build(PreprocessingConfig(), 0, 1.0)


def test_ratio_placeholder_resolved() -> None:
    grid = resolve_grid(CANDIDATES["xgboost"], 577.0)
    assert grid["classifier__scale_pos_weight"] == [1.0, 577.0]
