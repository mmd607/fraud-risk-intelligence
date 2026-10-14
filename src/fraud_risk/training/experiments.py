"""Controlled, reproducible comparison of candidate classifiers.

Protocol (see ``docs/model-comparison.md``):

1. The same seeded split as the baseline (train / validation / test).
2. For each available candidate: ``GridSearchCV`` with stratified K-fold on the **training
   partition only**, scored by average precision. Preprocessing (and SMOTE, where used) is inside
   the pipeline, so it is refitted within every training fold.
3. Each tuned candidate is refitted on the full training partition and scored on the
   **validation** partition (AP, threshold chosen there, precision at fixed recall, top-k).
4. **Selection objective:** highest mean CV AP on training folds (more positives, hence less
   noisy than the ~100 validation frauds). Validation AP is reported as a sanity check.
5. Only the **selected** model is evaluated on the **test** partition, once, with its
   validation-chosen threshold. Unselected candidates never see the test data.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from fraud_risk.core.exceptions import ConfigurationError, ModelError
from fraud_risk.core.logging import get_logger, log_event
from fraud_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_risk.data.splitting import SplitConfig, split_dataset, split_features_target
from fraud_risk.evaluation.metrics import (
    ThresholdStrategy,
    compute_classification_metrics,
    precision_recall_at_k,
    select_threshold,
)
from fraud_risk.features.preprocessing import PreprocessingConfig, transformed_feature_names
from fraud_risk.models.baseline import CLASSIFIER_STEP, PREPROCESS_STEP
from fraud_risk.models.candidates import CANDIDATES, DEFAULT_CANDIDATES, resolve_grid
from fraud_risk.training.artifacts import (
    ModelManifest,
    library_versions,
    new_model_id,
    save_model_bundle,
    utc_now_iso,
)
from fraud_risk.training.baseline import positive_scores, write_scores

_LOGGER = get_logger(__name__)
RECALL_LEVELS: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration of the model comparison."""

    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES
    cv_folds: int = 3
    n_jobs: int = -1
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.MAX_F1
    target_precision: float = 0.9
    review_capacities: tuple[int, ...] = (50, 100, 500)

    def __post_init__(self) -> None:
        unknown = [name for name in self.candidates if name not in CANDIDATES]
        if unknown:
            msg = f"Unknown candidates {unknown}; available: {sorted(CANDIDATES)}."
            raise ConfigurationError(msg)
        if not self.candidates:
            msg = "At least one candidate is required."
            raise ConfigurationError(msg)
        if self.cv_folds < 2:  # noqa: PLR2004
            msg = "cv_folds must be >= 2."
            raise ConfigurationError(msg)


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Outcome of tuning and validating one candidate."""

    name: str
    status: str
    reason: str = ""
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_ap_mean: float | None = None
    cv_ap_std: float | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    threshold: dict[str, Any] = field(default_factory=dict)
    precision_at_recall: dict[str, float] = field(default_factory=dict)
    search_seconds: float | None = None
    refit_seconds: float | None = None
    complexity: dict[str, Any] = field(default_factory=dict)
    uses_resampling: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "best_params": self.best_params,
            "cv_ap_mean": self.cv_ap_mean,
            "cv_ap_std": self.cv_ap_std,
            "validation": self.validation,
            "threshold": self.threshold,
            "precision_at_recall": self.precision_at_recall,
            "search_seconds": self.search_seconds,
            "refit_seconds": self.refit_seconds,
            "complexity": self.complexity,
            "uses_resampling": self.uses_resampling,
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Full comparison outcome."""

    candidates: tuple[CandidateResult, ...]
    selected: str
    selection_objective: str
    test: dict[str, Any]
    test_top_k: list[dict[str, Any]]
    model_id: str
    bundle_dir: Path
    report_path: Path
    split: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation."""
        return {
            "selected": self.selected,
            "selection_objective": self.selection_objective,
            "model_id": self.model_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "selected_test": self.test,
            "selected_test_top_k": self.test_top_k,
            "split": self.split,
        }

    def comparison_table(self) -> str:
        """Markdown table of the candidates (validation/CV only; test shown for the winner)."""
        lines = [
            "| Candidate | Status | CV AP (mean ± std) | Val AP | Val P@R=0.8 | Val F1 @thr | "
            "Search s | Refit s | Size KB |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in self.candidates:
            if c.status != "ok":
                lines.append(f"| {c.name} | {c.status}: {c.reason} | | | | | | | |")
                continue
            lines.append(
                f"| {c.name}{' **(selected)**' if c.name == self.selected else ''} | ok | "
                f"{c.cv_ap_mean:.4f} ± {c.cv_ap_std:.4f} | "
                f"{c.validation['average_precision']:.4f} | "
                f"{c.precision_at_recall.get('0.8', float('nan')):.4f} | "
                f"{c.validation['f1']:.4f} | {c.search_seconds:.1f} | {c.refit_seconds:.1f} | "
                f"{c.complexity.get('serialized_kb', float('nan')):.0f} |"
            )
        return "\n".join(lines)


def precision_at_recall_levels(
    labels: npt.ArrayLike, scores: npt.ArrayLike, levels: tuple[float, ...] = RECALL_LEVELS
) -> dict[str, float]:
    """Best precision achievable at recall >= each level (from the PR curve)."""
    precision, recall, _ = precision_recall_curve(np.asarray(labels), np.asarray(scores))
    return {
        str(level): float(precision[recall >= level].max()) if np.any(recall >= level) else 0.0
        for level in levels
    }


def model_complexity(model: Pipeline) -> dict[str, Any]:
    """Size and structural complexity of a fitted pipeline's classifier."""
    classifier = model.named_steps[CLASSIFIER_STEP]
    info: dict[str, Any] = {"serialized_kb": len(pickle.dumps(model)) / 1024}
    if hasattr(classifier, "coef_"):
        info["n_coefficients"] = int(np.asarray(classifier.coef_).size)
    if hasattr(classifier, "estimators_"):
        info["n_trees"] = len(classifier.estimators_)
        info["total_tree_nodes"] = int(sum(t.tree_.node_count for t in classifier.estimators_))
    if hasattr(classifier, "n_iter_") and not hasattr(classifier, "coef_"):
        info["n_boosting_iterations"] = int(np.asarray(classifier.n_iter_).max())
    if hasattr(classifier, "get_booster"):
        info["n_boosted_trees"] = len(classifier.get_booster().get_dump())
    return info


def _tune_candidate(
    name: str,
    config: ExperimentConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    positive_ratio: float,
) -> tuple[GridSearchCV, float]:
    spec = CANDIDATES[name]
    pipeline = spec.build(config.preprocessing, config.split.seed, positive_ratio)
    search = GridSearchCV(
        pipeline,
        resolve_grid(spec, positive_ratio),
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.split.seed),
        refit=False,
        n_jobs=config.n_jobs,
        error_score="raise",
    )
    start = perf_counter()
    search.fit(x_train, y_train)
    return search, perf_counter() - start


def run_model_comparison(
    frame: pd.DataFrame,
    *,
    dataset_sha256: str | None,
    config: ExperimentConfig,
    models_dir: Path,
    reports_dir: Path,
) -> ComparisonResult:
    """Run the comparison protocol and persist the selected model (see module docstring)."""
    seed = config.split.seed
    splits = split_dataset(frame, config.split)
    x_train, y_train = split_features_target(splits.train)
    x_validation, y_validation = split_features_target(splits.validation)
    positives = int(y_train.sum())
    positive_ratio = float((len(y_train) - positives) / positives)

    results: list[CandidateResult] = []
    fitted: dict[str, Pipeline] = {}
    validation_scores: dict[str, npt.NDArray[np.float64]] = {}
    for name in config.candidates:
        spec = CANDIDATES[name]
        if not spec.is_available():
            reason = f"optional dependency '{spec.requires}' not installed"
            results.append(CandidateResult(name=name, status="skipped", reason=reason))
            log_event(_LOGGER, "candidate_skipped", candidate=name, reason=reason)
            continue
        search, search_seconds = _tune_candidate(name, config, x_train, y_train, positive_ratio)
        best_index = int(search.best_index_)
        model = spec.build(config.preprocessing, seed, positive_ratio)
        model.set_params(**search.best_params_)
        start = perf_counter()
        model.fit(x_train, y_train)
        refit_seconds = perf_counter() - start
        scores = positive_scores(model, x_validation)
        selection = select_threshold(
            y_validation,
            scores,
            strategy=config.threshold_strategy,
            target_precision=config.target_precision,
        )
        result = CandidateResult(
            name=name,
            status="ok",
            best_params={
                k.removeprefix(f"{CLASSIFIER_STEP}__"): v for k, v in search.best_params_.items()
            },
            cv_ap_mean=float(search.cv_results_["mean_test_score"][best_index]),
            cv_ap_std=float(search.cv_results_["std_test_score"][best_index]),
            validation=compute_classification_metrics(
                y_validation, scores, threshold=selection.threshold
            ).to_dict(),
            threshold=selection.to_dict(),
            precision_at_recall=precision_at_recall_levels(y_validation, scores),
            search_seconds=search_seconds,
            refit_seconds=refit_seconds,
            complexity=model_complexity(model),
            uses_resampling=spec.uses_resampling,
        )
        results.append(result)
        fitted[name] = model
        validation_scores[name] = scores
        log_event(
            _LOGGER,
            "candidate_evaluated",
            candidate=name,
            cv_ap_mean=result.cv_ap_mean,
            validation_ap=float(result.validation["average_precision"]),
            search_seconds=round(search_seconds, 2),
        )

    completed = [r for r in results if r.status == "ok"]
    if not completed:
        msg = "No candidate could be trained (all skipped)."
        raise ModelError(msg)
    winner = max(completed, key=lambda r: r.cv_ap_mean or 0.0)
    model = fitted[winner.name]

    # Final, single evaluation of the selected model on the held-out test partition.
    x_test, y_test = split_features_target(splits.test)
    test_scores = positive_scores(model, x_test)
    threshold = float(winner.threshold["threshold"])
    test_metrics = compute_classification_metrics(y_test, test_scores, threshold=threshold)
    test_top_k = [
        precision_recall_at_k(y_test, test_scores, k).to_dict() for k in config.review_capacities
    ]

    model_id = new_model_id(winner.name)
    manifest = ModelManifest(
        model_id=model_id,
        model_name=winner.name,
        created_at=utc_now_iso(),
        seed=seed,
        target=TARGET_COLUMN,
        feature_names=list(FEATURE_COLUMNS),
        transformed_feature_names=list(
            transformed_feature_names(model.named_steps[PREPROCESS_STEP])
        ),
        preprocessing=config.preprocessing.to_dict(),
        model_params=winner.best_params,
        threshold=winner.threshold,
        metrics={
            "cv_ap_mean": winner.cv_ap_mean,
            "cv_ap_std": winner.cv_ap_std,
            "validation": winner.validation,
            "test": test_metrics.to_dict(),
        },
        dataset_sha256=dataset_sha256,
        split=splits.metadata.to_dict(),
        library_versions=library_versions(),
        notes="Selected by mean CV AP on training folds; test evaluated once for this model only.",
    )
    bundle_dir = save_model_bundle(model, manifest, models_dir)
    write_scores(
        bundle_dir / "scores_validation.csv",
        splits.validation.index,
        y_validation,
        validation_scores[winner.name],
    )
    write_scores(bundle_dir / "scores_test.csv", splits.test.index, y_test, test_scores)

    comparison = ComparisonResult(
        candidates=tuple(results),
        selected=winner.name,
        selection_objective="max mean CV average precision on training folds",
        test=test_metrics.to_dict(),
        test_top_k=test_top_k,
        model_id=model_id,
        bundle_dir=bundle_dir,
        report_path=reports_dir / "model_comparison.json",
        split=splits.metadata.to_dict(),
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.report_path.write_text(
        json.dumps(
            {"dataset_sha256": dataset_sha256, **comparison.to_dict()}, indent=2, default=str
        )
        + "\n",
        encoding="utf-8",
    )
    (reports_dir / "model_comparison.md").write_text(
        comparison.comparison_table() + "\n", encoding="utf-8"
    )
    return comparison
