"""Reproducible baseline training workflow: DummyClassifier vs logistic regression.

Protocol (see ``docs/modeling-baseline.md``):

1. Split once (stratified, duplicate-grouped, seeded) into train / validation / test.
2. Compare preprocessing options with stratified K-fold cross-validation **on the training
   partition only** (AP scoring); pick the best mean AP.
3. Fit the dummy and the logistic regression on the training partition.
4. Choose each model's decision threshold on the **validation** partition.
5. Evaluate both models once on the **test** partition with thresholds frozen.
6. Persist the logistic-regression bundle and a JSON report to Git-ignored local paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from fraud_risk.core.logging import get_logger, log_event
from fraud_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_risk.data.splitting import SplitConfig, split_dataset, split_features_target
from fraud_risk.evaluation.metrics import (
    ThresholdSelection,
    ThresholdStrategy,
    compute_classification_metrics,
    precision_recall_at_k,
    select_threshold,
)
from fraud_risk.features.preprocessing import (
    AmountStrategy,
    PreprocessingConfig,
    TimeStrategy,
    transformed_feature_names,
)
from fraud_risk.models.baseline import (
    PREPROCESS_STEP,
    LogisticRegressionConfig,
    build_dummy_model,
    build_logistic_regression,
)
from fraud_risk.training.artifacts import (
    ModelManifest,
    library_versions,
    new_model_id,
    save_model_bundle,
    utc_now_iso,
)

_LOGGER = get_logger(__name__)

DEFAULT_PREPROCESSING_CANDIDATES: tuple[PreprocessingConfig, ...] = (
    PreprocessingConfig(),
    PreprocessingConfig(amount=AmountStrategy.STANDARD),
    PreprocessingConfig(time=TimeStrategy.DROP),
    PreprocessingConfig(time=TimeStrategy.HOUR_CYCLICAL),
)


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Configuration of the baseline experiment."""

    split: SplitConfig = field(default_factory=SplitConfig)
    logistic_regression: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)
    preprocessing_candidates: tuple[PreprocessingConfig, ...] = DEFAULT_PREPROCESSING_CANDIDATES
    cv_folds: int = 5
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.MAX_F1
    target_precision: float = 0.9
    review_capacities: tuple[int, ...] = (50, 100, 500)


@dataclass(frozen=True, slots=True)
class PreprocessingComparison:
    """Cross-validated AP of one preprocessing option on training folds."""

    config: PreprocessingConfig
    cv_ap_mean: float
    cv_ap_std: float
    fold_scores: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return {
            "preprocessing": self.config.to_dict(),
            "cv_ap_mean": self.cv_ap_mean,
            "cv_ap_std": self.cv_ap_std,
            "fold_scores": list(self.fold_scores),
        }


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Validation-chosen threshold plus validation and test metrics for one model."""

    name: str
    threshold: ThresholdSelection
    validation: dict[str, object]
    test: dict[str, object]
    test_top_k: list[dict[str, object]]
    fit_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return {
            "name": self.name,
            "threshold": self.threshold.to_dict(),
            "validation": self.validation,
            "test": self.test,
            "test_top_k": self.test_top_k,
            "fit_seconds": self.fit_seconds,
        }


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Everything produced by :func:`run_baseline_training`."""

    model_id: str
    bundle_dir: Path
    report_path: Path
    preprocessing_comparison: tuple[PreprocessingComparison, ...]
    selected_preprocessing: PreprocessingConfig
    evaluations: tuple[ModelEvaluation, ...]
    split: dict[str, object]

    def evaluation(self, name: str) -> ModelEvaluation:
        """Look up a model evaluation by name."""
        return next(e for e in self.evaluations if e.name == name)

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation (report body)."""
        return {
            "model_id": self.model_id,
            "selected_preprocessing": self.selected_preprocessing.to_dict(),
            "preprocessing_comparison": [c.to_dict() for c in self.preprocessing_comparison],
            "split": self.split,
            "models": [e.to_dict() for e in self.evaluations],
            "metric_definitions": {
                "average_precision": "sklearn.metrics.average_precision_score: "
                "sum_n (R_n - R_{n-1}) P_n over descending score thresholds (no interpolation)",
                "thresholded_metrics": "positive when score >= threshold; threshold chosen on the "
                "validation partition only",
                "confusion_matrix": "labels [0, 1] -> [[TN, FP], [FN, TP]]",
            },
        }


def compare_preprocessing(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: tuple[PreprocessingConfig, ...],
    *,
    lr_config: LogisticRegressionConfig,
    seed: int,
    cv_folds: int,
) -> tuple[PreprocessingComparison, ...]:
    """Cross-validate a logistic regression per preprocessing option on training data only."""
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = []
    for candidate in candidates:
        model = build_logistic_regression(lr_config, candidate, seed=seed)
        scores = cross_val_score(
            model, x_train, y_train, cv=folds, scoring="average_precision", n_jobs=-1
        )
        values = tuple(float(s) for s in scores)
        results.append(
            PreprocessingComparison(
                candidate, float(np.mean(values)), float(np.std(values)), values
            )
        )
    return tuple(results)


def select_preprocessing(comparison: tuple[PreprocessingComparison, ...]) -> PreprocessingConfig:
    """One-standard-error rule: keep the first (documented default) candidate unless another one
    beats its mean CV AP by more than the standard error of that challenger's fold scores.

    This avoids "selecting" on differences that are within cross-validation noise.
    """
    if not comparison:
        msg = "At least one preprocessing candidate is required."
        raise ValueError(msg)
    default = comparison[0]
    best = max(comparison, key=lambda c: c.cv_ap_mean)
    standard_error = best.cv_ap_std / np.sqrt(max(len(best.fold_scores), 1))
    if best.cv_ap_mean - default.cv_ap_mean > standard_error:
        return best.config
    return default.config


def positive_scores(model: Pipeline, features: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Probability of the fraud class (column 1)."""
    return np.asarray(model.predict_proba(features)[:, 1], dtype="float64")


def evaluate_model(
    name: str,
    model: Pipeline,
    *,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: BaselineConfig,
    fit_seconds: float,
) -> tuple[ModelEvaluation, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Choose a threshold on validation, then evaluate once on test."""
    validation_scores = positive_scores(model, x_validation)
    selection = select_threshold(
        y_validation,
        validation_scores,
        strategy=config.threshold_strategy,
        target_precision=config.target_precision,
    )
    test_scores = positive_scores(model, x_test)
    evaluation = ModelEvaluation(
        name=name,
        threshold=selection,
        validation=compute_classification_metrics(
            y_validation, validation_scores, threshold=selection.threshold
        ).to_dict(),
        test=compute_classification_metrics(
            y_test, test_scores, threshold=selection.threshold
        ).to_dict(),
        test_top_k=[
            precision_recall_at_k(y_test, test_scores, k).to_dict()
            for k in config.review_capacities
        ],
        fit_seconds=fit_seconds,
    )
    return evaluation, validation_scores, test_scores


def write_scores(
    path: Path, index: pd.Index, labels: pd.Series, scores: npt.NDArray[np.float64]
) -> Path:
    """Persist row id, label, and score (no features) for later dashboards/analysis."""
    frame = pd.DataFrame({"row_id": index, "label": labels.to_numpy(), "score": scores})
    frame.to_csv(path, index=False)
    return path


def run_baseline_training(
    frame: pd.DataFrame,
    *,
    dataset_sha256: str | None,
    config: BaselineConfig,
    models_dir: Path,
    reports_dir: Path,
) -> BaselineResult:
    """Run the full baseline protocol and persist artifacts (see module docstring)."""
    seed = config.split.seed
    splits = split_dataset(frame, config.split)
    x_train, y_train = split_features_target(splits.train)
    x_validation, y_validation = split_features_target(splits.validation)
    x_test, y_test = split_features_target(splits.test)

    comparison = compare_preprocessing(
        x_train,
        y_train,
        config.preprocessing_candidates,
        lr_config=config.logistic_regression,
        seed=seed,
        cv_folds=config.cv_folds,
    )
    selected = select_preprocessing(comparison)
    log_event(
        _LOGGER,
        "preprocessing_selected",
        amount=selected.amount.value,
        time=selected.time.value,
        pca=selected.pca.value,
    )

    evaluations = []
    scores: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in (
        ("dummy_prior", build_dummy_model(selected)),
        (
            "logistic_regression",
            build_logistic_regression(config.logistic_regression, selected, seed=seed),
        ),
    ):
        start = perf_counter()
        model.fit(x_train, y_train)
        elapsed = perf_counter() - start
        evaluation, val_scores, test_scores = evaluate_model(
            name,
            model,
            x_validation=x_validation,
            y_validation=y_validation,
            x_test=x_test,
            y_test=y_test,
            config=config,
            fit_seconds=elapsed,
        )
        evaluations.append(evaluation)
        scores[name] = (val_scores, test_scores)
        fitted[name] = model
        test_ap = evaluation.test["average_precision"]
        log_event(
            _LOGGER,
            "model_evaluated",
            model=name,
            test_average_precision=test_ap if isinstance(test_ap, float) else None,
            threshold=evaluation.threshold.threshold,
        )

    best = fitted["logistic_regression"]
    lr_eval = evaluations[-1]
    model_id = new_model_id("logistic_regression")
    manifest = ModelManifest(
        model_id=model_id,
        model_name="logistic_regression",
        created_at=utc_now_iso(),
        seed=seed,
        target=TARGET_COLUMN,
        feature_names=list(FEATURE_COLUMNS),
        transformed_feature_names=list(
            transformed_feature_names(best.named_steps[PREPROCESS_STEP])
        ),
        preprocessing=selected.to_dict(),
        model_params=config.logistic_regression.to_dict(),
        threshold=lr_eval.threshold.to_dict(),
        metrics={"validation": lr_eval.validation, "test": lr_eval.test},
        dataset_sha256=dataset_sha256,
        split=splits.metadata.to_dict(),
        library_versions=library_versions(),
        notes="Baseline. Threshold chosen on validation; test used once for final evaluation.",
    )
    bundle_dir = save_model_bundle(best, manifest, models_dir)
    for partition, index, labels, values in (
        ("validation", splits.validation.index, y_validation, scores["logistic_regression"][0]),
        ("test", splits.test.index, y_test, scores["logistic_regression"][1]),
    ):
        write_scores(bundle_dir / f"scores_{partition}.csv", index, labels, values)

    result = BaselineResult(
        model_id=model_id,
        bundle_dir=bundle_dir,
        report_path=reports_dir / "baseline_metrics.json",
        preprocessing_comparison=comparison,
        selected_preprocessing=selected,
        evaluations=tuple(evaluations),
        split=splits.metadata.to_dict(),
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_sha256": dataset_sha256, **result.to_dict()}
    result.report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result
