"""Anomaly-detection training workflow (separate track from supervised classification).

Setup:

* Same seeded split as the supervised track.
* ``fit_on="all"`` (default, fully unsupervised): fit on **all training-partition features**,
  no labels used. The ~0.17 % frauds in training are treated as unlabelled points.
* ``fit_on="legitimate"`` (semi-supervised novelty detection): fit only on training rows labelled
  legitimate. This **uses training labels** to filter rows and is reported as label-informed.
* Labels of the validation and test partitions are used only to evaluate rankings.
* Hyperparameters are fixed a priori (not tuned on labels). Review budgets are fixed in config.
* The anomaly score is persisted separately and never combined with supervised scores here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter

import pandas as pd

from fraud_risk.anomaly.detectors import IsolationForestConfig, IsolationForestDetector
from fraud_risk.anomaly.evaluation import AnomalyRankingReport, evaluate_anomaly_ranking
from fraud_risk.core.logging import get_logger, log_event
from fraud_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_risk.data.splitting import SplitConfig, split_dataset, split_features_target
from fraud_risk.features.preprocessing import transformed_feature_names
from fraud_risk.training.artifacts import (
    ModelManifest,
    library_versions,
    new_model_id,
    save_model_bundle,
    utc_now_iso,
)
from fraud_risk.training.baseline import write_scores

_LOGGER = get_logger(__name__)


class FitOn(StrEnum):
    """Which training rows the detector is fitted on."""

    ALL = "all"
    LEGITIMATE = "legitimate"


@dataclass(frozen=True, slots=True)
class AnomalyConfig:
    """Anomaly-track configuration."""

    split: SplitConfig = field(default_factory=SplitConfig)
    detector: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    fit_on: FitOn = FitOn.ALL
    review_budgets: tuple[int, ...] = (50, 100, 500, 1000)


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    """Outcome of :func:`run_anomaly_training`."""

    model_id: str
    bundle_dir: Path
    report_path: Path
    fit_on: FitOn
    training_rows: int
    fit_seconds: float
    validation: AnomalyRankingReport
    test: AnomalyRankingReport

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return {
            "model_id": self.model_id,
            "detector": "isolation_forest",
            "fit_on": self.fit_on.value,
            "uses_training_labels": self.fit_on is FitOn.LEGITIMATE,
            "training_rows": self.training_rows,
            "fit_seconds": self.fit_seconds,
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
        }


def run_anomaly_training(
    frame: pd.DataFrame,
    *,
    dataset_sha256: str | None,
    config: AnomalyConfig,
    models_dir: Path,
    reports_dir: Path,
) -> AnomalyResult:
    """Fit the detector on training features and evaluate rankings on held-out partitions."""
    splits = split_dataset(frame, config.split)
    train = splits.train
    if config.fit_on is FitOn.LEGITIMATE:
        train = train.loc[train[TARGET_COLUMN] == 0]
    x_fit = train.loc[:, list(FEATURE_COLUMNS)]  # labels dropped before the detector sees data

    detector = IsolationForestDetector(config.detector)
    start = perf_counter()
    detector.fit(x_fit)
    fit_seconds = perf_counter() - start

    x_validation, y_validation = split_features_target(splits.validation)
    x_test, y_test = split_features_target(splits.test)
    validation_scores = detector.score(x_validation)
    test_scores = detector.score(x_test)
    validation_report = evaluate_anomaly_ranking(
        y_validation, validation_scores, config.review_budgets
    )
    test_report = evaluate_anomaly_ranking(y_test, test_scores, config.review_budgets)

    model_id = new_model_id(f"isolation_forest_{config.fit_on.value}")
    manifest = ModelManifest(
        model_id=model_id,
        model_name="isolation_forest",
        created_at=utc_now_iso(),
        seed=config.detector.seed,
        target=TARGET_COLUMN,
        feature_names=list(FEATURE_COLUMNS),
        transformed_feature_names=list(
            transformed_feature_names(detector.pipeline.named_steps["preprocess"])
        ),
        preprocessing=config.detector.preprocessing.to_dict(),
        model_params={**config.detector.to_dict(), "fit_on": config.fit_on.value},
        threshold={"strategy": "none", "note": "anomaly scores are rankings; no threshold"},
        metrics={"validation": validation_report.to_dict(), "test": test_report.to_dict()},
        dataset_sha256=dataset_sha256,
        split=splits.metadata.to_dict(),
        library_versions=library_versions(),
        notes="Anomaly score: higher = more unusual. Not a probability; not combined with the "
        "supervised model.",
    )
    bundle_dir = save_model_bundle(detector.pipeline, manifest, models_dir)
    write_scores(
        bundle_dir / "scores_validation.csv",
        splits.validation.index,
        y_validation,
        validation_scores,
    )
    write_scores(bundle_dir / "scores_test.csv", splits.test.index, y_test, test_scores)

    result = AnomalyResult(
        model_id=model_id,
        bundle_dir=bundle_dir,
        report_path=reports_dir / f"anomaly_metrics_{config.fit_on.value}.json",
        fit_on=config.fit_on,
        training_rows=len(x_fit),
        fit_seconds=fit_seconds,
        validation=validation_report,
        test=test_report,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    result.report_path.write_text(
        json.dumps({"dataset_sha256": dataset_sha256, **result.to_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    log_event(
        _LOGGER,
        "anomaly_detector_evaluated",
        fit_on=config.fit_on.value,
        validation_ap=validation_report.average_precision,
        test_ap=test_report.average_precision,
    )
    return result
