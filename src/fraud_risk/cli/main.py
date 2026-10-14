"""``fraud-risk`` command-line interface.

Each subcommand is a thin wrapper around package functions. Output is aggregate JSON only; no raw
transaction rows are ever printed. Exit codes: ``0`` success, ``1`` a domain error (message on
stderr), ``2`` invalid command-line usage (argparse).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from fraud_risk.anomaly.detectors import IsolationForestConfig
from fraud_risk.config.settings import Settings, load_settings
from fraud_risk.core.exceptions import ConfigurationError, DataValidationError, FraudRiskError
from fraud_risk.core.logging import configure_logging
from fraud_risk.data.ingestion import load_transactions
from fraud_risk.data.prepare import load_validated_dataset
from fraud_risk.data.quality_report import build_quality_report
from fraud_risk.data.splitting import SplitConfig, SplitStrategy, split_dataset
from fraud_risk.evaluation.metrics import ThresholdStrategy
from fraud_risk.explainability.report import build_explanation_report
from fraud_risk.models.candidates import DEFAULT_CANDIDATES
from fraud_risk.policy.config import load_policy
from fraud_risk.registry.model_registry import (
    LocalModelRegistry,
    ModelStage,
    current_code_revision,
    track_entry,
)
from fraud_risk.registry.tracking import (
    ExperimentTracker,
    MlflowTracker,
    NullTracker,
    default_tracking_uri,
)
from fraud_risk.risk.costs import CostAssumptions, evaluate_policy
from fraud_risk.risk.review_queue import capacity_tradeoff
from fraud_risk.training.anomaly import AnomalyConfig, FitOn, run_anomaly_training
from fraud_risk.training.artifacts import load_model_bundle
from fraud_risk.training.baseline import BaselineConfig, run_baseline_training
from fraud_risk.training.experiments import ExperimentConfig, run_model_comparison

Handler = Callable[[argparse.Namespace, Settings], dict[str, object]]

EXIT_OK = 0
EXIT_DOMAIN_ERROR = 1


def write_json(payload: dict[str, object], output: Path | None) -> None:
    """Print ``payload`` as JSON to stdout and optionally persist it to ``output``."""
    text = json.dumps(payload, indent=2, sort_keys=False, default=str)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    path: Path = args.path or settings.dataset_path
    result = load_transactions(path, allow_extra_columns=args.allow_extra_columns)
    return {"status": "ok", "dataset": result.metadata.to_dict()}


def _cmd_validate(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    path: Path = args.path or settings.dataset_path
    result = load_transactions(path, allow_extra_columns=args.allow_extra_columns)
    report = build_quality_report(result.frame, dataset_fingerprint=result.metadata.sha256)
    output: Path = args.report or settings.reports_dir / "data_quality.json"
    report.write_json(output)
    validation = report.validation
    payload: dict[str, object] = {
        "status": "ok" if validation.is_valid else "invalid",
        "report_path": str(output),
        "validation": validation.to_dict(),
    }
    if not validation.is_valid:
        codes = sorted({issue.code.value for issue in validation.errors})
        raise DataValidationError(f"Data validation failed with error codes {codes}. See {output}.")
    return payload


def _cmd_split(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    path: Path = args.path or settings.dataset_path
    result, _ = load_validated_dataset(path)
    config = SplitConfig(
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=settings.random_seed,
        strategy=SplitStrategy(args.strategy),
        group_duplicates=not args.no_group_duplicates,
    )
    splits = split_dataset(result.frame, config)
    output: Path = args.metadata_output or settings.reports_dir / "split_metadata.json"
    splits.metadata.write_json(output)
    return {
        "status": "ok",
        "metadata_path": str(output),
        "dataset_sha256": result.metadata.sha256,
        "split": splits.metadata.to_dict(),
    }


def _cmd_train_baseline(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    path: Path = args.path or settings.dataset_path
    result, _ = load_validated_dataset(path)
    config = BaselineConfig(
        split=SplitConfig(seed=settings.random_seed),
        cv_folds=args.cv_folds,
        threshold_strategy=ThresholdStrategy(args.threshold_strategy),
        target_precision=args.target_precision,
    )
    outcome = run_baseline_training(
        result.frame,
        dataset_sha256=result.metadata.sha256,
        config=config,
        models_dir=settings.models_dir,
        reports_dir=settings.reports_dir,
    )
    summary = {
        e.name: {
            "threshold": e.threshold.threshold,
            "test_average_precision": e.test["average_precision"],
            "test_roc_auc": e.test["roc_auc"],
            "test_precision": e.test["precision"],
            "test_recall": e.test["recall"],
            "test_f1": e.test["f1"],
        }
        for e in outcome.evaluations
    }
    return {
        "status": "ok",
        "model_id": outcome.model_id,
        "report_path": str(outcome.report_path),
        "selected_preprocessing": outcome.selected_preprocessing.to_dict(),
        "models": summary,
    }


def _cmd_compare_models(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    path: Path = args.path or settings.dataset_path
    result, _ = load_validated_dataset(path)
    config = ExperimentConfig(
        split=SplitConfig(seed=settings.random_seed),
        candidates=tuple(args.candidates),
        cv_folds=args.cv_folds,
    )
    outcome = run_model_comparison(
        result.frame,
        dataset_sha256=result.metadata.sha256,
        config=config,
        models_dir=settings.models_dir,
        reports_dir=settings.reports_dir,
    )
    return {
        "status": "ok",
        "selected": outcome.selected,
        "selection_objective": outcome.selection_objective,
        "model_id": outcome.model_id,
        "report_path": str(outcome.report_path),
        "candidates": [
            {
                "name": c.name,
                "status": c.status,
                "reason": c.reason,
                "cv_ap_mean": c.cv_ap_mean,
                "validation_ap": c.validation.get("average_precision"),
            }
            for c in outcome.candidates
        ],
        "selected_test": {
            key: outcome.test[key]
            for key in ("average_precision", "roc_auc", "precision", "recall", "f1", "threshold")
        },
    }


def _cmd_train_anomaly(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    path: Path = args.path or settings.dataset_path
    result, _ = load_validated_dataset(path)
    config = AnomalyConfig(
        split=SplitConfig(seed=settings.random_seed),
        detector=IsolationForestConfig(seed=settings.random_seed),
        fit_on=FitOn(args.fit_on),
    )
    outcome = run_anomaly_training(
        result.frame,
        dataset_sha256=result.metadata.sha256,
        config=config,
        models_dir=settings.models_dir,
        reports_dir=settings.reports_dir,
    )
    return {"status": "ok", "report_path": str(outcome.report_path), **outcome.to_dict()}


def _read_scores(bundle: Path, partition: str) -> tuple[list[int], list[float]]:
    import pandas as pd  # noqa: PLC0415 - only needed by this command

    path = bundle / f"scores_{partition}.csv"
    if not path.is_file():
        msg = f"No {partition} scores found in bundle '{bundle.name}'."
        raise ConfigurationError(msg)
    frame = pd.read_csv(path)
    return frame["label"].astype(int).tolist(), frame["score"].astype(float).tolist()


def _cmd_evaluate_policy(args: argparse.Namespace, _settings: Settings) -> dict[str, object]:
    policy = load_policy(args.policy)
    labels, scores = _read_scores(args.bundle, args.partition)
    costs = CostAssumptions(
        missed_fraud_cost=args.missed_fraud_cost,
        false_block_cost=args.false_block_cost,
        review_cost=args.review_cost,
    )
    outcome = evaluate_policy(labels, scores, policy, costs)
    capacities = [c for c in (50, 100, 150, 250, 500) if c <= len(scores)]
    return {
        "status": "ok",
        "partition": args.partition,
        "policy": policy.model_dump(mode="json"),
        "outcome": outcome.to_dict(),
        "capacity_tradeoff": [p.to_dict() for p in capacity_tradeoff(labels, scores, capacities)],
    }


def _cmd_explain(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    path: Path = args.path or settings.dataset_path
    result, _ = load_validated_dataset(path)
    bundle = load_model_bundle(args.bundle)
    if bundle.manifest.dataset_sha256 not in (None, result.metadata.sha256):
        msg = "The dataset differs from the one recorded in the bundle manifest."
        raise ConfigurationError(msg)
    report = build_explanation_report(
        bundle, result.frame, reports_dir=settings.reports_dir, n_local=args.n_local
    )
    importances = report.payload["global"]["importances"][: args.top]
    return {
        "status": "ok",
        "report_path": str(report.path),
        "local_method": report.payload["local_method"],
        "top_global_importances": importances,
        "caveat": report.payload["caveat"],
    }


def build_tracker(settings: Settings) -> ExperimentTracker:
    """Tracker from settings: disabled (no-op) unless FRAUD_RISK_TRACKING_ENABLED=true."""
    if not settings.tracking_enabled:
        return NullTracker()
    uri = settings.tracking_uri or default_tracking_uri(settings.artifacts_dir)
    return MlflowTracker(uri)


def _cmd_registry(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    registry = LocalModelRegistry(settings.registry_dir)
    if args.registry_command == "register":
        entry = registry.register(
            args.bundle, actor=args.actor, code_revision=current_code_revision(Path.cwd())
        )
        tracker = build_tracker(settings)
        run_id = track_entry(tracker, entry) if tracker.enabled else None
        return {"status": "ok", "entry": entry.model_dump(mode="json"), "tracking_run": run_id}
    if args.registry_command == "transition":
        entry = registry.transition(
            args.version, ModelStage(args.to), actor=args.actor, reason=args.reason
        )
        return {"status": "ok", "entry": entry.model_dump(mode="json")}
    entries = registry.entries()
    return {
        "status": "ok",
        "approved_version": next(
            (e.version for e in entries if e.stage is ModelStage.APPROVED), None
        ),
        "entries": [
            {
                "version": e.version,
                "model_id": e.model_id,
                "stage": e.stage.value,
                "test_average_precision": e.metrics.get("test", {}).get("average_precision"),
                "code_revision": e.code_revision,
            }
            for e in entries
        ],
    }


def _add_registry_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register model-registry commands: list, register, transition."""
    registry = sub.add_parser("registry", help="Local model registry (explicit lifecycle).")
    registry_sub = registry.add_subparsers(dest="registry_command", required=True)
    registry_sub.add_parser("list", help="List registered model versions.")
    register = registry_sub.add_parser("register", help="Register a bundle as a candidate.")
    register.add_argument("--bundle", type=Path, required=True)
    register.add_argument("--actor", required=True, help="Who is registering (audit trail).")
    transition = registry_sub.add_parser(
        "transition", help="Explicitly change a version's stage (never automatic)."
    )
    transition.add_argument("--version", type=int, required=True)
    transition.add_argument(
        "--to",
        choices=[s.value for s in ModelStage if s is not ModelStage.CANDIDATE],
        required=True,
    )
    transition.add_argument("--actor", required=True)
    transition.add_argument("--reason", required=True)
    registry.set_defaults(handler=_cmd_registry)


def _add_data_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register data commands: ingest, validate, split."""
    ingest = sub.add_parser("ingest", help="Load the local CSV and report dataset metadata.")
    ingest.add_argument("--path", type=Path, default=None, help="CSV path (default: settings).")
    ingest.add_argument("--allow-extra-columns", action="store_true")
    ingest.add_argument("--output", type=Path, default=None, help="Also write JSON here.")
    ingest.set_defaults(handler=_cmd_ingest)

    validate = sub.add_parser(
        "validate", help="Validate the CSV against the data contract and write a JSON report."
    )
    validate.add_argument("--path", type=Path, default=None, help="CSV path (default: settings).")
    validate.add_argument("--allow-extra-columns", action="store_true")
    validate.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report path (default: reports/data_quality.json).",
    )
    validate.set_defaults(handler=_cmd_validate)

    split = sub.add_parser("split", help="Split deterministically and persist split metadata only.")
    split.add_argument("--path", type=Path, default=None, help="CSV path (default: settings).")
    split.add_argument(
        "--strategy", choices=[s.value for s in SplitStrategy], default=SplitStrategy.STRATIFIED
    )
    split.add_argument("--test-size", type=float, default=0.2)
    split.add_argument("--validation-size", type=float, default=0.2)
    split.add_argument("--no-group-duplicates", action="store_true")
    split.add_argument(
        "--output",
        dest="metadata_output",
        type=Path,
        default=None,
        help="Split metadata path (default: reports/split_metadata.json).",
    )
    split.set_defaults(handler=_cmd_split)


def _add_model_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register model commands: train-baseline, compare-models, train-anomaly."""
    baseline = sub.add_parser(
        "train-baseline", help="Train dummy + logistic regression baselines and evaluate them."
    )
    baseline.add_argument("--path", type=Path, default=None, help="CSV path (default: settings).")
    baseline.add_argument("--cv-folds", type=int, default=5)
    baseline.add_argument(
        "--threshold-strategy",
        choices=[s.value for s in ThresholdStrategy if s is not ThresholdStrategy.FIXED],
        default=ThresholdStrategy.MAX_F1.value,
    )
    baseline.add_argument("--target-precision", type=float, default=0.9)
    baseline.set_defaults(handler=_cmd_train_baseline)

    compare = sub.add_parser(
        "compare-models", help="Tune candidates with CV on training folds and select one."
    )
    compare.add_argument("--path", type=Path, default=None, help="CSV path (default: settings).")
    compare.add_argument(
        "--candidates",
        nargs="+",
        choices=list(DEFAULT_CANDIDATES),
        default=list(DEFAULT_CANDIDATES),
    )
    compare.add_argument("--cv-folds", type=int, default=3)
    compare.set_defaults(handler=_cmd_compare_models)

    anomaly = sub.add_parser(
        "train-anomaly", help="Fit an Isolation Forest and evaluate anomaly rankings."
    )
    anomaly.add_argument("--path", type=Path, default=None, help="CSV path (default: settings).")
    anomaly.add_argument("--fit-on", choices=[f.value for f in FitOn], default=FitOn.ALL.value)
    anomaly.set_defaults(handler=_cmd_train_anomaly)


def _add_decision_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register decision and explanation commands: evaluate-policy, explain."""
    policy = sub.add_parser(
        "evaluate-policy",
        help="Apply a risk policy to a bundle's saved scores (illustrative costs).",
    )
    policy.add_argument("--policy", type=Path, required=True, help="Risk policy JSON file.")
    policy.add_argument("--bundle", type=Path, required=True, help="Model bundle directory.")
    policy.add_argument(
        "--partition",
        choices=["validation", "test"],
        default="validation",
        help="Tune on validation; use test only for the final check.",
    )
    policy.add_argument("--missed-fraud-cost", type=float, default=100.0)
    policy.add_argument("--false-block-cost", type=float, default=10.0)
    policy.add_argument("--review-cost", type=float, default=2.0)
    policy.add_argument("--output", type=Path, default=None, help="Also write JSON here.")
    policy.set_defaults(handler=_cmd_evaluate_policy)

    explain = sub.add_parser(
        "explain", help="Global permutation importance + local explanations for a bundle."
    )
    explain.add_argument("--bundle", type=Path, required=True, help="Model bundle directory.")
    explain.add_argument("--path", type=Path, default=None, help="CSV path (default: settings).")
    explain.add_argument("--n-local", type=int, default=5)
    explain.add_argument("--top", type=int, default=10)
    explain.set_defaults(handler=_cmd_explain)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all registered subcommands."""
    parser = argparse.ArgumentParser(prog="fraud-risk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_data_commands(sub)
    _add_model_commands(sub)
    _add_decision_commands(sub)
    _add_registry_commands(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        configure_logging(settings.log_level, json_format=settings.log_format == "json")
        handler: Handler = args.handler
        payload = handler(args, settings)
    except FraudRiskError as error:
        sys.stderr.write(f"error: {error}\n")
        return EXIT_DOMAIN_ERROR
    write_json(payload, getattr(args, "output", None))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
