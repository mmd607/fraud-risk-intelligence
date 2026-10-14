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

from fraud_risk.config.settings import Settings, load_settings
from fraud_risk.core.exceptions import DataValidationError, FraudRiskError
from fraud_risk.core.logging import configure_logging
from fraud_risk.data.ingestion import load_transactions
from fraud_risk.data.prepare import load_validated_dataset
from fraud_risk.data.quality_report import build_quality_report
from fraud_risk.data.splitting import SplitConfig, SplitStrategy, split_dataset
from fraud_risk.evaluation.metrics import ThresholdStrategy
from fraud_risk.models.candidates import DEFAULT_CANDIDATES
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


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all registered subcommands."""
    parser = argparse.ArgumentParser(prog="fraud-risk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

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
