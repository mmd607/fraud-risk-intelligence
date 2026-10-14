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
from fraud_risk.core.exceptions import FraudRiskError
from fraud_risk.core.logging import configure_logging
from fraud_risk.data.ingestion import load_transactions

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


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all registered subcommands."""
    parser = argparse.ArgumentParser(prog="fraud-risk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load the local CSV and report dataset metadata.")
    ingest.add_argument("--path", type=Path, default=None, help="CSV path (default: settings).")
    ingest.add_argument("--allow-extra-columns", action="store_true")
    ingest.add_argument("--output", type=Path, default=None, help="Also write JSON here.")
    ingest.set_defaults(handler=_cmd_ingest)
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
