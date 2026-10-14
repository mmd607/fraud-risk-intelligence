from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_risk.cli.main import main
from tests.support.synthetic import make_transactions, write_csv

pytestmark = pytest.mark.integration


def test_ingest_command_reports_metadata_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv(make_transactions(50, 5), tmp_path / "creditcard.csv")
    output = tmp_path / "out" / "ingest.json"
    assert main(["ingest", "--path", str(path), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"]["row_count"] == 50
    assert payload["dataset"]["target_counts"] == {"0": 45, "1": 5}
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_ingest_command_missing_file_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ingest", "--path", str(tmp_path / "nope.csv")]) == 1
    assert "DATA_NOT_FOUND" in capsys.readouterr().err
