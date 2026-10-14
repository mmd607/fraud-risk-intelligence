"""Test-phase T01: independent edge-case coverage for dataset ingestion."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from fraud_risk.cli.main import main
from fraud_risk.core.exceptions import DataError, DatasetNotFoundError, SchemaError
from fraud_risk.core.logging import configure_logging
from fraud_risk.data.ingestion import file_sha256, load_transactions
from fraud_risk.data.schema import FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN
from tests.support.synthetic import make_transactions, write_csv


def _csv_text(frame_rows: int = 6, quote_all: bool = False) -> str:
    frame = make_transactions(frame_rows, 1, seed=3)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, quoting=1 if quote_all else 0, lineterminator="\n")
    return buffer.getvalue()


def test_kaggle_style_quoted_values_are_parsed_as_numbers(tmp_path: Path) -> None:
    path = tmp_path / "quoted.csv"
    path.write_text(_csv_text(quote_all=True), encoding="utf-8")
    result = load_transactions(path)
    assert result.frame[TARGET_COLUMN].dtype == "int64"
    assert result.metadata.target_counts == {0: 5, 1: 1}


def test_utf8_bom_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + _csv_text().encode("utf-8"))
    assert load_transactions(path).metadata.columns[0] == "Time"


def test_whitespace_around_headers_is_stripped(tmp_path: Path) -> None:
    text = _csv_text()
    header, rest = text.split("\n", 1)
    padded = ",".join(f" {name} " for name in header.split(","))
    path = tmp_path / "padded.csv"
    path.write_text(padded + "\n" + rest, encoding="utf-8")
    assert load_transactions(path).metadata.columns == REQUIRED_COLUMNS


def test_column_order_is_not_assumed(tmp_path: Path) -> None:
    frame = make_transactions(20, 2)
    shuffled = frame[list(reversed(frame.columns))]
    result = load_transactions(write_csv(shuffled, tmp_path / "rev.csv"))
    assert set(result.metadata.columns) == set(REQUIRED_COLUMNS)
    assert result.metadata.target_counts == {0: 18, 1: 2}


def test_float_encoded_binary_target_is_accepted(tmp_path: Path) -> None:
    frame = make_transactions(20, 2).astype({TARGET_COLUMN: "float64"})
    result = load_transactions(write_csv(frame, tmp_path / "f.csv"))
    assert result.frame[TARGET_COLUMN].dtype == "int64"


def test_directory_path_is_not_a_dataset(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        load_transactions(tmp_path)


def test_empty_header_name_rejected(tmp_path: Path) -> None:
    text = _csv_text()
    header, rest = text.split("\n", 1)
    path = tmp_path / "blank.csv"
    path.write_text(header + ",\n" + rest.replace("\n", ",0\n"), encoding="utf-8")
    with pytest.raises(SchemaError) as excinfo:
        load_transactions(path)
    assert excinfo.value.code == "DATA_EMPTY_HEADER"


def test_all_errors_are_data_errors(tmp_path: Path) -> None:
    frame = make_transactions(10, 1).drop(columns=[TARGET_COLUMN])
    with pytest.raises(DataError):
        load_transactions(write_csv(frame, tmp_path / "x.csv"))


def test_infinite_values_are_numeric_and_left_for_validation(tmp_path: Path) -> None:
    frame = make_transactions(10, 1)
    frame.loc[0, "Amount"] = np.inf
    result = load_transactions(write_csv(frame, tmp_path / "inf.csv"))
    assert np.isinf(result.frame["Amount"]).sum() == 1


def test_target_with_only_legitimate_rows_loads(tmp_path: Path) -> None:
    frame = make_transactions(10, 0)
    result = load_transactions(write_csv(frame, tmp_path / "x.csv"))
    assert result.metadata.target_counts == {0: 10, 1: 0}
    assert result.metadata.positive_rate == 0.0


def test_error_message_lists_counts_not_raw_values(tmp_path: Path) -> None:
    frame = make_transactions(10, 1).astype({"Amount": "object"})
    frame.loc[2, "Amount"] = "SECRET-CARD-4111"
    with pytest.raises(SchemaError) as excinfo:
        load_transactions(write_csv(frame, tmp_path / "x.csv"))
    assert "SECRET-CARD-4111" not in str(excinfo.value)
    assert "Amount" in str(excinfo.value)


def test_input_file_is_not_modified(tmp_path: Path) -> None:
    path = write_csv(make_transactions(30, 3), tmp_path / "x.csv")
    before = file_sha256(path)
    load_transactions(path)
    assert file_sha256(path) == before


def test_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    a = write_csv(make_transactions(30, 3, seed=1), tmp_path / "a.csv")
    b = write_csv(make_transactions(30, 3, seed=2), tmp_path / "b.csv")
    assert load_transactions(a).metadata.sha256 != load_transactions(b).metadata.sha256


def test_metadata_is_json_serialisable_and_has_no_rows(tmp_path: Path) -> None:
    frame = make_transactions(25, 2)
    meta = load_transactions(write_csv(frame, tmp_path / "x.csv")).metadata
    text = json.dumps(meta.to_dict())
    some_value = f"{frame.loc[0, 'V1']:.6f}"
    assert some_value not in text
    assert set(meta.to_dict()) >= {"row_count", "target_counts", "missing_values", "sha256"}


def test_metadata_reports_full_directory_free_source_name(tmp_path: Path) -> None:
    path = write_csv(make_transactions(10, 1), tmp_path / "nested" / "data.csv")
    assert load_transactions(path).metadata.source_name == "data.csv"


def test_load_logs_aggregates_only(tmp_path: Path) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    frame = make_transactions(15, 2)
    load_transactions(write_csv(frame, tmp_path / "x.csv"))
    record = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert record["event"] == "dataset_loaded"
    assert record["rows"] == 15
    assert f"{frame.loc[0, 'V2']:.6f}" not in stream.getvalue()


def test_features_never_include_target() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 30
    assert REQUIRED_COLUMNS[-1] == TARGET_COLUMN


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_uses_dataset_path_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_csv(make_transactions(12, 2), tmp_path / "env.csv")
    monkeypatch.setenv("FRAUD_RISK_DATASET_PATH", str(path))
    assert main(["ingest"]) == 0
    assert json.loads(capsys.readouterr().out)["dataset"]["row_count"] == 12


def test_cli_allow_extra_columns_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frame = make_transactions(12, 2)
    frame["note"] = 0.0
    path = write_csv(frame, tmp_path / "extra.csv")
    assert main(["ingest", "--path", str(path)]) == 1
    assert "DATA_EXTRA_COLUMNS" in capsys.readouterr().err
    assert main(["ingest", "--path", str(path), "--allow-extra-columns"]) == 0
    assert json.loads(capsys.readouterr().out)["dataset"]["extra_columns"] == ["note"]


def test_cli_invalid_configuration_is_a_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FRAUD_RISK_RANDOM_SEED", "-1")
    assert main(["ingest", "--path", str(tmp_path / "x.csv")]) == 1
    assert "CONFIGURATION_ERROR" in capsys.readouterr().err


def test_cli_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
