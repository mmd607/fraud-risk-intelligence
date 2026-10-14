from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud_risk.core.exceptions import DataReadError, DatasetNotFoundError, SchemaError
from fraud_risk.data.ingestion import file_sha256, load_transactions
from fraud_risk.data.schema import FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN
from tests.support.synthetic import make_transactions, write_csv


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    return write_csv(make_transactions(200, 10), tmp_path / "creditcard.csv")


def test_loads_valid_csv_with_metadata(csv_path: Path) -> None:
    result = load_transactions(csv_path)
    meta = result.metadata
    assert meta.row_count == 200
    assert meta.target_counts == {0: 190, 1: 10}
    assert meta.positive_rate == pytest.approx(0.05)
    assert meta.columns == REQUIRED_COLUMNS
    assert meta.total_missing_values == 0
    assert meta.source_name == "creditcard.csv"
    assert meta.sha256 == file_sha256(csv_path)
    assert result.frame[TARGET_COLUMN].dtype == "int64"
    assert all(result.frame[c].dtype == "float64" for c in FEATURE_COLUMNS)


def test_missing_file_has_download_instructions(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match=r"kaggle.com/datasets/mlg-ulb/creditcardfraud"):
        load_transactions(tmp_path / "absent.csv")


def test_missing_columns_are_named(tmp_path: Path) -> None:
    frame = make_transactions().drop(columns=["V7", "Amount"])
    with pytest.raises(SchemaError, match="V7") as excinfo:
        load_transactions(write_csv(frame, tmp_path / "x.csv"))
    assert excinfo.value.code == "DATA_MISSING_COLUMNS"
    assert "Amount" in str(excinfo.value)


def test_duplicate_headers_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    frame = make_transactions(10, 1)
    header = ",".join([*REQUIRED_COLUMNS, "V1"])
    rows = [",".join(str(v) for v in [*row, 0.0]) for row in frame.itertuples(index=False)]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError) as excinfo:
        load_transactions(path)
    assert excinfo.value.code == "DATA_DUPLICATE_HEADER"


def test_extra_columns_rejected_by_default_and_allowed_on_request(tmp_path: Path) -> None:
    frame = make_transactions(20, 2)
    frame["merchant_note"] = 1.0
    path = write_csv(frame, tmp_path / "extra.csv")
    with pytest.raises(SchemaError) as excinfo:
        load_transactions(path)
    assert excinfo.value.code == "DATA_EXTRA_COLUMNS"
    assert load_transactions(path, allow_extra_columns=True).metadata.extra_columns == (
        "merchant_note",
    )


def test_non_numeric_feature_rejected(tmp_path: Path) -> None:
    frame = make_transactions(20, 2).astype({"V3": "object"})
    frame.loc[5, "V3"] = "abc"
    with pytest.raises(SchemaError, match="V3") as excinfo:
        load_transactions(write_csv(frame, tmp_path / "x.csv"))
    assert excinfo.value.code == "DATA_NON_NUMERIC"


@pytest.mark.parametrize("bad_label", [2, -1, 0.5])
def test_invalid_target_rejected(tmp_path: Path, bad_label: float) -> None:
    frame = make_transactions(20, 2).astype({TARGET_COLUMN: "float64"})
    frame.loc[0, TARGET_COLUMN] = bad_label
    with pytest.raises(SchemaError) as excinfo:
        load_transactions(write_csv(frame, tmp_path / "x.csv"))
    assert excinfo.value.code == "DATA_TARGET_INVALID"


def test_missing_target_rejected(tmp_path: Path) -> None:
    frame = make_transactions(20, 2).astype({TARGET_COLUMN: "float64"})
    frame.loc[3, TARGET_COLUMN] = None
    with pytest.raises(SchemaError) as excinfo:
        load_transactions(write_csv(frame, tmp_path / "x.csv"))
    assert excinfo.value.code == "DATA_TARGET_MISSING"


def test_header_only_file_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(",".join(REQUIRED_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError) as excinfo:
        load_transactions(path)
    assert excinfo.value.code == "DATA_EMPTY"


def test_zero_byte_file_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "zero.csv"
    path.write_bytes(b"")
    with pytest.raises(SchemaError) as excinfo:
        load_transactions(path)
    assert excinfo.value.code == "DATA_EMPTY"


def test_non_utf8_file_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "binary.csv"
    path.write_bytes(b"\xff\xfe\x00\x81\x82garbage")
    with pytest.raises(DataReadError):
        load_transactions(path)


def test_missing_feature_values_reported_not_imputed(tmp_path: Path) -> None:
    frame = make_transactions(30, 3)
    frame.loc[[1, 2], "V10"] = None
    result = load_transactions(write_csv(frame, tmp_path / "x.csv"))
    assert result.metadata.missing_values["V10"] == 2
    assert result.frame["V10"].isna().sum() == 2
    assert result.metadata.warnings


def test_duplicate_rows_counted_not_dropped(tmp_path: Path) -> None:
    frame = make_transactions(30, 3)
    frame = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
    result = load_transactions(write_csv(frame, tmp_path / "x.csv"))
    assert result.metadata.duplicate_row_count == 2
    assert result.metadata.row_count == 32
