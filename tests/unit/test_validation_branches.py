"""Test-phase T02: branch coverage, stability, and privacy checks for data validation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_risk.cli.main import main
from fraud_risk.data.contracts import ColumnKind, ColumnSpec, DataContract, creditcard_contract
from fraud_risk.data.quality_report import build_quality_report, summarize_column
from fraud_risk.data.schema import TARGET_COLUMN
from fraud_risk.data.validation import IssueCode, Severity, validate_dataframe
from tests.support.synthetic import make_transactions, write_csv

EXPECTED_CODES = {
    "EMPTY_DATASET": "DQ001",
    "MISSING_COLUMN": "DQ002",
    "UNEXPECTED_COLUMN": "DQ003",
    "NON_NUMERIC_COLUMN": "DQ004",
    "NULL_VALUES": "DQ005",
    "NON_FINITE_VALUES": "DQ006",
    "TARGET_OUT_OF_DOMAIN": "DQ007",
    "SINGLE_CLASS_TARGET": "DQ008",
    "SUSPICIOUS_POSITIVE_RATE": "DQ009",
    "DUPLICATE_ROWS": "DQ010",
    "VALUE_OUT_OF_RANGE": "DQ011",
    "TIME_NOT_SORTED": "DQ012",
    "BELOW_MIN_ROWS": "DQ013",
    "CONFLICTING_DUPLICATE_LABELS": "DQ014",
}


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_transactions(300, 3, seed=11)


def _with(contract: DataContract, **changes: object) -> DataContract:
    return dataclasses.replace(contract, **changes)  # type: ignore[arg-type]


def test_issue_codes_are_stable() -> None:
    assert {code.name: code.value for code in IssueCode} == EXPECTED_CODES


def test_issue_codes_are_documented() -> None:
    doc = (Path(__file__).resolve().parents[2] / "docs" / "data-quality.md").read_text("utf-8")
    for name, value in EXPECTED_CODES.items():
        assert value in doc
        assert name in doc


# --- contract construction -----------------------------------------------------------------------


def test_contract_rejects_duplicate_columns() -> None:
    spec = ColumnSpec("a", ColumnKind.FLOAT, "")
    target = ColumnSpec(TARGET_COLUMN, ColumnKind.BINARY_LABEL, "")
    with pytest.raises(ValueError, match="unique"):
        DataContract(columns=(spec, spec, target))


def test_contract_requires_declared_target() -> None:
    with pytest.raises(ValueError, match="Target column"):
        DataContract(columns=(ColumnSpec("a", ColumnKind.FLOAT, ""),))


@pytest.mark.parametrize(
    "changes",
    [
        {"positive_rate_bounds": (0.5, 0.1)},
        {"positive_rate_bounds": (-0.1, 0.1)},
        {"min_rows": 0},
        {"max_duplicate_fraction": 1.5},
    ],
)
def test_contract_rejects_invalid_parameters(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - message varies by parameter
        _with(creditcard_contract(), **changes)


def test_contract_spec_lookup() -> None:
    contract = creditcard_contract()
    assert contract.spec("Amount").min_value == 0.0
    with pytest.raises(KeyError):
        contract.spec("Unknown")


# --- validation branches -------------------------------------------------------------------------


def test_below_min_rows(frame: pd.DataFrame) -> None:
    report = validate_dataframe(frame, _with(creditcard_contract(), min_rows=1000))
    assert IssueCode.BELOW_MIN_ROWS in report.codes()
    assert not report.is_valid


def test_nullable_column_accepts_nulls(frame: pd.DataFrame) -> None:
    contract = creditcard_contract()
    columns = tuple(
        dataclasses.replace(spec, nullable=True) if spec.name == "V9" else spec
        for spec in contract.columns
    )
    frame.loc[0, "V9"] = np.nan
    report = validate_dataframe(frame, _with(contract, columns=columns))
    assert IssueCode.NULL_VALUES not in report.codes()


def test_max_value_range(frame: pd.DataFrame) -> None:
    contract = creditcard_contract()
    columns = tuple(
        dataclasses.replace(spec, max_value=1.0) if spec.name == "V1" else spec
        for spec in contract.columns
    )
    report = validate_dataframe(frame, _with(contract, columns=columns))
    issue = next(i for i in report.issues if i.code is IssueCode.VALUE_OUT_OF_RANGE)
    assert issue.column == "V1"
    assert issue.count == int((frame["V1"] > 1.0).sum())


def test_boolean_column_is_not_numeric(frame: pd.DataFrame) -> None:
    frame["V6"] = frame["V6"] > 0
    assert IssueCode.NON_NUMERIC_COLUMN in validate_dataframe(frame).codes()


def test_non_numeric_target_skips_target_checks(frame: pd.DataFrame) -> None:
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map({0: "no", 1: "yes"})
    codes = validate_dataframe(frame).codes()
    assert IssueCode.NON_NUMERIC_COLUMN in codes
    assert IssueCode.TARGET_OUT_OF_DOMAIN not in codes


def test_duplicates_below_tolerance_are_info(frame: pd.DataFrame) -> None:
    data = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = validate_dataframe(data, _with(creditcard_contract(), max_duplicate_fraction=0.5))
    issue = next(i for i in report.issues if i.code is IssueCode.DUPLICATE_ROWS)
    assert issue.severity is Severity.INFO


def test_time_sorting_check_can_be_disabled(frame: pd.DataFrame) -> None:
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    report = validate_dataframe(
        reversed_frame, _with(creditcard_contract(), expect_time_sorted=False)
    )
    assert IssueCode.TIME_NOT_SORTED not in report.codes()


def test_empty_frame_also_reports_missing_columns() -> None:
    report = validate_dataframe(pd.DataFrame({"Time": []}))
    codes = report.codes()
    assert codes >= {IssueCode.EMPTY_DATASET, IssueCode.MISSING_COLUMN}


def test_all_nan_target_does_not_crash(frame: pd.DataFrame) -> None:
    frame[TARGET_COLUMN] = np.nan
    report = validate_dataframe(frame)
    assert IssueCode.NULL_VALUES in report.codes()
    assert IssueCode.SINGLE_CLASS_TARGET not in report.codes()


def test_multiple_issues_in_one_frame(frame: pd.DataFrame) -> None:
    frame.loc[0, "V1"] = np.nan
    frame.loc[1, "Amount"] = -np.inf
    frame.loc[2, TARGET_COLUMN] = 7
    report = validate_dataframe(frame)
    assert {IssueCode.NULL_VALUES, IssueCode.NON_FINITE_VALUES, IssueCode.TARGET_OUT_OF_DOMAIN} <= (
        report.codes()
    )
    assert len(report.errors) >= 3
    payload = report.to_dict()
    assert payload["error_count"] == len(report.errors)
    assert payload["is_valid"] is False


def test_negative_infinity_not_double_counted_as_out_of_range(frame: pd.DataFrame) -> None:
    frame.loc[1, "Amount"] = -np.inf
    codes = [i.code for i in validate_dataframe(frame).issues]
    assert IssueCode.VALUE_OUT_OF_RANGE not in codes


def test_messages_never_contain_cell_values(frame: pd.DataFrame) -> None:
    frame.loc[0, "Amount"] = -123.456789
    frame.loc[1, TARGET_COLUMN] = 98765
    text = json.dumps(validate_dataframe(frame).to_dict())
    assert "123.456789" not in text
    assert "98765" not in text


def test_issue_serialisation_fields(frame: pd.DataFrame) -> None:
    frame.loc[0, "V2"] = np.nan
    issue = validate_dataframe(frame).to_dict()["issues"][0]  # type: ignore[index]
    assert set(issue) == {"code", "name", "severity", "message", "column", "count"}
    assert issue["code"] == "DQ005"
    assert issue["severity"] == "error"


# --- quality report ------------------------------------------------------------------------------


def test_summarize_all_missing_column() -> None:
    summary = summarize_column(pd.Series([np.nan, np.inf], name="x"))
    assert summary.count == 0
    assert summary.missing == 1
    assert summary.non_finite == 1
    assert summary.mean is None
    assert summary.quantiles["p50"] is None


def test_summarize_single_value_has_no_std() -> None:
    summary = summarize_column(pd.Series([2.0], name="x"))
    assert summary.std is None
    assert summary.minimum == summary.maximum == 2.0


def test_quality_report_is_aggregate_only(frame: pd.DataFrame) -> None:
    payload = build_quality_report(frame).to_dict()
    assert len(payload["columns"]) == 31  # type: ignore[arg-type]
    text = json.dumps(payload)
    assert f"{frame.loc[5, 'V7']!r}" not in text


def test_quality_report_does_not_mutate(frame: pd.DataFrame) -> None:
    snapshot = frame.copy(deep=True)
    build_quality_report(frame)
    pd.testing.assert_frame_equal(frame, snapshot)


# --- CLI --------------------------------------------------------------------------------


def test_cli_validate_valid_dataset(
    tmp_path: Path, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = write_csv(frame, tmp_path / "ok.csv")
    report = tmp_path / "out" / "q.json"
    assert main(["validate", "--path", str(csv), "--report", str(report)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert json.loads(report.read_text(encoding="utf-8"))["validation"]["is_valid"] is True


def test_cli_validate_invalid_dataset_writes_report_and_fails(
    tmp_path: Path, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    frame.loc[0, "Amount"] = -10.0
    csv = write_csv(frame, tmp_path / "bad.csv")
    report = tmp_path / "q.json"
    assert main(["validate", "--path", str(csv), "--report", str(report)]) == 1
    assert "DATA_VALIDATION_FAILED" in capsys.readouterr().err
    assert json.loads(report.read_text(encoding="utf-8"))["validation"]["is_valid"] is False


def test_cli_validate_default_report_location(
    tmp_path: Path, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FRAUD_RISK_REPORTS_DIR", str(tmp_path / "reports"))
    csv = write_csv(frame, tmp_path / "ok.csv")
    assert main(["validate", "--path", str(csv)]) == 0
    assert (tmp_path / "reports" / "data_quality.json").is_file()
