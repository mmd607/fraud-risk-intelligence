from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_risk.data.contracts import creditcard_contract
from fraud_risk.data.quality_report import build_quality_report
from fraud_risk.data.schema import REQUIRED_COLUMNS, TARGET_COLUMN
from fraud_risk.data.validation import IssueCode, Severity, validate_dataframe
from tests.support.synthetic import make_transactions


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_transactions(500, 5)


def test_clean_frame_is_valid(frame: pd.DataFrame) -> None:
    report = validate_dataframe(frame)
    assert report.is_valid
    assert report.issues == ()
    assert report.row_count == 500


def test_validation_does_not_mutate_input(frame: pd.DataFrame) -> None:
    frame.loc[0, "V1"] = np.nan
    snapshot = frame.copy(deep=True)
    validate_dataframe(frame)
    pd.testing.assert_frame_equal(frame, snapshot)


def test_validation_is_deterministic(frame: pd.DataFrame) -> None:
    frame.loc[[1, 2], "Amount"] = -5.0
    assert validate_dataframe(frame) == validate_dataframe(frame)


def test_empty_frame(frame: pd.DataFrame) -> None:
    report = validate_dataframe(frame.iloc[0:0])
    assert IssueCode.EMPTY_DATASET in report.codes()
    assert not report.is_valid


def test_missing_and_unexpected_columns(frame: pd.DataFrame) -> None:
    report = validate_dataframe(frame.drop(columns=["V5"]).assign(extra=1.0))
    issues = {i.code: i for i in report.issues}
    assert issues[IssueCode.MISSING_COLUMN].column == "V5"
    assert issues[IssueCode.MISSING_COLUMN].severity is Severity.ERROR
    assert issues[IssueCode.UNEXPECTED_COLUMN].severity is Severity.WARNING


def test_nulls_and_infinities_are_errors(frame: pd.DataFrame) -> None:
    frame.loc[[0, 1], "V2"] = np.nan
    frame.loc[3, "V3"] = np.inf
    report = validate_dataframe(frame)
    issues = {i.code: i for i in report.issues}
    assert issues[IssueCode.NULL_VALUES].count == 2
    assert issues[IssueCode.NON_FINITE_VALUES].count == 1
    assert not report.is_valid


def test_non_numeric_column(frame: pd.DataFrame) -> None:
    frame["V4"] = frame["V4"].astype(str)
    report = validate_dataframe(frame)
    assert IssueCode.NON_NUMERIC_COLUMN in report.codes()


def test_target_domain(frame: pd.DataFrame) -> None:
    frame.loc[0, TARGET_COLUMN] = 3
    report = validate_dataframe(frame)
    assert IssueCode.TARGET_OUT_OF_DOMAIN in report.codes()


def test_single_class_target(frame: pd.DataFrame) -> None:
    frame[TARGET_COLUMN] = 0
    assert IssueCode.SINGLE_CLASS_TARGET in validate_dataframe(frame).codes()


def test_suspicious_positive_rate_is_warning(frame: pd.DataFrame) -> None:
    frame.loc[:200, TARGET_COLUMN] = 1
    report = validate_dataframe(frame)
    assert IssueCode.SUSPICIOUS_POSITIVE_RATE in report.codes()
    assert report.is_valid


def test_duplicates_and_conflicting_labels(frame: pd.DataFrame) -> None:
    duplicate = frame.iloc[[0]].copy()
    conflicting = frame.iloc[[1]].copy()
    conflicting[TARGET_COLUMN] = 1 - conflicting[TARGET_COLUMN]
    report = validate_dataframe(pd.concat([frame, duplicate, conflicting], ignore_index=True))
    codes = report.codes()
    assert IssueCode.DUPLICATE_ROWS in codes
    assert IssueCode.CONFLICTING_DUPLICATE_LABELS in codes
    assert report.is_valid


def test_negative_amount_is_out_of_range(frame: pd.DataFrame) -> None:
    frame.loc[0, "Amount"] = -1.0
    issue = next(i for i in validate_dataframe(frame).issues)
    assert issue.code is IssueCode.VALUE_OUT_OF_RANGE
    assert issue.column == "Amount"


def test_unsorted_time_is_info(frame: pd.DataFrame) -> None:
    report = validate_dataframe(frame.iloc[::-1].reset_index(drop=True))
    issue = next(i for i in report.issues if i.code is IssueCode.TIME_NOT_SORTED)
    assert issue.severity is Severity.INFO
    assert report.is_valid


def test_quality_report_json_round_trip(frame: pd.DataFrame, tmp_path: Path) -> None:
    report = build_quality_report(frame, dataset_fingerprint="abc")
    path = report.write_json(tmp_path / "r" / "quality.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["validation"]["is_valid"] is True
    assert payload["dataset_fingerprint"] == "abc"
    assert payload["target_counts"] == {"0": 495, "1": 5}
    assert [c["name"] for c in payload["columns"]] == list(REQUIRED_COLUMNS)


def test_contract_feature_names_exclude_target() -> None:
    contract = creditcard_contract()
    assert TARGET_COLUMN not in contract.feature_names
    assert len(contract.feature_names) == 30
