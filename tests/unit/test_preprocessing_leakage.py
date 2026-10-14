"""Test-phase T03: leakage, reproducibility, and robustness of splitting and preprocessing."""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from fraud_risk.cli.main import main
from fraud_risk.data.schema import FEATURE_COLUMNS, PCA_COLUMNS, TARGET_COLUMN
from fraud_risk.data.splitting import (
    SplitConfig,
    SplitError,
    SplitStrategy,
    index_fingerprint,
    split_dataset,
    split_features_target,
)
from fraud_risk.features.preprocessing import (
    AmountStrategy,
    PcaStrategy,
    PreprocessingConfig,
    TimeStrategy,
    build_preprocessor,
    hour_of_day_cyclical,
)
from tests.support.synthetic import make_transactions, write_csv


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_transactions(3_000, 60, seed=21)


# --- splitting ---


def test_split_does_not_mutate_input(frame: pd.DataFrame) -> None:
    snapshot = frame.copy(deep=True)
    split_dataset(frame)
    pd.testing.assert_frame_equal(frame, snapshot)


@pytest.mark.parametrize("strategy", list(SplitStrategy))
def test_every_row_assigned_exactly_once(frame: pd.DataFrame, strategy: SplitStrategy) -> None:
    splits = split_dataset(frame, SplitConfig(strategy=strategy))
    combined = splits.train.index.append(splits.validation.index).append(splits.test.index)
    assert combined.is_unique
    assert set(combined) == set(frame.index)


def test_custom_sizes_are_respected(frame: pd.DataFrame) -> None:
    splits = split_dataset(frame, SplitConfig(test_size=0.1, validation_size=0.3))
    assert len(splits.test) == pytest.approx(300, abs=3)
    assert len(splits.validation) == pytest.approx(900, abs=3)


def test_metadata_counts_match_partitions(frame: pd.DataFrame) -> None:
    splits = split_dataset(frame)
    meta = splits.metadata
    assert meta.train.rows == len(splits.train)
    assert meta.test.positives == int(splits.test[TARGET_COLUMN].sum())
    assert meta.train.positives + meta.validation.positives + meta.test.positives == 60
    assert meta.seed == 42
    assert meta.strategy == "stratified"


def test_metadata_json_contains_no_rows(frame: pd.DataFrame, tmp_path: Path) -> None:
    splits = split_dataset(frame)
    path = splits.metadata.write_json(tmp_path / "meta.json")
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert set(payload) >= {"train", "validation", "test", "seed", "strategy"}
    assert f"{frame.loc[0, 'V1']!r}" not in text


def test_index_fingerprint_is_order_independent() -> None:
    assert index_fingerprint(pd.Index([3, 1, 2])) == index_fingerprint(pd.Index([1, 2, 3]))
    assert index_fingerprint(pd.Index([1, 2])) != index_fingerprint(pd.Index([1, 3]))


def test_ungrouped_split_can_leak_duplicates_but_grouped_cannot(frame: pd.DataFrame) -> None:
    data = pd.concat([frame] * 2, ignore_index=True)  # every row duplicated once
    key = list(FEATURE_COLUMNS)

    def overlap(config: SplitConfig) -> int:
        splits = split_dataset(data, config)
        train = set(map(tuple, splits.train[key].to_numpy()))
        test = set(map(tuple, splits.test[key].to_numpy()))
        return len(train & test)

    assert overlap(SplitConfig(group_duplicates=False)) > 0
    assert overlap(SplitConfig(group_duplicates=True)) == 0


def test_temporal_split_fails_when_partition_lacks_positives() -> None:
    data = make_transactions(1_000, 0, seed=1)
    data.loc[data.index[:30], TARGET_COLUMN] = 1  # only the earliest rows are fraud
    with pytest.raises(SplitError, match="both classes"):
        split_dataset(data, SplitConfig(strategy=SplitStrategy.TEMPORAL))


def test_non_unique_index_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(SplitError, match="unique"):
        split_dataset(frame.set_index(np.zeros(len(frame), dtype=int)))


def test_missing_target_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(SplitError, match="target"):
        split_dataset(frame.drop(columns=[TARGET_COLUMN]))


def test_split_features_target_requires_all_columns(frame: pd.DataFrame) -> None:
    with pytest.raises(Exception, match="V3"):
        split_features_target(frame.drop(columns=["V3"]))


def test_split_features_target_ignores_extra_columns(frame: pd.DataFrame) -> None:
    features, _ = split_features_target(frame.assign(Class_copy=frame[TARGET_COLUMN]))
    assert "Class_copy" not in features.columns


# --- preprocessing leakage ---


def test_fitting_on_full_data_differs_from_train_only(frame: pd.DataFrame) -> None:
    splits = split_dataset(frame)
    x_train, _ = split_features_target(splits.train)
    x_all, _ = split_features_target(frame)
    train_scaler = build_preprocessor().fit(x_train).named_transformers_["pca"]
    full_scaler = build_preprocessor().fit(x_all).named_transformers_["pca"]
    assert not np.allclose(train_scaler.mean_, full_scaler.mean_)


def test_preprocessor_refit_inside_each_cv_fold(frame: pd.DataFrame) -> None:
    x, y = split_features_target(frame)
    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("clf", LogisticRegression(max_iter=300))]
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    means = []
    for train_idx, _ in folds.split(x, y):
        fitted = clone(pipeline).fit(x.iloc[train_idx], y.iloc[train_idx])
        means.append(fitted.named_steps["preprocess"].named_transformers_["pca"].mean_[0])
        expected = x.iloc[train_idx]["V1"].mean()
        assert means[-1] == pytest.approx(expected)
    assert len(set(np.round(means, 12))) == 3
    scores = cross_val_score(pipeline, x, y, cv=folds, scoring="average_precision")
    assert scores.shape == (3,)


def test_target_column_never_reaches_estimator(frame: pd.DataFrame) -> None:
    transformed = build_preprocessor().fit_transform(frame)  # frame still contains Class
    assert TARGET_COLUMN not in transformed.columns
    assert transformed.shape[1] == len(FEATURE_COLUMNS)


def test_unfitted_preprocessor_cannot_transform(frame: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError):
        build_preprocessor().transform(frame)


def test_transform_fails_on_missing_input_column(frame: pd.DataFrame) -> None:
    fitted = build_preprocessor().fit(frame)
    with pytest.raises(ValueError, match="Amount"):
        fitted.transform(frame.drop(columns=["Amount"]))


def test_passthrough_strategies_preserve_values(frame: pd.DataFrame) -> None:
    config = PreprocessingConfig(amount=AmountStrategy.PASSTHROUGH, pca=PcaStrategy.PASSTHROUGH)
    transformed = build_preprocessor(config).fit_transform(frame)
    np.testing.assert_allclose(transformed["Amount"], frame["Amount"])
    np.testing.assert_allclose(transformed[list(PCA_COLUMNS)], frame[list(PCA_COLUMNS)])


def test_standard_and_robust_amount_differ(frame: pd.DataFrame) -> None:
    robust = build_preprocessor(PreprocessingConfig(amount=AmountStrategy.ROBUST))
    standard = build_preprocessor(PreprocessingConfig(amount=AmountStrategy.STANDARD))
    a = robust.fit_transform(frame)["Amount"]
    b = standard.fit_transform(frame)["Amount"]
    assert not np.allclose(a, b)


def test_hour_cyclical_values() -> None:
    encoded = hour_of_day_cyclical([0.0, 21_600.0, 86_400.0 + 43_200.0])
    np.testing.assert_allclose(encoded[0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(encoded[1], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(encoded[2], [0.0, -1.0], atol=1e-12)


def test_hour_cyclical_drops_raw_time(frame: pd.DataFrame) -> None:
    out = build_preprocessor(PreprocessingConfig(time=TimeStrategy.HOUR_CYCLICAL)).fit_transform(
        frame
    )
    assert "Time" not in out.columns
    assert np.all(np.abs(out[["Time_hour_sin", "Time_hour_cos"]].to_numpy()) <= 1.0)


def test_fitted_pipeline_survives_joblib_round_trip(frame: pd.DataFrame) -> None:
    x, y = split_features_target(frame)
    config = PreprocessingConfig(time=TimeStrategy.HOUR_CYCLICAL)
    model = Pipeline(
        [("preprocess", build_preprocessor(config)), ("clf", LogisticRegression(max_iter=300))]
    ).fit(x, y)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(restored.predict_proba(x), model.predict_proba(x))


# --- CLI ---


def test_cli_split_writes_metadata_only(
    frame: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = write_csv(frame, tmp_path / "d.csv")
    out = tmp_path / "meta.json"
    assert main(["split", "--path", str(csv), "--output", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["split"]["train"]["rows"] + payload["split"]["test"]["rows"] < len(frame)
    assert json.loads(out.read_text(encoding="utf-8"))["seed"] == 42
    assert list(tmp_path.iterdir()) == sorted([csv, out]) or set(tmp_path.iterdir()) == {csv, out}


def test_cli_split_refuses_invalid_data(
    frame: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame.loc[0, "Amount"] = -1.0
    csv = write_csv(frame, tmp_path / "bad.csv")
    assert main(["split", "--path", str(csv), "--output", str(tmp_path / "m.json")]) == 1
    assert "DATA_VALIDATION_FAILED" in capsys.readouterr().err
    assert not (tmp_path / "m.json").exists()


def test_cli_split_uses_configured_seed(
    frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = write_csv(frame, tmp_path / "d.csv")
    monkeypatch.setenv("FRAUD_RISK_RANDOM_SEED", "9")
    out = tmp_path / "m.json"
    assert main(["split", "--path", str(csv), "--output", str(out), "--strategy", "temporal"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["seed"] == 9
    assert payload["strategy"] == "temporal"
