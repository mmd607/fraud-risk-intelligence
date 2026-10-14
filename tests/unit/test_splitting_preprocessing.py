from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_risk.core.exceptions import ConfigurationError
from fraud_risk.data.schema import FEATURE_COLUMNS, PCA_COLUMNS, TARGET_COLUMN
from fraud_risk.data.splitting import (
    SplitConfig,
    SplitError,
    SplitStrategy,
    split_dataset,
    split_features_target,
)
from fraud_risk.features.preprocessing import (
    AmountStrategy,
    PreprocessingConfig,
    TimeStrategy,
    build_preprocessor,
    transformed_feature_names,
)
from tests.support.synthetic import make_transactions


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_transactions(2_000, 40, seed=5)


def test_split_is_reproducible(frame: pd.DataFrame) -> None:
    first = split_dataset(frame, SplitConfig(seed=7))
    second = split_dataset(frame, SplitConfig(seed=7))
    assert first.metadata == second.metadata
    pd.testing.assert_index_equal(first.test.index, second.test.index)


def test_different_seed_changes_membership(frame: pd.DataFrame) -> None:
    a = split_dataset(frame, SplitConfig(seed=1)).metadata.test.index_fingerprint
    b = split_dataset(frame, SplitConfig(seed=2)).metadata.test.index_fingerprint
    assert a != b


def test_stratified_split_preserves_class_ratio(frame: pd.DataFrame) -> None:
    splits = split_dataset(frame)
    overall = frame[TARGET_COLUMN].mean()
    for part in (splits.train, splits.validation, splits.test):
        assert part[TARGET_COLUMN].mean() == pytest.approx(overall, abs=0.006)
    sizes = [len(splits.train), len(splits.validation), len(splits.test)]
    assert sum(sizes) == len(frame)
    assert sizes[2] == pytest.approx(0.2 * len(frame), abs=5)


def test_partitions_are_disjoint(frame: pd.DataFrame) -> None:
    splits = split_dataset(frame)
    parts = [set(splits.train.index), set(splits.validation.index), set(splits.test.index)]
    assert not parts[0] & parts[1]
    assert not parts[0] & parts[2]
    assert not parts[1] & parts[2]


def test_exact_duplicates_stay_in_one_partition(frame: pd.DataFrame) -> None:
    data = pd.concat([frame, frame.iloc[:200]], ignore_index=True)
    splits = split_dataset(data, SplitConfig(seed=3, group_duplicates=True))
    key = list(FEATURE_COLUMNS)
    train_keys = set(map(tuple, splits.train[key].to_numpy().round(10)))
    test_keys = set(map(tuple, splits.test[key].to_numpy().round(10)))
    assert not train_keys & test_keys
    assert splits.metadata.duplicate_groups == 200


def test_temporal_split_orders_by_time(frame: pd.DataFrame) -> None:
    splits = split_dataset(frame, SplitConfig(strategy=SplitStrategy.TEMPORAL))
    assert splits.train["Time"].max() <= splits.validation["Time"].min()
    assert splits.validation["Time"].max() <= splits.test["Time"].min()


def test_invalid_split_config() -> None:
    with pytest.raises(ConfigurationError):
        SplitConfig(test_size=0.6, validation_size=0.5)
    with pytest.raises(ConfigurationError):
        SplitConfig(test_size=0.0)


def test_too_few_positives_cannot_be_stratified() -> None:
    with pytest.raises(SplitError):
        split_dataset(make_transactions(100, 2))


def test_features_exclude_target(frame: pd.DataFrame) -> None:
    features, target = split_features_target(frame)
    assert TARGET_COLUMN not in features.columns
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert target.dtype == "int64"


def test_preprocessor_is_fitted_on_training_data_only(frame: pd.DataFrame) -> None:
    splits = split_dataset(frame)
    x_train, _ = split_features_target(splits.train)
    x_test, _ = split_features_target(splits.test)
    preprocessor = build_preprocessor().fit(x_train)
    scaler = preprocessor.named_transformers_["pca"]
    np.testing.assert_allclose(scaler.mean_, x_train[list(PCA_COLUMNS)].mean().to_numpy())
    transformed_test = preprocessor.transform(x_test)
    assert not np.allclose(transformed_test["V1"].mean(), 0.0, atol=1e-12)


def test_preprocessor_drops_target_and_names_features(frame: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(frame)
    names = transformed_feature_names(preprocessor)
    assert TARGET_COLUMN not in names
    assert names == FEATURE_COLUMNS


@pytest.mark.parametrize(
    ("config", "expected_time_columns"),
    [
        (PreprocessingConfig(time=TimeStrategy.DROP), ()),
        (PreprocessingConfig(time=TimeStrategy.HOUR_CYCLICAL), ("Time_hour_sin", "Time_hour_cos")),
        (PreprocessingConfig(amount=AmountStrategy.ROBUST), ("Time",)),
    ],
)
def test_strategies_change_output_columns(
    frame: pd.DataFrame, config: PreprocessingConfig, expected_time_columns: tuple[str, ...]
) -> None:
    names = transformed_feature_names(build_preprocessor(config).fit(frame))
    assert tuple(n for n in names if n.startswith("Time")) == expected_time_columns
    assert "Amount" in names


def test_log_amount_is_standardised_on_train(frame: pd.DataFrame) -> None:
    transformed = build_preprocessor().fit_transform(frame)
    assert transformed["Amount"].mean() == pytest.approx(0.0, abs=1e-9)


def test_pipeline_sanity_with_sklearn_estimator(frame: pd.DataFrame) -> None:
    splits = split_dataset(frame)
    x_train, y_train = split_features_target(splits.train)
    x_val, _ = split_features_target(splits.validation)
    model = Pipeline(
        [("preprocess", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))]
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_val)[:, 1]
    assert scores.shape == (len(x_val),)
    assert np.all((scores >= 0) & (scores <= 1))
