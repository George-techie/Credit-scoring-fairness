"""Behavioral tests for credit_scoring.features."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_scoring.features import (
    MissingFeatureError,
    drop_demographics,
    prepare_features,
)
from credit_scoring.schema import DEMOGRAPHIC_COLUMNS


def test_drop_demographics_removes_protected_columns(raw_record):
    out = drop_demographics(raw_record)
    for col in DEMOGRAPHIC_COLUMNS:
        assert col not in out.columns


def test_drop_demographics_keeps_predictive_columns(raw_record):
    out = drop_demographics(raw_record)
    assert "AMT_INCOME_TOTAL" in out.columns
    assert "EXT_SOURCE_3" in out.columns


def test_drop_demographics_does_not_mutate_input(raw_record):
    before = list(raw_record.columns)
    drop_demographics(raw_record)
    assert list(raw_record.columns) == before


def test_prepare_orders_to_feature_names(raw_record, feature_names):
    out = prepare_features(raw_record, feature_names)
    assert list(out.columns) == feature_names


def test_prepare_drops_demographics_when_aligning(raw_record, feature_names):
    out = prepare_features(raw_record, feature_names)
    for col in DEMOGRAPHIC_COLUMNS:
        assert col not in out.columns


def test_prepare_preserves_values(raw_record, feature_names):
    out = prepare_features(raw_record, feature_names)
    assert out.loc[0, "EXT_SOURCE_3"] == pytest.approx(0.5)
    assert out.loc[0, "AMT_CREDIT"] == pytest.approx(400000.0)


def test_prepare_none_feature_names_returns_numeric(raw_record):
    out = prepare_features(raw_record, None)
    assert "CODE_GENDER" not in out.columns
    assert "EXT_SOURCE_1" in out.columns


def test_missing_required_feature_raises(raw_record, feature_names):
    record = raw_record.drop(columns=["EXT_SOURCE_3"])
    with pytest.raises(MissingFeatureError):
        prepare_features(record, feature_names)


def test_missing_feature_error_names_the_feature(raw_record, feature_names):
    record = raw_record.drop(columns=["EXT_SOURCE_3"])
    with pytest.raises(MissingFeatureError) as exc:
        prepare_features(record, feature_names)
    assert "EXT_SOURCE_3" in exc.value.missing


def test_missing_feature_not_silently_filled_with_zero(raw_record, feature_names):
    """A record missing a bureau score must NOT be scored as if that score is 0.

    0 is the maximum-risk extreme on the [0, 1] bureau scale, so silently
    substituting it would systematically penalise applicants whose upstream
    integration omitted an optional field.
    """
    record = raw_record.drop(columns=["EXT_SOURCE_1"])
    with pytest.raises(MissingFeatureError):
        prepare_features(record, feature_names)


def test_multiple_missing_features_all_reported(raw_record, feature_names):
    record = raw_record.drop(columns=["EXT_SOURCE_1", "EXT_SOURCE_3"])
    with pytest.raises(MissingFeatureError) as exc:
        prepare_features(record, feature_names)
    assert set(exc.value.missing) == {"EXT_SOURCE_1", "EXT_SOURCE_3"}


def test_extra_unexpected_columns_are_dropped(raw_record, feature_names):
    record = raw_record.copy()
    record["SOME_LEAKED_ID"] = 999
    out = prepare_features(record, feature_names)
    assert "SOME_LEAKED_ID" not in out.columns


def test_prepare_handles_multiple_rows(raw_record, feature_names):
    two = pd.concat([raw_record, raw_record], ignore_index=True)
    out = prepare_features(two, feature_names)
    assert len(out) == 2
    assert list(out.columns) == feature_names


def test_prepare_output_has_no_nans_when_input_complete(raw_record, feature_names):
    out = prepare_features(raw_record, feature_names)
    assert not out.isna().any().any()
