"""Behavioral tests for credit_scoring.explain."""

from __future__ import annotations

from credit_scoring.explain import UI_NAME_MAP, translate_feature_names


def test_translation_preserves_length():
    cols = ["EXT_SOURCE_1", "AMT_CREDIT", "FLAG_OWN_CAR"]
    assert len(translate_feature_names(cols)) == len(cols)


def test_translation_preserves_order():
    cols = ["AMT_CREDIT", "EXT_SOURCE_1"]
    out = translate_feature_names(cols)
    assert out == ["Requested Loan Amount", "Bureau Credit Score 1"]


def test_unknown_columns_pass_through():
    cols = ["EXT_SOURCE_1", "ENGINEERED_RATIO"]
    out = translate_feature_names(cols)
    assert out == ["Bureau Credit Score 1", "ENGINEERED_RATIO"]


def test_all_known_features_mapped():
    out = translate_feature_names(list(UI_NAME_MAP.keys()))
    assert out == list(UI_NAME_MAP.values())


def test_empty_input():
    assert translate_feature_names([]) == []
