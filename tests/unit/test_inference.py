"""Behavioral tests for credit_scoring.inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_scoring.inference import decide, predict_default_proba


@pytest.fixture
def X():
    return pd.DataFrame({"EXT_SOURCE_2": [0.0, 0.25, 0.5, 0.75, 1.0]})


def test_single_model_proba_matches_rule(X, proba_model):
    out = predict_default_proba(proba_model, X)
    expected = 1.0 - X["EXT_SOURCE_2"].to_numpy()
    assert out == pytest.approx(expected)


def test_proba_shape(X, proba_model):
    out = predict_default_proba(proba_model, X)
    assert out.shape == (len(X),)


def test_proba_in_unit_interval(X, proba_model):
    out = predict_default_proba(proba_model, X)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_hard_predictor_fallback(X, hard_model):
    out = predict_default_proba(hard_model, X)
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_ensemble_is_weighted_average(X, fakes):
    # Two identical sub-models -> ensemble equals either one.
    ens = fakes.FakeEnsemble([fakes.ProbaModel(), fakes.ProbaModel()], [1.0, 3.0])
    out = predict_default_proba(ens, X)
    expected = 1.0 - X["EXT_SOURCE_2"].to_numpy()
    assert out == pytest.approx(expected)


def test_ensemble_weights_are_normalised(fakes):
    # Weights need not sum to 1; result must still be a proper average.
    m_low = fakes.ProbaModel("low")
    m_high = fakes.ProbaModel("high")
    Xw = pd.DataFrame({"low": [0.2, 0.2], "high": [0.8, 0.8]})
    ens = fakes.FakeEnsemble([m_low, m_high], [2.0, 2.0])
    out = predict_default_proba(ens, Xw)
    # equal weights -> mean of (1-0.2)=0.8 and (1-0.8)=0.2 -> 0.5
    assert out == pytest.approx([0.5, 0.5])


def test_ensemble_unequal_weights(fakes):
    m_a = fakes.ProbaModel("a")
    m_b = fakes.ProbaModel("b")
    Xw = pd.DataFrame({"a": [0.0], "b": [1.0]})
    # default prob: a->1.0, b->0.0 ; weights 3:1 -> (3*1 + 1*0)/4 = 0.75
    ens = fakes.FakeEnsemble([m_a, m_b], [3.0, 1.0])
    out = predict_default_proba(ens, Xw)
    assert out == pytest.approx([0.75])


def test_zero_sum_weights_raises(X, fakes):
    ens = fakes.FakeEnsemble([fakes.ProbaModel(), fakes.ProbaModel()], [0.0, 0.0])
    with pytest.raises(ValueError):
        predict_default_proba(ens, X)


def test_negative_total_weights_raises(X, fakes):
    ens = fakes.FakeEnsemble([fakes.ProbaModel(), fakes.ProbaModel()], [-1.0, -1.0])
    with pytest.raises(ValueError):
        predict_default_proba(ens, X)


def test_decide_strictly_greater_than_threshold():
    out = decide([0.5], threshold=0.5)
    assert out.tolist() == [0]


def test_decide_above_threshold_fires():
    out = decide([0.51], threshold=0.5)
    assert out.tolist() == [1]


def test_decide_array():
    out = decide([0.1, 0.49, 0.5, 0.9])
    assert out.tolist() == [0, 0, 0, 1]


def test_decide_custom_threshold():
    out = decide([0.3, 0.31], threshold=0.3)
    assert out.tolist() == [0, 1]


def test_decide_returns_int_array():
    out = decide([0.9, 0.1])
    assert out.dtype == np.int_ or out.dtype == np.int64


def test_full_pipeline_decision(X, proba_model):
    proba = predict_default_proba(proba_model, X)
    decisions = decide(proba)
    assert decisions.tolist() == [1, 1, 0, 0, 0]
