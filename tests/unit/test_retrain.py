"""Behavioral tests for credit_scoring.retrain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_scoring.retrain import (
    MIN_RETRAIN_ROWS,
    make_validation_split,
    select_champion,
    should_retrain,
)


def test_should_retrain_below_threshold():
    assert should_retrain(MIN_RETRAIN_ROWS - 1) is False


def test_should_retrain_at_threshold():
    assert should_retrain(MIN_RETRAIN_ROWS) is True


def test_should_retrain_custom_min():
    assert should_retrain(10, min_rows=5) is True
    assert should_retrain(3, min_rows=5) is False


@pytest.fixture
def Xy():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_split_holds_out_validation(Xy):
    X, y = Xy
    X_tr, X_val, y_tr, y_val = make_validation_split(X, y, test_size=0.2)
    assert len(X_val) == 20
    assert len(X_tr) == 80


def test_split_no_train_val_overlap(Xy):
    X, y = Xy
    X_tr, X_val, y_tr, y_val = make_validation_split(X, y)
    # indices were reset; reconstruct via row content hashing
    tr_rows = {tuple(r) for r in X_tr.to_numpy()}
    val_rows = {tuple(r) for r in X_val.to_numpy()}
    assert tr_rows.isdisjoint(val_rows)


def test_split_does_not_leak_full_set_as_validation(Xy):
    X, y = Xy
    X_tr, X_val, _, _ = make_validation_split(X, y)
    # The buggy fallback returned X as both train and val (len == full set).
    assert len(X_val) < len(X)


def test_split_stratifies_when_possible(Xy):
    X, y = Xy
    _, _, y_tr, y_val = make_validation_split(X, y)
    # both classes should appear in validation for a balanced 100-row set
    assert set(y_val.unique()) == {0, 1}


def test_split_handles_tiny_single_class():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1, 1, 1, 1])
    # should not raise despite single class (stratify disabled internally)
    X_tr, X_val, y_tr, y_val = make_validation_split(X, y, test_size=0.25)
    assert len(X_val) >= 1


def test_select_champion_promotes_clear_winner():
    assert select_champion(0.70, 0.80) == "candidate"


def test_select_champion_keeps_incumbent_on_tie():
    assert select_champion(0.80, 0.80) == "incumbent"


def test_select_champion_requires_margin():
    # improvement smaller than MIN margin -> keep incumbent
    assert select_champion(0.800, 0.803) == "incumbent"


def test_select_champion_respects_margin_boundary():
    assert select_champion(0.800, 0.805) == "candidate"


def test_select_champion_nan_candidate_loses():
    assert select_champion(0.70, float("nan")) == "incumbent"


def test_select_champion_nan_incumbent_yields_candidate():
    assert select_champion(float("nan"), 0.60) == "candidate"


def test_select_champion_does_not_ship_regression():
    assert select_champion(0.85, 0.60) == "incumbent"
