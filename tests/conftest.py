"""Shared fixtures: deterministic fakes so tests need neither lightgbm nor a
trained artifact on disk, and run identically on every machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class ProbaModel:
    """Minimal scikit-learn-style classifier with a fixed scoring rule.

    Default probability is a deterministic function of EXT_SOURCE_2 (higher
    bureau score -> lower default probability), so tests can assert exact values.
    """

    def __init__(self, feature: str = "EXT_SOURCE_2"):
        self.feature = feature

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        p_default = 1.0 - X[self.feature].to_numpy(dtype=float).clip(0.0, 1.0)
        return np.column_stack([1.0 - p_default, p_default])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class HardModel:
    """A predictor that only emits 0/1 labels (no predict_proba)."""

    def __init__(self, threshold: float = 0.5, feature: str = "EXT_SOURCE_2"):
        self.threshold = threshold
        self.feature = feature

    def predict(self, X):
        X = pd.DataFrame(X)
        return (X[self.feature].to_numpy(dtype=float) < self.threshold).astype(int)


class FakeEnsemble:
    """Fairlearn-reduction-style ensemble: ``predictors_`` + ``weights_``."""

    def __init__(self, predictors, weights):
        self.predictors_ = list(predictors)
        self.weights_ = list(weights)


@pytest.fixture
def fakes():
    """Access to the fake model classes for tests that build custom configs."""
    from types import SimpleNamespace

    return SimpleNamespace(
        ProbaModel=ProbaModel, HardModel=HardModel, FakeEnsemble=FakeEnsemble
    )


@pytest.fixture
def feature_names():
    return [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "FLAG_OWN_CAR",
    ]


@pytest.fixture
def raw_record():
    """A single complete request row, including demographic columns."""
    return pd.DataFrame([{
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 400000.0,
        "AMT_ANNUITY": 24000.0,
        "AMT_GOODS_PRICE": 380000.0,
        "DAYS_BIRTH": -14000.0,
        "DAYS_EMPLOYED": -2000.0,
        "CNT_CHILDREN": 1.0,
        "CNT_FAM_MEMBERS": 3.0,
        "EXT_SOURCE_1": 0.6,
        "EXT_SOURCE_2": 0.7,
        "EXT_SOURCE_3": 0.5,
        "FLAG_OWN_CAR": 1,
        "CODE_GENDER": "F",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Working",
        "NAME_HOUSING_TYPE": "House / apartment",
    }])


@pytest.fixture
def proba_model():
    return ProbaModel()


@pytest.fixture
def hard_model():
    return HardModel()


@pytest.fixture
def ensemble(proba_model):
    return FakeEnsemble([ProbaModel(), ProbaModel()], [1.0, 3.0])


@pytest.fixture
def baseline_frame():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "EXT_SOURCE_1": rng.uniform(0.4, 0.6, n),
        "EXT_SOURCE_2": rng.uniform(0.4, 0.6, n),
        "EXT_SOURCE_3": rng.uniform(0.4, 0.6, n),
        "DAYS_BIRTH": rng.integers(-20000, -8000, n).astype(float),
        "prediction_prob": rng.uniform(0.1, 0.3, n),
    })
