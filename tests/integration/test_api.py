"""Integration tests for the FastAPI layer via TestClient.

Uses a tiny in-process fake model so no trained artifact or network is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_scoring.api import state
from credit_scoring.api.app import create_app


class _FakeModel:
    """Default probability = 1 - EXT_SOURCE_2 (deterministic)."""

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        p = 1.0 - X["EXT_SOURCE_2"].to_numpy(dtype=float).clip(0, 1)
        return np.column_stack([1 - p, p])


def _payload(**overrides):
    base = {
        "AMT_INCOME_TOTAL": 150000.0, "AMT_CREDIT": 400000.0,
        "AMT_ANNUITY": 24000.0, "AMT_GOODS_PRICE": 380000.0,
        "DAYS_BIRTH": -14000.0, "DAYS_EMPLOYED": -2000.0,
        "CNT_CHILDREN": 1.0, "CNT_FAM_MEMBERS": 3.0,
        "EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.7, "EXT_SOURCE_3": 0.5,
        "FLAG_OWN_CAR": 1, "CODE_GENDER": "F",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Working", "NAME_HOUSING_TYPE": "House / apartment",
    }
    base.update(overrides)
    return base


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate feedback.db
    with TestClient(create_app()) as c:
        state.model = _FakeModel()
        state.feature_names = None
        yield c
    state.model = None
    state.feature_names = None


def test_predict_returns_probability(client):
    r = client.post("/predict", json=_payload(EXT_SOURCE_2=0.7))
    assert r.status_code == 200
    body = r.json()
    assert body["default_probability"] == pytest.approx(0.3)
    assert body["default_prediction"] == 0


def test_predict_high_risk_flagged(client):
    r = client.post("/predict", json=_payload(EXT_SOURCE_2=0.1))
    assert r.status_code == 200
    assert r.json()["default_prediction"] == 1


def test_predict_model_unavailable_returns_503(client):
    state.model = None
    r = client.post("/predict", json=_payload())
    assert r.status_code == 503


def test_predict_rejects_malformed_request(client):
    r = client.post("/predict", json={"AMT_CREDIT": 1000})
    assert r.status_code == 422  # pydantic validation


def test_feedback_records_ok(client):
    r = client.post("/feedback", json=_payload(prediction_prob=0.3, ground_truth=0))
    assert r.status_code == 200
    assert r.json()["status"] == "success"
