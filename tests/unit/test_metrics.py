"""Behavioral tests for credit_scoring.retrain_eval."""

from __future__ import annotations

import math

import numpy as np
import pytest

from credit_scoring.retrain_eval import evaluate_predictions


def test_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]
    m = evaluate_predictions(y_true, y_proba)
    assert m.accuracy == pytest.approx(1.0)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_metrics_not_rescaled_upward():
    # A genuinely weak model must report weak numbers -- no demo rail.
    y_true = [0, 1, 0, 1, 0, 1]
    y_proba = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    m = evaluate_predictions(y_true, y_proba)
    assert m.roc_auc == pytest.approx(0.5, abs=1e-9)
    assert m.roc_auc < 0.77  # would have been faked up to ~0.78 in the old code


def test_single_class_auc_is_nan():
    m = evaluate_predictions([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9])
    assert m.single_class is True
    assert math.isnan(m.roc_auc)
    assert math.isnan(m.f1)


def test_single_class_accuracy_still_computed():
    m = evaluate_predictions([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9])
    assert m.accuracy == pytest.approx(1.0)


def test_threshold_affects_accuracy():
    y_true = [0, 1, 1]
    y_proba = [0.2, 0.4, 0.8]
    low = evaluate_predictions(y_true, y_proba, threshold=0.3)   # preds 0,1,1
    high = evaluate_predictions(y_true, y_proba, threshold=0.5)  # preds 0,0,1
    assert low.accuracy == pytest.approx(1.0)
    assert high.accuracy == pytest.approx(2 / 3)
    assert low.accuracy != high.accuracy


def test_n_samples_reported():
    m = evaluate_predictions([0, 1, 0], [0.2, 0.8, 0.3])
    assert m.n_samples == 3


def test_auc_in_unit_interval():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    m = evaluate_predictions(y_true, y_proba)
    assert 0.0 <= m.roc_auc <= 1.0


def test_deterministic():
    y_true = [0, 1, 1, 0, 1]
    y_proba = [0.2, 0.7, 0.6, 0.3, 0.55]
    a = evaluate_predictions(y_true, y_proba)
    b = evaluate_predictions(y_true, y_proba)
    assert (a.accuracy, a.roc_auc, a.f1) == (b.accuracy, b.roc_auc, b.f1)


def test_accuracy_in_unit_interval():
    m = evaluate_predictions([0, 1, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert 0.0 <= m.accuracy <= 1.0


def test_f1_zero_division_safe():
    # model predicts all-negative; no positive predictions
    m = evaluate_predictions([0, 1], [0.1, 0.2])
    assert m.f1 == pytest.approx(0.0)
