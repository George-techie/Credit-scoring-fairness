"""Behavioral tests for credit_scoring.fairness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from credit_scoring.fairness import audit_fairness, passes_four_fifths_rule


def test_perfect_parity_zero_difference():
    # both groups selected at the same 50% rate
    y_pred = [1, 0, 1, 0]
    sensitive = ["A", "A", "B", "B"]
    rep = audit_fairness(y_pred, sensitive)
    assert rep.demographic_parity_difference == pytest.approx(0.0)
    assert rep.disparate_impact_ratio == pytest.approx(1.0)


def test_selection_rate_per_group():
    y_pred = [1, 1, 0, 0, 0, 0]
    sensitive = ["A", "A", "A", "B", "B", "B"]
    rep = audit_fairness(y_pred, sensitive)
    assert rep.group("A").selection_rate == pytest.approx(2 / 3)
    assert rep.group("B").selection_rate == pytest.approx(0.0)


def test_demographic_parity_difference():
    y_pred = [1, 1, 1, 0, 0, 0]   # A: 100%, B: 0%
    sensitive = ["A", "A", "A", "B", "B", "B"]
    rep = audit_fairness(y_pred, sensitive)
    assert rep.demographic_parity_difference == pytest.approx(1.0)


def test_disparate_impact_ratio_80_percent():
    # A selected 100% (4/4), B selected 80% (4/5) -> DI = 0.8
    y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    sensitive = ["A", "A", "A", "A", "B", "B", "B", "B", "B"]
    rep = audit_fairness(y_pred, sensitive)
    assert rep.disparate_impact_ratio == pytest.approx(0.8)
    assert passes_four_fifths_rule(rep) is True


def test_four_fifths_rule_fails_below_threshold():
    y_pred = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # A 100%, B 0%
    sensitive = ["A"] * 5 + ["B"] * 5
    rep = audit_fairness(y_pred, sensitive)
    assert passes_four_fifths_rule(rep) is False


def test_disparate_impact_nan_when_max_zero():
    rep = audit_fairness([0, 0, 0, 0], ["A", "A", "B", "B"])
    assert math.isnan(rep.disparate_impact_ratio)


def test_equalized_odds_needs_y_true():
    rep = audit_fairness([1, 0, 1, 0], ["A", "A", "B", "B"])
    assert math.isnan(rep.equalized_odds_difference)
    assert math.isnan(rep.equal_opportunity_difference)


def test_equal_opportunity_difference():
    # group A: among true-positives, all caught (TPR=1); group B: none (TPR=0)
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 0, 0]
    sensitive = ["A", "A", "B", "B"]
    rep = audit_fairness(y_pred, sensitive, y_true=y_true)
    assert rep.group("A").tpr == pytest.approx(1.0)
    assert rep.group("B").tpr == pytest.approx(0.0)
    assert rep.equal_opportunity_difference == pytest.approx(1.0)


def test_equalized_odds_takes_max_of_tpr_fpr_gaps():
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_pred = [1, 1, 1, 1, 1, 0, 0, 0]   # A: TPR 1 FPR 1 ; B: TPR .5 FPR 0
    sensitive = ["A", "A", "A", "A", "B", "B", "B", "B"]
    rep = audit_fairness(y_pred, sensitive, y_true=y_true)
    # tpr gap = 0.5, fpr gap = 1.0 -> eod = 1.0
    assert rep.equalized_odds_difference == pytest.approx(1.0)


def test_tpr_nan_when_group_has_no_positives():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    sensitive = ["A", "A", "B", "B"]  # A has no true positives
    rep = audit_fairness(y_pred, sensitive, y_true=y_true)
    assert math.isnan(rep.group("A").tpr)


def test_single_group_is_trivially_fair():
    rep = audit_fairness([1, 0, 1], ["A", "A", "A"])
    assert rep.demographic_parity_difference == pytest.approx(0.0)
    assert rep.disparate_impact_ratio == pytest.approx(1.0)


def test_group_counts():
    rep = audit_fairness([1, 0, 1, 0, 1], ["A", "A", "B", "B", "B"])
    assert rep.group("A").n == 2
    assert rep.group("B").n == 3


def test_deterministic():
    y_pred = [1, 0, 1, 1, 0]
    sensitive = ["A", "B", "A", "B", "A"]
    a = audit_fairness(y_pred, sensitive)
    b = audit_fairness(y_pred, sensitive)
    assert a.demographic_parity_difference == b.demographic_parity_difference
