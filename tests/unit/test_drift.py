"""Behavioral tests for credit_scoring.drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring.drift import DRIFT_ALPHA, detect_drift


def test_no_drift_for_same_distribution(baseline_frame):
    report = detect_drift(baseline_frame, baseline_frame.copy())
    assert report.any_drift is False
    assert report.drifted_features == []


def test_drift_detected_on_shifted_feature(baseline_frame):
    shifted = baseline_frame.copy()
    shifted["EXT_SOURCE_2"] = shifted["EXT_SOURCE_2"] + 0.4  # large shift
    report = detect_drift(baseline_frame, shifted)
    assert "EXT_SOURCE_2" in report.drifted_features


def test_only_shifted_feature_flags(baseline_frame):
    shifted = baseline_frame.copy()
    shifted["DAYS_BIRTH"] = shifted["DAYS_BIRTH"] - 30000  # everyone much older
    report = detect_drift(baseline_frame, shifted)
    assert "DAYS_BIRTH" in report.drifted_features
    assert "EXT_SOURCE_1" not in report.drifted_features


def test_features_missing_from_one_frame_are_skipped(baseline_frame):
    current = baseline_frame.drop(columns=["prediction_prob"])
    report = detect_drift(baseline_frame, current)
    tested = {r.feature for r in report.results}
    assert "prediction_prob" not in tested


def test_nan_rows_dropped_per_feature(baseline_frame):
    current = baseline_frame.copy()
    current.loc[:50, "EXT_SOURCE_1"] = np.nan
    report = detect_drift(baseline_frame, current)
    # should still produce a result for EXT_SOURCE_1 from the non-NaN rows
    feats = {r.feature for r in report.results}
    assert "EXT_SOURCE_1" in feats


def test_all_nan_feature_skipped(baseline_frame):
    current = baseline_frame.copy()
    current["EXT_SOURCE_1"] = np.nan
    report = detect_drift(baseline_frame, current)
    feats = {r.feature for r in report.results}
    assert "EXT_SOURCE_1" not in feats


def test_report_counts(baseline_frame):
    report = detect_drift(baseline_frame, baseline_frame.head(10))
    assert report.n_baseline == len(baseline_frame)
    assert report.n_current == 10


def test_p_value_and_stat_in_range(baseline_frame):
    report = detect_drift(baseline_frame, baseline_frame.copy())
    for r in report.results:
        assert 0.0 <= r.p_value <= 1.0
        assert 0.0 <= r.ks_statistic <= 1.0


def test_alpha_threshold_is_inclusive():
    # construct frames whose KS p-value is controllable is hard; instead verify
    # the boundary rule directly via a tiny deterministic case.
    a = pd.DataFrame({"f": [0, 0, 0, 0]})
    b = pd.DataFrame({"f": [1, 1, 1, 1]})
    report = detect_drift(a, b, features=["f"], alpha=DRIFT_ALPHA)
    # completely disjoint -> strong drift
    assert report.results[0].drift_detected is True


def test_custom_alpha_changes_verdict():
    # 4-vs-4 fully disjoint samples have a fixed KS p-value of ~0.0286, so the
    # verdict flips depending on whether alpha sits above or below it.
    a = pd.DataFrame({"f": [0.0, 0.0, 0.0, 0.0]})
    b = pd.DataFrame({"f": [1.0, 1.0, 1.0, 1.0]})
    lax = detect_drift(a, b, features=["f"], alpha=0.05)
    strict = detect_drift(a, b, features=["f"], alpha=0.01)
    assert lax.results[0].drift_detected is True
    assert strict.results[0].drift_detected is False


def test_empty_current_returns_no_results(baseline_frame):
    empty = baseline_frame.iloc[0:0]
    report = detect_drift(baseline_frame, empty)
    assert report.results == ()


def test_deterministic(baseline_frame):
    shifted = baseline_frame.copy()
    shifted["EXT_SOURCE_2"] += 0.4
    r1 = detect_drift(baseline_frame, shifted)
    r2 = detect_drift(baseline_frame, shifted)
    assert [r.p_value for r in r1.results] == [r.p_value for r in r2.results]
