"""Population-stability / drift detection via the two-sample KS test.

The original monitor wired the KS computation directly into MLflow logging and
reached out to a tracking server at import time. Here the statistical logic is a
pure function returning structured results; persistence/logging is the caller's
concern.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from scipy.stats import ks_2samp

from .schema import DRIFT_FEATURES

# Significance level: a p-value at or below this rejects the null hypothesis that
# the two samples are drawn from the same distribution -> drift.
DRIFT_ALPHA = 0.05


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    ks_statistic: float
    p_value: float
    drift_detected: bool


@dataclass(frozen=True)
class DriftReport:
    results: tuple
    n_baseline: int
    n_current: int

    @property
    def any_drift(self) -> bool:
        return any(r.drift_detected for r in self.results)

    @property
    def drifted_features(self) -> list:
        return [r.feature for r in self.results if r.drift_detected]


def detect_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[str] = DRIFT_FEATURES,
    alpha: float = DRIFT_ALPHA,
) -> DriftReport:
    """Compare ``current`` against ``baseline`` feature-by-feature.

    Only features present in *both* frames are tested. NaNs are dropped per
    feature before the KS test. A feature drifts when its p-value <= ``alpha``.
    """
    results = []
    for feature in features:
        if feature not in baseline.columns or feature not in current.columns:
            continue
        b = baseline[feature].dropna()
        c = current[feature].dropna()
        if len(b) == 0 or len(c) == 0:
            continue
        stat, p_value = ks_2samp(b, c)
        results.append(
            FeatureDrift(
                feature=feature,
                ks_statistic=float(stat),
                p_value=float(p_value),
                drift_detected=bool(p_value <= alpha),
            )
        )
    return DriftReport(
        results=tuple(results),
        n_baseline=len(baseline),
        n_current=len(current),
    )
