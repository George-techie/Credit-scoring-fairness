"""Model inference: probability scoring and the lend/decline decision.

Supports both plain scikit-learn style estimators (``predict_proba``) and
Fairlearn-style reduction ensembles that expose ``predictors_`` and
``weights_``. Pure numpy; the estimator is duck-typed so this module never
imports lightgbm/fairlearn directly and is trivially testable with fakes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .schema import DEFAULT_DECISION_THRESHOLD


def _positive_class_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive (default) class for each row in ``X``."""
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        return proba[:, 1]
    # Fall back to a hard predictor that only emits 0/1 labels.
    return np.asarray(estimator.predict(X), dtype=float)


def predict_default_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return the per-row probability of default.

    For a Fairlearn reduction ensemble the result is the weight-normalised
    average of each sub-predictor's positive-class probability.
    """
    if hasattr(model, "predictors_") and hasattr(model, "weights_"):
        weights = np.asarray(model.weights_, dtype=float)
        total = weights.sum()
        if total <= 0:
            raise ValueError("Ensemble weights sum to a non-positive value.")
        weights = weights / total
        acc = np.zeros(len(X), dtype=float)
        for w, predictor in zip(weights, model.predictors_):
            acc += w * _positive_class_proba(predictor, X)
        return acc
    return _positive_class_proba(model, X)


def decide(
    proba: np.ndarray | Sequence[float] | float,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> np.ndarray:
    """Convert default probabilities to a 0/1 decision at ``threshold``.

    A borrower defaults-prediction fires when probability is strictly greater
    than the threshold.
    """
    proba = np.asarray(proba, dtype=float)
    return (proba > threshold).astype(int)
