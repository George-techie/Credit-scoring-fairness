"""Deterministic evaluation metrics for the credit-scoring model.

Extracted from the retraining script and stripped of the "presentation-mode"
rescaling rail that fabricated AUC/accuracy/F1 with ``random.uniform`` -- those
values are computed honestly here and are fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .schema import DEFAULT_DECISION_THRESHOLD


@dataclass(frozen=True)
class EvalMetrics:
    accuracy: float
    roc_auc: float
    f1: float
    n_samples: int
    single_class: bool


def evaluate_predictions(
    y_true,
    y_pred_proba,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> EvalMetrics:
    """Compute accuracy, ROC-AUC and F1 from probabilities and ground truth.

    ROC-AUC and F1 are undefined when the ground-truth labels contain a single
    class; in that case they are reported as ``nan`` (not silently zeroed and
    never rescaled upward).
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    y_pred = (y_pred_proba > threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    single_class = len(np.unique(y_true)) < 2
    if single_class:
        return EvalMetrics(acc, float("nan"), float("nan"), len(y_true), True)

    auc = float(roc_auc_score(y_true, y_pred_proba))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return EvalMetrics(acc, auc, f1, len(y_true), False)
