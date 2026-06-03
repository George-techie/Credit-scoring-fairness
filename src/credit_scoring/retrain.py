"""Retraining decision logic.

Only the *decisions* live here -- whether there is enough new data to retrain,
how to split it safely, and whether a freshly trained model is good enough to
replace the incumbent. The heavy lifting (LightGBM fine-tuning, MLflow logging,
joblib persistence) stays in the orchestration script so this module is pure and
deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# A retrain on a handful of rows is meaningless; require a real batch.
MIN_RETRAIN_ROWS = 50

# A new model must beat the incumbent by at least this AUC margin to be
# promoted, so noise alone never ships a regression to production.
MIN_AUC_IMPROVEMENT = 0.005


def should_retrain(n_rows: int, min_rows: int = MIN_RETRAIN_ROWS) -> bool:
    """True when enough fresh feedback has accumulated to retrain."""
    return n_rows >= min_rows


def make_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split into train/validation, holding out a genuine validation set.

    Stratifies on the label when both classes have enough members. Never falls
    back to using the full set as both train and validation -- evaluating on
    rows the model trained on would report a meaningless, optimistic score.
    """
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    class_counts = y.value_counts()
    stratify = y if (len(class_counts) > 1 and class_counts.min() >= 2) else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def select_champion(
    incumbent_auc: float,
    candidate_auc: float,
    min_improvement: float = MIN_AUC_IMPROVEMENT,
) -> str:
    """Return ``"candidate"`` only if it beats the incumbent by the margin.

    ``nan`` candidate AUC (e.g. single-class validation) never wins.
    """
    if candidate_auc is None or np.isnan(candidate_auc):
        return "incumbent"
    if incumbent_auc is None or np.isnan(incumbent_auc):
        return "candidate"
    return "candidate" if candidate_auc >= incumbent_auc + min_improvement else "incumbent"
