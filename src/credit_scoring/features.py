"""Feature preparation for inference and retraining.

This consolidates the column-alignment logic that previously lived (duplicated)
inside the ``/predict``, ``/explain`` and retraining code paths. Pure pandas;
no model, network or filesystem dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .schema import DEMOGRAPHIC_COLUMNS


class MissingFeatureError(ValueError):
    """Raised when a record is missing a feature the model requires.

    Missing predictive features must surface as an explicit error rather than
    being silently substituted with a placeholder value, because for several
    inputs (notably the bureau scores, which live on a [0, 1] scale where higher
    is safer) a placeholder of 0 is not neutral -- it is the maximum-risk
    extreme, and would silently bias the score.
    """

    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        super().__init__(
            "Missing required model feature(s): " + ", ".join(self.missing)
        )


def drop_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with protected demographic columns removed."""
    cols = [c for c in DEMOGRAPHIC_COLUMNS if c in df.columns]
    return df.drop(columns=cols)


def prepare_features(
    records: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Align raw request records to the exact feature contract the model expects.

    Parameters
    ----------
    records:
        One or more raw input rows. May contain demographic columns; these are
        always dropped before the frame reaches the model.
    feature_names:
        The ordered list of columns the trained model was fitted on. When
        provided, the returned frame contains *exactly* these columns, in this
        order. When ``None``, demographic columns are dropped and the remaining
        columns are returned unchanged.

    Returns
    -------
    pandas.DataFrame
        A frame ready to hand to the estimator.

    Raises
    ------
    MissingFeatureError
        If ``feature_names`` is supplied and any required feature is absent from
        ``records``. Missing features are never filled with a placeholder.
    """
    if feature_names is None:
        return drop_demographics(records)

    feature_names = list(feature_names)
    missing = [f for f in feature_names if f not in records.columns]
    if missing:
        raise MissingFeatureError(missing)

    # Select and order exactly to the model's contract. Because every required
    # column is guaranteed present, no fabricated values are introduced.
    return records.loc[:, feature_names].copy()
