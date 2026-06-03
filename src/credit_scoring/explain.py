"""Helpers for SHAP explanation alignment.

The plotting/SHAP call stays in the API layer, but the part that is easy to get
subtly wrong -- keeping the human-readable feature labels aligned with the exact
column order the model saw -- is isolated here as a pure function so it can be
unit-tested without importing shap or training a model.
"""

from __future__ import annotations

from typing import Mapping, Sequence

UI_NAME_MAP: Mapping[str, str] = {
    "AMT_INCOME_TOTAL": "Total Income",
    "AMT_CREDIT": "Requested Loan Amount",
    "AMT_ANNUITY": "Yearly Annuity",
    "AMT_GOODS_PRICE": "Goods Price",
    "DAYS_BIRTH": "Estimated Age",
    "DAYS_EMPLOYED": "Employment Duration",
    "CNT_CHILDREN": "Child Count",
    "CNT_FAM_MEMBERS": "Family Size",
    "EXT_SOURCE_1": "Bureau Credit Score 1",
    "EXT_SOURCE_2": "Bureau Credit Score 2",
    "EXT_SOURCE_3": "Bureau Credit Score 3",
    "FLAG_OWN_CAR": "Owns Car",
}


def translate_feature_names(columns: Sequence[str]) -> list:
    """Map raw model column names to UI labels, preserving order.

    Unknown columns pass through unchanged so the label list always has the same
    length and order as ``columns`` -- a mismatch here silently mislabels the
    SHAP contributions in the explanation plot.
    """
    return [UI_NAME_MAP.get(c, c) for c in columns]
