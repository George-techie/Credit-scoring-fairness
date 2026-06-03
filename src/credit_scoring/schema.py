"""Shared schema constants for the credit-scoring service.

Centralises the feature contract that was previously duplicated across the
FastAPI app, the retraining pipeline and the drift monitor. Importing this
module has no side effects.
"""

from __future__ import annotations

# Demographic / protected attributes that are tracked for fairness auditing but
# are NOT fed to the model as predictive inputs.
DEMOGRAPHIC_COLUMNS = (
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
)

# Numeric predictive features the baseline model is trained on.
NUMERIC_FEATURES = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "FLAG_OWN_CAR",
)

# External bureau scores live on a [0, 1] scale where HIGHER means safer.
# A value of 0 is therefore the maximum-risk extreme, never a neutral default.
BUREAU_SCORE_FEATURES = (
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)

# Features the drift monitor tracks against the training baseline.
DRIFT_FEATURES = (
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "prediction_prob",
)

DEFAULT_DECISION_THRESHOLD = 0.5
