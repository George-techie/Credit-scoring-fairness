"""Credit-scoring fairness service: core library.

Deterministic, side-effect-free building blocks shared by the FastAPI app, the
drift monitor and the retraining pipeline.
"""

from __future__ import annotations

from . import schema
from .drift import DRIFT_ALPHA, DriftReport, FeatureDrift, detect_drift
from .explain import UI_NAME_MAP, translate_feature_names
from .features import MissingFeatureError, drop_demographics, prepare_features
from .inference import decide, predict_default_proba
from .retrain import (
    MIN_AUC_IMPROVEMENT,
    MIN_RETRAIN_ROWS,
    make_validation_split,
    select_champion,
    should_retrain,
)
from .retrain_eval import EvalMetrics, evaluate_predictions

__all__ = [
    "schema",
    "prepare_features",
    "drop_demographics",
    "MissingFeatureError",
    "predict_default_proba",
    "decide",
    "detect_drift",
    "DriftReport",
    "FeatureDrift",
    "DRIFT_ALPHA",
    "evaluate_predictions",
    "EvalMetrics",
    "should_retrain",
    "make_validation_split",
    "select_champion",
    "MIN_RETRAIN_ROWS",
    "MIN_AUC_IMPROVEMENT",
    "translate_feature_names",
    "UI_NAME_MAP",
]
