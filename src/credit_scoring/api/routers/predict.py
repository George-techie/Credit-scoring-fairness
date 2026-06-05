"""/predict — score a borrower's probability of default."""

import pandas as pd
from fastapi import APIRouter, HTTPException

from ...features import MissingFeatureError, prepare_features
from ...inference import decide, predict_default_proba
from .. import state
from ..schemas import BorrowerFeatures

router = APIRouter()


@router.post("/predict")
def predict_default(features: BorrowerFeatures):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    raw = pd.DataFrame([features.model_dump()])
    try:
        X = prepare_features(raw, state.feature_names)
    except MissingFeatureError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    try:
        prob = float(predict_default_proba(state.model, X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e
    return {"default_probability": prob, "default_prediction": int(decide([prob])[0])}
