"""/explain — SHAP waterfall explanation for a single decision."""

import base64
import io

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ...explain import translate_feature_names
from ...features import MissingFeatureError, prepare_features
from .. import state
from ..schemas import BorrowerFeatures

router = APIRouter()


@router.post("/explain")
def explain_decision(features: BorrowerFeatures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    raw = pd.DataFrame([features.model_dump()])
    try:
        X = prepare_features(raw, state.feature_names)
    except MissingFeatureError as e:
        raise HTTPException(status_code=422, detail=str(e))

    model = state.model
    explainer_model = model
    if hasattr(model, "predictors_") and hasattr(model, "weights_"):
        best_idx = int(np.argmax(np.asarray(model.weights_)))
        explainer_model = model.predictors_[best_idx]

    try:
        explainer = shap.TreeExplainer(explainer_model)
        shap_values = explainer(X.values)
        sv = shap_values[:, :, 1][0] if len(shap_values.shape) == 3 else shap_values[0]
        sv.feature_names = np.array(translate_feature_names(list(X.columns)))

        plt.clf()
        shap.plots.waterfall(sv, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6.5)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return {"status": "success", "shap_base64": base64.b64encode(buf.read()).decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {e}")
