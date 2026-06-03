# Architecture

The scoring, feature, drift, retraining and explanation logic lives in an
importable, side-effect-free package (`src/credit_scoring`). The HTTP layer
(`credit_scoring.api`) is a thin FastAPI app split into per-resource routers,
and the operational entrypoints (drift monitoring, retraining) are scripts that
import the same package.

```
src/credit_scoring/
  schema.py        Feature contract (predictive vs demographic columns, scales)
  features.py      Request -> model feature alignment (no value fabrication)
  inference.py     Ensemble probability aggregation + decision threshold
  drift.py         Two-sample KS drift detection (pure; returns a DriftReport)
  retrain_eval.py  Honest accuracy / ROC-AUC / F1
  retrain.py       Retrain gating, leakage-free split, champion selection
  explain.py       SHAP feature-label alignment
  api/
    app.py             FastAPI factory + model loading
    schemas.py         Request/response models
    routers/           /predict, /feedback, /explain

scripts/             Operational entrypoints (drift monitor, retraining)
tests/unit/          Module-level behavioral tests
tests/integration/   API tests via FastAPI TestClient
```

## Data flow

A request hits `/predict`, is aligned to the model's feature contract by
`prepare_features`, scored by `predict_default_proba`, and thresholded by
`decide`. Ground-truth outcomes arrive at `/feedback` and accumulate in SQLite;
`scripts/monitor_drift.py` compares them to the training baseline, and
`scripts/retrain_pipeline.py` retrains and promotes a new model only when it
beats the incumbent.
