# Credit Scoring & Fairness Service

A credit-default scoring service with a fairness-aware model, drift monitoring,
and a feedback-driven retraining loop.

## Architecture

The scoring, feature, drift, retraining and explanation logic lives in an
importable, side-effect-free package; the FastAPI app, the drift monitor and the
retraining script are thin orchestration layers over it.

```
src/credit_scoring/
  schema.py        Feature contract (predictive vs demographic columns, scales)
  features.py      Request -> model feature alignment (no value fabrication)
  inference.py     Ensemble probability aggregation + decision threshold
  drift.py         Two-sample KS drift detection (pure; returns a DriftReport)
  retrain_eval.py  Honest accuracy / ROC-AUC / F1
  retrain.py       Retrain gating, leakage-free split, champion selection
  explain.py       SHAP feature-label alignment

fastapi_app.py      /predict, /feedback, /explain  (imports the package)
monitor_drift.py    Drift entrypoint (MLflow optional, via MLFLOW_TRACKING_URI)
retrain_pipeline.py Retrain entrypoint (MLflow optional)
tests/              71 behavioral tests
```

## Setup

```
pip install -r requirements.txt
pytest            # 71 tests
```

## Design notes

- Modules import without side effects; MLflow is engaged only when a tracking
  URI is supplied at call time, never at import.
- Metrics are computed honestly and are fully reproducible.
- Demographic attributes (CODE_GENDER, education, income, housing type) are
  tracked for fairness auditing but never fed to the model as predictors.
- External bureau scores are normalised to [0, 1] with higher = safer; the
  feature layer never substitutes a placeholder for a missing required feature.
