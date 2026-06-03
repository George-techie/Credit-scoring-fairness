"""Retraining entrypoint.

Thin orchestration around the deterministic decision logic in
``credit_scoring.retrain`` and the honest metrics in
``credit_scoring.retrain_eval``. The previous "presentation mode" rail that
overwrote real AUC/accuracy/F1 with synthetic values has been removed --
metrics reported here are the metrics actually achieved. MLflow is optional and
never touched at import time.
"""

import logging
import os
import sqlite3
from datetime import datetime

import joblib
import lightgbm as lgb
import pandas as pd

from credit_scoring.features import prepare_features
from credit_scoring.retrain import make_validation_split, select_champion, should_retrain
from credit_scoring.retrain_eval import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_NAME = "feedback.db"
MODEL_PATH = "model/lgb_model.joblib"

FEEDBACK_QUERY = """
    SELECT AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
           DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, CNT_FAM_MEMBERS,
           EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, FLAG_OWN_CAR,
           CODE_GENDER, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE, NAME_HOUSING_TYPE,
           ground_truth AS TARGET
    FROM feedback_v2
"""


def load_training_frame(db_name: str = DB_NAME) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    try:
        return pd.read_sql(FEEDBACK_QUERY, conn)
    finally:
        conn.close()


def retrain_model(tracking_uri: str | None = None) -> None:
    if not os.path.exists(DB_NAME) or not os.path.exists(MODEL_PATH):
        logging.error("Feedback DB and a baseline model are both required.")
        return

    df = load_training_frame()
    if not should_retrain(len(df)):
        logging.warning("Only %d feedback rows; below retrain threshold.", len(df))
        return

    artefact = joblib.load(MODEL_PATH)
    if isinstance(artefact, dict) and "model" in artefact:
        old_model, feature_names = artefact["model"], artefact.get("feature_names")
    else:
        old_model, feature_names = artefact, None

    y = df["TARGET"]
    X = prepare_features(df.drop(columns=["TARGET"]), feature_names)
    X_train, X_val, y_train, y_val = make_validation_split(X, y)

    old_booster = getattr(old_model, "booster_", old_model)
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
              "seed": 42, "verbose": -1}
    new_booster = lgb.train(
        params, lgb.Dataset(X_train, label=y_train), num_boost_round=10,
        init_model=old_booster, keep_training_booster=True,
    )

    metrics = evaluate_predictions(y_val, new_booster.predict(X_val))
    logging.info("Candidate metrics -> acc=%.4f auc=%.4f f1=%.4f",
                 metrics.accuracy, metrics.roc_auc, metrics.f1)

    incumbent_auc = evaluate_predictions(y_val, old_booster.predict(X_val)).roc_auc
    winner = select_champion(incumbent_auc, metrics.roc_auc)
    if winner == "candidate":
        out = f"lgb_model_v2_{datetime.now():%Y%m%d_%H%M%S}.joblib"
        joblib.dump(new_booster, out)
        logging.info("Promoted candidate -> %s", out)
    else:
        logging.info("Candidate did not beat incumbent; keeping current model.")

    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Credit_Scoring_Retraining")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", metrics.accuracy)
            mlflow.log_metric("roc_auc", metrics.roc_auc)
            mlflow.log_metric("f1_score", metrics.f1)


if __name__ == "__main__":
    retrain_model(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
