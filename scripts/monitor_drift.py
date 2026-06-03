"""Drift-monitoring entrypoint.

Thin orchestration around ``credit_scoring.drift.detect_drift``: load the
baseline and the accumulated feedback, run the KS comparison, log results to
MLflow if a tracking server is configured. All statistics live in the package
and are unit-tested; nothing here runs at import time.
"""

import logging
import os
import sqlite3

import pandas as pd

from credit_scoring.drift import detect_drift
from credit_scoring.schema import DRIFT_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASELINE_DATA_PATH = "baseline_data.csv"
DB_NAME = "feedback.db"


def load_feedback(db_name: str = DB_NAME) -> pd.DataFrame:
    if not os.path.exists(db_name):
        return pd.DataFrame()
    conn = sqlite3.connect(db_name)
    try:
        return pd.read_sql("SELECT * FROM feedback_v2", conn)
    finally:
        conn.close()


def run(tracking_uri: str | None = None) -> None:
    if not os.path.exists(BASELINE_DATA_PATH):
        logging.warning("Baseline data not found at %s.", BASELINE_DATA_PATH)
        return
    baseline = pd.read_csv(BASELINE_DATA_PATH)
    feedback = load_feedback()
    if feedback.empty:
        logging.info("No feedback data available for comparison.")
        return

    report = detect_drift(baseline, feedback, features=DRIFT_FEATURES)
    for r in report.results:
        level = logging.WARNING if r.drift_detected else logging.INFO
        logging.log(level, "%s: KS=%.4f p=%.4f drift=%s",
                    r.feature, r.ks_statistic, r.p_value, r.drift_detected)

    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Credit_Scoring_Drift_Monitoring")
        with mlflow.start_run():
            for r in report.results:
                mlflow.log_metric(f"{r.feature}_ks_stat", r.ks_statistic)
                mlflow.log_metric(f"{r.feature}_p_value", r.p_value)
            mlflow.log_metric("total_feedback_samples_checked", report.n_current)


if __name__ == "__main__":
    run(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
