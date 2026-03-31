import pandas as pd
import sqlite3
import mlflow
import os
from scipy.stats import ks_2samp
from datetime import datetime
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths and Config
BASELINE_DATA_PATH = "baseline_data.csv" # Provided training dataset features to benchmark against
DB_NAME = "feedback.db"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Set up MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Credit_Scoring_Drift_Monitoring")

def monitor_drift():
    logging.info("Starting drift monitoring test...")
    
    # Ensure baseline data exists to compare new feedback against
    if not os.path.exists(BASELINE_DATA_PATH):
        logging.warning(f"Baseline data not found at {BASELINE_DATA_PATH}. Creating a dummy baseline for demonstration purposes.")
        # Only in case you don't have baseline available for immediate testing
        dummy = pd.DataFrame({
            'EXT_SOURCE_1': [0.5, 0.4, 0.6, 0.5, 0.2], 
            'EXT_SOURCE_2': [0.5, 0.4, 0.6, 0.5, 0.2], 
            'EXT_SOURCE_3': [0.5, 0.4, 0.6, 0.5, 0.2], 
            'DAYS_BIRTH': [-15000, -10000, -20000, -14000, -12000],
            'prediction_prob': [0.1, 0.2, 0.4, 0.05, 0.8]
        })
        dummy.to_csv(BASELINE_DATA_PATH, index=False)
        
    if not os.path.exists(DB_NAME):
        logging.warning("No feedback database found. Skipping drift detection (generate some /feedback requests first).")
        return

    # Load baseline
    baseline_df = pd.read_csv(BASELINE_DATA_PATH)
    
    # Load newly collected ground truth / feedback data
    conn = sqlite3.connect(DB_NAME)
    try:
        feedback_df = pd.read_sql("SELECT * FROM feedback_v2", conn)
    except Exception as e:
        logging.error(f"Failed to read from DB (has /feedback been called yet?): {e}")
        conn.close()
        return
    finally:
        conn.close()

    if feedback_df.empty:
        logging.info("No feedback data available for comparison.")
        return
        
    logging.info(f"Loaded {len(feedback_df)} new records from feedback DB for drift analysis.")

    # Select numerical features for KS Test
    features_to_monitor = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'prediction_prob']
    
    with mlflow.start_run(run_name=f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        for feature in features_to_monitor:
            if feature in baseline_df.columns and feature in feedback_df.columns:
                # Perform Kolmogorov-Smirnov test for goodness of fit
                # Null hypothesis: identical distributions
                stat, p_value = ks_2samp(baseline_df[feature].dropna(), feedback_df[feature].dropna())
                
                # Log metrics to MLflow
                mlflow.log_metric(f"{feature}_ks_stat", stat)
                mlflow.log_metric(f"{feature}_p_value", p_value)
                
                # Check for significant drift
                if p_value < 0.05:
                    logging.warning(f"Significant drift detected in feature: {feature} (KS-stat: {stat:.4f}, p-value: {p_value:.4f})")
                else:
                    logging.info(f"No significant drift in feature: {feature} (KS-stat: {stat:.4f}, p-value: {p_value:.4f})")

        mlflow.log_metric("total_feedback_samples_checked", len(feedback_df))
        logging.info("Drift monitoring completed and tracked to MLflow.")

if __name__ == "__main__":
    monitor_drift()
