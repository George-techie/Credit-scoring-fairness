import time
import random
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import os

# Configuration
MLFLOW_URI = "http://localhost:5000"
DRIFT_EXP = "Credit_Scoring_Drift_Monitoring"
RETRAIN_EXP = "Credit_Scoring_Retraining"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

def get_or_create_experiment(name):
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    return client.create_experiment(name)

def seed_presentation_graphs():
    print("==========================================================")
    print("🚀 INSTANT MLOPS DEMO SEEDER")
    print("   Forging 15 days of historical MLflow metrics in 2 seconds...")
    print("==========================================================\n")
    
    # Create empty dummy model file for artifact logging
    with open("dummy_model.joblib", "w") as f:
        f.write("mock_model_data")
    
    drift_exp_id = get_or_create_experiment(DRIFT_EXP)
    retrain_exp_id = get_or_create_experiment(RETRAIN_EXP)
    
    # We want 15 data points spread exactly 1 day apart
    now = int(time.time() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    
    # 1. Seed Drift History
    print(f"📦 Forging 15 Data Drift runs in MLflow...")
    for i in range(15):
        run_time = now - ((14 - i) * day_ms)
        run = client.create_run(drift_exp_id, start_time=run_time)
        
        # Add a proper Run Name so it shows up beautifully in the UI
        client.set_tag(run.info.run_id, "mlflow.runName", f"batch_scan_{datetime.fromtimestamp(run_time/1000).strftime('%Y%m%d')}")
        
        if i < 7:
            ks_score = random.uniform(0.40, 0.80) 
        elif i < 12:
            ks_score = random.uniform(0.15, 0.35)
        else:
            ks_score = random.uniform(0.01, 0.04) 
            
        # Log parameters to populate the MLflow table view
        client.log_param(run.info.run_id, "feature_count", "12")
        client.log_param(run.info.run_id, "test_type", "Kolmogorov-Smirnov")
        
        client.log_metric(run.info.run_id, "prediction_prob_p_value", ks_score)
        client.set_terminated(run.info.run_id, status="FINISHED", end_time=run_time + 5000)
        
    # 2. Seed Retraining History
    print(f"📦 Forging 15 Continuous Retraining runs in MLflow...")
    for i in range(15):
        run_time = now - ((14 - i) * day_ms)
        run = client.create_run(retrain_exp_id, start_time=run_time)
        
        client.set_tag(run.info.run_id, "mlflow.runName", f"retrain_run_{datetime.fromtimestamp(run_time/1000).strftime('%Y%m%d')}")
        
        if i == 0:
            auc = 0.58
        elif i < 11:
            auc = 0.58 + (i * 0.015) + random.uniform(-0.01, 0.01)
        elif i < 13:
            auc = 0.61 + random.uniform(-0.02, 0.02)
        else:
            auc = 0.76 + random.uniform(0.0, 0.03)
            
        # Log the EXACT same params that retrain_pipeline.py logs!
        client.log_param(run.info.run_id, "objective", "binary")
        client.log_param(run.info.run_id, "metric", "auc")
        client.log_param(run.info.run_id, "learning_rate", "0.05")
        client.log_param(run.info.run_id, "num_fine_tuning_samples", str(int(100 + (i*15))))
        
        client.log_metric(run.info.run_id, "accuracy", auc - 0.05)
        client.log_metric(run.info.run_id, "roc_auc", auc)
        client.log_metric(run.info.run_id, "f1_score", auc - 0.08)
        
        # Log dummy artifact so the UI shows standard files
        client.log_artifact(run.info.run_id, "dummy_model.joblib")
        
        client.set_terminated(run.info.run_id, status="FINISHED", end_time=run_time + 10000)
        
    # Cleanup
    if os.path.exists("dummy_model.joblib"):
        os.remove("dummy_model.joblib")
        
    print("\n✅ SEEDING COMPLETE!")
    print("   Refresh your MLflow UI dashboard at: http://localhost:5000")
    print("   All forged metrics, parameters, and tags are correctly embedded!")

if __name__ == "__main__":
    seed_presentation_graphs()
