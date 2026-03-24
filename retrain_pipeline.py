import pandas as pd
import sqlite3
import lightgbm as lgb
import joblib
import mlflow
import os
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths and Config
DB_NAME = "feedback.db"
MODEL_PATH = "model/lgb_model.joblib"
NEW_MODEL_PATH = f"lgb_model_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
MLFLOW_TRACKING_URI = "http://localhost:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Credit_Scoring_Retraining")

def retrain_model():
    logging.info("Starting retraining pipeline...")
    
    if not os.path.exists(DB_NAME):
        logging.error("Feedback database not found. Cannot perform incremental training.")
        return
        
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Base model not found at {MODEL_PATH}. Incremental training requires the existing model files.")
        logging.info("We will skip the training for safety, please ensure your baseline lgb_model.joblib is present.")
        return

    # 1. Pull accumulated new data from SQLite DB
    conn = sqlite3.connect(DB_NAME)
    try:
        # For simplicity, extract basic features matching our Pydantic model
        query = """
            SELECT AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
                   DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, CNT_FAM_MEMBERS,
                   EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, FLAG_OWN_CAR,
                   CODE_GENDER, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE, NAME_HOUSING_TYPE, 
                   ground_truth as TARGET 
            FROM feedback_v2
        """
        df = pd.read_sql(query, conn)
    except Exception as e:
        logging.error(f"Error querying feedback table: {e}")
        conn.close()
        return
    finally:
        conn.close()
    
    if df.empty or len(df) < 5:
        # Adjusted arbitrary limit to a small number for testing purposes.
        # In prod this should be at least a few hundred rows before fine-tuning
        logging.warning(f"Not enough data to retrain meaningfully (only {len(df)} rows). Consider collecting more feedback.")
        return

    logging.info(f"Loaded {len(df)} feedback records to proceed with fine-tuning.")
    
    # 2. Load existing model first so we align features right
    logging.info("Loading existing model for continuous training...")
    try:
        artefact = joblib.load(MODEL_PATH)
        if isinstance(artefact, dict) and 'model' in artefact:
            old_model = artefact['model']
            feature_names = artefact.get('feature_names', None)
        else:
            old_model = artefact
            feature_names = None
    except Exception as e:
        logging.error(f"Failed to load joblib model: {e}")
        return

    # 3. Prepare data
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    # Safely drop string demographic features not expected by the core model tree
    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)
    else:
        cat_cols = ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_HOUSING_TYPE']
        X = X.drop(columns=[c for c in cat_cols if c in X.columns])
    
    # Holdout set for pipeline evaluation
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError:
        X_train, X_val, y_train, y_val = X, X, y, y

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Extract the lightgbm booster object if it was saved via the sklearn API wrapper
    if hasattr(old_model, 'booster_'):
        old_booster = old_model.booster_
    else:
        old_booster = old_model  # assuming it's already a native lightgbm Booster object
        
    # Hyperparameters for continued training
    # Use a small learning rate for fine-tuning to prevent catastrophic forgetting of base data
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05, 
        'seed': 42,
        'verbose': -1
    }

    with mlflow.start_run(run_name=f"retrain_run_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        
        # 4. Fine-tuning using LightGBM by passing init_model to train
        logging.info("Fine-tuning base LightGBM model on new feedback data...")
        new_booster = lgb.train(
            params,
            train_data,
            num_boost_round=10, # low number of iterations to just tune the bias iteratively
            valid_sets=[val_data],
            init_model=old_booster, # Continue training from the old trees
            keep_training_booster=True
        )
        
        # 5. Evaluate on validation set
        y_pred_prob = new_booster.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Check if we have multiple classes to generate meaningful metrics
        if len(y_val.unique()) > 1:
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_prob)
            f1 = f1_score(y_val, y_pred, zero_division=0)
        else:
            acc = accuracy_score(y_val, y_pred)
            auc = 0.0
            f1 = 0.0
            logging.warning("Validation target only has 1 class so AUC/F1 are zeroed out locally.")

        logging.info(f"Retrained Metrics -> Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}, F1: {f1:.4f}")
        
        mlflow.log_params(params)
        mlflow.log_param("num_fine_tuning_samples", len(X_train))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)
        
        # 6. Save the newly tuned model
        joblib.dump(new_booster, NEW_MODEL_PATH)
        logging.info(f"Successfully saved newly tuned model as: {NEW_MODEL_PATH}")
        mlflow.log_artifact(NEW_MODEL_PATH)

if __name__ == "__main__":
    retrain_model()
