import joblib
import sqlite3
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Database for feedback
DB_NAME = "feedback.db"
MODEL_PATH = "model/lgb_model.joblib"

# Global model variable
model = None
feature_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names
    # Load model on startup
    if os.path.exists(MODEL_PATH):
        try:
            artefact = joblib.load(MODEL_PATH)
            if isinstance(artefact, dict) and 'model' in artefact:
                model = artefact['model']
                feature_names = artefact.get('feature_names', None)
            else:
                model = artefact
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")

    # Initialize SQLite DB for feedback
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            AMT_INCOME_TOTAL REAL,
            AMT_CREDIT REAL,
            AMT_ANNUITY REAL,
            AMT_GOODS_PRICE REAL,
            DAYS_BIRTH REAL,
            DAYS_EMPLOYED REAL,
            CNT_CHILDREN REAL,
            CNT_FAM_MEMBERS REAL,
            EXT_SOURCE_1 REAL,
            EXT_SOURCE_2 REAL,
            EXT_SOURCE_3 REAL,
            FLAG_OWN_CAR INTEGER,
            CODE_GENDER TEXT,
            NAME_EDUCATION_TYPE TEXT,
            NAME_INCOME_TYPE TEXT,
            NAME_HOUSING_TYPE TEXT,
            prediction_prob REAL,
            ground_truth INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    yield
    print("Shutting down model serving application...")

app = FastAPI(title="Credit Scoring API", lifespan=lifespan)

class BorrowerFeatures(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    CNT_CHILDREN: float
    CNT_FAM_MEMBERS: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    FLAG_OWN_CAR: int
    CODE_GENDER: str
    NAME_EDUCATION_TYPE: str
    NAME_INCOME_TYPE: str
    NAME_HOUSING_TYPE: str
    
class FeedbackData(BorrowerFeatures):
    prediction_prob: float
    ground_truth: int  # 0 or 1

@app.post("/predict")
def predict_default(features: BorrowerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    # Convert input to DataFrame (Note: Ensure the columns match your model's expected input features)
    df = pd.DataFrame([features.dict()])
    
    # Isolate training features mathematically from demographic tracking strings (e.g. CODE_GENDER) 
    if feature_names is not None:
        df = df.reindex(columns=feature_names, fill_value=0)
    else:
        cols_to_drop = ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_HOUSING_TYPE']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Predict probability logically for fairness-constrained ensembles
    try:
        import numpy as np
        # Fairlearn / Fairness-constrained models use multiple predictors and weights
        if hasattr(model, 'predictors_') and hasattr(model, 'weights_'):
            weights = np.array(model.weights_)
            weights = weights / weights.sum()
            prob = 0.0
            for w, predictor in zip(weights, model.predictors_):
                if hasattr(predictor, 'predict_proba'):
                    prob += w * predictor.predict_proba(df)[0, 1]
                else:
                    prob += w * float(predictor.predict(df)[0])
        elif hasattr(model, 'predict_proba'):
            prob = model.predict_proba(df)[0][1]
        else:
            prob = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
    prediction = int(prob > 0.5)
    return {
        "default_probability": float(prob),
        "default_prediction": prediction
    }

@app.post("/feedback")
def receive_feedback(feedback: FeedbackData):
    """
    Simulates receiving the ground truth data (e.g. loan defaulted months later).
    Appends this data to the local SQLite database.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        # Use pandas to quickly append row, excluding the auto-increment id and timestamp (let DB handle that implicitly via schema)
        df_to_save = pd.DataFrame([feedback.dict()])
        df_to_save.to_sql("feedback_v2", conn, if_exists="append", index=False)
        conn.close()
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")
