# flask_app.py
# Flask REST API for Credit Scoring Fairness (Home Credit Dataset)
# SDG 8 (Decent Work) · SDG 10 (Reduced Inequalities)
# Run: python flask_app.py

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/lgb_model.joblib")
model      = None
features   = None

def load_model():
    global model, features
    if os.path.exists(MODEL_PATH):
        artefact = joblib.load(MODEL_PATH)
        model    = artefact["model"]
        features = artefact["feature_names"]
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}. Run the training notebook first.")

load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/model-info", methods=["GET"])
def model_info():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "model_type":    type(model).__name__,
        "num_features":  len(features),
        "dataset":       "Home Credit Default Risk",
        "sdg_alignment": [
            "SDG 8 - Decent Work and Economic Growth",
            "SDG 10 - Reduced Inequalities"
        ],
        "protected_attributes": [
            "CODE_GENDER",
            "NAME_EDUCATION_TYPE",
            "NAME_INCOME_TYPE",
            "NAME_HOUSING_TYPE",
        ],
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Extract demographic fields for fairness logging (not used in prediction)
    demographic_info = {
        "CODE_GENDER":         data.pop("CODE_GENDER", None),
        "NAME_EDUCATION_TYPE": data.pop("NAME_EDUCATION_TYPE", None),
        "NAME_INCOME_TYPE":    data.pop("NAME_INCOME_TYPE", None),
        "NAME_HOUSING_TYPE":   data.pop("NAME_HOUSING_TYPE", None),
    }

    try:
        df   = pd.DataFrame([data])
        df   = df.reindex(columns=features, fill_value=0)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0, 1]

        return jsonify({
            "default_prediction":  int(pred),
            "default_probability": round(float(prob), 4),
            "risk_label":          "High Risk" if pred == 1 else "Low Risk",
            "demographic_context": demographic_info,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
