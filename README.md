# Credit Scoring for Financial Inclusion

**SDG 8** — Decent Work and Economic Growth  
**SDG 10** — Reduced Inequalities

## Overview
This project builds a fair, interpretable credit default predictor for the *unbanked* — individuals with thin or no formal credit files. It uses alternative data (telecom/utility scores, socio-economic indicators) alongside standard financial features, and evaluates fairness across demographic groups.

## Dataset
Download from Kaggle: https://www.kaggle.com/c/home-credit-default-risk/data

Files needed — place them in the root folder before running the notebook:
- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `previous_application.csv`

## Project Structure
```
Credit-scoring-fairness/
├── credit_scoring_fairness.ipynb   # Main training and fairness analysis notebook
├── fastapi_app.py                  # FastAPI REST API for deployment & feedback
├── streamlit_app.py                # Streamlit UI with Groq LLM and MLflow health integration
├── start_services.py               # Single entrypoint launcher for all 3 web systems
├── monitor_drift.py                # Kolmogorov-Smirnov (KS) test for data drift
├── retrain_pipeline.py             # Automates fine-tuning LightGBM with MLflow metrics
├── requirements.txt                # Frontend dependencies
├── requirements_backend.txt        # Backend and MLOps dependencies
└── model/
    └── lgb_model.joblib            # Saved model (generated after running notebook)
```

## Fairness Analysis
The model is evaluated across four protected attributes:

| Attribute | Column | SDG Link |
|---|---|---|
| Gender | `CODE_GENDER` | SDG 10 — Reduced Inequalities |
| Education Level | `NAME_EDUCATION_TYPE` | SDG 10 — Equal access to opportunity |
| Income / Employment Type | `NAME_INCOME_TYPE` | SDG 8 — Decent Work |
| Housing Type | `NAME_HOUSING_TYPE` | SDG 10 — Socioeconomic exclusion |

## Setup & Run

Install all dependencies required for the frontend and backend:
```bash
pip install -r requirements.txt
pip install -r requirements_backend.txt
```

Launch the entire suite (FastAPI Backend, MLflow Server, and Streamlit UI) safely through the python orchestrator. This runs everything cleanly in a single console:
```bash
python start_services.py
```
This single command spins up:
- **Streamlit Application**: http://localhost:8501
- **FastAPI Swagger Docs**:  http://localhost:8000/docs
- **MLflow Dashboard**:      http://localhost:5000

## Contributors
- George-techie
- rh0se
- bixzare
- WanjaWhoopie

---

# Backend & MLOps Documentation

This section details the continuous training pipeline and MLOps backend for the Fairness-Aware Credit Scoring system. The architecture is designed for local development and tracking, utilizing standard offline tools rather than specific cloud provider SDKs.

## System Architecture

The backend consists of three core components:

1. **FastAPI Serving System (`fastapi_app.py`)**: Hosts the active LightGBM model for real-time predictions and provides a `/feedback` endpoint to securely collect downstream ground-truth labels directly from the Streamlit UI.
2. **Drift Monitoring (`monitor_drift.py`)**: Compares newly collected feedback data against a baseline training distribution using Kolmogorov-Smirnov (KS) tests, identifying feature and output probability drift.
3. **Continuous Retraining Pipeline (`retrain_pipeline.py`)**: Extracts accumulated feedback from the local database and fine-tunes the existing gradient boosting tree parameters without catastrophic forgetting.

All monitoring metrics, hyperparameters, and newly saved model artifacts are seamlessly tracked using a local **MLflow** instance.

## Essential Artifacts

Before starting the system, ensure the following artifacts are placed in this directory:
- `model/lgb_model.joblib`: Your initial trained LightGBM model (either an active `lgb.Booster` or the Scikit-Learn Wrapper `LGBMClassifier`).
- `baseline_data.csv`: A sample or complete set of the dataset you used to originally train the model. The drift monitor needs this to compare against new incoming feedback.

## Advanced Usage & Execution

While `python start_services.py` handles the active web applications effortlessly, you can evaluate model degradation and trigger the fine-tuning pipelines securely via standalone scripts:

### 1. Run Drift Monitoring

Whenever you want to check if the new data arriving in your feedback database significantly drifts from your baseline distributions, execute the script:

```bash
python monitor_drift.py
```

The script runs a KS-test on features and logs the p-values and test statistics directly to MLflow.

### 2. Trigger Incremental Model Retraining

If significant drift is detected or a large batch of feedback has been successfully collected via Streamlit's "Simulate Outcome", you can fine-tune the model:

```bash
python retrain_pipeline.py
```

This will:
- Load data from the SQLite feedback database (populated directly by Streamlit).
- Check if enough robust samples exist.
- Load `model/lgb_model.joblib`.
- Perform additional boosting iterations on the new data without catastrophic forgetting of the initial weights.
- Store the new model locally and log evaluation metrics to MLflow.
