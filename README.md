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
├── simulate_time_travel.py         # Presentation macro script to simulate Concept Drift
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

This project goes beyond a static model by implementing a fully local, continuous training MLOps pipeline.

## System Architecture

The backend consists of four core components:

1. **FastAPI Serving System (`fastapi_app.py`)**: Hosts the active LightGBM model for real-time predictions and provides a `/feedback` endpoint to securely collect downstream ground-truth labels directly from the Streamlit UI.
2. **Drift Monitoring (`monitor_drift.py`)**: Compares newly collected feedback data against a baseline training distribution using Kolmogorov-Smirnov (KS) tests, identifying feature and output probability drift.
3. **Continuous Retraining Pipeline (`retrain_pipeline.py`)**: Extracts accumulated feedback from the local database and fine-tunes the existing gradient boosting tree parameters without catastrophic forgetting.
4. **Interactive MLOps Dashboard**: Built natively into Streamlit (`tab3`), providing system architects with a real-time health-check of the active model via MLflow.

All monitoring metrics, hyperparameters, and newly saved model artifacts are seamlessly tracked using local **MLflow**.

## MLOps Control Center (Streamlit Native)

The Streamlit UI features a dedicated **MLOps Control Center** tab that acts as a central hub for pipeline management. It leverages:
- **Dynamic MLflow Connection**: The tab utilizes the `mlflow` Python client API to securely query the background tracking server (`localhost:5000`) and fetch the latest experiment histories (timestamps, metrics) directly into the frontend interface.
- **Visual Health Dashboards**: Live tracking data is parsed into native Streamlit interactive line charts visualizing the two core MLOps benchmarks: (1) **Data Concept Drift** via tracking the Kolmogorov-Smirnov statistical p-value, and (2) **Continuous Retraining Performance** via tracking the ROC-AUC score across subsequent fine-tuning iterations.
- **Interactive Subprocessor Commands**: Administrators can seamlessly execute the background Python pipelines (`monitor_drift.py` and `retrain_pipeline.py`) by clicking native UI buttons. Instead of blocking the Streamlit thread or requiring manual terminal commands, Streamlit safely encapsulates these scripts via `subprocess.run()`, catches their outputs, updates the MLflow charts natively, and immediately hot-reloads the visualizations upon success.

## Presentation & Interactive Features

To effectively demonstrate the robustness of the system to university panels or banking executives, this project includes several highly polished presentation tools:

### 1. Explainable AI (SHAP Waterfall)
The system leverages an active SHAP (SHapley Additive exPlanations) engine natively inside the FastAPI backend. During the Streamlit Assessment phase, FastAPI computationally extracts the exact feature contribution values from the Fairlearn ensemble, generates an in-memory `matplotlib` Waterfall Graph, translates the technical data science variables into plain English, and securely streams it back to Streamlit as a Base64 image payload. This physically shows your audience *why* the AI made the exact decision it did.

### 2. Generative AI Chatbot (Groq)
The Streamlit UI features a dedicated AI Assistant powered by Groq. This integrates a lightning-fast Large Language Model (LLM) loaded with domain-specific knowledge regarding credit scoring, fairness metrics, and macroeconomics. It allows loan officers to ask natural language questions regarding application assessments without ever leaving the secure platform.

### 3. Instant Demo Seeder (`seed_demo.py`) 🚀 Recommended for Live Demos
When presenting live, you want your MLOps control center to look populated and professional instantly. Connect your terminal and run:
```bash
python seed_demo.py
```
This script uses the `MlflowClient` API to aggressively forge 15 days of perfectly orchestrated historical metrics (both stable and drifting) directly into your MLflow backend in under 2 seconds. This guarantees your Streamlit dashboard displays a beautiful, predictable narrative arc reliably every single time.

### 4. Deep MLOps History Builder (`simulate_mlops_history.py`)
To mathematically prove the Integration Pipeline works end-to-end, you can run a heavy script that loops through organic data generation, drift detection, and automated LightGBM retraining 15 consecutive times. This script performs real Machine Learning math to calculate actual drift against raw databases. *(Note: Takes ~3 minutes to run).*
```bash
python simulate_mlops_history.py
```

### 5. Live Time Travel Simulator (`simulate_time_travel.py`)
You can forcefully simulate literal "Concept Drift" during your presentation by pumping 120 severely stressed synthetic applicant records directly into the SQLite feedback database:
```bash
python simulate_time_travel.py
```
Once executed, navigate to the **MLOps Control Center** in Streamlit and click **"Run Drift Monitor Batch Scan"**. The MLflow dashboard will instantly update, alerting the user to a severe statistical target drift (P-value < 0.05), effectively prompting an immediate Model Retraining phase.
