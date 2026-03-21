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
├── flask_app.py                    # Flask REST API for deployment
├── streamlit_app.py                # Streamlit UI for deployment
├── requirements.txt                # Python dependencies
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

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the Flask API:
```bash
python flask_app.py
# API available at http://localhost:5000
```

Run the Streamlit UI:
```bash
streamlit run streamlit_app.py
# UI available at http://localhost:8501
```

## Contributors
- George-techie
- rh0se
- bixzare
- WanjaWhoopie
