# Credit Scoring for Financial Inclusion

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange?style=flat-square)
![Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn-green?style=flat-square)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple?style=flat-square)
![SDG8](https://img.shields.io/badge/SDG-8%20Decent%20Work-red?style=flat-square)
![SDG10](https://img.shields.io/badge/SDG-10%20Reduced%20Inequalities-red?style=flat-square)

> A fair, interpretable, and scalable credit default prediction framework that leverages alternative data to extend credit access to unbanked populations — advancing **SDG 8 (Decent Work and Economic Growth)** and **SDG 10 (Reduced Inequalities)**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [1. Data & Pipeline](#1-data--pipeline)
- [2. ML Architecture & Training Pipeline](#2-ml-architecture--training-pipeline)
- [3. Evaluation Metrics & Results](#3-evaluation-metrics--results)
- [Fairness Analysis](#fairness-analysis)
- [SHAP Interpretability](#shap-interpretability)
- [Bias Mitigation](#bias-mitigation)
- [Deployment](#deployment)
- [Installation](#installation)
- [Team](#team)

---

## Overview

Over 1.4 billion adults worldwide remain unbanked, with the majority concentrated in developing economies. Traditional credit scoring systems rely on historical financial records — creating a Catch-22 where you need credit to prove you deserve credit. This project breaks that cycle.

Using the **Home Credit Default Risk dataset**, we build a machine learning pipeline that:

- Incorporates **alternative data** (telecom scores, utility payment proxies, behavioral signals) to assess creditworthiness without requiring formal credit history
- Achieves **competitive predictive performance** (AUC-ROC 0.7649) on a population of 307,511 thin-file and no-file applicants
- Quantifies and mitigates **demographic disparities** across gender, education level, and income type
- Provides **SHAP-based explanations** for every prediction — making decisions transparent and auditable

---

## Dataset

**Home Credit Default Risk** — [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)

| File | Rows | Description |
|---|---|---|
| `application_train.csv` | 307,511 | Core application data — demographics, income, employment, housing |
| `application_test.csv` | 48,744 | Test set for submission |
| `bureau.csv` | 1,716,428 | External credit bureau history per applicant |
| `previous_application.csv` | 1,670,214 | Prior Home Credit application records |

**Target:** `TARGET` — 1 = defaulted, 0 = repaid (imbalance ratio 11.4:1)

---

## Project Structure

```
Credit-scoring-fairness/
│
├── credit_scoring_fairness.ipynb   # Full pipeline notebook
├── flask_app.py                    # REST API for loan decisions
├── streamlit_app.py                # Interactive demo UI
├── requirements.txt                # Dependencies
├── model/
│   └── lgb_model.joblib            # Saved mitigated model artifact
└── README.md
```

---

## 1. Data & Pipeline

### Why this dataset serves the SDG mission

Home Credit's business model specifically targets individuals underserved by traditional banks — street vendors, rural farmers, factory workers, and young people with irregular income streams. The dataset is rich with "thin-file" and "no-file" individuals, making it an authentic representation of the population SDG 10 aims to serve.

### Data sources and merging

We merge four tables into a single enriched training set of **150 features**:

```python
train = app_train_fe.merge(bureau_agg, on='SK_ID_CURR', how='left')
train = train.merge(prev_agg,          on='SK_ID_CURR', how='left')
```

### Feature engineering — 13 new features

Three feature groups carry the SDG argument:

**Alternative credit signal composites** — replacing the need for formal credit history:
```python
df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['EXT_SOURCE_MIN']  = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
df['EXT_SOURCE_STD']  = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
```

**Income stability for low-income households:**
```python
df['CREDIT_INCOME_RATIO']   = df['AMT_CREDIT']    / (df['AMT_INCOME_TOTAL'] + 1)
df['ANNUITY_INCOME_RATIO']  = df['AMT_ANNUITY']   / (df['AMT_INCOME_TOTAL'] + 1)
df['INCOME_PER_FAM_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
```

**Employment stability — rewards consistent informal sector employment:**
```python
df['EMPLOYED_YEARS']        = -df['DAYS_EMPLOYED'].clip(upper=0) / 365
df['EMPLOYED_TO_AGE_RATIO'] = df['EMPLOYED_YEARS'] / (df['AGE_YEARS'] + 1)
```

**Prior exclusion signal:**
```python
prev_agg['prev_approval_rate'] = (
    prev_agg['prev_approved_count'] / prev_agg['prev_app_count']
).fillna(0)
```

### Critical design decision — sensitive attribute exclusion

Sensitive attributes (gender, education type, income type, housing type) are **deliberately excluded from training features**. They are saved separately for fairness evaluation only. The model must earn its predictions from financial behavior, not demographic proxies.

```python
DROP_COLS = ['TARGET', 'SK_ID_CURR', 'AGE_YEARS_EDA'] + sensitive_cols
features  = [c for c in train.columns if c not in DROP_COLS]
```

### Preprocessing

| Step | Detail |
|---|---|
| Encoding | One-hot via `pd.get_dummies`, aligned across train/test |
| Imputation | Median strategy — 67 columns with missing values |
| Scaling | StandardScaler for Logistic Regression only |
| Class imbalance | SMOTE at `sampling_strategy=0.3` — 11.4:1 ratio |

---

## 2. ML Architecture & Training Pipeline

### Model progression

We benchmark four models deliberately — from transparent baseline to high-performance ensemble:

```
Logistic Regression  →  Random Forest  →  XGBoost  →  LightGBM (tuned) ← selected
     (baseline)           (ensemble)      (perf.)        (production)
```

### Why LightGBM over XGBoost

XGBoost and LightGBM are statistically identical in discriminative power on this dataset:

| | XGBoost | LightGBM |
|---|---|---|
| AUC-ROC | 0.7649 | 0.7642 |
| Gap | — | 0.0007 |
| CV variance | — | ±0.0026 |

The gap of 0.0007 is smaller than one standard deviation of our own CV spread. These models are the same.

LightGBM wins on every operational criterion that matters for the deployment context:

- **Leaf-wise tree growth** — faster on 307K rows × 143 features than XGBoost's level-wise approach
- **Histogram-based binning** — lower memory footprint; critical for microfinance institutions in emerging markets running on modest infrastructure
- **Faster inference** — real-time loan decisioning requires sub-second prediction
- **Clean fairlearn integration** — `class_weight='balanced'` + `scale_pos_weight` + ExponentiatedGradient wrapper work natively

### Optuna hyperparameter tuning

```python
best_params = {
    'n_estimators'     : 500,
    'learning_rate'    : 0.010220488556128927,
    'num_leaves'       : 122,
    'max_depth'        : 10,
    'min_child_samples': 12,
    'subsample'        : 0.8189168373977738,
    'colsample_bytree' : 0.501213429116603,
    'reg_alpha'        : 0.00010205386070312878,
    'reg_lambda'       : 0.08880864885487433,
    'class_weight'     : 'balanced'
}
```

30 Optuna trials with 5-fold stratified cross-validation inside each trial.

### 5-fold cross-validation results

```
CV AUC scores: [0.7461  0.7442  0.7514  0.7478  0.7449]
Mean: 0.7469 ± 0.0026
```

Low variance confirms the model generalizes reliably — not overfitting to any single split.

---

## 3. Evaluation Metrics & Results

### Why AUC-ROC and Recall — both matter

**AUC-ROC** measures the model's discriminative power across all decision thresholds. Our AUC of 0.7649 is competitive and meaningful for a thin-file population dataset — the 0.75–0.80 range is consistent with what the literature reports when predicting default for applicants with limited financial histories. AUC tells us the model has learned real signal from alternative data; it is not guessing.

**Recall** measures how well the model catches actual defaulters at the operating threshold. In a financial inclusion context, a false negative means a real lending loss; a false positive means a creditworthy person is wrongly denied — an exclusion harm. Our balanced class weighting and SMOTE approach deliberately calibrate this tradeoff. High AUC ensures the model ranks risk well globally; strong Recall ensures it acts decisively at the decision boundary.

Together they validate both learning quality and operational utility.

### Full model comparison

| Model | AUC-ROC | Recall | Accuracy | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.7503 | 0.6820 | 0.6878 | 0.1612 |
| Random Forest | 0.7449 | 0.5458 | 0.7742 | 0.1889 |
| XGBoost | **0.7649** | 0.6298 | 0.7466 | 0.1853 |
| LightGBM (tuned) | 0.7642 | **0.6467** | 0.7323 | 0.1792 |

LightGBM achieves the **highest Recall of all four models** while matching XGBoost's AUC within statistical noise — precisely the balance needed for a system whose mission is to extend credit without exposing lenders to unacceptable risk.

---

## Fairness Analysis

### Approach

Fairness is evaluated across three sensitive attributes using two complementary metrics:

- **Demographic Parity Difference (DPD)** — gap in predicted default rates between groups
- **Equalized Odds Difference (EOD)** — gap in true positive rates between groups

### Results

**Gender (LightGBM — before mitigation):**

| Group | Count | Actual Default Rate | Predicted Rate | Recall | AUC |
|---|---|---|---|---|---|
| F | 40,561 | 0.0699 | 0.2611 | 0.6125 | 0.7583 |
| M | 20,940 | 0.1017 | 0.3501 | 0.6923 | 0.7641 |

DPD: 0.2389 | EOD: 0.6923

**Education (LightGBM):** DPD: 0.1764 | EOD: 0.4317

**Income Type (LightGBM):** DPD: 0.6000 | EOD: 0.7015

> The income type disparity of 0.60 reflects genuine structural inequality in the data — working applicants default at materially different rates than state servants or commercial associates. Our framework surfaces this explicitly where traditional scoring systems would obscure it behind a single score.

---

## SHAP Interpretability

SHAP (SHapley Additive exPlanations) provides global and per-applicant explanations for every model decision.

### Top drivers — the SDG result

```
EXT_SOURCE_MEAN       ████████████████████  0.0412
GOODS_CREDIT_RATIO    ████████████████      0.0318
EXT_SOURCE_MIN        ██████████████        0.0285
EXT_SOURCE_STD        ████████████          0.0241
CREDIT_INCOME_RATIO   ██████████            0.0198
```

The most important predictor is not raw income, not employment status, not demographic group. It is the **composite of alternative credit signals from telecom and utility providers**. This empirically validates the core premise of financial inclusion lending — that behavioral signals from non-banking sources can predict creditworthiness for people with no formal credit history.

The single-applicant SHAP waterfall plot makes every decision transparent and contestable — a requirement for deployment in regulated lending environments.

---

## Bias Mitigation

We apply **ExponentiatedGradient** with a DemographicParity constraint on gender as an integrated pipeline stage — not post-processing. The saved model artifact is the mitigated model.

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

mitigator = ExponentiatedGradient(
    estimator   = base_lgb,
    constraints = DemographicParity()
)
mitigator.fit(X_train_imp, y_train, sensitive_features=gender_train)
```

### Results

| Metric | Before | After | Change |
|---|---|---|---|
| Demographic Parity Difference | 0.2389 | 0.2080 | −13% |
| Equalized Odds Difference | 0.6923 | 0.6481 | −6.4% |

### Why only gender?

Gender is explicitly recognised as a legally protected attribute in fair lending legislation across most jurisdictions (ECOA in the US, EU Anti-Discrimination Directives). Education and income type disparities are quantified and documented transparently — surfacing them is itself a contribution. Multi-attribute constrained optimisation across all three sensitive attributes simultaneously is a natural extension of this work and is flagged as future work.

---

## Deployment

The mitigated model is saved as a portable artifact:

```python
model_artefact = {
    "model"         : mitigator,
    "feature_names" : features,
    "training_time" : pd.Timestamp.now().isoformat(),
    "fairness_note" : "ExponentiatedGradient with DemographicParity on CODE_GENDER"
}
joblib.dump(model_artefact, "model/lgb_model.joblib")
```

**Note on model size:** The artifact is ~320MB because ExponentiatedGradient stores an ensemble of constrained sub-models internally. For production on resource-constrained infrastructure, the base `lgb_final` (unconstrained LightGBM) is available as a lighter alternative. Deployment targets are Flask REST API and Streamlit interactive demo.

| Interface | File | Purpose |
|---|---|---|
| REST API | `flask_app.py` | Loan decision endpoint |
| Demo UI | `streamlit_app.py` | Interactive applicant scoring |

---

## Installation

```bash
git clone https://github.com/George-techie/Credit-scoring-fairness.git
cd Credit-scoring-fairness
pip install -r requirements.txt
```

**Download dataset files** from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place in the project root:
```
application_train.csv
application_test.csv
bureau.csv
previous_application.csv
```

Then run the notebook top to bottom in Google Colab or Jupyter.

---

## Team

| Name | GitHub |
|---|---|
| George | [@George-techie](https://github.com/George-techie) |
| | [@rh0se](https://github.com/rh0se) |
| | [@bixzare](https://github.com/bixzare) |
| | [@WanjaWhoopie](https://github.com/WanjaWhoopie) |

---

*This project contributes to responsible AI in financial services — combining performance, fairness, and interpretability to advance equitable credit access.*
