# Credit Scoring & Fairness Service

A production-style credit-default scoring service that treats fairness as a measured, enforceable property of the system rather than a footnote. It scores a borrower's probability of default, audits those decisions for bias across protected groups, explains each individual decision with SHAP, watches the live input distribution for drift, and retrains itself only when a new model genuinely beats the one already in production.

Built on the Home Credit default dataset. Aligned to **SDG 10 (Reduced Inequalities)** and **SDG 8 (Decent Work and Economic Growth)**.

![CI](https://github.com/George-techie/Credit-scoring-fairness/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)

---

## Why it matters

A credit score is a gate to economic participation: a loan, a home, working capital for a small business. A scoring model that is even slightly biased, once it runs at scale, quietly turns that bias into thousands of denials that fall on the same groups every time.

The problem is sharper in underbanked and emerging markets, where thin credit files and missing bureau data are the norm rather than the exception. Naive engineering makes it worse. If a missing bureau score is silently filled with `0`, and bureau scores run on a `[0, 1]` scale where higher means safer, then every applicant with an incomplete file is scored as if they had the worst possible credit history. That penalty lands on exactly the people financial inclusion is meant to reach.

This service is built around that reality:

- Protected attributes (gender, education, income type, housing type) are tracked and audited, but never fed to the model as predictors.
- Missing required features fail loudly instead of being filled with a value that silently biases the score.
- Fairness is computed on real decisions and reduced to a single regulatory pass/fail (the four-fifths rule), alongside the parity and error-rate gaps that show *where* a model is inequitable.
- Every decision can be explained in plain language, which is what a lender actually needs to deploy a model responsibly and defend it to a regulator.

## What it does

- **Score** a borrower's probability of default and convert it into a lend/decline decision at a configurable threshold. The serving layer is model-agnostic: it works with any scikit-learn-style estimator and natively supports Fairlearn reduction ensembles (the output of `ExponentiatedGradient` fairness mitigation), averaging their sub-predictors by normalised weight.
- **Audit** decisions for group fairness: per-group selection rate, demographic parity difference, disparate-impact ratio with the four-fifths check, and, when outcomes are known, equalized-odds and equal-opportunity differences.
- **Explain** any single decision with a SHAP waterfall, returned as an image and, in the UI, narrated in plain English by an LLM.
- **Monitor** the live feature distribution against the training baseline using a two-sample Kolmogorov–Smirnov test, and flag the features that have drifted.
- **Retrain** from accumulated ground-truth feedback, with a leakage-free validation split and a champion/challenger gate, so a new model ships only if it beats the incumbent by a real AUC margin.

## Design principles

The decisions that make this more than a notebook, each recorded as an [ADR](docs/adr):

- **One feature contract.** Predictive features, protected attributes, and the bureau-score scale are defined once in `credit_scoring.schema` and imported everywhere, so serving, monitoring, and retraining cannot drift out of sync. ([ADR 0001](docs/adr/0001-feature-contract.md))
- **No silent fill.** A missing model feature raises a `MissingFeatureError` that names it and surfaces as a clean `422` at the API boundary, instead of being fabricated into a systematically pessimistic score. ([ADR 0002](docs/adr/0002-no-silent-fill.md))
- **Honest metrics.** Accuracy, ROC-AUC, and F1 are computed directly from predictions and ground truth, and reported as `nan` where they are mathematically undefined (e.g. single-class validation) rather than zeroed or rescaled upward. ([ADR 0003](docs/adr/0003-fairness-metrics.md))
- **Side-effect-free core.** The `credit_scoring` package imports cleanly with no network or filesystem access; MLflow is engaged only when a tracking URI is supplied at call time. The FastAPI app, the drift monitor, and the retraining script are thin orchestration layers over it, which is why the statistics and decision logic are fully unit-testable without a server or a trained model.

## Architecture

```
src/credit_scoring/
  schema.py          Feature contract: predictive vs demographic columns, scales
  features.py        Request -> model feature alignment (no value fabrication)
  inference.py       Ensemble probability aggregation + decision threshold
  fairness.py        Group-fairness metrics (parity, disparate impact, odds)
  drift.py           Two-sample KS drift detection (pure; returns a DriftReport)
  retrain_eval.py    Honest accuracy / ROC-AUC / F1
  retrain.py         Retrain gating, leakage-free split, champion selection
  explain.py         SHAP feature-label alignment
  cli.py             Shell entrypoints: audit / drift / predict
  api/
    app.py           FastAPI factory + model loading + SQLite feedback store
    schemas.py       Request/response models
    routers/         /predict, /feedback, /explain

scripts/             Operational entrypoints (drift monitor, retraining)
streamlit_app.py     "FairCredit Africa" demo UI
tests/unit/          Module-level behavioural tests
tests/integration/   API tests via FastAPI TestClient
credit_scoring_fairness.ipynb   Model training (LightGBM + fairness mitigation)
```

**Data flow.** A request hits `/predict`, is aligned to the model's feature contract by `prepare_features`, scored by `predict_default_proba`, and thresholded by `decide`. Ground-truth outcomes arrive at `/feedback` and accumulate in SQLite. `scripts/monitor_drift.py` compares that accumulated traffic against the training baseline, and `scripts/retrain_pipeline.py` retrains and promotes a new model only when it clears the champion/challenger gate.

The package is the serving and operations layer; the model itself is trained in `credit_scoring_fairness.ipynb` and saved to `model/lgb_model.joblib` as a dict of `{"model", "feature_names"}`.

## Quickstart

```bash
# Core library + API + tests
pip install -r requirements.txt
pytest                       # 95 tests

# Serve the API
uvicorn credit_scoring.api.app:app --reload    # http://127.0.0.1:8000/docs

# Run the demo UI (installs Streamlit + Groq on top of the core deps)
pip install -r requirements.txt -r requirements-app.txt
streamlit run streamlit_app.py
```

The API loads `model/lgb_model.joblib` if it is present and otherwise returns `503` from the scoring endpoints. The Streamlit app falls back to a small synthetic model when no trained artifact is found, so it runs out of the box; set `GROQ_API_KEY` (env var or the sidebar) to enable the LLM narrative, which otherwise degrades to a template built from the same SHAP factors.

### Docker

```bash
docker build -t credit-scoring .
docker run -p 8000:8000 credit-scoring
```

## API

`POST /predict` — score a borrower's probability of default.

```bash
curl -s http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 450000, "AMT_ANNUITY": 24700,
  "AMT_GOODS_PRICE": 450000, "DAYS_BIRTH": -14235, "DAYS_EMPLOYED": -2384,
  "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2,
  "EXT_SOURCE_1": 0.65, "EXT_SOURCE_2": 0.71, "EXT_SOURCE_3": 0.58,
  "FLAG_OWN_CAR": 1, "CODE_GENDER": "F",
  "NAME_EDUCATION_TYPE": "Higher education", "NAME_INCOME_TYPE": "Working",
  "NAME_HOUSING_TYPE": "House / apartment"
}'
# -> {"default_probability": 0.083, "default_prediction": 0}
```

Demographic fields are accepted in the payload (they are needed for the feedback record) but are dropped before the frame ever reaches the model.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/predict` | POST | Default probability and lend/decline decision |
| `/explain` | POST | SHAP waterfall for the decision, as a base64 PNG |
| `/feedback` | POST | Persist the realised outcome (`ground_truth`) for retraining |

## CLI

The same operations run from the shell:

```bash
# Group fairness audit over a scored CSV
python -m credit_scoring audit --input scored.csv --pred-col prediction \
    --sensitive-col CODE_GENDER --truth-col TARGET

# KS drift between a baseline and current CSV
python -m credit_scoring drift --baseline baseline.csv --current live.csv

# Score a single applicant from a JSON file
python -m credit_scoring predict --model model/lgb_model.joblib --input applicant.json
```

## Fairness metrics

`audit_fairness` reports, per protected group, against the positive (predicted-default) decision:

| Metric | What it measures |
| --- | --- |
| Selection rate | Share of each group that receives the positive decision |
| Demographic parity difference | Largest gap in selection rate between groups |
| Disparate-impact ratio | Lowest group rate / highest group rate; the basis for the four-fifths rule |
| Equalized-odds difference | Largest gap in true- and false-positive rates (needs ground truth) |
| Equal-opportunity difference | Largest gap in true-positive rate (needs ground truth) |

`passes_four_fifths_rule` collapses the disparate-impact ratio into a single pass/fail at the regulatory 0.8 threshold. Metrics that are undefined for a group (a TPR with no positives, a ratio with an all-zero maximum) are returned as `nan` rather than silently treated as zero.

## Testing

95 behavioural tests, run on every push and pull request via GitHub Actions (`ruff check` then `pytest`, Python 3.12).

| Area | Tests |
| --- | --- |
| Inference & decision threshold | 15 |
| Retraining gates & splits | 15 |
| Feature alignment / no-silent-fill | 14 |
| Group fairness metrics | 13 |
| Drift detection | 12 |
| Honest evaluation metrics | 10 |
| CLI | 6 |
| SHAP label alignment | 5 |
| API (integration, `TestClient`) | 5 |

## Tech stack

Python 3.10+, FastAPI, Pydantic v2, scikit-learn, LightGBM, SHAP, SciPy (KS test), pandas, NumPy. Streamlit and Groq (Llama-3.3-70B) power the demo UI. MLflow is optional for experiment tracking. Tooling: pytest, ruff, pre-commit, Docker.

## Data

Home Credit Default Risk dataset (anonymised loan application records). The repository ships code, not data; point the training notebook and the baseline/feedback paths at your own copy.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
