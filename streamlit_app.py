"""FairCredit Africa — interactive UI.

Wired to the credit_scoring package: real probability of default, a real SHAP
explanation of each decision, and an LLM that turns the SHAP factors into a
plain-language reason. A second tab runs a group fairness audit.

Run:  pip install -r requirements-app.txt  &&  streamlit run streamlit_app.py
Set GROQ_API_KEY (env var or the sidebar box) to enable the LLM narrative;
without it the app falls back to a template explanation built from the same
SHAP factors, so it still runs.
"""

from __future__ import annotations

import os
import sys

# Make the src/ package importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st

from credit_scoring.fairness import audit_fairness, passes_four_fifths_rule
from credit_scoring.features import prepare_features
from credit_scoring.inference import decide, predict_default_proba
from credit_scoring.schema import BUREAU_SCORE_FEATURES, NUMERIC_FEATURES

MODEL_PATH = "model/lgb_model.joblib"
GROQ_MODEL = "llama-3.3-70b-versatile"


# --------------------------------------------------------------------------- #
# Model: load the trained artifact if present, else train a small demo model.
# --------------------------------------------------------------------------- #
def _synthetic_frame(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.normal(150000, 60000, n).clip(20000),
        "AMT_CREDIT": rng.normal(400000, 150000, n).clip(20000),
        "AMT_ANNUITY": rng.normal(24000, 9000, n).clip(2000),
        "AMT_GOODS_PRICE": rng.normal(380000, 140000, n).clip(20000),
        "DAYS_BIRTH": rng.integers(-25000, -7000, n).astype(float),
        "DAYS_EMPLOYED": rng.integers(-12000, -100, n).astype(float),
        "CNT_CHILDREN": rng.integers(0, 4, n).astype(float),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "EXT_SOURCE_1": rng.uniform(0, 1, n),
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "EXT_SOURCE_3": rng.uniform(0, 1, n),
        "FLAG_OWN_CAR": rng.integers(0, 2, n),
    })
    # Lower bureau scores -> higher default risk (with noise).
    risk = 1.0 - df[list(BUREAU_SCORE_FEATURES)].mean(axis=1)
    prob = (risk + rng.normal(0, 0.15, n)).clip(0.01, 0.99)
    df["TARGET"] = (rng.uniform(0, 1, n) < prob).astype(int)
    return df


@st.cache_resource(show_spinner=False)
def get_model():
    """Return (model, feature_names, source_label)."""
    import joblib

    if os.path.exists(MODEL_PATH):
        artefact = joblib.load(MODEL_PATH)
        if isinstance(artefact, dict) and "model" in artefact:
            return artefact["model"], artefact.get("feature_names", list(NUMERIC_FEATURES)), "loaded production model"
        return artefact, list(NUMERIC_FEATURES), "loaded production model"

    import lightgbm as lgb

    train = _synthetic_frame(3000)
    feats = list(NUMERIC_FEATURES)
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(train[feats], train["TARGET"])
    return model, feats, "demo model (trained on synthetic data)"


@st.cache_data(show_spinner=False)
def sample_population(n: int = 600) -> pd.DataFrame:
    """A synthetic population with a protected attribute, for the audit tab."""
    df = _synthetic_frame(n, seed=11)
    rng = np.random.default_rng(11)
    df["CODE_GENDER"] = rng.choice(["F", "M"], size=n)
    # Inject a mild disparity so the audit shows something to discuss.
    df.loc[df["CODE_GENDER"] == "F", "EXT_SOURCE_2"] *= 0.9
    return df


# --------------------------------------------------------------------------- #
# Explanation: SHAP factors + LLM narrative (template fallback if no key).
# --------------------------------------------------------------------------- #
def shap_contributions(model, X: pd.DataFrame):
    """Return a list of (feature, value, signed_contribution) sorted by impact."""
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer(X.values)
    vals = sv.values[0]
    if vals.ndim == 2:  # (features, classes)
        vals = vals[:, 1]
    rows = [(c, float(X.iloc[0][c]), float(v)) for c, v in zip(X.columns, vals)]
    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return rows


def _factor_lines(contribs, k: int = 5) -> str:
    out = []
    for feature, value, contrib in contribs[:k]:
        direction = "raised the risk" if contrib > 0 else "lowered the risk"
        out.append(f"- {feature} (value {value:.3g}) {direction}")
    return "\n".join(out)


def template_explanation(decision: int, prob: float, contribs) -> str:
    verdict = "declined" if decision == 1 else "approved"
    top = contribs[:3]
    drivers = ", ".join(f"{f}" for f, _, _ in top)
    return (
        f"This application would be **{verdict}** (estimated default probability "
        f"{prob:.0%}). The factors that weighed most on the decision were: "
        f"{drivers}.\n\n{_factor_lines(contribs)}"
    )


def llm_explanation(decision: int, prob: float, contribs, api_key: str) -> str:
    if not api_key:
        return template_explanation(decision, prob, contribs)
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        verdict = "declined" if decision == 1 else "approved"
        prompt = (
            "You are a credit analyst writing to a loan applicant. Using ONLY the "
            "factors listed below (do not invent any), explain in 3-4 plain, "
            "respectful sentences why the application was "
            f"{verdict} (estimated default probability {prob:.0%}). End with one "
            "concrete, factual suggestion for what could improve a future "
            "application, based only on these factors.\n\n"
            f"Factors, most influential first:\n{_factor_lines(contribs, k=6)}"
        )
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350,
        )
        return resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001 - surface any LLM error, keep app usable
        return template_explanation(decision, prob, contribs) + f"\n\n_(LLM unavailable: {e})_"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="FairCredit Africa", page_icon="🪙", layout="wide")
st.title("🪙 FairCredit Africa")
st.caption("Explainable, fairness-aware credit scoring")

model, feature_names, source_label = get_model()

with st.sidebar:
    st.subheader("Settings")
    st.info(f"Model: {source_label}")
    api_key = st.text_input(
        "GROQ_API_KEY (optional)",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Enables the LLM narrative. Without it, a template explanation is used.",
    )

score_tab, fairness_tab = st.tabs(["Score an applicant", "Fairness audit"])

with score_tab:
    st.subheader("Applicant details")
    c1, c2, c3 = st.columns(3)
    with c1:
        income = st.number_input("Annual income", value=150000.0, step=10000.0)
        credit = st.number_input("Loan amount", value=400000.0, step=10000.0)
        annuity = st.number_input("Annuity", value=24000.0, step=1000.0)
        goods = st.number_input("Goods price", value=380000.0, step=10000.0)
    with c2:
        age_years = st.slider("Age", 18, 75, 38)
        emp_years = st.slider("Years employed", 0, 40, 6)
        children = st.number_input("Children", value=1, step=1, min_value=0)
        fam = st.number_input("Family members", value=3, step=1, min_value=1)
    with c3:
        ext1 = st.slider("Bureau score 1", 0.0, 1.0, 0.6)
        ext2 = st.slider("Bureau score 2", 0.0, 1.0, 0.7)
        ext3 = st.slider("Bureau score 3", 0.0, 1.0, 0.5)
        own_car = 1 if st.checkbox("Owns a car", value=True) else 0

    if st.button("Score applicant", type="primary"):
        record = pd.DataFrame([{
            "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods, "DAYS_BIRTH": -age_years * 365.0,
            "DAYS_EMPLOYED": -emp_years * 365.0, "CNT_CHILDREN": float(children),
            "CNT_FAM_MEMBERS": float(fam), "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3, "FLAG_OWN_CAR": own_car,
        }])
        X = prepare_features(record, feature_names)
        prob = float(predict_default_proba(model, X)[0])
        decision = int(decide([prob])[0])

        m1, m2 = st.columns(2)
        m1.metric("Default probability", f"{prob:.1%}")
        m2.metric("Decision", "Decline" if decision == 1 else "Approve")

        try:
            contribs = shap_contributions(model, X)
            st.subheader("Why — explanation")
            st.markdown(llm_explanation(decision, prob, contribs, api_key))
            with st.expander("Underlying SHAP factors"):
                st.dataframe(pd.DataFrame(contribs, columns=["feature", "value", "contribution"]))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Explanation unavailable: {e}")

with fairness_tab:
    st.subheader("Group fairness audit")
    st.write(
        "Scores a synthetic population and audits decisions across a protected "
        "attribute using the package's fairness metrics."
    )
    attr = st.selectbox("Protected attribute", ["CODE_GENDER"])
    if st.button("Run audit"):
        pop = sample_population()
        X = prepare_features(pop.drop(columns=["TARGET", attr]), feature_names)
        proba = predict_default_proba(model, X)
        preds = decide(proba)
        report = audit_fairness(preds, pop[attr].to_numpy(), y_true=pop["TARGET"].to_numpy())

        st.markdown(
            f"**Disparate-impact ratio:** {report.disparate_impact_ratio:.2f} "
            f"({'passes' if passes_four_fifths_rule(report) else 'fails'} the 4/5ths rule)  \n"
            f"**Demographic parity difference:** {report.demographic_parity_difference:.3f}  \n"
            f"**Equalized-odds difference:** {report.equalized_odds_difference:.3f}"
        )
        st.dataframe(pd.DataFrame([
            {"group": g.group, "n": g.n, "selection_rate": round(g.selection_rate, 3),
             "TPR": round(g.tpr, 3), "FPR": round(g.fpr, 3)}
            for g in report.by_group
        ]))
