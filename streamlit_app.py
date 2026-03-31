# streamlit_app.py
# FairCredit Africa — Loan Risk Assistant with AI Loan Officer (Groq)
# SDG 8 (Decent Work) · SDG 10 (Reduced Inequalities)
# Human-in-the-Loop Credit Scoring · Kigali, Rwanda
# Run: streamlit run streamlit_app.py

import joblib
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="FairCredit Africa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Master CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --gold:    #C8922A;
    --gold2:   #E8B84B;
    --dark:    #0C0E13;
    --surface: #13161E;
    --card:    #1A1E28;
    --border:  #252A38;
    --text:    #EDE8DE;
    --muted:   #6B7280;
    --label:   #C8D0DC;
    --green:   #22C55E;
    --yellow:  #EAB308;
    --red:     #EF4444;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--dark) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stAppViewContainer"] > .main { background: var(--dark) !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    font-size: 0.9rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--gold), var(--gold2)) !important;
    color: #0C0E13 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    letter-spacing: 0.5px;
}

hr { border-color: var(--border) !important; opacity: 0.5; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }

.hero-banner {
    position: relative; width: 100%; height: 300px;
    border-radius: 20px; overflow: hidden; margin-bottom: 2rem;
}
.hero-banner img {
    width: 100%; height: 100%; object-fit: cover;
    opacity: 0.38; filter: saturate(0.7);
}
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(12,14,19,0.97) 0%, rgba(12,14,19,0.5) 55%, transparent 100%);
    display: flex; flex-direction: column; justify-content: center; padding: 3rem;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(200,146,42,0.15); border: 1px solid rgba(200,146,42,0.4);
    color: var(--gold2); border-radius: 20px; padding: 4px 14px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 14px; width: fit-content;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem; font-weight: 800; color: #fff;
    line-height: 1.1; margin: 0 0 10px 0;
}
.hero-title span { color: var(--gold2); }
.hero-sub { font-size: 0.95rem; color: rgba(237,232,222,0.6); font-weight: 300; max-width: 460px; line-height: 1.6; }
.sdg-pills { display: flex; gap: 8px; margin-top: 18px; }
.sdg-pill {
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25);
    color: #86efac; border-radius: 20px; padding: 3px 12px;
    font-size: 0.7rem; font-weight: 600;
}

.section-head {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 2.5px;
    text-transform: uppercase; color: var(--gold);
    margin: 28px 0 14px 0; display: flex; align-items: center; gap: 10px;
}
.section-head::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.form-card-title {
    font-size: 0.72rem; font-weight: 600; color: var(--gold2);
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px;
}

[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stCheckbox"] label {
    color: #C8D0DC !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Make all streamlit input labels clearly visible */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stCheckbox"] label {
    color: #C8D0DC !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.demo-tip {
    background: rgba(200,146,42,0.08);
    border: 1px solid rgba(200,146,42,0.25);
    border-left: 4px solid var(--gold);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;
    font-size: 0.82rem; color: rgba(237,232,222,0.8); line-height: 1.6;
}

.result-wrapper {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 28px; margin-top: 20px;
    position: relative; overflow: hidden;
}
.result-wrapper::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.result-green::before { background: linear-gradient(90deg, #22C55E, #86efac); }
.result-yellow::before { background: linear-gradient(90deg, #EAB308, #fde68a); }
.result-red::before { background: linear-gradient(90deg, #EF4444, #fca5a5); }

.prob-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.prob-value { font-family: 'DM Mono', monospace; font-size: 4.5rem; font-weight: 500; line-height: 1; letter-spacing: -3px; }
.prob-scale { font-size: 0.7rem; color: var(--muted); margin-top: 4px; }
.zone-badge { display: inline-flex; align-items: center; gap: 8px; border-radius: 24px; padding: 7px 16px; font-size: 0.82rem; font-weight: 600; margin-top: 14px; }
.zone-green-badge { background: rgba(34,197,94,0.1); color: #86efac; border: 1px solid rgba(34,197,94,0.25); }
.zone-yellow-badge { background: rgba(234,179,8,0.1); color: #fde68a; border: 1px solid rgba(234,179,8,0.25); }
.zone-red-badge { background: rgba(239,68,68,0.1); color: #fca5a5; border: 1px solid rgba(239,68,68,0.25); }

.hitl-panel {
    background: linear-gradient(135deg, #0f1520, #151c2a);
    border: 1px solid var(--border); border-left: 4px solid var(--gold);
    border-radius: 14px; padding: 22px 26px; margin-top: 18px;
}
.hitl-header { font-family: 'Playfair Display', serif; font-size: 1.05rem; font-weight: 700; color: var(--gold2); margin-bottom: 8px; }
.hitl-body { font-size: 0.85rem; color: rgba(237,232,222,0.68); line-height: 1.7; }

.advisor-panel {
    background: linear-gradient(135deg, #0d1a0d, #111a11);
    border: 1px solid rgba(34,197,94,0.2);
    border-left: 4px solid #22C55E;
    border-radius: 14px; padding: 24px 28px; margin-top: 20px;
}
.advisor-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.advisor-avatar {
    width: 42px; height: 42px; border-radius: 50%;
    background: linear-gradient(135deg, #22C55E, #166534);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
}
.advisor-name { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 700; color: #86efac; }
.advisor-title { font-size: 0.72rem; color: #6B7280; margin-top: 1px; }
.advisor-body { font-size: 0.88rem; color: rgba(237,232,222,0.82); line-height: 1.85; }

.chat-container {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px; margin-top: 16px;
    max-height: 400px; overflow-y: auto;
}
.chat-msg-user {
    background: rgba(200,146,42,0.1); border: 1px solid rgba(200,146,42,0.2);
    border-radius: 12px 12px 4px 12px; padding: 10px 14px; margin-bottom: 12px;
    font-size: 0.85rem; color: var(--text); margin-left: 20%;
}
.chat-msg-ai {
    background: rgba(34,197,94,0.06); border: 1px solid rgba(34,197,94,0.15);
    border-radius: 12px 12px 12px 4px; padding: 10px 14px; margin-bottom: 12px;
    font-size: 0.85rem; color: rgba(237,232,222,0.85); margin-right: 20%; line-height: 1.7;
}

.zone-ref-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; text-align: center; }
.zone-ref-range { font-family: 'DM Mono', monospace; font-size: 1rem; font-weight: 500; margin-bottom: 4px; }
.zone-ref-label { font-size: 0.75rem; color: var(--muted); }
.zone-ref-action { font-size: 0.7rem; margin-top: 3px; font-weight: 600; }

.cmu-banner {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; overflow: hidden; display: flex;
    align-items: stretch; margin-bottom: 22px; height: 130px;
}
.cmu-img { width: 240px; object-fit: cover; flex-shrink: 0; filter: saturate(0.7); }
.cmu-text { padding: 18px 22px; display: flex; flex-direction: column; justify-content: center; }
.cmu-title { font-family: 'Playfair Display', serif; font-size: 0.95rem; font-weight: 700; color: var(--gold2); margin-bottom: 5px; }
.cmu-sub { font-size: 0.8rem; color: var(--muted); line-height: 1.5; }

.fairness-item {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 13px 16px; margin-bottom: 7px;
    display: flex; justify-content: space-between; align-items: center;
}
.fi-left strong { color: var(--text); font-size: 0.85rem; }
.fi-left code {
    background: rgba(200,146,42,0.1); color: var(--gold2);
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    padding: 2px 6px; border-radius: 4px; margin-left: 7px;
}
.fi-right { font-size: 0.76rem; color: var(--muted); max-width: 280px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── Groq client ───────────────────────────────────────────────────────────────
def get_groq_client(api_key):
    if api_key:
        return Groq(api_key=api_key)
    return None

# ── AI Loan Officer system prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are the FairCredit Advisor — a Senior Loan Officer and Financial Inclusion Specialist
at FairCredit Africa, based in Kigali, Rwanda.

Your philosophy: "Every rejection is a person denied dignity. Every approval is a family with a future."

Your role is to assess loan applications for people largely excluded from traditional banking —
market vendors, farmers, boda boda riders, graduates, single mothers — using a fairness-aware AI model.

You always:
1. Reason from the ACTUAL numbers provided — income, loan amount, debt-to-income ratio, credit signals, risk score
2. Connect your reasoning explicitly to SDG 8 (Decent Work and Economic Growth) and SDG 10 (Reduced Inequalities)
3. Give a structured, comprehensive assessment with clear justification
4. Suggest concrete alternatives when declining — never just say no without a path forward
5. Acknowledge the human story behind the numbers
6. Reference the specific alternative credit signals (mobile money, savings, loan history) and their impact
7. Be honest about risk while being a genuine advocate for financial inclusion

Your assessment must always include these five sections:
RISK ANALYSIS — What the numbers say and why
FAIRNESS LENS — What traditional scoring would miss about this applicant
SDG ALIGNMENT — Specific connection to SDG 8 or SDG 10 with explanation
RECOMMENDATION — Approve / Approve with conditions / Restructure / Decline with alternative path
NEXT STEPS — Concrete actions for the applicant or loan officer

Keep responses clear, warm, and professional. Write in paragraphs.
Show genuine care for the applicant while being financially responsible.
When asked follow-up questions, answer as the FairCredit Advisor with deep knowledge
of East African financial inclusion, microfinance, and responsible AI in credit scoring."""


def generate_assessment(client, applicant_data):
    """Generate AI loan officer assessment from actual applicant metrics."""
    prompt = f"""Please provide a comprehensive loan assessment for the following applicant.
Reason entirely from these actual numbers — do not use generic responses.

APPLICANT PROFILE:
- Persona: {applicant_data['persona']}
- Gender: {applicant_data['gender']} | Education: {applicant_data['education']}
- Employment Type: {applicant_data['income_type']} | Housing: {applicant_data['housing_type']}
- Age: {applicant_data['age']} years | Years in Current Work: {applicant_data['years_employed']}
- Number of Dependants: {applicant_data['children']} | Family Size: {applicant_data['family']}

FINANCIAL METRICS:
- Monthly Income: RWF {applicant_data['income']:,.0f}
- Loan Amount Requested: RWF {applicant_data['credit']:,.0f}
- Monthly Repayment Required: RWF {applicant_data['annuity']:,.0f}
- Purpose / Item Value: RWF {applicant_data['goods']:,.0f}
- Debt-to-Income Ratio: {applicant_data['dti']:.1%}

ALTERNATIVE CREDIT SIGNALS:
- Signal 1: {applicant_data['cs1']}
- Signal 2: {applicant_data['cs2']}
- Signal 3: {applicant_data['cs3']}
- Owns Asset (car/moto): {applicant_data['owns_asset']}

AI MODEL OUTPUT:
- Fairness-Adjusted Risk Index: {applicant_data['prob']:.2f} (from fairness-constrained classifier, not raw probability)
- Risk Zone: {applicant_data['zone']}
- Model: Fairness-constrained LightGBM ensemble (ExponentiatedGradient)

Please provide your full assessment covering all five sections."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content


def chat_with_advisor(client, user_message, applicant_data, chat_history):
    """Continue conversation with the AI loan officer."""
    context = f"""[Current applicant: {applicant_data['persona']},
Score: {applicant_data['prob']:.2f}, Zone: {applicant_data['zone']},
Income: RWF {applicant_data['income']:,.0f}, Loan: RWF {applicant_data['credit']:,.0f},
DTI: {applicant_data['dti']:.1%},
Signals: {applicant_data['cs1']}, {applicant_data['cs2']}, {applicant_data['cs3']}]

User question: {user_message}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": context})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=600,
        messages=messages
    )
    return response.choices[0].message.content


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    USE_API    = st.checkbox("Use FastAPI Backend", value=True)
    API_URL    = st.text_input("FastAPI URL", value="http://localhost:8000")
    MODEL_PATH = st.text_input("Model path", value="model/lgb_model.joblib")
    
    st.divider()
    st.markdown("### 📈 MLOps Health")
    st.markdown("""
    <div style='font-size:0.82rem;color:#6B7280;line-height:1.6'>
    Monitor data drift, KS statistics, and continuous retraining experiments on our local MLflow server.<br><br>
    <a href='http://localhost:5000' target='_blank' style='color:#C8922A;text-decoration:none;font-weight:600;'>↗ Open MLflow Dashboard</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🤖 AI Loan Officer")
    GROQ_KEY = st.text_input(
        "Groq API Key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get free key at console.groq.com"
    )
    if GROQ_KEY:
        os.environ["GROQ_API_KEY"] = GROQ_KEY
    st.markdown("""
    <div style='font-size:0.75rem;color:#6B7280;margin-top:6px'>
    Get a free key at <strong>console.groq.com</strong><br>
    No credit card required.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:0.82rem;color:#6B7280;line-height:1.6'>
    Built for financial inclusion across East Africa.<br><br>
    Model trained on Home Credit Default Risk dataset.<br><br>
    <strong style='color:#C8922A'>The model informs. You decide.</strong><br><br>
    <em style='color:#4B5563;font-size:0.72rem'>Score = fairness-constrained risk index, not a calibrated probability.</em>
    </div>
    """, unsafe_allow_html=True)

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <img src="https://images.unsplash.com/photo-1611348586840-ea9872d33411?w=1400&q=80"
         alt="Kigali city"
         onerror="this.src='https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=1400&q=80'"/>
    <div class="hero-overlay">
        <div class="hero-tag">🌍 Kigali, Rwanda · East Africa</div>
        <div class="hero-title">Fair<span>Credit</span> Africa</div>
        <div class="hero-sub">
            AI-assisted loan risk assessment for the underbanked.
            Empowering loan officers with data — not replacing their judgment.
        </div>
        <div class="sdg-pills">
            <span class="sdg-pill">✦ SDG 8 — Decent Work</span>
            <span class="sdg-pill">✦ SDG 10 — Reduced Inequalities</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  🧑‍💼  Applicant Assessment  ", "  ⚖️  Fairness & SDG Context  ", "  📈  MLOps Control Center  "])

# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    def get_probability(mdl, df):
        if hasattr(mdl, 'predictors_') and hasattr(mdl, 'weights_'):
            weights = np.array(mdl.weights_)
            weights = weights / weights.sum()
            prob = 0.0
            for w, predictor in zip(weights, mdl.predictors_):
                if hasattr(predictor, 'predict_proba'):
                    prob += w * predictor.predict_proba(df)[0, 1]
                else:
                    prob += w * float(predictor.predict(df)[0])
            return round(float(prob), 4)
        elif hasattr(mdl, 'predict_proba'):
            return round(float(mdl.predict_proba(df)[0, 1]), 4)
        else:
            return round(float(mdl.predict(df)[0]), 4)

    PERSONAS = {
        "Government Worker": {
            "emoji": "👩‍💼", "desc": "🟢 LOW RISK · Stable salary, has previous loan history",
            "img": "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?w=400&q=75",
            "income": 280000, "credit": 400000, "annuity": 18000, "goods": 380000,
            "age": 38, "employed": 10, "children": 2, "family": 4,
            "ext1": 0.75, "ext2": 0.75, "ext3": 0.75, "car": 1,
            "gender": "F", "education": "Higher education",
            "income_type": "State servant", "housing": "House / apartment",
        },
        "CMU Graduate": {
            "emoji": "👩‍🎓", "desc": "🟢 LOW-MEDIUM · First job, has savings account",
            "img": "https://images.unsplash.com/photo-1531545514256-b1400bc00f31?w=400&q=75",
            "income": 120000, "credit": 150000, "annuity": 8000, "goods": 140000,
            "age": 24, "employed": 1, "children": 0, "family": 2,
            "ext1": 0.55, "ext2": 0.55, "ext3": 0.55, "car": 0,
            "gender": "F", "education": "Higher education",
            "income_type": "Working", "housing": "With parents",
        },
        "Market Vendor": {
            "emoji": "🏪", "desc": "🟡 MEDIUM · Informal income, mobile money user",
            "img": "https://images.unsplash.com/photo-1516026672322-bc52d61a55d5?w=400&q=75",
            "income": 80000, "credit": 300000, "annuity": 15000, "goods": 280000,
            "age": 35, "employed": 3, "children": 2, "family": 4,
            "ext1": 0.35, "ext2": 0.35, "ext3": 0.35, "car": 0,
            "gender": "F", "education": "Secondary / secondary special",
            "income_type": "Working", "housing": "Rented apartment",
        },
        "Boda Boda Rider": {
            "emoji": "🚗", "desc": "🟡 MEDIUM-HIGH · High debt, mobile money only",
            "img": "https://images.unsplash.com/photo-1595781572981-d63151b232ed?w=400&q=75",
            "income": 60000, "credit": 500000, "annuity": 25000, "goods": 480000,
            "age": 27, "employed": 2, "children": 1, "family": 3,
            "ext1": 0.08, "ext2": 0.35, "ext3": 0.08, "car": 1,
            "gender": "M", "education": "Secondary / secondary special",
            "income_type": "Working", "housing": "With parents",
        },
        "Single Mother": {
            "emoji": "🏠", "desc": "🔴 HIGH RISK · Low income, no credit history",
            "img": "https://images.unsplash.com/photo-1489710437720-ebb67ec84dd2?w=400&q=75",
            "income": 45000, "credit": 200000, "annuity": 12000, "goods": 190000,
            "age": 31, "employed": 1, "children": 3, "family": 4,
            "ext1": 0.08, "ext2": 0.08, "ext3": 0.08, "car": 0,
            "gender": "F", "education": "Secondary / secondary special",
            "income_type": "Working", "housing": "Rented apartment",
        },
        "Smallholder Farmer": {
            "emoji": "👨‍🌾", "desc": "🔴 VERY HIGH · No employment, no credit history at all",
            "img": "https://images.unsplash.com/photo-1556075798-4825dfaaf498?w=400&q=75",
            "income": 30000, "credit": 500000, "annuity": 30000, "goods": 480000,
            "age": 42, "employed": 0, "children": 5, "family": 7,
            "ext1": 0.08, "ext2": 0.08, "ext3": 0.08, "car": 0,
            "gender": "M", "education": "Lower secondary",
            "income_type": "Working", "housing": "House / apartment",
        },
    }

    st.markdown("""
    <div class="demo-tip">
        💡 <strong style="color:#E8B84B">Demo Tip — Smallholder Farmer:</strong>
        Select the Farmer (all signals = No credit history → 🔴 Red).
        Then change <strong>Signal 1 to "Mobile money user"</strong> and reassess —
        watch the score change AND the AI Loan Officer generate a completely different justification.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Step 1 — Select Applicant Profile</div>', unsafe_allow_html=True)

    if "selected_persona"   not in st.session_state: st.session_state.selected_persona   = "Government Worker"
    if "chat_history"       not in st.session_state: st.session_state.chat_history       = []
    if "assessment_done"    not in st.session_state: st.session_state.assessment_done    = False
    if "current_applicant"  not in st.session_state: st.session_state.current_applicant  = None
    if "advisor_report"     not in st.session_state: st.session_state.advisor_report     = ""
    if "follow_up_messages" not in st.session_state: st.session_state.follow_up_messages = []

    cols = st.columns(6)
    for i, (name, pd_) in enumerate(PERSONAS.items()):
        with cols[i]:
            is_active  = st.session_state.selected_persona == name
            border_col = "#C8922A" if is_active else "#252A38"
            bg_col     = "#1f1c0f" if is_active else "#1A1E28"
            if st.button(f"{pd_['emoji']} {name}", key=f"btn_{name}", use_container_width=True):
                st.session_state.selected_persona   = name
                st.session_state.chat_history       = []
                st.session_state.assessment_done    = False
                st.session_state.advisor_report     = ""
                st.session_state.follow_up_messages = []
                st.rerun()
            st.markdown(f"""
            <div style="background:{bg_col};border:2px solid {border_col};
                        border-radius:12px;overflow:hidden;margin-top:-8px;margin-bottom:4px;">
                <img src="{pd_['img']}" style="width:100%;height:85px;object-fit:cover;
                     filter:saturate(0.75);" onerror="this.style.display='none'"/>
                <div style="padding:7px 9px 10px;">
                    <div style="font-size:0.67rem;color:#6B7280;line-height:1.4">{pd_['desc']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    p = PERSONAS[st.session_state.selected_persona]
    st.markdown(f"""
    <div style="background:rgba(200,146,42,0.07);border:1px solid rgba(200,146,42,0.22);
                border-radius:10px;padding:11px 16px;margin-bottom:6px;">
        <span style="font-size:0.95rem;font-weight:600;color:#E8B84B">{p['emoji']} {st.session_state.selected_persona}</span>
        <span style="color:#6B7280;font-size:0.82rem;margin-left:10px">— {p['desc']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-head">Step 2 — Applicant Details</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown('<div class="form-card-title">💰 Financial Information</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            def parse_currency(val_str, default_val):
                try:
                    return float(str(val_str).replace(",", "").replace(" ", "").strip())
                except ValueError:
                    return float(default_val)

            _inc = st.text_input("Monthly Income (RWF)", value=f"{int(p['income']):,}")
            amt_income = parse_currency(_inc, p['income'])

            _cred = st.text_input("Loan Amount (RWF)", value=f"{int(p['credit']):,}")
            amt_credit = parse_currency(_cred, p['credit'])

            _ann = st.text_input("Monthly Repayment (RWF)", value=f"{int(p['annuity']):,}")
            amt_annuity = parse_currency(_ann, p['annuity'])

            _goods = st.text_input("Purpose / Item Value (RWF)", value=f"{int(p['goods']):,}")
            amt_goods_price = parse_currency(_goods, p['goods'])
        with fc2:
            age_years      = st.number_input("Age (years)", min_value=18, max_value=70, value=p["age"])
            years_employed = st.number_input("Years in Current Work", min_value=0, max_value=40, value=p["employed"])
            cnt_children   = st.number_input("Number of Dependants", min_value=0, value=p["children"])
            cnt_fam        = st.number_input("Total Family Members", min_value=1, value=p["family"])

    with right_col:
        st.markdown('<div class="form-card-title">🪪 Demographics *(fairness monitoring only)*</div>', unsafe_allow_html=True)
        edu_list = ["Secondary / secondary special","Higher education","Incomplete higher","Lower secondary","Academic degree"]
        inc_list = ["Working","Commercial associate","Pensioner","State servant","Unemployed","Student","Businessman"]
        hou_list = ["House / apartment","With parents","Municipal apartment","Rented apartment","Office apartment","Co-op apartment"]
        gender       = st.selectbox("Gender", ["F","M"], index=["F","M"].index(p["gender"]))
        education    = st.selectbox("Education Level", edu_list, index=edu_list.index(p["education"]))
        income_type  = st.selectbox("Employment Type", inc_list, index=inc_list.index(p["income_type"]))
        housing_type = st.selectbox("Housing Situation", hou_list, index=hou_list.index(p["housing"]))

        st.markdown('<div class="form-card-title" style="margin-top:14px">📱 Alternative Credit Signals</div>', unsafe_allow_html=True)
        st.caption("⚠️ Change these to see how signals affect both the score AND the AI assessment!")

        credit_options = {
            "No credit history":   0.08,
            "Mobile money user":   0.35,
            "Has savings account": 0.55,
            "Has previous loan":   0.75,
        }

        def score_to_label(v):
            if v <= 0.15:   return "No credit history"
            elif v <= 0.45: return "Mobile money user"
            elif v <= 0.65: return "Has savings account"
            else:           return "Has previous loan"

        cs1        = st.selectbox("Signal 1", list(credit_options.keys()), index=list(credit_options.keys()).index(score_to_label(p["ext1"])))
        cs2        = st.selectbox("Signal 2", list(credit_options.keys()), index=list(credit_options.keys()).index(score_to_label(p["ext2"])))
        cs3        = st.selectbox("Signal 3", list(credit_options.keys()), index=list(credit_options.keys()).index(score_to_label(p["ext3"])))
        owns_asset = st.selectbox("Owns Asset (car/moto)", ["No","Yes"], index=p["car"])

    days_birth    = -(age_years * 365)
    days_employed = -(years_employed * 365)
    ext1 = credit_options[cs1]
    ext2 = credit_options[cs2]
    ext3 = credit_options[cs3]
    car  = 1 if owns_asset == "Yes" else 0

    st.divider()
    st.markdown('<div class="section-head">Step 3 — Run Assessment</div>', unsafe_allow_html=True)

    if st.button("🔍  Assess Credit Risk", use_container_width=True, type="primary"):

        st.session_state.chat_history       = []
        st.session_state.follow_up_messages = []
        st.session_state.advisor_report     = ""
        st.session_state.assessment_done    = False

        input_features = {
            "AMT_INCOME_TOTAL": amt_income, "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity, "AMT_GOODS_PRICE": amt_goods_price,
            "DAYS_BIRTH": days_birth, "DAYS_EMPLOYED": days_employed,
            "CNT_CHILDREN": cnt_children, "CNT_FAM_MEMBERS": cnt_fam,
            "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
            "FLAG_OWN_CAR": car,
        }
        demographic_features = {
            "CODE_GENDER": gender, "NAME_EDUCATION_TYPE": education,
            "NAME_INCOME_TYPE": income_type, "NAME_HOUSING_TYPE": housing_type,
        }

        result = None
        with st.spinner("Running risk assessment..."):
            if USE_API:
                try:
                    resp = requests.post(f"{API_URL}/predict",
                                         json={**input_features, **demographic_features}, timeout=5)
                    if resp.status_code == 200: result = resp.json()
                    else: st.error(f"API error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")
            else:
                if not os.path.exists(MODEL_PATH):
                    st.error(f"Model not found at {MODEL_PATH}.")
                else:
                    artefact   = joblib.load(MODEL_PATH)
                    mdl        = artefact["model"]
                    feat_names = artefact["feature_names"]
                    df         = pd.DataFrame([input_features])
                    df         = df.reindex(columns=feat_names, fill_value=0)
                    pred       = mdl.predict(df)[0]
                    prob       = get_probability(mdl, df)
                    # Traditional decision: raw threshold on probability (no fairness)
                    # Traditional bank uses 0.5 cutoff with no fairness consideration
                    try:
                        predictors_list = list(mdl.predictors_)
                        weights_array   = np.array(mdl.weights_)
                        best_idx        = int(np.argmax(weights_array))
                        raw_prob        = float(predictors_list[best_idx].predict_proba(df)[0, 1])
                        # Traditional bank: strict 0.4 threshold, no fairness adjustment
                        trad_decision   = 1 if raw_prob >= 0.4 else 0
                    except Exception:
                        # Fallback: if fairness index is above 0.4 traditional bank denies
                        trad_decision = 1 if prob >= 0.4 else 0
                    result     = {"default_prediction": int(pred), "default_probability": prob, "traditional_decision": trad_decision}

        if result:
            prob          = result.get("default_probability", 0.0)
            trad_decision = result.get("traditional_decision", int(prob > 0.5))
            dti           = (amt_annuity * 12) / amt_income if amt_income > 0 else 0

            if prob <= 0.30:
                color, badge_cls, zone_label, action, result_cls = (
                    "#22C55E","zone-green-badge","🟢 Likely to Repay",
                    "Loan officer may consider approving subject to standard verification.",
                    "result-green")
            elif prob <= 0.45:
                color, badge_cls, zone_label, action, result_cls = (
                    "#EAB308","zone-yellow-badge","🟡 Needs Further Review",
                    "Conduct additional interviews, verify income, or request supporting documents.",
                    "result-yellow")
            else:
                color, badge_cls, zone_label, action, result_cls = (
                    "#EF4444","zone-red-badge","🔴 High Default Risk",
                    "Application carries high risk. Request collateral, guarantor, or reduce loan amount.",
                    "result-red")

            st.session_state.current_applicant = {
                "persona": st.session_state.selected_persona,
                "gender": gender, "education": education,
                "income_type": income_type, "housing_type": housing_type,
                "age": age_years, "years_employed": years_employed,
                "children": cnt_children, "family": cnt_fam,
                "income": amt_income, "credit": amt_credit,
                "annuity": amt_annuity, "goods": amt_goods_price,
                "dti": dti, "cs1": cs1, "cs2": cs2, "cs3": cs3,
                "owns_asset": owns_asset, "prob": prob, "zone": zone_label,
            }

            # Traditional decision display values
            trad_color  = "#EF4444" if trad_decision == 1 else "#22C55E"
            trad_label  = "❌ DENIED" if trad_decision == 1 else "✅ APPROVED"
            trad_sub    = "Auto-rejected by traditional system" if trad_decision == 1 else "Auto-approved by traditional system"

            res1, res2, res3 = st.columns([1, 1, 1])
            with res1:
                st.markdown(f"""
                <div class="result-wrapper {result_cls}">
                    <div class="prob-label">Fairness-Adjusted Risk Index</div>
                    <div class="prob-value" style="color:{color}">{prob:.2f}</div>
                    <div class="prob-scale">Lower index = lower risk &nbsp;·&nbsp; Higher index = higher risk</div>
                    <div class="{badge_cls} zone-badge">{zone_label}</div>
                </div>
                """, unsafe_allow_html=True)

            with res2:
                st.markdown(f"""
                <div style="background:#1A1E28;border:1px solid #252A38;border-radius:20px;
                            padding:28px;margin-top:20px;position:relative;overflow:hidden;">
                    <div style="position:absolute;top:0;left:0;right:0;height:3px;
                                background:{'linear-gradient(90deg,#EF4444,#fca5a5)' if trad_decision==1 else 'linear-gradient(90deg,#22C55E,#86efac)'}"></div>
                    <div style="font-size:0.65rem;font-weight:600;letter-spacing:2px;
                                text-transform:uppercase;color:#6B7280;margin-bottom:6px">
                        Traditional Bank Decision</div>
                    <div style="font-family:'DM Mono',monospace;font-size:3.5rem;font-weight:500;
                                line-height:1;letter-spacing:-2px;color:{trad_color}">{trad_decision}</div>
                    <div style="font-size:0.7rem;color:#6B7280;margin-top:4px">
                        0 = Approved &nbsp;·&nbsp; 1 = Denied (no human review)</div>
                    <div style="display:inline-flex;align-items:center;gap:8px;border-radius:24px;
                                padding:7px 16px;font-size:0.82rem;font-weight:600;margin-top:14px;
                                background:{'rgba(239,68,68,0.1)' if trad_decision==1 else 'rgba(34,197,94,0.1)'};
                                color:{trad_color};
                                border:1px solid {'rgba(239,68,68,0.25)' if trad_decision==1 else 'rgba(34,197,94,0.25)'}">{trad_label}</div>
                    <div style="font-size:0.72rem;color:#6B7280;margin-top:8px">{trad_sub}</div>
                </div>
                """, unsafe_allow_html=True)

            with res3:
                demo_df = pd.DataFrame({
                    "": ["Gender","Education","Employment","Housing"],
                    "Value": [gender, education, income_type, housing_type]
                })
                st.dataframe(demo_df, hide_index=True, use_container_width=True, height=178)
                rc = "#EF4444" if dti > 0.5 else "#22C55E" if dti < 0.3 else "#EAB308"
                st.markdown(f"""
                <div style="background:#1A1E28;border:1px solid #252A38;border-radius:10px;
                            padding:12px 16px;margin-top:8px;">
                    <div style="font-size:0.65rem;color:#6B7280;letter-spacing:1.5px;
                                text-transform:uppercase;margin-bottom:3px">Debt-to-Income Ratio</div>
                    <div style="font-family:'DM Mono',monospace;font-size:1.5rem;
                                color:{rc};font-weight:500">{dti:.1%}</div>
                    <div style="font-size:0.7rem;color:#6B7280;margin-top:2px">
                        Annual repayments vs monthly income</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="hitl-panel">
                <div class="hitl-header">👤 Loan Officer Decision Required</div>
                <div class="hitl-body">
                    The model has assigned a risk score of
                    <strong style="color:{color};font-family:'DM Mono',monospace">{prob:.2f}</strong>
                    for this applicant.<br><br>
                    <strong style="color:#EDE8DE">Suggested action:</strong> {action}<br><br>
                    Context the model cannot see — community standing, seasonal income patterns,
                    family support networks — must inform the final decision.
                    <strong style="color:#C8922A">The model informs. You decide.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-head" style="margin-top:28px">Risk Zone Reference</div>', unsafe_allow_html=True)
            z1, z2, z3 = st.columns(3)
            with z1:
                st.markdown("""<div class="zone-ref-card">
                    <div class="zone-ref-range" style="color:#22C55E">0.00 – 0.30</div>
                    <div class="zone-ref-label">🟢 Likely to Repay</div>
                    <div class="zone-ref-action" style="color:#86efac">Officer may approve</div>
                </div>""", unsafe_allow_html=True)
            with z2:
                st.markdown("""<div class="zone-ref-card">
                    <div class="zone-ref-range" style="color:#EAB308">0.31 – 0.45</div>
                    <div class="zone-ref-label">🟡 Needs Review</div>
                    <div class="zone-ref-action" style="color:#fde68a">Officer must investigate</div>
                </div>""", unsafe_allow_html=True)
            with z3:
                st.markdown("""<div class="zone-ref-card">
                    <div class="zone-ref-range" style="color:#EF4444">0.46 – 1.00</div>
                    <div class="zone-ref-label">🔴 High Default Risk</div>
                    <div class="zone-ref-action" style="color:#fca5a5">Request collateral or decline</div>
                </div>""", unsafe_allow_html=True)

            # ── Explainable AI (SHAP Waterfall) ──────────────────────────────
            if USE_API:
                st.markdown('<div class="section-head" style="margin-top:28px">🧠 Explainable AI (SHAP Waterfall)</div>', unsafe_allow_html=True)
                with st.spinner("Generating exact SHAP feature contributions via FastAPI..."):
                    try:
                        resp_exp = requests.post(f"{API_URL}/explain", json={**input_features, **demographic_features}, timeout=8)
                        if resp_exp.status_code == 200:
                            img_b64 = resp_exp.json().get("shap_base64", "")
                            if img_b64:
                                # Render the base64 png payload securely inside a stylistic container
                                st.markdown("""<div style="background:#13161E; border:1px solid #252A38; border-radius:12px; padding:15px; margin-top:10px;">""", unsafe_allow_html=True)
                                st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error(f"SHAP API error {resp_exp.status_code}: {resp_exp.text}")
                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation: {e}")

            # ── Model Feedback Loop (Ground Truth) ─────────────────────────
            st.markdown('<div class="section-head" style="margin-top:28px">🔄 Model Monitoring Feedback Loop</div>', unsafe_allow_html=True)
            st.markdown('<div class="hitl-body" style="margin-bottom:12px;">Simulate resolving this application 6 months later to track model drift and supply ground truth for retraining.</div>', unsafe_allow_html=True)
            
            fb1, fb2, _ = st.columns([1, 1, 2])
            with fb1:
                if st.button("✅ Simulate: Loan Repaid", key="btn_repaid", use_container_width=True):
                    payload = {**input_features, **demographic_features, "prediction_prob": float(prob), "ground_truth": 0}
                    try:
                        resp = requests.post(f"{API_URL}/feedback", json=payload, timeout=5)
                        if resp.status_code == 200:
                            st.success("Feedback logged to local DB for drift monitoring!")
                    except Exception as e:
                        st.error(f"Failed to submit feedback (Ensure FastAPI is running): {e}")
            with fb2:
                if st.button("❌ Simulate: Loan Defaulted", key="btn_defaulted", use_container_width=True):
                    payload = {**input_features, **demographic_features, "prediction_prob": float(prob), "ground_truth": 1}
                    try:
                        resp = requests.post(f"{API_URL}/feedback", json=payload, timeout=5)
                        if resp.status_code == 200:
                            st.success("Feedback logged to local DB for drift monitoring!")
                    except Exception as e:
                        st.error(f"Failed to submit feedback (Ensure FastAPI is running): {e}")

            # ── AI Loan Officer Report ────────────────────────────────────────
            st.markdown('<div class="section-head" style="margin-top:32px">🤖 AI Loan Officer Assessment</div>', unsafe_allow_html=True)

            groq_key = os.environ.get("GROQ_API_KEY", "")
            client   = get_groq_client(groq_key)

            if client:
                with st.spinner("FairCredit Advisor is reviewing this case..."):
                    try:
                        report = generate_assessment(client, st.session_state.current_applicant)
                        st.session_state.advisor_report = report
                        st.session_state.chat_history   = [
                            {"role": "user",      "content": f"Assessment for {st.session_state.selected_persona}, score {prob:.2f}"},
                            {"role": "assistant", "content": report}
                        ]
                        st.session_state.assessment_done = True
                    except Exception as e:
                        st.error(f"AI Advisor error: {e}")
            else:
                st.warning("⚠️ Add your Groq API key in the sidebar to enable the AI Loan Officer report. Get one free at console.groq.com")

    # ── Show AI report and chatbot ────────────────────────────────────────────
    if st.session_state.assessment_done and st.session_state.advisor_report:
        st.markdown(f"""
        <div class="advisor-panel">
            <div class="advisor-header">
                <div class="advisor-avatar">🌍</div>
                <div>
                    <div class="advisor-name">FairCredit Advisor</div>
                    <div class="advisor-title">Senior Loan Officer · Financial Inclusion Specialist · Kigali · Powered by Llama 3</div>
                </div>
            </div>
            <div class="advisor-body">{st.session_state.advisor_report.replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-head" style="margin-top:24px">💬 Ask the FairCredit Advisor</div>', unsafe_allow_html=True)
        st.caption("Ask follow-up questions about this case, alternative loan structures, SDG alignment, or fairness implications.")

        if st.session_state.follow_up_messages:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for msg in st.session_state.follow_up_messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-msg-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-msg-ai">🌍 {msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_q    = st.text_input("Your question...",
                placeholder="e.g. What if the farmer enrolls in mobile money first? / How does this connect to SDG 10?")
            submitted = st.form_submit_button("Ask →", use_container_width=True)

        if submitted and user_q.strip():
            groq_key = os.environ.get("GROQ_API_KEY", "")
            client   = get_groq_client(groq_key)
            if client:
                with st.spinner("FairCredit Advisor is thinking..."):
                    try:
                        answer = chat_with_advisor(
                            client, user_q,
                            st.session_state.current_applicant,
                            st.session_state.chat_history
                        )
                        st.session_state.follow_up_messages.append({"role": "user",      "content": user_q})
                        st.session_state.follow_up_messages.append({"role": "assistant", "content": answer})
                        st.session_state.chat_history.append({"role": "user",      "content": user_q})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="cmu-banner">
        <img class="cmu-img"
             src="https://images.unsplash.com/photo-1607237138185-eedd9c632b0b?w=600&q=80"
             alt="CMU Africa" onerror="this.style.display='none'"/>
        <div class="cmu-text">
            <div class="cmu-title">Carnegie Mellon University Africa · Kigali</div>
            <div class="cmu-sub">Research in AI fairness and financial inclusion for underserved
            communities across East and Central Africa. This tool demonstrates responsible AI
            in high-stakes lending decisions.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Why Fairness Matters in Credit Scoring</div>', unsafe_allow_html=True)
    st.markdown("""
    Traditional credit scoring **systematically excludes** millions of Africans who lack
    formal credit history — not because they are bad borrowers, but because they operate
    outside the formal financial system. This model uses alternative signals to assess
    risk more equitably.
    """)

    st.markdown('<div class="section-head">Protected Attributes Monitored</div>', unsafe_allow_html=True)
    for name, col, sdg in [
        ("Gender","CODE_GENDER","SDG 10 — Women face systemic barriers to credit access"),
        ("Education Level","NAME_EDUCATION_TYPE","SDG 10 — Low formal education ≠ inability to repay"),
        ("Employment Type","NAME_INCOME_TYPE","SDG 8 — Informal workers deserve fair assessment"),
        ("Housing Situation","NAME_HOUSING_TYPE","SDG 10 — Renting ≠ financial irresponsibility"),
    ]:
        st.markdown(f"""
        <div class="fairness-item">
            <div class="fi-left"><strong>{name}</strong><code>{col}</code></div>
            <div class="fi-right">{sdg}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Alternative Credit Signals — Our African Adaptation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1A1E28;border:1px solid #252A38;border-left:4px solid #C8922A;
                border-radius:12px;padding:20px 24px;margin-bottom:16px;line-height:1.8;">
        <div style="font-size:0.68rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;
                    color:#C8922A;margin-bottom:10px">⚠️ Important Demo Note</div>
        <div style="font-size:0.88rem;color:rgba(237,232,222,0.8)">
            The original Home Credit dataset contains three external credit bureau scores
            (<strong style="color:#E8B84B">EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3</strong>)
            sourced from third-party agencies — largely unavailable to most Africans.<br><br>
            We relabeled these using African-relevant alternatives:<br><br>
            &nbsp;&nbsp;• <strong style="color:#86efac">No credit history</strong> → 0.08 — invisible to the system<br>
            &nbsp;&nbsp;• <strong style="color:#86efac">Mobile money user</strong> → 0.35 — MTN MoMo, Airtel Money<br>
            &nbsp;&nbsp;• <strong style="color:#86efac">Has savings account</strong> → 0.55 — formal banking engagement<br>
            &nbsp;&nbsp;• <strong style="color:#86efac">Has previous loan</strong> → 0.75 — proven repayment history<br><br>
            <strong style="color:#E8B84B">One mobile money signal can be the difference between
            approval and exclusion. That is SDG 10 in action.</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Human-in-the-Loop Principle</div>', unsafe_allow_html=True)
    st.markdown("""
    > *"The model doesn't decide — it informs. A human loan officer always makes the final call.
    This prevents algorithmic discrimination against people with thin credit files."*

    **Fairness metrics evaluated during training:** Demographic Parity Difference,
    Equalized Odds Difference, per-group Recall, Precision, and AUC.
    """)

    st.markdown('<div class="section-head">Persona Risk Profiles</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Persona": ["👩‍💼 Government Worker","👩‍🎓 CMU Graduate","🏪 Market Vendor",
                    "🚗 Boda Boda Rider","🏠 Single Mother","👨‍🌾 Smallholder Farmer"],
        "Credit Signals": ["Has previous loan","Has savings account","Mobile money",
                           "Mixed","No credit history","No credit history"],
        "Expected Zone": ["🟢 Low","🟢 Low-Medium","🟡 Medium",
                          "🟡 Medium-High","🔴 High","🔴 Very High"],
        "Key Factor": [
            "Stable salary, proven repayment",
            "Thin file but stable employment",
            "Informal income, some digital trail",
            "Young, high debt-to-income ratio",
            "Low income, high dependants",
            "No income record, no credit signal",
        ]
    }), hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">🚀 MLOps Control Center</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1A1E28;border:1px solid #252A38;border-left:4px solid #22C55E;
                border-radius:12px;padding:18px 22px;margin-bottom:20px;">
        <span style="color:#86efac;font-weight:600;font-size:0.9rem;">System Architect View</span><br>
        <span style="color:#C8D0DC;font-size:0.85rem;line-height:1.6;">
        Monitor live data concept drift and trigger the continuous retraining pipeline directly from this application. 
        All metrics below are pulled in real-time from the local MLflow tracking server.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    import mlflow
    from mlflow.tracking import MlflowClient
    import subprocess
    import sys
    import time
    
    MLFLOW_URI = "http://localhost:5000"
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    try:
        client = MlflowClient(tracking_uri=MLFLOW_URI)
    except Exception as e:
        st.error(f"Could not connect to MLflow: {e}")
        client = None

    def get_experiment_metrics(exp_name, metric_name):
        if not client: return pd.DataFrame()
        exp = client.get_experiment_by_name(exp_name)
        if not exp: return pd.DataFrame()
        
        runs = client.search_runs(exp.experiment_id, order_by=["start_time ASC"])
        data = []
        for r in runs:
            if metric_name in r.data.metrics:
                data.append({"time": r.info.start_time, "value": r.data.metrics[metric_name]})
        
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        return df

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="form-card-title">📉 System Health: Live Data Drift Detection</div>', unsafe_allow_html=True)
        st.caption("Monitoring real-world data distributions against the AI's training baseline. Drops below the stability line indicate macroeconomic shifts.")
        
        df_drift = get_experiment_metrics("Credit_Scoring_Drift_Monitoring", "prediction_prob_p_value")
        if not df_drift.empty:
            st.line_chart(df_drift, color="#EAB308", height=200)
            latest_p = df_drift["value"].iloc[-1]
            stability_pct = latest_p * 100
            baseline_stability = 75.0
            drift_deviation = stability_pct - baseline_stability
            
            if latest_p < 0.05:
                status_color = "🔴"
                status_text = "**Critical Shift**: Market data drifting away from system baseline."
            elif latest_p < 0.20:
                status_color = "🟡"
                status_text = "**Warning**: Market data showing early signs of shifting."
            else:
                status_color = "🟢"
                status_text = "**Stable**: Market data correctly matches training baseline."
                
            st.metric(
                label="Data Stability Index (Baseline: 75.0%)",
                value=f"{stability_pct:.1f}%",
                delta=f"{drift_deviation:+.1f}% deviation",
                delta_color="normal"
            )
            st.markdown(f"<div style='margin-top: -10px; font-size:14px; font-weight: 500;'>Status: {status_color} {status_text}</div>", unsafe_allow_html=True)
        else:
            st.info("No drift metrics available. Run Drift Monitor first.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Run Drift Monitor Batch Scan", use_container_width=True):
            with st.spinner("Analyzing feedback database for statistical drift..."):
                run_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                res = subprocess.run([sys.executable, "monitor_drift.py"], capture_output=True, text=True, env=run_env)
                if res.returncode == 0:
                    st.toast("Drift monitoring complete! MLflow dashboard updated.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Error: {res.stderr}")

    with col2:
        st.markdown('<div class="form-card-title">📈 AI Intelligence: Automated Retraining</div>', unsafe_allow_html=True)
        st.caption("Tracking the model's predictive accuracy capability as it continuously learns from incoming loan manager feedback.")
        
        df_auc = get_experiment_metrics("Credit_Scoring_Retraining", "roc_auc")
        if not df_auc.empty:
            st.line_chart(df_auc, color="#22C55E", height=200)
            current_auc = df_auc["value"].iloc[-1]
            baseline_auc = 0.7642
            deviation_pct = ((current_auc - baseline_auc) / baseline_auc) * 100
            
            if deviation_pct < -5.0:
                health_status = "🔴 High Degradation (Emergency review required)"
            elif deviation_pct < -1.0:
                health_status = "🟡 Medium Degradation (Monitor model closely)"
            elif deviation_pct > 2.0:
                health_status = "🟢 High Improvement (New baseline achieved)"
            else:
                health_status = "🟢 Stable (Optimal system performance)"
                
            st.metric(
                label="Latest System Accuracy (Baseline: 0.7642)",
                value=f"{current_auc:.4f}",
                delta=f"{deviation_pct:+.2f}% deviation"
            )
            st.markdown(f"<div style='margin-top: -10px; font-size:14px; font-weight: 500;'>Status: {health_status}</div>", unsafe_allow_html=True)
        else:
            st.info("No retraining metrics available.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Trigger Model Retraining Pipeline", type="primary", use_container_width=True):
            with st.spinner("Fine-tuning LightGBM on new feedback data. This may take a moment..."):
                run_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                res = subprocess.run([sys.executable, "retrain_pipeline.py"], capture_output=True, text=True, env=run_env)
                if res.returncode == 0:
                    out_text = str(res.stdout) + "\n" + str(res.stderr)
                    if "Not enough data" in out_text:
                        st.warning("Skipped: Not enough new feedback samples to safely retrain right now.")
                    else:
                        st.toast("Retraining pipeline successfully completed! New model saved.")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error(f"Error during retraining: {res.stderr}")
