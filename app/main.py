"""
app/main.py — Fraud Detection Dashboard
Run: streamlit run app/main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import streamlit as st

from src.preprocessor import FraudPreprocessor
from src.models import train_all, metrics_table, evaluate
from src.explainer import FraudExplainer
from src.business_impact import BusinessImpact

# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 14px; }
.stApp { background: #08090e; color: #d1d5db; }

section[data-testid="stSidebar"] {
    background: #0d0f17 !important;
    border-right: 1px solid #1c2033 !important;
}
section[data-testid="stSidebar"] * { color: #6b7280 !important; }
section[data-testid="stSidebar"] h2 {
    font-family: 'Syne', sans-serif !important;
    color: #f87171 !important; font-size: 13px !important;
    font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
}

.topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding: 0.6rem 1.5rem; background: #0d0f17;
    border-bottom: 1px solid #1c2033; margin-bottom: 1.5rem;
}
.topbar-logo { font-family:'Syne',sans-serif; font-size:1.1rem;
    font-weight:800; color:#f9fafb; }
.topbar-logo span { color:#f87171; }
.topbar-tag { font-family:'JetBrains Mono',monospace; font-size:10px;
    padding:2px 8px; border-radius:3px; background:rgba(248,113,113,0.1);
    border:1px solid rgba(248,113,113,0.25); color:#f87171;
    letter-spacing:0.08em; margin-left:8px; }
.topbar-right { font-family:'JetBrains Mono',monospace;
    font-size:11px; color:#374151; letter-spacing:0.05em; }

.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr);
    gap:1px; background:#1c2033; border:1px solid #1c2033;
    border-radius:8px; overflow:hidden; margin-bottom:1.5rem; }
.kpi-card { background:#0d0f17; padding:1.1rem 1.25rem; }
.kpi-label { font-family:'JetBrains Mono',monospace; font-size:10px;
    color:#4b5563; text-transform:uppercase; letter-spacing:0.12em;
    margin-bottom:6px; }
.kpi-value { font-family:'JetBrains Mono',monospace; font-size:1.4rem;
    font-weight:500; line-height:1; margin-bottom:4px; }
.kpi-sub { font-size:11px; color:#4b5563;
    font-family:'JetBrains Mono',monospace; }
.red { color:#f87171; } .green { color:#34d399; }
.blue { color:#60a5fa; } .amber { color:#fbbf24; }
.purple { color:#a78bfa; }

.sec-head { display:flex; align-items:center; gap:10px;
    margin:1.5rem 0 0.9rem; padding-bottom:8px;
    border-bottom:1px solid #1c2033; }
.sec-title { font-family:'Syne',sans-serif; font-size:12px;
    font-weight:700; text-transform:uppercase;
    letter-spacing:0.12em; color:#9ca3af; }
.sec-line { flex:1; height:1px; background:#1c2033; }
.sec-badge { font-family:'JetBrains Mono',monospace; font-size:10px;
    padding:2px 7px; border-radius:3px;
    background:rgba(59,130,246,0.08); border:1px solid #1c2033;
    color:#374151; }

.alert-box { background:rgba(248,113,113,0.08);
    border:1px solid rgba(248,113,113,0.25); border-radius:6px;
    padding:1rem 1.25rem; margin-bottom:1rem; }
.alert-title { font-family:'Syne',sans-serif; font-size:14px;
    font-weight:700; color:#f87171; margin-bottom:4px; }
.alert-body { font-family:'JetBrains Mono',monospace; font-size:12px;
    color:#9ca3af; line-height:1.7; }

.legit-box { background:rgba(52,211,153,0.08);
    border:1px solid rgba(52,211,153,0.25); border-radius:6px;
    padding:1rem 1.25rem; margin-bottom:1rem; }
.legit-title { font-family:'Syne',sans-serif; font-size:14px;
    font-weight:700; color:#34d399; margin-bottom:4px; }

.stTabs [data-baseweb="tab-list"] {
    background:#0d0f17; border-bottom:1px solid #1c2033; gap:0; padding:0; }
.stTabs [data-baseweb="tab"] {
    font-family:'Inter',sans-serif !important; font-size:13px !important;
    font-weight:500; color:#4b5563 !important; padding:10px 22px !important;
    border-radius:0 !important; border-bottom:2px solid transparent; }
.stTabs [aria-selected="true"] {
    color:#f9fafb !important;
    border-bottom:2px solid #f87171 !important;
    background:transparent !important; }

.stButton > button { background:#f87171 !important; color:white !important;
    border:none !important; border-radius:6px !important;
    font-family:'Inter',sans-serif !important; font-weight:600 !important;
    font-size:13px !important; width:100% !important; }
.stButton > button:hover { background:#ef4444 !important; }

.stDataFrame thead th { background:#111420 !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:11px !important; color:#6b7280 !important; }
.stDataFrame tbody td { font-family:'JetBrains Mono',monospace !important;
    font-size:12px !important; color:#9ca3af !important; }

[data-testid="metric-container"] { background:#0d0f17;
    border:1px solid #1c2033; border-radius:6px; padding:0.75rem 1rem; }
[data-testid="metric-container"] label {
    font-family:'JetBrains Mono',monospace !important;
    font-size:10px !important; color:#4b5563 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family:'JetBrains Mono',monospace !important;
    font-size:1.1rem !important; color:#f9fafb !important; }

hr { border-color:#1c2033 !important; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:#08090e;
    border-bottom:1px solid #1c2033; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Chart theme helper
# ─────────────────────────────────────────────────────────────
BG     = "#08090e"
PANEL  = "#0d0f17"
GRID   = "rgba(255,255,255,0.04)"
BORDER = "#1c2033"
TEXT   = "#6b7280"
RED    = "#f87171"
GREEN  = "#34d399"
BLUE   = "#3b82f6"
AMBER  = "#fbbf24"

def dark_layout(fig, height=340, xt="", yt="", xfmt="", yfmt=""):
    fig.update_layout(height=height, plot_bgcolor=PANEL, paper_bgcolor=PANEL,
        font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
        margin=dict(l=55, r=20, t=30, b=45),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT)))
    fig.update_xaxes(title_text=xt, tickformat=xfmt,
        gridcolor=GRID, linecolor=BORDER, tickfont=dict(color=TEXT, size=10))
    fig.update_yaxes(title_text=yt, tickformat=yfmt,
        gridcolor=GRID, linecolor=BORDER, tickfont=dict(color=TEXT, size=10))
    return fig

# ─────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div>
    <span class="topbar-logo">FRAUD<span>SHIELD</span></span>
    <span class="topbar-tag">ML DETECTION</span>
    <span class="topbar-tag">SHAP EXPLAINABILITY</span>
  </div>
  <div class="topbar-right">
    NIRMIT PATEL &nbsp;·&nbsp; MBA CANDIDATE &nbsp;·&nbsp; PACE UNIVERSITY
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Detection Config")
    st.markdown("---")

    fp_cost = st.slider("FP cost ($) — false alert",
                         0.5, 20.0, 2.0, 0.5)
    fn_cost = st.slider("FN cost ($) — missed fraud",
                         50.0, 500.0, 150.0, 10.0)
    threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01,
        help="Lower = catch more fraud but more false alerts")

    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("Kaggle Credit Card Fraud Detection\n284,807 transactions · 492 fraud cases (0.17%)")

    st.markdown("---")
    run = st.button("▶  Train & Analyze")

# ─────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────
if not run:
    st.markdown("""
    <div style="max-width:700px;margin:2rem auto;">
      <div style="font-family:'Syne',sans-serif;font-size:2.2rem;
           font-weight:800;color:#f9fafb;line-height:1.1;margin-bottom:1rem;">
        ML-powered fraud detection<br>
        <span style="color:#f87171;">with SHAP explainability</span>
      </div>
      <div style="font-size:14px;color:#6b7280;line-height:1.8;margin-bottom:2rem;">
        Trains 4 models on 284,807 real credit card transactions.
        Uses SHAP to explain every prediction — not just who gets flagged,
        but <em>why</em>. Includes business cost analysis and optimal threshold selection.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, title, desc, color in [
        (c1, "4 ML Models", "LR · RF · XGBoost · LightGBM", "#3b82f6"),
        (c2, "SHAP Analysis", "Global + local explainability", "#f87171"),
        (c3, "Business Cost", "Optimal threshold selection", "#fbbf24"),
        (c4, "Live Scorer", "Score any transaction in real time", "#34d399"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0d0f17;border:1px solid #1c2033;
                 border-top:2px solid {color};border-radius:6px;
                 padding:1.1rem;height:90px;">
              <div style="font-family:'Syne',sans-serif;font-weight:700;
                   font-size:13px;color:#f9fafb;margin-bottom:6px;">{title}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                   color:#4b5563;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:2rem;font-family:'JetBrains Mono',
         monospace;font-size:12px;color:#374151;">
      Download creditcard.csv from Kaggle → place in data/ → click ▶ Train & Analyze
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# Load data & train
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on 284k transactions...")
def load_and_train():
    prep = FraudPreprocessor()
    df   = prep.load()
    X_train, X_test, y_train, y_test, feats = prep.preprocess(df, balance=True)
    results = train_all(X_train, X_test, y_train, y_test, threshold=0.5)

    # Build SHAP explainer on best model
    best_name = max(results, key=lambda k: results[k]["metrics"]["auc_pr"])
    best_model = results[best_name]["model"]
    idx = np.random.RandomState(42).choice(X_test.shape[0], 300, replace=False)
    explainer = FraudExplainer(best_model, X_train[:300], feats)

    return prep, df, X_train, X_test, y_train, y_test, feats, results, best_name, explainer, idx

try:
    prep, df, X_train, X_test, y_train, y_test, feats, results, best_name, explainer, idx = load_and_train()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Re-evaluate with current threshold
best_model = results[best_name]["model"]
m = evaluate(best_model, X_test, y_test, threshold=threshold)
bi = BusinessImpact(fp_cost=fp_cost, fn_cost=fn_cost)
impact = bi.compute_savings(m["tp"], m["fp"], m["tn"], m["fn"])

# ─────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">AUC-ROC</div>
    <div class="kpi-value green">{m['auc_roc']:.4f}</div>
    <div class="kpi-sub">best: {best_name}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">AUC-PR</div>
    <div class="kpi-value green">{m['auc_pr']:.4f}</div>
    <div class="kpi-sub">precision-recall</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Recall</div>
    <div class="kpi-value {'green' if m['recall']>0.8 else 'amber'}">{m['recall']:.1%}</div>
    <div class="kpi-sub">fraud caught</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Precision</div>
    <div class="kpi-value blue">{m['precision']:.1%}</div>
    <div class="kpi-sub">alerts correct</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Savings / 1k txns</div>
    <div class="kpi-value green">${impact['savings_per_1000_txns']:,.0f}</div>
    <div class="kpi-sub">vs no model</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Model Performance",
    "🔍  SHAP Explainability",
    "💰  Business Impact",
    "🎯  Live Transaction Scorer",
])

# ══ TAB 1 — Model Performance ══════════════════════════════════
with tab1:
    st.markdown("""<div class="sec-head">
      <div class="sec-title">Model Comparison</div>
      <div class="sec-line"></div>
      <div class="sec-badge">all 4 models</div>
    </div>""", unsafe_allow_html=True)
    st.dataframe(metrics_table(results), use_container_width=True)

    cl, cr = st.columns(2, gap="large")

    with cl:
        st.markdown("""<div class="sec-head">
          <div class="sec-title">ROC Curves</div>
          <div class="sec-line"></div>
        </div>""", unsafe_allow_html=True)

        fig_roc = go.Figure()
        colors_map = {
            "Logistic Regression": "#6b7280",
            "Random Forest":       "#fbbf24",
            "XGBoost":             RED,
            "LightGBM":            BLUE,
        }
        for name, r in results.items():
            m2 = r["metrics"]
            fig_roc.add_trace(go.Scatter(
                x=m2["fpr"], y=m2["tpr"], mode="lines",
                name=f"{name} ({m2['auc_roc']:.3f})",
                line=dict(color=colors_map.get(name, BLUE), width=2),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#374151", dash="dot", width=1),
            showlegend=False,
        ))
        dark_layout(fig_roc, height=340, xt="False Positive Rate",
                    yt="True Positive Rate")
        fig_roc.update_layout(showlegend=True)
        st.plotly_chart(fig_roc, use_container_width=True)

    with cr:
        st.markdown("""<div class="sec-head">
          <div class="sec-title">Precision-Recall Curves</div>
          <div class="sec-line"></div>
          <div class="sec-badge">better metric for imbalanced data</div>
        </div>""", unsafe_allow_html=True)

        fig_pr = go.Figure()
        for name, r in results.items():
            m2 = r["metrics"]
            fig_pr.add_trace(go.Scatter(
                x=m2["rec_curve"], y=m2["prec_curve"], mode="lines",
                name=f"{name} ({m2['auc_pr']:.3f})",
                line=dict(color=colors_map.get(name, BLUE), width=2),
            ))
        dark_layout(fig_pr, height=340, xt="Recall", yt="Precision")
        fig_pr.update_layout(showlegend=True)
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("""<div class="sec-head">
      <div class="sec-title">Confusion Matrix</div>
      <div class="sec-line"></div>
    </div>""", unsafe_allow_html=True)

    cm_data = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig_cm = go.Figure(go.Heatmap(
        z=cm_data,
        x=["Predicted: Legit", "Predicted: Fraud"],
        y=["Actual: Legit", "Actual: Fraud"],
        text=[[str(m["tn"]), str(m["fp"])], [str(m["fn"]), str(m["tp"])]],
        texttemplate="%{text}",
        textfont=dict(size=16, color="white",
                      family="JetBrains Mono, monospace"),
        colorscale=[[0, PANEL], [1, RED]],
        showscale=False,
    ))
    dark_layout(fig_cm, height=260)
    st.plotly_chart(fig_cm, use_container_width=True)

# ══ TAB 2 — SHAP ══════════════════════════════════════════════
with tab2:
    st.markdown("""<div class="sec-head">
      <div class="sec-title">Global Feature Importance</div>
      <div class="sec-line"></div>
      <div class="sec-badge">mean |SHAP| across 300 transactions</div>
    </div>""", unsafe_allow_html=True)

    X_sample = X_test[idx]
    fig_global = explainer.global_importance_plotly(X_sample, top_n=15)
    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("""<div class="sec-head">
      <div class="sec-title">Transaction-level Explanation</div>
      <div class="sec-line"></div>
      <div class="sec-badge">select any transaction</div>
    </div>""", unsafe_allow_html=True)

    # Pick fraud and legit examples
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]

    col_sel, col_thresh = st.columns([2, 1])
    with col_sel:
        txn_type = st.radio("Transaction type",
                            ["Fraud transaction", "Legitimate transaction"],
                            horizontal=True)
    with col_thresh:
        if txn_type == "Fraud transaction":
            txn_num = st.slider("Transaction #", 0,
                                min(20, len(fraud_idx)-1), 0)
            chosen_idx = fraud_idx[txn_num]
        else:
            txn_num = st.slider("Transaction #", 0,
                                min(20, len(legit_idx)-1), 0)
            chosen_idx = legit_idx[txn_num]

    x_single = X_test[chosen_idx]
    prob = float(best_model.predict_proba(x_single.reshape(1, -1))[0, 1])
    actual = int(y_test[chosen_idx])

    if prob >= threshold:
        st.markdown(f"""
        <div class="alert-box">
          <div class="alert-title">🚨 FRAUD ALERT — {prob:.1%} probability</div>
          <div class="alert-body">{explainer.business_reason(x_single)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="legit-box">
          <div class="legit-title">✓ LEGITIMATE — {1-prob:.1%} confidence</div>
          <div class="alert-body" style="color:#6b7280;">
            No significant fraud signals detected. Transaction appears normal.
          </div>
        </div>
        """, unsafe_allow_html=True)

    fig_wf = explainer.waterfall_plotly(x_single, top_n=12)
    st.plotly_chart(fig_wf, use_container_width=True)

    st.caption("Red bars push toward fraud · Green bars push toward legitimate")

# ══ TAB 3 — Business Impact ═══════════════════════════════════
with tab3:
    st.markdown("""<div class="sec-head">
      <div class="sec-title">Cost Analysis — All Models</div>
      <div class="sec-line"></div>
    </div>""", unsafe_allow_html=True)
    st.dataframe(bi.savings_summary_table(results), use_container_width=True)

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Fraud caught",         f"{impact['fraud_caught_pct']:.1%}")
    bm2.metric("False alert rate",     f"{impact['false_alert_rate']:.2%}")
    bm3.metric("Savings vs no model",  f"${impact['net_savings_vs_no_model']:,.0f}")
    bm4.metric("Savings vs rules",     f"${impact['net_savings_vs_naive_rule']:,.0f}")

    st.markdown("""<div class="sec-head">
      <div class="sec-title">Optimal Threshold — Business Cost Curve</div>
      <div class="sec-line"></div>
    </div>""", unsafe_allow_html=True)
    fig_cost = bi.threshold_cost_curve(best_model, X_test, y_test)
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown(f"""
    <div style="background:#0d0f17;border:1px solid #1c2033;border-left:3px solid #fbbf24;
         padding:1rem 1.25rem;font-family:'JetBrains Mono',monospace;
         font-size:12px;color:#6b7280;line-height:1.9;">
      <span style="color:#fbbf24;font-weight:500;">Assumption:</span>
      FP cost = ${fp_cost:.2f} (manual review friction) ·
      FN cost = ${fn_cost:.2f} (chargeback + investigation) ·
      Ratio = {fn_cost/fp_cost:.0f}:1<br>
      At this cost ratio, the model saves
      <span style="color:#34d399;">${impact['savings_per_1000_txns']:,.2f}</span>
      per 1,000 transactions compared to no detection system.
    </div>
    """, unsafe_allow_html=True)

# ══ TAB 4 — Live Scorer ═══════════════════════════════════════
with tab4:
    st.markdown("""<div class="sec-head">
      <div class="sec-title">Score a Transaction</div>
      <div class="sec-line"></div>
      <div class="sec-badge">real-time SHAP explanation</div>
    </div>""", unsafe_allow_html=True)

    st.caption("Adjust the sliders below to simulate a transaction. "
               "The model scores it instantly and SHAP explains the prediction.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        v1  = st.slider("V1",  -5.0, 5.0, -3.0, 0.1)
        v2  = st.slider("V2",  -5.0, 5.0,  0.0, 0.1)
        v3  = st.slider("V3",  -5.0, 5.0,  0.0, 0.1)
        v4  = st.slider("V4",  -5.0, 5.0,  0.5, 0.1)
        v5  = st.slider("V5",  -5.0, 5.0,  0.0, 0.1)
    with col_b:
        v10 = st.slider("V10", -5.0, 5.0, -3.5, 0.1)
        v11 = st.slider("V11", -5.0, 5.0,  0.0, 0.1)
        v12 = st.slider("V12", -5.0, 5.0, -4.0, 0.1)
        v14 = st.slider("V14", -5.0, 5.0, -5.0, 0.1)
        v16 = st.slider("V16", -5.0, 5.0,  0.0, 0.1)
    with col_c:
        amount = st.slider("Amount ($)", 0.0, 5000.0, 1200.0, 10.0)
        v17    = st.slider("V17", -5.0, 5.0, 0.0, 0.1)
        v18    = st.slider("V18", -5.0, 5.0, 0.0, 0.1)
        v19    = st.slider("V19", -5.0, 5.0, 0.0, 0.1)
        v21    = st.slider("V21", -5.0, 5.0, 0.0, 0.1)

    # Build a feature vector matching the training data format
    feature_vals = {f: 0.0 for f in feats}
    manual_vals  = {
        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
        "V10": v10, "V11": v11, "V12": v12, "V14": v14,
        "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V21": v21,
        "Amount_scaled": amount / 2500.0,
        "Time_scaled":   0.5,
    }
    for k, v in manual_vals.items():
        if k in feature_vals:
            feature_vals[k] = v

    x_manual = np.array([feature_vals[f] for f in feats])
    prob_manual = float(best_model.predict_proba(
        x_manual.reshape(1, -1))[0, 1])

    if prob_manual >= threshold:
        st.markdown(f"""
        <div class="alert-box">
          <div class="alert-title">🚨 FRAUD ALERT — {prob_manual:.1%} probability</div>
          <div class="alert-body">{explainer.business_reason(x_manual)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="legit-box">
          <div class="legit-title">✓ LEGITIMATE — {1-prob_manual:.1%} confidence</div>
          <div class="alert-body" style="color:#6b7280;">
            Transaction scored as legitimate at current threshold ({threshold}).
          </div>
        </div>
        """, unsafe_allow_html=True)

    fig_manual = explainer.waterfall_plotly(x_manual, top_n=12)
    st.plotly_chart(fig_manual, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.25rem 0;border-top:1px solid #1c2033;
     display:flex;justify-content:space-between;align-items:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#374151;">
    Built by <span style="color:#9ca3af;">Nirmit Patel</span>
    &nbsp;·&nbsp; MBA Candidate · Pace University · Business Analytics
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;display:flex;gap:16px;">
    <a href="https://github.com/Nirmitpatel889" target="_blank"
       style="color:#f87171;text-decoration:none;">GitHub</a>
    <a href="https://nirmit-patel-portfolio.netlify.app" target="_blank"
       style="color:#f87171;text-decoration:none;">Portfolio</a>
  </div>
</div>
""", unsafe_allow_html=True)
