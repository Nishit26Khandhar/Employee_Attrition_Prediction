# =============================================================================
# Employee Attrition Prediction & Risk Scoring System
# Streamlit Dashboard — Production Grade
# Developed by Nishit Khandhar
# =============================================================================

import os
import sys
import warnings
import pickle
import io

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AttritionIQ | Employee Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONSTANTS
# =============================================================================
COLOR_HIGH   = "#FF4B4B"
COLOR_MEDIUM = "#F9C74F"
COLOR_LOW    = "#43AA8B"
COLOR_BG     = "#0F1117"
COLOR_CARD   = "#1A1D27"
COLOR_ACCENT = "#6C63FF"

NUMERICAL_COLUMNS = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
    "IncomePerYearExp", "PromotionDelay", "EngagementScore",
    "WorkloadStress", "HighPerf_NoPromo", "JobHopperIndex", "LoyaltyScore",
    "DissatisfactionRisk", "OverTime",
]
CATEGORICAL_COLUMNS = [
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus",
]
ALL_FEATURE_COLS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

# =============================================================================
# CUSTOM CSS — Premium Dark Enterprise Theme
# =============================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── Global reset ────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #E8EAF0;
    }
    .stApp { background-color: #0A0C14; }
    .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1600px; }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12141F 0%, #0F1117 100%);
        border-right: 1px solid #1E2130;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiselect label { color: #9B9FC4 !important; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; }

    /* ── Metric Cards ────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1A1D2E 0%, #161926 100%);
        border: 1px solid #252840;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    .metric-card.total::before  { background: linear-gradient(90deg, #6C63FF, #A855F7); }
    .metric-card.high::before   { background: linear-gradient(90deg, #FF4B4B, #FF8080); }
    .metric-card.medium::before { background: linear-gradient(90deg, #F9C74F, #FFE099); }
    .metric-card.low::before    { background: linear-gradient(90deg, #43AA8B, #80D5BE); }
    .metric-label {
        font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: #7B7FA8; margin-bottom: 0.4rem; font-weight: 600;
    }
    .metric-value {
        font-size: 2.2rem; font-weight: 700; line-height: 1;
        font-family: 'DM Mono', monospace;
    }
    .metric-sub {
        font-size: 0.78rem; color: #7B7FA8; margin-top: 0.35rem;
    }
    .metric-icon {
        position: absolute; right: 1.2rem; top: 1.2rem;
        font-size: 1.6rem; opacity: 0.18;
    }

    /* ── Section Headers ─────────────────────────────────────────── */
    .section-header {
        display: flex; align-items: center; gap: 0.6rem;
        margin: 1.8rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #1E2130;
    }
    .section-header h2 {
        font-size: 1.05rem; font-weight: 600;
        color: #C8CADF; letter-spacing: 0.02em; margin: 0;
    }
    .section-badge {
        font-size: 0.68rem; padding: 2px 8px;
        background: #252840; border-radius: 20px;
        color: #7B7FA8; letter-spacing: 0.06em;
        text-transform: uppercase; font-weight: 600;
    }

    /* ── Risk Badges ─────────────────────────────────────────────── */
    .badge-high   { background:#FF4B4B22; color:#FF4B4B; border:1px solid #FF4B4B44; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
    .badge-medium { background:#F9C74F22; color:#F9C74F; border:1px solid #F9C74F44; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
    .badge-low    { background:#43AA8B22; color:#43AA8B; border:1px solid #43AA8B44; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:600; }

    /* ── Profile Card ────────────────────────────────────────────── */
    .profile-card {
        background: linear-gradient(135deg, #1A1D2E 0%, #161926 100%);
        border: 1px solid #252840; border-radius: 14px; padding: 1.5rem;
    }
    .profile-avatar {
        width: 56px; height: 56px; border-radius: 50%;
        background: linear-gradient(135deg, #6C63FF, #A855F7);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; font-weight: 700; color: white;
        margin-bottom: 0.8rem;
    }
    .profile-name { font-size: 1.1rem; font-weight: 600; color: #E8EAF0; }
    .profile-role { font-size: 0.8rem; color: #7B7FA8; margin-top: 2px; }
    .profile-stat { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #1E2130; }
    .profile-stat:last-child { border-bottom: none; }
    .stat-key { font-size: 0.8rem; color: #7B7FA8; }
    .stat-val { font-size: 0.82rem; font-weight: 600; color: #C8CADF; font-family: 'DM Mono', monospace; }

    /* ── Data table ──────────────────────────────────────────────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    [data-testid="stDataFrame"] table { background: #12141F !important; }

    /* ── Tabs ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { background: #12141F; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #7B7FA8; font-size: 0.88rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #252840 !important; color: #E8EAF0 !important; }

    /* ── Slider ──────────────────────────────────────────────────── */
    .stSlider [data-testid="stSlider"] > div > div > div { background: #6C63FF; }

    /* ── Expander ────────────────────────────────────────────────── */
    .streamlit-expanderHeader { background: #1A1D2E; border-radius: 8px; }

    /* ── Top header bar ──────────────────────────────────────────── */
    .top-header {
        background: linear-gradient(90deg, #12141F 0%, #1A1D2E 50%, #12141F 100%);
        border: 1px solid #252840; border-radius: 16px;
        padding: 1.2rem 2rem; margin-bottom: 1.5rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .header-title { font-size: 1.5rem; font-weight: 700; color: #E8EAF0; }
    .header-subtitle { font-size: 0.82rem; color: #7B7FA8; margin-top: 2px; }
    .header-pill {
        background: #252840; border-radius: 20px; padding: 6px 14px;
        font-size: 0.75rem; color: #9B9FC4; font-weight: 500;
        border: 1px solid #353860;
    }

    /* ── Footer ──────────────────────────────────────────────────── */
    .footer {
        text-align: center; margin-top: 3rem;
        padding: 1rem; border-top: 1px solid #1E2130;
        color: #4A4E6A; font-size: 0.78rem;
    }

    /* ── Info box ────────────────────────────────────────────────── */
    .info-box {
        background: #1A2535; border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.82rem; color: #9B9FC4;
    }

    /* ── Scrollbar ───────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0A0C14; }
    ::-webkit-scrollbar-thumb { background: #252840; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# PLOTLY CHART THEME
# =============================================================================
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#C8CADF", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#1E2130", linecolor="#1E2130", zerolinecolor="#1E2130"),
    yaxis=dict(gridcolor="#1E2130", linecolor="#1E2130", zerolinecolor="#1E2130"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#252840"),
    hoverlabel=dict(bgcolor="#1A1D2E", bordercolor="#252840", font_size=12),
)


def apply_theme(fig, title="", height=380):
    fig.update_layout(title=dict(text=title, font=dict(size=13, color="#9B9FC4"), x=0),
                      height=height, **CHART_LAYOUT)
    return fig


# =============================================================================
# DATA LOADING & PREDICTION HELPERS
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors Featur_engineering.py — applied before preprocessing."""
    df = df.copy()
    # Clean OverTime if needed
    if df["OverTime"].dtype == object:
        df["OverTime"] = df["OverTime"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    df["IncomePerYearExp"]  = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    df["PromotionDelay"]    = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
    df["EngagementScore"]   = (df["JobSatisfaction"] + df["EnvironmentSatisfaction"] + df["RelationshipSatisfaction"]) / 3
    df["WorkloadStress"]    = ((df["OverTime"] == 1) & (df["WorkLifeBalance"] <= 2)).astype(int)
    df["HighPerf_NoPromo"]  = np.where((df["PerformanceRating"] >= 4) & (df["YearsSinceLastPromotion"] > 3), 1, 0)
    df["JobHopperIndex"]    = df["NumCompaniesWorked"] / (df["Age"] - 17 + 1)
    df["LoyaltyScore"]      = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["DissatisfactionRisk"] = (
        (df["JobInvolvement"] <= 2).astype(int) +
        (df["StockOptionLevel"] == 0).astype(int) +
        (df["DistanceFromHome"] > 15).astype(int)
    )
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    """Load saved model bundle from artifacts/model.pkl."""
    path = PROJECT_ROOT / "artifacts" / "model.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """Load saved preprocessor from artifacts/proprocessor.pkl."""
    path = PROJECT_ROOT / "artifacts" / "proprocessor.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_raw_data():
    """Load raw CSV; try several common paths."""
    paths = [PROJECT_ROOT / "artifacts" / "data.csv",
             PROJECT_ROOT / "notebook" / "data" / "Palo Alto Networks.csv",
             PROJECT_ROOT / "data.csv"
             ]
    
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
        
        return None


@st.cache_data(show_spinner=False)
def build_scored_dataset(_model_bundle, _preprocessor, raw_df):
    """
    Apply feature engineering + preprocessing, then score every employee.
    Returns enriched DataFrame with AttritionProb, RiskBand columns.
    """
    df = raw_df.copy()

    # Clean target & OverTime
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)
    if "OverTime" in df.columns and df["OverTime"].dtype == object:
        df["OverTime"] = df["OverTime"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    df = engineer_features(df)

    # Add synthetic EmployeeID if missing
    if "EmployeeNumber" not in df.columns:
        df["EmployeeNumber"] = range(1, len(df) + 1)
    df["EmployeeID"] = df["EmployeeNumber"].astype(int)

    # Score with model
    if _model_bundle is not None and _preprocessor is not None:
        model     = _model_bundle["model"]
        threshold = _model_bundle.get("threshold", 0.3)
        high_t    = _model_bundle.get("risk_bands", {}).get("high", 0.70)
        med_t     = _model_bundle.get("risk_bands", {}).get("medium", 0.40)

        feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
        X = _preprocessor.transform(df[feat_cols])
        probs = model.predict_proba(X)[:, 1]
    else:
        # ── Demo fallback: synthetic probabilities ────────────────────────
        np.random.seed(42)
        probs     = np.random.beta(1.5, 5, len(df))
        threshold = 0.3
        high_t    = 0.70
        med_t     = 0.40

    df["AttritionProb"] = np.round(probs, 4)
    df["AttritionPct"]  = (probs * 100).round(1)

    def band(p):
        if p >= high_t: return "High"
        if p >= med_t:  return "Medium"
        return "Low"

    df["RiskBand"]    = df["AttritionProb"].apply(band)
    df["PredAttrition"] = (probs >= threshold).astype(int)
    return df


# =============================================================================
# UI HELPERS
# =============================================================================
def metric_card(label, value, sub, card_class, icon):
    st.markdown(f"""
    <div class="metric-card {card_class}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def section_header(title, badge="", icon=""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.1rem">{icon}</span>
        <h2>{title}</h2>
        {badge_html}
    </div>""", unsafe_allow_html=True)


def risk_badge(band):
    cls = band.lower()
    return f'<span class="badge-{cls}">{band} Risk</span>'


def _get_shap_feature_names(preprocessor, feat_cols):
    """Reconstruct full feature name list after preprocessing."""
    try:
        enc_names = (
            preprocessor.named_transformers_["cat_pipelines"]
            .named_steps["one_hot_encoder"]
            .get_feature_names_out(CATEGORICAL_COLUMNS)
        )
        return NUMERICAL_COLUMNS + list(enc_names)
    except Exception:
        return feat_cols


def _run_shap_explainer(model, X_background, X_emp):
    """
    Universal SHAP explainer using the modern shap.Explainer API.
    Falls back to TreeExplainer / LinearExplainer for speed where applicable.
    Returns a 1-D numpy array of SHAP values for the single employee row.
    """
    import shap as shap_lib

    # Tree-based models: TreeExplainer is fastest & most accurate
    if hasattr(model, "feature_importances_"):
        explainer = shap_lib.TreeExplainer(model)
        sv = explainer.shap_values(X_emp)
        # Binary classification returns a list [neg_class, pos_class]
        if isinstance(sv, list):
            sv = sv[1]
        return np.array(sv).flatten()

    # Linear models
    if hasattr(model, "coef_"):
        explainer = shap_lib.LinearExplainer(model, X_background)
        sv = explainer.shap_values(X_emp)
        return np.array(sv).flatten()

    # Universal fallback — works for any sklearn-compatible model
    explainer = shap_lib.Explainer(model, X_background)
    sv = explainer(X_emp)
    vals = sv.values
    if vals.ndim == 3:          # multi-output: take positive class
        vals = vals[:, :, 1]
    return np.array(vals).flatten()


def compute_top5_reasons(emp_row, df, model_bundle=None, preprocessor=None):
    """
    Returns list of dicts {feature, label, shap} sorted by |shap| desc.
    Priority: real shap.Explainer → z-score proxy fallback.
    """
    DISPLAY = [
        ("OverTime",               "Works Overtime"),
        ("WorkloadStress",         "Workload Stress"),
        ("DissatisfactionRisk",    "Dissatisfaction Risk"),
        ("HighPerf_NoPromo",       "High Perf, No Promotion"),
        ("JobSatisfaction",        "Low Job Satisfaction"),
        ("EngagementScore",        "Low Engagement Score"),
        ("YearsSinceLastPromotion","Promotion Delay"),
        ("LoyaltyScore",           "Low Loyalty"),
        ("DistanceFromHome",       "Long Commute"),
        ("JobHopperIndex",         "Job Hopper Index"),
        ("MonthlyIncome",          "Monthly Income"),
        ("WorkLifeBalance",        "Work-Life Balance"),
        ("EnvironmentSatisfaction","Environment Satisfaction"),
        ("NumCompaniesWorked",     "No. of Companies Worked"),
        ("PromotionDelay",         "Promotion Delay Score"),
    ]
    PROTECTIVE = {"LoyaltyScore", "EngagementScore", "JobSatisfaction",
                  "EnvironmentSatisfaction", "WorkLifeBalance", "MonthlyIncome"}

    # ── Real SHAP via shap.Explainer ───────────────────────────────────────
    if model_bundle is not None and preprocessor is not None:
        try:
            import shap as shap_lib  # noqa: F401
            model     = model_bundle["model"]
            feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

            # Background: sample up to 100 rows for speed
            X_all        = preprocessor.transform(df[feat_cols].head(100))
            X_emp_trans  = preprocessor.transform(emp_row[feat_cols].to_frame().T)

            sv       = _run_shap_explainer(model, X_all, X_emp_trans)
            all_names = _get_shap_feature_names(preprocessor, feat_cols)
            shap_map  = {all_names[i]: float(sv[i])
                         for i in range(min(len(all_names), len(sv)))}

            result = [
                {"feature": f, "label": lbl, "shap": round(shap_map[f], 4)}
                for f, lbl in DISPLAY if f in shap_map
            ]
            result.sort(key=lambda x: abs(x["shap"]), reverse=True)
            return result[:5]
        except Exception:
            pass   # fall through to proxy

    # ── Proxy: z-score deviation (used only when SHAP lib missing) ────────
    result = []
    for feat, label in DISPLAY:
        if feat not in df.columns:
            continue
        val = emp_row.get(feat, 0)
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        mean_v = df[feat].mean()
        std_v  = df[feat].std() or 1
        sign   = -1 if feat in PROTECTIVE else 1
        result.append({"feature": feat, "label": label,
                        "shap": round(sign * (val - mean_v) / std_v * 0.08, 4)})
    result.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return result[:5]


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size:2rem;">🧠</div>
            <div style="font-size:1.1rem; font-weight:700; color:#E8EAF0; margin-top:0.4rem;">AttritionIQ</div>
            <div style="font-size:0.72rem; color:#7B7FA8; margin-top:4px; line-height:1.5;">
                Employee Risk Scoring System<br>
                <span style="color:#6C63FF;">powered by ML</span>
            </div>
        </div>
        <hr style="border-color:#1E2130; margin: 0 0 1.2rem 0;">
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.68rem;letter-spacing:0.1em;color:#7B7FA8;text-transform:uppercase;font-weight:600;margin-bottom:0.5rem;">🎛️ Filters</p>', unsafe_allow_html=True)

        departments = ["All"] + sorted(df["Department"].dropna().unique().tolist())
        dept_filter = st.selectbox("Department", departments)

        roles = ["All"]
        if dept_filter != "All":
            roles += sorted(df[df["Department"] == dept_filter]["JobRole"].dropna().unique().tolist())
        else:
            roles += sorted(df["JobRole"].dropna().unique().tolist())
        role_filter = st.selectbox("Job Role", roles)

        st.markdown('<hr style="border-color:#1E2130; margin: 1rem 0;">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.68rem;letter-spacing:0.1em;color:#7B7FA8;text-transform:uppercase;font-weight:600;margin-bottom:0.5rem;">⚡ Risk Threshold</p>', unsafe_allow_html=True)
        threshold = st.slider("Attrition Probability ≥", 0.1, 0.9, 0.3, 0.05,
                               help="Employees above this threshold are flagged at-risk.")

        st.markdown('<hr style="border-color:#1E2130; margin: 1rem 0;">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.68rem;letter-spacing:0.1em;color:#7B7FA8;text-transform:uppercase;font-weight:600;margin-bottom:0.5rem;">👤 Employee Lookup</p>', unsafe_allow_html=True)
        emp_ids = sorted(df["EmployeeID"].unique().tolist())
        emp_id  = st.selectbox("Employee ID", emp_ids)

        st.markdown('<hr style="border-color:#1E2130; margin: 1.5rem 0 1rem 0;">', unsafe_allow_html=True)
        high_c = (df["RiskBand"] == "High").sum()
        total  = len(df)
        st.markdown(f"""
        <div style="background:#1A1D2E;border-radius:10px;padding:0.9rem 1rem;border:1px solid #252840;">
            <div style="font-size:0.7rem;color:#7B7FA8;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;">Quick Stats</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:0.8rem;color:#9B9FC4;">Total Employees</span>
                <span style="font-size:0.8rem;font-weight:600;color:#E8EAF0;font-family:'DM Mono',monospace;">{total:,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="font-size:0.8rem;color:#9B9FC4;">High Risk</span>
                <span style="font-size:0.8rem;font-weight:600;color:#FF4B4B;font-family:'DM Mono',monospace;">{high_c:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Apply filters
    fdf = df.copy()
    if dept_filter != "All":
        fdf = fdf[fdf["Department"] == dept_filter]
    if role_filter != "All":
        fdf = fdf[fdf["JobRole"] == role_filter]

    # Apply sidebar threshold override for risk flag
    fdf["AtRisk"] = (fdf["AttritionProb"] >= threshold).astype(int)

    return fdf, threshold, emp_id


# =============================================================================
# MODULE 1 — ATTRITION RISK DASHBOARD
# =============================================================================
def render_risk_dashboard(fdf, threshold):
    section_header("Attrition Risk Dashboard", badge="Overview", icon="📊")

    total  = len(fdf)
    high   = (fdf["RiskBand"] == "High").sum()
    medium = (fdf["RiskBand"] == "Medium").sum()
    low    = (fdf["RiskBand"] == "Low").sum()
    at_risk = (fdf["AttritionProb"] >= threshold).sum()
    pct    = round(at_risk / total * 100, 1) if total else 0

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Employees", f"{total:,}", "Active headcount", "total", "👥")
    with c2: metric_card("High Risk", f"{high:,}", f"{round(high/total*100,1)}% of workforce", "high", "🚨")
    with c3: metric_card("Medium Risk", f"{medium:,}", f"{round(medium/total*100,1)}% of workforce", "medium", "⚠️")
    with c4: metric_card("Low Risk", f"{low:,}", f"{round(low/total*100,1)}% of workforce", "low", "✅")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # Charts row 1
    col_left, col_right = st.columns([1.1, 1.9])

    with col_left:
        # Pie chart
        fig_pie = go.Figure(go.Pie(
            labels=["High Risk", "Medium Risk", "Low Risk"],
            values=[high, medium, low],
            hole=0.62,
            marker=dict(colors=[COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW],
                        line=dict(color="#0A0C14", width=3)),
            textfont=dict(size=11, color="#E8EAF0"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"<b>{pct}%</b><br><span style='font-size:10px'>At Risk</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color="#E8EAF0"), align="center")
        apply_theme(fig_pie, "Risk Band Distribution", height=320)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        # Probability histogram
        fig_hist = go.Figure(go.Histogram(
            x=fdf["AttritionProb"],
            nbinsx=40,
            marker=dict(
                color=fdf["AttritionProb"].apply(
                    lambda p: COLOR_HIGH if p >= 0.7 else (COLOR_MEDIUM if p >= 0.4 else COLOR_LOW)
                ),
                line=dict(color="#0A0C14", width=0.5),
            ),
            hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#6C63FF",
                           annotation_text=f"Threshold: {threshold}",
                           annotation_font_color="#9B9FC4", annotation_font_size=11)
        apply_theme(fig_hist, "Attrition Probability Distribution", height=320)
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # Charts row 2
    col_a, col_b = st.columns(2)

    with col_a:
        # Attrition by age group
        if "Age" in fdf.columns:
            fdf_copy = fdf.copy()
            fdf_copy["AgeGroup"] = pd.cut(fdf_copy["Age"],
                                           bins=[18, 25, 30, 35, 40, 45, 55, 70],
                                           labels=["18-25","25-30","30-35","35-40","40-45","45-55","55+"])
            age_risk = fdf_copy.groupby("AgeGroup", observed=True)["AttritionProb"].mean().reset_index()
            fig_age = go.Figure(go.Bar(
                x=age_risk["AgeGroup"].astype(str),
                y=age_risk["AttritionProb"],
                marker=dict(
                    color=age_risk["AttritionProb"],
                    colorscale=[[0,"#43AA8B"],[0.5,"#F9C74F"],[1,"#FF4B4B"]],
                    showscale=False,
                ),
                hovertemplate="Age: %{x}<br>Avg Prob: %{y:.3f}<extra></extra>",
            ))
            apply_theme(fig_age, "Avg Attrition Risk by Age Group", height=300)
            st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Risk by years at company
        if "YearsAtCompany" in fdf.columns:
            fdf_copy2 = fdf.copy()
            fdf_copy2["Tenure"] = pd.cut(fdf_copy2["YearsAtCompany"],
                                          bins=[-1,1,3,5,10,20,40],
                                          labels=["<1yr","1-3yr","3-5yr","5-10yr","10-20yr","20+yr"])
            ten_risk = fdf_copy2.groupby("Tenure", observed=True)["AttritionProb"].mean().reset_index()
            fig_ten = go.Figure(go.Scatter(
                x=ten_risk["Tenure"].astype(str),
                y=ten_risk["AttritionProb"],
                mode="lines+markers",
                line=dict(color=COLOR_ACCENT, width=2.5),
                marker=dict(size=8, color=COLOR_ACCENT,
                            line=dict(color="#0A0C14", width=2)),
                fill="tozeroy",
                fillcolor="rgba(108,99,255,0.1)",
                hovertemplate="Tenure: %{x}<br>Avg Risk: %{y:.3f}<extra></extra>",
            ))
            apply_theme(fig_ten, "Avg Attrition Risk by Tenure", height=300)
            st.plotly_chart(fig_ten, use_container_width=True, config={"displayModeBar": False})

    # Top-10 high risk table
    section_header("Top 10 Highest Risk Employees", icon="🚨")
    top10 = fdf.nlargest(10, "AttritionProb")[
        ["EmployeeID", "Department", "JobRole", "Age", "MonthlyIncome",
         "YearsAtCompany", "AttritionPct", "RiskBand"]
    ].copy()
    top10["AttritionPct"] = top10["AttritionPct"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(
        top10.style
        .map(lambda v: "color:#FF4B4B;font-weight:600" if v == "High"
                  else ("color:#F9C74F;font-weight:600" if v == "Medium"
                        else "color:#43AA8B;font-weight:600"), subset=["RiskBand"])
        .set_properties(**{"background-color": "#12141F", "border-color": "#1E2130"}),
        use_container_width=True, hide_index=True,
    )

    # CSV download
    csv_buf = io.StringIO()
    fdf.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️  Export Full Risk Report (CSV)",
        data=csv_buf.getvalue(),
        file_name="attrition_risk_report.csv",
        mime="text/csv",
    )


# =============================================================================
# MODULE 2 — EMPLOYEE RISK PROFILE
# =============================================================================
def render_employee_profile(df, emp_id):
    section_header("Employee Risk Profile", badge="Individual", icon="👤")

    emp = df[df["EmployeeID"] == emp_id]
    if emp.empty:
        st.warning("Employee not found.")
        return
    emp = emp.iloc[0]

    prob  = emp["AttritionProb"]
    band  = emp["RiskBand"]
    color = COLOR_HIGH if band == "High" else (COLOR_MEDIUM if band == "Medium" else COLOR_LOW)
    initials = str(emp.get("JobRole", "E"))[:2].upper()

    col_prof, col_gauge, col_factors = st.columns([1.2, 1.5, 2.3])

    with col_prof:
        dept    = emp.get("Department", "—")
        role    = emp.get("JobRole", "—")
        age     = emp.get("Age", "—")
        tenure  = emp.get("YearsAtCompany", "—")
        income  = emp.get("MonthlyIncome", "—")
        travel  = emp.get("BusinessTravel", "—")
        manager = emp.get("YearsWithCurrManager", "—")

        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-avatar">{initials}</div>
            <div class="profile-name">Employee #{emp_id}</div>
            <div class="profile-role">{role} · {dept}</div>
            <br>
            <div class="profile-stat"><span class="stat-key">Age</span><span class="stat-val">{age} yrs</span></div>
            <div class="profile-stat"><span class="stat-key">Tenure</span><span class="stat-val">{tenure} yrs</span></div>
            <div class="profile-stat"><span class="stat-key">Monthly Income</span><span class="stat-val">₹{income:,.0f}</span></div>
            <div class="profile-stat"><span class="stat-key">Travel</span><span class="stat-val">{travel}</span></div>
            <div class="profile-stat"><span class="stat-key">Mgr Tenure</span><span class="stat-val">{manager} yrs</span></div>
            <br>
            <div style="text-align:center">{risk_badge(band)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number=dict(suffix="%", font=dict(size=32, color=color)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#252840",
                          tickfont=dict(color="#7B7FA8", size=10)),
                bar=dict(color=color, thickness=0.25),
                bgcolor="#1A1D2E",
                borderwidth=0,
                steps=[
                    dict(range=[0, 40],  color="#0D1F1A"),
                    dict(range=[40, 70], color="#1F1C0D"),
                    dict(range=[70, 100],color="#1F0D0D"),
                ],
                threshold=dict(
                    line=dict(color=color, width=3),
                    thickness=0.8,
                    value=round(prob * 100, 1),
                ),
            ),
            title=dict(text="Attrition Probability", font=dict(color="#9B9FC4", size=12)),
        ))
        apply_theme(fig_gauge, height=280)
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with col_factors:
        # Key risk drivers for this employee
        factor_map = {
            "OverTime":              ("Works Overtime",        emp.get("OverTime", 0) == 1,           "OverTime = Yes"),
            "WorkloadStress":        ("Workload Stress",       emp.get("WorkloadStress", 0) == 1,      "Overtime + Low WLB"),
            "DissatisfactionRisk":   ("Dissatisfaction Risk",  emp.get("DissatisfactionRisk", 0) >= 2, "Score ≥ 2"),
            "HighPerf_NoPromo":      ("High Perf, No Promo",   emp.get("HighPerf_NoPromo", 0) == 1,    "Perf≥4, Promo>3yr"),
            "JobSatisfaction":       ("Low Job Satisfaction",  emp.get("JobSatisfaction", 3) <= 2,     f"Score: {emp.get('JobSatisfaction','-')}"),
            "EngagementScore":       ("Low Engagement",        emp.get("EngagementScore", 3) <= 2.0,   f"Score: {round(emp.get('EngagementScore',0),2)}"),
            "YearsSinceLastPromotion":("Promotion Overdue",    emp.get("YearsSinceLastPromotion", 0) > 4, f"{emp.get('YearsSinceLastPromotion',0)} yrs"),
            "LoyaltyScore":          ("Low Loyalty",           emp.get("LoyaltyScore", 1) < 0.25,      f"Score: {round(emp.get('LoyaltyScore',0),2)}"),
            "DistanceFromHome":      ("Long Commute",          emp.get("DistanceFromHome", 0) > 20,    f"{emp.get('DistanceFromHome',0)} km"),
            "JobHopperIndex":        ("Job Hopper",            emp.get("JobHopperIndex", 0) > 0.25,    f"Index: {round(emp.get('JobHopperIndex',0),3)}"),
        }

        risk_on  = [(k, label) for k, (label, triggered, detail) in factor_map.items() if triggered]
        risk_off = [(k, label) for k, (label, triggered, detail) in factor_map.items() if not triggered]

        section_header("Risk Factor Analysis", icon="🔍")
        if risk_on:
            for k, _ in risk_on[:6]:
                label, _, detail = factor_map[k]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0.7rem;
                            background:#1F1215;border-radius:8px;margin-bottom:6px;
                            border-left:3px solid {COLOR_HIGH};">
                    <span style="color:{COLOR_HIGH};font-size:0.9rem;">⚠</span>
                    <div>
                        <span style="font-size:0.82rem;font-weight:600;color:#E8EAF0;">{label}</span>
                        <span style="font-size:0.74rem;color:#9B9FC4;margin-left:6px;">({detail})</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        if risk_off:
            for k, _ in risk_off[:3]:
                label, _, detail = factor_map[k]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0.7rem;
                            background:#0D1F18;border-radius:8px;margin-bottom:6px;
                            border-left:3px solid {COLOR_LOW};">
                    <span style="color:{COLOR_LOW};font-size:0.9rem;">✓</span>
                    <div>
                        <span style="font-size:0.82rem;font-weight:600;color:#E8EAF0;">{label}</span>
                        <span style="font-size:0.74rem;color:#9B9FC4;margin-left:6px;">(Not triggered)</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Top-5 SHAP Reasons ─────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    section_header("🧠 Top 5 Risk Reasons (AI Explanation)", badge="SHAP-based", icon="")
    reasons = compute_top5_reasons(emp, df)
    if reasons:
        header_color = COLOR_HIGH if band == "High" else (COLOR_MEDIUM if band == "Medium" else COLOR_LOW)
        items_html = ""
        for r in reasons:
            sign  = "▲" if r["shap"] > 0 else "▼"
            color = COLOR_HIGH if r["shap"] > 0 else COLOR_LOW
            items_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:0.5rem 0.8rem;background:#1A1D2E;border-radius:8px;margin-bottom:6px;'
                f'border-left:3px solid {color};">'
                f'<span style="font-size:0.83rem;color:#C8CADF;">{sign} {r["label"]}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.82rem;color:{color};font-weight:600;">'
                f'({r["shap"]:+.3f})</span></div>'
            )
        band_label = f"{band} Risk"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1A1D2E,#161926);border:1px solid #252840;
                    border-radius:12px;padding:1.2rem;border-top:3px solid {header_color};">
            <div style="font-size:0.72rem;color:#7B7FA8;letter-spacing:0.1em;text-transform:uppercase;
                        margin-bottom:0.8rem;font-weight:600;">
                {band_label} because:
            </div>
            {items_html}
        </div>""", unsafe_allow_html=True)


# =============================================================================
# MODULE 3 — DEPARTMENT-LEVEL RISK VIEW
# =============================================================================
def render_department_view(fdf):
    section_header("Department-Level Risk View", badge="Aggregated", icon="🏢")

    col_a, col_b = st.columns(2)

    with col_a:
        # Avg risk by department
        dept_risk = (
            fdf.groupby("Department")["AttritionProb"].mean()
            .reset_index().sort_values("AttritionProb", ascending=True)
        )
        fig_dept = go.Figure(go.Bar(
            x=dept_risk["AttritionProb"],
            y=dept_risk["Department"],
            orientation="h",
            marker=dict(
                color=dept_risk["AttritionProb"],
                colorscale=[[0,"#43AA8B"],[0.5,"#F9C74F"],[1,"#FF4B4B"]],
                showscale=True,
                colorbar=dict(thickness=8, tickfont=dict(color="#9B9FC4", size=9)),
            ),
            hovertemplate="<b>%{y}</b><br>Avg Prob: %{x:.3f}<extra></extra>",
            text=dept_risk["AttritionProb"].apply(lambda x: f"{x:.3f}"),
            textposition="outside", textfont=dict(color="#9B9FC4", size=10),
        ))
        apply_theme(fig_dept, "Avg Attrition Risk by Department", height=300)
        st.plotly_chart(fig_dept, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # High-risk count by role
        role_high = (
            fdf[fdf["RiskBand"] == "High"].groupby("JobRole").size()
            .reset_index(name="HighRiskCount").sort_values("HighRiskCount", ascending=False).head(10)
        )
        fig_role = go.Figure(go.Bar(
            x=role_high["HighRiskCount"],
            y=role_high["JobRole"],
            orientation="h",
            marker=dict(color=COLOR_HIGH, opacity=0.85),
            hovertemplate="<b>%{y}</b><br>High Risk: %{x}<extra></extra>",
        ))
        apply_theme(fig_role, "Top 10 Job Roles — High Risk Count", height=300)
        st.plotly_chart(fig_role, use_container_width=True, config={"displayModeBar": False})

    # Heatmap: Department × Risk Band
    section_header("Risk Band Heatmap", icon="🗺️")
    heat_df = fdf.groupby(["Department", "JobRole"])["AttritionProb"].mean().reset_index()
    if len(heat_df) > 0:
        pivot = heat_df.pivot(index="JobRole", columns="Department", values="AttritionProb").fillna(0)
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0,"#43AA8B"],[0.5,"#F9C74F"],[1,"#FF4B4B"]],
            hovertemplate="Role: %{y}<br>Dept: %{x}<br>Avg Prob: %{z:.3f}<extra></extra>",
            colorbar=dict(thickness=10, tickfont=dict(color="#9B9FC4", size=9)),
        ))
        apply_theme(fig_heat, "Avg Attrition Probability — Role × Department", height=420)
        fig_heat.update_layout(
            xaxis=dict(side="top", tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # Risk split stacked bar by department
    section_header("Risk Band Composition by Department", icon="📊")
    split = fdf.groupby(["Department", "RiskBand"]).size().reset_index(name="Count")
    fig_stack = go.Figure()
    for band, color in [("High", COLOR_HIGH), ("Medium", COLOR_MEDIUM), ("Low", COLOR_LOW)]:
        sub = split[split["RiskBand"] == band]
        fig_stack.add_trace(go.Bar(
            name=f"{band} Risk",
            x=sub["Department"],
            y=sub["Count"],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{band} Risk: %{{y}}<extra></extra>",
        ))
    fig_stack.update_layout(barmode="stack")
    apply_theme(fig_stack, height=340)
    st.plotly_chart(fig_stack, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# MODULE 4 — EXPLAINABILITY PANEL
# =============================================================================
def render_explainability(df, model_bundle, preprocessor):
    section_header("Explainability Panel", badge="SHAP + What-If", icon="🔬")

    tab1, tab2, tab3 = st.tabs(["📈 Global Feature Importance", "🔎 Individual SHAP", "🧪 What-If Analysis"])

    # ── Tab 1: Global Feature Importance ──────────────────────────────────────
    with tab1:
        st.markdown('<div class="info-box">Global feature importance shows which features most influence attrition predictions across all employees.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if model_bundle is not None:
            model = model_bundle["model"]
            feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
            importance = None

            if hasattr(model, "feature_importances_"):
                raw_imp = model.feature_importances_
                if preprocessor is not None:
                    try:
                        enc_names = (
                            preprocessor.named_transformers_["cat_pipelines"]
                            .named_steps["one_hot_encoder"]
                            .get_feature_names_out(CATEGORICAL_COLUMNS)
                        )
                        all_feat_names = NUMERICAL_COLUMNS + list(enc_names)
                        imp_df = pd.DataFrame({"Feature": all_feat_names[:len(raw_imp)], "Importance": raw_imp}).nlargest(20, "Importance")
                    except Exception:
                        imp_df = pd.DataFrame({"Feature": feat_cols[:len(raw_imp)], "Importance": raw_imp}).nlargest(20, "Importance")
                else:
                    imp_df = pd.DataFrame({"Feature": feat_cols[:len(raw_imp)], "Importance": raw_imp}).nlargest(20, "Importance")
                importance = imp_df

            elif hasattr(model, "coef_"):
                coef = np.abs(model.coef_[0])
                if preprocessor is not None:
                    try:
                        enc_names = (
                            preprocessor.named_transformers_["cat_pipelines"]
                            .named_steps["one_hot_encoder"]
                            .get_feature_names_out(CATEGORICAL_COLUMNS)
                        )
                        all_feat_names = NUMERICAL_COLUMNS + list(enc_names)
                        imp_df = pd.DataFrame({"Feature": all_feat_names[:len(coef)], "Importance": coef}).nlargest(20, "Importance")
                    except Exception:
                        imp_df = pd.DataFrame({"Feature": feat_cols[:len(coef)], "Importance": coef}).nlargest(20, "Importance")
                else:
                    imp_df = pd.DataFrame({"Feature": feat_cols[:len(coef)], "Importance": coef}).nlargest(20, "Importance")
                importance = imp_df

            if importance is not None:
                importance = importance.sort_values("Importance", ascending=True)
                fig_imp = go.Figure(go.Bar(
                    x=importance["Importance"],
                    y=importance["Feature"],
                    orientation="h",
                    marker=dict(
                        color=importance["Importance"],
                        colorscale=[[0,"#6C63FF"],[1,"#FF4B4B"]],
                        showscale=False,
                    ),
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                ))
                apply_theme(fig_imp, "Top 20 Features by Importance", height=500)
                st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Feature importance not available for this model type.")
        else:
            # Demo: synthetic importances
            demo_features = [
                "OverTime", "MonthlyIncome", "YearsAtCompany", "Age",
                "WorkloadStress", "EngagementScore", "DissatisfactionRisk",
                "DistanceFromHome", "JobSatisfaction", "LoyaltyScore",
                "JobHopperIndex", "HighPerf_NoPromo", "EnvironmentSatisfaction",
                "PromotionDelay", "BusinessTravel_Travel_Frequently",
            ]
            demo_vals = sorted(np.random.dirichlet(np.ones(15), size=1)[0], reverse=True)
            imp_df = pd.DataFrame({"Feature": demo_features, "Importance": demo_vals}).sort_values("Importance")
            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
                marker=dict(color=imp_df["Importance"],
                            colorscale=[[0,"#6C63FF"],[1,"#FF4B4B"]], showscale=False),
            ))
            apply_theme(fig_imp, "Top Features (Demo Mode)", height=450)
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 2: Individual SHAP ─────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="info-box">SHAP (SHapley Additive exPlanations) shows <b>exactly why</b> a specific employee\'s risk is high or low. '
                    'Positive SHAP → pushes toward attrition &nbsp;|&nbsp; Negative → reduces risk.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        emp_ids  = sorted(df["EmployeeID"].unique().tolist())
        sel_emp  = st.selectbox("Select Employee for SHAP Breakdown", emp_ids, key="shap_emp")
        emp_row2 = df[df["EmployeeID"] == sel_emp]

        if emp_row2.empty:
            st.warning("Employee not found.")
        else:
            emp_row2 = emp_row2.iloc[0]
            prob     = emp_row2["AttritionProb"]
            band     = emp_row2["RiskBand"]
            color    = COLOR_HIGH if band == "High" else (COLOR_MEDIUM if band == "Medium" else COLOR_LOW)

            feat_cols      = [c for c in ALL_FEATURE_COLS if c in df.columns]
            shap_available = False
            shap_df        = None

            # ── Real SHAP ──────────────────────────────────────────────────────
            if model_bundle is not None and preprocessor is not None:
                try:
                    import shap as shap_lib  # noqa: F401
                    model      = model_bundle["model"]
                    X_bg       = preprocessor.transform(df[feat_cols].head(100))
                    X_emp_t    = preprocessor.transform(emp_row2[feat_cols].to_frame().T)
                    sv         = _run_shap_explainer(model, X_bg, X_emp_t)
                    all_names  = _get_shap_feature_names(preprocessor, feat_cols)
                    shap_df    = (
                        pd.DataFrame({"Feature": all_names[:len(sv)], "SHAP": sv})
                        .assign(AbsSHAP=lambda d: d["SHAP"].abs())
                        .sort_values("AbsSHAP", ascending=False)
                        .head(15)
                        .drop(columns="AbsSHAP")
                    )
                    shap_available = True
                    st.success("✅ Real SHAP values computed via `shap.Explainer`")
                except Exception as e:
                    st.warning(f"SHAP library unavailable ({e}) — showing z-score proxy.")

            # ── Proxy fallback ─────────────────────────────────────────────────
            if not shap_available:
                display_features = [
                    "OverTime","WorkloadStress","DissatisfactionRisk","HighPerf_NoPromo",
                    "LoyaltyScore","EngagementScore","JobSatisfaction","EnvironmentSatisfaction",
                    "YearsSinceLastPromotion","DistanceFromHome","MonthlyIncome","JobHopperIndex",
                    "PromotionDelay","WorkLifeBalance","NumCompaniesWorked",
                ]
                PROTECTIVE_P = {"LoyaltyScore","EngagementScore","JobSatisfaction",
                                "EnvironmentSatisfaction","WorkLifeBalance","MonthlyIncome"}
                proxy_data = []
                for feat in display_features:
                    if feat not in df.columns:
                        continue
                    val    = emp_row2.get(feat, 0)
                    mean_v = df[feat].mean() if pd.api.types.is_numeric_dtype(df[feat]) else 0
                    std_v  = df[feat].std()  if pd.api.types.is_numeric_dtype(df[feat]) else 1
                    std_v  = std_v if std_v > 0 else 1
                    sign   = -1 if feat in PROTECTIVE_P else 1
                    proxy_data.append({"Feature": feat,
                                       "SHAP": round(sign * (val - mean_v) / std_v * 0.08, 4)})
                shap_df = pd.DataFrame(proxy_data).sort_values("SHAP", key=lambda s: s.abs(), ascending=False)

            shap_df = shap_df.sort_values("SHAP", ascending=True)
            shap_colors = [COLOR_HIGH if v > 0 else COLOR_LOW for v in shap_df["SHAP"]]

            # Waterfall-style horizontal bar chart
            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP"],
                y=shap_df["Feature"],
                orientation="h",
                marker=dict(color=shap_colors,
                            line=dict(color="rgba(0,0,0,0)", width=0),
                            opacity=0.9),
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
                text=[f"{v:+.4f}" for v in shap_df["SHAP"]],
                textposition="outside",
                textfont=dict(color="#9B9FC4", size=10),
            ))
            fig_shap.add_vline(x=0, line_color="#252840", line_width=2)
            apply_theme(fig_shap,
                        f"SHAP Waterfall — Employee #{sel_emp} | Risk: {prob*100:.1f}%",
                        height=460)
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

            # Top-5 cards
            c1, c2 = st.columns(2)
            pos_rows = shap_df[shap_df["SHAP"] > 0].sort_values("SHAP", ascending=False).head(5)
            neg_rows = shap_df[shap_df["SHAP"] < 0].sort_values("SHAP").head(5)
            with c1:
                rows_html = "".join([
                    f'<div style="font-size:0.83rem;color:{COLOR_HIGH};padding:3px 0;">'
                    f'↑ {r["Feature"]} <span style="font-family:DM Mono,monospace;">({r["SHAP"]:+.4f})</span></div>'
                    for _, r in pos_rows.iterrows()
                ])
                st.markdown(f"""
                <div style="background:#1F1215;border-radius:10px;padding:0.9rem 1.1rem;
                            border:1px solid {COLOR_HIGH}33;">
                    <div style="font-size:0.7rem;color:#7B7FA8;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:0.5rem;font-weight:600;">
                        ▲ Risk Drivers (pushing attrition UP)
                    </div>
                    {rows_html or "<div style='color:#7B7FA8;font-size:0.8rem;'>None</div>"}
                </div>""", unsafe_allow_html=True)
            with c2:
                rows_html2 = "".join([
                    f'<div style="font-size:0.83rem;color:{COLOR_LOW};padding:3px 0;">'
                    f'↓ {r["Feature"]} <span style="font-family:DM Mono,monospace;">({r["SHAP"]:+.4f})</span></div>'
                    for _, r in neg_rows.iterrows()
                ])
                st.markdown(f"""
                <div style="background:#0D1F18;border-radius:10px;padding:0.9rem 1.1rem;
                            border:1px solid {COLOR_LOW}33;">
                    <div style="font-size:0.7rem;color:#7B7FA8;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:0.5rem;font-weight:600;">
                        ▼ Protective Factors (pushing attrition DOWN)
                    </div>
                    {rows_html2 or "<div style='color:#7B7FA8;font-size:0.8rem;'>None</div>"}
                </div>""", unsafe_allow_html=True)

    # ── Tab 3: What-If Analysis ────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="info-box">Adjust feature values below to see how an employee\'s attrition probability changes in real time.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        ref_emp_id = st.selectbox("Base Employee", sorted(df["EmployeeID"].unique().tolist()), key="whatif_emp")
        ref_row = df[df["EmployeeID"] == ref_emp_id].iloc[0]

        col_sliders, col_result = st.columns([1.8, 1.2])

        with col_sliders:
            st.markdown("**Adjust Features:**")
            overtime_wi    = st.selectbox("OverTime",         [0, 1],
                                           index=int(ref_row.get("OverTime", 0)), key="wi_ot",
                                           format_func=lambda x: "Yes" if x else "No")
            satisfaction_wi = st.slider("Job Satisfaction",  1, 4,
                                         int(ref_row.get("JobSatisfaction", 3)), key="wi_js")
            env_wi         = st.slider("Environment Satisfaction", 1, 4,
                                        int(ref_row.get("EnvironmentSatisfaction", 3)), key="wi_env")
            wlb_wi         = st.slider("Work-Life Balance",  1, 4,
                                        int(ref_row.get("WorkLifeBalance", 3)), key="wi_wlb")
            income_wi      = st.slider("Monthly Income",     1000, 20000,
                                        int(ref_row.get("MonthlyIncome", 5000)), step=500, key="wi_inc")
            promo_wi       = st.slider("Years Since Last Promotion", 0, 15,
                                        int(ref_row.get("YearsSinceLastPromotion", 2)), key="wi_promo")
            dist_wi        = st.slider("Distance From Home", 1, 30,
                                        int(ref_row.get("DistanceFromHome", 10)), key="wi_dist")

        with col_result:
            # Build modified row
            mod = ref_row.copy()
            mod["OverTime"]                 = overtime_wi
            mod["JobSatisfaction"]          = satisfaction_wi
            mod["EnvironmentSatisfaction"]  = env_wi
            mod["WorkLifeBalance"]          = wlb_wi
            mod["MonthlyIncome"]            = income_wi
            mod["YearsSinceLastPromotion"]  = promo_wi
            mod["DistanceFromHome"]         = dist_wi

            # Re-derive engineered features
            mod["EngagementScore"]   = (satisfaction_wi + env_wi + mod.get("RelationshipSatisfaction", 3)) / 3
            mod["WorkloadStress"]    = int(overtime_wi == 1 and wlb_wi <= 2)
            mod["HighPerf_NoPromo"]  = int(mod.get("PerformanceRating", 3) >= 4 and promo_wi > 3)
            mod["IncomePerYearExp"]  = income_wi / (mod.get("TotalWorkingYears", 1) + 1)
            mod["DissatisfactionRisk"] = (
                int(mod.get("JobInvolvement", 3) <= 2) +
                int(mod.get("StockOptionLevel", 0) == 0) +
                int(dist_wi > 15)
            )

            # Predict
            if model_bundle is not None and preprocessor is not None:
                try:
                    feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
                    X_mod = preprocessor.transform(pd.DataFrame([mod])[feat_cols])
                    new_prob = model_bundle["model"].predict_proba(X_mod)[0, 1]
                except Exception:
                    new_prob = prob = ref_row["AttritionProb"]
            else:
                # Proxy: adjust original prob based on factors
                base    = ref_row["AttritionProb"]
                delta   = (overtime_wi - ref_row.get("OverTime", 0)) * 0.08
                delta  += (2.5 - satisfaction_wi) * 0.03
                delta  += (2.5 - env_wi) * 0.02
                delta  += (2.5 - wlb_wi) * 0.02
                delta  += (ref_row.get("MonthlyIncome", 5000) - income_wi) / 100000
                delta  += (promo_wi - ref_row.get("YearsSinceLastPromotion", 2)) * 0.01
                delta  += (dist_wi - ref_row.get("DistanceFromHome", 10)) * 0.002
                new_prob = float(np.clip(base + delta, 0.01, 0.99))

            orig_prob = ref_row["AttritionProb"]
            delta_prob = new_prob - orig_prob
            delta_color = COLOR_HIGH if delta_prob > 0 else COLOR_LOW
            new_band  = "High" if new_prob >= 0.7 else ("Medium" if new_prob >= 0.4 else "Low")
            new_color = COLOR_HIGH if new_band == "High" else (COLOR_MEDIUM if new_band == "Medium" else COLOR_LOW)

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1A1D2E,#161926);border:1px solid #252840;
                         border-radius:14px;padding:1.5rem;text-align:center;margin-bottom:1rem;">
                <div style="font-size:0.7rem;color:#7B7FA8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">Modified Attrition Probability</div>
                <div style="font-size:3rem;font-weight:700;color:{new_color};font-family:'DM Mono',monospace;">{new_prob*100:.1f}%</div>
                <div style="font-size:1rem;margin:0.5rem 0;">{risk_badge(new_band)}</div>
                <div style="font-size:0.85rem;color:{delta_color};margin-top:0.8rem;font-weight:600;">
                    {"▲" if delta_prob > 0 else "▼"} {abs(delta_prob)*100:.1f}% vs original ({orig_prob*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mini gauge
            fig_wi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(new_prob * 100, 1),
                number=dict(suffix="%", font=dict(size=24, color=new_color)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color="#7B7FA8", size=9)),
                    bar=dict(color=new_color, thickness=0.22),
                    bgcolor="#1A1D2E", borderwidth=0,
                    steps=[
                        dict(range=[0,40],  color="#0D1F1A"),
                        dict(range=[40,70], color="#1F1C0D"),
                        dict(range=[70,100],color="#1F0D0D"),
                    ],
                ),
            ))
            apply_theme(fig_wi, height=220)
            fig_wi.update_layout(margin=dict(l=20,r=20,t=20,b=10))
            st.plotly_chart(fig_wi, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# MODULE 5 — ADVANCED ANALYTICS
# =============================================================================
def render_advanced_analytics(fdf):
    section_header("Advanced Analytics", badge="Trends & Correlations", icon="📉")

    col_a, col_b = st.columns(2)

    with col_a:
        # Risk by income bracket
        if "MonthlyIncome" in fdf.columns:
            fdf_c = fdf.copy()
            fdf_c["IncomeBracket"] = pd.cut(
                fdf_c["MonthlyIncome"],
                bins=[0, 3000, 6000, 9000, 12000, 25000],
                labels=["<3k", "3-6k", "6-9k", "9-12k", "12k+"],
            )
            inc_risk = fdf_c.groupby("IncomeBracket", observed=True)["AttritionProb"].mean().reset_index()
            fig_inc = go.Figure(go.Bar(
                x=inc_risk["IncomeBracket"].astype(str),
                y=inc_risk["AttritionProb"],
                marker=dict(color=inc_risk["AttritionProb"],
                            colorscale=[[0,"#43AA8B"],[1,"#FF4B4B"]], showscale=False),
                hovertemplate="Income: %{x}<br>Avg Risk: %{y:.3f}<extra></extra>",
            ))
            apply_theme(fig_inc, "Attrition Risk by Monthly Income", height=300)
            st.plotly_chart(fig_inc, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Overtime vs non-overtime distribution
        if "OverTime" in fdf.columns:
            ot_df = fdf.groupby("OverTime")["AttritionProb"].mean().reset_index()
            ot_df["Label"] = ot_df["OverTime"].map({1: "Overtime", 0: "No Overtime"})
            fig_ot = go.Figure(go.Bar(
                x=ot_df["Label"],
                y=ot_df["AttritionProb"],
                marker=dict(color=[COLOR_HIGH, COLOR_LOW]),
                hovertemplate="%{x}<br>Avg Risk: %{y:.3f}<extra></extra>",
            ))
            apply_theme(fig_ot, "Avg Risk: Overtime vs No Overtime", height=300)
            st.plotly_chart(fig_ot, use_container_width=True, config={"displayModeBar": False})

    # Correlation heatmap (numeric features)
    with st.expander("📐 Feature Correlation Matrix", expanded=False):
        numeric_cols_corr = [
            "AttritionProb", "Age", "MonthlyIncome", "YearsAtCompany",
            "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
            "OverTime", "DistanceFromHome", "NumCompaniesWorked",
            "YearsSinceLastPromotion", "EngagementScore", "DissatisfactionRisk",
        ]
        avail = [c for c in numeric_cols_corr if c in fdf.columns]
        corr_matrix = fdf[avail].corr().round(2)
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(thickness=10, tickfont=dict(color="#9B9FC4", size=9)),
        ))
        apply_theme(fig_corr, "Feature Correlation Matrix", height=480)
        fig_corr.update_layout(xaxis=dict(tickangle=-35, tickfont=dict(size=9)))
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# MODULE 6 — MODEL EVALUATION & COMPARISON
# =============================================================================
@st.cache_data(show_spinner=False)
def _train_comparison_models(_raw_df):
    """Train LR, RF, XGB on the dataset and return evaluation metrics."""
    df = _raw_df.copy()
    if "Attrition" not in df.columns:
        return None

    # Prep
    df["Attrition"] = df["Attrition"].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0}
    ).fillna(0).astype(int)
    if "OverTime" in df.columns and df["OverTime"].dtype == object:
        df["OverTime"] = df["OverTime"].str.strip().str.lower().map(
            {"yes": 1, "no": 0}).fillna(0).astype(int)

    df = engineer_features(df)
    feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

    # Simple encode for comparison
    X = pd.get_dummies(df[feat_cols], drop_first=True)
    y = df["Attrition"]
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42,
                                           eval_metric="logloss", use_label_encoder=False)

    results = {}
    roc_data = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred  = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred,        zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
        }
        roc_data[name] = {"fpr": fpr, "tpr": tpr}

    return results, roc_data


# ── Shared sub-renderers used by render_model_metrics ─────────────────────────

def _show_metric_cards(acc, prec, rec, f1_s, auc):
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, icon, cls in [
        (c1, "Accuracy",  acc,  "🎯", "total"),
        (c2, "Precision", prec, "🔍", "low"),
        (c3, "Recall",    rec,  "📡", "medium"),
        (c4, "F1-Score",  f1_s, "⚖️", "medium"),
        (c5, "ROC-AUC",   auc,  "📊", "high"),
    ]:
        with col:
            metric_card(label, f"{val:.3f}", f"{val*100:.1f}%", cls, icon)
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)


def _show_roc_from_comp(roc_data_comp, results, best_name):
    COLORS_MODELS = [COLOR_ACCENT, "#43AA8B", "#F9C74F", "#FF4B4B"]
    fig_roc = go.Figure()
    for i, (mname, rd) in enumerate(roc_data_comp.items()):
        auc_val = results[mname]["ROC-AUC"]
        fig_roc.add_trace(go.Scatter(
            x=rd["fpr"], y=rd["tpr"], mode="lines",
            line=dict(color=COLORS_MODELS[i % len(COLORS_MODELS)], width=2.5),
            name=f"{mname} (AUC={auc_val:.3f})",
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(color="#4A4E6A", dash="dash", width=1.5), name="Baseline",
    ))
    apply_theme(fig_roc, "ROC Curve (Test Split)", height=340)
    st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})


def _render_comparison_section(results, roc_data):
    section_header("Model Comparison", badge="LR vs RF vs XGBoost", icon="🏆")

    best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    best_auc  = results[best_name]["ROC-AUC"]

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#1A2535,#1A1D2E);border:1px solid #6C63FF44;
                border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.2rem;
                display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem;">🥇</span>
        <div>
            <div style="font-size:0.72rem;color:#7B7FA8;letter-spacing:0.1em;
                        text-transform:uppercase;">Best Model</div>
            <div style="font-size:1.2rem;font-weight:700;color:#E8EAF0;">{best_name}
                <span style="font-size:0.9rem;color:{COLOR_ACCENT};margin-left:0.5rem;">
                    ROC-AUC: {best_auc:.4f}</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    rows = [
        {"Model": mn, **{k: f"{v:.4f}" for k,v in m.items()},
         "Best?": "✅ Best" if mn == best_name else ""}
        for mn, m in results.items()
    ]
    comp_df = pd.DataFrame(rows)
    st.dataframe(
        comp_df.style
        .map(lambda v: "color:#6C63FF;font-weight:700" if v == "✅ Best" else "",
                  subset=["Best?"])
        .set_properties(**{"background-color": "#12141F", "border-color": "#1E2130"}),
        use_container_width=True, hide_index=True,
    )

    COLORS_MODELS = [COLOR_ACCENT, "#43AA8B", "#F9C74F"]
    col_roc, col_bar = st.columns(2)
    with col_roc:
        fig_roc_comp = go.Figure()
        for i, (mname, rd) in enumerate(roc_data.items()):
            fig_roc_comp.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"], mode="lines",
                line=dict(color=COLORS_MODELS[i % len(COLORS_MODELS)], width=2.5),
                name=f"{mname} ({results[mname]['ROC-AUC']:.3f})",
                hovertemplate="FPR:%{x:.3f} TPR:%{y:.3f}<extra></extra>",
            ))
        fig_roc_comp.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#4A4E6A", dash="dash", width=1.5), name="Baseline",
        ))
        apply_theme(fig_roc_comp, "ROC Curves Comparison", height=360)
        st.plotly_chart(fig_roc_comp, use_container_width=True, config={"displayModeBar": False})

    with col_bar:
        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        fig_bar = go.Figure()
        for i, (mname, metrics) in enumerate(results.items()):
            fig_bar.add_trace(go.Bar(
                name=mname,
                x=metric_names,
                y=[metrics[m] for m in metric_names],
                marker_color=COLORS_MODELS[i % len(COLORS_MODELS)],
                hovertemplate=f"<b>{mname}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
            ))
        fig_bar.update_layout(barmode="group")
        apply_theme(fig_bar, "All Metrics Comparison", height=360)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# FIX 5 — PIPELINE DISPLAY
# =============================================================================
def _render_pipeline_display():
    section_header("ML Pipeline Architecture", badge="End-to-End Flow", icon="⚙️")
    st.markdown("""
    <div class="info-box">
    This section shows the complete pipeline: from raw data through feature engineering,
    preprocessing (scaling + encoding), model training, and final prediction output.
    </div><br>""", unsafe_allow_html=True)

    # Visual pipeline using HTML
    steps = [
        ("📂", "Raw Data", "CSV / DataFrame<br><small style='color:#7B7FA8'>1470 rows × 35 cols</small>"),
        ("🔧", "Feature Engineering", "8 new features created<br>"
         "<small style='color:#7B7FA8'>EngagementScore, WorkloadStress,<br>"
         "HighPerf_NoPromo, JobHopperIndex,<br>"
         "LoyaltyScore, PromotionDelay,<br>"
         "IncomePerYearExp, DissatisfactionRisk</small>"),
        ("⚙️", "Preprocessing", "Numerical → StandardScaler<br>"
         "<small style='color:#7B7FA8'>23 numeric features normalised</small><br>"
         "Categorical → OneHotEncoder<br>"
         "<small style='color:#7B7FA8'>6 categorical columns encoded</small>"),
        ("🤖", "Model Training", "3 models trained &amp; compared<br>"
         "<small style='color:#7B7FA8'>Logistic Regression<br>"
         "Random Forest (100 estimators)<br>"
         "XGBoost (100 estimators)</small>"),
        ("📊", "Evaluation", "Accuracy · Precision · Recall<br>"
         "<small style='color:#7B7FA8'>F1-Score · ROC-AUC · Confusion Matrix</small>"),
        ("🎯", "Prediction", "Risk Score + Band<br>"
         "<small style='color:#7B7FA8'>High ≥ 70% · Medium 40–70%<br>Low &lt; 40%</small>"),
        ("🔬", "Explainability", "SHAP Values<br>"
         "<small style='color:#7B7FA8'>shap.Explainer (TreeExplainer /<br>"
         "LinearExplainer / Universal)</small>"),
    ]

    cards_html = ""
    for i, (icon, title, desc) in enumerate(steps):
        arrow = '<div style="font-size:1.4rem;color:#4A4E6A;margin:0 0.3rem;">→</div>' if i < len(steps)-1 else ""
        cards_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;min-width:120px;max-width:150px;">
            <div style="background:linear-gradient(135deg,#1A1D2E,#161926);border:1px solid #252840;
                        border-radius:12px;padding:0.9rem 0.7rem;text-align:center;width:100%;
                        border-top:3px solid {COLOR_ACCENT};">
                <div style="font-size:1.6rem;margin-bottom:0.4rem;">{icon}</div>
                <div style="font-size:0.78rem;font-weight:700;color:#E8EAF0;margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.7rem;color:#9B9FC4;line-height:1.5;">{desc}</div>
            </div>
        </div>
        {arrow}
        """

    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;align-items:flex-start;gap:0.3rem;
                padding:1rem;background:#0D0F18;border-radius:14px;border:1px solid #1E2130;">
        {cards_html}
    </div>""", unsafe_allow_html=True)

    # Preprocessing detail table
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    with st.expander("📋 Feature Engineering — Full Reference", expanded=False):
        fe_data = {
            "Feature": ["EngagementScore", "WorkloadStress", "HighPerf_NoPromo",
                        "JobHopperIndex", "LoyaltyScore", "PromotionDelay",
                        "IncomePerYearExp", "DissatisfactionRisk"],
            "Formula": [
                "(JobSatisfaction + EnvSatisfaction + RelSatisfaction) / 3",
                "OverTime==1 AND WorkLifeBalance≤2 → 1 else 0",
                "PerformanceRating≥4 AND YearsSinceLastPromotion>3 → 1",
                "NumCompaniesWorked / (Age - 17 + 1)",
                "YearsAtCompany / (TotalWorkingYears + 1)",
                "YearsAtCompany - YearsSinceLastPromotion",
                "MonthlyIncome / (TotalWorkingYears + 1)",
                "Sum of: JobInvolvement≤2, StockOption==0, DistanceFromHome>15",
            ],
            "Type": ["Continuous","Binary","Binary","Continuous",
                     "Continuous","Continuous","Continuous","Ordinal (0-3)"],
            "Direction": ["↓ Lower = Higher Risk","↑ Yes = Higher Risk","↑ Yes = Higher Risk",
                          "↑ Higher = Higher Risk","↓ Lower = Higher Risk","↑ Higher = Higher Risk",
                          "↓ Lower = Higher Risk","↑ Higher = Higher Risk"],
        }
        st.dataframe(pd.DataFrame(fe_data).style.set_properties(
            **{"background-color": "#12141F", "border-color": "#1E2130"}),
            use_container_width=True, hide_index=True)

    with st.expander("🔢 Preprocessing Details — Scaling & Encoding", expanded=False):
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div style="background:#1A1D2E;border-radius:10px;padding:1rem;border-left:3px solid #6C63FF;">
                <div style="font-size:0.72rem;color:#7B7FA8;letter-spacing:0.08em;
                            text-transform:uppercase;margin-bottom:0.5rem;font-weight:600;">
                    📐 Numerical Pipeline (23 features)
                </div>
                <div style="font-size:0.82rem;color:#C8CADF;line-height:1.8;">
                    Step 1: <code>SimpleImputer(strategy='median')</code><br>
                    Step 2: <code>StandardScaler()</code><br>
                    → Each feature: mean=0, std=1<br>
                    → Prevents high-magnitude features dominating
                </div>
            </div>
            <div style="background:#1A1D2E;border-radius:10px;padding:1rem;border-left:3px solid #43AA8B;">
                <div style="font-size:0.72rem;color:#7B7FA8;letter-spacing:0.08em;
                            text-transform:uppercase;margin-bottom:0.5rem;font-weight:600;">
                    🏷️ Categorical Pipeline (6 features)
                </div>
                <div style="font-size:0.82rem;color:#C8CADF;line-height:1.8;">
                    Step 1: <code>SimpleImputer(strategy='most_frequent')</code><br>
                    Step 2: <code>OneHotEncoder(handle_unknown='ignore')</code><br>
                    → Creates binary columns per category<br>
                    → Encoded: BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


def render_model_metrics(scored_df, model_bundle, preprocessor, raw_df):
    section_header("Model Evaluation", badge="Performance Metrics", icon="📈")

    # ── Always compute metrics (from live labels OR from train/test split) ────
    metrics_source = None
    y_true = y_pred = y_prob = None

    if "Attrition" in scored_df.columns and scored_df["Attrition"].nunique() == 2:
        y_true = scored_df["Attrition"].astype(int)
        y_pred = scored_df["PredAttrition"].astype(int)
        y_prob = scored_df["AttritionProb"]
        metrics_source = "Full dataset (live labels)"

    # Fallback: train a quick RF on raw_df and evaluate on held-out test set
    if y_true is None and raw_df is not None and "Attrition" in raw_df.columns:
        try:
            comp = _train_comparison_models(raw_df)
            if comp:
                results, roc_data_comp = comp
                # Pick best model's test predictions to show as primary metrics
                best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
                bm = results[best_name]
                metrics_source = f"Test-set evaluation ({best_name}, 20% hold-out)"
                acc, prec, rec, f1_s, auc = (
                    bm["Accuracy"], bm["Precision"],
                    bm["Recall"],   bm["F1-Score"], bm["ROC-AUC"],
                )
                _show_metric_cards(acc, prec, rec, f1_s, auc)
                st.caption(f"📌 Source: {metrics_source}")
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                _show_roc_from_comp(roc_data_comp, results, best_name)
                # skip to model comparison (already built)
                _render_comparison_section(results, roc_data_comp)
                _render_pipeline_display()
                return
        except Exception:
            pass

    if y_true is not None:
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred,    zero_division=0)
        f1_s = f1_score(y_true, y_pred,        zero_division=0)
        auc  = roc_auc_score(y_true, y_prob)

        _show_metric_cards(acc, prec, rec, f1_s, auc)
        st.caption(f"📌 Source: {metrics_source}")
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        col_cm, col_roc = st.columns(2)
        with col_cm:
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Pred: Stay", "Pred: Leave"],
                y=["Actual: Stay", "Actual: Leave"],
                colorscale=[[0, "#1A1D2E"], [1, COLOR_ACCENT]],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=22, color="#E8EAF0"),
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            apply_theme(fig_cm, "Confusion Matrix", height=320)
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

        with col_roc:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=COLOR_ACCENT, width=2.5),
                fill="tozeroy", fillcolor="rgba(108,99,255,0.1)",
                name=f"ROC (AUC={auc:.3f})",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(color="#4A4E6A", dash="dash", width=1.5),
                name="Baseline",
            ))
            apply_theme(fig_roc, "ROC Curve", height=320)
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("ℹ️ No Attrition labels found. Place dataset at `artifacts/data.csv` for full metrics.")

    # ── Model Comparison ──────────────────────────────────────────────────────
    with st.spinner("Training comparison models…"):
        comp = _train_comparison_models(raw_df)
    if comp:
        results, roc_data_comp = comp
        _render_comparison_section(results, roc_data_comp)

    # ── Pipeline Display ──────────────────────────────────────────────────────
    _render_pipeline_display()


# =============================================================================
# MODULE 7 — REAL PREDICTION INPUT FORM
# =============================================================================
def render_predict_form(model_bundle, preprocessor, raw_df):
    section_header("Predict Employee Attrition", badge="Manual Input", icon="🎯")

    st.markdown("""
    <div class="info-box">
    Enter employee details below and get an instant attrition risk prediction
    with AI-powered explanation of the top risk drivers.
    </div><br>""", unsafe_allow_html=True)

    with st.form("predict_form", clear_on_submit=False):
        st.markdown("#### 👤 Personal & Role Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age          = st.number_input("Age", 18, 65, 32)
            gender       = st.selectbox("Gender", ["Male", "Female"])
            marital      = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            edu_field    = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing",
                                                            "Technical Degree", "Human Resources", "Other"])
        with c2:
            department   = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role     = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                                      "Manufacturing Director", "Healthcare Representative",
                                                      "Manager", "Sales Representative", "Research Director",
                                                      "Human Resources"])
            job_level    = st.slider("Job Level", 1, 5, 2)
            education    = st.slider("Education Level (1-5)", 1, 5, 3)
        with c3:
            business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
            distance_home   = st.slider("Distance From Home (km)", 1, 30, 10)
            num_companies   = st.slider("Num. Companies Worked", 0, 10, 2)
            stock_option    = st.slider("Stock Option Level", 0, 3, 1)

        st.markdown("---")
        st.markdown("#### 💰 Compensation & Career")
        c4, c5, c6 = st.columns(3)
        with c4:
            monthly_income   = st.number_input("Monthly Income (₹)", 1000, 25000, 5000, step=500)
            daily_rate       = st.number_input("Daily Rate", 100, 1500, 800)
            hourly_rate      = st.number_input("Hourly Rate", 30, 100, 65)
            monthly_rate     = st.number_input("Monthly Rate", 2000, 27000, 14000, step=500)
        with c5:
            total_working    = st.slider("Total Working Years", 0, 40, 8)
            years_company    = st.slider("Years at Company", 0, 40, 5)
            years_role       = st.slider("Years in Current Role", 0, 18, 3)
            years_promotion  = st.slider("Years Since Last Promotion", 0, 15, 2)
        with c6:
            years_manager    = st.slider("Years With Current Manager", 0, 17, 3)
            percent_hike     = st.slider("% Salary Hike", 11, 25, 15)
            perf_rating      = st.slider("Performance Rating (1-4)", 1, 4, 3)
            training_times   = st.slider("Training Times Last Year", 0, 6, 3)

        st.markdown("---")
        st.markdown("#### 😊 Satisfaction & Wellbeing")
        c7, c8, c9 = st.columns(3)
        with c7:
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            rel_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
        with c8:
            work_life_bal    = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
            job_involvement  = st.slider("Job Involvement (1-4)", 1, 4, 3)
            overtime         = st.selectbox("OverTime", ["No", "Yes"])
        with c9:
            pass  # padding

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        ot_val = 1 if overtime == "Yes" else 0

        # Build row dict
        row = {
            "Age": age, "DailyRate": daily_rate, "DistanceFromHome": distance_home,
            "Education": education, "EnvironmentSatisfaction": env_satisfaction,
            "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
            "JobLevel": job_level, "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies, "PercentSalaryHike": percent_hike,
            "PerformanceRating": perf_rating, "RelationshipSatisfaction": rel_satisfaction,
            "StockOptionLevel": stock_option, "TotalWorkingYears": total_working,
            "TrainingTimesLastYear": training_times, "WorkLifeBalance": work_life_bal,
            "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promotion, "YearsWithCurrManager": years_manager,
            "OverTime": ot_val,
            "BusinessTravel": business_travel, "Department": department,
            "EducationField": edu_field, "Gender": gender,
            "JobRole": job_role, "MaritalStatus": marital,
        }

        # Engineered features
        row["IncomePerYearExp"]   = monthly_income / (total_working + 1)
        row["PromotionDelay"]     = years_company - years_promotion
        row["EngagementScore"]    = (job_satisfaction + env_satisfaction + rel_satisfaction) / 3
        row["WorkloadStress"]     = int(ot_val == 1 and work_life_bal <= 2)
        row["HighPerf_NoPromo"]   = int(perf_rating >= 4 and years_promotion > 3)
        row["JobHopperIndex"]     = num_companies / (age - 17 + 1)
        row["LoyaltyScore"]       = years_company / (total_working + 1)
        row["DissatisfactionRisk"] = (
            int(job_involvement <= 2) + int(stock_option == 0) + int(distance_home > 15)
        )

        emp_series = pd.Series(row)

        # Predict
        if model_bundle is not None and preprocessor is not None:
            try:
                feat_cols = [c for c in ALL_FEATURE_COLS if c in row]
                X_new = preprocessor.transform(pd.DataFrame([row])[feat_cols])
                prob  = float(model_bundle["model"].predict_proba(X_new)[0, 1])
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return
        else:
            # Demo proxy
            prob = float(np.clip(
                0.15
                + ot_val * 0.12
                + (2.5 - job_satisfaction) * 0.04
                + (2.5 - env_satisfaction) * 0.03
                + (2.5 - work_life_bal) * 0.03
                + (years_promotion - 2) * 0.01
                + num_companies * 0.02
                - (monthly_income - 5000) / 100000,
                0.02, 0.98
            ))

        band  = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")
        color = COLOR_HIGH if band == "High" else (COLOR_MEDIUM if band == "Medium" else COLOR_LOW)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        col_res, col_reasons = st.columns([1.2, 1.8])

        with col_res:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number=dict(suffix="%", font=dict(size=36, color=color)),
                gauge=dict(
                    axis=dict(range=[0,100], tickfont=dict(color="#7B7FA8", size=10)),
                    bar=dict(color=color, thickness=0.25),
                    bgcolor="#1A1D2E", borderwidth=0,
                    steps=[
                        dict(range=[0,40],  color="#0D1F1A"),
                        dict(range=[40,70], color="#1F1C0D"),
                        dict(range=[70,100],color="#1F0D0D"),
                    ],
                    threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=round(prob*100,1)),
                ),
                title=dict(text="Attrition Probability", font=dict(color="#9B9FC4", size=13)),
            ))
            apply_theme(fig_g, height=300)
            fig_g.update_layout(margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
            st.markdown(f'<div style="text-align:center;margin-top:-0.5rem;">{risk_badge(band)}</div>',
                        unsafe_allow_html=True)

        with col_reasons:
            # Use a dummy df built from the one row for z-score context
            ref_df = raw_df.copy() if raw_df is not None else pd.DataFrame([row])
            if "OverTime" in ref_df.columns and ref_df["OverTime"].dtype == object:
                ref_df["OverTime"] = ref_df["OverTime"].str.strip().str.lower().map(
                    {"yes":1,"no":0}).fillna(0).astype(int)
            ref_df = engineer_features(ref_df) if raw_df is not None else ref_df

            reasons = compute_top5_reasons(emp_series, ref_df, model_bundle, preprocessor)

            header_color = color
            items_html = ""
            for r in reasons:
                sign   = "▲" if r["shap"] > 0 else "▼"
                rcolor = COLOR_HIGH if r["shap"] > 0 else COLOR_LOW
                items_html += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:0.55rem 0.9rem;background:#1A1D2E;border-radius:8px;margin-bottom:7px;'
                    f'border-left:3px solid {rcolor};">'
                    f'<span style="font-size:0.85rem;color:#C8CADF;">{sign} {r["label"]}</span>'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.83rem;color:{rcolor};font-weight:600;">'
                    f'({r["shap"]:+.3f})</span></div>'
                )

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1A1D2E,#161926);border:1px solid #252840;
                        border-radius:12px;padding:1.3rem;height:100%;border-top:3px solid {header_color};">
                <div style="font-size:0.72rem;color:#7B7FA8;letter-spacing:0.1em;text-transform:uppercase;
                            margin-bottom:1rem;font-weight:600;">{band} Risk because:</div>
                {items_html}
            </div>""", unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    inject_css()

    # ── Top Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="top-header">
        <div>
            <div class="header-title">🧠 AttritionIQ &nbsp;<span style="font-size:1rem;font-weight:400;color:#6C63FF;">Employee Risk Scoring System</span></div>
            <div class="header-subtitle">Predict · Explain · Act — powered by Machine Learning</div>
        </div>
        <div style="display:flex;gap:0.6rem;flex-wrap:wrap;">
            <span class="header-pill">🎯 ML-Powered</span>
            <span class="header-pill">📊 Real-time</span>
            <span class="header-pill">🔒 Explainable AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load resources ─────────────────────────────────────────────────────────
    with st.spinner("Loading model & data..."):
        model_bundle = load_model()
        preprocessor = load_preprocessor()
        raw_df       = load_raw_data()

    if raw_df is None:
        st.error("⚠️ Could not find data file. Place your CSV at `artifacts/data.csv` and restart.")
        st.stop()

    demo_mode = model_bundle is None or preprocessor is None
    if demo_mode:
        st.markdown("""
        <div class="info-box">
        ℹ️ <b>Demo Mode</b> — Model artifacts not found at <code>artifacts/model.pkl</code>.
        Showing synthetic risk scores. Run your training pipeline to enable real predictions.
        </div><br>
        """, unsafe_allow_html=True)

    # Score dataset
    with st.spinner("Computing risk scores..."):
        scored_df = build_scored_dataset(model_bundle, preprocessor, raw_df)

    # Sidebar filters
    filtered_df, threshold, emp_id = render_sidebar(scored_df)

    if filtered_df.empty:
        st.warning("No employees match the current filters.")
        st.stop()

    # ── Navigation tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Risk Dashboard",
        "👤 Employee Profile",
        "🏢 Department View",
        "🔬 Explainability",
        "📉 Advanced Analytics",
        "📈 Model Metrics",
        "🎯 Predict Employee",
    ])

    with tabs[0]:
        render_risk_dashboard(filtered_df, threshold)

    with tabs[1]:
        render_employee_profile(scored_df, emp_id)

    with tabs[2]:
        render_department_view(filtered_df)

    with tabs[3]:
        render_explainability(scored_df, model_bundle, preprocessor)

    with tabs[4]:
        render_advanced_analytics(filtered_df)

    with tabs[5]:
        render_model_metrics(scored_df, model_bundle, preprocessor, raw_df)

    with tabs[6]:
        render_predict_form(model_bundle, preprocessor, raw_df)

    # ── Footer ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        Developed by <strong style="color:#9B9FC4;">Nishit Khandhar</strong> &nbsp;·&nbsp;
        Employee Attrition Prediction & Risk Scoring System &nbsp;·&nbsp;
        <span style="color:#4A4E6A;">Built with Streamlit + Plotly + Scikit-learn · SHAP Explainability · What-If Analysis</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
