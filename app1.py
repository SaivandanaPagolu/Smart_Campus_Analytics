"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     AI-POWERED SMART CAMPUS ANALYTICS SYSTEM                                ║
║     Student & Resource Management — IEEE Research Project                   ║
║                                                                              ║
║  📌 Title  : AI-Powered Smart Campus Analytics System for                   ║
║              Student and Resource Management                                  ║
║  🎯 SDGs   : SDG 4 (Quality Education) · SDG 9 (Innovation & Infrastructure)║
║              SDG 10 (Reduced Inequalities) · SDG 11 (Sustainable Cities)    ║
║  🧠 Models : GPA Regression · Pass/Fail Classification · At-Risk Detection  ║
║  🛠  Stack  : Python · Streamlit · Scikit-learn · Plotly · Groq AI          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import pickle
import joblib
import warnings
import os
import io
from datetime import datetime
from dotenv import load_dotenv
try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 🔑 SECRETS — loaded from environment variables / .env file / st.secrets
#    NEVER hardcode API keys or DB passwords directly in source code.
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()  # reads a local .env file if present (see .env.example)

def get_secret(key: str, default: str = "") -> str:
    """
    Look up a secret with this priority:
    1. Streamlit's st.secrets (used on Streamlit Community Cloud)
    2. Environment variable (loaded from .env locally, or set by the host)
    3. Provided default (usually empty)
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL   = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

MYSQL_CONFIG = {
    "host":     get_secret("MYSQL_HOST", "localhost"),
    "port":     int(get_secret("MYSQL_PORT", "3306")),
    "user":     get_secret("MYSQL_USER", "root"),
    "password": get_secret("MYSQL_PASSWORD", ""),
    "database": get_secret("MYSQL_DATABASE", "smart_campus"),
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Campus Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME CSS — Blue & White University  (sidebar fully visible)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, .stApp {
    background-color: #f0f4fa;
    color: #1a2a4a;
    font-family: 'Inter', sans-serif;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a2463 0%, #1b4fa8 60%, #1565c0 100%) !important;
    border-right: 3px solid rgba(255,255,255,0.15) !important;
}

/* All text in sidebar: white */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* Sidebar selectbox & multiselect controls */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="select"] input,
section[data-testid="stSidebar"] [class*="control"],
section[data-testid="stSidebar"] [class*="placeholder"],
section[data-testid="stSidebar"] [class*="singleValue"] {
    background-color: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.3) !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #ffffff !important;
}

/* Tags / chips inside multiselect */
section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: rgba(255,255,255,0.25) !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] span,
section[data-testid="stSidebar"] [data-baseweb="tag"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* Dropdown menu that pops out */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {
    background-color: #0d2d6b !important;
    color: #ffffff !important;
}
[data-baseweb="popover"] li:hover {
    background-color: rgba(255,255,255,0.15) !important;
}

/* Sidebar markdown hr */
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25) !important;
}

/* Sidebar stSelectbox label */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {
    color: #e3f2fd !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── METRIC CARDS ── */
div[data-testid="stMetric"] {
    background: white; border: 1px solid #dce8f7;
    border-radius: 12px; padding: 14px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.08);
}
.kpi-card {
    background: white; border: 1px solid #dce8f7;
    border-radius: 14px; padding: 22px 18px;
    text-align: center; box-shadow: 0 4px 16px rgba(21,101,192,0.10);
    transition: transform .2s, box-shadow .2s; height: 130px;
    display: flex; flex-direction: column; justify-content: center;
    border-top: 4px solid #1565c0;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(21,101,192,0.18); }
.kpi-label { font-size: 11px; color: #5c7a9e; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; font-weight: 600; }
.kpi-value { font-size: 28px; font-weight: 700; color: #0a2463; line-height: 1.1; }
.kpi-sub   { font-size: 12px; margin-top: 6px; font-weight: 500; }
.kpi-sub.good    { color: #2e7d32; }
.kpi-sub.bad     { color: #c62828; }
.kpi-sub.neutral { color: #f57c00; }

/* ── ALERT BOXES ── */
.alert-red    { background:#fff5f5; border-left:4px solid #e53935; border-radius:8px; padding:12px 16px; margin:5px 0; color:#b71c1c; font-size:13px; }
.alert-yellow { background:#fffde7; border-left:4px solid #f9a825; border-radius:8px; padding:12px 16px; margin:5px 0; color:#e65100; font-size:13px; }
.alert-green  { background:#f1f8e9; border-left:4px solid #43a047; border-radius:8px; padding:12px 16px; margin:5px 0; color:#1b5e20; font-size:13px; }
.alert-blue   { background:#e3f2fd; border-left:4px solid #1565c0; border-radius:8px; padding:12px 16px; margin:5px 0; color:#0d47a1; font-size:13px; }
.alert-good   { background:#f1f8e9; border-left:4px solid #43a047; border-radius:8px; padding:12px 16px; margin:5px 0; color:#1b5e20; font-size:13px; }
.alert-neutral{ background:#fffde7; border-left:4px solid #f9a825; border-radius:8px; padding:12px 16px; margin:5px 0; color:#e65100; font-size:13px; }
.alert-bad    { background:#fff5f5; border-left:4px solid #e53935; border-radius:8px; padding:12px 16px; margin:5px 0; color:#b71c1c; font-size:13px; }

/* ── AI RECOMMENDATION CARD ── */
.ai-rec-container {
    background: linear-gradient(135deg, #e8f0fe 0%, #f0f4fa 100%);
    border: 1px solid #90b4e8; border-radius: 14px;
    padding: 22px 26px; margin: 10px 0;
    position: relative; overflow: hidden;
}
.ai-rec-container::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1565c0, #42a5f5, #1565c0);
}
.ai-rec-header { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 700; color: #0a2463; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.ai-rec-content { color: #1a2a4a; font-size: 13px; line-height: 1.8; }

/* ── MISC ── */
.section-title {
    font-size: 15px; font-weight: 700; color: #0a2463;
    border-bottom: 2px solid #bbdefb; padding-bottom: 8px;
    margin: 20px 0 14px 0; text-transform: uppercase; letter-spacing: 0.5px;
}
.page-banner {
    background: linear-gradient(90deg, #0a2463 0%, #1565c0 60%, #1e88e5 100%);
    border-radius: 14px; padding: 22px 28px; margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(21,101,192,0.25);
}
.page-banner h1 { font-size: 22px; font-weight: 700; color: white; margin: 0; font-family: 'Playfair Display', serif; }
.page-banner p  { font-size: 13px; color: #90caf9; margin: 5px 0 0 0; }

.sdg-badge {
    display: inline-block; padding: 4px 10px;
    border-radius: 20px; font-size: 11px; font-weight: 700;
    margin: 2px; letter-spacing: 0.5px;
}
.sdg-4  { background: #c8e6c9; color: #1b5e20; }
.sdg-9  { background: #ffe0b2; color: #bf360c; }
.sdg-10 { background: #e1bee7; color: #4a148c; }
.sdg-11 { background: #b2ebf2; color: #006064; }

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
    border: none !important; color: white !important;
    font-weight: 600 !important; letter-spacing: 0.5px !important;
}
div[data-testid="stButton"] > button {
    border: 1px solid #90b4e8 !important; color: #0a2463 !important;
    background: white !important; font-weight: 500 !important;
}
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LIGHT_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="#f8fbff",
    font=dict(color="#1a2a4a", family="Inter"),
    xaxis=dict(gridcolor="#e3eaf5", showgrid=True, linecolor="#dce8f7"),
    yaxis=dict(gridcolor="#e3eaf5", showgrid=True, linecolor="#dce8f7"),
    margin=dict(l=30, r=20, t=50, b=30),
    legend=dict(bgcolor="white", bordercolor="#dce8f7", borderwidth=1)
)

COLORS = ["#1565c0","#42a5f5","#26a69a","#ef5350","#ab47bc","#ff7043",
          "#66bb6a","#ffa726","#29b6f6","#ec407a","#8d6e63","#78909c"]
UNI_BLUE   = "#1565c0"
UNI_LIGHT  = "#42a5f5"
UNI_GREEN  = "#2e7d32"
UNI_RED    = "#c62828"
UNI_ORANGE = "#e65100"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# SAFE MODEL LOADER  (NumPy cross-version compatible)
# ─────────────────────────────────────────────────────────────────────────────
class _SafeUnpickler(pickle.Unpickler):
    class _DummyBitGenerator:
        def __setstate__(self, state): pass
        def __reduce__(self): return (self.__class__, ())

    def find_class(self, module, name):
        if module in ("numpy.random._pickle", "numpy.random.bit_generator", "numpy.random"):
            if name in ("__bit_generator_ctor", "__randomstate_ctor", "BitGenerator"):
                return lambda *a, **kw: _SafeUnpickler._DummyBitGenerator()
        if module.startswith("numpy.random") and (
            name in ("MT19937","PCG64","Philox","SFC64","Generator","RandomState") or "BitGenerator" in name
        ):
            return lambda *a, **kw: _SafeUnpickler._DummyBitGenerator()
        return super().find_class(module, name)


def _safe_load(fpath: str):
    try:
        with open(fpath, "rb") as f:
            return _SafeUnpickler(f).load()
    except Exception:
        pass
    try:
        return joblib.load(fpath)
    except Exception as e:
        raise RuntimeError(f"Could not load {fpath}: {e}") from e


@st.cache_resource
def load_models():
    warnings.filterwarnings('ignore')
    model_files = {
        "gpa_regressor":        "models/gpa_regressor.pkl",
        "pass_fail_classifier": "models/pass_fail_classifier.pkl",
        "at_risk_model":        "models/at_risk_model.pkl",
        "scaler_gpa":           "models/scaler_gpa.pkl",
        "scaler_pf":            "models/scaler_pf.pkl",
        "scaler_risk":          "models/scaler_risk.pkl",
        "le_pass_fail":         "models/le_pass_fail.pkl",
        "le_risk":              "models/le_risk.pkl",
        "encoders":             "models/encoders.pkl",
        "feature_cols":         "models/feature_cols.pkl",
    }
    models = {}
    for key, fname in model_files.items():
        fpath = os.path.join(DATA_DIR, fname)
        models[key] = _safe_load(fpath) if os.path.exists(fpath) else None

    # ── Optional extras: model comparison leaderboard + feature importance ──
    # These are produced by the notebook (see "Model Metrics Export" cell).
    # They're optional so the app still runs fine on older model folders
    # that were trained before this export step existed.
    extra_files = {
        "model_metrics":       "models/model_metrics.pkl",       # accuracy/AUC leaderboard for all classifiers
        "feature_importance":  "models/feature_importance.pkl",  # top predictors for pass/fail + at-risk models
    }
    for key, fname in extra_files.items():
        fpath = os.path.join(DATA_DIR, fname)
        models[key] = _safe_load(fpath) if os.path.exists(fpath) else None

    return models


def encode_input(raw: dict, encoders: dict, feature_cols: list) -> pd.DataFrame:
    warnings.filterwarnings('ignore')
    row = pd.DataFrame([raw])
    for col, le in encoders.items():
        if col in row.columns:
            known = set(le.classes_)
            def _match(val, known=known, le=le):
                if val in known: return val
                for k in known:
                    if str(k).lower() == str(val).lower(): return k
                return list(known)[0]
            row[col] = row[col].astype(str).apply(_match)
            row[col] = le.transform(row[col])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    return row[feature_cols]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    student   = pd.read_csv(os.path.join(DATA_DIR, "student_dataset.csv"))
    faculty   = pd.read_csv(os.path.join(DATA_DIR, "faculty_dataset.csv"))
    classroom = pd.read_csv(os.path.join(DATA_DIR, "classroom_dataset.csv"))

    student["gpa_band"] = pd.cut(student["current_gpa"],
        bins=[0,4,5.5,7,8.5,10], labels=["<4.0","4-5.5","5.5-7.0","7.0-8.5","8.5+"])
    student["attendance_band"] = pd.cut(student["attendance_percentage"],
        bins=[0,60,75,85,100], labels=["<60%","60-75%","75-85%","85%+"])
    student["risk_label"]  = student["at_risk_student"].map({1:"At Risk", 0:"Safe"})
    student["pass_label"]  = student["pass_fail"].map({1:"Pass", 0:"Fail"})

    student["engagement_score"] = (
        student["library_usage"] * 0.3 +
        student["online_course_completion"] * 0.3 +
        student["login_frequency"] * 0.2 +
        student["study_hours_per_week"] * 0.5
    ).round(2)

    student["placement_score"] = (
        student["current_gpa"] * 7 +
        student["online_course_completion"] * 0.15 +
        student["extracurricular_score"] * 0.10 +
        student["lab_performance"] * 0.05
    ).round(2)
    student["placement_ready"] = (
        (student["current_gpa"] >= 7.0) &
        (student["backlogs"] == 0) &
        (student["attendance_percentage"] >= 75)
    )

    faculty["performance_score"] = (
        faculty["student_feedback_score"] * 20 +
        faculty["research_output"] * 5 -
        (faculty["workload_hours"] - faculty["workload_hours"].mean()).abs()
    )
    faculty["workload_status"] = faculty["workload_hours"].apply(
        lambda v: "🔴 Overloaded" if v > 22 else ("🟡 High" if v > 18 else "🟢 Normal")
    )

    classroom["status_color"] = classroom["maintenance_status"].map(
        {"Good":"green","Average":"orange","Poor":"red"}
    )
    classroom["utilization_band"] = pd.cut(classroom["utilization_rate"],
        bins=[0,40,60,80,100],
        labels=["Low (<40%)","Medium (40-60%)","High (60-80%)","Very High (80%+)"])

    # ── BACKEND MySQL DATABASE CONNECTIVITY ──────────────────────────────────
    # MYSQL_CONFIG is now defined once near the top of the file, loaded from
    # environment variables / .env / st.secrets — see get_secret() above.
    if not MYSQL_AVAILABLE:
        st.session_state["db_connected"] = False
        st.session_state["db_error"] = "mysql-connector-python not installed. Run: pip install mysql-connector-python"
    else:
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor()

            def _sync_table(df, table_name):
                """Drop-and-recreate table, then insert all rows."""
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                cols = []
                for col, dtype in df.dtypes.items():
                    if "int" in str(dtype):
                        cols.append(f"`{col}` BIGINT")
                    elif "float" in str(dtype):
                        cols.append(f"`{col}` DOUBLE")
                    else:
                        cols.append(f"`{col}` VARCHAR(255)")
                cursor.execute(f"CREATE TABLE `{table_name}` ({', '.join(cols)})")
                placeholders = ", ".join(["%s"] * len(df.columns))
                insert_sql = f"INSERT INTO `{table_name}` VALUES ({placeholders})"
                rows = [tuple(None if (isinstance(v, float) and v != v) else v for v in row)
                        for row in df.itertuples(index=False, name=None)]
                cursor.executemany(insert_sql, rows)
                conn.commit()

            _sync_table(student,   "students")
            _sync_table(faculty,   "faculty")
            _sync_table(classroom, "classrooms")

            # Sync log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `sync_log` (
                    `synced_at` VARCHAR(30),
                    `student_rows` INT,
                    `faculty_rows` INT,
                    `classroom_rows` INT
                )
            """)
            cursor.execute(
                "INSERT INTO `sync_log` VALUES (%s, %s, %s, %s)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(student), len(faculty), len(classroom))
            )
            conn.commit()
            cursor.close()
            conn.close()
            st.session_state["db_connected"] = True
            st.session_state["db_config"]    = MYSQL_CONFIG
        except Exception as _db_err:
            st.session_state["db_connected"] = False
            st.session_state["db_error"]     = str(_db_err)
    # ─────────────────────────────────────────────────────────────────────────
    return student, faculty, classroom


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub="", sub_class="neutral"):
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub {sub_class}">{sub}</div>
    </div>"""

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def banner(icon, title, desc, sdgs=None):
    sdg_html = ""
    if sdgs:
        sdg_html = "<div style='margin-top:8px;'>"
        for s in sdgs:
            sdg_html += f'<span class="sdg-badge sdg-{s}">🌍 SDG {s}</span>'
        sdg_html += "</div>"
    st.markdown(f"""<div class="page-banner">
        <h1>{icon} {title}</h1>
        <p>{desc}</p>
        {sdg_html}
    </div>""", unsafe_allow_html=True)

def fmt_pct(v): return f"{v:.1f}%"
def fmt_num(v): return f"{v:,.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# GROQ API CALL  (replaces Cohere)
# ─────────────────────────────────────────────────────────────────────────────
def call_groq(system_prompt: str, user_message: str, max_tokens: int = 600) -> str:
    """
    Calls Groq's OpenAI-compatible chat completions endpoint.
    Returns the assistant reply text, or an error string.
    """
    if not GROQ_API_KEY:
        return ("❌ No Groq API key found. Set GROQ_API_KEY as an environment "
                "variable, in a local .env file, or in Streamlit secrets — "
                "see .env.example for details.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"❌ Groq API error ({resp.status_code}): {resp.text[:300]}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# AI RECOMMENDATIONS ENGINE  (Groq)
# ─────────────────────────────────────────────────────────────────────────────
def render_ai_section(context: dict, page_type: str, section_key: str):
    section("🤖 AI-POWERED INSIGHTS (Groq AI)")

    active_key  = f"active_card_{section_key}"
    content_key = f"card_content_{section_key}"
    trans_key   = f"translation_{section_key}"
    lang_key    = f"lang_{section_key}"

    for k, default in [(active_key, None),(content_key, {}),(trans_key, ""),(lang_key, "Telugu")]:
        if k not in st.session_state:
            st.session_state[k] = default

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: clicked_analysis  = st.button("📊 AI Analysis",       key=f"btn_a_{section_key}", use_container_width=True)
    with b2: clicked_causes    = st.button("🔍 Problem Causes",    key=f"btn_c_{section_key}", use_container_width=True)
    with b3: clicked_recs      = st.button("✅ Recommendations",   key=f"btn_r_{section_key}", use_container_width=True)
    with b4: clicked_plan      = st.button("🗓️ Action Plan",       key=f"btn_p_{section_key}", use_container_width=True)
    with b5: clicked_outcomes  = st.button("🎯 Expected Outcomes", key=f"btn_o_{section_key}", use_container_width=True)

    def build_data_summary():
        lines = [f"Smart Campus Analytics System — {page_type.replace('_',' ').title()} Data:"]
        lines.append("(IEEE Research Project | SDG 4, SDG 9, SDG 10, SDG 11)")
        for k, v in context.items():
            lines.append(f"- {k.replace('_',' ').title()}: {v}")
        return "\n".join(lines)

    CARD_CONFIG = {
        "analysis":  {
            "spinner":      "📊 Generating AI Analysis…",
            "system":       "You are a university academic analyst for an IEEE Smart Campus project aligned with SDG 4 (Quality Education). Write 3-4 sentences summarizing the overall campus performance based on the data. Be specific with numbers. No bullet points — clear paragraph form only.",
            "title":        "📊 AI ANALYSIS",
            "header_color": "#0a2463",
            "border_color": "#1565c0",
        },
        "causes":    {
            "spinner":      "🔍 Identifying Problem Causes…",
            "system":       "You are a university academic analyst. List 3-4 bullet points identifying root causes visible in the campus data, referencing relevant SDGs where appropriate. Format: **Cause title:** one sentence explanation.",
            "title":        "🔍 PROBLEM CAUSES",
            "header_color": "#e65100",
            "border_color": "#f57c00",
        },
        "recs":      {
            "spinner":      "✅ Generating Recommendations…",
            "system":       "You are a university academic analyst. List 4-5 bullet points of specific, actionable recommendations aligned with SDG 4 (student success), SDG 9 (digital innovation), SDG 11 (campus sustainability). Format: **Action title:** one sentence on what to do.",
            "title":        "✅ RECOMMENDATIONS",
            "header_color": "#1b5e20",
            "border_color": "#2e7d32",
        },
        "plan":      {
            "spinner":      "🗓️ Building Action Plan…",
            "system":       (
                "You are a university operations expert supporting SDG 4 and SDG 11 goals. "
                "Create a 30-Day Campus Improvement Action Plan. Structure:\n"
                "**Week 1 (Days 1-7):** 2 immediate actions\n"
                "**Week 2 (Days 8-14):** 2 short-term actions\n"
                "**Week 3-4 (Days 15-30):** 2 medium-term actions\n"
                "Each action: clear owner role (e.g. Dean, HOD, Counselor) and expected impact."
            ),
            "title":        "🗓️ 30-DAY ACTION PLAN",
            "header_color": "#4a148c",
            "border_color": "#7b1fa2",
        },
        "outcomes":  {
            "spinner":      "🎯 Projecting Expected Outcomes…",
            "system":       "You are a university academic analyst. Project Expected Outcomes if the recommendations are implemented within 30 days, citing relevant SDG targets. List 4-5 bullet points. Format: **Metric name:** expected improvement with a % estimate. Be realistic and data-driven.",
            "title":        "🎯 EXPECTED OUTCOMES",
            "header_color": "#b71c1c",
            "border_color": "#c62828",
        },
    }

    triggered = None
    if clicked_analysis:   triggered = "analysis"
    elif clicked_causes:   triggered = "causes"
    elif clicked_recs:     triggered = "recs"
    elif clicked_plan:     triggered = "plan"
    elif clicked_outcomes: triggered = "outcomes"

    if triggered:
        st.session_state[active_key] = triggered
        st.session_state[trans_key]  = ""
        if triggered not in st.session_state[content_key]:
            cfg      = CARD_CONFIG[triggered]
            base_msg = build_data_summary()
            if triggered in ("plan","outcomes") and "recs" in st.session_state[content_key]:
                base_msg += f"\n\nExisting Recommendations:\n{st.session_state[content_key]['recs']}"
            with st.spinner(cfg["spinner"]):
                result = call_groq(cfg["system"], base_msg, max_tokens=600)
                st.session_state[content_key][triggered] = result

    active = st.session_state[active_key]
    if active:
        cfg  = CARD_CONFIG[active]
        text = st.session_state[content_key].get(active, "")
        html = text.replace("\n", "<br>")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ai-rec-container" style="border-color:{cfg['border_color']};">
            <div class="ai-rec-header" style="color:{cfg['header_color']};">{cfg['title']}</div>
            <div class="ai-rec-content">{html}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:#5c7a9e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🌐 TRANSLATE THIS REPORT</div>', unsafe_allow_html=True)

        lang_col, btn_col, clear_col = st.columns([2,1,1])
        LANGUAGES = ["Telugu","Hindi","Tamil","Kannada","Marathi","Bengali","Gujarati","Punjabi","Urdu","Spanish","French","German","Arabic","Chinese","Japanese"]
        with lang_col:
            lang_choice = st.selectbox("Select Language", LANGUAGES,
                index=LANGUAGES.index(st.session_state[lang_key]),
                key=f"lang_select_{section_key}", label_visibility="collapsed")
            st.session_state[lang_key] = lang_choice
        with btn_col:
            translate_clicked = st.button(f"🌐 Translate to {lang_choice}", key=f"btn_translate_{section_key}", use_container_width=True)
        with clear_col:
            if st.button("🗑️ Clear", key=f"clear_{section_key}", use_container_width=True):
                st.session_state[active_key]  = None
                st.session_state[content_key] = {}
                st.session_state[trans_key]   = ""
                st.rerun()

        if translate_clicked:
            with st.spinner(f"🌐 Translating to {lang_choice}…"):
                sys_msg = (
                    f"You are a professional translator. Translate the following campus analytics report "
                    f"into {lang_choice}. Preserve all formatting and bullet points. "
                    f"Only return the translated text, nothing else."
                )
                result = call_groq(sys_msg, text, max_tokens=700)
                st.session_state[trans_key] = result

        if st.session_state[trans_key]:
            trans_html = st.session_state[trans_key].replace("\n","<br>")
            st.markdown(f"""
            <div class="ai-rec-container" style="border-color:#7b1fa2;margin-top:8px;">
                <div class="ai-rec-header" style="color:#4a148c;">🌐 {lang_choice.upper()} TRANSLATION</div>
                <div class="ai-rec-content">{trans_html}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 0 — ABOUT / PROJECT INFO
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    banner("📌", "AI-Powered Smart Campus Analytics System",
           "IEEE Research Project — Student & Resource Management using Machine Learning",
           sdgs=["4","9","10","11"])

    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:24px 28px;border:1px solid #dce8f7;box-shadow:0 4px 16px rgba(21,101,192,0.08);">
        <h3 style="color:#0a2463;margin-top:0;">📄 Abstract</h3>
        <p style="color:#1a2a4a;font-size:13px;line-height:1.9;">
        The rapid digitalization of educational institutions has led to the generation of vast amounts of academic
        and administrative data. This <b>AI-Powered Smart Campus Analytics System</b> integrates machine learning
        algorithms with data analytics techniques to process student academic records, attendance, faculty utilization,
        and infrastructure usage.
        </p>
        <p style="color:#1a2a4a;font-size:13px;line-height:1.9;">
        The system employs predictive models to <b>identify at-risk students</b>, forecast academic outcomes,
        and recommend personalized interventions. It also analyzes institutional resources to improve allocation
        efficiency and reduce operational costs.
        </p>
        <p style="color:#1a2a4a;font-size:13px;line-height:1.9;">
        Experimental results demonstrate improved prediction accuracy and efficient resource utilization compared
        to traditional methods, contributing to the development of <b>intelligent educational ecosystems</b>.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:24px 28px;border:1px solid #dce8f7;box-shadow:0 4px 16px rgba(21,101,192,0.08);">
        <h3 style="color:#0a2463;margin-top:0;">🎯 Objectives</h3>
        <ul style="color:#1a2a4a;font-size:13px;line-height:2;">
            <li>Predict student GPA, Pass/Fail &amp; At-Risk status</li>
            <li>Identify weak students for early intervention</li>
            <li>Optimize resource usage (classrooms, faculty, labs)</li>
            <li>Provide real-time analytics dashboards</li>
            <li>Enable data-driven decision making</li>
        </ul>
        <h3 style="color:#0a2463;">⚙️ Key Modules</h3>
        <ul style="color:#1a2a4a;font-size:13px;line-height:2;">
            <li>📊 Student Performance Prediction (ML Model)</li>
            <li>📅 Attendance &amp; Behavior Analysis</li>
            <li>🏫 Resource Allocation Optimization</li>
            <li>👨‍🏫 Faculty Workload Analysis</li>
            <li>🤖 AI Dashboard &amp; Visualization (Streamlit)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    sdg_info = [
        ("4","Quality Education","#c8e6c9","#1b5e20",
         "Predicts student performance, identifies at-risk students, improves learning outcomes using AI, and enables data-driven teaching strategies."),
        ("9","Industry, Innovation & Infrastructure","#ffe0b2","#bf360c",
         "Uses AI and data analytics, builds smart digital infrastructure for campuses, and promotes innovation in education systems."),
        ("11","Sustainable Cities & Communities","#b2ebf2","#006064",
         "Smart campuses are part of smart cities. Efficient resource utilization (classrooms, energy, faculty) supports campus sustainability."),
        ("10","Reduced Inequalities","#e1bee7","#4a148c",
         "Identifies disadvantaged students across socioeconomic groups and provides equal learning opportunities through targeted interventions."),
    ]
    for col, (num, name, bg, text, desc) in zip([c1,c2,c3,c4], sdg_info):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border-radius:14px;padding:20px;height:200px;">
                <div style="font-size:28px;font-weight:900;color:{text};">SDG {num}</div>
                <div style="font-size:12px;font-weight:700;color:{text};margin-bottom:8px;">{name}</div>
                <div style="font-size:11px;color:#333;line-height:1.7;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("🧠 TECHNOLOGY STACK")
    t1, t2, t3, t4, t5 = st.columns(5)
    tech = [
        ("🐍","Python","Pandas · NumPy · Scikit-learn"),
        ("🤖","Machine Learning","Regression · Classification · GBM"),
        ("📊","Visualization","Plotly · Streamlit Dashboard"),
        ("🗄️","Data","CSV Datasets · 500+ Student Records"),
        ("⚡","AI Engine","Groq LLM · NLP Chatbot"),
    ]
    for col, (icon,name,detail) in zip([t1,t2,t3,t4,t5], tech):
        with col:
            st.markdown(kpi_card(name, icon, detail, "neutral"), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_executive(s, f, c):
    banner("🎓","Campus Executive Dashboard",
           "University-wide KPIs: student performance, faculty efficiency, and campus utilization",
           sdgs=["4","9","11"])

    total_students = len(s)
    pass_rate      = s["pass_fail"].mean() * 100
    avg_gpa        = s["current_gpa"].mean()
    at_risk_count  = s["at_risk_student"].sum()
    avg_attendance = s["attendance_percentage"].mean()
    avg_feedback   = f["student_feedback_score"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kpi_card("Total Students", fmt_num(total_students), "Enrolled","neutral"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Pass Rate", fmt_pct(pass_rate), "✅ Above target" if pass_rate>=75 else "⚠️ Below 75% target","good" if pass_rate>=75 else "bad"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Avg GPA", f"{avg_gpa:.2f}", "Target: 7.0+" if avg_gpa<7 else "✅ On Track","good" if avg_gpa>=7 else "neutral"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("At-Risk Students", fmt_num(at_risk_count), f"{at_risk_count/total_students*100:.1f}% flagged","bad" if at_risk_count>500 else "neutral"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Avg Attendance", fmt_pct(avg_attendance), "⚠️ Below 75%" if avg_attendance<75 else "✅ Good","bad" if avg_attendance<75 else "good"), unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("Faculty Feedback", f"{avg_feedback:.2f}/5", "Student-rated avg","good" if avg_feedback>=4 else "neutral"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        section("🏛️ DEPARTMENT-WISE PASS RATE")
        dept_pass = s.groupby("department")["pass_fail"].mean().reset_index()
        dept_pass["pass_pct"] = dept_pass["pass_fail"]*100
        dept_pass = dept_pass.sort_values("pass_pct", ascending=False)
        bar_colors = [UNI_BLUE if v>=85 else (UNI_ORANGE if v>=75 else UNI_RED) for v in dept_pass["pass_pct"]]
        fig = go.Figure(go.Bar(
            x=dept_pass["department"], y=dept_pass["pass_pct"],
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in dept_pass["pass_pct"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Pass Rate: %{y:.1f}%<extra></extra>"
        ))
        fig.add_hline(y=75, line_color=UNI_RED, line_dash="dot", line_width=1.5,
                      annotation_text="75% Target", annotation_font_color=UNI_RED)
        fig.update_layout(**LIGHT_LAYOUT, height=320, showlegend=False,
                          yaxis_title="Pass Rate (%)", yaxis_range=[0,110])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("📊 GPA DISTRIBUTION BY DEPARTMENT")
        fig2 = px.box(s, x="department", y="current_gpa", color="department",
                      color_discrete_sequence=COLORS,
                      labels={"current_gpa":"GPA","department":"Department"})
        fig2.update_layout(**LIGHT_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    section("📈 AT-RISK STUDENTS BY YEAR & DEPARTMENT (SDG 4 — Early Intervention)")
    risk_heat  = s.groupby(["year_of_study","department"])["at_risk_student"].mean().reset_index()
    risk_pivot = risk_heat.pivot(index="department", columns="year_of_study", values="at_risk_student")
    fig3 = go.Figure(go.Heatmap(
        z=risk_pivot.values*100,
        x=[f"Year {col}" for col in risk_pivot.columns],
        y=risk_pivot.index.tolist(),
        colorscale=[[0,"#e3f2fd"],[0.5,"#90caf9"],[1,"#b71c1c"]],
        hovertemplate="Dept: <b>%{y}</b><br>Year: <b>%{x}</b><br>At-Risk: <b>%{z:.1f}%</b><extra></extra>",
        colorbar=dict(title=dict(text="At-Risk %",font=dict(color="#1a2a4a")),tickfont=dict(color="#1a2a4a"))
    ))
    fig3.update_layout(paper_bgcolor="white", font=dict(color="#1a2a4a"),
                       height=300, margin=dict(l=80,r=40,t=40,b=40))
    st.plotly_chart(fig3, use_container_width=True)

    exec_context = {
        "total_students": total_students,
        "pass_rate": fmt_pct(pass_rate),
        "average_gpa": f"{avg_gpa:.2f}",
        "at_risk_students": int(at_risk_count),
        "average_attendance": fmt_pct(avg_attendance),
        "faculty_feedback_score": f"{avg_feedback:.2f}/5",
        "worst_department_pass": dept_pass.iloc[-1]["department"],
        "best_department_pass":  dept_pass.iloc[0]["department"],
        "sdg_alignment": "SDG 4 (Quality Education), SDG 9 (Innovation), SDG 11 (Sustainability)",
    }
    render_ai_section(exec_context, "executive", "exec")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — STUDENT PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def page_student(s):
    banner("📚","Student Performance Analytics",
           "Deep-dive into GPA, attendance, backlogs, study habits, and at-risk identification",
           sdgs=["4","10"])

    avg_gpa        = s["current_gpa"].mean()
    avg_attendance = s["attendance_percentage"].mean()
    avg_internal   = s["internal_marks"].mean()
    avg_backlogs   = s["backlogs"].mean()
    fail_count     = (s["pass_fail"]==0).sum()
    risk_count     = s["at_risk_student"].sum()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kpi_card("Avg GPA", f"{avg_gpa:.2f}", "/10 scale","good" if avg_gpa>=7 else "neutral"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Avg Attendance", fmt_pct(avg_attendance), "⚠️ Low" if avg_attendance<75 else "✅ Good","bad" if avg_attendance<75 else "good"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Avg Internal Marks", f"{avg_internal:.1f}", "/100","neutral"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Avg Backlogs", f"{avg_backlogs:.2f}", "Per student","bad" if avg_backlogs>1 else "good"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Failed Students", fmt_num(fail_count), f"{fail_count/len(s)*100:.1f}% of total","bad"), unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("At-Risk Students", fmt_num(risk_count), f"{risk_count/len(s)*100:.1f}% flagged","bad" if risk_count>500 else "neutral"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📉 GPA vs ATTENDANCE SCATTER")
        sample = s.sample(min(len(s),1500), random_state=42)
        fig = px.scatter(sample, x="attendance_percentage", y="current_gpa",
                         color="risk_label",
                         color_discrete_map={"At Risk":UNI_RED,"Safe":UNI_BLUE},
                         size="study_hours_per_week", size_max=12,
                         hover_data={"department":True,"year_of_study":True,"backlogs":True},
                         labels={"attendance_percentage":"Attendance %","current_gpa":"Current GPA"})
        fig.add_vline(x=75, line_color=UNI_ORANGE, line_dash="dash", line_width=1.5)
        fig.add_hline(y=7.0, line_color=UNI_ORANGE, line_dash="dash", line_width=1.5)
        fig.update_layout(**LIGHT_LAYOUT, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🎯 PASS/FAIL BY STRESS LEVEL")
        pf_stress = s.groupby(["stress_level","pass_label"]).size().reset_index(name="count")
        fig2 = px.bar(pf_stress, x="stress_level", y="count", color="pass_label",
                      color_discrete_map={"Pass":UNI_BLUE,"Fail":UNI_RED}, barmode="group",
                      labels={"stress_level":"Stress Level","count":"Students","pass_label":"Result"})
        fig2.update_layout(**LIGHT_LAYOUT, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("🏠 HOSTEL vs DAY SCHOLAR — GPA (SDG 10)")
        hostel_summary = s.groupby("hostel_status").agg(
            avg_gpa=("current_gpa","mean"),
            avg_attendance=("attendance_percentage","mean"),
            count=("student_id","count")
        ).reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=hostel_summary["hostel_status"], y=hostel_summary["avg_gpa"],
            marker_color=[UNI_BLUE, UNI_LIGHT][:len(hostel_summary)],
            text=[f"{v:.2f}" for v in hostel_summary["avg_gpa"]], textposition="outside",
            name="Avg GPA"
        ))
        fig3.update_layout(**LIGHT_LAYOUT, height=340, showlegend=False,
                           yaxis_title="Average GPA", xaxis_title="Status",
                           title="Avg GPA — Hostel vs Day Scholar")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("📖 STUDY HOURS vs GPA BY DEPARTMENT")
        dept_study = s.groupby("department").agg(
            avg_study=("study_hours_per_week","mean"),
            avg_gpa=("current_gpa","mean"),
            count=("student_id","count")
        ).reset_index()
        fig4 = px.scatter(dept_study, x="avg_study", y="avg_gpa", text="department",
                          size="count", color="department", color_discrete_sequence=COLORS,
                          labels={"avg_study":"Avg Study Hours/Week","avg_gpa":"Avg GPA"})
        fig4.update_traces(textposition="top center", textfont_size=10)
        fig4.update_layout(**LIGHT_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    section("🔍 TOP 20 AT-RISK STUDENTS — Immediate Intervention Required (SDG 4)")
    at_risk_df = s[s["at_risk_student"]==1][["student_id","department","year_of_study",
        "current_gpa","attendance_percentage","backlogs","stress_level","pass_label"]].copy()
    at_risk_df = at_risk_df.sort_values("current_gpa").head(20)
    at_risk_df.columns = ["Student ID","Dept","Year","GPA","Attendance %","Backlogs","Stress","Result"]
    st.dataframe(at_risk_df, use_container_width=True, hide_index=True)

    section("⚖️ GPA EQUITY BY SOCIOECONOMIC STATUS (SDG 10 — Reduced Inequalities)")
    ses_order = [x for x in ["Low","Medium","High"] if x in s["socio_economic_status"].unique()]
    ses_gpa   = s.groupby("socio_economic_status")["current_gpa"].mean().reindex(ses_order)
    fig5 = go.Figure(go.Bar(
        x=ses_gpa.index, y=ses_gpa.values,
        marker_color=[UNI_RED,UNI_ORANGE,UNI_GREEN][:len(ses_order)],
        text=[f"{v:.2f}" for v in ses_gpa.values], textposition="outside"
    ))
    fig5.update_layout(**LIGHT_LAYOUT, height=300, showlegend=False,
                       yaxis_title="Avg GPA", title="GPA Equity Across Socioeconomic Groups")
    st.plotly_chart(fig5, use_container_width=True)

    student_context = {
        "average_gpa": f"{avg_gpa:.2f}",
        "average_attendance": fmt_pct(avg_attendance),
        "average_internal_marks": f"{avg_internal:.1f}",
        "average_backlogs": f"{avg_backlogs:.2f}",
        "failed_students": int(fail_count),
        "fail_rate": fmt_pct(fail_count/len(s)*100),
        "at_risk_students": int(risk_count),
        "at_risk_rate": fmt_pct(risk_count/len(s)*100),
        "worst_department": s.groupby("department")["current_gpa"].mean().idxmin(),
        "best_department":  s.groupby("department")["current_gpa"].mean().idxmax(),
        "sdg_4_focus": "Early identification of at-risk students for timely intervention",
        "sdg_10_equity": f"GPA gap between Low vs High SES: {abs(s[s['socio_economic_status']=='High']['current_gpa'].mean() - s[s['socio_economic_status']=='Low']['current_gpa'].mean()):.2f}",
    }
    render_ai_section(student_context, "student_performance", "student")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ATTENDANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_attendance(s):
    banner("📅","Attendance Analytics",
           "Track attendance patterns across departments, years, and student demographics",
           sdgs=["4"])

    avg_att  = s["attendance_percentage"].mean()
    below_75 = (s["attendance_percentage"] < 75).sum()
    below_60 = (s["attendance_percentage"] < 60).sum()
    above_85 = (s["attendance_percentage"] >= 85).sum()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Avg Attendance", fmt_pct(avg_att), "All students","bad" if avg_att<75 else "good"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Below 75%", fmt_num(below_75), f"{below_75/len(s)*100:.1f}% — Detention Risk","bad"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Critical (<60%)", fmt_num(below_60), "Immediate action needed","bad"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Excellent (85%+)", fmt_num(above_85), f"{above_85/len(s)*100:.1f}% — Star students","good"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📊 ATTENDANCE BAND DISTRIBUTION")
        att_band = s["attendance_band"].value_counts().reset_index()
        att_band.columns = ["Band","Count"]
        att_band = att_band.sort_values("Band")
        colors_map = {"<60%":UNI_RED,"60-75%":UNI_ORANGE,"75-85%":UNI_LIGHT,"85%+":UNI_GREEN}
        fig = px.pie(att_band, names="Band", values="Count",
                     color="Band", color_discrete_map=colors_map, hole=0.5)
        fig.update_layout(paper_bgcolor="white", font=dict(color="#1a2a4a"), height=320,
                          margin=dict(l=10,r=10,t=50,b=10), legend=dict(bgcolor="white"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🏛️ DEPARTMENT-WISE AVERAGE ATTENDANCE")
        dept_att = s.groupby("department")["attendance_percentage"].mean().reset_index()
        dept_att = dept_att.sort_values("attendance_percentage")
        bar_colors = [UNI_RED if v<75 else (UNI_ORANGE if v<80 else UNI_BLUE) for v in dept_att["attendance_percentage"]]
        fig2 = go.Figure(go.Bar(
            y=dept_att["department"], x=dept_att["attendance_percentage"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in dept_att["attendance_percentage"]], textposition="outside"
        ))
        fig2.add_vline(x=75, line_color=UNI_RED, line_dash="dot", line_width=1.5)
        fig2.update_layout(**LIGHT_LAYOUT, height=320, showlegend=False, xaxis_title="Avg Attendance %")
        st.plotly_chart(fig2, use_container_width=True)

    section("📉 ATTENDANCE vs GPA — YEAR-WISE TREND")
    fig3 = px.scatter(s.sample(min(len(s),2000), random_state=1),
                   x="attendance_percentage", y="current_gpa",
                   color="year_of_study", facet_col="year_of_study",
                   color_discrete_sequence=COLORS,
                   labels={"attendance_percentage":"Attendance %","current_gpa":"GPA","year_of_study":"Year"})
    fig3.update_layout(**LIGHT_LAYOUT, height=320)
    st.plotly_chart(fig3, use_container_width=True)

    att_context = {
        "average_attendance": fmt_pct(avg_att),
        "students_below_75_percent": int(below_75),
        "critical_below_60_percent": int(below_60),
        "excellent_above_85_percent": int(above_85),
        "worst_department_attendance": s.groupby("department")["attendance_percentage"].mean().idxmin(),
        "best_department_attendance":  s.groupby("department")["attendance_percentage"].mean().idxmax(),
        "correlation_attendance_gpa":  f"{s['attendance_percentage'].corr(s['current_gpa']):.2f}",
        "sdg_4_concern": "Attendance <75% directly impacts SDG 4 learning outcomes",
    }
    render_ai_section(att_context, "attendance", "attendance")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — FACULTY ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_faculty(f):
    banner("👨‍🏫","Faculty Analytics",
           "Workload distribution, feedback scores, research output, and performance evaluation",
           sdgs=["9","4"])

    avg_feedback = f["student_feedback_score"].mean()
    avg_workload = f["workload_hours"].mean()
    avg_research = f["research_output"].mean()
    avg_subjects = f["subjects_handled"].mean()
    overloaded   = (f["workload_hours"] > 22).sum()
    top_perf     = f.loc[f["performance_score"].idxmax(),"faculty_id"]

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(kpi_card("Avg Feedback Score", f"{avg_feedback:.2f}/5", "Student-rated","good" if avg_feedback>=4 else "neutral"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Avg Workload", f"{avg_workload:.1f} hrs", "Per week","bad" if avg_workload>22 else "good"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Avg Research Output", f"{avg_research:.1f}", "Publications/year","neutral"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Avg Subjects", f"{avg_subjects:.1f}", "Handled per faculty","neutral"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Overloaded Faculty", fmt_num(overloaded), ">22 hrs/week","bad" if overloaded>10 else "neutral"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📊 FEEDBACK SCORE BY DEPARTMENT (SDG 4)")
        dept_fb = f.groupby("department")["student_feedback_score"].mean().reset_index()
        dept_fb = dept_fb.sort_values("student_feedback_score", ascending=False)
        bar_colors = [UNI_BLUE if v>=4.0 else (UNI_ORANGE if v>=3.5 else UNI_RED) for v in dept_fb["student_feedback_score"]]
        fig = go.Figure(go.Bar(
            x=dept_fb["department"], y=dept_fb["student_feedback_score"],
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in dept_fb["student_feedback_score"]], textposition="outside"
        ))
        fig.add_hline(y=4.0, line_color=UNI_RED, line_dash="dot", line_width=1.5,
                      annotation_text="4.0 Target", annotation_font_color=UNI_RED)
        fig.update_layout(**LIGHT_LAYOUT, height=320, showlegend=False,
                          yaxis_title="Feedback Score", yaxis_range=[0,5.5])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("⚖️ WORKLOAD vs FEEDBACK (SDG 9 — Work Balance)")
        fig2 = px.scatter(f, x="workload_hours", y="student_feedback_score",
                          color="department", size="research_output", size_max=18,
                          color_discrete_sequence=COLORS,
                          hover_data={"faculty_id":True,"experience_years":True,"subjects_handled":True},
                          labels={"workload_hours":"Workload Hours/Week","student_feedback_score":"Feedback Score"})
        fig2.add_vline(x=22, line_color=UNI_RED, line_dash="dash", line_width=1.5,
                       annotation_text="Max Recommended", annotation_font_color=UNI_RED)
        fig2.update_layout(**LIGHT_LAYOUT, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("🔬 RESEARCH OUTPUT BY DEPARTMENT (SDG 9)")
        dept_res = f.groupby("department")["research_output"].sum().reset_index()
        dept_res = dept_res.sort_values("research_output", ascending=False)
        fig3 = px.bar(dept_res, x="department", y="research_output", color="department",
                      color_discrete_sequence=COLORS,
                      labels={"department":"Department","research_output":"Total Research Output"})
        fig3.update_layout(**LIGHT_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("📈 EXPERIENCE vs FEEDBACK SCORE")
        fig4 = px.scatter(f, x="experience_years", y="student_feedback_score",
                          color="department", trendline="ols", color_discrete_sequence=COLORS,
                          labels={"experience_years":"Experience (Years)","student_feedback_score":"Feedback Score"})
        fig4.update_layout(**LIGHT_LAYOUT, height=320)
        st.plotly_chart(fig4, use_container_width=True)

    section("📋 FACULTY PERFORMANCE TABLE")
    fdisp = f[["faculty_id","department","experience_years","subjects_handled",
               "workload_hours","student_feedback_score","research_output","performance_score","workload_status"]].copy()
    fdisp["performance_score"] = fdisp["performance_score"].round(2)
    fdisp.columns = ["Faculty ID","Dept","Exp (Yrs)","Subjects","Workload Hrs","Feedback","Research","Perf Score","Status"]
    st.dataframe(fdisp.sort_values("Perf Score", ascending=False), use_container_width=True, hide_index=True)

    section("🎯 AT-RISK STUDENT ↔ FACULTY ASSIGNMENT (SDG 4 & SDG 9)")
    st.markdown("""
    <p style='color:#1a2a4a;font-size:13px;margin-bottom:10px;'>
    At-risk students are matched to <b>low-workload faculty</b> within the same department for targeted mentoring.
    Faculty with fewer than <b>18 hrs/week</b> workload are prioritised.
    </p>""", unsafe_allow_html=True)

    # Load student data fresh for cross-page use (s may be filtered)
    try:
        _s_all = pd.read_csv(os.path.join(DATA_DIR, "student_dataset.csv"))
    except Exception:
        _s_all = None

    if _s_all is not None:
        at_risk_students = _s_all[_s_all["at_risk_student"] == 1][
            ["student_id", "department", "current_gpa", "attendance_percentage", "backlogs"]
        ].copy().sort_values("current_gpa").head(20).reset_index(drop=True)

        low_wl_faculty = f[f["workload_hours"] < 18][
            ["faculty_id", "department", "workload_hours", "student_feedback_score"]
        ].copy()

        assignment_rows = []
        for _, stu in at_risk_students.iterrows():
            dept_fac = low_wl_faculty[low_wl_faculty["department"] == stu["department"]]
            if dept_fac.empty:
                dept_fac = low_wl_faculty  # fallback: any low-workload faculty
            if dept_fac.empty:
                assigned_fid, assigned_wl = "Unassigned", "—"
            else:
                best_row = dept_fac.sort_values("workload_hours").iloc[0]
                assigned_fid = best_row["faculty_id"]
                assigned_wl = f"{best_row['workload_hours']:.0f} hrs"
            assignment_rows.append({
                "Student ID": stu["student_id"],
                "Department": stu["department"],
                "GPA": round(stu["current_gpa"], 2),
                "Attendance %": round(stu["attendance_percentage"], 1),
                "Backlogs": int(stu["backlogs"]),
                "Assigned Faculty": assigned_fid,
                "Faculty Workload": assigned_wl,
            })

        assign_df = pd.DataFrame(assignment_rows)
        st.dataframe(assign_df, use_container_width=True, hide_index=True)

        assigned_count = (assign_df["Assigned Faculty"] != "Unassigned").sum()
        unassigned_count = (assign_df["Assigned Faculty"] == "Unassigned").sum()
        col_a, col_b = st.columns(2)
        col_a.success(f"✅ {assigned_count} students successfully matched to a low-workload faculty mentor")
        if unassigned_count:
            col_b.warning(f"⚠️ {unassigned_count} students could not be matched (no low-workload faculty in their dept)")
    else:
        st.info("Student dataset not available for assignment. Please ensure student_dataset.csv is in the data folder.")

    faculty_context = {
        "average_feedback_score": f"{avg_feedback:.2f}/5",
        "average_workload_hours": f"{avg_workload:.1f} hrs/week",
        "average_research_output": f"{avg_research:.1f}",
        "overloaded_faculty_count": int(overloaded),
        "worst_department_feedback": f.groupby("department")["student_feedback_score"].mean().idxmin(),
        "best_department_feedback":  f.groupby("department")["student_feedback_score"].mean().idxmax(),
        "top_performer_faculty": top_perf,
        "sdg_9_innovation": f"Research output total: {f['research_output'].sum()}; Avg: {avg_research:.1f}/faculty",
    }
    render_ai_section(faculty_context, "faculty", "faculty")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — PLACEMENT TRACKER
# ─────────────────────────────────────────────────────────────────────────────
def page_placement(s):
    banner("🎯","Placement Tracker",
           "Estimate placement readiness based on GPA, skills, online learning, and extracurriculars",
           sdgs=["4","9"])

    ready_count     = s["placement_ready"].sum()
    avg_place_score = s["placement_score"].mean()
    zero_backlog    = (s["backlogs"]==0).sum()
    high_gpa        = (s["current_gpa"]>=8.0).sum()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Placement Ready", fmt_num(ready_count), f"{ready_count/len(s)*100:.1f}% of students","good" if ready_count/len(s)>0.6 else "neutral"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Avg Placement Score", f"{avg_place_score:.1f}", "/composite 100","neutral"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Zero Backlogs", fmt_num(zero_backlog), f"{zero_backlog/len(s)*100:.1f}% clean","good"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("GPA ≥ 8.0", fmt_num(high_gpa), "High achievers","good"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("🏛️ PLACEMENT READINESS BY DEPARTMENT")
        dept_ready = s.groupby("department")["placement_ready"].mean().reset_index()
        dept_ready["pct"] = dept_ready["placement_ready"]*100
        dept_ready = dept_ready.sort_values("pct", ascending=False)
        bar_c = [UNI_BLUE if v>=70 else (UNI_ORANGE if v>=50 else UNI_RED) for v in dept_ready["pct"]]
        fig = go.Figure(go.Bar(
            x=dept_ready["department"], y=dept_ready["pct"],
            marker_color=bar_c, text=[f"{v:.1f}%" for v in dept_ready["pct"]], textposition="outside"
        ))
        fig.update_layout(**LIGHT_LAYOUT, height=320, showlegend=False,
                          yaxis_title="Placement Ready %", yaxis_range=[0,110])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("📚 ONLINE COURSE COMPLETION vs GPA")
        fig2 = px.scatter(s.sample(min(len(s),1500),random_state=5),
                          x="online_course_completion", y="current_gpa",
                          color="placement_ready",
                          color_discrete_map={True:UNI_BLUE, False:UNI_RED},
                          hover_data={"department":True,"backlogs":True},
                          labels={"online_course_completion":"Online Course Completion %","current_gpa":"GPA"})
        fig2.update_layout(**LIGHT_LAYOUT, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    section("🎓 TOP 30 STUDENTS BY PLACEMENT SCORE")
    top_students = s.nlargest(30,"placement_score")[["student_id","department","year_of_study",
        "current_gpa","backlogs","attendance_percentage","online_course_completion","placement_score","placement_ready"]].copy()
    top_students["placement_ready"] = top_students["placement_ready"].map({True:"✅ Ready", False:"⚠️ Needs Work"})
    top_students.columns = ["Student ID","Dept","Year","GPA","Backlogs","Attendance %","Online Courses","Place Score","Status"]
    st.dataframe(top_students, use_container_width=True, hide_index=True)

    place_context = {
        "placement_ready_students": int(ready_count),
        "placement_readiness_rate": fmt_pct(ready_count/len(s)*100),
        "average_placement_score": f"{avg_place_score:.1f}",
        "students_with_zero_backlogs": int(zero_backlog),
        "students_with_gpa_above_8": int(high_gpa),
        "worst_department_readiness": dept_ready.iloc[-1]["department"],
        "best_department_readiness":  dept_ready.iloc[0]["department"],
    }
    render_ai_section(place_context, "placement", "placement")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — CAMPUS & CLASSROOM UTILIZATION
# ─────────────────────────────────────────────────────────────────────────────
def page_campus(c):
    banner("🏫","Campus & Classroom Utilization",
           "Monitor room utilization, equipment availability, and maintenance status across blocks",
           sdgs=["11","9"])

    avg_util    = c["utilization_rate"].mean()
    poor_maint  = (c["maintenance_status"]=="Poor").sum()
    no_equip    = (c["equipment_availability"]=="No").sum()
    total_rooms = len(c)
    avg_cap     = c["capacity"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(kpi_card("Total Classrooms", fmt_num(total_rooms), "Across all blocks","neutral"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Avg Utilization", fmt_pct(avg_util), "Room usage rate","good" if avg_util>=60 else "bad"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Avg Capacity", f"{avg_cap:.0f}", "Seats per room","neutral"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Poor Maintenance", fmt_num(poor_maint), "Needs urgent attention","bad"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("No Equipment", fmt_num(no_equip), "Missing AV/Tech gear","bad" if no_equip>10 else "neutral"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("🏗️ UTILIZATION BY BUILDING (SDG 11)")
        bld_util = c.groupby("building")["utilization_rate"].mean().reset_index()
        bld_util = bld_util.sort_values("utilization_rate", ascending=False)
        bar_c = [UNI_BLUE if v>=65 else (UNI_ORANGE if v>=50 else UNI_RED) for v in bld_util["utilization_rate"]]
        fig = go.Figure(go.Bar(
            x=bld_util["building"], y=bld_util["utilization_rate"],
            marker_color=bar_c, text=[f"{v:.1f}%" for v in bld_util["utilization_rate"]], textposition="outside"
        ))
        fig.update_layout(**LIGHT_LAYOUT, height=320, showlegend=False,
                          yaxis_title="Avg Utilization %", yaxis_range=[0,110])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🔧 MAINTENANCE STATUS BREAKDOWN")
        maint = c["maintenance_status"].value_counts().reset_index()
        maint.columns = ["Status","Count"]
        fig2 = px.pie(maint, names="Status", values="Count",
                      color="Status", color_discrete_map={"Good":UNI_GREEN,"Average":UNI_ORANGE,"Poor":UNI_RED}, hole=0.5)
        fig2.update_layout(paper_bgcolor="white", font=dict(color="#1a2a4a"), height=320,
                           margin=dict(l=10,r=10,t=50,b=10), legend=dict(bgcolor="white"))
        st.plotly_chart(fig2, use_container_width=True)

    section("📊 CAPACITY vs UTILIZATION — ALL ROOMS")
    fig3 = px.scatter(c, x="capacity", y="utilization_rate",
                      color="maintenance_status", symbol="equipment_availability",
                      color_discrete_map={"Good":UNI_GREEN,"Average":UNI_ORANGE,"Poor":UNI_RED},
                      hover_data={"classroom_id":True,"building":True,"floor":True},
                      labels={"capacity":"Room Capacity","utilization_rate":"Utilization Rate %"})
    fig3.update_layout(**LIGHT_LAYOUT, height=340)
    st.plotly_chart(fig3, use_container_width=True)

    section("📋 CLASSROOM DETAIL TABLE")
    cdisp = c[["classroom_id","building","floor","capacity","utilization_rate",
               "equipment_availability","maintenance_status","utilization_band"]].copy()
    cdisp["maintenance_flag"] = cdisp["maintenance_status"].map({"Good":"🟢 Good","Average":"🟡 Average","Poor":"🔴 Poor"})
    cdisp["equip_flag"] = cdisp["equipment_availability"].map({"Yes":"✅ Available","No":"❌ Missing"})
    disp = cdisp[["classroom_id","building","floor","capacity","utilization_rate","utilization_band","equip_flag","maintenance_flag"]].copy()
    disp.columns = ["Room ID","Building","Floor","Capacity","Utilization %","Util Band","Equipment","Maintenance"]
    st.dataframe(disp.sort_values("Utilization %", ascending=False), use_container_width=True, hide_index=True)

    campus_context = {
        "total_classrooms": total_rooms,
        "average_utilization_rate": fmt_pct(avg_util),
        "rooms_with_poor_maintenance": int(poor_maint),
        "rooms_without_equipment": int(no_equip),
        "worst_building_utilization": bld_util.iloc[-1]["building"],
        "best_building_utilization":  bld_util.iloc[0]["building"],
        "average_room_capacity": f"{avg_cap:.0f} seats",
        "sdg_11_sustainability": f"Under-utilized rooms (<40%): {(c['utilization_rate']<40).sum()}",
    }
    render_ai_section(campus_context, "campus_utilization", "campus")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — LIBRARY & DIGITAL ENGAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def page_library(s):
    banner("📚","Library & Digital Engagement",
           "Library usage, online course completion, login frequency and their impact on academic outcomes",
           sdgs=["4","9"])

    avg_lib      = s["library_usage"].mean()
    avg_online   = s["online_course_completion"].mean()
    avg_login    = s["login_frequency"].mean()
    high_lib     = (s["library_usage"]>=15).sum()
    corr_lib_gpa = s["library_usage"].corr(s["current_gpa"])

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(kpi_card("Avg Library Usage", f"{avg_lib:.1f}", "Visits/semester","neutral"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Online Course Avg", fmt_pct(avg_online), "Completion rate","good" if avg_online>=50 else "neutral"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Avg Login Freq.", f"{avg_login:.1f}", "Platform logins/week","neutral"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("High Library Users", fmt_num(high_lib), "≥15 visits","good"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Library-GPA Corr.", f"{corr_lib_gpa:.2f}", "Positive = good","good" if corr_lib_gpa>0.2 else "neutral"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📖 LIBRARY USAGE vs GPA")
        fig = px.scatter(s.sample(min(len(s),1500),random_state=7),
                         x="library_usage", y="current_gpa",
                         color="department", size="online_course_completion", size_max=10,
                         color_discrete_sequence=COLORS, trendline="ols",
                         labels={"library_usage":"Library Visits","current_gpa":"GPA"})
        fig.update_layout(**LIGHT_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("💻 ONLINE COURSE COMPLETION BY DEPARTMENT (SDG 9)")
        dept_oc = s.groupby("department")["online_course_completion"].mean().reset_index()
        dept_oc = dept_oc.sort_values("online_course_completion", ascending=False)
        fig2 = go.Figure(go.Bar(
            x=dept_oc["department"], y=dept_oc["online_course_completion"],
            marker_color=COLORS[:len(dept_oc)],
            text=[f"{v:.1f}%" for v in dept_oc["online_course_completion"]], textposition="outside"
        ))
        fig2.update_layout(**LIGHT_LAYOUT, height=340, showlegend=False, yaxis_title="Avg Online Completion %")
        st.plotly_chart(fig2, use_container_width=True)

    section("🔗 DIGITAL ENGAGEMENT CORRELATION MATRIX")
    corr_cols   = ["library_usage","online_course_completion","login_frequency","study_hours_per_week","current_gpa"]
    corr_matrix = s[corr_cols].corr().round(2)
    fig3 = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns.tolist(), y=corr_matrix.index.tolist(),
        colorscale=[[0,"#ef5350"],[0.5,"white"],[1,"#1565c0"]], zmid=0, zmin=-1, zmax=1,
        text=corr_matrix.values.round(2), texttemplate="%{text}",
        hovertemplate="<b>%{y} vs %{x}</b>: %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="Correlation",font=dict(color="#1a2a4a")),tickfont=dict(color="#1a2a4a"))
    ))
    fig3.update_layout(paper_bgcolor="white", font=dict(color="#1a2a4a"),
                       height=380, margin=dict(l=150,r=40,t=40,b=100))
    st.plotly_chart(fig3, use_container_width=True)

    lib_context = {
        "average_library_usage": f"{avg_lib:.1f} visits/semester",
        "average_online_course_completion": fmt_pct(avg_online),
        "average_login_frequency": f"{avg_login:.1f} logins/week",
        "library_gpa_correlation": f"{corr_lib_gpa:.2f}",
        "high_library_users": int(high_lib),
        "worst_department_online": dept_oc.iloc[-1]["department"],
        "best_department_online":  dept_oc.iloc[0]["department"],
    }
    render_ai_section(lib_context, "library_engagement", "library")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — AI CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
def page_chatbot(s, f, c):
    banner("🤖","Campus AI Chatbot",
           "Ask questions about student performance, faculty, placements, and campus in natural language",
           sdgs=["4","9"])

    total_students  = len(s)
    pass_rate       = s["pass_fail"].mean()*100
    avg_gpa         = s["current_gpa"].mean()
    avg_attendance  = s["attendance_percentage"].mean()
    at_risk         = s["at_risk_student"].sum()
    avg_feedback    = f["student_feedback_score"].mean()
    avg_workload    = f["workload_hours"].mean()
    overloaded_fac  = (f["workload_hours"]>22).sum()
    avg_utilization = c["utilization_rate"].mean()
    poor_rooms      = (c["maintenance_status"]=="Poor").sum()

    dept_pass    = s.groupby("department")["pass_fail"].mean()*100
    worst_dept   = dept_pass.idxmin()
    best_dept    = dept_pass.idxmax()
    place_ready  = s["placement_ready"].sum()
    place_pct    = s["placement_ready"].mean()*100

    dept_fb      = f.groupby("department")["student_feedback_score"].mean()
    worst_fac    = dept_fb.idxmin()
    best_fac     = dept_fb.idxmax()

    low_gpa  = s[s["socio_economic_status"]=="Low"]["current_gpa"].mean()
    high_gpa = s[s["socio_economic_status"]=="High"]["current_gpa"].mean()
    equity_gap = abs(high_gpa - low_gpa)

    def generate_response(msg_raw: str) -> str:
        msg = msg_raw.lower()

        if any(k in msg for k in ["hi","hello","hey","help","what can"]):
            return ("👋 **Hello! I'm your Smart Campus Analytics AI.**\n\n"
                    "This system aligns with **SDG 4** (Quality Education), **SDG 9** (Innovation), "
                    "**SDG 10** (Reduced Inequalities), and **SDG 11** (Sustainable Cities).\n\n"
                    "You can ask me:\n\n"
                    "- 📊 *What is the overall pass rate?*\n"
                    "- ⚠️ *How many students are at risk?*\n"
                    "- 👨‍🏫 *How is faculty performing?*\n"
                    "- 🏫 *What is classroom utilization?*\n"
                    "- 🎓 *How many students are placement ready?*\n"
                    "- ⚖️ *What is the equity situation? (SDG 10)*\n"
                    "- 🌍 *What is the SDG impact?*\n"
                    "- 🧠 *Give me recommendations*")

        if any(k in msg for k in ["sdg","sustainable","goal"]):
            return (f"🌍 **SDG Impact Report:**\n\n"
                    f"**📚 SDG 4 — Quality Education:**\n"
                    f"- Pass Rate: **{pass_rate:.1f}%** (target: 80%+)\n"
                    f"- At-Risk Students: **{at_risk:,}** flagged for intervention\n"
                    f"- Avg GPA: **{avg_gpa:.2f}/10**\n\n"
                    f"**⚖️ SDG 10 — Reduced Inequalities:**\n"
                    f"- GPA Gap (High SES vs Low SES): **{equity_gap:.2f} points**\n"
                    f"- Action: Targeted support programs for Low SES students\n\n"
                    f"**🏗️ SDG 9 — Innovation & Infrastructure:**\n"
                    f"- Research Output (total): **{f['research_output'].sum()}** publications\n"
                    f"- Faculty using digital tools: All {len(f)} faculty\n\n"
                    f"**🌱 SDG 11 — Sustainable Campus:**\n"
                    f"- Avg Room Utilization: **{avg_utilization:.1f}%**\n"
                    f"- Rooms needing maintenance: **{poor_rooms}** (urgent)")

        if any(k in msg for k in ["equity","inequalit","ses","socio","sdg 10","sdg10"]):
            return (f"⚖️ **SDG 10 — Equity Analysis:**\n\n"
                    f"- GPA (Low SES): **{low_gpa:.2f}** | GPA (High SES): **{high_gpa:.2f}**\n"
                    f"- Equity Gap: **{equity_gap:.2f} GPA points**\n"
                    f"- At-Risk among Low SES: **{s[s['socio_economic_status']=='Low']['at_risk_student'].mean()*100:.1f}%**\n\n"
                    f"**Action:** Implement need-based scholarships, tutoring programs, and mentoring for Low SES students.")

        if any(k in msg for k in ["pass rate","pass %","passing"]):
            return (f"✅ **Pass Rate Analysis (SDG 4):**\n\n"
                    f"- Overall Pass Rate: **{pass_rate:.1f}%**\n"
                    f"- Best Department: **{best_dept}** ({dept_pass[best_dept]:.1f}%)\n"
                    f"- Worst Department: **{worst_dept}** ({dept_pass[worst_dept]:.1f}%)\n\n"
                    f"**SDG 4 Action:** Focus remedial support on `{worst_dept}` department.")

        if any(k in msg for k in ["gpa","grade point","academic performance"]):
            return (f"📊 **GPA Summary (SDG 4):**\n\n"
                    f"- Average GPA: **{avg_gpa:.2f}/10**\n"
                    f"- Students with GPA ≥ 8.0: **{(s['current_gpa']>=8).sum():,}**\n"
                    f"- Students with GPA < 5.0: **{(s['current_gpa']<5).sum():,}**\n"
                    f"- Highest GPA Dept: **{s.groupby('department')['current_gpa'].mean().idxmax()}**\n"
                    f"- Lowest GPA Dept: **{s.groupby('department')['current_gpa'].mean().idxmin()}**")

        if any(k in msg for k in ["at risk","atrisk","risk","struggling"]):
            return (f"⚠️ **At-Risk Students (SDG 4 — Intervention):**\n\n"
                    f"- Total At-Risk: **{at_risk:,}** ({at_risk/total_students*100:.1f}% of students)\n"
                    f"- Triggers: Low attendance (<75%), backlogs, GPA < 5.0, High stress\n\n"
                    f"**ML Model predicts these students** using GPA Regression + At-Risk Classifier.\n"
                    f"**Action:** Assign academic mentors and counselors immediately.")

        if any(k in msg for k in ["attendance","absent","presence"]):
            below75 = (s["attendance_percentage"]<75).sum()
            return (f"📅 **Attendance Overview (SDG 4):**\n\n"
                    f"- Average Attendance: **{avg_attendance:.1f}%**\n"
                    f"- Below 75% (detention risk): **{below75:,} students**\n"
                    f"- Below 60% (critical): **{(s['attendance_percentage']<60).sum():,} students**\n\n"
                    f"**Action:** Issue attendance warnings and parent notifications for <60% students.")

        if any(k in msg for k in ["faculty","teacher","professor"]):
            return (f"👨‍🏫 **Faculty Analytics (SDG 9):**\n\n"
                    f"- Average Feedback Score: **{avg_feedback:.2f}/5**\n"
                    f"- Average Workload: **{avg_workload:.1f} hrs/week**\n"
                    f"- Overloaded Faculty (>22hrs): **{overloaded_fac}**\n"
                    f"- Total Research Output: **{f['research_output'].sum()}** publications\n"
                    f"- Best Feedback Dept: **{best_fac}**  |  Worst: **{worst_fac}**\n\n"
                    f"**Action:** Redistribute workload from {overloaded_fac} overloaded faculty.")

        if any(k in msg for k in ["classroom","room","campus","building","utiliz","sdg 11","sdg11"]):
            return (f"🏫 **Campus Utilization (SDG 11 — Sustainability):**\n\n"
                    f"- Average Room Utilization: **{avg_utilization:.1f}%**\n"
                    f"- Rooms with Poor Maintenance: **{poor_rooms}**\n"
                    f"- Rooms without Equipment: **{(c['equipment_availability']=='No').sum()}**\n"
                    f"- Under-utilized rooms (<40%): **{(c['utilization_rate']<40).sum()}**\n\n"
                    f"**SDG 11 Action:** Prioritize maintenance and optimize room scheduling.")

        if any(k in msg for k in ["placement","job","career","employ"]):
            return (f"🎓 **Placement Readiness (SDG 4):**\n\n"
                    f"- Placement Ready: **{place_ready:,}** ({place_pct:.1f}%)\n"
                    f"- Criteria: GPA ≥ 7.0, Zero backlogs, Attendance ≥ 75%\n"
                    f"- Avg Online Course Completion: **{s['online_course_completion'].mean():.1f}%**\n\n"
                    f"**Action:** Targeted coaching for students near the placement threshold.")

        if any(k in msg for k in ["recommend","suggest","improve","action","what should"]):
            recs = []
            if pass_rate < 80: recs.append(f"📚 **Remedial Classes** (SDG 4) — Pass rate {pass_rate:.1f}% below 80% target")
            if avg_attendance < 75: recs.append(f"📅 **Attendance Drive** (SDG 4) — Average {avg_attendance:.1f}% below 75%")
            if at_risk > 1000: recs.append(f"⚠️ **Mentor {at_risk:,} At-Risk Students** (SDG 4) — Immediate counselors needed")
            if overloaded_fac > 5: recs.append(f"👨‍🏫 **Balance Faculty Workload** (SDG 9) — {overloaded_fac} overloaded")
            if poor_rooms > 50: recs.append(f"🏫 **Campus Maintenance** (SDG 11) — {poor_rooms} rooms need urgent repair")
            if equity_gap > 0.5: recs.append(f"⚖️ **Equity Programs** (SDG 10) — {equity_gap:.2f} GPA gap between SES groups")
            if not recs: recs.append("✅ **Campus performing well** — maintain current practices")
            return "🧠 **AI Recommendations (SDG-Aligned):**\n\n" + "\n\n".join(recs)

        if any(k in msg for k in ["summary","overview","report","status","how is"]):
            return (f"📋 **Smart Campus Summary:**\n\n"
                    f"| Metric | Value | SDG |\n|---|---|---|\n"
                    f"| Total Students | {total_students:,} | SDG 4 |\n"
                    f"| Pass Rate | {pass_rate:.1f}% | SDG 4 |\n"
                    f"| Average GPA | {avg_gpa:.2f} | SDG 4 |\n"
                    f"| At-Risk Students | {at_risk:,} | SDG 4 |\n"
                    f"| Avg Attendance | {avg_attendance:.1f}% | SDG 4 |\n"
                    f"| Placement Ready | {place_pct:.1f}% | SDG 4 |\n"
                    f"| SES Equity Gap | {equity_gap:.2f} GPA pts | SDG 10 |\n"
                    f"| Research Output | {f['research_output'].sum()} | SDG 9 |\n"
                    f"| Room Utilization | {avg_utilization:.1f}% | SDG 11 |\n"
                    f"| Overloaded Faculty | {overloaded_fac} | SDG 9 |")

        return (f"🤖 Quick Snapshot — Pass Rate: **{pass_rate:.1f}%** | Avg GPA: **{avg_gpa:.2f}** | At-Risk: **{at_risk:,}**\n\n"
                f"Try: *'Give me recommendations'*, *'What is the SDG impact?'*, or *'How many students are at risk?'*")

    if "campus_chat" not in st.session_state:
        st.session_state.campus_chat = [{"role":"assistant",
            "content":("👋 **Welcome to the Smart Campus Analytics AI Chatbot!**\n\n"
                       "This system supports **SDG 4**, **SDG 9**, **SDG 10**, and **SDG 11**.\n\n"
                       "Try: *'What is the SDG impact?'*, *'How many students are at risk?'* or *'Give me recommendations'*")}]

    for msg in st.session_state.campus_chat:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    st.markdown("**⚡ Quick Questions:**")
    qcols = st.columns(4)
    quick_qs = ["What is the SDG impact?", "How many students are at risk?",
                "Give me recommendations", "Full campus summary"]
    for i, qc in enumerate(qcols):
        with qc:
            if st.button(quick_qs[i], key=f"cq_{i}", use_container_width=True):
                st.session_state.campus_chat.append({"role":"user","content":quick_qs[i]})
                st.session_state.campus_chat.append({"role":"assistant","content":generate_response(quick_qs[i])})
                st.rerun()

    if prompt := st.chat_input("Ask about your campus data… (e.g. 'What is the SDG 10 equity gap?')"):
        st.session_state.campus_chat.append({"role":"user","content":prompt})
        st.session_state.campus_chat.append({"role":"assistant","content":generate_response(prompt)})
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.campus_chat = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 9 — STUDENT PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
def page_prediction():
    banner("🔮","Student Outcome Predictor",
           "Predict GPA, Pass/Fail, and At-Risk status using ML models trained on campus data",
           sdgs=["4","9"])

    models = load_models()
    reg    = models.get("gpa_regressor")
    clf_pf = models.get("pass_fail_classifier")
    clf_rk = models.get("at_risk_model")
    sc_gpa = models.get("scaler_gpa")
    sc_pf  = models.get("scaler_pf")
    sc_rk  = models.get("scaler_risk")
    le_pf_ = models.get("le_pass_fail")
    le_rk_ = models.get("le_risk")
    encs   = models.get("encoders")
    fcols  = models.get("feature_cols")

    required_keys = ["gpa_regressor","pass_fail_classifier","at_risk_model",
                      "scaler_gpa","scaler_pf","scaler_risk",
                      "le_pass_fail","le_risk","encoders","feature_cols"]
    missing = [k for k in required_keys if models.get(k) is None]
    if missing:
        st.error(f"❌ Model file(s) not found in `models/` folder: {', '.join(missing)}\n\nRun the Jupyter notebook first to train and save models.")
        return

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi_card("GPA Predictor","Regressor","Predicts numeric GPA","good"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Pass/Fail","Classifier","Pass or Fail prediction","good"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("At-Risk Model","Classifier","Risk detection (SDG 4)","neutral"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Features", str(len(fcols)), "Training attributes","neutral"), unsafe_allow_html=True)

    # ── MODEL PERFORMANCE (incl. Logistic Regression accuracy) & ML INTERPRETATION ──
    metrics_df = models.get("model_metrics")
    fi_data    = models.get("feature_importance")

    if metrics_df is not None or fi_data is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        section("🧮 MODEL PERFORMANCE & INTERPRETATION")

        mcol1, mcol2 = st.columns(2)

        with mcol1:
            st.markdown('<div style="font-size:13px;font-weight:700;color:#1565c0;margin-bottom:8px;">📊 Pass/Fail Classifier Leaderboard</div>', unsafe_allow_html=True)
            if metrics_df is not None:
                lr_row = metrics_df[metrics_df["Model"].str.contains("Logistic", case=False, na=False)]
                if not lr_row.empty:
                    lr_acc = float(lr_row.iloc[0]["Accuracy"]) * 100
                    st.markdown(kpi_card("Logistic Regression Accuracy", f"{lr_acc:.1f}%",
                                         "On held-out test set", "good" if lr_acc>=75 else "neutral"),
                                unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                display_df = metrics_df.copy()
                for col in ["Accuracy","AUC","CV Acc"]:
                    if col in display_df.columns:
                        display_df[col] = (display_df[col]*100).round(1).astype(str) + "%"
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Model comparison metrics not available. Re-run the notebook's 'Model Metrics Export' cell to generate `models/model_metrics.pkl`.")

        with mcol2:
            st.markdown('<div style="font-size:13px;font-weight:700;color:#1565c0;margin-bottom:8px;">🔍 Feature Importance — What Drives the Prediction</div>', unsafe_allow_html=True)
            if fi_data is not None and "pass_fail" in fi_data:
                fi_series = fi_data["pass_fail"].sort_values(ascending=False).head(10)
                fig_fi = go.Figure(go.Bar(
                    x=fi_series.values[::-1], y=fi_series.index[::-1], orientation="h",
                    marker_color=UNI_BLUE
                ))
                fig_fi.update_layout(**LIGHT_LAYOUT, height=320,
                                     xaxis_title="Importance Score")
                fig_fi.update_layout(margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_fi, use_container_width=True)
                st.caption("Top factors the Pass/Fail model relies on most — useful for explaining *why* a prediction was made, not just what it is.")
            else:
                st.info("Feature importance not available. Re-run the notebook's 'Model Metrics Export' cell to generate `models/feature_importance.pkl`.")

    st.markdown("<br>", unsafe_allow_html=True)
    section("📝 ENTER STUDENT DETAILS")

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown('<div style="font-size:13px;font-weight:700;color:#1565c0;margin-bottom:8px;">👤 Demographics</div>', unsafe_allow_html=True)
        r1c1,r1c2,r1c3,r1c4 = st.columns(4)
        with r1c1: gender       = st.selectbox("Gender",["Female","Male"])
        with r1c2: age          = st.number_input("Age", min_value=17, max_value=30, value=20)
        with r1c3: department   = st.selectbox("Department",["CIVIL","CSE","ECE","EEE","MBA","MCA","MECH"])
        with r1c4: year_of_study= st.selectbox("Year of Study",[1,2,3,4])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:700;color:#1565c0;margin-bottom:8px;">🏠 Background (SDG 10 — Equity Factors)</div>', unsafe_allow_html=True)
        r2c1,r2c2,r2c3 = st.columns(3)
        with r2c1: socio_economic_status = st.selectbox("Socio-Economic Status",["Low","Medium","High"])
        with r2c2: parental_education    = st.selectbox("Parental Education",["No Degree","Diploma","Bachelor","Master","PhD"])
        with r2c3: hostel_status         = st.selectbox("Hostel Status",["Day Scholar","Hosteller"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:700;color:#1565c0;margin-bottom:8px;">📊 Academic Performance</div>', unsafe_allow_html=True)
        r3c1,r3c2,r3c3,r3c4,r3c5 = st.columns(5)
        with r3c1: attendance_percentage  = st.slider("Attendance %",0.0,100.0,75.0,step=0.5)
        with r3c2: internal_marks         = st.slider("Internal Marks",0.0,100.0,65.0,step=0.5)
        with r3c3: assignment_score       = st.slider("Assignment Score",0.0,100.0,70.0,step=0.5)
        with r3c4: lab_performance        = st.slider("Lab Performance",0.0,100.0,70.0,step=0.5)
        with r3c5: previous_sem_gpa       = st.slider("Previous Sem GPA",0.0,10.0,6.5,step=0.1)

        r3c6,r3c7 = st.columns(2)
        with r3c6: backlogs           = st.number_input("Backlogs", min_value=0, max_value=15, value=0)
        with r3c7: discipline_records = st.number_input("Discipline Records", min_value=0, max_value=10, value=0)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:700;color:#1565c0;margin-bottom:8px;">📚 Engagement & Study Habits (SDG 4)</div>', unsafe_allow_html=True)
        r4c1,r4c2,r4c3 = st.columns(3)
        with r4c1: study_hours_per_week      = st.slider("Study Hours / Week",0.0,50.0,15.0,step=0.5)
        with r4c2: extracurricular_score     = st.slider("Extracurricular Score",0.0,100.0,50.0,step=0.5)
        with r4c3: login_frequency           = st.slider("Login Frequency",0.0,100.0,45.0,step=0.5)

        r4c4,r4c5 = st.columns(2)
        with r4c4: library_usage             = st.slider("Library Usage (visits)",0.0,50.0,10.0,step=0.5)
        with r4c5: online_course_completion  = st.slider("Online Course Completion %",0.0,100.0,50.0,step=0.5)

        r4c6,r4c7,_ = st.columns(3)
        with r4c6: participation = st.selectbox("Class Participation",["Low","Medium","High"])
        with r4c7: stress_level  = st.selectbox("Stress Level",["Low","Medium","High"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Run Prediction", use_container_width=True, type="primary")

    if submitted:
        raw_input = {
            "gender":gender,"age":age,"department":department,
            "year_of_study":year_of_study,"socio_economic_status":socio_economic_status,
            "parental_education":parental_education,"hostel_status":hostel_status,
            "attendance_percentage":attendance_percentage,"internal_marks":internal_marks,
            "assignment_score":assignment_score,"lab_performance":lab_performance,
            "previous_sem_gpa":previous_sem_gpa,"backlogs":backlogs,
            "study_hours_per_week":study_hours_per_week,"participation":participation,
            "extracurricular_score":extracurricular_score,"login_frequency":login_frequency,
            "library_usage":library_usage,"online_course_completion":online_course_completion,
            "discipline_records":discipline_records,"stress_level":stress_level,
        }

        warnings.filterwarnings('ignore')
        input_df = encode_input(raw_input, encs, fcols)

        predicted_gpa = float(reg.predict(sc_gpa.transform(input_df))[0])
        predicted_gpa = max(0.0, min(10.0, predicted_gpa))

        pf_raw    = clf_pf.predict(sc_pf.transform(input_df))[0]
        pf_label  = le_pf_.inverse_transform([pf_raw])[0]
        pf_proba  = clf_pf.predict_proba(sc_pf.transform(input_df))[0]
        pass_prob = float(pf_proba[1])*100
        fail_prob = float(pf_proba[0])*100

        rk_raw        = clf_rk.predict(sc_rk.transform(input_df))[0]
        predicted_risk= int(le_rk_.inverse_transform([rk_raw])[0])
        risk_proba    = clf_rk.predict_proba(sc_rk.transform(input_df))[0]
        safe_prob     = float(risk_proba[0])*100
        risk_prob_val = float(risk_proba[1])*100

        gpa_label = ("Excellent" if predicted_gpa>=8.5 else
                     "Good" if predicted_gpa>=7.0 else
                     "Average" if predicted_gpa>=5.5 else "Poor")

        # ── Clear prediction AI insights so they regenerate fresh ──
        for _k in ["active_card_prediction", "card_content_prediction",
                   "translation_prediction", "lang_prediction"]:
            st.session_state.pop(_k, None)

        st.session_state["pred_context"] = {
            "predicted_gpa": f"{predicted_gpa:.2f}",
            "gpa_performance_band": gpa_label,
            "at_risk_status": "At Risk" if predicted_risk==1 else "Safe",
            "at_risk_probability": f"{risk_prob_val:.1f}%",
            "safe_probability": f"{safe_prob:.1f}%",
            "attendance_percentage": fmt_pct(attendance_percentage),
            "backlogs": backlogs,
            "previous_sem_gpa": previous_sem_gpa,
            "study_hours_per_week": study_hours_per_week,
            "stress_level": stress_level,
            "socio_economic_status": socio_economic_status,
            "department": department,
            "sdg_4_intervention": "Student identified for targeted academic support" if predicted_risk==1 else "Student on track with SDG 4 goals",
            "sdg_10_equity": f"SES: {socio_economic_status} — requires equity-aware support" if socio_economic_status=="Low" else f"SES: {socio_economic_status}",
        }

        # ── Store everything needed to redraw the results panel below,
        #    so it survives reruns triggered by the AI-insight buttons ──
        st.session_state["pred_full"] = {
            "predicted_gpa": predicted_gpa,
            "pf_label": pf_label,
            "pass_prob": pass_prob,
            "fail_prob": fail_prob,
            "predicted_risk": predicted_risk,
            "risk_prob_val": risk_prob_val,
            "safe_prob": safe_prob,
            "gpa_label": gpa_label,
            "attendance_percentage": attendance_percentage,
            "internal_marks": internal_marks,
            "assignment_score": assignment_score,
            "lab_performance": lab_performance,
            "previous_sem_gpa": previous_sem_gpa,
            "study_hours_per_week": study_hours_per_week,
            "extracurricular_score": extracurricular_score,
            "online_course_completion": online_course_completion,
            "backlogs": backlogs,
            "stress_level": stress_level,
            "socio_economic_status": socio_economic_status,
        }

    # ── Render results from session_state whenever we have them —
    #    on the submit run AND on every later rerun (e.g. AI insight clicks) ──
    if "pred_full" in st.session_state:
        pf = st.session_state["pred_full"]
        predicted_gpa          = pf["predicted_gpa"]
        pf_label                = pf["pf_label"]
        pass_prob               = pf["pass_prob"]
        fail_prob               = pf["fail_prob"]
        predicted_risk          = pf["predicted_risk"]
        risk_prob_val           = pf["risk_prob_val"]
        safe_prob               = pf["safe_prob"]
        gpa_label               = pf["gpa_label"]
        attendance_percentage   = pf["attendance_percentage"]
        internal_marks          = pf["internal_marks"]
        assignment_score        = pf["assignment_score"]
        lab_performance         = pf["lab_performance"]
        previous_sem_gpa        = pf["previous_sem_gpa"]
        study_hours_per_week    = pf["study_hours_per_week"]
        extracurricular_score   = pf["extracurricular_score"]
        online_course_completion= pf["online_course_completion"]
        backlogs                = pf["backlogs"]
        stress_level            = pf["stress_level"]
        socio_economic_status   = pf["socio_economic_status"]

        st.markdown("<br>", unsafe_allow_html=True)
        section("🎯 PREDICTION RESULTS")

        gpa_class  = "good" if predicted_gpa>=7.0 else ("neutral" if predicted_gpa>=5.5 else "bad")
        risk_class = "bad" if predicted_risk==1 else "good"
        risk_lbl   = "⚠️ AT RISK" if predicted_risk==1 else "✅ SAFE"
        pf_display = "✅ PASS" if int(pf_label)==1 else "❌ FAIL"
        pf_class   = "good" if int(pf_label)==1 else "bad"

        r1,r2,r3,r4,r5 = st.columns(5)
        with r1: st.markdown(kpi_card("Predicted GPA", f"{predicted_gpa:.2f}", f"{gpa_label} Performance", gpa_class), unsafe_allow_html=True)
        with r2: st.markdown(kpi_card("Pass / Fail", pf_display, f"Confidence: {max(pass_prob,fail_prob):.1f}%", pf_class), unsafe_allow_html=True)
        with r3: st.markdown(kpi_card("Risk Status", risk_lbl, f"Confidence: {max(safe_prob,risk_prob_val):.1f}%", risk_class), unsafe_allow_html=True)
        with r4: st.markdown(kpi_card("Safe Probability", f"{safe_prob:.1f}%", "Will not be at risk","good" if safe_prob>70 else "neutral"), unsafe_allow_html=True)
        with r5: st.markdown(kpi_card("At-Risk Probability", f"{risk_prob_val:.1f}%", "Risk of failing/dropout","bad" if risk_prob_val>30 else "good"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            section("📊 PREDICTED GPA GAUGE")
            gpa_color = "#2e7d32" if predicted_gpa>=7 else ("#e65100" if predicted_gpa>=5.5 else "#c62828")
            fig_gpa = go.Figure(go.Indicator(
                mode="gauge+number", value=predicted_gpa,
                number={"suffix":"/10","font":{"color":gpa_color,"size":36}},
                title={"text":"Predicted GPA","font":{"color":"#1a2a4a","size":14}},
                gauge={"axis":{"range":[0,10]},"bar":{"color":gpa_color},"bgcolor":"#f8fbff",
                       "steps":[{"range":[0,5.5],"color":"#ffebee"},{"range":[5.5,7.0],"color":"#fff8e1"},{"range":[7.0,10],"color":"#e8f5e9"}],
                       "threshold":{"line":{"color":"#1565c0","width":3},"thickness":0.8,"value":7.0}}
            ))
            fig_gpa.update_layout(paper_bgcolor="white",font_color="#1a2a4a",height=300,margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig_gpa, use_container_width=True)

        with col2:
            section("⚠️ AT-RISK PROBABILITY GAUGE")
            risk_color = "#c62828" if risk_prob_val>50 else ("#e65100" if risk_prob_val>25 else "#2e7d32")
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=risk_prob_val,
                number={"suffix":"%","font":{"color":risk_color,"size":36}},
                delta={"reference":30,"valueformat":".1f","suffix":"%"},
                title={"text":"At-Risk Probability","font":{"color":"#1a2a4a","size":14}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":risk_color},"bgcolor":"#f8fbff",
                       "steps":[{"range":[0,25],"color":"#e8f5e9"},{"range":[25,50],"color":"#fff8e1"},{"range":[50,100],"color":"#ffebee"}],
                       "threshold":{"line":{"color":"#c62828","width":3},"thickness":0.8,"value":50}}
            ))
            fig_risk.update_layout(paper_bgcolor="white",font_color="#1a2a4a",height=300,margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig_risk, use_container_width=True)

        section("📡 INPUT PROFILE RADAR")
        radar_vals = [attendance_percentage,internal_marks,assignment_score,
                      lab_performance,previous_sem_gpa*10,study_hours_per_week*2,
                      extracurricular_score,online_course_completion]
        radar_cats = ["Attendance","Internal Marks","Assignment","Lab","Prev GPA×10",
                      "Study Hrs×2","Extra-Curr","Online Courses"]
        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals+[radar_vals[0]], theta=radar_cats+[radar_cats[0]],
            fill='toself', fillcolor='rgba(21,101,192,0.15)',
            line=dict(color=UNI_BLUE,width=2), marker=dict(color=UNI_BLUE,size=7)
        ))
        fig_radar.update_layout(paper_bgcolor="white",
            polar=dict(bgcolor="#f8fbff",
                       radialaxis=dict(visible=True,range=[0,100],tickfont=dict(color="#5c7a9e")),
                       angularaxis=dict(tickfont=dict(color="#1a2a4a",size=12))),
            height=380, margin=dict(l=40,r=40,t=40,b=40), font=dict(color="#1a2a4a"))
        st.plotly_chart(fig_radar, use_container_width=True)

        section("💡 ACTIONABLE INSIGHTS (SDG-Aligned)")
        insights = []
        if attendance_percentage < 75:
            insights.append(("bad", f"📅 **Low Attendance ({attendance_percentage:.1f}%)** (SDG 4) — Below 75% threshold. Compulsory counselling recommended."))
        if backlogs > 0:
            insights.append(("bad", f"📚 **{backlogs} Backlog(s)** (SDG 4) — Assign subject-specific tutoring and create a clearance plan."))
        if previous_sem_gpa < 5.5:
            insights.append(("bad", f"📉 **Low Previous GPA ({previous_sem_gpa:.1f})** (SDG 4) — Remedial academic support needed urgently."))
        if study_hours_per_week < 10:
            insights.append(("neutral", f"⏰ **Low Study Hours ({study_hours_per_week:.1f}/week)** (SDG 4) — Encourage structured study; target 15+ hours."))
        if stress_level == "High":
            insights.append(("neutral", "🧠 **High Stress Level** (SDG 4) — Refer to campus counsellor for mental health support."))
        if socio_economic_status == "Low":
            insights.append(("neutral", "⚖️ **Low SES Background** (SDG 10) — Eligible for need-based scholarship and mentoring programs."))
        if predicted_gpa >= 8.0:
            insights.append(("good", f"🌟 **High Achiever (GPA {predicted_gpa:.2f})** (SDG 4) — Consider for merit scholarship and leadership programmes."))
        if predicted_risk == 0 and attendance_percentage >= 85:
            insights.append(("good", "✅ **Strong Academic Profile** (SDG 4) — Student on track; maintain current engagement level."))
        if not insights:
            insights.append(("blue", "📋 Student profile looks average. Monitor regularly and provide encouragement."))

        for kind, msg in insights:
            st.markdown(f'<div class="alert-{kind}">{msg}</div>', unsafe_allow_html=True)

    if "pred_full" not in st.session_state:
        st.markdown("""
        <div style="background:white;border:2px dashed #90b4e8;border-radius:14px;
                    padding:40px;text-align:center;margin-top:20px;">
            <div style="font-size:48px;margin-bottom:12px;">🔮</div>
            <div style="font-size:18px;font-weight:700;color:#0a2463;margin-bottom:8px;">Ready to Predict</div>
            <div style="font-size:13px;color:#5c7a9e;">Fill in the student details above and click <b>Run Prediction</b><br>
            to get GPA prediction, Pass/Fail, and At-Risk classification (SDG 4).</div>
        </div>""", unsafe_allow_html=True)

    if "pred_context" in st.session_state:
        render_ai_section(st.session_state["pred_context"], "student_prediction", "prediction")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — also returns a filter-signature so we can clear AI cache on change
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar(s, f, c):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:14px 0 10px 0;'>
            <div style='font-size:42px;'>🎓</div>
            <div style='font-size:15px;font-weight:700;color:white;font-family:Playfair Display,serif;'>Smart Campus Analytics</div>
            <div style='font-size:10px;color:#90caf9;margin-top:2px;'>AI-Powered | IEEE Research Project</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center;padding:4px 0 12px 0;'>
            <span style='background:rgba(255,255,255,0.15);border-radius:20px;padding:3px 8px;font-size:10px;color:#e3f2fd;margin:2px;display:inline-block;'>🌍 SDG 4</span>
            <span style='background:rgba(255,255,255,0.15);border-radius:20px;padding:3px 8px;font-size:10px;color:#e3f2fd;margin:2px;display:inline-block;'>SDG 9</span>
            <span style='background:rgba(255,255,255,0.15);border-radius:20px;padding:3px 8px;font-size:10px;color:#e3f2fd;margin:2px;display:inline-block;'>SDG 10</span>
            <span style='background:rgba(255,255,255,0.15);border-radius:20px;padding:3px 8px;font-size:10px;color:#e3f2fd;margin:2px;display:inline-block;'>SDG 11</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        page = st.selectbox("📌 Navigation", [
            "📌 About This Project",
            "🎓 Executive Dashboard",
            "📚 Student Performance",
            "📅 Attendance Analytics",
            "👨‍🏫 Faculty Analytics",
            "🎯 Placement Tracker",
            "🏫 Campus & Classroom",
            "📖 Library & Digital",
            "🔮 Student Predictor",
            "🤖 AI Chatbot",
            "🗄️ Database Status",
        ])

        st.markdown("---")
        st.markdown('<div style="font-size:11px;color:black;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🎛️ FILTERS</div>', unsafe_allow_html=True)

        departments = sorted(s["department"].unique().tolist())
        sel_depts   = st.multiselect("🏛️ Department", departments)
        years       = sorted(s["year_of_study"].unique().tolist())
        sel_years   = st.multiselect("📅 Year of Study", years)
        genders     = s["gender"].unique().tolist()
        sel_genders = st.multiselect("👤 Gender", genders)

        st.markdown("---")
        st.markdown(f"""
        <div style='font-size:11px;color:#90caf9;line-height:2;'>
            👥 {len(s):,} Students<br>
            👨‍🏫 {len(f)} Faculty<br>
            🏫 {len(c)} Classrooms<br>
            🏛️ {s['department'].nunique()} Departments
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if st.session_state.get("db_connected"):
            cfg = st.session_state.get("db_config", {})
            st.markdown(f"""<div style='font-size:11px;color:#a5d6a7;'>🟢 <b>MySQL Connected</b><br><span style='color:#90caf9;font-size:10px;'>{cfg.get('database','smart_campus')}@{cfg.get('host','localhost')}</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='font-size:11px;color:#ef9a9a;'>🔴 <b>DB Offline</b></div>""", unsafe_allow_html=True)

    # Build a signature string of the current filter state
    filter_sig = f"{sorted(sel_depts)}|{sorted(sel_years)}|{sorted(sel_genders)}"
    return page, sel_depts, sel_years, sel_genders, filter_sig


# ─────────────────────────────────────────────────────────────────────────────
# AI SESSION STATE CLEANER — wipe all card caches when filters change
# ─────────────────────────────────────────────────────────────────────────────
def clear_ai_cache_if_filters_changed(filter_sig: str):
    prev_sig_key = "_prev_filter_sig"
    prev_sig = st.session_state.get(prev_sig_key, None)

    if prev_sig is not None and prev_sig != filter_sig:
        # Filters changed — clear every AI section's cached content
        ai_keys_to_clear = [k for k in st.session_state.keys()
                            if k.startswith(("active_card_", "card_content_", "translation_", "lang_"))]
        for k in ai_keys_to_clear:
            del st.session_state[k]

    # Save the current signature
    st.session_state[prev_sig_key] = filter_sig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — DATABASE STATUS (Backend Connectivity)
# ─────────────────────────────────────────────────────────────────────────────
def page_database():
    banner("🗄️", "Database Status", "Backend MySQL connectivity — live sync of campus datasets", sdgs=["9", "11"])

    db_ok  = st.session_state.get("db_connected", False)
    db_err = st.session_state.get("db_error", "")
    cfg    = st.session_state.get("db_config", {"host": "localhost", "database": "smart_campus", "user": "root", "port": 3306})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        status_label = "🟢 Connected" if db_ok else "🔴 Disconnected"
        st.markdown(kpi_card("DB Status", status_label, "MySQL backend", "good" if db_ok else "bad"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Host", cfg.get("host", "localhost"), f"Port {cfg.get('port', 3306)}", "neutral"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Database", cfg.get("database", "smart_campus"), f"User: {cfg.get('user','root')}", "neutral"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Engine", "MySQL", "mysql-connector-python", "neutral"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not MYSQL_AVAILABLE:
        st.error("❌ mysql-connector-python is not installed.")
        st.code("pip install mysql-connector-python", language="bash")
        return

    if db_ok:
        section("📊 TABLE SUMMARY")
        try:
            conn = mysql.connector.connect(**cfg)
            tables = ["students", "faculty", "classrooms", "sync_log"]
            rows = []
            for t in tables:
                try:
                    cur = conn.cursor()
                    cur.execute(f"SELECT COUNT(*) FROM `{t}`")
                    count = cur.fetchone()[0]
                    rows.append({"Table": t, "Rows": int(count), "Status": "✅ OK"})
                    cur.close()
                except Exception:
                    rows.append({"Table": t, "Rows": 0, "Status": "❌ Missing"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            section("🕐 SYNC LOG (Last 5 Syncs)")
            try:
                cur = conn.cursor()
                cur.execute("SELECT * FROM sync_log ORDER BY synced_at DESC LIMIT 5")
                log_rows = cur.fetchall()
                cur.close()
                if log_rows:
                    log_df = pd.DataFrame(log_rows, columns=["Synced At", "Student Rows", "Faculty Rows", "Classroom Rows"])
                    st.dataframe(log_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Sync log is empty.")
            except Exception:
                st.info("Sync log not yet available.")

            section("🔍 LIVE SQL QUERY")
            st.markdown("<p style='font-size:13px;color:#1a2a4a;'>Run a custom SELECT query on the MySQL campus database:</p>", unsafe_allow_html=True)
            default_query = "SELECT department, COUNT(*) as total, AVG(current_gpa) as avg_gpa FROM students GROUP BY department ORDER BY avg_gpa DESC"
            user_query = st.text_area("SQL Query", value=default_query, height=80)
            if st.button("▶ Run Query", type="primary"):
                try:
                    cur = conn.cursor()
                    cur.execute(user_query)
                    result_rows = cur.fetchall()
                    col_names   = [desc[0] for desc in cur.description]
                    cur.close()
                    result_df = pd.DataFrame(result_rows, columns=col_names)
                    st.success(f"✅ Query returned {len(result_df)} rows")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                except Exception as qe:
                    st.error(f"❌ Query error: {qe}")
            conn.close()
        except Exception as e:
            st.error(f"Could not connect to MySQL: {e}")
    else:
        st.error(f"❌ MySQL not connected. Error: {db_err}")
        st.markdown("""
        <div style='background:#fff3e0;border-radius:8px;padding:14px;font-size:13px;color:#1a2a4a;line-height:2;'>
        <b>🔧 To fix this, check the following:</b><br>
        &nbsp;&nbsp;1. MySQL server is running on your machine<br>
        &nbsp;&nbsp;2. Database <code>smart_campus</code> exists → <code>CREATE DATABASE smart_campus;</code><br>
        &nbsp;&nbsp;3. Correct <b>host / user / password</b> set as environment variables (see <code>.env.example</code>)<br>
        &nbsp;&nbsp;4. Run: <code>pip install mysql-connector-python</code>
        </div>""", unsafe_allow_html=True)

    section("ℹ️ DATABASE ARCHITECTURE")
    st.markdown("""
    <div style='background:#e3f2fd;border-radius:8px;padding:16px;font-size:13px;color:#1a2a4a;line-height:1.8;'>
    <b>📁 MySQL Tables:</b><br>
    &nbsp;&nbsp;• <code>students</code> — All student academic records (500+ rows, 25 columns)<br>
    &nbsp;&nbsp;• <code>faculty</code> — Faculty workload, feedback, research data<br>
    &nbsp;&nbsp;• <code>classrooms</code> — Room utilization and maintenance status<br>
    &nbsp;&nbsp;• <code>sync_log</code> — Tracks every data sync with timestamp<br><br>
    <b>🔄 Sync Strategy:</b> CSV → MySQL on every app startup (drop & recreate)<br>
    <b>⚙️ Config Location:</b> credentials loaded via environment variables / <code>.env</code> / <code>st.secrets</code> — never hardcoded (see <code>get_secret()</code> near the top of app.py)<br>
    <b>🌍 SDG 9:</b> Digital infrastructure for data-driven campus management
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("🔄 Loading campus data…"):
        student, faculty, classroom = load_data()

    page, sel_depts, sel_years, sel_genders, filter_sig = build_sidebar(student, faculty, classroom)

    # ── Auto-clear AI insights when sidebar filters change ──────────────────
    clear_ai_cache_if_filters_changed(filter_sig)

    s = student.copy()
    if sel_depts:   s = s[s["department"].isin(sel_depts)]
    if sel_years:   s = s[s["year_of_study"].isin(sel_years)]
    if sel_genders: s = s[s["gender"].isin(sel_genders)]

    f = faculty.copy()
    if sel_depts: f = f[f["department"].isin(sel_depts)]

    c = classroom.copy()

    if s.empty:
        st.warning("⚠️ No data matches selected filters. Please adjust the sidebar filters.")
        return

    if   page == "📌 About This Project":  page_about()
    elif page == "🎓 Executive Dashboard": page_executive(s, f, c)
    elif page == "📚 Student Performance": page_student(s)
    elif page == "📅 Attendance Analytics":page_attendance(s)
    elif page == "👨‍🏫 Faculty Analytics":   page_faculty(f)
    elif page == "🎯 Placement Tracker":   page_placement(s)
    elif page == "🏫 Campus & Classroom":  page_campus(c)
    elif page == "📖 Library & Digital":   page_library(s)
    elif page == "🔮 Student Predictor":   page_prediction()
    elif page == "🤖 AI Chatbot":          page_chatbot(s, f, c)
    elif page == "🗄️ Database Status":    page_database()


if __name__ == "__main__":
    main()
