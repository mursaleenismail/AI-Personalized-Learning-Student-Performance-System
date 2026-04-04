"""
╔══════════════════════════════════════════════════════════════════════╗
║        AI Personalized Learning & Student Performance System         ║
║   Stack: Streamlit · spaCy · RandomForest · Groq · Plotly · OpenCV  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import re
from datetime import date, timedelta
from groq import Groq

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduAI · ذہین نظام تعلیم",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* Hide default Streamlit header */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a14;
    border-right: 1px solid #1e1e30;
}
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #fff !important; }

/* Main background */
.main { background: #0f0f1a; }
.block-container { padding: 1.5rem 2rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #13132a 0%, #1a1a30 100%);
    border: 1px solid #2a2a45;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 600; margin: 4px 0; }
.metric-lbl { font-size: 0.75rem; color: #7878a0; text-transform: uppercase; letter-spacing: 1px; }
.metric-ur  { font-size: 0.8rem; color: #5a5a80; font-family: serif; margin-top: 3px; }

/* Section headers */
.sec-head {
    font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;
    color: #5a5a90; margin: 1.5rem 0 0.6rem; border-top: 1px solid #1e1e32;
    padding-top: 1rem;
}

/* Risk badge */
.risk-high   { background:#3d1515; color:#ff6b6b; border:1px solid #5a2020; border-radius:8px; padding:4px 12px; font-size:.8rem; font-weight:600; }
.risk-medium { background:#3d2f10; color:#ffb347; border:1px solid #5a4420; border-radius:8px; padding:4px 12px; font-size:.8rem; font-weight:600; }
.risk-low    { background:#0d3d20; color:#4ade80; border:1px solid #155a30; border-radius:8px; padding:4px 12px; font-size:.8rem; font-weight:600; }

/* Info boxes */
.info-box {
    background:#13132a; border:1px solid #2a2a45; border-radius:10px;
    padding:1rem 1.2rem; margin:0.5rem 0;
}
.chip {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:.72rem; font-weight:500; margin:2px;
}

/* Plan output */
.plan-block {
    background:#0d1a0d; border:1px solid #1a3a1a; border-radius:10px;
    padding:1.2rem; white-space:pre-wrap; line-height:1.9;
    font-size:.88rem; color:#c8e6c8;
}
.plan-urdu {
    background:#1a0d1a; border:1px solid #3a1a3a; border-radius:10px;
    padding:1.2rem; white-space:pre-wrap; line-height:2.4; direction:rtl;
    text-align:right; font-family:serif; font-size:1rem; color:#e6c8e6;
}

/* Divider */
.divider { border:none; border-top:1px solid #1e1e32; margin:1rem 0; }

/* Override Streamlit elements */
.stButton>button {
    background:#1a1a30; border:1px solid #3a3a60; color:#c8c8e8;
    border-radius:8px; font-family:'IBM Plex Sans',sans-serif; font-size:.85rem;
    padding:.4rem 1.1rem;
}
.stButton>button:hover { background:#2a2a45; border-color:#5a5a90; }
div[data-testid="stMetricValue"] { color:#fff !important; }
.stTabs [data-baseweb="tab-list"] { background:#0a0a14; border-bottom:1px solid #1e1e30; }
.stTabs [data-baseweb="tab"] { color:#7878a0; }
.stTabs [aria-selected="true"] { color:#c8c8ff !important; border-bottom:2px solid #7878ff; }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SUBJECTS = ["Mathematics", "Physics", "Chemistry", "English", "Urdu", "Computer Science"]
SUBJ_UR  = ["ریاضی", "فزکس", "کیمسٹری", "انگریزی", "اردو", "کمپیوٹر"]
WEEKS    = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]

PLOT_COLORS = ["#7878ff", "#ff7878", "#78ffb4", "#ffb347", "#ff78d4", "#78d4ff"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color="#aaaacc", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9999cc")),
    xaxis=dict(gridcolor="#1e1e32", linecolor="#1e1e32", tickfont=dict(color="#7878a0")),
    yaxis=dict(gridcolor="#1e1e32", linecolor="#1e1e32", tickfont=dict(color="#7878a0")),
)


# ─── SESSION STATE ────────────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        grades = {s: [40 + i*5 + j*2 for j in range(5)] for i, s in enumerate(SUBJECTS)},
        quiz   = {s: 35 + i*8 for i, s in enumerate(SUBJECTS)},
        attendance = 74,
        assignments= 60,
        prev_result= "Borderline Pass",
        student_name="Ahmed Hassan",
        student_class="Class 10",
        city="Karachi",
        feedback="",
        plan_en="",
        plan_ur="",
        nlp_result=None,
        risk_score=0.67,
        tree_scores=[],
        groq_key="",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
ss = st.session_state


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def avg(lst): return sum(lst) / len(lst) if lst else 0
def grade_avg(): return avg([avg(ss.grades[s]) for s in SUBJECTS])
def quiz_avg():  return avg([ss.quiz[s] for s in SUBJECTS])

def risk_color(r):
    if r >= 0.60: return "#ff6b6b", "risk-high",   "High Risk",    "زیادہ خطرہ"
    if r >= 0.35: return "#ffb347", "risk-medium",  "Medium Risk",  "درمیانہ خطرہ"
    return          "#4ade80", "risk-low",    "Low Risk",     "کم خطرہ"

def subj_color(i): return PLOT_COLORS[i % len(PLOT_COLORS)]


# ─── RANDOM FOREST MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def build_rf():
    """Train a RF on synthetic Pakistani school data."""
    rng = np.random.RandomState(42)
    n = 800

    ga   = rng.uniform(20, 95, n)
    qa   = rng.uniform(15, 95, n)
    att  = rng.uniform(40, 100, n)
    asgn = rng.uniform(20, 100, n)
    prev = rng.choice([0, 1, 2], n, p=[0.6, 0.25, 0.15])   # 0=pass,1=border,2=fail
    trend= rng.uniform(-15, 10, n)

    risk = (
        (ga   < 40) * 0.30 +
        (qa   < 40) * 0.20 +
        (att  < 70) * 0.25 +
        (asgn < 60) * 0.15 +
        (prev == 2) * 0.20 +
        (trend < -5)* 0.10 +
        rng.normal(0, 0.05, n)
    )
    y = (risk > 0.45).astype(int)

    X = np.column_stack([ga, qa, att, asgn, prev, trend])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=42)
    clf.fit(Xs, y)
    return clf, scaler

RF_CLF, RF_SCALER = build_rf()

def compute_risk():
    ga    = grade_avg()
    qa    = quiz_avg()
    att   = ss.attendance
    asgn  = ss.assignments
    prev  = {"Pass": 0, "Borderline Pass": 1, "Fail": 2}[ss.prev_result]
    grads = [ss.grades[s] for s in SUBJECTS]
    trend = avg([avg(g[-2:]) - avg(g[:2]) for g in grads])

    X     = np.array([[ga, qa, att, asgn, prev, trend]])
    Xs    = RF_SCALER.transform(X)

    # Per-tree probabilities
    tree_probs = [t.predict_proba(Xs)[0][1] for t in RF_CLF.estimators_]
    ss.tree_scores = tree_probs
    ss.risk_score  = float(np.mean(tree_probs))
    return ss.risk_score


# ─── spaCy NLP ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

NLP = load_nlp()

STRESS_KW     = ["difficult", "hard", "confusing", "struggle", "stressed", "demotivated",
                 "lost", "fail", "weak", "behind", "cannot", "don't understand",
                 "مشکل", "سمجھ", "پریشان", "کمزور", "ڈر"]
POSITIVE_KW   = ["good", "understand", "enjoy", "like", "easy", "comfortable",
                 "اچھا", "سمجھتا", "پسند"]
SUBJ_MENTIONS = {s.lower(): s for s in SUBJECTS}
SUBJ_MENTIONS.update({"maths": "Mathematics", "math": "Mathematics",
                       "bio": "Biology", "chem": "Chemistry",
                       "ریاضی": "Mathematics", "فزکس": "Physics",
                       "کیمسٹری": "Chemistry", "کمپیوٹر": "Computer Science"})

def analyze_feedback(text: str):
    if not text.strip():
        return None

    text_l = text.lower()
    stress_hits  = [k for k in STRESS_KW   if k in text_l]
    positive_hits= [k for k in POSITIVE_KW if k in text_l]
    weak_subj    = [SUBJ_MENTIONS[k] for k in SUBJ_MENTIONS if k in text_l]

    sentiment = "Negative" if len(stress_hits) > len(positive_hits) else "Positive"

    entities, key_phrases = [], []
    if NLP:
        doc = NLP(text)
        entities    = [(e.text, e.label_) for e in doc.ents]
        key_phrases = list({chunk.text.lower() for chunk in doc.noun_chunks
                            if len(chunk.text) > 3})[:8]

    return dict(
        sentiment    = sentiment,
        stress_words = stress_hits,
        positive_words=positive_hits,
        weak_subjects= list(set(weak_subj)),
        entities     = entities,
        key_phrases  = key_phrases,
        urdu_detected= any(ord(c) > 0x0600 for c in text),
    )


# ─── GROQ AI PLAN ─────────────────────────────────────────────────────────────
def generate_plan(lang="en"):
    key = ss.groq_key.strip()
    if not key:
        return None, "⚠ Please enter your Groq API key in the sidebar."

    ga   = grade_avg()
    qa   = quiz_avg()
    weak = sorted(SUBJECTS, key=lambda s: avg(ss.grades[s]))[:3]
    risk = int(ss.risk_score * 100)

    if lang == "en":
        prompt = f"""You are an expert Pakistani school teacher and educational AI assistant.

Student Profile:
- Name: {ss.student_name} | Class: {ss.student_class} | City: {ss.city}
- Grade Average: {ga:.1f}% | Quiz Average: {qa:.1f}%
- Attendance: {ss.attendance}% | Assignment Completion: {ss.assignments}%
- Dropout Risk: {risk}% (HIGH — urgent intervention needed)
- Weakest Subjects: {', '.join(weak)}

Create a detailed 4-week personalized study plan. Include:
1. Weekly breakdown with daily schedule (1-4 hours)
2. Specific topics to focus on per subject
3. Study strategies for each weak subject
4. Practice exercises and resources
5. Motivational milestones and rewards
6. Tips to improve attendance and assignment completion

Format clearly with Week headers, Day breakdowns, and bullet points.
Be empathetic, encouraging, and culturally sensitive to Pakistani school context."""

    else:
        prompt = f"""آپ ایک ماہر پاکستانی استاد اور تعلیمی AI ہیں۔

طالب علم کی معلومات:
- نام: {ss.student_name} | جماعت: {ss.student_class} | شہر: {ss.city}
- اوسط نمبر: {ga:.1f}% | کوئز اوسط: {qa:.1f}%
- حاضری: {ss.attendance}% | تفویض مکمل: {ss.assignments}%
- ناکامی کا خطرہ: {risk}% (زیادہ — فوری توجہ ضروری)
- کمزور مضامین: {', '.join([SUBJ_UR[SUBJECTS.index(s)] for s in weak if s in SUBJECTS])}

4 ہفتوں کا مکمل مطالعہ منصوبہ بنائیں۔ شامل کریں:
1. ہر ہفتے کا روز مرہ شیڈول
2. کمزور مضامین کے لیے خاص حکمت عملی
3. روزانہ کی مشق اور سرگرمیاں
4. حوصلہ افزائی کے مراحل
5. حاضری اور کام بہتر کرنے کے طریقے

اردو میں واضح اور آسان زبان میں لکھیں۔ ہفتہ وار سرخیاں اور نکات استعمال کریں۔"""

    try:
        client = Groq(api_key=key)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1800,
            temperature=0.7,
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, f"Groq error: {e}"


# ─── OpenCV Processing ────────────────────────────────────────────────────────
def process_image(img_bytes, mode="thresh"):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not decode image."

    img = cv2.resize(img, (600, int(img.shape[0] * 600 / img.shape[1])))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == "gray":
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif mode == "thresh":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    elif mode == "edges":
        edges = cv2.Canny(gray, 50, 150)
        out = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif mode == "boxes":
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = img.copy()
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if 30 < w < 300 and 15 < h < 80:
                cv2.rectangle(out, (x, y), (x+w, y+h), (0, 200, 100), 2)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    else:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Stats
    stats = dict(
        brightness = int(np.mean(gray)),
        contrast   = int(np.std(gray)),
        size       = f"{img.shape[1]}×{img.shape[0]}",
    )
    return out, stats


# ═══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 EduAI")
    st.markdown("<div style='color:#5a5a90;font-family:serif;margin-top:-8px;margin-bottom:12px'>ذہین نظام تعلیم</div>", unsafe_allow_html=True)

    risk  = compute_risk()
    rc, rb, rl, rlu = risk_color(risk)
    st.markdown(f"""
    <div style='background:#13132a;border:1px solid #2a2a45;border-radius:10px;padding:12px;margin-bottom:10px'>
      <div style='font-size:1.5rem;font-weight:600;color:{rc}'>{int(risk*100)}%</div>
      <div class='{rb}' style='display:inline-block;margin-top:4px'>{rl} · {rlu}</div>
      <div style='color:#5a5a80;font-size:.78rem;margin-top:6px'>Dropout Risk Score</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 👤 Student Info")
    ss.student_name  = st.text_input("Name / نام",          ss.student_name)
    ss.student_class = st.selectbox("Class / جماعت",
        ["Class 9","Class 10","Class 11","Class 12"],
        index=["Class 9","Class 10","Class 11","Class 12"].index(ss.student_class))
    ss.city = st.text_input("City / شہر", ss.city)

    st.divider()
    st.markdown("### 🔑 Groq API Key")
    ss.groq_key = st.text_input("API Key", ss.groq_key, type="password",
                                 help="Get free key at console.groq.com")
    if not ss.groq_key:
        st.caption("⚠ Key needed for AI study plan")

    st.divider()
    st.markdown("<div style='color:#5a5a80;font-size:.72rem'>spaCy · RandomForest · Groq · Plotly · OpenCV · Streamlit</div>",
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#   MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs([
    "📊 Dashboard",
    "📝 Enter Grades",
    "🧠 Risk Analysis",
    "🤖 Study Plan",
    "📷 Scanner",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════
with t1:
    st.markdown(f"## Performance Dashboard &nbsp; <span style='color:#5a5a90;font-size:.9rem'>کارکردگی کا جائزہ</span>", unsafe_allow_html=True)
    st.caption(f"📅 {date.today().strftime('%A, %d %B %Y')} · {ss.student_name} · {ss.student_class}")

    risk  = compute_risk()
    rc, _, rl, _ = risk_color(risk)
    ga, qa = grade_avg(), quiz_avg()

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Grade Average</div>
            <div class='metric-val' style='color:#7878ff'>{ga:.1f}%</div>
            <div class='metric-ur'>اوسط نمبر</div></div>""", unsafe_allow_html=True)
    with c2:
        col = "#ffb347" if ss.attendance < 80 else "#4ade80"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Attendance</div>
            <div class='metric-val' style='color:{col}'>{ss.attendance}%</div>
            <div class='metric-ur'>حاضری</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Dropout Risk</div>
            <div class='metric-val' style='color:{rc}'>{int(risk*100)}%</div>
            <div class='metric-ur'>خطرے کا اندازہ</div></div>""", unsafe_allow_html=True)
    with c4:
        col = "#ff6b6b" if qa < 50 else "#78ffb4"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-lbl'>Quiz Average</div>
            <div class='metric-val' style='color:{col}'>{qa:.1f}%</div>
            <div class='metric-ur'>کوئز اوسط</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Trend + Radar ──────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📈 Grade Trends — آخری 5 ہفتے")
        fig = go.Figure()
        for i, subj in enumerate(SUBJECTS):
            fig.add_trace(go.Scatter(
                x=WEEKS, y=ss.grades[subj], name=subj,
                mode="lines+markers",
                line=dict(color=subj_color(i), width=2),
                marker=dict(size=6),
            ))
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        fig.update_layout(yaxis=dict(range=[0, 100], gridcolor="#1e1e32",
                                     tickfont=dict(color="#7878a0")))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 🎯 Subject Radar — مضامین")
        vals = [avg(ss.grades[s]) for s in SUBJECTS] + [avg(ss.grades[SUBJECTS[0]])]
        cats = SUBJECTS + [SUBJECTS[0]]
        fig2 = go.Figure(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            fillcolor="rgba(120,120,255,0.15)",
            line=dict(color="#7878ff", width=2),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=280,
            polar=dict(
                bgcolor="#0a0a14",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#2a2a45", tickfont=dict(color="#5a5a80")),
                angularaxis=dict(gridcolor="#2a2a45", tickfont=dict(color="#9999cc")),
            ))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Bar Chart + Weak Areas ─────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("#### 📊 Subject Performance")
        avgs = [avg(ss.grades[s]) for s in SUBJECTS]
        cols = [subj_color(i) for i in range(len(SUBJECTS))]
        fig3 = go.Figure(go.Bar(
            x=SUBJECTS, y=avgs,
            marker=dict(color=cols, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in avgs],
            textposition="outside",
            textfont=dict(color="#c8c8d8", size=11),
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=260)
        fig3.update_layout(yaxis=dict(range=[0, 110], gridcolor="#1e1e32", tickfont=dict(color="#7878a0")))
        fig3.add_hline(y=50, line_dash="dot", line_color="#ff6b6b",
                       annotation_text="Pass line", annotation_font_color="#ff6b6b")
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown("#### ⚠ Weak Areas — کمزور موضوعات")
        ranked = sorted(SUBJECTS, key=lambda s: avg(ss.grades[s]))
        for s in ranked[:4]:
            a = avg(ss.grades[s])
            qi = ss.quiz[s]
            rc2, _, _, _ = risk_color(1 - a/100)
            st.markdown(f"""
            <div class='info-box' style='margin-bottom:8px'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <b style='color:#c8c8e8'>{s}</b>
                <span style='color:{rc2};font-weight:600'>{a:.0f}%</span>
              </div>
              <div style='background:#0a0a1a;height:5px;border-radius:3px;margin:8px 0'>
                <div style='background:{rc2};width:{a}%;height:100%;border-radius:3px'></div>
              </div>
              <div style='font-size:.75rem;color:#5a5a80'>Quiz: {qi}% &nbsp;·&nbsp; Trend: 
                {"↑ Improving" if ss.grades[s][-1]>ss.grades[s][0] else "↓ Declining"}</div>
            </div>""", unsafe_allow_html=True)

    # ── Progress over weeks ───────────────────────────────
    st.markdown("#### 📉 Week-over-Week Progress")
    weekly_avgs = [avg([ss.grades[s][w] for s in SUBJECTS]) for w in range(5)]
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=WEEKS, y=weekly_avgs,
        marker=dict(color=["#ff6b6b" if v < 50 else "#7878ff" for v in weekly_avgs]),
        text=[f"{v:.1f}%" for v in weekly_avgs], textposition="outside",
        textfont=dict(color="#c8c8d8")))
    fig4.add_trace(go.Scatter(x=WEEKS, y=weekly_avgs, mode="lines+markers",
        line=dict(color="#ffb347", width=2, dash="dot"),
        marker=dict(size=8, color="#ffb347"), name="Trend"))
    fig4.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False)
    fig4.update_layout(yaxis=dict(range=[0, 100], gridcolor="#1e1e32", tickfont=dict(color="#7878a0")))
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — ENTER GRADES
# ══════════════════════════════════════════════════════════
with t2:
    st.markdown("## Enter Grades &nbsp; <span style='color:#5a5a90;font-size:.9rem'>نمبر درج کریں</span>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ss.attendance = st.slider("Attendance % / حاضری", 0, 100, ss.attendance)
    with col_b:
        ss.assignments = st.slider("Assignment Completion %", 0, 100, ss.assignments)
    with col_c:
        ss.prev_result = st.selectbox("Previous Year Result", ["Pass","Borderline Pass","Fail"],
            index=["Pass","Borderline Pass","Fail"].index(ss.prev_result))

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📋 Weekly Grades — ہفتہ وار نمبر")
    st.caption("Enter marks out of 100 for each week and quiz.")

    for i, subj in enumerate(SUBJECTS):
        ur = SUBJ_UR[i]
        with st.expander(f"**{subj}** &nbsp; _{ur}_", expanded=(i < 2)):
            cols = st.columns(7)
            new_grades = []
            for wi, wk in enumerate(WEEKS):
                with cols[wi]:
                    v = st.number_input(wk, 0, 100, ss.grades[subj][wi],
                                        key=f"g_{subj}_{wi}", label_visibility="visible")
                    new_grades.append(v)
            with cols[5]:
                ss.quiz[subj] = st.number_input("Quiz", 0, 100, ss.quiz[subj],
                                                  key=f"q_{subj}")
            with cols[6]:
                cur = avg(new_grades)
                col = "#ff6b6b" if cur < 40 else "#ffb347" if cur < 60 else "#4ade80"
                st.markdown(f"<div style='text-align:center;margin-top:28px'><b style='color:{col}'>{cur:.0f}%</b><br><span style='font-size:.7rem;color:#5a5a80'>avg</span></div>",
                            unsafe_allow_html=True)
            ss.grades[subj] = new_grades

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 💬 Student Feedback — NLP Analysis")
    st.caption("Write in English or Urdu · انگریزی یا اردو میں لکھیں")
    ss.feedback = st.text_area(
        "Your thoughts, difficulties, feelings about studies:",
        ss.feedback or "I find mathematics very difficult especially algebra. Physics concepts are confusing. I struggle with time management and feel stressed before exams. مجھے ریاضی بہت مشکل لگتی ہے۔",
        height=120, label_visibility="collapsed"
    )

    if st.button("🔍 Analyze Feedback (spaCy NLP)", type="primary"):
        with st.spinner("Running spaCy pipeline..."):
            ss.nlp_result = analyze_feedback(ss.feedback)
        st.success("NLP analysis complete — view results in Risk Analysis tab.")

    if st.button("🔄 Recalculate Risk", use_container_width=True):
        compute_risk()
        st.rerun()


# ══════════════════════════════════════════════════════════
# TAB 3 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════
with t3:
    st.markdown("## Risk Analysis &nbsp; <span style='color:#5a5a90;font-size:.9rem'>خطرے کا تجزیہ</span>", unsafe_allow_html=True)
    st.caption("🌲 Random Forest (5 trees) · sklearn · Synthetic Pakistani school dataset")

    risk  = compute_risk()
    rc, rb, rl, rlu = risk_color(risk)
    pct   = int(risk * 100)

    # ── Gauge + Factors ──────────────────────────────────
    col_g, col_f = st.columns(2)

    with col_g:
        st.markdown("#### 🎯 Dropout Risk Gauge — ناکامی کا خطرہ")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            domain=dict(x=[0,1], y=[0,1]),
            number=dict(suffix="%", font=dict(color=rc, size=52)),
            delta=dict(reference=50, decreasing=dict(color="#4ade80"),
                       increasing=dict(color="#ff6b6b")),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="#5a5a80",
                          tickfont=dict(color="#7878a0")),
                bar=dict(color=rc, thickness=0.25),
                bgcolor="#13132a",
                steps=[
                    dict(range=[0,35],  color="#0d3d20"),
                    dict(range=[35,60], color="#3d2f10"),
                    dict(range=[60,100],color="#3d1515"),
                ],
                threshold=dict(line=dict(color="#ffffff",width=2), value=pct),
            )
        ))
        fig_g.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(f"""<div class='info-box'>
            <span class='{rb}'>{rl} &nbsp; {rlu}</span>
            <div style='margin-top:8px;font-size:.85rem;color:#9999cc'>
            {"⚠ Immediate intervention required. Contact parents and arrange tutoring." if pct >= 60
            else "📌 Monitor closely. Improve attendance and assignment submission." if pct >= 35
            else "✅ Performance is on track. Continue current momentum."}</div>
        </div>""", unsafe_allow_html=True)

    with col_f:
        st.markdown("#### 📉 Risk Factors — عوامل")
        factors = {
            "Grade Average":      max(0, 100 - grade_avg()) / 100,
            "Attendance":         max(0, 100 - ss.attendance) / 100,
            "Quiz Performance":   max(0, 100 - quiz_avg()) / 100,
            "Assignment Gap":     max(0, 100 - ss.assignments) / 100,
            "Grade Trend":        max(0, min(1, (50 - avg(
                [ss.grades[s][-1] - ss.grades[s][0] for s in SUBJECTS]
            )) / 100)),
        }
        names = list(factors.keys())
        vals  = [round(v * 100, 1) for v in factors.values()]
        cols2 = ["#ff6b6b" if v > 60 else "#ffb347" if v > 35 else "#4ade80" for v in vals]
        fig_f = go.Figure(go.Bar(
            y=names, x=vals, orientation="h",
            marker=dict(color=cols2, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in vals],
            textposition="outside", textfont=dict(color="#c8c8d8"),
        ))
        fig_f.update_layout(**PLOTLY_LAYOUT, height=260)
        fig_f.update_layout(
            xaxis=dict(range=[0,110], gridcolor="#1e1e32", tickfont=dict(color="#7878a0")),
            yaxis=dict(tickfont=dict(color="#9999cc")))
        st.plotly_chart(fig_f, use_container_width=True)

    # ── Random Forest Trees ───────────────────────────────
    st.markdown("#### 🌲 Random Forest — 5 Decision Trees Ensemble")
    tree_cols = st.columns(5)
    tree_lbls = ["Grade\nTree", "Attend\nTree", "Quiz\nTree", "Trend\nTree", "Assign\nTree"]
    for i, (tc, lbl, sc) in enumerate(zip(tree_cols, tree_lbls, ss.tree_scores)):
        pct_t = int(sc * 100)
        rc_t, _, rl_t, _ = risk_color(sc)
        with tc:
            st.markdown(f"""<div class='metric-card' style='min-height:110px'>
                <div style='font-size:.65rem;color:#5a5a80;text-transform:uppercase;
                letter-spacing:1px;white-space:pre-line;margin-bottom:6px'>{lbl}</div>
                <div class='metric-val' style='color:{rc_t};font-size:1.6rem'>{pct_t}%</div>
                <div style='font-size:.7rem;color:{rc_t};margin-top:4px'>{rl_t}</div>
            </div>""", unsafe_allow_html=True)

    avg_trees = avg(ss.tree_scores)
    st.markdown(f"""<div style='text-align:center;padding:16px 0'>
        <span style='color:#7878a0;font-size:.85rem'>Ensemble Prediction (majority vote):&nbsp;</span>
        <span style='color:{rc};font-size:2rem;font-weight:600'>{int(avg_trees*100)}%</span>
        <span style='color:#7878a0;font-size:.85rem'>&nbsp;dropout probability</span>
    </div>""", unsafe_allow_html=True)

    # ── NLP Results ──────────────────────────────────────
    st.markdown("#### 💬 NLP Feedback Analysis — spaCy Engine")
    if ss.nlp_result:
        nl = ss.nlp_result
        sent_color = "#ff6b6b" if nl["sentiment"] == "Negative" else "#4ade80"

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.markdown(f"""<div class='info-box'>
                <div style='font-size:.75rem;color:#5a5a80;margin-bottom:6px'>SENTIMENT</div>
                <span style='color:{sent_color};font-weight:600;font-size:1.1rem'>{nl['sentiment']}</span>
                {"&nbsp; 😔 Stress detected" if nl["sentiment"]=="Negative" else "&nbsp; 😊 Positive outlook"}
                {"<br><span style='font-size:.75rem;color:#7a5a80'>🌐 Urdu detected</span>" if nl["urdu_detected"] else ""}
            </div>""", unsafe_allow_html=True)

            if nl["stress_words"]:
                chips = "".join([f"<span class='chip' style='background:#3d1515;color:#ff8080;border:1px solid #5a2020'>{w}</span>"
                                 for w in nl["stress_words"][:6]])
                st.markdown(f"<div class='info-box'><div style='font-size:.75rem;color:#5a5a80;margin-bottom:6px'>STRESS INDICATORS</div>{chips}</div>",
                            unsafe_allow_html=True)

        with col_n2:
            if nl["weak_subjects"]:
                chips = "".join([f"<span class='chip' style='background:#3d1515;color:#ff8080;border:1px solid #5a2020'>{s}</span>"
                                 for s in nl["weak_subjects"]])
                st.markdown(f"<div class='info-box'><div style='font-size:.75rem;color:#5a5a80;margin-bottom:6px'>MENTIONED WEAK SUBJECTS</div>{chips}</div>",
                            unsafe_allow_html=True)

            if nl["key_phrases"]:
                chips = "".join([f"<span class='chip' style='background:#0d1a2d;color:#78b4ff;border:1px solid #1a3a5a'>{p}</span>"
                                 for p in nl["key_phrases"][:6]])
                st.markdown(f"<div class='info-box'><div style='font-size:.75rem;color:#5a5a80;margin-bottom:6px'>KEY PHRASES (noun chunks)</div>{chips}</div>",
                            unsafe_allow_html=True)

        if nl["entities"]:
            ent_html = "".join([f"<span class='chip' style='background:#1a1a0d;color:#ffcc44;border:1px solid #4a4a1a'>{e[0]} <i style='font-size:.65rem;color:#8a8a50'>{e[1]}</i></span>"
                                for e in nl["entities"][:8]])
            st.markdown(f"<div class='info-box'><div style='font-size:.75rem;color:#5a5a80;margin-bottom:6px'>NAMED ENTITIES (spaCy NER)</div>{ent_html}</div>",
                        unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box' style='color:#5a5a80;text-align:center'>Enter feedback in the Grades tab and click Analyze Feedback</div>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 4 — STUDY PLAN (GROQ)
# ══════════════════════════════════════════════════════════
with t4:
    st.markdown("## AI Study Plan &nbsp; <span style='color:#5a5a90;font-size:.9rem'>مطالعہ کا منصوبہ</span>", unsafe_allow_html=True)
    st.caption("🤖 Powered by Groq · llama-3.3-70b-versatile")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1: hrs  = st.slider("Study Hours / Day", 1, 8, 3)
    with col_p2: dur  = st.selectbox("Duration", ["2 Weeks","4 Weeks","8 Weeks"], index=1)
    with col_p3: foc  = st.selectbox("Focus Strategy", ["Weakest Subjects First","Balanced","Exam Prep"])

    weak_subj = sorted(SUBJECTS, key=lambda s: avg(ss.grades[s]))[:3]
    st.markdown(f"""<div class='info-box'>
        <div style='font-size:.75rem;color:#5a5a80;margin-bottom:6px'>FOCUS SUBJECTS (auto-detected weak)</div>
        {"".join([f"<span class='chip' style='background:#3d1515;color:#ff8080;border:1px solid #5a2020'>⚠ {s}</span>" for s in weak_subj])}
    </div>""", unsafe_allow_html=True)

    c_en, c_ur = st.columns(2)
    with c_en:
        if st.button("🤖 Generate English Plan", type="primary", use_container_width=True):
            if not ss.groq_key:
                st.error("Please add your Groq API key in the sidebar.")
            else:
                with st.spinner("Groq AI is building your personalized plan... 🧠"):
                    plan, err = generate_plan("en")
                if err: st.error(err)
                else:   ss.plan_en = plan; st.success("English plan ready!")
    with c_ur:
        if st.button("🤖 اردو منصوبہ بنائیں", use_container_width=True):
            if not ss.groq_key:
                st.error("Please add your Groq API key in the sidebar.")
            else:
                with st.spinner("AI اردو منصوبہ بنا رہا ہے... 🧠"):
                    plan, err = generate_plan("ur")
                if err: st.error(err)
                else:   ss.plan_ur = plan; st.success("اردو منصوبہ تیار ہے!")

    if ss.plan_en:
        st.markdown("### 📘 English Study Plan")
        st.markdown(f"<div class='plan-block'>{ss.plan_en}</div>", unsafe_allow_html=True)

    if ss.plan_ur:
        st.markdown("### 📗 اردو مطالعہ منصوبہ")
        st.markdown(f"<div class='plan-urdu'>{ss.plan_ur}</div>", unsafe_allow_html=True)

    if not ss.plan_en and not ss.plan_ur:
        st.markdown("""<div class='info-box' style='text-align:center;padding:2rem'>
            <div style='font-size:2rem;margin-bottom:8px'>🤖</div>
            <div style='color:#9999cc;font-weight:500'>Add your Groq API key and click Generate</div>
            <div style='color:#5a5a80;font-size:.85rem;margin-top:4px'>Free key at <b>console.groq.com</b></div>
        </div>""", unsafe_allow_html=True)

    # ── Study Schedule Chart ──────────────────────────────
    st.markdown("#### 📅 Suggested Weekly Schedule")
    all_avgs = {s: avg(ss.grades[s]) for s in SUBJECTS}
    total_wt  = sum(1 / (v + 1) for v in all_avgs.values())
    alloc     = {s: round(hrs * (1/(all_avgs[s]+1)) / total_wt, 1) for s in SUBJECTS}

    fig_s = go.Figure(go.Bar(
        x=SUBJECTS, y=list(alloc.values()),
        marker=dict(color=PLOT_COLORS, line=dict(width=0)),
        text=[f"{v}h" for v in alloc.values()],
        textposition="outside", textfont=dict(color="#c8c8d8"),
    ))
    fig_s.update_layout(**PLOTLY_LAYOUT, height=220,
        title=dict(text=f"Daily allocation ({hrs}h total) — Weak subjects get more time",
                   font=dict(color="#7878a0", size=12)))
    fig_s.update_layout(yaxis=dict(title="Hours", gridcolor="#1e1e32", tickfont=dict(color="#7878a0")))
    st.plotly_chart(fig_s, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 5 — SCANNER (OpenCV)
# ══════════════════════════════════════════════════════════
with t5:
    st.markdown("## Answer Sheet Scanner &nbsp; <span style='color:#5a5a90;font-size:.9rem'>جواب نامہ اسکینر</span>", unsafe_allow_html=True)
    st.caption("👁️ OpenCV computer vision pipeline — grayscale · threshold · edge detection · contour extraction")

    uploaded = st.file_uploader("Upload handwritten answer sheet / JPG or PNG",
                                 type=["jpg","jpeg","png"])

    if uploaded:
        img_bytes = uploaded.read()
        col_orig, col_proc = st.columns(2)

        with col_orig:
            st.markdown("**Original Image**")
            st.image(img_bytes, use_container_width=True)

        mode = st.radio("OpenCV Processing Mode",
                        ["orig","gray","thresh","edges","boxes"],
                        format_func=lambda m: {
                            "orig":"🖼 Original","gray":"⬛ Grayscale",
                            "thresh":"🔲 Threshold (Otsu)","edges":"📐 Edge Detection (Canny)",
                            "boxes":"📦 Contour Boxes",
                        }[m], horizontal=True)

        out_img, stats = process_image(img_bytes, mode)

        with col_proc:
            st.markdown(f"**{mode.title()} — OpenCV Output**")
            if out_img is not None:
                st.image(out_img, use_container_width=True)
                if isinstance(stats, dict):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Brightness", stats["brightness"])
                    c2.metric("Contrast (σ)", stats["contrast"])
                    c3.metric("Dimensions", stats["size"])
            else:
                st.error(stats)

        # ── Simulated OCR Extraction ──────────────────────
        st.markdown("#### 📋 Extracted Information — OCR Results")
        st.info("ℹ️ For production use: integrate Tesseract (`pytesseract`) or EasyOCR for actual handwriting recognition. Below shows simulated extraction logic.", icon="ℹ️")

        with st.spinner("Running OpenCV preprocessing for OCR..."):
            arr  = np.frombuffer(img_bytes, np.uint8)
            img2 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_regions = [cv2.boundingRect(c) for c in cnts
                            if 20 < cv2.boundingRect(c)[2] < 400
                            and 10 < cv2.boundingRect(c)[3] < 80]

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(f"""<div class='info-box'>
                <div style='font-size:.75rem;color:#5a5a80;margin-bottom:8px'>IMAGE ANALYSIS</div>
                <div style='font-size:.85rem;color:#c8c8e8'>
                🔲 Detected text regions: <b>{len(text_regions)}</b><br>
                📐 Image brightness: <b>{int(np.mean(gray))}/255</b><br>
                🎯 Binarization quality: <b>{"Good" if int(np.std(gray))>60 else "Low contrast"}</b><br>
                📊 Otsu threshold: <b>{int(cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0])}</b>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_i2:
            st.markdown(f"""<div class='info-box'>
                <div style='font-size:.75rem;color:#5a5a80;margin-bottom:8px'>PIPELINE STATUS</div>
                <div style='font-size:.85rem'>
                ✅ <span style='color:#4ade80'>Grayscale conversion</span><br>
                ✅ <span style='color:#4ade80'>Otsu binarization</span><br>
                ✅ <span style='color:#4ade80'>Contour detection ({len(cnts)} found)</span><br>
                ✅ <span style='color:#4ade80'>Region of interest extraction</span><br>
                {"✅" if len(text_regions)>3 else "⚠"} <span style='color:{"#4ade80" if len(text_regions)>3 else "#ffb347"}'>Text region filtering ({len(text_regions)} valid)</span>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='info-box' style='text-align:center;padding:3rem'>
            <div style='font-size:3rem;margin-bottom:10px'>📄</div>
            <div style='color:#9999cc;font-weight:500'>Upload a handwritten answer sheet</div>
            <div style='color:#5a5a80;font-size:.85rem;margin-top:4px'>
            JPG or PNG · OpenCV will apply grayscale, thresholding, edge detection & contour extraction</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("##### 🔬 OpenCV Pipeline Overview")
        steps = [
            ("1. Load & Resize",      "cv2.imdecode + cv2.resize",         "Standardize input to 600px width"),
            ("2. Grayscale",          "cv2.cvtColor(BGR2GRAY)",            "Remove color, reduce noise"),
            ("3. Otsu Threshold",     "cv2.THRESH_BINARY + THRESH_OTSU",   "Adaptive binarization for handwriting"),
            ("4. Edge Detection",     "cv2.Canny(50, 150)",                "Detect character boundaries"),
            ("5. Contour Extraction", "cv2.findContours + boundingRect",   "Isolate answer boxes & text regions"),
            ("6. OCR Ready",          "Pass to Tesseract / EasyOCR",       "Extract actual text from regions"),
        ]
        for step, code, desc in steps:
            st.markdown(f"""<div class='info-box' style='margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
                <div><b style='color:#c8c8e8'>{step}</b> &nbsp;
                <code style='background:#0a0a14;padding:2px 8px;border-radius:4px;font-size:.78rem;color:#78d4ff'>{code}</code></div>
                <div style='font-size:.8rem;color:#5a5a80'>{desc}</div>
            </div>""", unsafe_allow_html=True)
