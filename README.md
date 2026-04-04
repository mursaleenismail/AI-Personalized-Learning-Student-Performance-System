# 🎓 EduAI · AI Personalized Learning & Student Performance System
### ذہین نظام تعلیم

---

## Stack
| Component | Technology |
|-----------|-----------|
| 👁️ Computer Vision | OpenCV — grayscale, threshold, Canny edges, contours |
| 💬 NLP | spaCy — NER, noun chunks, sentiment from feedback |
| 🧠 Machine Learning | scikit-learn Random Forest — dropout risk prediction |
| 🤖 AI / LLM | Groq (llama-3.3-70b-versatile) — Urdu/English study plan |
| 📊 Data Science | Plotly — trend, radar, gauge, bar charts |
| 🖥️ UI | Streamlit — full dark-themed multi-tab dashboard |

---

## Setup

### 1. Clone / download files
```
app.py
requirements.txt
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 4. Get Groq API key (free)
- Go to https://console.groq.com
- Create a free account → API Keys → Create Key

### 5. Run
```bash
streamlit run app.py
```

---

## Features

### 📊 Tab 1 — Dashboard
- 4 metric cards: Grade avg, Attendance, Dropout Risk %, Quiz avg
- Plotly line chart — grade trends per subject over 5 weeks
- Radar chart — all subjects at a glance
- Bar chart — subject performance vs pass line
- Week-over-week progress bar chart

### 📝 Tab 2 — Enter Grades
- Sliders: attendance %, assignment completion %, previous result
- Expandable per-subject input: 5 weekly grades + quiz score
- Feedback text area (Urdu + English supported)
- One-click spaCy NLP analysis

### 🧠 Tab 3 — Risk Analysis
- Random Forest (5 trees) ensemble — trained on synthetic Pakistani school data
- Plotly gauge chart for dropout probability
- Horizontal bar chart — risk factor breakdown
- Per-tree scores shown as metric cards
- spaCy NLP output: sentiment, stress keywords, weak subjects, NER entities, noun chunks

### 🤖 Tab 4 — Study Plan (Groq AI)
- Generates detailed 4-week plan in English via llama-3.3-70b-versatile
- Generates detailed 4-week plan in Urdu (اردو)
- Auto-detects 3 weakest subjects from grades
- Daily hour allocation chart (inverse-weighted by performance)

### 📷 Tab 5 — Answer Sheet Scanner
- Upload JPG/PNG of handwritten answer sheet
- OpenCV modes: Original, Grayscale, Otsu Threshold, Canny Edges, Contour Boxes
- Real-time image stats: brightness, contrast, dimensions, Otsu threshold value
- Contour detection counts text regions
- Pipeline overview with code references

---

## Architecture

```
Student Input (Grades + Feedback + Image)
        │
        ├── spaCy NLP ────────► Sentiment, Entities, Weak Subjects
        │
        ├── Random Forest ────► Dropout Risk Score (0–100%)
        │   (5 decision trees,
        │    sklearn ensemble)
        │
        ├── Groq LLM ─────────► Personalized Study Plan (EN + UR)
        │   (llama-3.3-70b)
        │
        ├── OpenCV ───────────► Grayscale → Threshold → Edges → Contours
        │
        └── Plotly ───────────► Line, Radar, Gauge, Bar, Scatter charts
                                  (all rendered in Streamlit)
```
