# 🎓 AI-Powered Smart Campus Analytics System
### For Student and Resource Management | IEEE Research Project

---

## 📄 Abstract
An AI-powered analytics system that integrates machine learning with data analytics to predict student performance, identify at-risk students, and optimize campus resource allocation — aligned with **SDG 4, 9, 10, and 11**.

---

## 🌍 SDG Alignment
| SDG | Goal | How This Project Contributes |
|-----|------|------------------------------|
| 📚 SDG 4 | Quality Education | GPA prediction, at-risk detection, attendance analysis |
| 🏗️ SDG 9 | Innovation & Infrastructure | AI/ML models, digital campus dashboard |
| ⚖️ SDG 10 | Reduced Inequalities | SES equity analysis, targeted interventions |
| 🌱 SDG 11 | Sustainable Cities | Classroom & resource utilization optimization |

---

## 📁 Project Structure
```
smart_campus/
├── app.py                                  # Streamlit dashboard (main app)
├── Smart_Campus_Analytics_Updated.ipynb    # Jupyter notebook (ML training)
├── requirements.txt                        # Python dependencies
├── .env.example                            # Template for local secrets (copy to .env)
├── .streamlit/secrets.toml.example         # Template for Streamlit Cloud secrets
├── .gitignore                              # Keeps real secrets out of version control
├── student_dataset.csv                     # Student records (500+ rows, 25 features)
├── faculty_dataset.csv                     # Faculty data
├── classroom_dataset.csv                   # Classroom utilization data
├── models/                                 # Auto-created after running notebook
│   ├── gpa_regressor.pkl
│   ├── pass_fail_classifier.pkl
│   ├── at_risk_model.pkl
│   ├── scaler_gpa.pkl
│   ├── scaler_pf.pkl
│   ├── scaler_risk.pkl
│   ├── le_pass_fail.pkl
│   ├── le_risk.pkl
│   ├── encoders.pkl
│   └── feature_cols.pkl
└── plots/                                  # Auto-created after running notebook
    ├── eda_overview.png
    ├── correlation_heatmap.png
    ├── gpa_regression.png
    ├── classification_results.png
    ├── atrisk_confusion.png
    ├── feature_importance.png
    ├── resource_analysis.png
    └── sdg_dashboard.png
```

---

## 🚀 Setup & Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Secrets (API key & database password)
No credentials are hardcoded in the source — the app reads them from environment
variables at runtime. Choose **one** of the two options below:

**Option A — Local `.env` file (recommended for local dev)**
```bash
cp .env.example .env
# then edit .env and fill in your real GROQ_API_KEY and MYSQL_PASSWORD
```

**Option B — Streamlit secrets (recommended for Streamlit Cloud deploys)**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml with your real values
```

Both `.env` and `.streamlit/secrets.toml` are already listed in `.gitignore`,
so they won't accidentally get committed to version control.

### Step 3: Train ML Models (run the notebook)
```bash
jupyter notebook Smart_Campus_Analytics_Updated.ipynb
# Run ALL cells top-to-bottom (Kernel → Restart & Run All)
```

### Step 4: Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🧠 ML Models
| Model | Task | Algorithm | Target |
|-------|------|-----------|--------|
| GPA Regressor | Regression | Gradient Boosting / Random Forest | `current_gpa` |
| Pass/Fail Classifier | Classification | Best of 6 models | `pass_fail` |
| At-Risk Detector | Classification | Gradient Boosting | `at_risk_student` |

---

## 📊 Dashboard Pages
1. **📌 About** — Project overview, SDG alignment
2. **🎓 Executive Dashboard** — University-wide KPIs
3. **📚 Student Performance** — GPA, attendance, at-risk heatmaps
4. **📅 Attendance Analytics** — Band distribution, department trends
5. **👨‍🏫 Faculty Analytics** — Workload, feedback, research output
6. **🎯 Placement Tracker** — Placement readiness scoring
7. **🏫 Campus & Classroom** — Utilization, maintenance status
8. **📖 Library & Digital** — Engagement correlation analysis
9. **🔮 Student Predictor** — Real-time ML prediction with gauges
10. **🤖 AI Chatbot** — Natural language campus Q&A (Cohere-powered)

---

## 🛠 Tech Stack
- **Python** — Pandas, NumPy, Scikit-learn
- **Streamlit** — Interactive web dashboard
- **Plotly** — Interactive charts and gauges
- **Cohere** — AI chatbot and recommendations engine
- **Jupyter** — ML training notebook
