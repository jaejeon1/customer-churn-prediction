# 📊 ChurnSense — Customer Churn Prediction

> *Identifying at-risk customers before they leave — with explainable AI*

🔗 **Live App:** [ChurnSense on Streamlit](YOUR_STREAMLIT_URL)
📁 **GitHub:** [customer-churn-prediction](YOUR_GITHUB_URL)

---

## 🧩 The Business Problem

Every month, thousands of telecom customers quietly decide to leave —
and most companies don't know until it's too late.

| Metric | Value |
|---|---|
| Annual churn rate | **26%** |
| Customers lost per year | **~13,000** |
| Average monthly spend | **€35 / customer** |
| Annual revenue at risk | **€5.46 million** |

> The retention team had no way to identify *who* was about to leave
> until they already had. They were reacting — not preventing.

---

## 💡 The Solution

**ChurnSense** is a machine learning system that scores every customer's
churn probability in real time — and explains *why* — so retention teams
can act before it's too late.

| Metric | Score |
|---|---|
| AUC-ROC | **0.900** |
| Recall | **0.940** — catches 94% of actual churners |
| Precision | **0.520** |
| Decision Threshold | **0.35** (optimised for recall) |
| Selected Model | **Logistic Regression** |

---

## 🏗 Architecture

```
Customer Data (P5_customer_churn.csv)
            ↓
  Feature Engineering (22 features)
            ↓
  Logistic Regression Pipeline (.pkl)
            ↓
  FastAPI /predict endpoint (main.py)
            ↓  HTTP POST JSON
  Streamlit Frontend (app.py)
            ↓
  SHAP Explanation + Confidence Score
            ↓
  Business User sees Result + Recommendations
```

**Deployment:** Local (laptop) — run `./start.sh` to launch both
FastAPI backend and Streamlit frontend simultaneously.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── app.py                  # Streamlit web application (frontend)
├── main.py                 # FastAPI backend — /predict endpoint
├── requirements.txt        # Python dependencies
├── start.sh                # One-command startup script
├── README.md               # Project documentation
│
├── churn_model.pkl         # Trained Logistic Regression pipeline
├── shap_explainer.pkl      # SHAP explainer for interpretability
├── feature_names.json      # Ordered feature list for prediction
└── model_config.json       # Model metadata and performance metrics
```

---

## 🔄 Analysis Workflow

```
1. Data Loading
   └── Dataset: P5_customer_churn.csv (50,000 customers, 22 features)

2. Missing Value Handling
   ├── Domain-logic imputation (days_since_last_purchase → 999)
   ├── Group-based median imputation (monthly_spend by contract & plan)
   └── Indicator columns for satisfaction_score and nps_score

3. Exploratory Data Analysis
   ├── Univariate distributions
   ├── Churn rate by categorical segments
   └── Boxplots of numerical features vs churn

4. Cohort & Segment Analysis
   ├── Tenure cohorts (<6m, 6-12m, 1-2yr, 2-5yr, 5yr+)
   └── Spend tier segmentation (Low / Mid / High / Premium)

5. Feature Engineering
   ├── Contract type ordinal encoding
   ├── Tenure risk score
   ├── Interaction features: new_mtm_flag, support_call_rate, frustration_index
   └── Log transforms: monthly_spend, return_rate

6. Model Training & Comparison
   ├── Logistic Regression (Pipeline with StandardScaler)
   ├── Random Forest
   └── XGBoost

7. Model Evaluation
   ├── AUC-ROC, Recall, Precision, F1
   ├── Confusion matrices (all 3 models)
   └── Threshold tuning (0.5 → 0.35)

8. SHAP Explainability
   ├── Global feature importance (beeswarm + bar chart)
   ├── Individual prediction explanation (waterfall)
   └── Feature dependence plot (contract type vs tenure)

9. Deployment
   └── FastAPI backend + Streamlit frontend
```

---

## 📈 Model Comparison

| Model | AUC-ROC | Recall | Precision | F1 | Missed Churners |
|---|---|---|---|---|---|
| ✅ **Logistic Regression** | **0.900** | **0.940** | 0.520 | 0.670 | **235** |
| XGBoost | 0.892 | 0.852 | 0.585 | 0.694 | 308 |
| Random Forest | 0.895 | 0.621 | 0.698 | 0.658 | 788 |

**Why Logistic Regression?**
- Highest recall — catches 94% of actual churners
- Only 235 missed churners vs 788 for Random Forest
- Fully interpretable coefficients for stakeholder communication
- Lower computational cost for real-time scoring

---

## 🔑 Key Findings

| Finding | Detail |
|---|---|
| Contract type is #1 churn driver | Month-to-Month = **45.2% churn** vs **2.4%** annual — 19x difference |
| Early tenure is highest-risk window | **41% churn in first 6 months** → drops to 3% after 5 years |
| Support calls signal frustration | 3+ calls in 6 months = strong churn indicator |
| Spend and region are NOT predictive | Churn rate flat (~26%) across all spend tiers and regions |
| Satisfaction score is a weak signal | Likely data quality issue — flagged for investigation |

---

## 🔍 Top Churn Drivers (SHAP Analysis)

> All 3 models agree on the top 2 drivers — high-confidence findings.

| Rank | Feature | SHAP Value | Insight |
|---|---|---|---|
| 🥇 1 | Contract Type | **+2.09** | Month-to-Month → highest churn risk |
| 🥈 2 | Customer Tenure | **+0.99** | Short tenure → high churn risk |
| 🥉 3 | Satisfaction Score | **+0.41** | Low satisfaction → increased risk |
| 4 | Support Calls (6m) | **+0.33** | More calls → higher churn risk |
| 5 | Return Rate | **+0.18** | High returns → product friction |
| 6 | NPS Score | **+0.16** | Low NPS → exit intent signal |

---

## 💰 Business Impact & ROI

| | Value |
|---|---|
| Churners correctly identified | **12,220 / year** (94% recall) |
| Customers saved (25% retention rate) | **3,055 / year** |
| Protected revenue | **€1.28M / year** |
| Year 1 deployment cost | **€13,000** |
| **Payback period** | **< 4 days** |

---

## 💡 Business Recommendations

| Priority | Action | Target Segment |
|---|---|---|
| 🔴 Immediate | Proactive outreach within 48h | Month-to-Month, tenure < 6 months |
| 🟠 Short-term | Annual contract upgrade incentives | All Month-to-Month customers |
| 🟡 Operational | Flag customers with 3+ support calls | All segments |
| 🟢 Strategic | Remove gender & region in model v2 | Data team |

---

## ⚖️ Ethics & Legal

| Risk | Mitigation |
|---|---|
| **GDPR Article 22** — automated customer profiling | No adverse decisions without human review |
| **EU AI Act** — limited-risk AI system | Full SHAP explainability; transparency obligations met |
| **Demographic bias** — gender, region features | SHAP confirmed near-zero impact (< 0.01); removing in v2 |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start both FastAPI and Streamlit with one command
chmod +x start.sh
./start.sh
```

**Then open:**
- Streamlit App → `http://localhost:8501`
- FastAPI Docs → `http://localhost:8000/docs`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API status check |
| GET | `/model-info` | Model metrics and features |
| POST | `/predict` | Churn prediction + confidence + SHAP |

**Example request to `/predict`:**
```json
{
  "contract_type": "Month-to-Month",
  "customer_tenure_months": 3,
  "monthly_spend_eur": 35.0,
  "satisfaction_score": 2.5,
  "nps_score": 4,
  "num_support_calls_last_6m": 5,
  "return_rate_pct": 12.0,
  "num_purchases_last_3m": 1,
  "payment_method": "Credit Card",
  "region": "West",
  "gender": "Male",
  "has_premium_plan": 0
}
```

**Example response:**
```json
{
  "prediction": 1,
  "risk_label": "High Risk",
  "churn_probability": 0.87,
  "confidence": 0.87,
  "top_risk_factors": [
    {"feature": "contract_type_encoded", "shap_value": 1.73},
    {"feature": "customer_tenure_months", "shap_value": 1.04}
  ]
}
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| pandas & numpy | Data manipulation |
| scikit-learn | Model training & evaluation |
| XGBoost | Gradient boosting comparison model |
| SHAP | Model explainability |
| matplotlib & seaborn | Visualisation |
| FastAPI + Uvicorn | REST API backend |
| Streamlit | Web application frontend |

---

## 👤 Author

**YOUR NAME** — Data Analytics Portfolio Project
Google Data Analytics Certificate · Google Business Intelligence Certificate (2025)

📧 YOUR_EMAIL
🔗 YOUR_LINKEDIN
