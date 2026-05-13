import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background — clean light */
.stApp { background-color: #f8f9fb; }

/* Hide default header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* App title bar */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 28px 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 32px;
}
.app-logo {
    background: linear-gradient(135deg, #e8623a, #c0392b);
    color: white;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
}
.app-title {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.3px;
}
.app-title span { color: #e8623a; }
.app-pills {
    margin-left: auto;
    display: flex;
    gap: 10px;
}
.pill {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 100px;
    padding: 4px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.pill strong { color: #0891b2; }

/* Risk badge */
.badge-high {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fef2ee;
    border: 1px solid #fca58a;
    color: #c0392b;
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-low {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf9;
    border: 1px solid #6ee7d4;
    color: #0f766e;
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Confidence bar */
.conf-wrap { margin: 20px 0; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}
.conf-track {
    background: #e5e7eb;
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
}
.conf-fill-high {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #f59e0b, #e8623a);
}
.conf-fill-low {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #34d399, #0891b2);
}
.threshold-note {
    font-size: 11px;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    margin-top: 5px;
}

/* Section headers */
.section-head {
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9ca3af;
    padding-bottom: 10px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 16px;
    margin-top: 24px;
}

/* Recommendation items */
.rec-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 10px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 13px;
    color: #374151;
    line-height: 1.5;
}
.rec-item:last-child { border-bottom: none; }

/* Plain text explanation */
.explanation-box {
    background: #fff7f5;
    border: 1px solid #fde8e0;
    border-left: 3px solid #e8623a;
    border-radius: 0 10px 10px 0;
    padding: 14px 16px;
    font-size: 13px;
    color: #374151;
    line-height: 1.6;
    margin: 16px 0;
}
.explanation-box.safe {
    background: #f0fdf9;
    border: 1px solid #ccfbf0;
    border-left: 3px solid #0d9488;
}

/* Metric overrides */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #111827 !important; font-size: 26px !important; font-weight: 600 !important; }

/* Tab styling */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    color: #6b7280;
}

/* Labels */
label, .stSelectbox label, .stNumberInput label {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Checkbox */
[data-testid="stCheckbox"] span { color: #374151 !important; font-size: 13px !important; }

/* Divider */
hr { border-color: #e5e7eb; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #d1d5db;
    border-radius: 10px;
}

/* Info box */
[data-testid="stAlert"] {
    background: #f0f9ff !important;
    border: 1px solid #bae6fd !important;
    color: #0369a1 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    with open("feature_names.json") as f:
        feature_names = json.load(f)
    with open("model_config.json") as f:
        config = json.load(f)
    return model, explainer, feature_names, config

try:
    model, explainer, FEATURE_NAMES, CONFIG = load_artifacts()
    THRESHOLD = CONFIG["threshold"]
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Could not load model files: {e}\n\nMake sure churn_model.pkl, shap_explainer.pkl, feature_names.json and model_config.json are in the same folder as app.py")
    st.stop()

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(contract_type, tenure, monthly_spend, satisfaction_score,
                       nps_score, support_calls, return_rate, purchases,
                       payment_method, region, gender,
                       sat_missing=False, nps_missing=False):

    contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    contract_encoded = contract_map[contract_type]

    if tenure < 6: tenure_risk = 4
    elif tenure < 12: tenure_risk = 3
    elif tenure < 24: tenure_risk = 2
    elif tenure < 60: tenure_risk = 1
    else: tenure_risk = 0

    new_mtm_flag = int(tenure < 6 and contract_type == "Month-to-Month")
    support_call_rate = support_calls / (tenure + 1)
    frustration_index = support_calls * (5 - satisfaction_score)
    log_monthly_spend = np.log1p(monthly_spend)
    log_return_rate = np.log1p(return_rate)

    row = {
        "contract_type_encoded": contract_encoded,
        "tenure_risk_score": tenure_risk,
        "customer_tenure_months": tenure,
        "log_monthly_spend": log_monthly_spend,
        "satisfaction_score": satisfaction_score,
        "nps_score": nps_score,
        "num_support_calls_last_6m": support_calls,
        "support_call_rate": support_call_rate,
        "frustration_index": frustration_index,
        "new_mtm_flag": new_mtm_flag,
        "log_return_rate": log_return_rate,
        "num_purchases_last_3m": purchases,
        "satisfaction_score_missing": int(sat_missing),
        "nps_score_missing": int(nps_missing),
        "payment_method_Credit Card": int(payment_method == "Credit Card"),
        "payment_method_Digital Wallet": int(payment_method == "Digital Wallet"),
        "payment_method_Invoice": int(payment_method == "Invoice"),
        "region_East": int(region == "East"),
        "region_North": int(region == "North"),
        "region_South": int(region == "South"),
        "region_West": int(region == "West"),
        "gender_Male": int(gender == "Male"),
    }
    return pd.DataFrame([row])[FEATURE_NAMES]


def predict_single(X):
    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba >= THRESHOLD)
    X_scaled = model.named_steps["scaler"].transform(X)
    shap_vals = explainer(X_scaled)
    shap_contributions = dict(zip(FEATURE_NAMES, shap_vals.values[0].tolist()))
    return proba, prediction, shap_contributions


FEATURE_LABELS = {
    "contract_type_encoded": "Contract Type",
    "customer_tenure_months": "Customer Tenure",
    "satisfaction_score": "Satisfaction Score",
    "num_support_calls_last_6m": "Support Calls (6m)",
    "log_return_rate": "Return Rate",
    "nps_score": "NPS Score",
    "new_mtm_flag": "New Month-to-Month Flag",
    "tenure_risk_score": "Tenure Risk Score",
    "frustration_index": "Frustration Index",
    "support_call_rate": "Support Call Rate",
    "log_monthly_spend": "Monthly Spend",
    "num_purchases_last_3m": "Purchases (3m)",
    "satisfaction_score_missing": "Satisfaction Missing",
    "nps_score_missing": "NPS Missing",
}

RECOMMENDATIONS_HIGH = [
    ("🎯", "Assign a retention specialist — reach out within 48 hours."),
    ("📋", "Offer an annual contract upgrade incentive — reduces churn risk ~19x."),
    ("📞", "Prioritise resolving open support tickets immediately."),
    ("🎁", "Consider a personalised loyalty reward or discount."),
]
RECOMMENDATIONS_LOW = [
    ("✅", "Customer is stable — no immediate intervention required."),
    ("📈", "Good candidate for upsell or premium plan upgrade."),
    ("💬", "Schedule a satisfaction check-in to maintain engagement."),
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">CS</div>
  <div class="app-title">Churn<span>Sense</span></div>
  <div class="app-pills">
    <div class="pill">Model: <strong>Logistic Regression</strong></div>
    <div class="pill">AUC-ROC <strong>0.900</strong></div>
    <div class="pill">Recall <strong>0.940</strong></div>
    <div class="pill">Threshold <strong>0.35</strong></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Single Customer Analysis", "📂  Batch CSV Upload"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Customer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_gap, col_result = st.columns([1.1, 0.08, 1])

    # ── Form ──────────────────────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="section-head">Contract & Account</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            contract_type = st.selectbox("Contract Type *", ["Month-to-Month", "One Year", "Two Year"])
        with c2:
            tenure = st.number_input("Tenure (months) *", min_value=0, max_value=999, value=6, step=1)

        c3, c4 = st.columns(2)
        with c3:
            monthly_spend = st.number_input("Monthly Spend (€) *", min_value=0.0, value=35.0, step=1.0)
        with c4:
            payment_method = st.selectbox("Payment Method *", ["Credit Card", "Bank Transfer", "Digital Wallet", "Invoice"])

        st.markdown('<div class="section-head">Satisfaction & Engagement</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            sat_unknown = st.checkbox("Satisfaction unknown")
            satisfaction_score = st.number_input("Satisfaction Score (1–5)", min_value=1.0, max_value=5.0,
                                                  value=3.5, step=0.1, disabled=sat_unknown)
        with c6:
            nps_unknown = st.checkbox("NPS unknown")
            nps_score = st.number_input("NPS Score (0–10)", min_value=0, max_value=10,
                                         value=5, step=1, disabled=nps_unknown)

        c7, c8 = st.columns(2)
        with c7:
            support_calls = st.number_input("Support Calls (last 6m) *", min_value=0, value=2, step=1)
        with c8:
            return_rate = st.number_input("Return Rate (%) *", min_value=0.0, value=5.0, step=0.5)

        st.markdown('<div class="section-head">Purchase Behaviour & Demographics</div>', unsafe_allow_html=True)
        c9, c10, c11 = st.columns(3)
        with c9:
            purchases = st.number_input("Purchases (last 3m) *", min_value=0, value=3, step=1)
        with c10:
            region = st.selectbox("Region *", ["West", "East", "North", "South", "Central"])
        with c11:
            gender = st.selectbox("Gender *", ["Male", "Female"])

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Analyse Churn Risk →", type="primary", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    with col_result:
        if run_btn:
            # Use medians for missing
            sat_val = 3.5 if sat_unknown else satisfaction_score
            nps_val = 5 if nps_unknown else nps_score

            X = engineer_features(
                contract_type, tenure, monthly_spend, sat_val, nps_val,
                support_calls, return_rate, purchases,
                payment_method, region, gender,
                sat_missing=sat_unknown, nps_missing=nps_unknown
            )
            proba, prediction, shap_vals = predict_single(X)
            pct = round(proba * 100, 1)
            is_high = prediction == 1
            risk_cls = "high" if is_high else "low"
            risk_label = "⚠ HIGH RISK" if is_high else "✓ LOW RISK"
            badge_cls = "badge-high" if is_high else "badge-low"
            fill_cls = "conf-fill-high" if is_high else "conf-fill-low"
            color = "#e8623a" if is_high else "#0d9488"

            # ── Risk badge + probability ──
            st.markdown(f'<div class="{badge_cls}">{risk_label}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Churn Probability", f"{pct}%")
            with m2:
                confidence = max(proba, 1 - proba)
                st.metric("Model Confidence", f"{round(confidence * 100, 1)}%")
            with m3:
                st.metric("Decision Threshold", f"{int(THRESHOLD*100)}%")

            # ── Confidence bar ──
            st.markdown(f"""
            <div class="conf-wrap">
              <div class="conf-label">
                <span>Churn Probability</span>
                <span>{pct}%</span>
              </div>
              <div class="conf-track">
                <div class="{fill_cls}" style="width:{pct}%"></div>
              </div>
              <div class="threshold-note">▲ Decision threshold at 35% — above this = High Risk</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Plain English explanation ──
            st.markdown('<div class="section-head">Why This Prediction</div>', unsafe_allow_html=True)

            sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
            top_pos = [(k, v) for k, v in sorted_shap if v > 0][:2]
            top_neg = [(k, v) for k, v in sorted_shap if v < 0][:1]

            risk_drivers = ", ".join([f"**{FEATURE_LABELS.get(k, k)}**" for k, v in top_pos])
            protect = f"**{FEATURE_LABELS.get(top_neg[0][0], top_neg[0][0])}**" if top_neg else None

            if is_high:
                explanation = f"This customer is flagged as **High Risk** ({pct}% churn probability). "
                if top_pos:
                    explanation += f"The primary drivers are {risk_drivers}. "
                if protect:
                    explanation += f"{protect} partially reduces the risk, but not enough to change the outcome."
                box_cls = "explanation-box"
            else:
                explanation = f"This customer is **Low Risk** ({pct}% churn probability). "
                if top_neg:
                    explanation += f"Key protective factors include {protect}. "
                if top_pos:
                    explanation += f"Minor risk signals from {risk_drivers} are present but insufficient to trigger concern."
                box_cls = "explanation-box safe"

            st.markdown(f'<div class="{box_cls}">{explanation}</div>', unsafe_allow_html=True)

            # ── SHAP bar chart ──
            st.markdown('<div class="section-head">Top Feature Contributions (SHAP)</div>', unsafe_allow_html=True)

            top_features = sorted(
                [(FEATURE_LABELS.get(k, k), v) for k, v in shap_vals.items()
                 if k in FEATURE_LABELS],
                key=lambda x: abs(x[1]), reverse=True
            )[:8]

            labels = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            colors_bar = ["#e8623a" if v > 0 else "#0d9488" for v in values]

            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#f8f9fb")
            bars = ax.barh(labels[::-1], values[::-1], color=colors_bar[::-1],
                           height=0.55, edgecolor="none")
            ax.axvline(0, color="#e5e7eb", linewidth=1)
            ax.set_xlabel("SHAP Value (impact on churn prediction)", color="#6b7280",
                          fontsize=9, labelpad=8)
            ax.tick_params(colors="#374151", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_color("#1e2230")

            pos_patch = mpatches.Patch(color="#e8623a", label="Increases churn risk")
            neg_patch = mpatches.Patch(color="#0d9488", label="Decreases churn risk")
            ax.legend(handles=[pos_patch, neg_patch], fontsize=8,
                      facecolor="#ffffff", edgecolor="#e5e7eb",
                      labelcolor="#374151", loc="lower right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # ── Recommendations ──
            st.markdown('<div class="section-head">Recommended Actions</div>', unsafe_allow_html=True)
            recs = RECOMMENDATIONS_HIGH if is_high else RECOMMENDATIONS_LOW
            for icon, text in recs:
                st.markdown(f'<div class="rec-item"><span>{icon}</span><span>{text}</span></div>',
                            unsafe_allow_html=True)

        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("👈  Fill in the customer details and click **Analyse Churn Risk** to see results.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-head">Upload Customer CSV</div>
    """, unsafe_allow_html=True)

    # Template download
    template_cols = [
        "contract_type", "customer_tenure_months", "monthly_spend_eur",
        "satisfaction_score", "nps_score", "num_support_calls_last_6m",
        "return_rate_pct", "num_purchases_last_3m", "payment_method",
        "region", "gender", "has_premium_plan"
    ]
    sample_data = pd.DataFrame([
        ["Month-to-Month", 3, 35.0, 2.5, 4, 5, 12.0, 1, "Credit Card", "West", "Male", 0],
        ["One Year", 24, 55.0, 4.2, 7, 1, 3.0, 8, "Bank Transfer", "North", "Female", 1],
        ["Two Year", 60, 80.0, 4.8, 9, 0, 1.0, 15, "Digital Wallet", "South", "Male", 1],
    ], columns=template_cols)

    csv_template = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV Template",
        csv_template,
        file_name="churnsense_template.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"],
                                 help="Must match the template column names above.")

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ {len(df_raw):,} customers loaded — running predictions...")

            results = []
            errors = []

            for i, row in df_raw.iterrows():
                try:
                    sat_missing = pd.isna(row.get("satisfaction_score"))
                    nps_missing = pd.isna(row.get("nps_score"))
                    sat_val = 3.5 if sat_missing else float(row["satisfaction_score"])
                    nps_val = 5.0 if nps_missing else float(row["nps_score"])

                    X = engineer_features(
                        contract_type=row["contract_type"],
                        tenure=float(row["customer_tenure_months"]),
                        monthly_spend=float(row["monthly_spend_eur"]),
                        satisfaction_score=sat_val,
                        nps_score=nps_val,
                        support_calls=float(row["num_support_calls_last_6m"]),
                        return_rate=float(row["return_rate_pct"]),
                        purchases=float(row["num_purchases_last_3m"]),
                        payment_method=row["payment_method"],
                        region=row["region"],
                        gender=row["gender"],
                        sat_missing=sat_missing,
                        nps_missing=nps_missing
                    )

                    proba = model.predict_proba(X)[0, 1]
                    prediction = int(proba >= THRESHOLD)
                    confidence = max(proba, 1 - proba)

                    results.append({
                        "churn_probability": round(proba, 4),
                        "churn_probability_pct": f"{round(proba*100,1)}%",
                        "confidence": round(confidence, 4),
                        "confidence_pct": f"{round(confidence*100,1)}%",
                        "risk_label": "High Risk" if prediction == 1 else "Low Risk",
                        "prediction": prediction,
                    })
                except Exception as e:
                    results.append({
                        "churn_probability": None,
                        "churn_probability_pct": "Error",
                        "confidence": None,
                        "confidence_pct": "Error",
                        "risk_label": f"Error: {str(e)}",
                        "prediction": None,
                    })

            df_results = pd.concat([df_raw.reset_index(drop=True),
                                     pd.DataFrame(results)], axis=1)

            # ── Summary stats ──
            st.markdown('<div class="section-head">Batch Summary</div>', unsafe_allow_html=True)

            valid = df_results[df_results["prediction"].notna()]
            n_total = len(valid)
            n_high = int(valid["prediction"].sum())
            n_low = n_total - n_high
            avg_prob = valid["churn_probability"].mean()
            avg_conf = valid["confidence"].mean()

            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                st.metric("Total Customers", f"{n_total:,}")
            with b2:
                st.metric("High Risk", f"{n_high:,}", delta=f"{round(n_high/n_total*100,1)}%",
                          delta_color="inverse")
            with b3:
                st.metric("Low Risk", f"{n_low:,}", delta=f"{round(n_low/n_total*100,1)}%")
            with b4:
                st.metric("Avg Churn Probability", f"{round(avg_prob*100,1)}%")
            with b5:
                st.metric("Avg Model Confidence", f"{round(avg_conf*100,1)}%")

            # ── Charts ──
            st.markdown('<div class="section-head">Risk Distribution</div>', unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)

            with ch1:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                fig2.patch.set_facecolor("#12151c")
                ax2.set_facecolor("#12151c")
                ax2.hist(valid["churn_probability"], bins=30,
                         color="#e8623a", alpha=0.8, edgecolor="none")
                ax2.axvline(THRESHOLD, color="#0d9488", linewidth=1.5,
                            linestyle="--", label=f"Threshold ({THRESHOLD})")
                ax2.set_xlabel("Churn Probability", color="#6b7280", fontsize=9)
                ax2.set_ylabel("Count", color="#6b7280", fontsize=9)
                ax2.set_title("Probability Distribution", color="#9ca3af", fontsize=10)
                ax2.tick_params(colors="#6b7280", labelsize=8)
                ax2.spines[:].set_visible(False)
                ax2.legend(fontsize=8, facecolor="#ffffff", edgecolor="#e5e7eb",
                           labelcolor="#374151")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

            with ch2:
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                fig3.patch.set_facecolor("#12151c")
                wedges, texts, autotexts = ax3.pie(
                    [n_high, n_low],
                    labels=["High Risk", "Low Risk"],
                    colors=["#e8623a", "#0d9488"],
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops=dict(edgecolor="#0e1117", linewidth=2)
                )
                for t in texts: t.set_color("#9ca3af"); t.set_fontsize(9)
                for at in autotexts: at.set_color("white"); at.set_fontsize(9); at.set_fontweight("bold")
                ax3.set_title("High vs Low Risk Split", color="#9ca3af", fontsize=10)
                fig3.patch.set_facecolor("#12151c")
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close()

            # ── Results table ──
            st.markdown('<div class="section-head">Full Results</div>', unsafe_allow_html=True)

            display_cols = list(df_raw.columns) + [
                "risk_label", "churn_probability_pct", "confidence_pct"
            ]
            st.dataframe(
                df_results[display_cols].style.apply(
                    lambda row: [
                        "background-color: rgba(232,98,58,0.08)" if row.get("risk_label") == "High Risk"
                        else "background-color: rgba(78,205,196,0.05)"
                        for _ in row
                    ], axis=1
                ),
                use_container_width=True,
                height=400
            )

            # ── Download ──
            st.markdown("<br>", unsafe_allow_html=True)
            csv_out = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Results CSV",
                csv_out,
                file_name="churnsense_predictions.csv",
                mime="text/csv",
                type="primary"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}\n\nMake sure your CSV matches the template format.")

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("⬆ Upload a CSV file to run batch predictions across all customers at once.")
        st.markdown("""
        **Required columns:**
        `contract_type` · `customer_tenure_months` · `monthly_spend_eur` · `satisfaction_score` ·
        `nps_score` · `num_support_calls_last_6m` · `return_rate_pct` · `num_purchases_last_3m` ·
        `payment_method` · `region` · `gender` · `has_premium_plan`
        """)
