import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── API config ─────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #f8f9fb; }
#MainMenu, footer, header { visibility: hidden; }

.app-header {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0 28px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 32px;
}
.app-logo {
    background: linear-gradient(135deg, #e8623a, #c0392b); color: white;
    font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500;
    width: 36px; height: 36px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
}
.app-title { font-size: 20px; font-weight: 600; color: #111827; letter-spacing: -0.3px; }
.app-title span { color: #e8623a; }
.app-pills { margin-left: auto; display: flex; gap: 10px; }
.pill {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 100px;
    padding: 4px 12px; font-family: 'DM Mono', monospace; font-size: 11px; color: #6b7280;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.pill strong { color: #0891b2; }
.api-status-ok {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf9; border: 1px solid #6ee7d4; color: #0f766e;
    border-radius: 100px; padding: 4px 12px; font-size: 11px; font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.api-status-err {
    display: inline-flex; align-items: center; gap: 6px;
    background: #fef2ee; border: 1px solid #fca58a; color: #c0392b;
    border-radius: 100px; padding: 4px 12px; font-size: 11px; font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.badge-high {
    display: inline-flex; align-items: center; gap: 6px;
    background: #fef2ee; border: 1px solid #fca58a; color: #c0392b;
    border-radius: 100px; padding: 6px 18px; font-size: 13px; font-weight: 600; letter-spacing: 0.5px;
}
.badge-low {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf9; border: 1px solid #6ee7d4; color: #0f766e;
    border-radius: 100px; padding: 6px 18px; font-size: 13px; font-weight: 600; letter-spacing: 0.5px;
}
.conf-wrap { margin: 20px 0; }
.conf-label {
    display: flex; justify-content: space-between; font-size: 12px; color: #6b7280;
    margin-bottom: 6px; font-family: 'DM Mono', monospace;
}
.conf-track { background: #e5e7eb; border-radius: 100px; height: 8px; overflow: hidden; }
.conf-fill-high { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #f59e0b, #e8623a); }
.conf-fill-low  { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #34d399, #0891b2); }
.threshold-note { font-size: 11px; color: #9ca3af; font-family: 'DM Mono', monospace; margin-top: 5px; }
.section-head {
    font-size: 10px; font-family: 'DM Mono', monospace; letter-spacing: 1.5px;
    text-transform: uppercase; color: #9ca3af; padding-bottom: 10px;
    border-bottom: 1px solid #e5e7eb; margin-bottom: 16px; margin-top: 24px;
}
.rec-item {
    display: flex; gap: 10px; align-items: flex-start; padding: 10px 0;
    border-bottom: 1px solid #f3f4f6; font-size: 13px; color: #374151; line-height: 1.5;
}
.rec-item:last-child { border-bottom: none; }
.explanation-box {
    background: #fff7f5; border: 1px solid #fde8e0; border-left: 3px solid #e8623a;
    border-radius: 0 10px 10px 0; padding: 14px 16px; font-size: 13px;
    color: #374151; line-height: 1.6; margin: 16px 0;
}
.explanation-box.safe {
    background: #f0fdf9; border: 1px solid #ccfbf0; border-left: 3px solid #0d9488;
}
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #111827 !important; font-size: 26px !important; font-weight: 600 !important; }
[data-testid="stTabs"] button { font-family: 'DM Sans', sans-serif; font-size: 14px; font-weight: 500; }
label { color: #374151 !important; font-size: 13px !important; font-weight: 500 !important; }
[data-testid="stCheckbox"] span { color: #374151 !important; font-size: 13px !important; }
hr { border-color: #e5e7eb; }
[data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; }
[data-testid="stAlert"] {
    background: #f0f9ff !important; border: 1px solid #bae6fd !important;
    color: #0369a1 !important; border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
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

# ── API helpers ────────────────────────────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def call_predict(payload: dict):
    """POST to /predict — all feature engineering happens in FastAPI."""
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

# ── Header ─────────────────────────────────────────────────────────────────────
api_ok = check_api_health()
model_info = get_model_info()
auc = model_info["metrics"]["auc_roc"] if model_info else "—"
recall = model_info["metrics"]["recall"] if model_info else "—"
threshold = model_info["threshold"] if model_info else 0.35

st.markdown(f"""
<div class="app-header">
  <div class="app-logo">CS</div>
  <div class="app-title">Churn<span>Sense</span></div>
  <div class="app-pills">
    <div class="pill">Model <strong>Logistic Regression</strong></div>
    <div class="pill">AUC-ROC <strong>{auc}</strong></div>
    <div class="pill">Recall <strong>{recall}</strong></div>
    <div class="pill">Threshold <strong>{threshold}</strong></div>
  </div>
  <div style="margin-left:16px">
    {"<span class='api-status-ok'>● API Connected</span>"
     if api_ok else
     "<span class='api-status-err'>● API Offline</span>"}
  </div>
</div>
""", unsafe_allow_html=True)

if not api_ok:
    st.error("⚠️ FastAPI backend is not running.\n\nOpen a terminal and run:\n```\nuvicorn main:app --reload\n```")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍  Single Customer Analysis",
    "📂  Batch CSV Upload",
    "🔌  API Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Customer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, _, col_result = st.columns([1.1, 0.08, 1])

    with col_form:
        st.markdown('<div class="section-head">Contract & Account</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: contract_type = st.selectbox("Contract Type *", ["Month-to-Month", "One Year", "Two Year"])
        with c2: tenure = st.number_input("Tenure (months) *", min_value=0, max_value=999, value=6, step=1)
        c3, c4 = st.columns(2)
        with c3: monthly_spend = st.number_input("Monthly Spend (€) *", min_value=0.0, value=35.0, step=1.0)
        with c4: payment_method = st.selectbox("Payment Method *", ["Credit Card", "Bank Transfer", "Digital Wallet", "Invoice"])

        st.markdown('<div class="section-head">Satisfaction & Engagement</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            sat_unknown = st.checkbox("Satisfaction unknown")
            satisfaction_score = st.number_input("Satisfaction Score (1–5)", min_value=1.0, max_value=5.0, value=3.5, step=0.1, disabled=sat_unknown)
        with c6:
            nps_unknown = st.checkbox("NPS unknown")
            nps_score = st.number_input("NPS Score (0–10)", min_value=0, max_value=10, value=5, step=1, disabled=nps_unknown)
        c7, c8 = st.columns(2)
        with c7: support_calls = st.number_input("Support Calls (last 6m) *", min_value=0, value=2, step=1)
        with c8: return_rate = st.number_input("Return Rate (%) *", min_value=0.0, value=5.0, step=0.5)

        st.markdown('<div class="section-head">Purchase Behaviour & Demographics</div>', unsafe_allow_html=True)
        c9, c10, c11 = st.columns(3)
        with c9: purchases = st.number_input("Purchases (last 3m) *", min_value=0, value=3, step=1)
        with c10: region = st.selectbox("Region *", ["West", "East", "North", "South", "Central"])
        with c11: gender = st.selectbox("Gender *", ["Male", "Female"])
        has_premium = st.selectbox("Premium Plan *", ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Analyse Churn Risk →", type="primary", use_container_width=True)

    with col_result:
        if run_btn:
            payload = {
                "contract_type": contract_type,
                "customer_tenure_months": float(tenure),
                "monthly_spend_eur": float(monthly_spend),
                "satisfaction_score": None if sat_unknown else float(satisfaction_score),
                "nps_score": None if nps_unknown else int(nps_score),
                "num_support_calls_last_6m": float(support_calls),
                "return_rate_pct": float(return_rate),
                "num_purchases_last_3m": float(purchases),
                "payment_method": payment_method,
                "region": region,
                "gender": gender,
                "has_premium_plan": 1 if has_premium == "Yes" else 0
            }

            with st.spinner("Calling /predict endpoint..."):
                try:
                    data = call_predict(payload)

                    proba      = data["churn_probability"]
                    prediction = data["prediction"]
                    confidence = data.get("confidence", max(proba, 1 - proba))
                    shap_vals  = data.get("all_shap_values", {})
                    top_risk   = data.get("top_risk_factors", [])
                    top_prot   = data.get("top_protective_factors", [])
                    pct        = round(proba * 100, 1)
                    conf_pct   = round(confidence * 100, 1)
                    is_high    = prediction == 1
                    fill_cls   = "conf-fill-high" if is_high else "conf-fill-low"
                    badge_cls  = "badge-high" if is_high else "badge-low"
                    badge_txt  = "⚠ HIGH RISK" if is_high else "✓ LOW RISK"

                    st.markdown(f'<div class="{badge_cls}">{badge_txt}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("Churn Probability", f"{pct}%")
                    with m2: st.metric("Model Confidence", f"{conf_pct}%")
                    with m3: st.metric("Decision Threshold", f"{int(threshold*100)}%")

                    st.markdown(f"""
                    <div class="conf-wrap">
                      <div class="conf-label"><span>Churn Probability</span><span>{pct}%</span></div>
                      <div class="conf-track">
                        <div class="{fill_cls}" style="width:{pct}%"></div>
                      </div>
                      <div class="threshold-note">▲ Decision threshold at {int(threshold*100)}% — above this = High Risk</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Plain English
                    st.markdown('<div class="section-head">Why This Prediction</div>', unsafe_allow_html=True)
                    risk_names = [FEATURE_LABELS.get(f["feature"], f["feature"]) for f in top_risk[:2]]
                    prot_name  = FEATURE_LABELS.get(top_prot[0]["feature"], top_prot[0]["feature"]) if top_prot else None
                    risk_str   = " and ".join([f"**{n}**" for n in risk_names])
                    box_cls    = "explanation-box" if is_high else "explanation-box safe"

                    if is_high:
                        explanation = (f"This customer is flagged as **High Risk** ({pct}% churn probability). "
                                       f"Primary drivers: {risk_str}. "
                                       + (f"**{prot_name}** partially reduces risk but not enough to change the outcome." if prot_name else ""))
                    else:
                        explanation = (f"This customer is **Low Risk** ({pct}% churn probability). "
                                       + (f"Key protective factor: **{prot_name}**. " if prot_name else "")
                                       + (f"Minor risk signals from {risk_str} are present but insufficient." if risk_names else ""))

                    st.markdown(f'<div class="{box_cls}">{explanation}</div>', unsafe_allow_html=True)

                    # SHAP chart
                    if shap_vals:
                        st.markdown('<div class="section-head">Feature Contributions (SHAP)</div>', unsafe_allow_html=True)
                        top_feats = sorted(
                            [(FEATURE_LABELS.get(k, k), v) for k, v in shap_vals.items() if k in FEATURE_LABELS],
                            key=lambda x: abs(x[1]), reverse=True
                        )[:8]
                        labels = [f[0] for f in top_feats]
                        values = [f[1] for f in top_feats]
                        colors_bar = ["#e8623a" if v > 0 else "#0d9488" for v in values]

                        fig, ax = plt.subplots(figsize=(6, 3.5))
                        fig.patch.set_facecolor("#ffffff")
                        ax.set_facecolor("#f8f9fb")
                        ax.barh(labels[::-1], values[::-1], color=colors_bar[::-1], height=0.55, edgecolor="none")
                        ax.axvline(0, color="#e5e7eb", linewidth=1)
                        ax.set_xlabel("SHAP Value", color="#6b7280", fontsize=9, labelpad=8)
                        ax.tick_params(colors="#374151", labelsize=9, length=0)
                        ax.spines[:].set_visible(False)
                        pos_patch = mpatches.Patch(color="#e8623a", label="Increases churn risk")
                        neg_patch = mpatches.Patch(color="#0d9488", label="Decreases churn risk")
                        ax.legend(handles=[pos_patch, neg_patch], fontsize=8,
                                  facecolor="#ffffff", edgecolor="#e5e7eb", labelcolor="#374151", loc="lower right")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                    # Recommendations
                    st.markdown('<div class="section-head">Recommended Actions</div>', unsafe_allow_html=True)
                    for icon, text in (RECOMMENDATIONS_HIGH if is_high else RECOMMENDATIONS_LOW):
                        st.markdown(f'<div class="rec-item"><span>{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)

                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ API Error {e.response.status_code}: {e.response.json().get('detail', str(e))}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach FastAPI. Make sure `uvicorn main:app --reload` is running.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("👈  Fill in the customer details and click **Analyse Churn Risk** to see results.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">Upload Customer CSV</div>', unsafe_allow_html=True)

    sample = pd.DataFrame([
        ["Month-to-Month", 3, 35.0, 2.5, 4, 5, 12.0, 1, "Credit Card", "West", "Male", 0],
        ["One Year", 24, 55.0, 4.2, 7, 1, 3.0, 8, "Bank Transfer", "North", "Female", 1],
        ["Two Year", 60, 80.0, 4.8, 9, 0, 1.0, 15, "Digital Wallet", "South", "Male", 1],
    ], columns=["contract_type","customer_tenure_months","monthly_spend_eur",
                "satisfaction_score","nps_score","num_support_calls_last_6m",
                "return_rate_pct","num_purchases_last_3m","payment_method","region","gender","has_premium_plan"])

    st.download_button("⬇ Download CSV Template", sample.to_csv(index=False).encode("utf-8"),
                       file_name="churnsense_template.csv", mime="text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅ {len(df_raw):,} customers loaded — sending to /predict...")

        results = []
        progress = st.progress(0)
        for i, row in df_raw.iterrows():
            try:
                payload = {
                    "contract_type": row["contract_type"],
                    "customer_tenure_months": float(row["customer_tenure_months"]),
                    "monthly_spend_eur": float(row["monthly_spend_eur"]),
                    "satisfaction_score": None if pd.isna(row.get("satisfaction_score")) else float(row["satisfaction_score"]),
                    "nps_score": None if pd.isna(row.get("nps_score")) else int(row["nps_score"]),
                    "num_support_calls_last_6m": float(row["num_support_calls_last_6m"]),
                    "return_rate_pct": float(row["return_rate_pct"]),
                    "num_purchases_last_3m": float(row["num_purchases_last_3m"]),
                    "payment_method": row["payment_method"],
                    "region": row["region"],
                    "gender": row["gender"],
                    "has_premium_plan": int(row.get("has_premium_plan", 0))
                }
                data = call_predict(payload)
                proba = data["churn_probability"]
                conf  = data.get("confidence", max(proba, 1 - proba))
                results.append({
                    "risk_label": data["risk_label"],
                    "churn_probability_pct": f"{round(proba*100,1)}%",
                    "confidence_pct": f"{round(conf*100,1)}%",
                    "prediction": data["prediction"],
                    "churn_probability": round(proba, 4),
                    "confidence": round(conf, 4),
                })
            except Exception as e:
                results.append({"risk_label": f"Error: {e}", "churn_probability_pct": "—",
                                 "confidence_pct": "—", "prediction": None,
                                 "churn_probability": None, "confidence": None})
            progress.progress((i + 1) / len(df_raw))

        progress.empty()
        df_out = pd.concat([df_raw.reset_index(drop=True), pd.DataFrame(results)], axis=1)

        # Summary
        st.markdown('<div class="section-head">Batch Summary</div>', unsafe_allow_html=True)
        valid  = df_out[df_out["prediction"].notna()]
        n_tot  = len(valid)
        n_high = int(valid["prediction"].sum())
        n_low  = n_tot - n_high

        b1, b2, b3, b4, b5 = st.columns(5)
        with b1: st.metric("Total Customers", f"{n_tot:,}")
        with b2: st.metric("High Risk", f"{n_high:,}", delta=f"{round(n_high/n_tot*100,1)}%", delta_color="inverse")
        with b3: st.metric("Low Risk",  f"{n_low:,}",  delta=f"{round(n_low/n_tot*100,1)}%")
        with b4: st.metric("Avg Churn Probability", f"{round(valid['churn_probability'].mean()*100,1)}%")
        with b5: st.metric("Avg Confidence", f"{round(valid['confidence'].mean()*100,1)}%")

        # Charts
        st.markdown('<div class="section-head">Risk Distribution</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            fig2.patch.set_facecolor("#ffffff"); ax2.set_facecolor("#f8f9fb")
            ax2.hist(valid["churn_probability"], bins=30, color="#e8623a", alpha=0.8, edgecolor="none")
            ax2.axvline(threshold, color="#0d9488", linewidth=1.5, linestyle="--", label=f"Threshold ({threshold})")
            ax2.set_xlabel("Churn Probability", color="#6b7280", fontsize=9)
            ax2.set_ylabel("Count", color="#6b7280", fontsize=9)
            ax2.set_title("Probability Distribution", color="#374151", fontsize=10)
            ax2.tick_params(colors="#6b7280", labelsize=8, length=0); ax2.spines[:].set_visible(False)
            ax2.legend(fontsize=8, facecolor="#ffffff", edgecolor="#e5e7eb", labelcolor="#374151")
            plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()
        with ch2:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            fig3.patch.set_facecolor("#ffffff")
            wedges, texts, autotexts = ax3.pie([n_high, n_low], labels=["High Risk", "Low Risk"],
                colors=["#e8623a", "#0d9488"], autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="#ffffff", linewidth=2))
            for t in texts: t.set_color("#374151"); t.set_fontsize(9)
            for at in autotexts: at.set_color("white"); at.set_fontsize(9); at.set_fontweight("bold")
            ax3.set_title("High vs Low Risk", color="#374151", fontsize=10)
            plt.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close()

        st.markdown('<div class="section-head">Full Results</div>', unsafe_allow_html=True)
        display_cols = list(df_raw.columns) + ["risk_label", "churn_probability_pct", "confidence_pct"]
        st.dataframe(df_out[display_cols], use_container_width=True, height=400)
        st.download_button("⬇ Download Results CSV", df_out.to_csv(index=False).encode("utf-8"),
                           file_name="churnsense_predictions.csv", mime="text/csv", type="primary")
    else:
        st.info("⬆ Upload a CSV to run batch predictions across all customers.")
        st.markdown("""**Required columns:** `contract_type` · `customer_tenure_months` · `monthly_spend_eur` ·
        `satisfaction_score` · `nps_score` · `num_support_calls_last_6m` · `return_rate_pct` ·
        `num_purchases_last_3m` · `payment_method` · `region` · `gender` · `has_premium_plan`""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — API Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">API Connection</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Base URL**")
        st.code(API_URL, language="bash")
        st.markdown("**Endpoints**")
        st.code("GET  /health      → API status\nGET  /model-info  → metrics & features\nPOST /predict     → churn prediction", language="bash")
        st.markdown("**Example Request Body**")
        st.code("""{
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
}""", language="json")

    with c2:
        st.markdown("**Live /health Response**")
        try:
            st.code(str(requests.get(f"{API_URL}/health", timeout=3).json()), language="json")
        except:
            st.error("API not reachable")

        st.markdown("**Live /model-info Response**")
        if model_info:
            st.code(str(model_info), language="json")
        else:
            st.error("Could not fetch model info")

        st.markdown("**Interactive API Docs**")
        st.markdown(f"[Open FastAPI Swagger UI →]({API_URL}/docs)")
        st.caption("Full interactive documentation — test the /predict endpoint live from the browser.")
