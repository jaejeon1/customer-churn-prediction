from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import json
import numpy as np
import pandas as pd
from typing import Optional

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability using Logistic Regression. AUC-ROC: 0.900, Recall: 0.940",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
model = joblib.load("churn_model.pkl")
explainer = joblib.load("shap_explainer.pkl")
with open("feature_names.json") as f:
    FEATURE_NAMES = json.load(f)
with open("model_config.json") as f:
    CONFIG = json.load(f)

THRESHOLD = CONFIG["threshold"]  # 0.35


# ── Input schema (raw business inputs only) ──────────────────────────────────
class CustomerInput(BaseModel):
    contract_type: str = Field(..., description="Month-to-Month | One Year | Two Year")
    customer_tenure_months: float = Field(..., ge=0, le=999, description="Months as customer")
    monthly_spend_eur: float = Field(..., ge=0, description="Monthly spend in EUR")
    satisfaction_score: Optional[float] = Field(None, ge=1.0, le=5.0, description="1–5 scale, null if unknown")
    nps_score: Optional[float] = Field(None, ge=0, le=10, description="0–10 scale, null if unknown")
    num_support_calls_last_6m: float = Field(..., ge=0, description="Support calls in last 6 months")
    return_rate_pct: float = Field(..., ge=0, description="Return rate percentage")
    num_purchases_last_3m: float = Field(..., ge=0, description="Purchases in last 3 months")
    payment_method: str = Field(..., description="Credit Card | Bank Transfer | Digital Wallet | Invoice")
    region: str = Field(..., description="West | East | North | South | Central")
    gender: str = Field(..., description="Male | Female")
    has_premium_plan: int = Field(..., ge=0, le=1, description="1 if premium plan, 0 otherwise")

    @validator("contract_type")
    def validate_contract(cls, v):
        valid = ["Month-to-Month", "One Year", "Two Year"]
        if v not in valid:
            raise ValueError(f"contract_type must be one of {valid}")
        return v

    @validator("payment_method")
    def validate_payment(cls, v):
        valid = ["Credit Card", "Bank Transfer", "Digital Wallet", "Invoice"]
        if v not in valid:
            raise ValueError(f"payment_method must be one of {valid}")
        return v

    @validator("region")
    def validate_region(cls, v):
        valid = ["West", "East", "North", "South", "Central"]
        if v not in valid:
            raise ValueError(f"region must be one of {valid}")
        return v

    @validator("gender")
    def validate_gender(cls, v):
        valid = ["Male", "Female"]
        if v not in valid:
            raise ValueError(f"gender must be one of {valid}")
        return v


# ── Feature engineering (mirrors notebook exactly) ───────────────────────────
def engineer_features(data: CustomerInput) -> pd.DataFrame:
    # Handle missing satisfaction/nps
    sat_missing = 1 if data.satisfaction_score is None else 0
    nps_missing = 1 if data.nps_score is None else 0

    satisfaction_score = data.satisfaction_score if data.satisfaction_score is not None else 3.5  # median
    nps_score = data.nps_score if data.nps_score is not None else 5.0  # median

    # Encoded contract type
    contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    contract_encoded = contract_map[data.contract_type]

    # Tenure risk score
    t = data.customer_tenure_months
    if t < 6:
        tenure_risk = 4
    elif t < 12:
        tenure_risk = 3
    elif t < 24:
        tenure_risk = 2
    elif t < 60:
        tenure_risk = 1
    else:
        tenure_risk = 0

    # Engineered features
    new_mtm_flag = int(t < 6 and data.contract_type == "Month-to-Month")
    support_call_rate = data.num_support_calls_last_6m / (t + 1)
    frustration_index = data.num_support_calls_last_6m * (5 - satisfaction_score)
    log_monthly_spend = np.log1p(data.monthly_spend_eur)
    log_return_rate = np.log1p(data.return_rate_pct)

    # One-hot encode payment method
    pm_credit_card = int(data.payment_method == "Credit Card")
    pm_digital_wallet = int(data.payment_method == "Digital Wallet")
    pm_invoice = int(data.payment_method == "Invoice")

    # One-hot encode region
    region_east = int(data.region == "East")
    region_north = int(data.region == "North")
    region_south = int(data.region == "South")
    region_west = int(data.region == "West")

    # Gender
    gender_male = int(data.gender == "Male")

    row = {
        "contract_type_encoded": contract_encoded,
        "tenure_risk_score": tenure_risk,
        "customer_tenure_months": t,
        "log_monthly_spend": log_monthly_spend,
        "satisfaction_score": satisfaction_score,
        "nps_score": nps_score,
        "num_support_calls_last_6m": data.num_support_calls_last_6m,
        "support_call_rate": support_call_rate,
        "frustration_index": frustration_index,
        "new_mtm_flag": new_mtm_flag,
        "log_return_rate": log_return_rate,
        "num_purchases_last_3m": data.num_purchases_last_3m,
        "satisfaction_score_missing": sat_missing,
        "nps_score_missing": nps_missing,
        "payment_method_Credit Card": pm_credit_card,
        "payment_method_Digital Wallet": pm_digital_wallet,
        "payment_method_Invoice": pm_invoice,
        "region_East": region_east,
        "region_North": region_north,
        "region_South": region_south,
        "region_West": region_west,
        "gender_Male": gender_male,
    }

    return pd.DataFrame([row])[FEATURE_NAMES]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health", "/model-info"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "model_type": CONFIG["model_type"],
        "threshold": THRESHOLD,
        "metrics": {
            "auc_roc": CONFIG["auc_roc"],
            "recall": CONFIG["recall"],
            "precision": CONFIG["precision"],
            "f1": CONFIG["f1"]
        },
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        # Engineer features
        X = engineer_features(customer)

        # Predict probability
        proba = model.predict_proba(X)[0, 1]
        prediction = int(proba >= THRESHOLD)
        risk_label = "High Risk" if prediction == 1 else "Low Risk"

        # SHAP explanation
        X_scaled = model.named_steps["scaler"].transform(X)
        shap_vals = explainer(X_scaled)
        shap_contributions = dict(zip(FEATURE_NAMES, shap_vals.values[0].tolist()))

        # Top 3 risk factors (positive SHAP = increases churn risk)
        top_factors = sorted(
            shap_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # Top 3 protective factors (negative SHAP = reduces churn risk)
        protective_factors = sorted(
            shap_contributions.items(),
            key=lambda x: x[1]
        )[:3]

        return {
            "prediction": prediction,
            "risk_label": risk_label,
            "churn_probability": round(float(proba), 4),
            "threshold_used": THRESHOLD,
            "top_risk_factors": [
                {"feature": k, "shap_value": round(v, 4)}
                for k, v in top_factors if v > 0
            ],
            "top_protective_factors": [
                {"feature": k, "shap_value": round(v, 4)}
                for k, v in protective_factors if v < 0
            ],
            "all_shap_values": {k: round(v, 4) for k, v in shap_contributions.items()},
            "model_version": "1.0.0"
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
