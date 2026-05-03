
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL AND SCALER
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🏦 Credit Risk Assessment System")
st.markdown("### Predict customer default probability using Machine Learning")
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR — CUSTOMER INPUT
# ─────────────────────────────────────────
st.sidebar.header("📋 Enter Customer Details")
st.sidebar.markdown("Fill in the customer information below:")

age = st.sidebar.slider(
    "Age", 
    min_value=18, max_value=100, value=35,
    help="Customer age in years"
)

monthly_income = st.sidebar.number_input(
    "Monthly Income ($)",
    min_value=0, max_value=25000, value=5000, step=100,
    help="Customer monthly income"
)

revolving_utilization = st.sidebar.slider(
    "Credit Utilization Rate",
    min_value=0.0, max_value=1.0, value=0.3, step=0.01,
    help="How much of credit limit is being used (0 = 0%, 1 = 100%)"
)

debt_ratio = st.sidebar.slider(
    "Debt Ratio",
    min_value=0.0, max_value=1.0, value=0.3, step=0.01,
    help="Monthly debt payments divided by monthly income"
)

num_open_loans = st.sidebar.slider(
    "Number of Open Loans & Credit Lines",
    min_value=0, max_value=30, value=5,
    help="Total number of open credit lines and loans"
)

num_real_estate_loans = st.sidebar.slider(
    "Number of Real Estate Loans",
    min_value=0, max_value=10, value=1,
    help="Number of mortgage and real estate loans"
)

num_dependents = st.sidebar.slider(
    "Number of Dependents",
    min_value=0, max_value=10, value=0,
    help="Number of dependents (family members relying on customer)"
)

st.sidebar.markdown("### Late Payment History")

late_30_59 = st.sidebar.slider(
    "Times 30-59 Days Late",
    min_value=0, max_value=10, value=0,
    help="Number of times 30-59 days past due"
)

late_60_89 = st.sidebar.slider(
    "Times 60-89 Days Late",
    min_value=0, max_value=10, value=0,
    help="Number of times 60-89 days past due"
)

late_90 = st.sidebar.slider(
    "Times 90+ Days Late",
    min_value=0, max_value=10, value=0,
    help="Number of times 90+ days past due"
)

# ─────────────────────────────────────────
# FEATURE ENGINEERING (automatic)
# ─────────────────────────────────────────
def engineer_features(age, monthly_income, revolving_utilization,
                       debt_ratio, num_open_loans, num_real_estate_loans,
                       num_dependents, late_30_59, late_60_89, late_90):

    total_late = late_30_59 + late_60_89 + late_90
    debt_to_income = debt_ratio / (monthly_income + 1)
    income_per_dep = monthly_income / (num_dependents + 1)
    high_util = 1 if revolving_utilization > 0.75 else 0
    age_risk = age * total_late
    loans_to_inc = num_open_loans / (monthly_income + 1)

    features = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": num_open_loans,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": num_real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": num_dependents,
        "total_late_payments": total_late,
        "debt_to_income": debt_to_income,
        "income_per_dependent": income_per_dep,
        "high_utilization": high_util,
        "age_risk_score": age_risk,
        "loans_to_income": loans_to_inc
    }

    return pd.DataFrame([features])

# ─────────────────────────────────────────
# PREDICTION BUTTON
# ─────────────────────────────────────────
st.markdown("## 🎯 Risk Assessment Result")

if st.button("🔍 Predict Default Risk", type="primary", use_container_width=True):

    # Build feature dataframe
    input_df = engineer_features(
        age, monthly_income, revolving_utilization,
        debt_ratio, num_open_loans, num_real_estate_loans,
        num_dependents, late_30_59, late_60_89, late_90
    )

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    prob_pct = prob * 100

    # Risk category
    if prob_pct < 20:
        risk_level = "LOW RISK"
        color = "🟢"
        recommendation = "APPROVE"
        rec_color = "green"
    elif prob_pct < 50:
        risk_level = "MEDIUM RISK"
        color = "🟡"
        recommendation = "MANUAL REVIEW"
        rec_color = "orange"
    else:
        risk_level = "HIGH RISK"
        color = "🔴"
        recommendation = "REJECT"
        rec_color = "red"

    # ── Results Layout ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Default Probability",
            value=f"{prob_pct:.1f}%"
        )

    with col2:
        st.metric(
            label="Risk Level",
            value=f"{color} {risk_level}"
        )

    with col3:
        st.metric(
            label="Recommendation",
            value=recommendation
        )

    # ── Risk Bar ──
    st.markdown("### Risk Probability Gauge")
    st.progress(int(prob_pct))

    # ── Customer Summary ──
    st.markdown("---")
    st.markdown("### 📊 Customer Profile Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Personal Details**")
        st.write(f"- Age: {age} years")
        st.write(f"- Monthly Income: ${monthly_income:,}")
        st.write(f"- Number of Dependents: {num_dependents}")
        st.write(f"- Real Estate Loans: {num_real_estate_loans}")

    with col2:
        st.markdown("**Financial Behavior**")
        st.write(f"- Credit Utilization: {revolving_utilization*100:.0f}%")
        st.write(f"- Debt Ratio: {debt_ratio:.2f}")
        st.write(f"- Total Late Payments: {late_30_59 + late_60_89 + late_90}")
        st.write(f"- Open Loans & Credit Lines: {num_open_loans}")

    # ── Key Risk Factors ──
    st.markdown("---")
    st.markdown("### ⚠️ Key Risk Factors")

    total_late = late_30_59 + late_60_89 + late_90

    factors = []
    if revolving_utilization > 0.75:
        factors.append(f"🔴 Very high credit utilization ({revolving_utilization*100:.0f}%)")
    if total_late > 0:
        factors.append(f"🔴 History of late payments ({total_late} times total)")
    if age < 30:
        factors.append(f"🟡 Young age increases risk ({age} years)")
    if monthly_income < 3000:
        factors.append(f"🟡 Low monthly income (${monthly_income:,})")
    if debt_ratio > 0.5:
        factors.append(f"🟡 High debt ratio ({debt_ratio:.2f})")

    if factors:
        for f in factors:
            st.write(f)
    else:
        st.write("🟢 No major risk factors identified — customer appears financially stable")

    # ── Disclaimer ──
    st.markdown("---")
    st.caption(
        "⚠️ This prediction is generated by a machine learning model trained on historical data. "
        "It is intended to assist — not replace — human judgment in credit decisions."
    )

else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Default Risk**")

    # Show sample stats while waiting
    st.markdown("### 📈 About This Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Random Forest")
    col2.metric("ROC-AUC Score", "0.84")
    col3.metric("Training Data", "149,735 customers")
    col4.metric("Features Used", "16")
