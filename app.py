import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    return model

model = load_model()

# OPTIMAL THRESHOLD (tuned from F1 optimization)
THRESHOLD = 0.68

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🏦 Credit Risk Assessment System")
st.markdown("### Predict customer default probability using Machine Learning")
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────
st.sidebar.header("📋 Enter Customer Details")

age = st.sidebar.slider("Age", 18, 100, 35)
monthly_income = st.sidebar.number_input(
    "Monthly Income ($)", 0, 25000, 5000, 100)
revolving_utilization = st.sidebar.slider(
    "Credit Utilization Rate", 0.0, 1.0, 0.3, 0.01)
debt_ratio = st.sidebar.slider(
    "Debt Ratio", 0.0, 1.0, 0.3, 0.01)
num_open_loans = st.sidebar.slider(
    "Number of Open Loans & Credit Lines", 0, 30, 5)
num_real_estate_loans = st.sidebar.slider(
    "Number of Real Estate Loans", 0, 10, 1)
num_dependents = st.sidebar.slider(
    "Number of Dependents", 0, 10, 0)

st.sidebar.markdown("### Late Payment History")
late_30_59 = st.sidebar.slider("Times 30-59 Days Late", 0, 10, 0)
late_60_89 = st.sidebar.slider("Times 60-89 Days Late", 0, 10, 0)
late_90 = st.sidebar.slider("Times 90+ Days Late", 0, 10, 0)

# ─────────────────────────────────────────
# FEATURE ENGINEERING
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

    return pd.DataFrame([{
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
    }])

# ─────────────────────────────────────────
# LLM RISK EXPLANATION
# ─────────────────────────────────────────
def get_llm_explanation(customer_data, prob, risk_level):
    """Use Claude API to generate natural language risk explanation"""
    
    prompt = f"""You are a credit risk analyst at a bank. 
A customer has been assessed with the following profile:

Age: {customer_data['age']} years
Monthly Income: ${customer_data['MonthlyIncome']:,}
Credit Utilization: {customer_data['RevolvingUtilizationOfUnsecuredLines']*100:.0f}%
Debt Ratio: {customer_data['DebtRatio']:.2f}
Total Late Payments: {customer_data['total_late_payments']}
Times 90+ Days Late: {customer_data['NumberOfTimes90DaysLate']}
Number of Dependents: {customer_data['NumberOfDependents']}
Open Credit Lines: {customer_data['NumberOfOpenCreditLinesAndLoans']}

The ML model has assigned:
Default Probability: {prob*100:.1f}%
Risk Level: {risk_level}

Write a 3-4 sentence explanation of why this customer 
received this risk assessment. Be specific about which 
factors are most concerning or reassuring. 
Write in plain English that a bank employee would 
tell the customer. Do not use technical ML terms.
Do not start with "The customer" — vary your opening."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        data = response.json()
        return data["content"][0]["text"]
    except:
        return None

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
st.markdown("## 🎯 Risk Assessment Result")

if st.button("🔍 Predict Default Risk", 
             type="primary", use_container_width=True):

    input_df = engineer_features(
        age, monthly_income, revolving_utilization,
        debt_ratio, num_open_loans, num_real_estate_loans,
        num_dependents, late_30_59, late_60_89, late_90
    )

    prob = model.predict_proba(input_df)[0][1]
    prob_pct = prob * 100

    # Use optimized threshold
    if prob >= THRESHOLD:
        risk_level = "HIGH RISK"
        color = "🔴"
        recommendation = "REJECT"
    elif prob >= 0.35:
        risk_level = "MEDIUM RISK"
        color = "🟡"
        recommendation = "MANUAL REVIEW"
    else:
        risk_level = "LOW RISK"
        color = "🟢"
        recommendation = "APPROVE"

    # Results
    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{prob_pct:.1f}%")
    col2.metric("Risk Level", f"{color} {risk_level}")
    col3.metric("Recommendation", recommendation)

    st.markdown("### Risk Probability Gauge")
    st.progress(min(int(prob_pct), 100))

    # ── LLM Explanation ──
    st.markdown("---")
    st.markdown("### 🤖 AI Risk Analysis")

    with st.spinner("Generating risk analysis..."):
        customer_dict = input_df.iloc[0].to_dict()
        explanation = get_llm_explanation(
            customer_dict, prob, risk_level
        )

    if explanation:
        st.info(explanation)
    else:
        # Fallback to rule-based if API fails
        total_late = late_30_59 + late_60_89 + late_90
        factors = []
        if revolving_utilization > 0.75:
            factors.append(f"Very high credit utilization ({revolving_utilization*100:.0f}%)")
        if total_late > 0:
            factors.append(f"History of late payments ({total_late} times)")
        if age < 30:
            factors.append(f"Young age increases statistical risk")
        if monthly_income < 3000:
            factors.append(f"Low monthly income (${monthly_income:,})")

        if factors:
            st.warning("Key Risk Factors:\n" + 
                      "\n".join([f"• {f}" for f in factors]))
        else:
            st.success("No major risk factors — customer appears financially stable")

    # ── Customer Summary ──
    st.markdown("---")
    st.markdown("### 📊 Customer Profile Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Personal Details**")
        st.write(f"- Age: {age} years")
        st.write(f"- Monthly Income: ${monthly_income:,}")
        st.write(f"- Dependents: {num_dependents}")
        st.write(f"- Real Estate Loans: {num_real_estate_loans}")

    with col2:
        st.markdown("**Financial Behavior**")
        st.write(f"- Credit Utilization: {revolving_utilization*100:.0f}%")
        st.write(f"- Debt Ratio: {debt_ratio:.2f}")
        st.write(f"- Total Late Payments: {late_30_59+late_60_89+late_90}")
        st.write(f"- Open Credit Lines: {num_open_loans}")

    st.markdown("---")
    st.caption(
        "⚠️ This prediction is generated by a machine learning model. "
        "Optimal threshold tuned at 0.68 using F1 score optimization. "
        "Evaluated using PR-AUC (0.32) for honest imbalanced class assessment."
    )

else:
    st.info("👈 Fill in customer details and click Predict Default Risk")
    st.markdown("### 📈 About This Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Random Forest")
    col2.metric("ROC-AUC", "0.84")
    col3.metric("PR-AUC", "0.32")
    col4.metric("Threshold", "0.68")
