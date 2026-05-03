# 🏦 Credit Risk Scoring System

An end-to-end machine learning system that predicts the probability of a customer defaulting on a loan — built to simulate real-world credit risk assessment used by banks like JPMC and HDFC.

## 🔗 Live Demo
👉 [Click here to try the app](https://credit-risk-scoring-system-prgrrftxraazqqtdofp9ft.streamlit.app/) ← we'll add this link after deployment

---

## 📌 Problem Statement
Banks lose crores every year from loan defaults. This system uses a customer's financial history to predict their probability of defaulting — enabling smarter, data-driven lending decisions.

---

## 🗂️ Project Highlights

| | |
|---|---|
| **Dataset** | 150,000 real customer records (Give Me Some Credit) |
| **Best Model** | Random Forest |
| **ROC-AUC Score** | 0.84 |
| **Key Challenge** | Severe class imbalance (6.7% defaulters) |
| **Solution** | SMOTE oversampling + threshold tuning |
| **Explainability** | SHAP values for every prediction |

---

## 🔍 What Makes This Project Unique

- ✅ **End-to-end pipeline** — from raw data to live deployed app
- ✅ **Real-world data cleaning** — handled missing values, outliers, invalid entries
- ✅ **Feature Engineering** — created 6 new features; `age_risk_score` became the #1 most important feature
- ✅ **Class Imbalance handled** — used SMOTE to fix 93:7 imbalance
- ✅ **3 Models compared** — Logistic Regression, Random Forest, XGBoost
- ✅ **Explainable AI** — SHAP shows WHY each customer is flagged as risky
- ✅ **Live Deployment** — Streamlit web app anyone can use

---

## 📊 EDA Findings

- Customers **under 30** have an 11.1% default rate — nearly 5x higher than 70+ age group
- Even **one instance** of 90+ days late payment increases default probability by 6x
- **Credit utilization above 75%** is a strong predictor of default
- `age_risk_score` (age × total late payments) became the strongest predictor

---

## 🤖 Model Comparison

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.8088 |
| XGBoost (Tuned) | 0.8193 |
| **Random Forest** ✅ | **0.8400** |

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, xgboost
- **Visualization** — matplotlib, seaborn
- **Explainability** — SHAP
- **Deployment** — Streamlit Cloud

---

## 📁 Project Structure

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author
Built as part of a data science portfolio project targeting roles in 
financial analytics and data science at product-based companies.
