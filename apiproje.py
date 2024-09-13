import numpy as np
import pandas as pd
import joblib
import streamlit as st

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load models
loaded_rf_model = joblib.load("rf_model.pkl")
loaded_xgboost_model = joblib.load("xgboost_model.pkl")
loaded_lgbm_model = joblib.load("lgbm_model.pkl")
loaded_catboost_model = joblib.load("catboost_model.pkl")

# Load the column names used during training
with open("training_columns.pkl", "rb") as f:
    training_columns = joblib.load(f)

# Define the input form
st.title("Customer Churn Prediction")

# Collect input data
customer_id = st.text_input("Customer ID")
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

# Create a DataFrame from the input data
data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Calculate custom features
data["NEW_TENURE_YEAR"] = data["tenure"].apply(lambda x: "0-1 Year" if x <= 12 else "1-2 Year" if x <= 24 else "2-3 Year" if x <= 36 else "3-4 Year" if x <= 48 else "4-5 Year" if x <= 60 else "5-6 Year")
data["NEW_Engaged"] = data["Contract"].apply(lambda x: 1 if x != "Month-to-month" else 0)
data["NEW_noProt"] = data[["OnlineBackup", "DeviceProtection", "TechSupport"]].apply(lambda x: 1 if (x == "No").sum() > 1 else 0, axis=1)
data["NEW_Young_Not_Engaged"] = data.apply(lambda x: 1 if (x["NEW_Engaged"] == 0 and x["SeniorCitizen"] == 0) else 0, axis=1)
data["NEW_TotalServices"] = data[["PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]].apply(lambda x: (x == "Yes").sum(), axis=1)
data["NEW_FLAG_ANY_STREAMING"] = data[["StreamingTV", "StreamingMovies"]].apply(lambda x: 1 if (x == "Yes").any() else 0, axis=1)
data["NEW_FLAG_AutoPayment"] = data["PaymentMethod"].apply(lambda x: 1 if "automatic" in x else 0)
data["NEW_AVG_Charges"] = data["TotalCharges"] / data["tenure"]
data["NEW_Increase"] = data["NEW_AVG_Charges"] / data["MonthlyCharges"]
data["NEW_AVG_Service_Fee"] = data["NEW_AVG_Charges"] - data["MonthlyCharges"]

# One-Hot Encode categorical variables
data = pd.get_dummies(data)

# Align the input data with the training columns
data = data.reindex(columns=training_columns, fill_value=0)

# Predict using each model
if st.button("Predict Churn"):
    prediction_rf = loaded_rf_model.predict(data)
    prediction_xgboost = loaded_xgboost_model.predict(data)
    prediction_lgbm = loaded_lgbm_model.predict(data)
    prediction_catboost = loaded_catboost_model.predict(data)

    st.write(f"Random Forest Prediction: {'Churn' if prediction_rf[0] else 'No Churn'}")
    st.write(f"XGBoost Prediction: {'Churn' if prediction_xgboost[0] else 'No Churn'}")
    st.write(f"LGBM Prediction: {'Churn' if prediction_lgbm[0] else 'No Churn'}")
    st.write(f"CatBoost Prediction: {'Churn' if prediction_catboost[0] else 'No Churn'}")
