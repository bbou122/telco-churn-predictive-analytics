# streamlit_app.py 
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("**XGBoost • 0.86 AUC • Instant predictions**")

# Load data + train model once (cached)
@st.cache_resource
def get_model():
    df = pd.read_csv("data/raw/telco_churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # minimal feature engineering (same as notebook)
    df['Tenure_Group'] = pd.cut(df['tenure'], [0,12,24,48,60,100], labels=['0-1yr','1-2yr','2-4yr','4-5yr','5+yr'])
    df['Num_Services'] = df[['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']].apply(lambda x: x.str.count('Yes')).sum(axis=1)
    df['Month_to_Month'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Fiber_Optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

    X = pd.get_dummies(df.drop(['customerID','Churn'], axis=1), drop_first=True)
    y = (df['Churn'] == 'Yes').astype(int)

    model = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.02,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01,
                              random_state=42, eval_metric='auc')
    model.fit(X, y)
    return model, X.columns

model, cols = get_model()

# Sidebar upload
uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv")

if uploaded:
    new = pd.read_csv(uploaded)
    orig = new.copy()

    # same preprocessing
    new['TotalCharges'] = pd.to_numeric(new['TotalCharges'], errors='coerce')
    new['TotalCharges'].fillna(new['TotalCharges'].median(), inplace=True)
    new['Tenure_Group'] = pd.cut(new['tenure'], [0,12,24,48,60,100], labels=['0-1yr','1-2yr','2-4yr','4-5yr','5+yr'])
    new['Num_Services'] = new[['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']].apply(lambda x: x.str.count('Yes')).sum(axis=1)
    new['Month_to_Month'] = (new['Contract'] == 'Month-to-month').astype(int)
    new['Fiber_Optic'] = (new['InternetService'] == 'Fiber optic').astype(int)
    new['Electronic_Check'] = (new['PaymentMethod'] == 'Electronic check').astype(int)

    X_new = pd.get_dummies(new.drop(columns=['customerID'], errors='ignore'), drop_first=True)
    X_new = X_new.reindex(columns=cols, fill_value=0)

    probs = model.predict_proba(X_new)[:,1]
    pred = (probs >= 0.5).astype(int)

    result = orig[['customerID']].copy()
    result['Churn_Probability'] = np.round(probs, 3)
    result['Prediction'] = np.where(pred==1, "Will Churn", "Will Stay")
    result = result.sort_values('Churn_Probability', ascending=False)

    st.success(f"Scored {len(result)} customers instantly!")
    st.dataframe(result.style.background_gradient(cmap='Reds', subset=['Churn_Probability']))

else:
    st.info("↑ Upload your telco CSV on the left to see live predictions")
    st.balloons()
