# streamlit_app.py – FULL VERSION WITH SHAP (deploys on Streamlit Cloud, works locally on 3.10)
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import urllib.request
import os
import socket  # For timeout
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("**Pre-trained XGBoost • 0.84+ AUC • Instant Predictions with SHAP Explanations**")

# ——— LOAD MODEL & FEATURES ———
@st.cache_resource
def load_model_and_features():
    model_url = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/model/model.json"
    feat_url  = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/model/feature_names.csv"
    
    try:
        socket.setdefaulttimeout(30)
        
        local_model = "temp_model.json"
        urllib.request.urlretrieve(model_url, local_model)
        
        model = xgb.XGBClassifier()
        model.load_model(local_model)
        
        features = pd.read_csv(feat_url)["feature"].tolist()
        
        if os.path.exists(local_model):
            os.remove(local_model)
        
        # Confirm engineered features (from notebook)
        engineered = ['Month_to_Month', 'Fiber_Optic', 'No_TechSupport', 'Num_Services']
        if all(f in features for f in engineered):
            st.write("Model loaded — engineered features confirmed!")
        else:
            st.warning("Engineered features missing — re-train model.json")
        
        return model, features
    except Exception as e:
        st.error(f"Load failed: {e}. Check GitHub files.")
        st.stop()
        return None, None

model, feature_names = load_model_and_features()

# ——— PREPROCESSING ———
def preprocess(df):
    df = df.copy()
    
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median() if not df["TotalCharges"].isna().all() else 0, inplace=True)
    else:
        df["TotalCharges"] = 0
    
    df['Tenure_Group'] = pd.cut(df.get('tenure', 0), bins=[0,12,24,48,60,999], labels=['0-1yr','1-2yr','2-4yr','4-5yr','5+yr'])
    df['MonthlyCharges_Group'] = pd.qcut(df.get('MonthlyCharges', [0]*len(df)), q=4, labels=['Low','Medium','High','VeryHigh'], duplicates='drop')
    df['TotalCharges_per_Month'] = df['TotalCharges'] / (df.get('tenure', 0) + 1)
    df['Has_Internet'] = (df.get('InternetService', 'No') != 'No').astype(int)
    
    services = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for col in services:
        if col not in df.columns:
            df[col] = 'No'
    df['Num_Services'] = df[services].apply(lambda x: x.str.contains("Yes", case=False, na=False)).sum(axis=1)
    
    df['No_TechSupport'] = (df.get('TechSupport', 'No') == 'No').astype(int)
    df['Fiber_Optic'] = (df.get('InternetService', 'No') == 'Fiber optic').astype(int)
    df['Month_to_Month'] = (df.get('Contract', 'Month-to-month') == 'Month-to-month').astype(int)
    df['Electronic_Check'] = (df.get('PaymentMethod', 'Electronic check') == 'Electronic check').astype(int)
    
    X = pd.get_dummies(df.drop(columns=['customerID'], errors='ignore'), drop_first=True)
    X = X.reindex(columns=feature_names, fill_value=0)
    return X

# ——— UI & PREDICTION ———
st.sidebar.header("Upload Customer Data")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        X = preprocess(data)
        probs = model.predict_proba(X)[:, 1]
        
        result = pd.DataFrame({
            "customerID": data.get("customerID", pd.Series(range(len(data)))),
            "Churn_Probability": np.round(probs, 3),
            "Prediction": np.where(probs >= 0.5, "Will Churn", "Will Stay")
        }).sort_values("Churn_Probability", ascending=False)
        
        st.success(f"Scored {len(result)} customers!")
        st.dataframe(result.style.background_gradient(cmap="Reds", subset=["Churn_Probability"]))
        st.download_button("Download", result.to_csv(index=False), "predictions.csv")
        
        # SHAP for highest-risk customer (as in your notebook)
        if len(result) > 0:
            top_index = result.index[0]
            st.subheader(f"SHAP Explanation for Highest-Risk Customer: {result.loc[top_index, 'customerID']}")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[top_index:top_index+1])
            shap.initjs()  # For force plot
            fig, ax = plt.subplots()
            shap.force_plot(explainer.expected_value, shap_values, X.iloc[top_index], matplotlib=True, show=False)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {e}. Check CSV format.")
else:
    st.info("Upload CSV to start!")
    st.markdown("**Test file:** [Download](https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/data/raw/telco_churn.csv)")
    st.balloons()
