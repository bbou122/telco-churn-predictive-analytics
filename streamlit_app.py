# streamlit_app.py 
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("**Pre-trained XGBoost • Instant Predictions**")

@st.cache_resource
def load_model():
    model_url = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/model/model.json"
    feat_url  = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/model/feature_names.csv"
    
    model = xgb.XGBClassifier()
    model.load_model(model_url)
    features = pd.read_csv(feat_url)["feature"].tolist()
    return model, features

model, feature_names = load_model()

def preprocess(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,df['tenure'].max()], 
                                labels=['0-1yr','1-2yr','2-4yr','4-5yr','5+yr'])
    df['MonthlyCharges_Group'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low','Medium','High','VeryHigh'])
    df['TotalCharges_per_Month'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['Has_Internet'] = (df['InternetService'] != 'No').astype(int)
    services = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['Num_Services'] = df[services].apply(lambda x: x.str.contains("Yes")).sum(axis=1)
    df['No_TechSupport'] = (df['TechSupport'] == 'No').astype(int)
    df['Fiber_Optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Month_to_Month'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

    X = pd.get_dummies(df.drop('customerID', axis=1, errors='ignore'), drop_first=True)
    X = X.reindex(columns=feature_names, fill_value=0)
    return X

st.sidebar.header("Upload Customers")
uploaded = st.sidebar.file_uploader("CSV file", type="csv")

if uploaded:
    data = pd.read_csv(uploaded)
    X = preprocess(data)
    probs = model.predict_proba(X)[:, 1]
    
    result = pd.DataFrame({
        "customerID": data["customerID"],
        "Churn_Probability": np.round(probs, 3),
        "Prediction": np.where(probs >= 0.5, "Will Churn", "Will Stay")
    }).sort_values("Churn_Probability", ascending=False)
    
    st.success(f"Scored {len(result)} customers!")
    st.dataframe(result.style.background_gradient(cmap="Reds", subset=["Churn_Probability"]))
    st.download_button("Download", result.to_csv(index=False), "predictions.csv")
else:
    st.info("Upload CSV → instant predictions!")
    st.balloons()
