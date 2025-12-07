# streamlit_app.py 
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import urllib.request
import os
import socket  # For timeout
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("**Pre-trained XGBoost • 0.84+ AUC • Instant Predictions**")

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
        
        # Confirm engineered features
        engineered = ['Month_to_Month', 'Fiber_Optic', 'No_TechSupport', 'Num_Services']
        if all(f in features for f in engineered):
            st.write("Model loaded — engineered features confirmed!")
        else:
            st.warning("Engineered features missing — re-train model.json")
        
        return model, features
    except Exception as e:
        st.error(f"Load failed: {e}. Check GitHub files and connection.")
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
    return X, df  # Return original df for segmentation

# ——— RETENTION SUGGESTIONS (rule-based on key features) ———
def get_suggestion(row):
    suggestions = []
    if row.get('Month_to_Month', 0) == 1:
        suggestions.append("Offer long-term contract discount")
    if row.get('Fiber_Optic', 0) == 1:
        suggestions.append("Improve fiber service quality")
    if row.get('No_TechSupport', 0) == 1:
        suggestions.append("Bundle tech support package")
    if row.get('Num_Services', 0) < 3:
        suggestions.append("Upsell add-on services")
    return "; ".join(suggestions) or "Monitor for retention"

# ——— UI & PREDICTION ———
st.sidebar.header("Upload Customer Data")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded, encoding='latin1')  # Handles encoding issues
        X, processed_data = preprocess(data)
        probs = model.predict_proba(X)[:, 1]
        
        result = pd.DataFrame({
            "customerID": data.get("customerID", pd.Series(range(len(data)))),
            "Churn_Probability": np.round(probs, 3),
            "Prediction": np.where(probs >= 0.5, "Will Churn", "Will Stay")
        }).sort_values("Churn_Probability", ascending=False)
        
        # ADD: Retention Suggestions column
        processed_data = processed_data.reset_index(drop=True)  # Align indices
        result = result.reset_index(drop=True)
        result['Retention Suggestion'] = processed_data.apply(get_suggestion, axis=1)
        
        st.success(f"Scored {len(result)} customers!")
        st.dataframe(result.style.background_gradient(cmap="Reds", subset=["Churn_Probability"]))
        st.download_button("Download Predictions", result.to_csv(index=False), "churn_predictions.csv", "text/csv")
        
        # ADD 3: Summary Stats Section
        st.subheader("Summary Stats")
        avg_prob = np.mean(result["Churn_Probability"])
        high_risk_count = len(result[result["Churn_Probability"] >= 0.5])
        high_risk_pct = (high_risk_count / len(result) * 100) if len(result) > 0 else 0
        top_risk_segment = result.iloc[0]["Prediction"] if len(result) > 0 else "N/A"
        st.markdown(f"""
        - Average Churn Probability: **{avg_prob:.3f}**
        - High-Risk Customers (Probability ≥ 0.5): **{high_risk_count} ({high_risk_pct:.1f}%)**
        - Top Risk Prediction: **{top_risk_segment}**
        """)
        
        # ADD 1: Feature Importance Chart (smaller size)
        st.subheader("Top Churn Drivers (Feature Importance)")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))  # Smaller size
        ax.barh(imp_df["Feature"], imp_df["Importance"], color='skyblue')
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Features Driving Churn")
        ax.invert_yaxis()
        st.pyplot(fig)
        
        # ADD 2: Churn Distribution Pie Chart (smaller size)
        st.subheader("Churn Distribution")
        churn_counts = result["Prediction"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))  # Smaller size
        ax_pie.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
        ax_pie.set_title("Percentage Will Churn vs Will Stay")
        st.pyplot(fig_pie)
        
        # ADD: Segmentation of At Risk Customers
        st.subheader("Segmentation of High-Risk Customers (Probability ≥ 0.5)")
        high_risk = result[result["Churn_Probability"] >= 0.5].copy()
        if not high_risk.empty:
            high_risk['Contract'] = processed_data['Contract']
            segment = high_risk.groupby('Contract').agg({
                'Churn_Probability': 'mean',
                'customerID': 'count'
            }).rename(columns={'customerID': 'Count', 'Churn_Probability': 'Avg Probability'})
            segment['Suggestion'] = segment.index.map(lambda x: "Focus on loyalty programs for month-to-month" if x == 'Month-to-month' else "Strengthen renewal offers")
            st.dataframe(segment.style.background_gradient(cmap="Oranges", subset=["Avg Probability"]))
        else:
            st.write("No high-risk customers in this batch.")
    except Exception as e:
        st.error(f"Error: {e}. Check CSV format or encoding.")
else:
    st.info("Upload CSV to start!")
    st.markdown("**Test file:** [Download](https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/data/raw/telco_churn.csv)")
    st.balloons()

