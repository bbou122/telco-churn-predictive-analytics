# streamlit_app.py 
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("**XGBoost Model • 0.84+ AUC • Instant Predictions for New Customers**")

# Load and train model (cached for speed - retrains in ~3 seconds using public URL)
@st.cache_resource
def load_model():
    # Load from direct public URL (no local file needed)
    url = "https://raw.githubusercontent.com/ironhack-datalabs/datamad0820/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Feature engineering from your notebook
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,df['tenure'].max()], labels=['0-1yr','1-2yr','2-4yr','4-5yr','5+yr'])
    df['MonthlyCharges_Group'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low','Medium','High','VeryHigh'])
    df['TotalCharges_per_Month'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['Has_Internet'] = (df['InternetService'] != 'No').astype(int)
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Num_Services'] = df[services].apply(lambda x: x.str.contains('Yes')).sum(axis=1)
    df['No_TechSupport'] = (df['TechSupport'] == 'No').astype(int)
    df['Fiber_Optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Month_to_Month'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = (df['Churn'] == 'Yes').astype(int)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01,
        random_state=42, eval_metric='auc', early_stopping_rounds=50
    )
    model.fit(X_encoded, y, eval_set=[(X_encoded, y)], verbose=False)
    return model, X_encoded.columns

model, feature_names = load_model()

# Sidebar for upload
st.sidebar.header("Upload New Customers")
uploaded_file = st.sidebar.file_uploader("Drop a CSV with the same columns as the original data", type="csv")

# Main app
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    original_data = new_data.copy()
    
    # Same preprocessing as training
    new_data['TotalCharges'] = pd.to_numeric(new_data['TotalCharges'], errors='coerce')
    new_data['TotalCharges'].fillna(new_data['TotalCharges'].median(), inplace=True)
    
    # Same feature engineering
    new_data['Tenure_Group'] = pd.cut(new_data['tenure'], bins=[0,12,24,48,60,new_data['tenure'].max()], labels=['0-1yr','1-2yr','2-4yr','4-5yr','5+yr'])
    new_data['MonthlyCharges_Group'] = pd.qcut(new_data['MonthlyCharges'], q=4, labels=['Low','Medium','High','VeryHigh'])
    new_data['TotalCharges_per_Month'] = new_data['TotalCharges'] / (new_data['tenure'] + 1)
    new_data['Has_Internet'] = (new_data['InternetService'] != 'No').astype(int)
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    new_data['Num_Services'] = new_data[services].apply(lambda x: x.str.contains('Yes')).sum(axis=1)
    new_data['No_TechSupport'] = (new_data['TechSupport'] == 'No').astype(int)
    new_data['Fiber_Optic'] = (new_data['InternetService'] == 'Fiber optic').astype(int)
    new_data['Month_to_Month'] = (new_data['Contract'] == 'Month-to-month').astype(int)
    new_data['Electronic_Check'] = (new_data['PaymentMethod'] == 'Electronic check').astype(int)
    
    X_new = pd.get_dummies(new_data.drop('customerID', axis=1, errors='ignore'), drop_first=True)
    X_new = X_new.reindex(columns=feature_names, fill_value=0)
    
    probs = model.predict_proba(X_new)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    result = original_data[['customerID']].copy()
    result['Churn_Probability'] = np.round(probs, 3)
    result['Prediction'] = np.where(preds==1, "Will Churn", "Will Stay")
    result = result.sort_values('Churn_Probability', ascending=False)
    
    st.success(f"Scored {len(result)} customers in real-time!")
    st.dataframe(result.style.background_gradient(cmap='Reds', subset=['Churn_Probability']))
    
else:
    st.info("Upload a CSV on the left to get instant predictions")
    st.markdown("### Key Insights from Model: Month-to-month contracts and fiber optic service are top churn drivers!")
