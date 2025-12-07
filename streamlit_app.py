# streamlit_app.py — FULL DEBUG VERSION

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import urllib.request
import os
import traceback
import tempfile

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor — DEBUG MODE")
st.markdown("**This version prints full debug info to solve your model loading issue.**")

# ============================================================
#   DEBUG LOADER — INSERTED RIGHT AFTER IMPORTS (CORRECT SPOT)
# ============================================================
@st.cache_resource(show_spinner=False)
def debug_load_model_and_features(model_url: str, feat_url: str):
    debug_info = {"model_url": model_url, "feat_url": feat_url}

    try:
        # -------------------
        # Download model.json
        # -------------------
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_model.close()  # close temp file so urlretrieve can write to it
        urllib.request.urlretrieve(model_url, tmp_model.name)

        # Model file size + preview
        debug_info["model_file_size_bytes"] = os.path.getsize(tmp_model.name)
        with open(tmp_model.name, "rb") as f:
            debug_info["model_first_bytes"] = f.read(300)

        # -----------------------
        # Download feature_names
        # -----------------------
        tmp_feat = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_feat.close()
        urllib.request.urlretrieve(feat_url, tmp_feat.name)

        debug_info["feature_file_size_bytes"] = os.path.getsize(tmp_feat.name)

        # Load feature file
        try:
            feat_df = pd.read_csv(tmp_feat.name)
            debug_info["feature_csv_head"] = feat_df.head(5).to_dict()

            if "feature" in feat_df.columns:
                feature_list = feat_df["feature"].astype(str).tolist()
            else:
                # Fallback: assume first column is features
                feature_list = feat_df.iloc[:, 0].astype(str).tolist()

            debug_info["num_features_loaded"] = len(feature_list)

        except Exception:
            debug_info["feature_load_error"] = traceback.format_exc()
            feature_list = []

        # -----------------------
        # Try loading the model
        # -----------------------
        load_attempts = {}
        model_obj = None

        # Attempt 1 – XGBClassifier
        try:
            m1 = xgb.XGBClassifier()
            m1.load_model(tmp_model.name)
            model_obj = m1
            debug_info["model_loaded_via"] = "XGBClassifier"
        except Exception:
            load_attempts["XGBClassifier"] = traceback.format_exc()

        # Attempt 2 – Booster
        if model_obj is None:
            try:
                booster = xgb.Booster()
                booster.load_model(tmp_model.name)
                model_obj = booster
                debug_info["model_loaded_via"] = "Booster"
            except Exception:
                load_attempts["Booster"] = traceback.format_exc()

        debug_info["load_attempt_errors"] = load_attempts

        return model_obj, feature_list, debug_info

    except Exception:
        debug_info["fatal_error"] = traceback.format_exc()
        return None, [], debug_info


# ============================================================
#   LOAD MODEL + FEATURES (WITH DEBUG)
# ============================================================
MODEL_URL = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/model/model.json"
FEAT_URL  = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/model/feature_names.csv"

model, feature_names, debug_info = debug_load_model_and_features(MODEL_URL, FEAT_URL)

# Display debug info on screen
st.subheader("🔍 Debug Information")
st.json(debug_info)

# If model failed, stop here
if model is None:
    st.error("❌ Model failed to load. See debug information above.")
    st.stop()

# ============================================================
#   PREPROCESSING FUNCTION
# ============================================================
def preprocess(df):
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median() if df["TotalCharges"].notna().any() else 0, inplace=True)
    else:
        df["TotalCharges"] = 0

    df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,999], labels=['0-1yr','1-2yr','2-4yr','4-5yr','5+yr'])
    df['MonthlyCharges_Group'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low','Medium','High','VeryHigh'], duplicates='drop')
    df['TotalCharges_per_Month'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['Has_Internet'] = (df['InternetService'] != 'No').astype(int)

    services = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for col in services:
        if col not in df.columns:
            df[col] = 'No'

    df['Num_Services'] = df[services].apply(lambda x: x.str.contains("Yes", case=False, na=False)).sum(axis=1)
    df['No_TechSupport'] = (df['TechSupport'] == 'No').astype(int)
    df['Fiber_Optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Month_to_Month'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

    X = pd.get_dummies(df.drop(columns=['customerID'], errors='ignore'), drop_first=True)
    X = X.reindex(columns=feature_names, fill_value=0)

    return X


# ============================================================
#   UI + PREDICTION
# ============================================================
st.sidebar.header("Upload Customer CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        X = preprocess(df)

        # If Booster object, use predict() instead of predict_proba()
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(X)
            probs = model.predict(dmatrix)
        else:
            probs = model.predict_proba(X)[:, 1]

        output = pd.DataFrame({
            "customerID": df.get("customerID", range(len(df))),
            "Churn_Probability": np.round(probs, 3),
            "Prediction": np.where(probs >= 0.5, "Will Churn", "Will Stay")
        }).sort_values("Churn_Probability", ascending=False)

        st.success("Predictions generated!")
        st.dataframe(output)
        st.download_button("Download Predictions", output.to_csv(index=False), "predictions.csv")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

else:
    st.info("Upload a CSV file to generate predictions.")
