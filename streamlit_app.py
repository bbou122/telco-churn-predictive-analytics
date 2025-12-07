# streamlit_app.py 
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import urllib.request
import tempfile
import os
import time
from typing import List, Tuple

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("**Pre-trained XGBoost • Instant Predictions • Robust preprocessing & error handling**")

# -------------------------
# Utilities
# -------------------------
def download_to_temp(url: str, suffix: str = "") -> str:
    """Download a URL to a temp file and return local path."""
    tmp_dir = tempfile.gettempdir()
    local_path = os.path.join(tmp_dir, f"streamlit_temp_{int(time.time()*1000)}{suffix}")
    urllib.request.urlretrieve(url, local_path)
    return local_path

def safe_load_feature_list(feat_url: str) -> List[str]:
    """Load features from CSV — tolerant to different column names."""
    df = pd.read_csv(feat_url)
    # If CSV has a column named "feature" use it, else take first column
    if "feature" in df.columns:
        return df["feature"].astype(str).tolist()
    else:
        return df.iloc[:, 0].astype(str).tolist()

# -------------------------
# Load model & features
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_features(model_url: str, feat_url: str) -> Tuple[xgb.XGBClassifier, List[str]]:
    """Download model.json and feature list, load model and return (model, feature_names)."""
    # Download model to a temp file (do not delete immediately)
    local_model_path = download_to_temp(model_url, suffix=".json")
    # Load model
    model = xgb.XGBClassifier()
    try:
        model.load_model(local_model_path)
    except Exception as e:
        # Provide a helpful error message on failure
        raise RuntimeError(f"Failed to load XGBoost model from '{model_url}': {e}")
    # Load features
    try:
        feature_names = safe_load_feature_list(feat_url)
    except Exception as e:
        raise RuntimeError(f"Failed to load feature names from '{feat_url}': {e}")
    # Return model and features (keep temp file — container cleans up)
    return model, feature_names

# Replace these URLs with your repo's raw URLs
MODEL_URL = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/model/model.json"
FEAT_URL  = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/model/feature_names.csv"

# Try load and show friendly error if it fails
try:
    model, feature_names = load_model_and_features(MODEL_URL, FEAT_URL)
except Exception as e:
    st.error("Model or feature file failed to load.")
    st.exception(e)
    st.stop()

# -------------------------
# Preprocessing
# -------------------------
def safe_qcut(series: pd.Series, q: int, labels: List[str]) -> pd.Series:
    """Robust qcut: if qcut fails fallback to quantile-based binning or simple cut."""
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        # If qcut fails (too few unique values), use cut with quantiles or constant label
        unique_vals = series.dropna().unique()
        if len(unique_vals) >= 2:
            try:
                qs = np.linspace(0, 1, min(q, len(unique_vals)))
                bins = series.quantile(qs).unique()
                # if bins length < 2 fallback
                if len(bins) >= 2:
                    # create labels matching number of bins-1 if duplicates were dropped
                    n_bins = len(bins) - 1
                    alt_labels = labels[:n_bins] if len(labels) >= n_bins else [f"bin{i}" for i in range(n_bins)]
                    return pd.cut(series, bins=bins, labels=alt_labels, include_lowest=True)
            except Exception:
                pass
        # final fallback: single label
        return pd.Series([labels[len(labels)//2]] * len(series), index=series.index)

def preprocess(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Preprocess dataframe to match training features.
    - Safe handling of missing columns
    - Stable dtypes
    - One-hot -> reindex to feature_names with fill_value=0
    """
    df = df.copy()

    # Ensure numeric columns exist and have sensible defaults
    # tenure and MonthlyCharges should exist for feature engineering. If missing, create reasonable defaults.
    if "tenure" not in df.columns:
        df["tenure"] = 0
    if "MonthlyCharges" not in df.columns:
        df["MonthlyCharges"] = 0.0
    # TotalCharges safe convert
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # if all NaN fallback to 0
        if df["TotalCharges"].notna().any():
            df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        else:
            df["TotalCharges"].fillna(0.0, inplace=True)
    else:
        df["TotalCharges"] = 0.0

    # Tenure group (use df['tenure'].max() so bins adapt to uploaded data)
    max_tenure = int(max( df["tenure"].max() if pd.notna(df["tenure"].max()) else 0, 1 ))
    df["Tenure_Group"] = pd.cut(
        df["tenure"],
        bins=[0,12,24,48,60, max_tenure],
        labels=['0-1yr','1-2yr','2-4yr','4-5yr','5+yr'],
        include_lowest=True
    )

    # MonthlyCharges group (robust)
    df["MonthlyCharges_Group"] = safe_qcut(df["MonthlyCharges"].astype(float), q=4, labels=['Low','Medium','High','VeryHigh'])

    # Ratio
    df["TotalCharges_per_Month"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan) + 1)
    df["TotalCharges_per_Month"].fillna(0.0, inplace=True)

    # Internet flag
    df["InternetService"] = df.get("InternetService", "").fillna("")
    df["Has_Internet"] = (df["InternetService"].astype(str).str.lower() != "no").astype(int)

    # Services: make sure columns exist and are strings
    services = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for col in services:
        if col not in df.columns:
            df[col] = "No"
        df[col] = df[col].astype(str).fillna("No")

    # Count Yes services robustly (case-insensitive)
    df['Num_Services'] = df[services].apply(lambda col: col.str.contains("yes", case=False, na=False)).sum(axis=1).astype(int)

    # Other engineered flags (use .get to avoid KeyError)
    df['No_TechSupport'] = (df.get('TechSupport', '').astype(str).str.lower() == 'no').astype(int)
    df['Fiber_Optic'] = (df.get('InternetService', '').astype(str).str.lower() == 'fiber optic').astype(int) | (df.get('InternetService', '').astype(str).str.lower() == 'fiber').astype(int)
    df['Month_to_Month'] = (df.get('Contract', '').astype(str).str.lower() == 'month-to-month').astype(int)
    df['Electronic_Check'] = (df.get('PaymentMethod', '').astype(str).str.lower() == 'electronic check').astype(int)

    # One-hot encode and reindex to training features
    X = pd.get_dummies(df.drop(columns=['customerID'], errors='ignore'), drop_first=True)
    # If any feature names in feature_names are missing in X, they'll be filled with 0
    X = X.reindex(columns=feature_names, fill_value=0)

    return X

# -------------------------
# UI: upload and scoring
# -------------------------
st.sidebar.header("Upload customer CSV")
uploaded = st.sidebar.file_uploader("Choose a CSV file with same columns used during training", type="csv")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Could not read uploaded CSV. Ensure it's a valid CSV file.")
        st.exception(e)
        st.stop()

    # Basic sanity check
    if data.shape[0] == 0:
        st.warning("Uploaded file contains no rows.")
        st.stop()

    # Preprocess
    try:
        X = preprocess(data, feature_names)
    except Exception as e:
        st.error("Preprocessing failed. Check that required columns exist (e.g., tenure, MonthlyCharges).")
        st.exception(e)
        st.stop()

    # Predict - guard for shape mismatch
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error("Prediction failed — likely a feature mismatch between model and uploaded data.")
        # Helpful debug info
        st.write("Model expects these features (first 40 shown):")
        st.write(feature_names[:40])
        st.write("Your prepared dataframe has these columns (first 40 shown):")
        st.write(list(X.columns[:40]))
        st.exception(e)
        st.stop()

    # Build result table
    ids = data.get("customerID", pd.Series(range(len(data))))
    result = pd.DataFrame({
        "customerID": ids,
        "Churn_Probability": np.round(probs, 3),
        "Prediction": np.where(probs >= 0.5, "Will Churn", "Will Stay")
    }).sort_values("Churn_Probability", ascending=False)

    st.success(f"Successfully scored {len(result)} customers!")
    st.dataframe(result.style.background_gradient(subset=["Churn_Probability"]))

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="churn_predictions.csv", mime="text/csv")

else:
    st.info("Upload a CSV to receive instant churn predictions.")
    st.markdown("**Sample/test file:** [Download sample here](https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/data/raw/telco_churn.csv)")
    st.balloons()
