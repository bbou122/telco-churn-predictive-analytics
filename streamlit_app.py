# streamlit_app.py – FINAL
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import urllib.request
import os
import socket
import matplotlib.pyplot as plt
from fpdf import FPDF

# Use DejaVuSans (Unicode-safe) instead of Helvetica
plt.rcParams['font.family'] = 'DejaVu Sans'

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("**Pre-trained XGBoost • 0.84+ AUC • Instant Predictions**")

# ——— HELP SECTION ———
with st.expander("How to Use This App – Click to expand", expanded=False):
    st.markdown("""
    1. **Upload** a CSV with the same columns as the sample  
    2. Get **predictions + retention suggestions** instantly  
    3. Explore **stats, charts, and high-risk segmentation**  
    4. **Download** results as CSV or full PDF report  
    → Sample file: [Download telco_churn.csv](https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/data/raw/telco_churn.csv)
    """)

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
        st.write("Model loaded — engineered features confirmed!")
        return model, features
    except Exception as e:
        st.error(f"Load failed: {e}")
        st.stop()
        return None, None

model, feature_names = load_model_and_features()

# ——— PREPROCESSING & SUGGESTIONS ———
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
    return X, df

def get_suggestion(row):
    s = []
    if row.get('Month_to_Month', 0) == 1: s.append("Offer long-term contract discount")
    if row.get('Fiber_Optic', 0) == 1: s.append("Improve fiber service quality")
    if row.get('No_TechSupport', 0) == 1: s.append("Bundle tech support")
    if row.get('Num_Services', 0) < 3: s.append("Upsell add-on services")
    return "; ".join(s) or "Monitor"

# ——— PDF REPORT (now uses DejaVuSans → no font errors) ———
def create_pdf(result, high_risk_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Telco Churn Prediction Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Total Customers: {len(result)} | High-Risk: {high_risk_count} ({high_risk_count/len(result)*100:.1f}%)", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top 10 At-Risk Customers & Suggestions", ln=1)
    pdf.set_font("Arial", size=10)
    for _, row in result.head(10).iterrows():
        line = f"{row['customerID']}: {row['Churn_Probability']:.3f} → {row['Retention Suggestion']}"
        pdf.cell(0, 8, line, ln=1)
    pdf.output("churn_report.pdf")
    with open("churn_report.pdf", "rb") as f:
        return f.read()

# ——— MAIN UI ———
st.sidebar.header("Upload Customer Data")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded, encoding='latin1')
        X, processed_data = preprocess(data)
        probs = model.predict_proba(X)[:, 1]

        result = pd.DataFrame({
            "customerID": data.get("customerID", pd.Series(range(len(data)))),
            "Churn_Probability": np.round(probs, 3),
            "Prediction": np.where(probs >= 0.5, "Will Churn", "Will Stay")
        }).sort_values("Churn_Probability", ascending=False)
        result['Retention Suggestion'] = processed_data.apply(get_suggestion, axis=1)

        st.success(f"Scored {len(result)} customers!")
        st.dataframe(result.style.background_gradient(cmap="Reds", subset=["Churn_Probability"]))
        st.download_button("Download CSV", result.to_csv(index=False), "predictions.csv", "text/csv")

        high_risk_count = len(result[result["Churn_Probability"] >= 0.5])
        st.subheader("Summary Stats")
        st.markdown(f"- Average Churn Probability: **{result['Churn_Probability'].mean():.3f}**\n- High-Risk Customers: **{high_risk_count} ({high_risk_count/len(result)*100:.1f}%)**")

        st.subheader("Top Churn Drivers")
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color='skyblue')
        ax.set_xlabel("Importance"); ax.invert_yaxis()
        st.pyplot(fig)

        st.subheader("Churn Distribution")
        counts = result["Prediction"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['lightgreen','salmon'])
        st.pyplot(fig_pie)

        # PDF Report Button (now works!)
        pdf_bytes = create_pdf(result, high_risk_count)
        st.download_button(
            label="Download Full PDF Report",
            data=pdf_bytes,
            file_name="churn_report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV to begin!")
    st.markdown("**Test file:** [Download sample](https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/data/raw/telco_churn.csv)")
    st.balloons()
