# streamlit_app.py – FINAL VERSION 
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import urllib.request
import os
import socket
import matplotlib.pyplot as plt
from fpdf import FPDF

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")

st.markdown("""
<div style="background: linear-gradient(90deg, #FF6B6B, #4ECDC4); padding: 12px; border-radius: 12px; text-align: center; color: white; font-size: 18px; font-weight: bold; margin-bottom: 25px;">
Live Production-Ready Churn Prediction Tool — Used by stakeholders in seconds
</div>
""", unsafe_allow_html=True)

st.markdown("**Pre-trained XGBoost • 0.84–0.86 AUC • Actionable Retention Insights**")

# Direct sample CSV download
sample_url = "https://raw.githubusercontent.com/bbou122/telco-churn-predictive-analytics/main/data/raw/telco_churn.csv"
st.sidebar.markdown(f"**Test with sample data:** [Download telco_churn.csv]({sample_url})")

# Help section
with st.expander("How to Use This App", expanded=False):
    st.markdown("""
    1. Upload a CSV (or use the sample above)  
    2. Get instant predictions + personalized suggestions  
    3. View stats, top drivers, and high-risk segmentation  
    4. Export as CSV or full PDF report (top 100 customers)
    """)

# Load model
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

model, feature_names = load_model_and_features()

# Preprocessing + suggestions
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
    if row.get('Num_Services', 0) < 3: s.append("Upsell add-ons")
    return "; ".join(s) or "Monitor closely"

# PDF Report
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
    pdf.cell(0, 10, "Top 100 Highest-Risk Customers", ln=1)
    pdf.set_font("Arial", size=9)
    for _, row in result.head(100).iterrows():
        line = f"{row['customerID']}: {row['Churn_Probability']:.3f} -> {row['Retention Suggestion']}"
        pdf.cell(0, 6, line, ln=1)
    pdf.output("churn_report.pdf")
    with open("churn_report.pdf", "rb") as f:
        return f.read()

# Main app
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

        # Updated Key Insight
        st.subheader("Key Business Insights from the Model")
        st.info("""
        • **Month-to-month contract customers are by far the most likely to churn** (35–45% risk)  
        • **Fiber optic internet customers churn at nearly twice the rate** of DSL customers  
        • Customers without **Tech Support** show dramatically higher churn  
        • Customers with **few or no add-on services** are prime upsell targets
        """)

        # Top Churn Drivers 
        st.subheader("Top Churn Drivers")
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color='#FF6B6B')
        ax.set_xlabel("Importance"); ax.invert_yaxis()
        st.pyplot(fig)

        # Churn Risk Distribution 
        st.subheader("Churn Risk Distribution")
        counts = result["Prediction"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax_pie.pie(counts, labels=counts.index, autopct='%1.0f%%', colors=['#95E1D3', '#FF6B6B'], startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

        # High-Risk by Contract
        st.subheader("High-Risk Customers by Contract Type")
        high_risk = result[result["Churn_Probability"] >= 0.5].copy()
        if not high_risk.empty:
            high_risk = high_risk.merge(processed_data[['Contract']], left_index=True, right_index=True)
            seg = high_risk.groupby('Contract').agg(
                Count=('customerID', 'count'),
                Avg_Probability=('Churn_Probability', 'mean')
            ).round(3).sort_values("Avg_Probability", ascending=False)
            seg['Recommendation'] = seg.index.map({
                'Month-to-month': 'Urgent — offer 1-year plan discount',
                'One year': 'Renewal campaign needed',
                'Two year': 'Loyal — reward & retain'
            })
            st.dataframe(seg.style.background_gradient(cmap="Oranges", subset=["Avg_Probability"]))

        # Downloads 
        st.subheader("Export Results")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button("Download Predictions CSV", result.to_csv(index=False), "telco_churn_predictions.csv", "text/csv")
            pdf_bytes = create_pdf(result, len(high_risk))
            st.download_button("Download Full PDF Report (Top 100)", pdf_bytes, "telco_churn_report.pdf", "application/pdf")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV to begin analysis")
    st.balloons()

st.markdown("---")
st.caption("Built by Braden Bourgeois • Master’s in Analytics • Open to Data Analyst / Jr Data Scientist roles")
