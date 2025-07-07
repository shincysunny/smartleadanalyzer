import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import joblib


model = joblib.load("lead_classifier_bert.joblib")
scaler = joblib.load("scaler_bert.joblib")
encoder = joblib.load("encoder_bert.joblib")
bert = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="SmartLeadAnalyzer", layout="wide")
st.title(" SmartLeadAnalyzer")
st.markdown("Upload a CSV of leads or enter a single description for instant prediction!")


upload = st.file_uploader(" Upload a CSV file with 'Description' column", type="csv")

def featurize(desc_series):
    
    embeddings = bert.encode(desc_series.tolist(), show_progress_bar=False)
    df = pd.DataFrame(desc_series, columns=["Description"])

    
    df["desc_len"] = df["Description"].apply(len)
    df["has_ai"] = df["Description"].str.lower().str.contains("ai|ml|gpt").astype(int)
    df["has_saas"] = df["Description"].str.lower().str.contains("saas|platform|crm").astype(int)

    manual = scaler.transform(df[["desc_len", "has_ai", "has_saas"]])
    features = np.hstack([embeddings, manual])
    return features


st.subheader(" Predict One Lead")
input_text = st.text_area("Enter a lead description", height=100)

if st.button("Analyze"):
    if input_text.strip():
        features = featurize(pd.Series([input_text]))
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features).max()
        st.success(f"Predicted Lead Tag: **{prediction}**")
        st.info(f"Confidence: **{prob:.2f}**")


if upload:
    df = pd.read_csv(upload)
    if "Description" not in df.columns:
        st.error("CSV must contain a 'Description' column.")
    else:
        st.subheader(" Analyzing Leads...")
        features = featurize(df["Description"])
        preds = model.predict(features)
        df["Predicted Tag"] = encoder.inverse_transform(preds)
        st.dataframe(df[["Description", "Predicted Tag"]])

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(" Download Results CSV", csv_out, file_name="predicted_leads.csv")
