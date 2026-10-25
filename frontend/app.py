import os
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="wide")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")

print("BACKEND_URL: " + BACKEND_URL)
st.title("SuperKart Sales Forecast")
st.write("Estimate sales for a product-store combination or upload a batch CSV.")
st.caption("Predictions support planning and should be combined with inventory, promotions, and operational context.")

with st.form("single_prediction"):
    c1, c2 = st.columns(2)
    with c1:
        Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
        Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027, format="%.3f")
        Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
        Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC", "Other"])
    with c2:
        Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        Store_Location_City_Type = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"])
        Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=16)
        Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
    submitted = st.form_submit_button("Predict sales", type="primary")

if submitted:
    payload = {
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Product_Id_char": Product_Id_char,
        "Store_Age_Years": Store_Age_Years,
        "Product_Type_Category": Product_Type_Category,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
        response.raise_for_status()
        st.metric("Predicted product-store sales", f'{response.json()["Sales"]:,.2f}')
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")

st.subheader("Batch prediction")
uploaded_file = st.file_uploader("Upload a CSV with the 10 model features", type=["csv"])
if uploaded_file is not None:
    try:
        batch_preview = pd.read_csv(uploaded_file)
        st.dataframe(batch_preview.head(10), use_container_width=True)
        uploaded_file.seek(0)
        if st.button("Predict batch"):
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=120,
            )
            response.raise_for_status()
            predictions = pd.DataFrame(response.json()["predictions"])
            st.dataframe(predictions, use_container_width=True)
            st.download_button("Download prediction results", predictions.to_csv(index=False), "superkart_predictions.csv", "text/csv")
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
