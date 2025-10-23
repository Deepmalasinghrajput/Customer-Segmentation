import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------- Background Image & CSS -------------------
image_url = 'https://ezranking.s3.eu-west-2.amazonaws.com/blog/wp-content/uploads/2022/06/10115059/Customer-Segmentation.jpg'

st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("{image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }}

    /* Title */
    h1 {{
        color: #ffd700;
        text-align: center;
        font-size: 38px !important;
        font-weight: 700;
    }}

    /* Input labels */
    .stNumberInput label {{
        font-size: 16px !important;
        color: #ffffff;
        font-weight: 500;
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        background-color: #2e86de;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: 600;
        transition: 0.3s;
    }}

    div.stButton > button:first-child:hover {{
        background-color: #1b4f72;
        transform: scale(1.03);
    }}

    /* Prediction Box */
    .prediction-box {{
        background-color: rgba(255, 255, 255, 0.15);
        border: 2px solid #ffd700;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #fff;
        box-shadow: 0px 0px 15px #ffd700;
        margin-top: 20px;
    }}

    /* Alerts */
    .stAlert {{
        border-radius: 10px;
        font-size: 16px;
    }}

    /* Footer */
    .footer {{
        position: relative;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: #ffffff;
        margin-top: 50px;
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------- Load Models -------------------
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# ------------------- App Title -------------------
st.title('🧩 Customer Segmentation App')

# ------------------- Centered Instructions -------------------
st.markdown("""
    <p style='text-align: center; font-size: 18px; color: #ffffff;'>
        Enter customer details below to predict which segment they belong to 👇
    </p>
""", unsafe_allow_html=True)

# ------------------- Input Fields -------------------
age = st.number_input('Age', min_value=18, max_value=100, value=35)
income = st.number_input('Income', min_value=0, max_value=200000, value=50000)
total_spending = st.number_input('Total Spending (sum of purchases)', min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input('Number of Web Purchases', min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input('Number of Store Purchases', min_value=0, max_value=100, value=10)
num_web_visits = st.number_input('Number of Web Visits per Month', min_value=0, max_value=50, value=3)
recency = st.number_input('Recency (days since last purchase)', min_value=0, max_value=365, value=30)

# ------------------- Prepare Input -------------------
input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Total_Spending': [total_spending],
    'NumWebPurchases': [num_web_purchases],
    'NumStorePurchases': [num_store_purchases],
    'NumWebVisitsMonth': [num_web_visits],
    'Recency': [recency]
})

input_scaled = scaler.transform(input_data)

# ------------------- Cluster Info -------------------
cluster_info = {
    0: " Premium Customer – High income, high spending",
    2: " Digital Buyer – High web purchases, low store purchases",
    5: " Dormant Customer – Low recency, inactive",
    6: " Budget Customer – Low income, low spending"
}

# ------------------- Prediction -------------------
if st.button(' Predict Segment'):
    cluster = kmeans.predict(input_scaled)[0]
    message = cluster_info.get(cluster,"This is a new or mixed-type customer.")

    st.markdown(f"""
        <div class='prediction-box'>
            Predicted Segment: <strong>Cluster {cluster}</strong><br>
            {message}
        </div>
    """, unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown("<div class='footer'>✨ Built by Deepmala | Customer Segmentation Project </div>", unsafe_allow_html=True)
