# ============================================================
# 🚖 UBER FARE PREDICTION APP
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import requests
from io import BytesIO

# ============================================================
# 1️⃣ Custom Function Definitions
# ============================================================
# Function to calculate Haversine distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two sets of coordinates
    in kilometers.
    """
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = R * c
    return distance_km

# Function to create new features from raw data
def create_features(df):
    """
    Creates the necessary features for the model from raw data.
    """
    df_copy = df.copy()

    # Check if 'pickup_datetime' column exists, if not, use datetime.now()
    if 'pickup_datetime' not in df_copy.columns:
        df_copy['pickup_datetime'] = datetime.now()

    # Calculate trip distance
    df_copy['trip_distance_km'] = df_copy.apply(
        lambda row: haversine(
            row['pickup_latitude'], row['pickup_longitude'],
            row['dropoff_latitude'], row['dropoff_longitude']
        ), axis=1
    )

    # Extract time features from 'pickup_datetime'
    df_copy['year'] = df_copy['pickup_datetime'].dt.year
    df_copy['month'] = df_copy['pickup_datetime'].dt.month
    df_copy['day'] = df_copy['pickup_datetime'].dt.day
    df_copy['dayofweek'] = df_copy['pickup_datetime'].dt.dayofweek
    df_copy['hour'] = df_copy['pickup_datetime'].dt.hour
    
    # Select features to be used for prediction
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 
                'dropoff_latitude', 'passenger_count', 'trip_distance_km',
                'year', 'month', 'day', 'dayofweek', 'hour']
    
    return df_copy[features]

# ============================================================
# 2️⃣ Load Model Pipeline
# ============================================================
def download_file(url, destination_path):
    """
    Downloads a file from the given URL and saves it to the destination path.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"✅ File '{os.path.basename(destination_path)}' downloaded successfully.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error: Failed to download file from URL. {e}")
        st.stop()


@st.cache_resource
def load_assets():
    """
    Loads the saved model pipeline using joblib.
    If the file does not exist, it tries to download it from the URL.
    """
    pipeline_file_name = "pipeline.pkl"
    
    # GANTI DENGAN URL UNDUHAN LANGSUNG PIPELINE.PKL ANDA
    pipeline_url = "https://example.com/your-pipeline-file.pkl"
    
    if not os.path.exists(pipeline_file_name):
        st.info(f"⏳ Downloading pipeline '{pipeline_file_name}'...")
        download_file(pipeline_url, pipeline_file_name)

    try:
        pipeline = joblib.load(pipeline_file_name)
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

# Load model pipeline when the app starts
pipeline = load_assets()

# ============================================================
# 3️⃣ Streamlit App Configuration
# ============================================================
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="centered"
)

st.title("🚖 Uber Fare Prediction App")
st.markdown("Masukkan detail perjalanan Anda untuk memprediksi tarif Uber")

# ============================================================
# 4️⃣ User Input Form
# ============================================================
with st.form("fare_form"):
    st.subheader("📝 Trip Details Input")
    
    pickup_date = st.date_input("Pickup Date", value=datetime.now().date())
    pickup_time = st.time_input("Pickup Time", value=datetime.now().time())

    col1, col2 = st.columns(2)
    with col1:
        pickup_latitude = st.number_input("Pickup Latitude", value=40.738354, format="%.6f")
        pickup_longitude = st.number_input("Pickup Longitude", value=-73.999817, format="%.6f")
    
    with col2:
        dropoff_latitude = st.number_input("Dropoff Latitude", value=40.723217, format="%.6f")
        dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.999512, format="%.6f")

    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

    submitted = st.form_submit_button("🔮 Predict Fare")

    if submitted:
        try:
            pickup_datetime_full = datetime.combine(pickup_date, pickup_time)
            
            input_data = pd.DataFrame([{
                'pickup_longitude': pickup_longitude,
                'pickup_latitude': pickup_latitude,
                'dropoff_longitude': dropoff_longitude,
                'dropoff_latitude': dropoff_latitude,
                'passenger_count': passenger_count,
                'pickup_datetime': pickup_datetime_full
            }])
            
            processed_data = create_features(input_data)
            
            prediction = pipeline.predict(processed_data)
            
            st.subheader("✅ Prediction Successful!")
            st.success(f"Predicted Uber fare: ${prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
