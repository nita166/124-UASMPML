# ============================================================
# 🚖 APLIKASI PREDIKSI TARIF UBER
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ============================================================
# 1️⃣ Definisi Fungsi Kustom
# ============================================================
# Fungsi untuk menghitung jarak Haversine antara dua koordinat
def haversine(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak Haversine antara dua titik koordinat
    dalam kilometer.
    """
    R = 6371  # Radius bumi dalam kilometer
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = R * c
    return distance_km

# Fungsi untuk membuat fitur baru dari data mentah
def create_features(df):
    """
    Membuat fitur-fitur baru yang diperlukan oleh model
    dari data mentah.
    """
    df_copy = df.copy()

    # Periksa apakah kolom 'pickup_datetime' ada, jika tidak, gunakan datetime.now()
    if 'pickup_datetime' not in df_copy.columns:
        df_copy['pickup_datetime'] = datetime.now()

    # Hitung jarak perjalanan
    df_copy['trip_distance_km'] = df_copy.apply(
        lambda row: haversine(
            row['pickup_latitude'], row['pickup_longitude'],
            row['dropoff_latitude'], row['dropoff_longitude']
        ), axis=1
    )

    # Ekstrak fitur waktu dari 'pickup_datetime'
    df_copy['year'] = df_copy['pickup_datetime'].dt.year
    df_copy['month'] = df_copy['pickup_datetime'].dt.month
    df_copy['day'] = df_copy['pickup_datetime'].dt.day
    df_copy['dayofweek'] = df_copy['pickup_datetime'].dt.dayofweek
    df_copy['hour'] = df_copy['pickup_datetime'].dt.hour
    
    # Pilih fitur yang akan digunakan untuk prediksi
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 
                'dropoff_latitude', 'passenger_count', 'trip_distance_km',
                'year', 'month', 'day', 'dayofweek', 'hour']
    
    return df_copy[features]

# ============================================================
# 2️⃣ Load Model
# ============================================================
@st.cache_resource
def load_model():
    """
    Memuat model Decision Tree yang telah disimpan menggunakan joblib.
    """
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan.")
        st.stop()
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ============================================================
# 3️⃣ Konfigurasi Aplikasi Streamlit
# ============================================================
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="centered"
)

st.title("🚖 Aplikasi Prediksi Tarif Uber")
st.markdown("Masukkan detail perjalanan Anda untuk memprediksi tarif Uber")

# ============================================================
# 4️⃣ Form Input Pengguna
# ============================================================
with st.form("fare_form"):
    st.subheader("📝 Detail Perjalanan")
    
    pickup_date = st.date_input("Tanggal Penjemputan", value=datetime.now().date())
    pickup_time = st.time_input("Waktu Penjemputan", value=datetime.now().time())

    col1, col2 = st.columns(2)
    with col1:
        pickup_latitude = st.number_input("Pickup Latitude", value=40.738354, format="%.6f")
        pickup_longitude = st.number_input("Pickup Longitude", value=-73.999817, format="%.6f")
    
    with col2:
        dropoff_latitude = st.number_input("Dropoff Latitude", value=40.723217, format="%.6f")
        dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.999512, format="%.6f")

    passenger_count = st.number_input("Jumlah Penumpang", min_value=1, max_value=6, value=1)

    submitted = st.form_submit_button("🔮 Prediksi Tarif")

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
            
            prediction = model.predict(processed_data)
            
            st.subheader("✅ Prediksi Berhasil!")
            st.success(f"Tarif Uber yang diprediksi: ${prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")