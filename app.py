import streamlit as st
import pandas as pd
import pickle
import os
import requests
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Weather Event Predictor", page_icon="🌦", layout="centered")

# Google Drive direct download link
# Replace <FILE_ID> with your Google Drive file ID
MODEL_URL = "https://drive.google.com/file/d/1bxeLow-PCHdOQqzWkWOcB3L32MPC3RBU/view?usp=sharing>"
MODEL_PATH = "model.pkl"

# -------------------------------
# Download large file from Google Drive
# -------------------------------
def download_file_from_google_drive(url, destination):
    """Download a large file from Google Drive handling confirmation"""
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Check for confirmation token for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        params = {"confirm": token}
        response = session.get(url, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            download_file_from_google_drive(MODEL_URL, MODEL_PATH)
    
    # Debug: check file size
    st.write(f"Downloaded model size: {os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB")

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌦 Predictive Analysis of Meteorological Events")
st.write("A machine learning app to forecast meteorological events using Random Forest.")

st.sidebar.header("Enter Weather Parameters")

def user_input_features():
    temp = st.sidebar.slider("Temperature (°C)", -10.0, 45.0, 20.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
    wind_kph = st.sidebar.slider("Wind Speed (kph)", 0.0, 150.0, 10.0)
    pressure_mb = st.sidebar.slider("Pressure (mb)", 900.0, 1100.0, 1013.0)
    precip_mm = st.sidebar.slider("Precipitation (mm)", 0.0, 200.0, 0.0)

    data = {
        'temperature_celsius': temp,
        'humidity': humidity,
        'wind_kph': wind_kph,
        'pressure_mb': pressure_mb,
        'precip_mm': precip_mm
    }
    return pd.DataFrame(data, index=[0])

# Collect user inputs
input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write(f"Predicted Event: **{prediction[0]}**")

st.subheader("Prediction Probability")
st.write(pd.DataFrame(prediction_proba, columns=model.classes_))
