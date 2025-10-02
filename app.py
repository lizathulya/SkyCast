import streamlit as st
import pandas as pd
import pickle
import requests
import os
import gdown
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Weather Event Predictor", page_icon="🌦", layout="centered")

# Replace with your actual Google Drive FILE_ID
# Example link: https://drive.google.com/file/d/1AbC1234XYZ/view?usp=sharing
# FILE_ID = "1AbC1234XYZ"
FILE_ID = "1bxeLow-PCHdOQqzWkWOcB3L32MPC3RBU"
MODEL_URL = f"https://drive.google.com/file/d/1bxeLow-PCHdOQqzWkWOcB3L32MPC3RBU/view?usp=sharing"
MODEL_PATH = "model.pkl"

# -------------------------------
# Download Model from Google Drive (if not cached)
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    st.write(f"Downloaded model size: {os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB")

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌦 Predictive Analysis of Meteorological Events")
st.write("A machine learning app to forecast meteorological events using Random Forest.")

# Sidebar for inputs
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


