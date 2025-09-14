import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load Trained Model
# -------------------------------
# Make sure you save your trained model as model.pkl after training
# Example:
# with open("model.pkl", "wb") as f:
#     pickle.dump(best_rf, f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Weather Event Predictor", page_icon="🌦", layout="centered")

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
