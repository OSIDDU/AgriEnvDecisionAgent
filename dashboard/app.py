import streamlit as st
import requests

st.set_page_config(page_title="Agri-Env Decision Agent", layout="centered")
st.title("Agri-Env Decision Agent")

api_url = st.text_input("API base URL", "http://localhost:8000")

st.header("🌾 Crop Yield Prediction")
with st.form("yield_form"):
    soil_pH = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    N = st.number_input("Nitrogen (N)", 0.0, 300.0, 40.0)
    P = st.number_input("Phosphorus (P)", 0.0, 300.0, 25.0)
    K = st.number_input("Potassium (K)", 0.0, 300.0, 20.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 300.0)
    temperature = st.number_input("Temperature (°C)", -20.0, 60.0, 28.0)
    submitted_yield = st.form_submit_button("Predict Yield")
    if submitted_yield:
        payload = {
            "soil_pH": soil_pH,
            "N": N,
            "P": P,
            "K": K,
            "rainfall": rainfall,
            "temperature": temperature
        }
        try:
            r = requests.post(f"{api_url}/predict/yield", json=payload, timeout=5)
            st.json(r.json())
        except Exception as e:
            st.error(f"API error: {e}")

st.header("🌫 AQI Prediction")
with st.form("aqi_form"):
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 120.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 170.0)
    co = st.number_input("CO (ppm)", 0.0, 50.0, 0.9)
    no2 = st.number_input("NO2 (µg/m³)", 0.0, 1000.0, 48.0)
    temp = st.number_input("Temperature (°C)", -20.0, 60.0, 30.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    submitted_aqi = st.form_submit_button("Predict AQI")
    if submitted_aqi:
        payload = {
            "pm25": pm25,
            "pm10": pm10,
            "co": co,
            "no2": no2,
            "temp": temp,
            "humidity": humidity
        }
        try:
            r = requests.post(f"{api_url}/predict/aqi", json=payload, timeout=5)
            st.json(r.json())
        except Exception as e:
            st.error(f"API error: {e}")
st.markdown("---")
