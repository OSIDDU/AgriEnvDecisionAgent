import os
import numpy as np
import joblib

MODEL_DIR = "models"

_yield_model = joblib.load(os.path.join(MODEL_DIR, "yield_model.pkl"))
_aqi_model = joblib.load(os.path.join(MODEL_DIR, "aqi_model.pkl"))


def predict_yield_tool(soil_pH, N, P, K, rainfall, temperature) -> float:
    X = np.array([[soil_pH, N, P, K, rainfall, temperature]])
    return float(_yield_model.predict(X)[0])


def predict_aqi_tool(pm25, pm10, co, no2, temp, humidity) -> float:
    X = np.array([[pm25, pm10, co, no2, temp, humidity]])
    return float(_aqi_model.predict(X)[0])

