from fastapi import FastAPI
from pydantic import BaseModel
from agent.tools import predict_yield_tool, predict_aqi_tool

app = FastAPI(title="Agri-Env Decision Agent API")


class YieldInput(BaseModel):
    soil_pH: float
    N: float
    P: float
    K: float
    rainfall: float
    temperature: float


class AQIInput(BaseModel):
    pm25: float
    pm10: float
    co: float
    no2: float
    temp: float
    humidity: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/yield")
def predict_yield(body: YieldInput):
    y = predict_yield_tool(**body.dict())
    advisory = "OK"
    if y < 3:
        advisory = "Low yield expected – adjust inputs."
    return {"predicted_yield": y, "advisory": advisory}


@app.post("/predict/aqi")
def predict_aqi(body: AQIInput):
    a = predict_aqi_tool(**body.dict())
    advisory = "OK"
    if a > 150:
        advisory = "Unhealthy AQI – limit exposure."
    return {"predicted_aqi": a, "advisory": advisory}

