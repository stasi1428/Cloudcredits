#!/usr/bin/env python3
"""
FastAPI app for Stock Price Forecasting with LSTM
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib, json
from pathlib import Path
import tensorflow as tf

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Load model + scaler
MODEL_PATH = next(MODELS_DIR.glob("lstm_stock_*.h5"))
SCALER_PATH = next(MODELS_DIR.glob("scaler_stock_*.pkl"))
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Define input schema
class SequenceInput(BaseModel):
    sequence: list[float]  # last 60 normalized price values

app = FastAPI(
    title="Stock Price Forecasting API",
    description="Predicts next-day stock price using an LSTM model",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": str(MODEL_PATH.name),
        "scaler": str(SCALER_PATH.name),
        "metrics": metrics
    }

@app.post("/predict")
def predict(data: SequenceInput):
    seq = np.array(data.sequence).reshape(1, len(data.sequence), 1)
    pred_scaled = model.predict(seq)
    pred = scaler.inverse_transform(pred_scaled)
    return {"prediction": float(pred[0][0])}