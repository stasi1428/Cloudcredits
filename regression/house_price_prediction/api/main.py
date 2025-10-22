#!/usr/bin/env python3
"""
FastAPI app for Boston Housing Price Prediction
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Load model
MODEL_PATH = MODELS_DIR / "house_price_model.joblib"
model = joblib.load(MODEL_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Define input schema (all features from BostonHousing.csv except target 'medv')
class HouseFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    b: float
    lstat: float

# Initialize app
app = FastAPI(
    title="Boston Housing Price Prediction API",
    description="Predicts median home values using Ridge Regression",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": "Ridge Regression (joblib)",
        "metrics": metrics
    }

@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert input to numpy array
    X = np.array([[features.crim, features.zn, features.indus, features.chas,
                   features.nox, features.rm, features.age, features.dis,
                   features.rad, features.tax, features.ptratio, features.b,
                   features.lstat]])
    # Predict
    pred = model.predict(X)
    return {"prediction": float(pred[0])}