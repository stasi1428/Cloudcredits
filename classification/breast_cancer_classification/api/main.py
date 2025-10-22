#!/usr/bin/env python3
"""
FastAPI app for Breast Cancer Classification (Wisconsin Dataset)
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

# Load scaler and model (using tuned Random Forest as default)
SCALER_PATH = MODELS_DIR / "breast_scaler.pkl"
MODEL_PATH = MODELS_DIR / "breast_rf.pkl"
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Define input schema (all numeric features from dataset except target 'y')
class CancerFeatures(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

# Initialize app
app = FastAPI(
    title="Breast Cancer Classification API",
    description="Predicts tumor type (Benign or Malignant) using a tuned Random Forest model",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": MODEL_PATH.name,
        "scaler": SCALER_PATH.name,
        "metrics": metrics,
        "classes": {"0": "Benign", "1": "Malignant"}
    }

@app.post("/predict")
def predict(features: CancerFeatures):
    # Convert input to numpy array
    X = np.array([[getattr(features, field) for field in features.__fields__]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return {"prediction": int(pred), "label": "Malignant" if pred == 1 else "Benign"}