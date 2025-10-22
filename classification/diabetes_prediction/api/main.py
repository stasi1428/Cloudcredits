#!/usr/bin/env python3
"""
FastAPI app for Diabetes Prediction (Pima Indians Dataset)
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

# Load model (either KNN or Logistic Regression depending on training outcome)
MODEL_PATHS = list(MODELS_DIR.glob("diabetes_*_pipeline.joblib"))
if not MODEL_PATHS:
    raise FileNotFoundError("No trained diabetes model found in models/ directory")
MODEL_PATH = MODEL_PATHS[0]
model = joblib.load(MODEL_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Define input schema (Pima Indians dataset features)
class PatientFeatures(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Initialize app
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes outcome using the tuned best model (KNN or Logistic Regression)",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": MODEL_PATH.name,
        "metrics": metrics,
        "classes": {"0": "No Diabetes", "1": "Diabetes"}
    }

@app.post("/predict")
def predict(features: PatientFeatures):
    X = np.array([[features.Pregnancies, features.Glucose, features.BloodPressure,
                   features.SkinThickness, features.Insulin, features.BMI,
                   features.DiabetesPedigreeFunction, features.Age]])
    pred = model.predict(X)[0]
    return {"prediction": int(pred), "label": "Diabetes" if pred == 1 else "No Diabetes"}