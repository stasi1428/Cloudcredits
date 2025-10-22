#!/usr/bin/env python3
"""
FastAPI app for Titanic Survival Prediction
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
MODEL_PATH = MODELS_DIR / "titanic_model.joblib"
model = joblib.load(MODEL_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Define input schema (features after preprocessing/feature engineering)
class PassengerFeatures(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str
    family_size: int

# Initialize app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts passenger survival using a tuned Random Forest model",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": "Random Forest (tuned)",
        "metrics": metrics,
        "classes": {"0": "Did not survive", "1": "Survived"}
    }

@app.post("/predict")
def predict(features: PassengerFeatures):
    X = np.array([[features.pclass, features.sex, features.age,
                   features.sibsp, features.parch, features.fare,
                   features.embarked, features.family_size]], dtype=object)
    pred = model.predict(X)[0]
    return {"prediction": int(pred), "label": "Survived" if pred == 1 else "Did not survive"}