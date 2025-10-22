#!/usr/bin/env python3
"""
FastAPI app for Iris Flower Classification
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
MODEL_PATH = MODELS_DIR / "iris_classifier.joblib"
model = joblib.load(MODEL_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Define input schema (Iris dataset features)
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize app
app = FastAPI(
    title="Iris Classification API",
    description="Predicts iris species using a tuned Decision Tree model",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": "Decision Tree (tuned)",
        "metrics": metrics,
        "classes": ["setosa", "versicolor", "virginica"]
    }

@app.post("/predict")
def predict(features: IrisFeatures):
    X = np.array([[features.sepal_length, features.sepal_width,
                   features.petal_length, features.petal_width]])
    pred = model.predict(X)[0]
    return {"prediction": int(pred)}