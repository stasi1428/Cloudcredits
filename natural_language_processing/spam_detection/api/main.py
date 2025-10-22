#!/usr/bin/env python3
"""
FastAPI app for Spam Detection (Enron Dataset)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Load model (either Naive Bayes or SVM depending on training outcome)
MODEL_PATHS = list(MODELS_DIR.glob("spam_*_pipeline.joblib"))
if not MODEL_PATHS:
    raise FileNotFoundError("No trained spam detection model found in models/ directory")
MODEL_PATH = MODEL_PATHS[0]
model = joblib.load(MODEL_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Define input schema
class EmailInput(BaseModel):
    message: str

# Initialize app
app = FastAPI(
    title="Spam Detection API",
    description="Classifies emails as ham or spam using the tuned best model (Naive Bayes or Linear SVM)",
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
        "classes": {"0": "Ham", "1": "Spam"}
    }

@app.post("/predict")
def predict(data: EmailInput):
    pred = model.predict([data.message])[0]
    return {"prediction": int(pred), "label": "Spam" if pred == 1 else "Ham"}