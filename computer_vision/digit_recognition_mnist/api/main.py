#!/usr/bin/env python3
"""
FastAPI app for MNIST Digit Recognition (CNN)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
from pathlib import Path
import tensorflow as tf

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Load model
MODEL_PATH = MODELS_DIR / "mnist_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Define input schema (flattened 28x28 grayscale image)
class DigitImage(BaseModel):
    pixels: list[float]  # length = 784

# Initialize app
app = FastAPI(
    title="MNIST Digit Recognition API",
    description="Predicts handwritten digits (0–9) using a trained CNN model",
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
        "classes": list(range(10))
    }

@app.post("/predict")
def predict(data: DigitImage):
    arr = np.array(data.pixels, dtype="float32").reshape(1, 28, 28, 1) / 255.0
    probs = model.predict(arr)[0]
    pred = int(np.argmax(probs))
    return {
        "prediction": pred,
        "probabilities": probs.tolist()
    }