#!/usr/bin/env python3
"""
FastAPI app for IMDb Sentiment Analysis
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, re
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Load models
NB_MODEL_PATH = MODELS_DIR / "imdb_nb_pipeline.joblib"
TOKENIZER_PATH = MODELS_DIR / "imdb_tokenizer.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "imdb_lstm_model.h5"

nb_model = joblib.load(NB_MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Constants
MAX_LEN = 200

# Input schema
class ReviewInput(BaseModel):
    review: str

# Text cleaning (same as training)
def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

# Initialize app
app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="Predicts sentiment (positive/negative) using Naive Bayes or LSTM",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "models": {
            "Naive Bayes": NB_MODEL_PATH.name,
            "LSTM": LSTM_MODEL_PATH.name
        },
        "metrics": metrics,
        "classes": {"0": "Negative", "1": "Positive"}
    }

@app.post("/predict_nb")
def predict_nb(data: ReviewInput):
    text = clean_text(data.review)
    pred = nb_model.predict([text])[0]
    return {"prediction": int(pred), "label": "Positive" if pred == 1 else "Negative"}

@app.post("/predict_lstm")
def predict_lstm(data: ReviewInput):
    text = clean_text(data.review)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = lstm_model.predict(padded)[0][0]
    pred = int(prob > 0.5)
    return {
        "prediction": pred,
        "label": "Positive" if pred == 1 else "Negative",
        "probability": float(prob)
    }