#!/usr/bin/env python3
"""
FastAPI app for MovieLens Recommendation System
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle, json
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "data" / "MovieLens_ratings.csv"

# Load model
MODEL_PATH = MODELS_DIR / "best_svd_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load metrics
METRICS_PATH = RESULTS_DIR / "metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Load dataset (for candidate movies)
df = pd.read_csv(DATA_PATH)

# Input schema
class UserRequest(BaseModel):
    userId: int
    n: int = 10  # number of recommendations

# Initialize app
app = FastAPI(
    title="MovieLens Recommendation API",
    description="Generates top-N movie recommendations using a tuned SVD model",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": MODEL_PATH.name,
        "metrics": metrics
    }

@app.post("/recommend")
def recommend(req: UserRequest):
    user_id = req.userId
    n = req.n
    all_movie_ids = df['movieId'].unique()
    seen = df[df.userId == user_id]['movieId'].tolist()
    candidates = [m for m in all_movie_ids if m not in seen]
    preds = [(m, model.predict(user_id, m).est) for m in candidates]
    top_n = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    return {
        "userId": user_id,
        "recommendations": [
            {"movieId": int(mid), "predicted_rating": float(rating)}
            for mid, rating in top_n
        ]
    }