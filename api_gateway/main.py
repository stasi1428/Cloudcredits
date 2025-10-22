#!/usr/bin/env python3
"""
Unified API Gateway for All ML Projects
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Base paths
BASE_DIR = Path(__file__).parent.parent

# Initialize app
app = FastAPI(
    title="Unified ML API Gateway",
    description="Single entry point for all ML project APIs",
    version="1.0"
)

# ---------------------------
# 1. Boston Housing
# ---------------------------
house_model = joblib.load(BASE_DIR / "boston_housing/models/house_price_model.joblib")
with open(BASE_DIR / "boston_housing/results/metrics.json") as f:
    house_metrics = json.load(f)

class HouseFeatures(BaseModel):
    crim: float; zn: float; indus: float; chas: int; nox: float; rm: float
    age: float; dis: float; rad: int; tax: float; ptratio: float; b: float; lstat: float

@app.post("/predict/house")
def predict_house(data: HouseFeatures):
    X = np.array([[getattr(data, f) for f in data.__fields__]])
    pred = house_model.predict(X)[0]
    return {"prediction": float(pred)}

# ---------------------------
# 2. Titanic Survival
# ---------------------------
titanic_model = joblib.load(BASE_DIR / "titanic_survival/models/titanic_model.joblib")
with open(BASE_DIR / "titanic_survival/results/metrics.json") as f:
    titanic_metrics = json.load(f)

class PassengerFeatures(BaseModel):
    pclass: int; sex: str; age: float; sibsp: int; parch: int
    fare: float; embarked: str; family_size: int

@app.post("/predict/titanic")
def predict_titanic(data: PassengerFeatures):
    X = np.array([[getattr(data, f) for f in data.__fields__]], dtype=object)
    pred = titanic_model.predict(X)[0]
    return {"prediction": int(pred), "label": "Survived" if pred == 1 else "Did not survive"}

# ---------------------------
# 3. Diabetes Prediction
# ---------------------------
diabetes_model = joblib.load(list((BASE_DIR / "diabetes_prediction/models").glob("diabetes_*_pipeline.joblib"))[0])
with open(BASE_DIR / "diabetes_prediction/results/metrics.json") as f:
    diabetes_metrics = json.load(f)

class PatientFeatures(BaseModel):
    Pregnancies: int; Glucose: float; BloodPressure: float; SkinThickness: float
    Insulin: float; BMI: float; DiabetesPedigreeFunction: float; Age: int

@app.post("/predict/diabetes")
def predict_diabetes(data: PatientFeatures):
    X = np.array([[getattr(data, f) for f in data.__fields__]])
    pred = diabetes_model.predict(X)[0]
    return {"prediction": int(pred), "label": "Diabetes" if pred == 1 else "No Diabetes"}

# ---------------------------
# 4. Breast Cancer
# ---------------------------
scaler = joblib.load(BASE_DIR / "breast_cancer_classification/models/breast_scaler.pkl")
breast_model = joblib.load(BASE_DIR / "breast_cancer_classification/models/breast_rf.pkl")
with open(BASE_DIR / "breast_cancer_classification/results/metrics.json") as f:
    breast_metrics = json.load(f)

class CancerFeatures(BaseModel):
    # Only showing a few; include all 30 features in practice
    radius_mean: float; texture_mean: float; perimeter_mean: float; area_mean: float
    smoothness_mean: float; compactness_mean: float; concavity_mean: float
    concave_points_mean: float; symmetry_mean: float; fractal_dimension_mean: float
    # ... (add the rest)

@app.post("/predict/cancer")
def predict_cancer(data: CancerFeatures):
    X = np.array([[getattr(data, f) for f in data.__fields__]])
    X_scaled = scaler.transform(X)
    pred = breast_model.predict(X_scaled)[0]
    return {"prediction": int(pred), "label": "Malignant" if pred == 1 else "Benign"}

# ---------------------------
# 5. MNIST Digit Recognition
# ---------------------------
mnist_model = tf.keras.models.load_model(BASE_DIR / "mnist_digit_recognition/models/mnist_cnn.h5")
with open(BASE_DIR / "mnist_digit_recognition/results/metrics.json") as f:
    mnist_metrics = json.load(f)

class DigitImage(BaseModel):
    pixels: list[float]  # length 784

@app.post("/predict/mnist")
def predict_mnist(data: DigitImage):
    arr = np.array(data.pixels, dtype="float32").reshape(1, 28, 28, 1) / 255.0
    probs = mnist_model.predict(arr)[0]
    pred = int(np.argmax(probs))
    return {"prediction": pred, "probabilities": probs.tolist()}

# ---------------------------
# 6. IMDb Sentiment
# ---------------------------
nb_model = joblib.load(BASE_DIR / "imdb_sentiment/models/imdb_nb_pipeline.joblib")
tokenizer = joblib.load(BASE_DIR / "imdb_sentiment/models/imdb_tokenizer.joblib")
lstm_model = tf.keras.models.load_model(BASE_DIR / "imdb_sentiment/models/imdb_lstm_model.h5")
with open(BASE_DIR / "imdb_sentiment/results/metrics.json") as f:
    imdb_metrics = json.load(f)

MAX_LEN = 200
class ReviewInput(BaseModel):
    review: str

@app.post("/predict/imdb/nb")
def predict_imdb_nb(data: ReviewInput):
    pred = nb_model.predict([data.review])[0]
    return {"prediction": int(pred), "label": "Positive" if pred == 1 else "Negative"}

@app.post("/predict/imdb/lstm")
def predict_imdb_lstm(data: ReviewInput):
    seq = tokenizer.texts_to_sequences([data.review])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = lstm_model.predict(padded)[0][0]
    pred = int(prob > 0.5)
    return {"prediction": pred, "label": "Positive" if pred == 1 else "Negative", "probability": float(prob)}

# ---------------------------
# 7. Spam Detection
# ---------------------------
spam_model = joblib.load(list((BASE_DIR / "spam_detection/models").glob("spam_*_pipeline.joblib"))[0])
with open(BASE_DIR / "spam_detection/results/metrics.json") as f:
    spam_metrics = json.load(f)

class EmailInput(BaseModel):
    message: str

@app.post("/predict/spam")
def predict_spam(data: EmailInput):
    pred = spam_model.predict([data.message])[0]
    return {"prediction": int(pred), "label": "Spam" if pred == 1 else "Ham"}

# ---------------------------
# 8. MovieLens Recommendation
# ---------------------------
with open(BASE_DIR / "movie_recommendation/models/best_svd_model.pkl", "rb") as f:
    recommender = pickle.load(f)
with open(BASE_DIR / "movie_recommendation/results/metrics.json") as f:
    rec_metrics = json.load(f)
df_movies = pd.read_csv(BASE_DIR / "movie_recommendation/data/MovieLens_ratings.csv")

class UserRequest(BaseModel):
    userId: int
    n: int = 10

@app.post("/recommend/movies")
def recommend_movies(req: UserRequest):
    all_movie_ids = df_movies['movieId'].unique()
    seen = df_movies[df_movies.userId == req.userId]['movieId'].tolist()
    candidates = [m for m in all_movie_ids if m not in seen]
    preds = [(m, recommender.predict(req.userId, m).est) for m in candidates]
    top_n = sorted(preds, key=lambda x: x[1], reverse=True)[:req.n]
    return {"userId": req.userId,
            "recommendations": [{"movieId": int(mid), "predicted_rating": float(r)} for mid, r in top_n]}