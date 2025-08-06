#!/usr/bin/env python3
"""
movielens_recommender.py

A full workflow to build & evaluate a MovieLens-based
collaborative filtering recommender:

1. Define Problem
2. Load & Inspect Data
3. Build Surprise Dataset
4. Train/Test Split
5. Model Selection (SVD & KNN)
6. Training
7. Evaluation (RMSE)
8. Hyperparameter Tuning
9. Persist Best Model
10. (Bonus) Generate Recommendations
"""

# 0. Ensure scikit-surprise is installed:
#    pip install scikit-surprise

import pandas as pd
from pathlib import Path

# 1. Define the Problem
#    Given past user ratings, predict unknown ratings and recommend top-N movies.

# 2. Load & Inspect Data
DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts"
    "/Cloudcredits Datasets/MovieLens_ratings.csv"
)
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print(df.head())
print("\nRating distribution:\n", df['rating'].describe())
print("Unique users:", df['userId'].nunique())
print("Unique movies:", df['movieId'].nunique(), "\n")

# 3. Build Surprise Dataset
try:
    from surprise import Reader, Dataset
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The 'surprise' library is not installed. "
        "Install it with:\n\n    pip install scikit-surprise\n"
    ) from e

# MovieLens ratings go from 0.5 to 5.0
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# 4. Train/Test Split
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 5. Model Selection
from surprise import SVD, KNNBasic

algo_svd = SVD(
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

algo_knn = KNNBasic(
    sim_options={'name': 'cosine', 'user_based': False},
    verbose=False
)

# 6. Training
print("Training SVD...")
algo_svd.fit(trainset)

print("Training item-based KNN...")
algo_knn.fit(trainset)

# 7. Evaluation
from surprise import accuracy

print("\nEvaluating on test set:")
pred_svd = algo_svd.test(testset)
pred_knn = algo_knn.test(testset)

rmse_svd = accuracy.rmse(pred_svd, verbose=True)
rmse_knn = accuracy.rmse(pred_knn, verbose=True)

# 8. Hyperparameter Tuning (Grid Search on SVD)
from surprise.model_selection import GridSearchCV

param_grid = {
    'n_factors': [20, 50, 100],
    'n_epochs':  [10, 20, 30],
    'lr_all':    [0.002, 0.005],
    'reg_all':   [0.02, 0.05]
}

gs = GridSearchCV(
    SVD, param_grid,
    measures=['rmse'],
    cv=3,
    n_jobs=-1,
    joblib_verbose=1
)
print("\nStarting Grid Search for SVD hyperparameters...")
gs.fit(data)

print("Best SVD RMSE:", gs.best_score['rmse'])
print("Best params:", gs.best_params['rmse'])

best_svd = gs.best_estimator['rmse']
best_svd.fit(trainset)
pred_best = best_svd.test(testset)
best_rmse = accuracy.rmse(pred_best, verbose=True)

# 9. Persist Best Model
import pickle

MODEL_PATH = Path(__file__).parent / "best_svd_model.pkl"
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_svd, f)
print(f"\nSaved best SVD model to: {MODEL_PATH}")

# 10. (Bonus) Generate Top-10 Recommendations for a Given User
def recommend(user_id, algo, n=10):
    """Return top-n movie IDs predicted for user_id."""
    all_movie_ids = df['movieId'].unique()
    seen = df[df.userId == user_id]['movieId'].tolist()
    candidates = [m for m in all_movie_ids if m not in seen]
    preds = [(m, algo.predict(user_id, m).est) for m in candidates]
    top_n = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    return top_n

if __name__ == "__main__":
    user = 1
    print(f"\nTop 10 recommendations for user {user}:")
    for movie_id, est_rating in recommend(user, best_svd, n=10):
        print(f"MovieID {movie_id} — Predicted Rating: {est_rating:.2f}")