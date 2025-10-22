#!/usr/bin/env python3
"""
MovieLens Recommendation Pipeline
"""

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, json, pickle
from pathlib import Path
from surprise import Reader, Dataset, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "MovieLens_ratings.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    return df

def run_eda(df):
    plt.figure(figsize=(6,4))
    sns.histplot(df['rating'], bins=20, kde=False)
    plt.title("Rating Distribution")
    plt.savefig(RESULTS_DIR / "rating_distribution.png")
    plt.close()

def train_and_evaluate(df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId','movieId','rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    algo_svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    algo_knn = KNNBasic(sim_options={'name':'cosine','user_based':False}, verbose=False)

    algo_svd.fit(trainset)
    algo_knn.fit(trainset)

    pred_svd = algo_svd.test(testset)
    pred_knn = algo_knn.test(testset)

    rmse_svd = accuracy.rmse(pred_svd, verbose=False)
    rmse_knn = accuracy.rmse(pred_knn, verbose=False)

    # Hyperparameter tuning
    param_grid = {
        'n_factors':[20,50,100],
        'n_epochs':[10,20,30],
        'lr_all':[0.002,0.005],
        'reg_all':[0.02,0.05]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
    gs.fit(data)
    best_svd = gs.best_estimator['rmse']
    best_svd.fit(trainset)
    pred_best = best_svd.test(testset)
    best_rmse = accuracy.rmse(pred_best, verbose=False)

    metrics = {
        "SVD": {"rmse": float(rmse_svd)},
        "KNN": {"rmse": float(rmse_knn)},
        "Tuned SVD": {"rmse": float(best_rmse), "best_params": gs.best_params['rmse']}
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(MODELS_DIR / "best_svd_model.pkl", 'wb') as f:
        pickle.dump(best_svd, f)

    return best_svd, metrics

def recommend(user_id, algo, df, n=10):
    all_movie_ids = df['movieId'].unique()
    seen = df[df.userId == user_id]['movieId'].tolist()
    candidates = [m for m in all_movie_ids if m not in seen]
    preds = [(m, algo.predict(user_id, m).est) for m in candidates]
    return sorted(preds, key=lambda x: x[1], reverse=True)[:n]

def main():
    df = load_and_clean()
    run_eda(df)
    best_svd, results = train_and_evaluate(df)
    print(json.dumps(results, indent=4))
    user = 1
    print(f"\nTop 10 recommendations for user {user}:")
    for movie_id, est_rating in recommend(user, best_svd, df, n=10):
        print(f"MovieID {movie_id} — Predicted Rating: {est_rating:.2f}")

if __name__ == "__main__":
    main()