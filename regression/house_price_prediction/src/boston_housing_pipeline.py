#!/usr/bin/env python3
"""
Boston Housing Pipeline
"""

import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
import json, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "BostonHousing.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna()
    return df

def run_eda(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png")
    plt.close()

def build_preprocessor(X):
    numeric_features = X.columns.tolist()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler = StandardScaler()
    return ColumnTransformer([
        ("poly", poly, numeric_features),
        ("scale", scaler, numeric_features)
    ])

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    results = {}

    # Linear Regression
    base = Pipeline([("preprocess", preprocessor), ("regressor", LinearRegression())])
    base.fit(X_train, y_train)
    y_pred = base.predict(X_test)
    results["linear_regression"] = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    # Ridge
    ridge = Pipeline([("preprocess", preprocessor),
                      ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results["ridge"] = {
        "alpha": float(ridge.named_steps["ridge"].alpha_),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    # Lasso
    lasso = Pipeline([("preprocess", preprocessor),
                      ("lasso", LassoCV(alphas=np.logspace(-3, 1, 30), cv=5, max_iter=5000))])
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    results["lasso"] = {
        "alpha": float(lasso.named_steps["lasso"].alpha_),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # Save best model (example: ridge)
    joblib.dump(ridge, MODELS_DIR / "house_price_model.joblib")

    return results

def main():
    df = load_and_clean()
    run_eda(df)
    X, y = df.drop("medv", axis=1), df["medv"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X)
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()