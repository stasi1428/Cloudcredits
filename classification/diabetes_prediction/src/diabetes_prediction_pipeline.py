#!/usr/bin/env python3
"""
Diabetes Prediction Pipeline (Pima Indians Dataset)
"""

import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "pima_indian_diabetes.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    for col in zero_cols:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def run_eda(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="Outcome", data=df, palette="Set2")
    plt.title("Outcome Distribution")
    plt.savefig(RESULTS_DIR / "outcome_distribution.png")
    plt.close()

    features = df.columns.drop("Outcome")
    for col in features:
        plt.figure()
        sns.histplot(df[col], kde=True, color="steelblue")
        plt.title(f"{col} Distribution")
        plt.savefig(RESULTS_DIR / f"{col}_distribution.png")
        plt.close()

        plt.figure()
        sns.boxplot(x="Outcome", y=col, data=df, palette="Pastel1")
        plt.title(f"{col} by Outcome")
        plt.savefig(RESULTS_DIR / f"{col}_boxplot.png")
        plt.close()

    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Matrix")
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png")
    plt.close()

def train_and_evaluate(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn_pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    lr_pipeline  = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(solver="liblinear", random_state=42))])

    knn_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    models = {"KNN": knn_pipeline, "Logistic Regression": lr_pipeline}
    metrics = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "classification_report": classification_report(y_test, preds, digits=3, output_dict=True)
        }

    # Hyperparameter tuning
    knn_params = {"knn__n_neighbors": [3, 5, 7, 9], "knn__weights": ["uniform", "distance"]}
    grid_knn = GridSearchCV(knn_pipeline, knn_params, cv=5, scoring="f1", n_jobs=-1)
    grid_knn.fit(X_train, y_train)

    lr_params = {"lr__C": [0.01, 0.1, 1.0, 10.0], "lr__penalty": ["l1", "l2"]}
    grid_lr = GridSearchCV(lr_pipeline, lr_params, cv=5, scoring="f1", n_jobs=-1)
    grid_lr.fit(X_train, y_train)

    knn_f1 = f1_score(y_test, grid_knn.best_estimator_.predict(X_test))
    lr_f1  = f1_score(y_test, grid_lr.best_estimator_.predict(X_test))

    if knn_f1 >= lr_f1:
        best_model, model_name = grid_knn.best_estimator_, "knn"
    else:
        best_model, model_name = grid_lr.best_estimator_, "logreg"

    metrics["Tuned Best"] = {"model": model_name, "f1": float(max(knn_f1, lr_f1))}

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(best_model, MODELS_DIR / f"diabetes_{model_name}_pipeline.joblib")
    return metrics

def main():
    df = load_and_clean()
    run_eda(df)
    results = train_and_evaluate(df)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()