#!/usr/bin/env python3
"""
Iris Classification Pipeline
"""

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, json, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "iris.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna()
    return df

def run_eda(df):
    sns.pairplot(df, hue="species", corner=True)
    plt.suptitle("Pairplot of Iris Features", y=1.02)
    plt.savefig(RESULTS_DIR / "pairplot.png")
    plt.close()

def train_and_evaluate(df):
    X = df.drop("species", axis=1)
    y = LabelEncoder().fit_transform(df["species"])

    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer([("scale", StandardScaler(), numeric_features)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Models
    dt_pipeline = Pipeline([("preprocess", preprocessor),
                            ("classifier", DecisionTreeClassifier(random_state=42))])
    lr_pipeline = Pipeline([("preprocess", preprocessor),
                            ("classifier", LogisticRegression(max_iter=200, random_state=42))])

    dt_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    models = {"Decision Tree": dt_pipeline, "Logistic Regression": lr_pipeline}
    metrics = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        cm    = confusion_matrix(y_test, preds).tolist()
        metrics[name] = {"accuracy": float(acc), "confusion_matrix": cm}

    # Hyperparameter tuning for Decision Tree
    param_grid = {"classifier__max_depth": [2, 3, 4, 5, None]}
    grid_dt = GridSearchCV(dt_pipeline, param_grid, cv=5)
    grid_dt.fit(X_train, y_train)
    best_dt = grid_dt.best_estimator_
    best_acc = accuracy_score(y_test, best_dt.predict(X_test))
    metrics["Tuned Decision Tree"] = {
        "accuracy": float(best_acc),
        "best_params": grid_dt.best_params_
    }

    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save best model
    joblib.dump(best_dt, MODELS_DIR / "iris_classifier.joblib")

    return metrics

def main():
    df = load_and_clean()
    run_eda(df)
    results = train_and_evaluate(df)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()