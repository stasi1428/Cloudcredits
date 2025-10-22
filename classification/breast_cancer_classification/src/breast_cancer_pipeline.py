#!/usr/bin/env python3
"""
Breast Cancer Classification Pipeline
"""

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "breast_cancer_wisconsin.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    if "" in df.columns:
        df = df.drop(columns=[""])
    df["y"] = df["y"].map({"B": 0, "M": 1})
    return df

def run_eda(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="y", data=df)
    plt.title("Benign (0) vs Malignant (1)")
    plt.savefig(RESULTS_DIR / "class_balance.png")
    plt.close()

    plt.figure(figsize=(12,10))
    sns.heatmap(df.drop(columns=["y"]).corr(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation")
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png")
    plt.close()

def train_and_evaluate(df):
    X = df.drop(columns=["y"])
    y = df["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    svm_clf = SVC(kernel="rbf", probability=True, random_state=42)
    rf_clf  = RandomForestClassifier(random_state=42)

    svm_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    # Hyperparameter tuning for RF
    rf_params = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
    grid_rf = GridSearchCV(rf_clf, rf_params, cv=5, scoring="recall", n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_

    metrics = {}
    for name, model in [("SVM", svm_clf), ("RandomForest", rf_clf), ("Tuned RF", best_rf)]:
        preds = model.predict(X_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "classification_report": classification_report(y_test, preds, target_names=["Benign","Malignant"], output_dict=True)
        }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(scaler, MODELS_DIR / "breast_scaler.pkl")
    joblib.dump(svm_clf, MODELS_DIR / "breast_svm.pkl")
    joblib.dump(best_rf, MODELS_DIR / "breast_rf.pkl")

    return metrics

def main():
    df = load_and_clean()
    run_eda(df)
    results = train_and_evaluate(df)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()