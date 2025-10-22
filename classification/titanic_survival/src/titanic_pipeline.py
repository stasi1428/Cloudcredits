#!/usr/bin/env python3
"""
Titanic Survival Classification Pipeline
"""

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "titanic.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    # Feature engineering
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df = df.drop(["passengerid", "name", "ticket", "cabin"], axis=1)
    return df

def run_eda(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="survived", data=df)
    plt.title("Survival Counts")
    plt.savefig(RESULTS_DIR / "survival_counts.png")
    plt.close()

    plt.figure(figsize=(8,4))
    sns.kdeplot(df.loc[df.survived == 0, "age"].dropna(), label="Not Survived", shade=True)
    sns.kdeplot(df.loc[df.survived == 1, "age"].dropna(), label="Survived", shade=True)
    plt.title("Age Distribution by Survival")
    plt.legend()
    plt.savefig(RESULTS_DIR / "age_distribution.png")
    plt.close()

    num_cols = ["pclass", "age", "sibsp", "parch", "fare", "survived"]
    plt.figure(figsize=(6,5))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numeric Feature Correlations")
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png")
    plt.close()

def train_and_evaluate(df):
    y = df["survived"].values
    X = df.drop("survived", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_feats = ["pclass", "age", "sibsp", "parch", "fare", "family_size"]
    categorical_feats = ["sex", "embarked"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_feats),
        ("cat", OneHotEncoder(drop="first"), categorical_feats)
    ])

    logreg_pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", LogisticRegression(max_iter=500, random_state=42))
    ])

    rf_pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    logreg_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    models = {"Logistic Regression": logreg_pipeline, "Random Forest": rf_pipeline}
    metrics = {}
    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec  = recall_score(y_test, preds)
        metrics[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "classification_report": classification_report(y_test, preds, digits=3, output_dict=True)
        }

    # Hyperparameter tuning for Random Forest
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5]
    }
    grid_rf = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    tuned_preds = best_rf.predict(X_test)
    metrics["Tuned Random Forest"] = {
        "accuracy": float(accuracy_score(y_test, tuned_preds)),
        "precision": float(precision_score(y_test, tuned_preds)),
        "recall": float(recall_score(y_test, tuned_preds)),
        "best_params": grid_rf.best_params_
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(best_rf, MODELS_DIR / "titanic_model.joblib")
    return metrics

def main():
    df = load_and_clean()
    run_eda(df)
    results = train_and_evaluate(df)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()