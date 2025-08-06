#!/usr/bin/env python3
"""
diabetes_prediction_pipeline.py

A complete ML workflow to predict diabetes onset for Pima Indian patients:
1. Define Problem
2. Load & Clean Data
3. Exploratory Data Analysis
4. Feature Engineering
5. Train/Test Split
6. Model Selection
7. Training
8. Evaluation
9. Hyperparameter Tuning
10. Deployment (serialize best model)

Dataset

The Pima Indians Diabetes CSV has columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 = no diabetes, 1 = diabetes)
"""

# 1. Define the Problem
#    Binary classification: predict Outcome (1 if diabetic, 0 otherwise).

# 2. Load & Clean Data
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts"
    "/Cloudcredits Datasets/pima_indian_diabetes.csv"
)

df = pd.read_csv(DATA_PATH)
print("Initial shape:", df.shape)
print("Outcome distribution:\n", df["Outcome"].value_counts(), "\n")

# Replace zeroes in columns where zero is invalid with NaN
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Impute missing values with median
for col in zero_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

print("Missing values after imputation:\n", df.isna().sum(), "\n")

# 3. Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# 3.1 Outcome count
plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=df, palette="Set2")
plt.title("Outcome Distribution (0 = No Diabetes, 1 = Diabetes)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()

# 3.2 Feature distributions
features = df.columns.drop("Outcome")
plt.figure(figsize=(14,10))
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True, color="steelblue")
    plt.title(f"{col} Distribution")
plt.tight_layout()
plt.show()

# 3.3 Boxplots to check outliers
plt.figure(figsize=(14,10))
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x="Outcome", y=col, data=df, palette="Pastel1")
    plt.title(f"{col} by Outcome")
plt.tight_layout()
plt.show()

# 3.4 Correlation heatmap
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Matrix")
plt.show()

# 4. Feature Engineering
X = df.drop("Outcome", axis=1)
y = df["Outcome"].values

# 5. Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Model Selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn",    KNeighborsClassifier())
])

logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(solver="liblinear", random_state=42))
])

# 7. Training
print("Training K-Nearest Neighbors...")
knn_pipeline.fit(X_train, y_train)

print("Training Logistic Regression...")
logreg_pipeline.fit(X_train, y_train)

# 8. Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def evaluate(name, model, X_val, y_val):
    preds = model.predict(X_val)
    acc  = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec  = recall_score(y_val, preds)
    print(f"\n{name} Performance on Test Set:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(classification_report(y_val, preds, digits=3))

evaluate("K-Nearest Neighbors", knn_pipeline, X_test, y_test)
evaluate("Logistic Regression", logreg_pipeline, X_test, y_test)

# 9. Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# KNN grid
knn_params = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights":     ["uniform", "distance"]
}

grid_knn = GridSearchCV(
    knn_pipeline, knn_params,
    cv=5, scoring="f1", n_jobs=-1, verbose=1
)
print("\nTuning KNN...")
grid_knn.fit(X_train, y_train)
print("Best KNN params:", grid_knn.best_params_)

# LR grid
lr_params = {
    "lr__C":       [0.01, 0.1, 1.0, 10.0],
    "lr__penalty": ["l1", "l2"]
}

grid_lr = GridSearchCV(
    logreg_pipeline, lr_params,
    cv=5, scoring="f1", n_jobs=-1, verbose=1
)
print("\nTuning Logistic Regression...")
grid_lr.fit(X_train, y_train)
print("Best LR params:", grid_lr.best_params_)

# Evaluate tuned models
evaluate("Tuned KNN", grid_knn.best_estimator_, X_test, y_test)
evaluate("Tuned Logistic Regression", grid_lr.best_estimator_, X_test, y_test)

# 10. Deployment: Serialize Best Model
import joblib
from sklearn.metrics import f1_score

knn_f1 = f1_score(y_test, grid_knn.best_estimator_.predict(X_test))
lr_f1  = f1_score(y_test, grid_lr.best_estimator_.predict(X_test))

if knn_f1 >= lr_f1:
    best_model = grid_knn.best_estimator_
    model_name = "knn"
else:
    best_model = grid_lr.best_estimator_
    model_name = "logreg"

OUT_PATH = Path(__file__).parent / f"diabetes_{model_name}_pipeline.joblib"
joblib.dump(best_model, OUT_PATH)
print(f"\nSaved best pipeline ({model_name}) to: {OUT_PATH}")