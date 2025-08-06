#!/usr/bin/env python3
"""
breast_cancer_pipeline.py

A full ML workflow to classify tumors as benign or malignant:
1. Define the Problem
2. Collect & Prepare Data
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Split the Data
6. Model Selection
7. Training
8. Evaluation
9. Improvement (Hyperparameter Tuning)
10. Deployment (serialize pipelines & models)

About Dataset
The Breast Cancer Wisconsin dataset contains 569 samples:
- 30 numerical features per sample (mean, SE, worst of 10 nucleus measurements)
- y: “B” for benign or “M” for malignant
"""

# 1. Define the Problem
#    Binary classification: predict tumor class (benign=0, malignant=1).

# 2. Collect & Prepare Data
import pandas as pd
from pathlib import Path
import numpy as np

DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts/"
    "Cloudcredits Datasets/breast_cancer_wisconsin.csv"
)
df = pd.read_csv(DATA_PATH)

# Drop unnamed index column if present
if "" in df.columns:
    df = df.drop(columns=[""])

# Encode target: B→0, M→1
df["y"] = df["y"].map({"B": 0, "M": 1})

# Check missing values
print("Missing values per column:\n", df.isna().sum())

# 3. Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Class balance
plt.figure(figsize=(6,4))
sns.countplot(x="y", data=df)
plt.title("Benign (0) vs Malignant (1)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.drop(columns=["y"]).corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation")
plt.show()

# 4. Feature Engineering
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=["y"])
y = df["y"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split the Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Model Selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

svm_clf = SVC(kernel="rbf", probability=True, random_state=42)
rf_clf  = RandomForestClassifier(random_state=42)

# 7. Training
print("Training SVM...")
svm_clf.fit(X_train, y_train)

print("Training Random Forest...")
rf_clf.fit(X_train, y_train)

# 8. Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

for name, model in [("SVM", svm_clf), ("RandomForest", rf_clf)]:
    preds = model.predict(X_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, preds, target_names=["Benign","Malignant"]))

# 9. Improvement (Hyperparameter Tuning for Random Forest)
from sklearn.model_selection import GridSearchCV

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10]
}

grid_rf = GridSearchCV(
    rf_clf, rf_params, cv=5, scoring="recall", n_jobs=-1, verbose=1
)
print("Tuning Random Forest for recall...")
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)

# Re-evaluate tuned RF
pred_rf = best_rf.predict(X_test)
print("\nTuned RF Classification Report:")
print(classification_report(y_test, pred_rf, target_names=["Benign","Malignant"]))

# 10. Deployment (serialize pipelines & models)
import joblib

OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)

# Save scaler
SCALER_OUT = OUT_DIR / "breast_scaler.pkl"
joblib.dump(scaler, SCALER_OUT)

# Save SVM and RF models
SVM_OUT = OUT_DIR / "breast_svm.pkl"
RF_OUT  = OUT_DIR / "breast_rf.pkl"
joblib.dump(svm_clf, SVM_OUT)
joblib.dump(best_rf, RF_OUT)

print(f"Saved scaler to: {SCALER_OUT}")
print(f"Saved SVM model to: {SVM_OUT}")
print(f"Saved RF model to: {RF_OUT}")