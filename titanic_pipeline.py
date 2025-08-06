#!/usr/bin/env python3
"""
titanic_pipeline.py

A full ML workflow to predict Titanic survivors:
1. Define Problem
2. Load & Clean Data
3. Exploratory Data Analysis
4. Feature Engineering
5. Train/Test Split
6. Model Selection
7. Training
8. Evaluation
9. Improvement
10. Deployment (serialize pipeline)

About Dataset

The dataset typically includes the following columns:
1. PassengerId: A unique identifier for each passenger.
2. Survived: This column indicates whether a passenger survived (1) or did not survive (0).
3. Pclass (Ticket class): A proxy for socio-economic status, with 1 being the highest class and 3 the lowest.
4. Name: The name of the passenger.
5. Sex: The gender of the passenger.
6. Age: The age of the passenger. (Note: There might be missing values in this column.)
7. SibSp: The number of siblings or spouses the passenger had aboard the Titanic.
8. Parch: The number of parents or children the passenger had aboard the Titanic.
9. Ticket: The ticket number.
10. Fare: The amount of money the passenger paid for the ticket.
"""

# 1. Define the Problem
#    Predict whether a passenger survived (1) or not (0) based on demographics and ticket info.

# 2. Load & Clean Data
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts"
    "/Cloudcredits Datasets/titanic.csv"
)
df = pd.read_csv(DATA_PATH)

# normalize column names to lowercase for consistency
df.columns = df.columns.str.lower()

# Quick shape & null‐count check
print("Original shape:", df.shape)
print("Nulls per column:\n", df.isnull().sum(), "\n")

# 3. Exploratory Data Analysis (EDA)
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of survivors
plt.figure(figsize=(6,4))
sns.countplot(x="survived", data=df)
plt.title("Survival Counts (0 = No, 1 = Yes)")
plt.show()

# Age distribution by survival
plt.figure(figsize=(8,4))
sns.kdeplot(df.loc[df.survived == 0, "age"].dropna(), label="Not Survived", shade=True)
sns.kdeplot(df.loc[df.survived == 1, "age"].dropna(), label="Survived", shade=True)
plt.title("Age Distribution by Survival")
plt.legend()
plt.show()

# Correlation heatmap of numeric features
num_cols = ["pclass", "age", "sibsp", "parch", "fare", "survived"]
plt.figure(figsize=(6,5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numeric Feature Correlations")
plt.show()

# 4. Feature Engineering
# Create family_size feature
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Impute missing age with median
df["age"] = df["age"].fillna(df["age"].median())

# Impute missing embarked with mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Drop columns not useful for prediction
df = df.drop(["passengerid", "name", "ticket", "cabin"], axis=1)

# Encode target
y = df["survived"].values

# 5. Prepare Features and Split
# Separate features
X = df.drop("survived", axis=1)

from sklearn.model_selection import train_test_split

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Build Preprocessing + Model Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numeric and categorical feature lists
numeric_feats = ["pclass", "age", "sibsp", "parch", "fare", "family_size"]
categorical_feats = ["sex", "embarked"]

# Preprocessing transformer
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_feats),
    ("cat", OneHotEncoder(drop="first"), categorical_feats)
])

# 7. Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Pipelines
logreg_pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("clf",    LogisticRegression(max_iter=500, random_state=42))
])

rf_pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("clf",    RandomForestClassifier(n_estimators=100, random_state=42))
])

# 8. Training
logreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# 9. Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

models = {
    "Logistic Regression": logreg_pipeline,
    "Random Forest":      rf_pipeline
}

for name, pipe in models.items():
    preds = pipe.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    print(f"\n{name} Performance:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print("  Classification Report:")
    print(classification_report(y_test, preds, digits=3))

# 10. Improvement: Hyperparameter Tuning for Random Forest
from sklearn.model_selection import GridSearchCV

param_grid = {
    "clf__n_estimators":     [100, 200, 300],
    "clf__max_depth":        [None, 5, 10],
    "clf__min_samples_split": [2, 5]
}

grid_rf = GridSearchCV(
    rf_pipeline, param_grid,
    cv=5, scoring="accuracy", n_jobs=-1
)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
best_params = grid_rf.best_params_
print(f"\nTuned Random Forest best params: {best_params}")

# Re-evaluate tuned model
tuned_preds = best_rf.predict(X_test)
print("Tuned RF Accuracy :", accuracy_score(y_test, tuned_preds))
print("Tuned RF Precision:", precision_score(y_test, tuned_preds))
print("Tuned RF Recall   :", recall_score(y_test, tuned_preds))

# 11. Deployment: Serialize the best pipeline
import joblib

# Choose best performing: grid_rf or logreg
best_pipeline = best_rf
DEPLOY_PATH = Path(__file__).parent / "titanic_model.joblib"
joblib.dump(best_pipeline, DEPLOY_PATH)
print(f"\nSerialized pipeline saved to: {DEPLOY_PATH}")