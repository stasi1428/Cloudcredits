#!/usr/bin/env python3
"""
iris_classification_pipeline.py

A full ML workflow to classify Iris species:
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
"""

# 1. Define the Problem
#    Predict iris species based on sepal and petal measurements.

# 2. Load & Clean Data
import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts"
    "/Cloudcredits Datasets/iris.csv"
)

df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)
print("Nulls per column:\n", df.isnull().sum(), "\n")

df.columns = [col.strip().lower() for col in df.columns]
print("Columns:", df.columns.tolist(), "\n")

# 3. Exploratory Data Analysis (optional plotting)
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue="species", corner=True)
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# 4. Feature Engineering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

X = df.drop("species", axis=1)
y = LabelEncoder().fit_transform(df["species"])

numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer([
    ("scale", StandardScaler(), numeric_features)
])

# 5. Split the Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Choose Models
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

dt_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

lr_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=200, random_state=42))
])

# 7. Train the Models
dt_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)

# 8. Evaluate the Models
from sklearn.metrics import accuracy_score, confusion_matrix

models = {
    "Decision Tree": dt_pipeline,
    "Logistic Regression": lr_pipeline
}

results = {}
for name, model in models.items():
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    cm    = confusion_matrix(y_test, preds)
    results[name] = (acc, cm)
    print(f"\n{name} Accuracy: {acc:.3f}")
    print(f"{name} Confusion Matrix:\n{cm}")

# 9. Improve the Model
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__max_depth": [2, 3, 4, 5, None]
}
grid_dt = GridSearchCV(dt_pipeline, param_grid, cv=5)
grid_dt.fit(X_train, y_train)

best_dt = grid_dt.best_estimator_
best_acc = accuracy_score(y_test, best_dt.predict(X_test))
print(f"\nTuned Decision Tree Accuracy: {best_acc:.3f} (depth={grid_dt.best_params_['classifier__max_depth']})")

# 10. Deploy the Best Model
import joblib
DEPLOY_PATH = Path(__file__).parent / "iris_classifier.joblib"
joblib.dump(best_dt, DEPLOY_PATH)
print(f"Serialized pipeline saved to: {DEPLOY_PATH}")