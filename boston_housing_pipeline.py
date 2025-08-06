#!/usr/bin/env python3
"""
boston_housing_pipeline.py

A full ML workflow to predict Boston house prices:
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
1. crim: Per capita crime rate by town.
2. zn: Proportion of large residential lots (over 25,000 sq. ft.).
3. indus: Proportion of non-retail business acres per town.
4. Chas: Binary variable indicating if the property is near Charles River (1 for yes, 0 for no).
5. nox: Concentration of nitrogen oxides in the air.
6. rm: Average number of rooms per dwelling.
7. age: Proportion of old owner-occupied units built before 1940.
8. dis: Weighted distances to Boston employment centers.
9. rad: Index of accessibility to radial highways.
10. tax: Property tax rate per $10,000.
"""

# 1. Define the Problem
#    Predict median house value (medv) using features from the Boston Housing dataset.

# 2. Collect and Prepare Data
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts"
    "/Cloudcredits Datasets/BostonHousing.csv"
)

# Load
df = pd.read_csv(DATA_PATH)

# Quick shape & null‐count check
print("Original shape:", df.shape)
print("Nulls per column:\n", df.isnull().sum(), "\n")

# Normalize column names to lowercase, stripped of whitespace
df.columns = [col.strip().lower() for col in df.columns]
print("Normalized columns:", df.columns.tolist(), "\n")

# Handle missing values by dropping any row with nulls
df = df.dropna()
print("Shape after dropping missing:", df.shape, "\n")

# 3. Exploratory Data Analysis (EDA)
import seaborn as sns
import matplotlib.pyplot as plt

# Compute & plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# 4. Feature Engineering
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer

# Split out features X and target y
X = df.drop("medv", axis=1)
y = df["medv"]

numeric_features = X.columns.tolist()
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
    ("poly", poly_transformer, numeric_features),
    ("scale", scaler,   numeric_features)
], remainder="drop")

# 5. Split the Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Choose a Model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

base_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor",  LinearRegression())
])

# 7. Train the Model
base_pipeline.fit(X_train, y_train)

# 8. Evaluate the Model
from sklearn.metrics import mean_squared_error, r2_score

y_pred = base_pipeline.predict(X_test)
mse    = mean_squared_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\nBase Linear Regression Performance:")
print(f"  MSE: {mse:.3f}")
print(f"  R² : {r2:.3f}")

# 9. Improve the Model
from sklearn.linear_model import RidgeCV, LassoCV

# Ridge with cross-validation
ridge_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("ridge",      RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))
])
ridge_pipeline.fit(X_train, y_train)

ridge_alpha = ridge_pipeline.named_steps["ridge"].alpha_
ridge_pred  = ridge_pipeline.predict(X_test)
ridge_mse   = mean_squared_error(y_test, ridge_pred)
ridge_r2    = r2_score(y_test, ridge_pred)

# Lasso with cross-validation
lasso_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("lasso",      LassoCV(alphas=np.logspace(-3, 1, 30), cv=5, max_iter=5000))
])
lasso_pipeline.fit(X_train, y_train)

lasso_alpha = lasso_pipeline.named_steps["lasso"].alpha_
lasso_pred  = lasso_pipeline.predict(X_test)
lasso_mse   = mean_squared_error(y_test, lasso_pred)
lasso_r2    = r2_score(y_test, lasso_pred)

print(f"\nRidge Regression (alpha={ridge_alpha:.4f}):")
print(f"  MSE: {ridge_mse:.3f} | R²: {ridge_r2:.3f}")
print(f"\nLasso Regression (alpha={lasso_alpha:.4f}):")
print(f"  MSE: {lasso_mse:.3f} | R²: {lasso_r2:.3f}")

# 10. Deploy the Model (serialize pipeline)
import joblib

# Select the best‐performing model for deployment (e.g., ridge_pipeline)
DEPLOY_PATH = Path(__file__).parent / "house_price_model.joblib"
joblib.dump(ridge_pipeline, DEPLOY_PATH)
print(f"\nSerialized pipeline saved to: {DEPLOY_PATH}")