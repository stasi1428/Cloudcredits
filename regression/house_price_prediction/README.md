\# 🏠 House Price Prediction (Boston Housing)



\## 📖 Overview

This project predicts median house prices in Boston using multiple regression techniques.  

The objective is to understand how socio‑economic and structural features influence housing values, and to benchmark different regression models for accuracy and interpretability.



---



\## 📂 Project Structure

house\_price\_prediction/

├── data/        # BostonHousing.csv (raw dataset)

├── src/         # pipeline.py (10-step ML workflow)

├── results/     # metrics.json, correlation\_heatmap.png

├── models/      # house\_price\_model.joblib (best model)

├── api/         # FastAPI app for deployment

└── README.md    # this file



---



\## 🛠️ Workflow

This project follows the \*\*10-step ML workflow\*\*:



1\. Define Problem  

2\. Load \& Clean Data  

3\. Exploratory Data Analysis (EDA)  

4\. Feature Engineering (polynomial features, scaling)  

5\. Train/Test Split  

6\. Model Selection (Linear, Ridge, Lasso)  

7\. Training  

8\. Evaluation (MSE, R²)  

9\. Improvement (regularization, hyperparameter tuning)  

10\. Deployment (FastAPI-ready model)



---



\## 📊 Dataset

\- \*\*Source\*\*: Boston Housing dataset (UCI / sklearn variant)  

\- \*\*Size\*\*: 506 rows × 14 features  

\- \*\*Key Features\*\*: crime rate, average number of rooms, property tax rate, pupil‑teacher ratio, etc.  

\- \*\*Target Variable\*\*: `medv` (median value of owner‑occupied homes in $1000s)



---



\## 🤖 Models Used

\- \*\*Linear Regression\*\* (baseline)  

\- \*\*Ridge Regression\*\* (cross‑validated alpha)  

\- \*\*Lasso Regression\*\* (cross‑validated alpha)  

\- \*\*Final Selected Model\*\*: Ridge Regression (saved as `house\_price\_model.joblib`)



---



\## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "linear\_regression": {"mse": ..., "r2": ...},

&nbsp;   "ridge": {"alpha": ..., "mse": ..., "r2": ...},

&nbsp;   "lasso": {"alpha": ..., "mse": ..., "r2": ...}

}



Generated plots:

\- `results/correlation\_heatmap.png` → feature correlation matrix



---



\## 🚀 Deployment

The trained model is deployed via \*\*FastAPI\*\*.



Run locally:

cd api

uvicorn main:app --reload



Endpoint:

\- `POST /predict` → returns model predictions



---



## 📜 License
This project is licensed under the [MIT License](../../LICENSE).

