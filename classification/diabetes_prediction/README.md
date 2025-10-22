# 🩺 Diabetes Prediction (Pima Indians Dataset)



## 📖 Overview

This project predicts the likelihood of diabetes in patients using the **Pima Indians Diabetes dataset**.  

The goal is to benchmark **K-Nearest Neighbors (KNN)** and **Logistic Regression** classifiers, evaluate their performance, and select the best model through hyperparameter tuning.



---



## 📂 Project Structure

diabetes\_prediction/  

├── data/        # pima\_indian\_diabetes.csv (raw dataset)  

├── src/         # pipeline.py (10-step ML workflow)  

├── results/     # metrics.json, outcome\_distribution.png, feature distributions, boxplots, correlation\_heatmap.png  

├── models/      # diabetes\_knn\_pipeline.joblib or diabetes\_logreg\_pipeline.joblib (best model)  

├── api/         # FastAPI app for deployment  

└── README.md    # this file  



---



## 🛠️ Workflow

This project follows the **10-step ML workflow**:



1. Define Problem  

2. Load \& Clean Data (replace invalid zeros with NaN, impute with median)  

3. Exploratory Data Analysis (EDA) → outcome distribution, feature histograms, boxplots, correlation heatmap  

4. Feature Engineering → scaling numeric features  

5. Train/Test Split (80/20, stratified)  

6. Model Selection → KNN, Logistic Regression  

7. Training  

8. Evaluation (accuracy, precision, recall, F1-score, classification report)  

9. Improvement → hyperparameter tuning (GridSearchCV for KNN and Logistic Regression)  

10. Deployment (FastAPI-ready model)



---



## 📊 Dataset

- **Source**: Pima Indians Diabetes dataset (`pima\_indian\_diabetes.csv`)  

- **Size**: 768 rows × 9 columns  

- **Features**: Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Pregnancies  

- **Target Variable**: `Outcome` (0 = No Diabetes, 1 = Diabetes)  

- **Preprocessing**: Replaced invalid zero values with median imputation



---



## 🤖 Models Used

- **K-Nearest Neighbors (KNN)**  

- **Logistic Regression** (liblinear solver)  

- **Final Selected Model**: Best tuned model (KNN or Logistic Regression, depending on F1-score)  

&nbsp; - Saved as `diabetes\_knn\_pipeline.joblib` or `diabetes\_logreg\_pipeline.joblib`



---



## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "KNN": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "f1": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "Logistic Regression": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "f1": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "Tuned Best": {

&nbsp;       "model": "knn" or "logreg",

&nbsp;       "f1": ...

&nbsp;   }

}



Generated plots:

- `results/outcome\_distribution.png` → class distribution  

- `results/\*\_distribution.png` → feature histograms  

- `results/\*\_boxplot.png` → feature vs outcome boxplots  

- `results/correlation\_heatmap.png` → feature correlations  



---



## 🚀 Deployment

The trained model is deployed via **FastAPI**.



Run locally:

cd api  

uvicorn main:app --reload  



Endpoint:

- `POST /predict` → returns diabetes prediction



---



## 📜 License

This project is licensed under the [MIT License]((https://github.com/stasi1428/Cloudcredits/blob/main/LICENSE)).

