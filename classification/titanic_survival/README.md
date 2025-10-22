# 🚢 Titanic Survival Prediction



## 📖 Overview

This project predicts passenger survival on the Titanic using machine learning models.  

The dataset includes demographic and ticket information, and the goal is to benchmark **Logistic Regression** and **Random Forest** classifiers, with hyperparameter tuning to optimize performance.



---



## 📂 Project Structure

titanic\_survival/  

├── data/        # titanic.csv (raw dataset)  

├── src/         # pipeline.py (10-step ML workflow)  

├── results/     # metrics.json, survival\_counts.png, age\_distribution.png, correlation\_heatmap.png  

├── models/      # titanic\_model.joblib (best model)  

├── api/         # FastAPI app for deployment  

└── README.md    # this file  



---



## 🛠️ Workflow

This project follows the **10-step ML workflow**:



1. Define Problem  

2. Load \& Clean Data (feature engineering: family size, imputation for age/embarked, drop irrelevant columns)  

3. Exploratory Data Analysis (EDA) → survival counts, age distribution, correlation heatmap  

4. Feature Engineering → scaling numeric features, one-hot encoding categorical features  

5. Train/Test Split (80/20, stratified)  

6. Model Selection → Logistic Regression, Random Forest  

7. Training  

8. Evaluation (accuracy, precision, recall, classification report)  

9. Improvement → hyperparameter tuning (Random Forest depth, estimators, min\_samples\_split)  

10. Deployment (FastAPI-ready model)



---



## 📊 Dataset

- **Source**: Titanic dataset (`titanic.csv`)  

- **Size**: ~891 rows × 12+ features  

- **Features**: pclass, sex, age, sibsp, parch, fare, embarked, family\_size  

- **Target Variable**: `survived` (0 = No, 1 = Yes)



---



## 🤖 Models Used

- **Logistic Regression** (baseline)  

- **Random Forest Classifier** (baseline + tuned with GridSearchCV)  

- **Final Selected Model**: Tuned Random Forest (saved as `titanic\_model.joblib`)



---



## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "Logistic Regression": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "Random Forest": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "Tuned Random Forest": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "best\_params": {...}

&nbsp;   }

}



Generated plots:

- `results/survival\_counts.png` → survival distribution  

- `results/age\_distribution.png` → age distribution by survival  

- `results/correlation\_heatmap.png` → numeric feature correlations  



---



## 🚀 Deployment

The trained model is deployed via **FastAPI**.



Run locally:

cd api  

uvicorn main:app --reload  



Endpoint:

- `POST /predict` → returns survival prediction



---



## 📜 License

This project is licensed under the [MIT License](../../LICENSE).

