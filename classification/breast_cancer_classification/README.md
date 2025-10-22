# 🎗️ Breast Cancer Classification (Wisconsin Dataset)



## 📖 Overview

This project classifies breast cancer tumors as **benign (0)** or **malignant (1)** using the **Breast Cancer Wisconsin dataset**.  

The goal is to benchmark **Support Vector Machines (SVM)** and **Random Forest** classifiers, evaluate their performance, and select the best model through hyperparameter tuning.



---



## 📂 Project Structure

breast\_cancer\_classification/  

├── data/        # breast\_cancer\_wisconsin.csv (raw dataset)  

├── src/         # pipeline.py (10-step ML workflow)  

├── results/     # metrics.json, class\_balance.png, correlation\_heatmap.png  

├── models/      # breast\_scaler.pkl, breast\_svm.pkl, breast\_rf.pkl  

├── api/         # FastAPI app for deployment  

└── README.md    # this file  



---



## 🛠️ Workflow

This project follows the **10-step ML workflow**:



1. Define Problem  

2. Load \& Clean Data (map labels: B → 0, M → 1; drop empty columns)  

3. Exploratory Data Analysis (EDA) → class balance, correlation heatmap  

4. Feature Engineering → standard scaling of numeric features  

5. Train/Test Split (80/20, stratified)  

6. Model Selection → SVM, Random Forest  

7. Training  

8. Evaluation (accuracy, precision, recall, F1-score, classification report)  

9. Improvement → hyperparameter tuning (Random Forest depth, estimators)  

10. Deployment (FastAPI-ready model)



---



## 📊 Dataset

- **Source**: Breast Cancer Wisconsin dataset (`breast\_cancer\_wisconsin.csv`)  

- **Size**: ~569 rows × 30 features  

- **Features**: cell nucleus characteristics (radius, texture, smoothness, etc.)  

- **Target Variable**: `y` (0 = Benign, 1 = Malignant)



---



## 🤖 Models Used

- **Support Vector Machine (SVM)** with RBF kernel  

- **Random Forest Classifier** (baseline + tuned with GridSearchCV)  

- **Final Selected Model**: Tuned Random Forest (saved as `breast\_rf.pkl`)  

- Additional artifacts: `breast\_scaler.pkl` (scaler), `breast\_svm.pkl` (SVM model)



---



## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "SVM": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "f1": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "RandomForest": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "f1": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "Tuned RF": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "f1": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   }

}



Generated plots:

- `results/class\_balance.png` → benign vs malignant distribution  

- `results/correlation\_heatmap.png` → feature correlations  



---



## 🚀 Deployment

The trained model is deployed via **FastAPI**.



Run locally:

cd api  

uvicorn main:app --reload  



Endpoint:

- `POST /predict` → returns tumor classification (benign or malignant)



---



## 📜 License

This project is licensed under the [MIT License](../../LICENSE).

