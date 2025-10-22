# 🌸 Iris Flower Classification



## 📖 Overview

This project classifies iris flowers into three species (*Setosa, Versicolor, Virginica*) using classical machine learning models.  

The goal is to benchmark **Decision Tree** and **Logistic Regression** classifiers on the well-known Iris dataset, and to demonstrate model selection, evaluation, and hyperparameter tuning.



---



## 📂 Project Structure

iris\_classification/  

├── data/        # iris.csv (raw dataset)  

├── src/         # pipeline.py (10-step ML workflow)  

├── results/     # metrics.json, pairplot.png  

├── models/      # iris\_classifier.joblib (best model)  

├── api/         # FastAPI app for deployment  

└── README.md    # this file  



---



## 🛠️ Workflow

This project follows the **10-step ML workflow**:



1. Define Problem  

2. Load \& Clean Data (drop missing values, normalize column names)  

3. Exploratory Data Analysis (EDA) → pairplot of features by species  

4. Feature Engineering → scaling numeric features  

5. Train/Test Split (80/20, stratified)  

6. Model Selection → Decision Tree, Logistic Regression  

7. Training  

8. Evaluation (accuracy, confusion matrix)  

9. Improvement → hyperparameter tuning (Decision Tree depth)  

10. Deployment (FastAPI-ready model)



---



## 📊 Dataset

- **Source**: Iris dataset (`iris.csv`)  

- **Size**: 150 rows × 5 columns  

- **Features**: sepal length, sepal width, petal length, petal width  

- **Target Variable**: `species` (Setosa, Versicolor, Virginica)



---



## 🤖 Models Used

- **Decision Tree Classifier** (baseline, tuned with GridSearchCV)  

- **Logistic Regression** (max\_iter=200)  

- **Final Selected Model**: Tuned Decision Tree (saved as `iris\_classifier.joblib`)



---



## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "Decision Tree": {"accuracy": ..., "confusion\_matrix": \[...]},

&nbsp;   "Logistic Regression": {"accuracy": ..., "confusion\_matrix": \[...]},

&nbsp;   "Tuned Decision Tree": {"accuracy": ..., "best\_params": {"classifier\_\_max\_depth": ...}}

}



Generated plots:

- `results/pairplot.png` → pairwise feature visualization by species



---



## 🚀 Deployment

The trained model is deployed via **FastAPI**.



Run locally:

cd api  

uvicorn main:app --reload  



Endpoint:

- `POST /predict` → returns predicted species



---



## 📜 License

This project is licensed under the [MIT License](LICENSE).

