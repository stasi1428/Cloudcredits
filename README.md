# Cloudcredits: Machine Learning Portfolio

This repository contains **10 end-to-end ML projects** spanning regression, classification, NLP, computer vision, and recommendation systems.  

Each project is fully reproducible, with pipelines, results, and deployable APIs.

---

## 📂 Repository Structure

Each project follows the same structure:

project_name/  
├── data/        # raw dataset  
├── src/         # pipeline.py (10-step ML workflow)  
├── results/     # metrics.json, plots, reports  
├── models/      # serialized trained models  
├── api/         # FastAPI app for deployment  
└── README.md    # project-specific documentation  

---

## 📊 Projects

| Domain              | Project                          | Folder |
|---------------------|----------------------------------|--------|
| Regression          | House Price Prediction           | [regression/house_price_prediction](regression/house_price_prediction) |
| Regression          | Stock Price Forecasting (LSTM)   | [regression/stock_price_forecasting](regression/stock_price_forecasting) |
| Classification      | Iris Flower Classification       | [classification/iris_classification](classification/iris_classification) |
| Classification      | Titanic Survival Prediction      | [classification/titanic_survival](classification/titanic_survival) |
| Classification      | Diabetes Prediction              | [classification/diabetes_prediction](classification/diabetes_prediction) |
| Classification      | Breast Cancer Classification     | [classification/breast_cancer_classification](classification/breast_cancer_classification) |
| NLP                 | Sentiment Analysis (IMDb)        | [natural_language_processing/sentiment_analysis_imdb](natural_language_processing/sentiment_analysis_imdb) |
| NLP                 | Spam Detection (Enron)           | [natural_language_processing/spam_detection](natural_language_processing/spam_detection) |
| Computer Vision     | Digit Recognition (MNIST)        | [computer_vision/digit_recognition_mnist](computer_vision/digit_recognition_mnist) |
| Recommendation Sys. | Movie Recommendation (MovieLens) | [recommendation_systems/movie_recommendation](recommendation_systems/movie_recommendation) |

---

## ⚙️ Workflow

All pipelines implement a **10-step ML workflow**:

1. Define Problem  
2. Load & Clean Data  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Train/Test Split  
6. Model Selection  
7. Training  
8. Evaluation  
9. Improvement (tuning, alt models)  
10. Deployment (API-ready model)

---

## 🚀 Deployment

Each project includes an `api/` folder with a **FastAPI app** exposing a `/predict` endpoint.  

To run an API locally:

cd project_folder/api  
uvicorn main:app --reload  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).