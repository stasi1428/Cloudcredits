\# 📧 Spam Detection (Enron Dataset)



\## 📖 Overview

This project classifies emails as \*\*ham (legitimate)\*\* or \*\*spam\*\* using the \*\*Enron email dataset\*\*.  

The goal is to benchmark \*\*Naive Bayes\*\* and \*\*Linear SVM\*\* classifiers, evaluate their performance, and select the best model through hyperparameter tuning.



---



\## 📂 Project Structure

spam\_detection/  

├── data/        # enron\_email\_data.csv (raw dataset)  

├── src/         # pipeline.py (10-step ML workflow)  

├── results/     # metrics.json, category\_counts.png, message\_length\_distribution.png  

├── models/      # spam\_nb\_pipeline.joblib or spam\_svm\_pipeline.joblib (best model)  

├── api/         # FastAPI app for deployment  

└── README.md    # this file  



---



\## 🛠️ Workflow

This project follows the \*\*10-step ML workflow\*\*:



1\. Define Problem  

2\. Load \& Clean Data (remove HTML tags, lowercase, strip punctuation, normalize whitespace)  

3\. Exploratory Data Analysis (EDA) → category counts, message length distribution  

4\. Feature Engineering → TF-IDF vectorization (unigrams + bigrams)  

5\. Train/Test Split (80/20, stratified)  

6\. Model Selection → Naive Bayes, Linear SVM  

7\. Training  

8\. Evaluation (accuracy, precision, recall, F1-score, classification report)  

9\. Improvement → hyperparameter tuning (GridSearchCV for both models)  

10\. Deployment (FastAPI-ready model)



---



\## 📊 Dataset

\- \*\*Source\*\*: Enron email dataset (`enron\_email\_data.csv`)  

\- \*\*Size\*\*: Thousands of emails (ham + spam)  

\- \*\*Features\*\*: `message` (email text)  

\- \*\*Target Variable\*\*: `category` (ham = 0, spam = 1)  

\- \*\*Preprocessing\*\*: HTML tag removal, lowercasing, punctuation removal, whitespace normalization



---



\## 🤖 Models Used

\- \*\*Naive Bayes (MultinomialNB)\*\* with TF-IDF features  

\- \*\*Linear SVM (LinearSVC)\*\* with TF-IDF features  

\- \*\*Final Selected Model\*\*: Best tuned model (Naive Bayes or SVM, depending on F1-score)  

&nbsp; - Saved as `spam\_nb\_pipeline.joblib` or `spam\_svm\_pipeline.joblib`



---



\## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "Naive Bayes": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "f1": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "Linear SVM": {

&nbsp;       "accuracy": ...,

&nbsp;       "precision": ...,

&nbsp;       "recall": ...,

&nbsp;       "f1": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "Tuned Naive Bayes": {...},

&nbsp;   "Tuned Linear SVM": {...}

}



Generated plots:

\- `results/category\_counts.png` → ham vs spam distribution  

\- `results/message\_length\_distribution.png` → message length by category  



---



\## 🚀 Deployment

The trained model is deployed via \*\*FastAPI\*\*.



Run locally:

cd api  

uvicorn main:app --reload  



Endpoint:

\- `POST /predict` → returns spam/ham classification



---



\## 📜 License

This project is licensed under the \[MIT License](../../LICENSE).

