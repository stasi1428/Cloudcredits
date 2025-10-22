\# 🎬 IMDb Sentiment Analysis



\## 📖 Overview

This project performs \*\*binary sentiment classification\*\* (positive vs negative) on the \*\*IMDb movie reviews dataset\*\*.  

Two approaches are benchmarked: a \*\*Naive Bayes classifier with TF-IDF features\*\* and a \*\*deep learning LSTM model\*\* for sequence-based sentiment analysis.



---



\## 📂 Project Structure

sentiment\_analysis\_imdb/  

├── data/        # IMDB Dataset.csv (raw dataset)  

├── src/         # pipeline.py (10-step ML workflow)  

├── results/     # metrics.json, review\_length\_distribution.png  

├── models/      # imdb\_nb\_pipeline.joblib, imdb\_tokenizer.joblib, imdb\_lstm\_model.h5  

├── api/         # FastAPI app for deployment  

└── README.md    # this file  



---



\## 🛠️ Workflow

This project follows the \*\*10-step ML workflow\*\*:



1\. Define Problem  

2\. Load \& Clean Data (remove HTML tags, non-alphabetic chars, lowercase text)  

3\. Exploratory Data Analysis (EDA) → review length distribution by sentiment  

4\. Feature Engineering → TF-IDF features for Naive Bayes, tokenization + padding for LSTM  

5\. Train/Test Split (80/20, stratified)  

6\. Model Selection → Naive Bayes, LSTM  

7\. Training (Naive Bayes with TF-IDF, LSTM with embeddings)  

8\. Evaluation (accuracy, F1-score, classification report)  

9\. Improvement → hyperparameter tuning, deeper LSTM layers  

10\. Deployment (FastAPI-ready models)



---



\## 📊 Dataset

\- \*\*Source\*\*: IMDb movie reviews dataset (`IMDB Dataset.csv`)  

\- \*\*Size\*\*: 50,000 reviews (25k positive, 25k negative)  

\- \*\*Features\*\*: `review` (text)  

\- \*\*Target Variable\*\*: `sentiment` (positive = 1, negative = 0)  

\- \*\*Preprocessing\*\*: HTML tag removal, punctuation removal, lowercasing, tokenization



---



\## 🤖 Models Used

\- \*\*Naive Bayes (MultinomialNB)\*\* with TF-IDF features  

\- \*\*LSTM (Long Short-Term Memory)\*\* with embedding + dropout layers  

\- \*\*Final Artifacts\*\*:  

&nbsp; - `imdb\_nb\_pipeline.joblib` (Naive Bayes pipeline)  

&nbsp; - `imdb\_tokenizer.joblib` (Tokenizer for LSTM)  

&nbsp; - `imdb\_lstm\_model.h5` (trained LSTM model)



---



\## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "Naive Bayes": {

&nbsp;       "accuracy": ...,

&nbsp;       "f1": ...,

&nbsp;       "classification\_report": {...}

&nbsp;   },

&nbsp;   "LSTM": {

&nbsp;       "accuracy": ...,

&nbsp;       "f1": ...

&nbsp;   }

}



Generated plots:

\- `results/review\_length\_distribution.png` → histogram of review lengths by sentiment



---



\## 🚀 Deployment

The trained models are deployed via \*\*FastAPI\*\*.



Run locally:

cd api  

uvicorn main:app --reload  



Endpoints:

\- `POST /predict\_nb` → returns sentiment prediction using Naive Bayes  

\- `POST /predict\_lstm` → returns sentiment prediction using LSTM  



---



\## 📜 License

This project is licensed under the \[MIT License](../../LICENSE).

