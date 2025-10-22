# 🎥 Movie Recommendation System (MovieLens)



## 📖 Overview

This project builds a **movie recommendation system** using the **MovieLens dataset**.  

The goal is to predict user ratings for unseen movies and generate personalized top‑N recommendations using collaborative filtering techniques.



---



## 📂 Project Structure

movie\_recommendation/  

├── data/        # MovieLens\_ratings.csv (raw dataset)  

├── src/         # pipeline.py (10-step ML workflow)  

├── results/     # metrics.json, rating\_distribution.png  

├── models/      # best\_svd\_model.pkl (trained model)  

├── api/         # FastAPI app for deployment  

└── README.md    # this file  



---



## 🛠️ Workflow

This project follows the **10-step ML workflow**:



1. Define Problem  

2. Load \& Clean Data (ratings dataset)  

3. Exploratory Data Analysis (EDA) → rating distribution histogram  

4. Feature Engineering → user–item matrix for collaborative filtering  

5. Train/Test Split (80/20)  

6. Model Selection → SVD (matrix factorization), KNN (item-based collaborative filtering)  

7. Training  

8. Evaluation (RMSE)  

9. Improvement → hyperparameter tuning (GridSearchCV for SVD)  

10. Deployment (FastAPI-ready recommender)



---



## 📊 Dataset

- **Source**: MovieLens ratings dataset (`MovieLens\_ratings.csv`)  

- **Size**: Varies (commonly 100k+ ratings)  

- **Features**: `userId`, `movieId`, `rating`  

- **Target Variable**: Predicted rating for unseen movies



---



## 🤖 Models Used

- **SVD (Singular Value Decomposition)** — matrix factorization  

- **KNNBasic** — item-based collaborative filtering with cosine similarity  

- **Final Selected Model**: Tuned SVD (saved as `best\_svd\_model.pkl`)



---



## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "SVD": {"rmse": ...},

&nbsp;   "KNN": {"rmse": ...},

&nbsp;   "Tuned SVD": {"rmse": ..., "best\_params": {...}}

}



Generated plots:

- `results/rating\_distribution.png` → histogram of rating frequencies



---



## 🎯 Recommendations

The system can generate top‑N recommendations for any user.  

Example (User 1):



Top 10 recommendations for user 1:

MovieID 318    — Predicted Rating: 5.00

MovieID 898    — Predicted Rating: 5.00

MovieID 904    — Predicted Rating: 5.00

MovieID 930    — Predicted Rating: 5.00

MovieID 1283   — Predicted Rating: 5.00

MovieID 750    — Predicted Rating: 5.00

MovieID 858    — Predicted Rating: 5.00

MovieID 1201   — Predicted Rating: 5.00

MovieID 177593 — Predicted Rating: 5.00  

...



---



## 🚀 Deployment

The trained model is deployed via **FastAPI**.



Run locally:  

cd api  

uvicorn main:app --reload  



Endpoint:  

- `POST /recommend` → returns top‑N movie recommendations for a given user



---



## 📜 License

This project is licensed under the [MIT License](LICENSE).

