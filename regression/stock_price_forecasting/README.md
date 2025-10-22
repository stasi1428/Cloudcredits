\# 📈 Stock Price Forecasting with LSTM



\## 📖 Overview

This project forecasts stock prices using a \*\*Long Short-Term Memory (LSTM)\*\* neural network.  

The goal is to capture temporal dependencies in financial time series data and evaluate the model’s predictive performance on unseen stock price sequences.



---



\## 📂 Project Structure

stock\_price\_forecasting/  

├── data/        # YahooFinance\_Industry.csv (raw dataset)  

├── src/         # pipeline.py (10-step ML workflow)  

├── results/     # metrics.json, \*\_price\_trend.png  

├── models/      # lstm\_stock\_\*.h5 (trained model), scaler\_stock\_\*.pkl  

├── api/         # FastAPI app for deployment  

└── README.md    # this file  



---



\## 🛠️ Workflow

This project follows the \*\*10-step ML workflow\*\*:



1\. Define Problem  

2\. Load \& Clean Data (parse dates, clean price column)  

3\. Exploratory Data Analysis (EDA) → price trend plots  

4\. Feature Engineering → sequence generation (60-day lookback)  

5\. Train/Test Split (80/20)  

6\. Model Selection → LSTM architecture with dropout + dense layers  

7\. Training (20 epochs, batch size 32)  

8\. Evaluation (MAE, RMSE)  

9\. Improvement (hyperparameter tuning, deeper LSTM layers)  

10\. Deployment (FastAPI-ready model)



---



\## 📊 Dataset

\- \*\*Source\*\*: Yahoo Finance industry dataset (`YahooFinance\_Industry.csv`)  

\- \*\*Features\*\*: Date, Symbol, Close/Price  

\- \*\*Target Variable\*\*: Stock closing price  

\- \*\*Preprocessing\*\*:  

&nbsp; - Removed formatting artifacts (commas, symbols)  

&nbsp; - Normalized prices with MinMaxScaler  

&nbsp; - Created 60-day rolling sequences for supervised learning  



---



\## 🤖 Model Architecture

\- \*\*LSTM Layer\*\* (50 units, return sequences)  

\- \*\*Dropout\*\* (0.2)  

\- \*\*LSTM Layer\*\* (50 units, return final sequence)  

\- \*\*Dropout\*\* (0.2)  

\- \*\*Dense Layer\*\* (25 units, ReLU activation)  

\- \*\*Dense Output Layer\*\* (1 unit)  

\- \*\*Optimizer\*\*: Adam  

\- \*\*Loss Function\*\*: Mean Squared Error  



---



\## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:



{

&nbsp;   "mae": ...,

&nbsp;   "rmse": ...

}



Generated plots:

\- `results/AAPL\_price\_trend.png` → stock price trend over time



---



\## 🚀 Deployment

The trained model and scaler are saved for deployment:



\- Model: `models/lstm\_stock\_AAPL.h5`  

\- Scaler: `models/scaler\_stock\_AAPL.pkl`  



Run locally:

cd api  

uvicorn main:app --reload  



Endpoint:

\- `POST /predict` → returns forecasted stock prices



---



\## 📜 License

This project is licensed under the \[MIT License](../../LICENSE).

