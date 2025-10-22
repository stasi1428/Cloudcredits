#!/usr/bin/env python3
"""
Stock Price Forecasting with LSTM
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, json, joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "YahooFinanace_Industry.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    price_col = 'Close' if 'Close' in df.columns else 'Price'
    df[price_col] = (df[price_col].astype(str)
                     .str.replace(",", "", regex=False)
                     .str.replace(r"[^\d\.]", "", regex=True)
                     .astype(float))
    date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df, price_col

def run_eda(df, price_col, symbol_label):
    plt.figure(figsize=(10,4))
    plt.plot(df[price_col].values, label=price_col)
    plt.title(f"{price_col} Over Time ({symbol_label})")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(RESULTS_DIR / f"{symbol_label}_price_trend.png")
    plt.close()

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def build_model(seq_len):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    df, price_col = load_and_clean()
    symbol_label = "AAPL" if "Symbol" in df.columns else "ALL"
    if "Symbol" in df.columns and symbol_label in df["Symbol"].unique():
        df = df[df["Symbol"] == symbol_label].copy()
    run_eda(df, price_col, symbol_label)

    prices = df[[price_col]].values
    scaler = MinMaxScaler(feature_range=(0,1))
    prices_scaled = scaler.fit_transform(prices)

    SEQ_LEN = 60
    X, y = create_sequences(prices_scaled, SEQ_LEN)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = build_model(SEQ_LEN)
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=20, batch_size=32, verbose=2)

    pred_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(pred_scaled)
    actuals     = scaler.inverse_transform(y_test)

    mae  = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    metrics = {"mae": float(mae), "rmse": float(rmse)}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    model.save(MODELS_DIR / f"lstm_stock_{symbol_label}.h5")
    joblib.dump(scaler, MODELS_DIR / f"scaler_stock_{symbol_label}.pkl")

    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()