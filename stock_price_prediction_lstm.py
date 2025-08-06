#!/usr/bin/env python3
"""
stock_price_prediction_lstm.py

A full ML workflow to forecast future stock prices using LSTM:

1. Define the Problem
2. Collect & Prepare Data
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Split the Data
6. Choose a Model
7. Train the Model
8. Evaluate the Model
9. Improve the Model
10. Deploy the Model (optional)
"""

# 1. Define the Problem
#    We want to predict tomorrow’s closing price given the past 60 days of prices.
#    This is a univariate regression problem (time-series forecasting).

# 2. Collect & Prepare Data
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts/"
    "Cloudcredits Datasets/YahooFinanace_Industry.csv"
)
df = pd.read_csv(DATA_PATH)
print("Raw data shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2.1 Clean Price column: remove commas, strip non-numeric, cast to float
price_col = 'Close' if 'Close' in df.columns else 'Price'
if price_col not in df.columns:
    raise KeyError("CSV must contain a 'Price' or 'Close' column.")

df[price_col] = (
    df[price_col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d\.]", "", regex=True)
                .astype(float)
)

# 2.2 Detect or warn about missing date column
date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Sorted by date column: {date_col}")
else:
    print("Warning: no date column found; assuming rows are already chronological.")

# 2.3 List symbols & let you pick one
if 'Symbol' in df.columns:
    symbols = df['Symbol'].unique().tolist()
    print("Available symbols:", symbols)
    chosen_symbol = 'AAPL'  # change as needed, or set to None to skip
    if chosen_symbol in symbols:
        df = df[df['Symbol'] == chosen_symbol].copy()
        print(f"Filtered to symbol '{chosen_symbol}' → {len(df)} rows")
    else:
        print(f"Symbol '{chosen_symbol}' not in list; using full dataset")
        chosen_symbol = None
else:
    chosen_symbol = None
    print("No 'Symbol' column; using all rows")

# 3. Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(df[price_col].values, label=price_col)
plt.title(f"{price_col} Over Time ({chosen_symbol or 'ALL'})")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.show()

# 4. Feature Engineering
# Create sequences of length 60 to forecast the next value
from sklearn.preprocessing import MinMaxScaler

prices = df[[price_col]].values
if len(prices) < 61:
    raise ValueError("Not enough data to build sequences (need ≥61 rows).")

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(prices_scaled, SEQ_LEN)
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# 5. Split the Data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# 6. Choose a Model
# We’ll use a two-layer LSTM followed by Dense output
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 7. Train the Model
EPOCHS = 20
BATCH_SIZE = 32

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

# 8. Evaluate the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error

pred_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(pred_scaled)
actuals     = scaler.inverse_transform(y_test)

mae  = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
print(f"\nTest MAE:  {mae:.3f}")
print(f"Test RMSE: {rmse:.3f}")

# 9. Improve the Model
# Suggestions:
#  - Add EarlyStopping callback
#  - Experiment with more LSTM layers or units
#  - Tune SEQ_LEN, batch_size, or learning_rate
#  - Try different optimizers (Adam, RMSprop)

# 10. Deploy the Model (optional)
# Save model and scaler for later inference or API deployment
OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)

symbol_label = chosen_symbol or 'ALL'
model_path  = OUT_DIR / f"lstm_stock_{symbol_label}.h5"
scaler_path = OUT_DIR / f"scaler_stock_{symbol_label}.pkl"

model.save(model_path)
import joblib
joblib.dump(scaler, scaler_path)

print(f"\nSaved model    to: {model_path}")
print(f"Saved scaler   to: {scaler_path}")