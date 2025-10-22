#!/usr/bin/env python3
"""
MNIST Digit Recognition Pipeline (CNN)
"""

import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "mnist_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    n_rows, n_cols = df.shape
    if "label" in df.columns:
        y = df["label"].values
        X = df.drop("label", axis=1).values
    elif n_cols == 784:
        from tensorflow.keras.datasets import mnist
        (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
        X = np.concatenate([X_train_full, X_test_full], axis=0)
        y = np.concatenate([y_train_full, y_test_full], axis=0)
    else:
        raise ValueError("CSV must include 'label' column or 784 pixel columns.")
    return X, y

def run_eda(X, y):
    imgs = X.reshape(-1, 28, 28) if X.ndim == 2 else X
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(f"Label: {y[i]}")
        ax.axis("off")
    plt.suptitle("Sample Digits")
    plt.savefig(RESULTS_DIR / "sample_digits.png")
    plt.close()

def train_and_evaluate(X, y):
    if X.ndim == 2:
        X = X.reshape(-1, 28, 28, 1)
    else:
        X = X[..., np.newaxis]
    X = X.astype("float32") / 255.0
    y_cat = to_categorical(y, num_classes=10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1,
              epochs=20, batch_size=128, callbacks=[early_stop], verbose=2)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)

    metrics = {"accuracy": float(acc), "confusion_matrix": cm.tolist()}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    save_model(model, MODELS_DIR / "mnist_cnn.h5")
    return metrics

def main():
    X, y = load_and_clean()
    run_eda(X, y)
    results = train_and_evaluate(X, y)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()