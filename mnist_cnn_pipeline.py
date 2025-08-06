#!/usr/bin/env python3
"""
mnist_cnn_pipeline.py

A full ML workflow to recognize handwritten digits using CNN:
1. Define Problem
2. Load & Clean Data
3. Exploratory Data Analysis
4. Feature Engineering
5. Train/Test Split
6. Model Selection
7. Training
8. Evaluation
9. Improvement
10. Deployment (serialize model)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils      import to_categorical
from tensorflow.keras.models     import Sequential, save_model
from tensorflow.keras.layers     import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks  import EarlyStopping

# 1–2. Load & Inspect
DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts"
    "/Cloudcredits Datasets/mnist_dataset.csv"
)
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)
print("Columns sample:", df.columns.tolist()[:5], "…", df.columns.tolist()[-5:])

# Detect whether we got labels in the CSV
n_rows, n_cols = df.shape

if "label" in df.columns:
    # Standard Kaggle “train.csv”: 785 columns (label + 784 pixels)
    print("Found 'label' column.")
    y = df["label"].values
    X = df.drop("label", axis=1).values

elif n_cols == 784:
    # Pixel-only CSV — we can't train without labels, so load Keras MNIST
    print("No 'label' column and 784 cols → falling back to built-in MNIST loader.")
    from tensorflow.keras.datasets import mnist

    # (60k train + 10k test)
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
    X = np.concatenate([X_train_full, X_test_full], axis=0)
    y = np.concatenate([y_train_full, y_test_full], axis=0)
    print(f"Using built-in MNIST: {X.shape[0]} samples with labels")

else:
    raise ValueError(
        "Input CSV must either include a 'label' column (785 cols) "
        "or be exactly 784 pixel columns (no labels)."
    )

# 3. Quick EDA: plot first ten samples
# If X came from df.values, reshape accordingly; if it came from loader, it's already 28×28
if X.ndim == 2:
    imgs = X.reshape(-1, 28, 28)
else:
    imgs = X

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(imgs[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.suptitle("Sample Digits")
plt.show()

# 4. Preprocessing
if X.ndim == 2:
    X = X.reshape(-1, 28, 28, 1)
else:
    X = X[..., np.newaxis]

X = X.astype("float32") / 255.0
y_cat = to_categorical(y, num_classes=10)

# 5. Train/Test Split
#   We stratify on y (not y_cat) to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)
print("Train/test shapes:",
      X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 6. Build CNN
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
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 7. Train
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20, batch_size=128,
    callbacks=[early_stop], verbose=2
)

# 8. Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)

# 9. Improve
# (e.g. data augmentation, learning-rate scheduling, ensembling…)

# 10. Deploy
DEPLOY_DIR = Path(__file__).parent / "deployment"
DEPLOY_DIR.mkdir(exist_ok=True)
save_model(model, DEPLOY_DIR / "mnist_cnn.h5")
print("Model saved to", DEPLOY_DIR)