#!/usr/bin/env python3
"""
imdb_sentiment_pipeline.py

A full ML workflow to classify IMDb movie reviews as positive or negative:
1. Define Problem
2. Load & Clean Data
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Train/Test Split
6. Model Selection
7. Training
8. Evaluation
9. Improvement (Hyperparameter Tuning)
10. Deployment (serialize pipelines & models)

About Dataset
The IMDB Dataset.csv contains 50,000 movie reviews:
- review: raw HTML-tagged text
- sentiment: “positive” or “negative”
"""

# 1. Define the Problem
#    Binary classification: predict whether a review is positive (1) or negative (0).

# 2. Load & Clean Data
import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts"
    "/Cloudcredits Datasets/IMDB Dataset.csv"
)

df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)
print(df['sentiment'].value_counts(), "\n")

# Drop any missing rows (if present)
df.dropna(subset=['review', 'sentiment'], inplace=True)

# Clean review text: strip HTML, non-letters, lowercase
def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)       # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # remove non-letters
    return re.sub(r'\s+', ' ', text).strip().lower()

df['clean_review'] = df['review'].apply(clean_text)
df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# 3. Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Review length distribution by sentiment
df['length'] = df['clean_review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,4))
sns.histplot(data=df, x='length', hue='sentiment', bins=50, alpha=0.6)
plt.title("Review Length Distribution by Sentiment")
plt.xlabel("Word Count")
plt.show()

# 4. Feature Engineering
#   - Naive Bayes: TF-IDF on unigrams + bigrams
#   - LSTM: Tokenize + pad sequences

# 5. Train/Test Split
from sklearn.model_selection import train_test_split

X = df['clean_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Model Selection
# 6.1. Naive Bayes pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
    ('clf',   MultinomialNB(alpha=0.1))
])

# 6.2. LSTM model (TensorFlow / Keras)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_VOCAB = 20000
MAX_LEN   = 200

tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(X_train)

# Convert texts to padded sequences
X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
X_te_seq = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN)

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=MAX_VOCAB, output_dim=128, input_length=MAX_LEN),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
lstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 7. Training
print("Training Naive Bayes...")
nb_pipeline.fit(X_train, y_train)

print("\nTraining LSTM...")
lstm_history = lstm_model.fit(
    X_tr_seq, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 8. Evaluation
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 8.1. Naive Bayes
y_pred_nb = nb_pipeline.predict(X_test)
print("\n--- Naive Bayes Evaluation ---")
print("Accuracy :", accuracy_score(y_test, y_pred_nb))
print("F1 Score :", f1_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, digits=3))

# 8.2. LSTM
loss, acc = lstm_model.evaluate(X_te_seq, y_test, verbose=0)
y_pred_lstm = (lstm_model.predict(X_te_seq) > 0.5).astype(int)
print("\n--- LSTM Evaluation ---")
print(f"Accuracy : {acc:.3f}")
print("F1 Score :", f1_score(y_test, y_pred_lstm))

# 9. Improvement: Hyperparameter Tuning for Naive Bayes
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__max_df':    [0.8, 0.9, 1.0],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__alpha':       [0.01, 0.1, 1.0]
}

grid_nb = GridSearchCV(
    nb_pipeline, param_grid,
    cv=3, scoring='f1', n_jobs=-1, verbose=1
)
print("\nGrid search for Naive Bayes...")
grid_nb.fit(X_train, y_train)

print("Best NB params:", grid_nb.best_params_)
best_nb = grid_nb.best_estimator_

# Re‐evaluate tuned Naive Bayes
y_pred_best_nb = best_nb.predict(X_test)
print("Tuned NB F1 Score:", f1_score(y_test, y_pred_best_nb))

# 10. Deployment: Serialize Pipelines & Models
import joblib

# 10.1. Save Naive Bayes pipeline
NB_OUT = Path(__file__).parent / "imdb_nb_pipeline.joblib"
joblib.dump(best_nb, NB_OUT)
print(f"Saved Naive Bayes pipeline to: {NB_OUT}")

# 10.2. Save tokenizer and LSTM model
TOKEN_OUT = Path(__file__).parent / "imdb_tokenizer.joblib"
MODEL_OUT = Path(__file__).parent / "imdb_lstm_model.h5"
joblib.dump(tokenizer, TOKEN_OUT)
lstm_model.save(MODEL_OUT)
print(f"Saved tokenizer to: {TOKEN_OUT}")
print(f"Saved LSTM model to: {MODEL_OUT}")