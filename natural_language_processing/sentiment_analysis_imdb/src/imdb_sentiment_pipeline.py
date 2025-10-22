#!/usr/bin/env python3
"""
IMDb Sentiment Analysis Pipeline
"""

import re, json, joblib
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "IMDB Dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=['review','sentiment'], inplace=True)
    df['clean_review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'negative':0,'positive':1})
    return df

def run_eda(df):
    df['length'] = df['clean_review'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x='length', hue='sentiment', bins=50, alpha=0.6)
    plt.title("Review Length Distribution by Sentiment")
    plt.xlabel("Word Count")
    plt.savefig(RESULTS_DIR / "review_length_distribution.png")
    plt.close()

def train_and_evaluate(df):
    X, y = df['clean_review'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    # Naive Bayes
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    nb_pipeline.fit(X_train, y_train)
    y_pred_nb = nb_pipeline.predict(X_test)

    # LSTM
    MAX_VOCAB, MAX_LEN = 20000, 200
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    tokenizer.fit_on_texts(X_train)
    X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_te_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

    lstm_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=MAX_VOCAB, output_dim=128, input_length=MAX_LEN),
        tf.keras.layers.SpatialDropout1D(0.2),
        tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_tr_seq, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

    loss, acc = lstm_model.evaluate(X_te_seq, y_test, verbose=0)
    y_pred_lstm = (lstm_model.predict(X_te_seq) > 0.5).astype(int)

    # Metrics
    metrics = {
        "Naive Bayes": {
            "accuracy": float(accuracy_score(y_test, y_pred_nb)),
            "f1": float(f1_score(y_test, y_pred_nb)),
            "classification_report": classification_report(y_test, y_pred_nb, digits=3, output_dict=True)
        },
        "LSTM": {
            "accuracy": float(acc),
            "f1": float(f1_score(y_test, y_pred_lstm))
        }
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save models
    joblib.dump(nb_pipeline, MODELS_DIR / "imdb_nb_pipeline.joblib")
    joblib.dump(tokenizer, MODELS_DIR / "imdb_tokenizer.joblib")
    lstm_model.save(MODELS_DIR / "imdb_lstm_model.h5")

    return metrics

def main():
    df = load_and_clean()
    run_eda(df)
    results = train_and_evaluate(df)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()