#!/usr/bin/env python3
"""
Spam Detection Pipeline (Enron Dataset)
"""

import re, json, joblib
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Paths
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "enron_email_data.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df.dropna(subset=['message'], inplace=True)
    df['clean_msg'] = df['message'].apply(preprocess)
    df['label'] = df['category'].map({'ham': 0, 'spam': 1})
    return df

def run_eda(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='category', data=df)
    plt.title("Email Categories")
    plt.savefig(RESULTS_DIR / "category_counts.png")
    plt.close()

    df['length'] = df['message'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8,4))
    sns.histplot(df, x='length', hue='category', bins=50, alpha=0.6)
    plt.title("Message Length by Category")
    plt.savefig(RESULTS_DIR / "message_length_distribution.png")
    plt.close()

def train_and_evaluate(df):
    X, y = df['clean_msg'], df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
        ('clf', LinearSVC(C=1.0, max_iter=10000, random_state=42))
    ])

    nb_pipeline.fit(X_train, y_train)
    svm_pipeline.fit(X_train, y_train)

    # Hyperparameter tuning
    nb_params = {'tfidf__max_df':[0.8,0.9,1.0],'tfidf__ngram_range':[(1,1),(1,2)],'clf__alpha':[0.01,0.1,1.0]}
    grid_nb = GridSearchCV(nb_pipeline, nb_params, cv=3, scoring='f1', n_jobs=-1)
    grid_nb.fit(X_train, y_train)

    svm_params = {'tfidf__max_df':[0.8,0.9,1.0],'tfidf__ngram_range':[(1,1),(1,2)],'clf__C':[0.1,1.0,10.0]}
    grid_svm = GridSearchCV(svm_pipeline, svm_params, cv=3, scoring='f1', n_jobs=-1)
    grid_svm.fit(X_train, y_train)

    nb_f1  = f1_score(y_test, grid_nb.best_estimator_.predict(X_test))
    svm_f1 = f1_score(y_test, grid_svm.best_estimator_.predict(X_test))
    best_model = grid_nb.best_estimator_ if nb_f1 >= svm_f1 else grid_svm.best_estimator_
    model_name = "nb" if nb_f1 >= svm_f1 else "svm"

    metrics = {}
    for name, model in [("Naive Bayes", nb_pipeline), ("Linear SVM", svm_pipeline),
                        ("Tuned Naive Bayes", grid_nb.best_estimator_), ("Tuned Linear SVM", grid_svm.best_estimator_)]:
        preds = model.predict(X_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "classification_report": classification_report(y_test, preds, digits=3, output_dict=True)
        }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(best_model, MODELS_DIR / f"spam_{model_name}_pipeline.joblib")
    return metrics

def main():
    df = load_and_clean()
    run_eda(df)
    results = train_and_evaluate(df)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()