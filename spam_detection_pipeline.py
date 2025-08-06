#!/usr/bin/env python3
"""
spam_detection_pipeline.py

A full ML workflow to classify Enron emails as spam or ham:
1. Define Problem
2. Load & Clean Data
3. Exploratory Data Analysis
4. Feature Engineering
5. Train/Test Split
6. Model Selection
7. Training
8. Evaluation
9. Improvement (hyperparameter tuning)
10. Deployment (serialize best model)

About Dataset

The Enron email CSV has columns:
- Category: “ham” or “spam”
- Message : the raw email text
"""

# 1. Define the Problem
#    Binary classification: predict spam (1) vs. ham (0) from email text.

# 2. Load & Clean Data
import re
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(
    "C:/Users/Stasis Mukwenya/Documents/Code Scripts"
    "/Cloudcredits Datasets/enron_email_data.csv"
)

df = pd.read_csv(DATA_PATH)
# normalize column names
df.columns = df.columns.str.strip().str.lower()

print("Original shape:", df.shape)
print(df['category'].value_counts(), "\n")

# Drop any rows with missing messages
df.dropna(subset=['message'], inplace=True)

# 3. Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Spam vs Ham counts
plt.figure(figsize=(6,4))
sns.countplot(x='category', data=df)
plt.title("Email Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# Message length distribution
df['length'] = df['message'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,4))
sns.histplot(df, x='length', hue='category', bins=50, alpha=0.6)
plt.title("Message Length by Category")
plt.xlabel("Word Count")
plt.show()

# 4. Feature Engineering
# Clean text: remove punctuation, lowercase
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)      # strip HTML tags (if any)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # keep alphanumeric + spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_msg'] = df['message'].apply(preprocess)
df['label'] = df['category'].map({'ham': 0, 'spam': 1})

# 5. Train/Test Split
from sklearn.model_selection import train_test_split

X = df['clean_msg']
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 6. Model Selection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Multinomial Naive Bayes pipeline
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
    ('clf',   MultinomialNB(alpha=0.1))
])

# Linear SVM pipeline
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
    ('clf',   LinearSVC(C=1.0, max_iter=10000, random_state=42))
])

# 7. Training
print("Training Naive Bayes...")
nb_pipeline.fit(X_train, y_train)

print("Training Linear SVM...")
svm_pipeline.fit(X_train, y_train)

# 8. Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def evaluate(name, model):
    preds = model.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    print(f"\n{name} Performance:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(classification_report(y_test, preds, digits=3))

evaluate("Naive Bayes", nb_pipeline)
evaluate("Linear SVM", svm_pipeline)

# 9. Improvement: Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Tune Naive Bayes
nb_params = {
    'tfidf__max_df':    [0.8, 0.9, 1.0],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__alpha':       [0.01, 0.1, 1.0]
}

grid_nb = GridSearchCV(
    nb_pipeline, nb_params,
    cv=3, scoring='f1', n_jobs=-1, verbose=1
)
print("\nTuning Naive Bayes...")
grid_nb.fit(X_train, y_train)
print("Best NB params:", grid_nb.best_params_)

# Tune Linear SVM
svm_params = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1.0, 10.0]
}

grid_svm = GridSearchCV(
    svm_pipeline, svm_params,
    cv=3, scoring='f1', n_jobs=-1, verbose=1
)
print("\nTuning Linear SVM...")
grid_svm.fit(X_train, y_train)
print("Best SVM params:", grid_svm.best_params_)

# Evaluate tuned models
evaluate("Tuned Naive Bayes", grid_nb.best_estimator_)
evaluate("Tuned Linear SVM",  grid_svm.best_estimator_)

# 10. Deployment: Serialize Best Model
import joblib

# Compare F1 scores on test set
from sklearn.metrics import f1_score

nb_f1  = f1_score(y_test, grid_nb.best_estimator_.predict(X_test))
svm_f1 = f1_score(y_test, grid_svm.best_estimator_.predict(X_test))

best_model = grid_nb.best_estimator_ if nb_f1 >= svm_f1 else grid_svm.best_estimator_
model_name = "nb" if nb_f1 >= svm_f1 else "svm"

OUT_PATH = Path(__file__).parent / f"spam_{model_name}_pipeline.joblib"
joblib.dump(best_model, OUT_PATH)
print(f"\nSaved best pipeline ({model_name}) to: {OUT_PATH}")