# 🔢 MNIST Digit Recognition (CNN)

## 📖 Overview
This project classifies handwritten digits (0–9) from the **MNIST dataset** using a **Convolutional Neural Network (CNN)**.  
The goal is to build a deep learning model capable of recognizing digits from pixel images with high accuracy.

---

## 📂 Project Structure
digit_recognition_mnist/  
├── data/        # mnist_dataset.csv (raw dataset or fallback to keras.datasets.mnist)  
├── src/         # pipeline.py (10-step ML workflow)  
├── results/     # metrics.json, sample_digits.png  
├── models/      # mnist_cnn.h5 (trained CNN model)  
├── api/         # FastAPI app for deployment  
└── README.md    # this file  

---

## 🛠️ Workflow
This project follows the **10-step ML workflow**:
1. Define Problem  
2. Load & Clean Data (CSV with labels or fallback to keras MNIST dataset)  
3. Exploratory Data Analysis (EDA) → sample digit visualization  
4. Feature Engineering → reshape, normalize pixel values, one-hot encode labels  
5. Train/Test Split (80/20, stratified)  
6. Model Selection → Convolutional Neural Network (CNN)  
7. Training (with early stopping)  
8. Evaluation (accuracy, confusion matrix)  
9. Improvement → tuning CNN architecture, dropout, batch size  
10. Deployment (FastAPI-ready model)

---

## 📊 Dataset
- **Source**: MNIST dataset (mnist_dataset.csv or keras.datasets.mnist)  
- **Size**: 70,000 images (60,000 train + 10,000 test)  
- **Features**: 28×28 grayscale pixel values (784 features if flattened)  
- **Target Variable**: digit label (0–9)

---

## 🤖 Model Architecture
- **Conv2D** (32 filters, 3×3, ReLU)  
- **MaxPooling2D** (2×2)  
- **Conv2D** (64 filters, 3×3, ReLU)  
- **MaxPooling2D** (2×2)  
- **Flatten**  
- **Dense** (128 units, ReLU)  
- **Dropout** (0.5)  
- **Dense** (10 units, softmax)  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  

---

## 📈 Results
Evaluation metrics are stored in results/metrics.json. Example structure:
{
  "accuracy": ...,
  "confusion_matrix": [...]
}

Generated plots:
- results/sample_digits.png → visualization of sample digits with labels

---

## 🚀 Deployment
The trained model is deployed via **FastAPI**.

Run locally:
cd api  
uvicorn main:app --reload

Endpoint:
- POST /predict → returns predicted digit

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
