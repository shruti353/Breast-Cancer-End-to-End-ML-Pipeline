
# Breast Cancer End-to-End Machine Learning Pipeline

An end-to-end machine learning project demonstrating the complete ML lifecycle —
from model training and experiment tracking to deployment as a live application.

---

## 🔍 Problem Statement
Predict whether a breast tumor is **Benign** or **Malignant** using diagnostic
features derived from medical imaging data.

This is a binary classification problem with real-world healthcare relevance.

---

## 🧠 Machine Learning Pipeline

1. Data ingestion using the Breast Cancer Wisconsin dataset (sklearn)
2. Train-test split and preprocessing
3. Training multiple ML models:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
   - Gradient Boosting
4. Experiment tracking and comparison using **MLflow**
5. Best model selection based on evaluation metrics
6. Model serialization using `joblib`
7. Inference pipeline for unseen data
8. Deployment using **Gradio + Hugging Face Spaces**

---

## 🚀 Live Deployment

The trained model is deployed as an interactive web application:

🔗 **Hugging Face Space**  
https://huggingface.co/spaces/shrutithakkar/breast_cancer_prediction

Users can input feature values and receive real-time predictions.

---

## 🛠 Tech Stack

- Python
- scikit-learn
- MLflow
- NumPy
- Gradio
- Hugging Face Spaces
- Git & GitHub

---

## 📈 Model Evaluation

Multiple models were trained and compared

Best model selected based on accuracy

Model tested on unseen data before deployment

## 📌 Key Learnings

End-to-end ML workflow implementation

Experiment tracking with MLflow

Model selection and serialization

Deployment of ML models

Debugging real-world MLOps issues

## 🔮 Future Improvements

Feature selection to reduce input dimensionality

Probability-based predictions with confidence scores

Model monitoring and drift detection

 ** CI/CD for automated retraining and deployment**
=======
---
Title: Breast Cancer Prediction
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

## Breast Cancer Prediction App

This is a deployed machine learning model using Gradio.

### 🔗 Source Code
Full end-to-end ML pipeline is available here:

👉 https://github.com/shruti353/Breast-Cancer-End-to-End-ML-Pipeline
>>>>>>> 07409c7e7f9e71be7049e903bf27f40884fff83b
