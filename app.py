import gradio as gr
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

model = joblib.load("breast_cancer_model.pkl")
data = load_breast_cancer()

def predict(*features):
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)[0]
    return "Benign" if pred == 1 else "Malignant"

inputs = [gr.Number(label=feat) for feat in data.feature_names]

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Prediction (MLflow + ML)",
    description="Best model selected using MLflow and deployed on Hugging Face"
)

app.launch()
