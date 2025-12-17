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

inputs = [gr.Number(label=f) for f in data.feature_names]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Prediction",
    description="MLflow trained model deployed on Hugging Face Spaces"
)
