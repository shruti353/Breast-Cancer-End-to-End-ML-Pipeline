import gradio as gr
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load model
model = joblib.load("breast_cancer_model.pkl")
data = load_breast_cancer()

def predict(*features):
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)[0]
    return "Benign" if pred == 1 else "Malignant"

inputs = [gr.Number(label=f) for f in data.feature_names]

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Prediction",
    description="MLflow-trained sklearn model deployed on Hugging Face Spaces"
)

# ✅ THIS is what keeps the container alive
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
