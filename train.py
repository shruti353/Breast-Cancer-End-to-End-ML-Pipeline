import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier()
}

best_acc = 0
best_model = None

mlflow.set_experiment("Breast Cancer Prediction")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name)

        if acc > best_acc:
            best_acc = acc
            best_model = model

# Save best model
joblib.dump(best_model, "breast_cancer_model.pkl")
print("Best model saved with accuracy:", best_acc)
