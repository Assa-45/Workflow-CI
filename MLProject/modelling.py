import mlflow
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import set_config
from sklearn.utils import estimator_html_repr
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5002")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("student-performance-prediction")

csv_path = "data/python-learning-performance_preprocessing.csv"
data = pd.read_csv(csv_path)

X = data.drop(columns=['passed_exam'])
y = data['passed_exam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42 
)

input_example = X_train.iloc[:5]
model = LogisticRegression(
    dual=True, 
    max_iter=10000, 
    penalty='l2', 
    solver='liblinear',
    random_state=42 
)

with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")  # Debug info
    
    model.fit(X_train, y_train)

    # prediksi data latih
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)

    # mencatat parameter model
    mlflow.log_params(model.get_params())

    # membuat estimator diagram
    set_config(display='diagram')
    html_representation = estimator_html_repr(model)
    with open("estimator.html", "w", encoding="utf-8") as f:
        f.write(html_representation)
    
    # menyimpan file json metrics
    metrics_data = {
        "training_accuracy_score": accuracy_score(y_train, y_pred),
        "training_f1_score": f1_score(y_train, y_pred, average='weighted'),
        "training_precision_score": precision_score(y_train, y_pred, average='weighted'),
        "training_recall_score": recall_score(y_train, y_pred, average='weighted'),
        "training_log_loss": log_loss(y_train, y_proba),
        "training_roc_auc": roc_auc_score(y_train, y_proba[:, 1])
    }

    with open("metrics_info.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    mlflow.log_metrics(metrics_data)

    # menyimpan gambar confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, ax=ax)
    plt.title("Confusion Matrix - Training")
    plt.savefig("training_confusion_matrix.png")
    plt.close(fig)

    # log semua artifak
    mlflow.log_artifact("estimator.html")
    mlflow.log_artifact("metrics_info.json")
    mlflow.log_artifact("training_confusion_matrix.png")
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)
    
