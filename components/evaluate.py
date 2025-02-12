from kfp.dsl import component

@component
def evaluate(input_path: str, model_path: str, metrics_path: str):
    import numpy as np
    import joblib
    from sklearn.metrics import accuracy_score
    import os

    X_test = np.load(f"{input_path}/X_test.npy")
    y_test = np.load(f"{input_path}/y_test.npy")
    
    model = joblib.load(f"{model_path}/model.pkl")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    os.makedirs(metrics_path, exist_ok=True)
    with open(f"{metrics_path}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
