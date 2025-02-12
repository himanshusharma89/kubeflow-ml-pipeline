from kfp.dsl import component

@component
def train(input_path: str, model_path: str) -> str:
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import os

    # Load the data
    X = np.load(f"{input_path}/X_train.npy")
    y = np.load(f"{input_path}/y_train.npy")

    # Train a model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the model
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "model.joblib")
    joblib.dump(model, model_file)

    return model_file  # Returning the path to the model as output
