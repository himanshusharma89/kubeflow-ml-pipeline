#model_development.py
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, component

@dsl.component(base_image="python:3.9")
def train_model(
    train_features: Input[Dataset],
    train_labels: Input[Dataset],
    model_output: Output[Model]
):
    """Trains and saves a classification model."""
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn", "joblib"], check=True)
    
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from joblib import dump
    
    X = pd.read_csv(train_features.path)
    y = pd.read_csv(train_labels.path)['label']
    
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        multi_class='multinomial'
    )
    model.fit(X, y)
    
    dump(model, model_output.path)