#feature_preparation.py
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, component

@dsl.component(base_image="python:3.9")
def process_features(
    raw_data: Input[Dataset],
    train_features: Output[Dataset],
    test_features: Output[Dataset],
    train_labels: Output[Dataset],
    test_labels: Output[Dataset]
):
    """Prepares data by normalizing and splitting into training/testing sets."""
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split
    
    dataset = pd.read_csv(raw_data.path)
    assert dataset.notna().all().all(), "Dataset contains missing values"
    
    features = dataset.drop(columns=['label'])
    target = dataset['label']
    
    scaler = RobustScaler()
    normalized_features = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, 
        target,
        test_size=0.25,
        random_state=42,
        stratify=target
    )
    
    pd.DataFrame(X_train, columns=features.columns).to_csv(train_features.path, index=False)
    pd.DataFrame(X_test, columns=features.columns).to_csv(test_features.path, index=False)
    pd.DataFrame(y_train, columns=['label']).to_csv(train_labels.path, index=False)
    pd.DataFrame(y_test, columns=['label']).to_csv(test_labels.path, index=False)