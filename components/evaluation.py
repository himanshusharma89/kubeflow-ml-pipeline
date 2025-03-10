from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, component

@dsl.component(base_image="python:3.9")
def evaluate_model(
    test_features: Input[Dataset],
    test_labels: Input[Dataset],
    model_file: Input[Model],
    evaluation_output: Output[Dataset]
):
    """Evaluates model performance and generates a report."""
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn", "seaborn", "joblib"], check=True)
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    from joblib import load
    
    X_test = pd.read_csv(test_features.path)
    y_true = pd.read_csv(test_labels.path)['label']
    model = load(model_file.path)
    
    y_pred = model.predict(X_test)
    
    metrics = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    
    results = {
        'metrics': metrics,
        'confusion_matrix': conf_matrix.tolist()
    }
    pd.DataFrame([results]).to_json(evaluation_output.path)