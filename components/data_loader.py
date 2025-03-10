#data_acquisition.py
from kfp import dsl
from kfp.dsl import Output, Dataset, component

@dsl.component(base_image="python:3.9")
def fetch_data(data_output: Output[Dataset]):
    """Loads and saves the dataset."""
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)
    
    from sklearn.datasets import load_iris
    import pandas as pd
    
    iris = load_iris()
    dataset = pd.DataFrame(
        iris.data,
        columns=[col.replace(' ', '_').lower() for col in iris.feature_names]
    )
    dataset['label'] = iris.target
    
    dataset.to_csv(data_output.path, index=False)