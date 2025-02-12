from kfp.dsl import component

@component
def preprocess(output_path: str) -> str:  # Output type as string (output path)
    import numpy as np
    from sklearn import datasets
    import os

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    os.makedirs(output_path, exist_ok=True)
    np.save(f"{output_path}/X_train.npy", X)
    np.save(f"{output_path}/y_train.npy", y)

    return output_path  # Return output path as string
