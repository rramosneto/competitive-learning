import pandas as pd

from competitive_learning.model import Dataset


def load_csv(file):
    if "iris" or "Iris" in file:
        return load_iris(file)


def load_iris(file):
    df = pd.read_csv(file)
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    dataset = Dataset.from_pandas(df.drop(columns=["species"]))

    return df, dataset
