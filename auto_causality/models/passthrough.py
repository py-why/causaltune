import pandas as pd
import numpy as np


class Passthrough:
    def __init__(self, col_name: str):
        self.col_name = col_name

    def fit(self, *args, **kwargs) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            p = X[self.col_name].values
        else:
            p = X[:, -1]
        out = np.zeros((len(X), 2))
        out[:, 1] = p
        out[:, 0] = 1 - p
        return out

    def predict(self, X: pd.DataFrame):
        return X[self.col_name].values
