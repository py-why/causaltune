from typing import Union, List

import pandas as pd
import numpy as np


class Passthrough:
    def __init__(self, col_name: Union[str, List[str]], include_control: bool = False):
        self.col_names = [col_name] if isinstance(col_name, str) else col_name
        assert all([isinstance(c, str) for c in self.col_names])

        self.include_control = False if len(self.col_names) == 1 else include_control

    def fit(self, *args, **kwargs) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            p = X[self.col_names].values
        else:  # EconML converts data to numpy, the below is fragile, depends on
            # column ordering in numpy and auto-causality
            p = X[:, -len(self.col_names) :]

        if self.include_control:
            out = p
        else:
            out = np.zeros((len(X), 1 + len(self.col_names)))
            out[:, 1:] = p
            out[:, 0] = 1 - p.sum(axis=1)

        assert all(out.reshape(-1) > 0.0)
        assert all(out.sum(axis=1) < 1.001)
        assert all(out.sum(axis=1) > 0.999)

        return out

    def predict(self, X: pd.DataFrame):
        return X[self.col_name].values
