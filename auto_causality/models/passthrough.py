from typing import Union, List
import copy

import pandas as pd
import numpy as np


def feature_filter(ModelType: type, col_names: List[str], first_cols: bool = False):
    """
    Constrain a model to only use a subset of the supplied features
    @param ModelType: The class of the model to be thus constrained
    @param col_names: The names of the columns to keep
    @param first_cols: If input is ndarray, take first or last N columns?
    @return: The constrained class, inheriting from the original one
    """

    class FilterFeatures(ModelType):
        def __init__(self, *args, **kwargs):
            super(FilterFeatures, self).__init__(*args, **kwargs)

            self.col_names = copy.deepcopy(col_names)
            assert all([isinstance(c, str) for c in self.col_names])

            self.first_cols = first_cols

        def filter_X(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if isinstance(X, pd.DataFrame):
                return X[self.col_names].values
            else:  # EconML converts data to numpy, the below is fragile, depends on
                # column ordering in numpy and auto-causality
                if self.first_cols:
                    return X[:, : len(self.col_names)]
                else:
                    return X[:, -len(self.col_names) :]

        def fit(self, X: Union[pd.DataFrame, np.ndarray], *args, **kwargs) -> None:
            super().fit(self.filter_X(X), *args, **kwargs)

        def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], *args, **kwargs):
            return super().predict_proba(self.filter_X(X), *args, **kwargs)

        def predict(self, X: Union[pd.DataFrame, np.ndarray], *args, **kwargs):
            return super().predict(self.filter_X(X), *args, **kwargs)

        def score(self, X: Union[pd.DataFrame, np.ndarray], *args, **kwargs):
            return super().score(self.filter_X(X), *args, **kwargs)

    return FilterFeatures


class PassthroughInner:
    def __init__(self, include_control: bool = False):
        """
        Init method
        @param include_control: is the probability of control included,
        that is do all the probabilities sum up to one?
        """
        self.include_control = include_control

    def fit(self, *args, **kwargs) -> None:
        pass

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            p = X.values
        else:  # EconML converts data to numpy, the below is fragile, depends on
            # column ordering in numpy and auto-causality
            p = X

        if self.include_control:
            out = p
        else:
            first_col = 1 - p.sum(axis=1, keepdims=True)
            out = np.concatenate([first_col, p], axis=1)

        assert all(out.reshape(-1) > 0.0)
        assert all(out.sum(axis=1) < 1.001)
        assert all(out.sum(axis=1) > 0.999)

        return out

    # This is needed for the FilterFeatures wrapper to work
    def score(self, *args, **kwargs):
        return 0.0


def passthrough_model(col_names: Union[str, List[str]], include_control: bool = False):
    if isinstance(col_names, str):
        col_names = [col_names]

    pt_class = feature_filter(PassthroughInner, col_names, first_cols=False)
    return pt_class(include_control)
