from typing import List, Union

import pandas as pd
import numpy as np

from auto_causality.models.wrapper import DoWhyWrapper, DoWhyMethods
from auto_causality.models.transformed_outcome import TransformedOutcomeFitter


class TransformedOutcome(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=TransformedOutcomeFitter, **kwargs)


class Dummy(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=DummyModel, **kwargs)


class DummyModel(DoWhyMethods):
    def __init__(self, *args, **kwargs):
        pass

    def fit(
        self,
        df: pd.DataFrame,
    ):
        pass

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return np.random.normal(size=(len(X),))
