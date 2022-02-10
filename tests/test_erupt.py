import pytest

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

from auto_causality.erupt import ERUPT


def binary_erupt_df(mylen: int):
    treatment = np.zeros(mylen).astype(int)
    treatment[: int(mylen / 2)] = 1
    X = np.apply_along_axis(lambda x: 1 - 2 * (x % 2), 0, np.array(range(mylen)))
    df = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": (treatment * X).astype(float),
            "X": X,
        }
    )
    return df


def n_ary_erupt_df(mylen: int, treatments: int = 3):
    treatment = np.zeros(mylen).astype(int)
    treatment[: int(mylen / 2)] = 1
    X = np.apply_along_axis(lambda x: 1 - 2 * (x % 2), 0, np.array(range(mylen)))
    df = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": (treatment * X).astype(float),
            "X": X,
        }
    )
    return df


class TestErupt(object):
    def test_binary_erupt_optimal(self):
        df = binary_erupt_df(100)
        df["policy"] = df["X"] > 0
        self.call_erupt(df, 0.5)

    def test_binary_erupt_worst(self):
        df = binary_erupt_df(100)
        df["policy"] = df["X"] < 0
        self.call_erupt(df, -0.5)

    def test_binary_erupt_zero(self):
        df = binary_erupt_df(100)
        df["policy"] = 0
        df["policy"].loc[:25] = 1
        df["policy"].loc[75:] = 1
        self.call_erupt(df, 0.0)

    def call_erupt(self, df: pd.DataFrame, target_score: float):
        e = ERUPT("treatment", DummyClassifier(strategy="prior"), X_names=["X"])
        e.fit(df)
        out = e.score(df, outcome=df["outcome"], policy=df["policy"])
        assert out == target_score, "Wrong binary ERUPT score"


if __name__ == "__main__":
    pytest.main([__file__])
