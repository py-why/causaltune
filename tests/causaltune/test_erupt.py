from typing import List
import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from causaltune.score.erupt_old import ERUPTOld


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


def n_ary_erupt_df(mylen: int, n: int = 3):
    mylen = mylen
    treatment = np.zeros(mylen * n * n).astype(int)
    for i in range(n):
        treatment[(i * mylen * n) : ((i + 1) * mylen * n)] = i

    X = np.apply_along_axis(lambda x: x % n, 0, np.array(range(mylen * n * n)))
    outcome = -np.ones(mylen * n * n).astype(int)
    outcome[X == treatment] = 1

    df = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "X": X,
        }
    )

    df = pd.concat([df, pd.get_dummies(df["X"], prefix="X_")], axis=1)
    return df


class TestErupt(object):
    def test_multi_erupt_optimal(self):
        df = n_ary_erupt_df(100, 3)
        df["policy"] = df["X"]
        self.call_erupt(df, 1.0, [c for c in df.columns if "X_" in c])

    def test_multi_erupt_constant(self):
        df = n_ary_erupt_df(100, 3)
        df["policy"] = 1
        self.call_erupt(df, -1 / 3, [c for c in df.columns if "X_" in c])

    def test_multi_erupt_worst(self):
        df = n_ary_erupt_df(100, 3)
        df["policy"] = df["X"].apply(lambda x: (x + 1) % 3)
        self.call_erupt(df, -1.0, [c for c in df.columns if "X_" in c])

    def test_binary_erupt_optimal(self):
        df = binary_erupt_df(100)
        df["policy"] = df["X"] > 0
        self.call_erupt(df, 0.5, ["X"])

    def test_binary_erupt_worst(self):
        df = binary_erupt_df(100)
        df["policy"] = df["X"] < 0
        self.call_erupt(df, -0.5, ["X"])

    def test_binary_erupt_zero(self):
        df = binary_erupt_df(100)
        df["policy"] = 0
        df["policy"].loc[:25] = 1
        df["policy"].loc[75:] = 1
        self.call_erupt(df, 0.0, ["X"])

    def call_erupt(self, df: pd.DataFrame, target_score: float, X_names: List[str]):
        e = ERUPTOld("treatment", DummyClassifier(strategy="prior"), X_names=X_names)
        e.fit(df)
        out = e.score(df, outcome=df["outcome"], policy=df["policy"])
        assert out == pytest.approx(target_score, rel=1e-5), "Wrong binary ERUPT score"


if __name__ == "__main__":
    pytest.main([__file__])
