from typing import Callable, List

import pandas as pd


# implementation of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957
class ERUPT:
    def __init__(
        self,
        treatment: str,
        propensity_model,
        clip: float = 0.05,
        remove_tiny: bool = True,
    ):
        self.treatment = treatment
        self.propensity = propensity_model
        self.clip = clip
        self.remove_tiny = remove_tiny

    def fit(self, df: pd.DataFrame, regressors: List[str] = None):
        if regressors is None:
            regressors = [c for c in df.columns if c != self.treatment]
        self.treatment_X = regressors
        self.propensity.fit(X=df[self.treatment_X], y=df[self.treatment])

    def score(
        self, df: pd.DataFrame, outcome: pd.Series, policy: Callable
    ) -> pd.Series:
        w = self.weights(df, policy)
        return (w * outcome).mean()

    def weights(self, df: pd.DataFrame, policy: Callable) -> pd.Series:
        W = df[self.treatment].astype(int)
        assert all(
            [x in [0, 1] for x in W.unique()]
        ), "Only boolean treatments supported as yet"

        d = pd.Series(index=df.index, data=policy(df).astype(int))
        assert all(
            [x in [0, 1] for x in d.unique()]
        ), "Only boolean treatments supported as yet"

        p = self.propensity.predict_proba(df)[:, 1]

        weight = 1 / p
        weight[W == 0] = 1 / (1 - p[W == 0])
        weight[d != W] = 0.0

        if self.remove_tiny:
            weight[weight > 1 / self.clip] = 0.0
        else:
            weight[weight > 1 / self.clip] = 1 / self.clip

        # and just for paranoia's sake let's normalize, though it shouldn't matter for big samples
        weight *= len(df) / sum(weight)

        return weight
