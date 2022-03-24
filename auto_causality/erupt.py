from typing import Callable, List, Optional, Union

import pandas as pd
import numpy as np


# implementation of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957
# we assume treatment takes integer values from 0 to n
class ERUPT:
    def __init__(
        self,
        treatment_name: str,
        propensity_model,
        X_names: None = Optional[List[str]],
        clip: float = 0.05,
        remove_tiny: bool = True,
    ):
        self.treatment_name = treatment_name
        self.propensity_model = propensity_model
        self.X_names = X_names
        self.clip = clip
        self.remove_tiny = remove_tiny

    def fit(self, df: pd.DataFrame):
        if self.X_names is None:
            self.X_names = [c for c in df.columns if c != self.treatment_name]
        self.propensity_model.fit(X=df[self.X_names], y=df[self.treatment_name])

    def score(
        self, df: pd.DataFrame, outcome: pd.Series, policy: Callable
    ) -> pd.Series:
        # TODO: make it accept both array and callable as policy
        w = self.weights(df, policy)
        return (w * outcome).mean()

    def weights(
        self, df: pd.DataFrame, policy: Union[Callable, np.ndarray, pd.Series]
    ) -> pd.Series:

        W = df[self.treatment_name].astype(int)
        assert all(
            [x >= 0 for x in W.unique()]
        ), "Treatment values must be non-negative integers"

        if callable(policy):
            policy = policy(df).astype(int)
        if isinstance(policy, pd.Series):
            policy = policy.values
        policy = np.array(policy)

        d = pd.Series(index=df.index, data=policy)
        assert all(
            [x >= 0 for x in d.unique()]
        ), "Policy values must be non-negative integers"

        p = self.propensity_model.predict_proba(df[self.X_names])
        # normalize to hopefully avoid NaNs
        p = np.maximum(p, 1e-4)

        weight = np.zeros(len(df))

        for i in W.unique():
            weight[W == i] = 1 / p[:, i][W == i]

        weight[d != W] = 0.0

        if self.remove_tiny:
            weight[weight > 1 / self.clip] = 0.0
        else:
            weight[weight > 1 / self.clip] = 1 / self.clip

        # and just for paranoia's sake let's normalize, though it shouldn't matter for big samples
        weight *= len(df) / sum(weight)

        assert not np.isnan(weight.sum()), "NaNs in ERUPT weights"

        return pd.Series(index=df.index, data=weight)
