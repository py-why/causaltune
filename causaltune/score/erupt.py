from typing import Callable, List, Optional, Union
import copy

import pandas as pd
import numpy as np

from causaltune.score.erupt_core import erupt

# implementation of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957
# we assume treatment takes integer values from 0 to n


class DummyPropensity:
    def __init__(self, p: pd.Series, treatment: pd.Series):
        n_vals = max(treatment) + 1
        out = np.zeros((len(treatment), n_vals))
        for i, pp in enumerate(p.values):
            out[i, treatment.values[i]] = pp
        self.p = out

    def fit(self, *args, **kwargs):
        pass

    def predict_proba(self):
        return self.p


def normalize_policy(
    df: pd.DataFrame,
    policy: Union[Callable, np.ndarray, pd.Series],
    num_treatments: int,
) -> np.ndarray:
    """
    Convert policy to matrix of probabilities
    :param df: Potential input to Callable policy
    :param policy:
    :param num_treatments: number of treatments
    :return:
    """
    # Handle policy input
    if callable(policy):
        policy = policy(df)
    if isinstance(policy, pd.Series) or isinstance(policy, pd.DataFrame):
        policy = np.squeeze(policy.values)

    n = len(policy)
    if len(policy.shape) == 1:  # vector of treatment indices
        # Convert to matrix of probabilities
        p = np.zeros((n, num_treatments))
        p[
            np.arange(n),
            policy.astype(np.int32),
        ] = 1.0
    else:
        p = policy

    assert isinstance(p, np.ndarray)
    assert p.shape == (n, num_treatments)
    return p


class ERUPT:
    def __init__(
        self,
        treatment_name: str,
        propensity_model,
        X_names: Optional[List[str]] = None,
        clip: float = 0.05,
        remove_tiny: bool = True,
        time_budget: Optional[float] = 30.0,  # Add default time budget
    ):
        """
        Initialize ERUPT with thompson sampling capability.

        Args:
            treatment_name (str): Name of treatment column
            propensity_model: Model for estimating propensity scores
            X_names (Optional[List[str]]): Names of feature columns
            clip (float): Clipping threshold for propensity scores
            remove_tiny (bool): Whether to remove tiny weights
            time_budget (Optional[float]): Time budget for AutoML propensity fitting
        """
        self.treatment_name = treatment_name
        self.propensity_model = copy.deepcopy(propensity_model)

        # If propensity model is AutoML, ensure it has time_budget
        if (
            hasattr(self.propensity_model, "time_budget")
            and self.propensity_model.time_budget is None
        ):
            self.propensity_model.time_budget = time_budget

        self.X_names = X_names
        self.clip = clip
        self.remove_tiny = remove_tiny

    def fit(self, df: pd.DataFrame):
        if self.X_names is None:
            self.X_names = [c for c in df.columns if c != self.treatment_name]
        self.propensity_model.fit(df[self.X_names], df[self.treatment_name])

    def score(
        self,
        df: pd.DataFrame,
        outcome: pd.Series,
        policy: Union[Callable, np.ndarray, pd.Series],
    ) -> float:
        actual_treatment = df[self.treatment_name].astype(int)
        treatments = np.unique(actual_treatment)
        assert (
            len(treatments) == max(treatments) + 1
        ), "The treatments must be integers from 0 to N-1, every treatment must be present in the sample"
        assert min(treatments) == 0, "The treatments must be integers from 0 to N-1"

        new_policy = normalize_policy(df, policy, len(treatments))
        propensity = self.propensity_vector(df, actual_treatment)

        return erupt(
            actual_propensity=propensity,
            actual_treatment=actual_treatment,
            actual_outcome=outcome.values,
            hypothetical_policy=new_policy,
            clip=self.clip,
            remove_tiny=self.remove_tiny,
        )

    def propensity_vector(
        self, df: pd.DataFrame, actual_treatment: np.ndarray
    ) -> np.ndarray:
        # Get propensity scores with better handling of edge cases
        if self.propensity_model.__class__.__name__ == "DummyPropensity":
            p = self.propensity_model.predict_proba()
        else:
            p = self.propensity_model.predict_proba(df[self.X_names])

        # Initialize weights
        propensity = p[np.arange(len(df)), actual_treatment]
        return propensity
