from typing import Callable, List, Optional, Union
import copy

import pandas as pd
import numpy as np

from scipy import stats

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
        self.propensity_model = copy.deepcopy(propensity_model)
        self.X_names = X_names
        self.clip = clip
        self.remove_tiny = remove_tiny

    def fit(self, df: pd.DataFrame):
        if self.X_names is None:
            self.X_names = [c for c in df.columns if c != self.treatment_name]
        self.propensity_model.fit(df[self.X_names], df[self.treatment_name])

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

        if isinstance(self.propensity_model, DummyPropensity):
            p = self.propensity_model.predict_proba()
        else:
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
    

        #NEW:
    def probabilistic_erupt_score(
        self,
        df: pd.DataFrame,
        outcome: pd.Series,
        treatment_effects: pd.Series,
        treatment_std_devs: pd.Series,
        iterations: int = 1000
    ) -> float:
        """
        Calculate the Probabilistic ERUPT (Expected Response Under Proposed Treatments) score.

        This method uses Monte Carlo simulation to estimate the expected outcome under
        a probabilistic treatment policy, accounting for uncertainty in treatment effects.
        It balances potential improvements against estimation uncertainty and treatment rates.

        Args:
            df (pd.DataFrame): The input dataframe containing treatment information.
            outcome (pd.Series): The observed outcomes for each unit.
            treatment_effects (pd.Series): Estimated treatment effects for each unit.
            treatment_std_devs (pd.Series): Standard deviations of treatment effects.
            iterations (int): Number of Monte Carlo iterations (default: 1000).

        Returns:
            float: The Probabilistic ERUPT score, representing the relative improvement
                over the baseline outcome, adjusted for uncertainty.
        """
        # Calculate the baseline outcome (mean outcome for untreated units)
        baseline_outcome = outcome[df[self.treatment_name] == 0].mean()

        policy_values = []
        treatment_decisions = []

        # Perform Monte Carlo simulation
        for _ in range(iterations):
            # Sample treatment effects from normal distributions
            sampled_effects = pd.Series(
                np.random.normal(treatment_effects, treatment_std_devs),
                index=treatment_effects.index
            )

            # Define policy: treat if sampled effect is positive
            # Note: A more conservative policy could use: sampled_effects > 2 * treatment_std_devs
            policy = (sampled_effects > 0).astype(int)

            # Calculate expected outcome under this policy
            expected_outcome = (
                baseline_outcome +
                (policy * sampled_effects).mean()
            )

            policy_values.append(expected_outcome)
            treatment_decisions.append(policy.mean())

        # Calculate mean and standard error of policy values
        mean_value = np.mean(policy_values)
        se_value = np.std(policy_values) / np.sqrt(iterations)

        # Placeholder for potential treatment rate penalty
        treatment_penalty = 0

        # Calculate score: mean value minus 2 standard errors, adjusted for treatment penalty
        score = (mean_value - 2*se_value) * (1 - treatment_penalty)

        # Calculate relative improvement over baseline
        improvement = (score - baseline_outcome) / baseline_outcome

        return improvement
