from typing import Callable, List, Optional, Union
import copy

import pandas as pd
import numpy as np

from dowhy.causal_estimator import CausalEstimate

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


class ERUPTOld:
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
        # TODO: make it accept both array and callable as policy
        w = self.weights(df, policy)

        out = (w * outcome).mean()
        return out

    def weights(
        self, df: pd.DataFrame, policy: Union[Callable, np.ndarray, pd.Series]
    ) -> pd.Series:
        W = df[self.treatment_name].astype(int)
        assert all(
            [x >= 0 for x in W.unique()]
        ), "Treatment values must be non-negative integers"

        # Handle policy input
        if callable(policy):
            policy = policy(df).astype(int)
        if isinstance(policy, pd.Series):
            policy = policy.values
        policy = np.array(policy)
        d = pd.Series(index=df.index, data=policy)
        assert all(
            [x >= 0 for x in d.unique()]
        ), "Policy values must be non-negative integers"

        # Get propensity scores with better handling of edge cases
        if self.propensity_model.__class__.__name__ == "DummyPropensity":
            p = self.propensity_model.predict_proba()
        else:
            p = self.propensity_model.predict_proba(df[self.X_names])

        # Clip propensity scores to avoid division by zero or extreme weights
        min_clip = max(1e-6, self.clip)  # Ensure minimum clip is not too small
        p = np.clip(p, min_clip, 1 - min_clip)
        self._p = copy.deepcopy(p)

        # Initialize weights
        weight = np.zeros(len(df))

        # Calculate weights with safer operations
        for i in W.unique():
            mask = W == i
            p_i = p[:, i][mask]
            # Add small constant to denominator to prevent division by zero
            weight[mask] = 1 / (p_i + 1e-10)

        # Zero out weights where policy disagrees with actual treatment
        weight[d != W] = 0.0

        # Handle extreme weights
        if self.remove_tiny:
            weight[weight > 1 / self.clip] = 0.0
        else:
            weight[weight > 1 / self.clip] = 1 / self.clip

        self._weight1 = copy.deepcopy(weight)

        # Normalize weights
        sum_weight = weight.sum()
        if sum_weight > 0:
            weight *= len(df) / sum_weight
        else:
            # If all weights are zero, use uniform weights
            weight = np.ones(len(df)) / len(df)

        # Final check for NaNs
        if np.any(np.isnan(weight)):
            # Replace any remaining NaNs with uniform weights
            weight = np.ones(len(df)) / len(df)

        self._weight2 = copy.deepcopy(weight)

        return pd.Series(index=df.index, data=weight)

    def probabilistic_erupt_score(
        self,
        df: pd.DataFrame,
        outcome: pd.Series,
        estimate: CausalEstimate,
        n_samples: int = 1000,
        clip: Optional[float] = None,
    ) -> float:
        """
        Calculate ERUPT score using Thompson sampling to create a probabilistic policy.

        Args:
            df (pd.DataFrame): Input dataframe
            outcome (pd.Series): Observed outcomes
            estimate (CausalEstimate): Causal estimate containing the estimator
            n_samples (int): Number of Thompson sampling iterations
            clip (float): Optional clipping value for effect std estimates

        Returns:
            float: Thompson sampling ERUPT score
        """
        est = estimate.estimator
        cate_estimate = est.effect(df)
        if len(cate_estimate.shape) > 1 and cate_estimate.shape[1] == 1:
            cate_estimate = cate_estimate.reshape(-1)

        # Get standard errors using established methods if available
        try:
            if "Econml" in str(type(est)):
                effect_stds = est.effect_stderr(df)
            else:
                # Use empirical std as proxy for uncertainty
                effect_stds = np.std(cate_estimate) * np.ones_like(cate_estimate) * 0.5

            effect_stds = np.squeeze(effect_stds)
            if clip:
                effect_stds = np.clip(effect_stds, clip, None)

        except Exception:
            # If standard error estimation fails, use empirical std
            effect_stds = np.std(cate_estimate) * np.ones_like(cate_estimate) * 0.5
            if clip:
                effect_stds = np.clip(effect_stds, clip, None)

        # Ensure propensity scores are available
        if not hasattr(self, "propensity_model"):
            return 0.0

        # Cache propensity predictions to avoid recomputing
        try:
            if isinstance(self.propensity_model, DummyPropensity):
                p = self.propensity_model.predict_proba()
            else:
                p = self.propensity_model.predict_proba(df[self.X_names])
            p = np.maximum(p, 1e-4)
        except Exception:
            return 0.0

        # Perform Thompson sampling using matrix operations
        n_units = len(df)
        scores = np.zeros(n_samples)

        # Pre-calculate base weights
        W = df[self.treatment_name].astype(int)
        base_weights = np.zeros(len(df))
        for i in W.unique():
            base_weights[W == i] = 1 / p[:, i][W == i]

        # Sample n_samples sets of effects
        samples = np.random.normal(
            loc=cate_estimate.reshape(-1, 1),
            scale=effect_stds.reshape(-1, 1),
            size=(n_units, n_samples),
        )

        # Convert sampled effects to binary policies
        sampled_policies = (samples > 0).astype(int)

        # Calculate scores efficiently
        for i in range(n_samples):
            policy = sampled_policies[:, i]
            weights = base_weights.copy()
            weights[policy != W] = 0.0

            if self.remove_tiny:
                weights[weights > 1 / self.clip] = 0.0
            else:
                weights[weights > 1 / self.clip] = 1 / self.clip

            if weights.sum() > 0:
                weights *= len(df) / weights.sum()
                scores[i] = (weights * outcome.values).mean()

        # Return mean non-zero score
        valid_scores = scores[scores != 0]
        if len(valid_scores) > 0:
            return np.mean(valid_scores)
        return 0.0
