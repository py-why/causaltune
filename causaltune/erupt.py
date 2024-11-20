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
        self, df: pd.DataFrame, outcome: pd.Series, policy: Callable
    ) -> pd.Series:
        # TODO: make it accept both array and callable as policy
        w = self.weights(df, policy)
        return np.round((w * outcome).mean(), decimals=12)

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
        if isinstance(self.propensity_model, DummyPropensity):
            p = self.propensity_model.predict_proba()
        else:
            try:
                p = self.propensity_model.predict_proba(df[self.X_names])
            except Exception:
                # Fallback to safe defaults if prediction fails
                p = np.full((len(df), 2), 0.5)

        # Clip propensity scores to avoid division by zero or extreme weights
        min_clip = max(1e-6, self.clip)  # Ensure minimum clip is not too small
        p = np.clip(p, min_clip, 1 - min_clip)

        # Initialize weights
        weight = np.zeros(len(df))

        try:
            # Calculate weights with safer operations
            for i in W.unique():
                mask = W == i
                p_i = p[:, i][mask]
                # Add small constant to denominator to prevent division by zero
                weight[mask] = 1 / (p_i + 1e-10)
        except Exception:
            # If something goes wrong, return safe weights
            weight = np.ones(len(df))

        # Zero out weights where policy disagrees with actual treatment
        weight[d != W] = 0.0

        # Handle extreme weights
        if self.remove_tiny:
            weight[weight > 1 / self.clip] = 0.0
        else:
            weight[weight > 1 / self.clip] = 1 / self.clip

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

    def thompson_weights(
        self,
        df: pd.DataFrame,
        cate_estimate: np.ndarray,
        effect_stds: np.ndarray,
        n_samples: int = 1,
    ) -> pd.Series:
        """Helper method to get weights for a single Thompson sampling iteration"""
        samples = np.random.normal(cate_estimate, effect_stds)
        policy = (samples > 0).astype(int)
        return self.weights(df, lambda x: policy)

    # def probabilistic_erupt_score(
    #     self,
    #     df: pd.DataFrame,
    #     outcome: pd.Series,
    #     estimate: CausalEstimate,
    #     cate_estimate: np.ndarray,
    #     sd_threshold: float = 1e-2,
    #     iterations: int = 1000
    # ) -> float:
    #     """
    #     Calculate the Probabilistic ERUPT score using Thompson sampling to select
    #     optimal treatments under uncertainty.

    #     This implementation utilizes Thompson sampling by selecting treatments that
    #     maximize expected outcomes based on sampled treatment effects. For each iteration,
    #     effects are sampled from posterior distributions and treatments are assigned
    #     to maximize the expected outcome.

    #     Args:
    #         df (pd.DataFrame): Input dataframe with treatment data
    #         outcome (pd.Series): Observed outcomes for each unit
    #         estimate (CausalEstimate): Causal estimate to evaluate
    #         cate_estimate (np.ndarray): Array with CATE estimates
    #         sd_threshold (float): Minimum standard deviation to consider meaningful variation
    #         iterations (int): Number of Thompson sampling iterations

    #     Returns:
    #         float: Probabilistic ERUPT score or 0 if variance estimation not available
    #     """
    #     est = estimate.estimator

    #     # Check if estimator supports inference
    #     if not hasattr(est, 'inference') or not hasattr(est, 'effect_stderr'):
    #         return 0

    #     try:
    #         # Get standard errors
    #         effect_stds = est.effect_stderr(df)

    #         # Check if we got valid standard errors
    #         if effect_stds is None:
    #             return 0

    #         # Check for meaningful heterogeneity in treatment effects
    #         cate_std = np.std(cate_estimate)
    #         if cate_std < sd_threshold:
    #             return 0

    #         unique_treatments = df[self.treatment_name].unique()
    #         treatment_scores = {treatment: [] for treatment in unique_treatments}

    #         # Normalize standard errors relative to effect size variation
    #         effect_stds = np.maximum(effect_stds, cate_std * 0.1)  # Prevent overconfidence

    #         # Calculate baseline outcome for reference
    #         baseline_outcome = outcome[df[self.treatment_name] == 0].mean()

    #         # Perform Thompson sampling iterations
    #         for _ in range(iterations):
    #             # Sample effects while maintaining relative relationships
    #             sampled_effects = np.random.normal(cate_estimate, effect_stds)

    #             # Apply treatment policy based on sampled effects
    #             policy = (sampled_effects > np.median(sampled_effects)).astype(int)

    #             # Calculate weights for this policy
    #             weights = self.weights(df, policy)

    #             # Skip if weights sum to zero
    #             if weights.sum() == 0:
    #                 continue

    #             # Calculate mean outcome under this policy
    #             weighted_outcome = (weights * outcome).sum() / weights.sum()
    #             treatment_scores[1].append(weighted_outcome)  # Store under treatment=1

    #         # If no valid iterations, return 0
    #         if not any(scores for scores in treatment_scores.values()):
    #             return 0

    #         # Calculate improvement over baseline
    #         average_treatment_outcome = np.mean(treatment_scores[1])
    #         relative_improvement = (average_treatment_outcome - baseline_outcome) / abs(baseline_outcome)

    #         return relative_improvement

    #     except (AttributeError, ValueError) as e:
    #         return 0

    # def probabilistic_erupt_score(
    #     self,
    #     df: pd.DataFrame,
    #     outcome: pd.Series,
    #     estimate: CausalEstimate,
    #     cate_estimate: np.ndarray,
    #     sd_threshold: float = 1e-2,
    #     iterations: int = 1000,
    # ) -> float:
    #     """[Previous docstring remains the same]"""
    #     est = estimate.estimator

    #     print(
    #         f"\nDebugging Probabilistic ERUPT for estimator: {est.__class__.__name__}"
    #     )
    #     print("CATE estimate summary:")
    #     print(f"Mean: {np.mean(cate_estimate):.4f}")
    #     print(f"Std: {np.std(cate_estimate):.4f}")
    #     print(f"Min: {np.min(cate_estimate):.4f}")
    #     print(f"Max: {np.max(cate_estimate):.4f}")

    #     try:
    #         # Different approaches to get standard errors based on estimator type
    #         effect_stds = None

    #         # For DML and DR learners
    #         if hasattr(est, "effect_stderr"):
    #             try:
    #                 effect_stds = est.effect_stderr(df)
    #                 if effect_stds is not None:
    #                     # Ensure correct shape
    #                     effect_stds = np.squeeze(effect_stds)
    #                 print("Got std errors from effect_stderr")
    #             except Exception as e:
    #                 print(f"effect_stderr failed: {str(e)}")

    #         # For metalearners
    #         if effect_stds is None and hasattr(est, "effect_inference"):
    #             try:
    #                 inference_result = est.effect_inference(df)
    #                 if hasattr(inference_result, "stderr"):
    #                     effect_stds = inference_result.stderr
    #                     effect_stds = np.squeeze(effect_stds)
    #                 print("Got std errors from effect_inference")
    #             except Exception as e:
    #                 print(f"effect_inference failed: {str(e)}")

    #         # If we still don't have valid standard errors, try inference method
    #         if effect_stds is None and hasattr(est, "inference"):
    #             try:
    #                 inference_result = est.inference()
    #                 if hasattr(inference_result, "stderr"):
    #                     effect_stds = inference_result.stderr
    #                     effect_stds = np.squeeze(effect_stds)
    #                 print("Got std errors from inference")
    #             except Exception as e:
    #                 print(f"inference failed: {str(e)}")

    #         # Final check if we got valid standard errors
    #         if effect_stds is None:
    #             print("Could not obtain valid standard errors")
    #             return 0

    #         # Check shapes match
    #         if effect_stds.shape != cate_estimate.shape:
    #             print(
    #                 f"Shape mismatch: effect_stds {effect_stds.shape} vs cate_estimate {cate_estimate.shape}"
    #             )
    #             effect_stds = np.broadcast_to(effect_stds, cate_estimate.shape)

    #         print("\nStandard errors summary:")
    #         print(f"Mean: {np.mean(effect_stds):.4f}")
    #         print(f"Std: {np.std(effect_stds):.4f}")
    #         print(f"Min: {np.min(effect_stds):.4f}")
    #         print(f"Max: {np.max(effect_stds):.4f}")

    #         # Check for meaningful heterogeneity
    #         cate_std = np.std(cate_estimate)
    #         if cate_std < sd_threshold:
    #             print(
    #                 f"CATE std {cate_std:.4f} below threshold {sd_threshold} - returning 0"
    #             )
    #             return 0

    #         unique_treatments = df[self.treatment_name].unique()
    #         print(f"\nUnique treatments: {unique_treatments}")
    #         treatment_scores = {treatment: [] for treatment in unique_treatments}

    #         # Normalize standard errors relative to effect size variation
    #         effect_stds = np.maximum(effect_stds, cate_std * 0.1)

    #         # Calculate baseline
    #         baseline_outcome = outcome[df[self.treatment_name] == 0].mean()
    #         print(f"Baseline outcome: {baseline_outcome:.4f}")

    #         print("\nStarting Thompson sampling iterations...")

    #         # Perform Thompson sampling iterations
    #         for _ in range(iterations):
    #             # Sample effects from posterior distributions for each treatment
    #             sampled_effects = {
    #                 treatment: np.random.normal(cate_estimate, effect_stds)
    #                 for treatment in unique_treatments
    #             }

    #             # Select treatment with highest sampled effect
    #             chosen_treatment = max(
    #                 sampled_effects, key=lambda k: np.mean(sampled_effects[k])
    #             )

    #             # Calculate weights for the chosen treatment policy
    #             weights = self.weights(
    #                 df, lambda x: np.array([chosen_treatment] * len(x))
    #             )

    #             # # Calculate mean outcome under this policy
    #             if weights.sum() > 0:
    #                 mean_outcome = (weights * outcome).sum() / weights.sum()
    #                 treatment_scores[chosen_treatment].append(mean_outcome)

    #         # Calculate final score
    #         if not any(scores for scores in treatment_scores.values()):
    #             print("No valid treatment scores")
    #             return 0

    #         average_outcomes = np.mean(
    #             [np.mean(scores) for scores in treatment_scores.values() if scores]
    #         )

    #         relative_improvement = (average_outcomes - baseline_outcome) / abs(
    #             baseline_outcome
    #         )
    #         print(f"Final relative improvement: {relative_improvement:.4f}")

    #         return relative_improvement

    #     except Exception as e:
    #         print(f"Exception occurred: {str(e)}")
    #         return 0
