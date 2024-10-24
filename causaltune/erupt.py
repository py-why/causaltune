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

        # and just for paranoia's sake let's normalize, though it shouldn't
        # matter for big samples
        weight *= len(df) / sum(weight)

        assert not np.isnan(weight.sum()), "NaNs in ERUPT weights"

        return pd.Series(index=df.index, data=weight)
    

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

    def probabilistic_erupt_score(
        self, 
        df: pd.DataFrame, 
        outcome: pd.Series,
        estimate: CausalEstimate,
        cate_estimate: np.ndarray,
        sd_threshold: float = 1e-2,
        iterations: int = 1000
    ) -> float:
        """[Previous docstring remains the same]"""
        est = estimate.estimator
        
        print(f"\nDebugging Probabilistic ERUPT for estimator: {est.__class__.__name__}")
        print(f"CATE estimate summary:")
        print(f"Mean: {np.mean(cate_estimate):.4f}")
        print(f"Std: {np.std(cate_estimate):.4f}")
        print(f"Min: {np.min(cate_estimate):.4f}")
        print(f"Max: {np.max(cate_estimate):.4f}")
        
        try:
            # Different approaches to get standard errors based on estimator type
            effect_stds = None
            
            # For DML and DR learners
            if hasattr(est, 'effect_stderr'):
                try:
                    effect_stds = est.effect_stderr(df)
                    if effect_stds is not None:
                        # Ensure correct shape
                        effect_stds = np.squeeze(effect_stds)
                    print("Got std errors from effect_stderr")
                except Exception as e:
                    print(f"effect_stderr failed: {str(e)}")
            
            # For metalearners
            if effect_stds is None and hasattr(est, 'effect_inference'):
                try:
                    inference_result = est.effect_inference(df)
                    if hasattr(inference_result, 'stderr'):
                        effect_stds = inference_result.stderr
                        effect_stds = np.squeeze(effect_stds)
                    print("Got std errors from effect_inference")
                except Exception as e:
                    print(f"effect_inference failed: {str(e)}")
            
            # If we still don't have valid standard errors, try inference method
            if effect_stds is None and hasattr(est, 'inference'):
                try:
                    inference_result = est.inference()
                    if hasattr(inference_result, 'stderr'):
                        effect_stds = inference_result.stderr
                        effect_stds = np.squeeze(effect_stds)
                    print("Got std errors from inference")
                except Exception as e:
                    print(f"inference failed: {str(e)}")
            
            # Final check if we got valid standard errors
            if effect_stds is None:
                print("Could not obtain valid standard errors")
                return 0
                
            # Check shapes match
            if effect_stds.shape != cate_estimate.shape:
                print(f"Shape mismatch: effect_stds {effect_stds.shape} vs cate_estimate {cate_estimate.shape}")
                effect_stds = np.broadcast_to(effect_stds, cate_estimate.shape)
                
            print(f"\nStandard errors summary:")
            print(f"Mean: {np.mean(effect_stds):.4f}")
            print(f"Std: {np.std(effect_stds):.4f}")
            print(f"Min: {np.min(effect_stds):.4f}")
            print(f"Max: {np.max(effect_stds):.4f}")
            
            # Check for meaningful heterogeneity
            cate_std = np.std(cate_estimate)
            if cate_std < sd_threshold:
                print(f"CATE std {cate_std:.4f} below threshold {sd_threshold} - returning 0")
                return 0

            unique_treatments = df[self.treatment_name].unique()
            print(f"\nUnique treatments: {unique_treatments}")
            treatment_scores = {treatment: [] for treatment in unique_treatments}
            
            # Normalize standard errors relative to effect size variation
            effect_stds = np.maximum(effect_stds, cate_std * 0.1)
            
            # Calculate baseline
            baseline_outcome = outcome[df[self.treatment_name] == 0].mean()
            print(f"Baseline outcome: {baseline_outcome:.4f}")

            print("\nStarting Thompson sampling iterations...")
            
            # Perform Thompson sampling iterations
            for _ in range(iterations):
                # Sample effects from posterior distributions for each treatment
                sampled_effects = {
                    treatment: np.random.normal(cate_estimate, effect_stds)
                    for treatment in unique_treatments
                }
                
                # Select treatment with highest sampled effect
                chosen_treatment = max(sampled_effects, key=lambda k: np.mean(sampled_effects[k]))
                
                # Calculate weights for the chosen treatment policy
                weights = self.weights(
                    df, 
                    lambda x: np.array([chosen_treatment] * len(x))
                )
                
                # Calculate mean outcome under this policy
                if weights.sum() > 0:
                    mean_outcome = (weights * outcome).sum() / weights.sum()
                    treatment_scores[chosen_treatment].append(mean_outcome)

            # Calculate final score
            if not any(scores for scores in treatment_scores.values()):
                print("No valid treatment scores")
                return 0
                
            average_outcomes = np.mean(
                [np.mean(scores) for scores in treatment_scores.values() if scores]
            )
            
            relative_improvement = (average_outcomes - baseline_outcome) / abs(baseline_outcome)
            print(f"Final relative improvement: {relative_improvement:.4f}")
            
            return relative_improvement
            
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            return 0