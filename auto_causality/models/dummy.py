from typing import List, Callable, Any

import numpy as np
import pandas as pd

# from auto_causality.models.monkey_patches import PropensityScoreWeightingEstimator
from auto_causality.models.wrapper import DoWhyMethods, DoWhyWrapper
from auto_causality.scoring import Scorer

# from auto_causality.scoring import ate


# Let's engage in a bit of monkey patching as we wait for this to be merged into DoWhy
# #TODO: delete this as soon as PR #485 is merged into dowhy
def effect(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
    # combining them in this way allows to override method_params from kwargs
    extra_params = {**self.method_params, **kwargs}
    new_estimator = type(self)(
        data=df,
        identified_estimand=self._target_estimand,
        treatment=self._target_estimand.treatment_variable,
        outcome=self._target_estimand.outcome_variable,
        test_significance=False,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        target_units=self._target_units,
        effect_modifiers=self._effect_modifier_names,
        **extra_params,
    )
    scalar_effect = new_estimator.estimate_effect()
    return np.ones(len(df)) * scalar_effect.value


# class OutOfSamplePSWEstimator(PropensityScoreWeightingEstimator):
#     """
#     A flavor of PSWEstimator that doesn't refit the propensity function
#     when doing out-of-sample evaluation
#     """
#
#     def __init__(self, *args, recalculate_propensity_score=False, **kwargs):
#         # for the case when this is not invoked via Econml wrapper,
#         # need to merge init_args in
#         init_params = kwargs.pop("init_params", {})
#         kwargs = {
#             **kwargs,
#             **init_params,
#             "recalculate_propensity_score": recalculate_propensity_score,
#         }
#         super().__init__(*args, **kwargs)
#
#         # force fitting for the first time
#         self.recalculate_propensity_score = True
#         self._estimate_effect()
#         self.recalculate_propensity_score = recalculate_propensity_score
#
#     def effect(self, df: pd.DataFrame, **kwargs):
#
#         effect = super().effect(
#             df,
#             propensity_model=self.propensity_model,
#             **kwargs,
#         )
#
#         return effect

# # TODO: delete this once PR #486 is merged in dowhy
# def _estimate_effect(self):
#     self._refresh_propensity_score()
#     return super()._estimate_effect()


class DummyModel(DoWhyMethods):
    def __init__(
        self,
        propensity_modifiers: List[str],
        outcome_modifiers: List[str],
        effect_modifiers: List[str],
        treatment: str,
        outcome: str,
        control_value: Any,
    ):
        self.propensity_modifiers = propensity_modifiers
        self.outcome_modifiers = outcome_modifiers
        self.effect_modifiers = effect_modifiers
        self.treatment = treatment
        self.outcome = outcome

    def fit(
        self,
        df: pd.DataFrame,
    ):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean_, _, _ = Scorer.naive_ate(X[self.treatment], X[self.outcome])
        return np.ones(len(X)) * mean_ * (1 + 0.01 * np.random.normal(size=(len(X),)))


class PropensityScoreWeighter(DoWhyMethods):
    def __init__(
        self,
        propensity_modifiers: List[str],
        outcome_modifiers: List[str],
        effect_modifiers: List[str],
        treatment: str,
        outcome: str,
        propensity_model: Callable,
        min_ps_score: float = 0.05,
        control_value: Any = 0,
    ):
        self.outcome_modifiers = outcome_modifiers
        self.effect_modifiers = effect_modifiers
        self.propensity_modifiers = (
            propensity_modifiers
            if propensity_modifiers
            else self.effect_modifiers + self.outcome_modifiers
        )

        self.treatment = treatment
        self.outcome = outcome
        self.propensity_model = propensity_model
        self.min_ps_score = min_ps_score
        self._control_value = control_value

    def fit(self, df: pd.DataFrame):
        self._treatment_value = sorted(
            [v for v in df[self.treatment].unique() if v != self._control_value]
        )
        self.propensity_model.fit(df[self.propensity_modifiers], df[self.treatment])

    def predict(self, X: pd.DataFrame):
        p = self.propensity_model.predict_proba(X[self.propensity_modifiers])
        p = np.clip(p, self.min_ps_score, 1 - self.min_ps_score)
        est = np.ones((len(X), len(self._treatment_value)))
        for i, v in enumerate(self._treatment_value):
            base = X[self.treatment] == self._control_value
            treat = X[self.treatment] == v
            base_outcome = weighted_average(
                X.loc[base, self.outcome].values, 1 / p[base, 0]
            )
            treat_outcome = weighted_average(
                X.loc[treat, self.outcome].values, 1 / p[treat, 0]
            )
            est[:, i] *= treat_outcome - base_outcome
        return est


def weighted_average(x, w):
    return (x * w).sum() / w.sum()


class MultivaluePSW(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=PropensityScoreWeighter, **kwargs)
        self.identifier_method = "backdoor"


class Dummy(MultivaluePSW):
    identifier_method = "backdoor"
    """
    Apply a small random disturbance so the effect values are slightly different
    across units
    """

    def effect(self, df: pd.DataFrame, **kwargs):
        effect = super(MultivaluePSW, self).effect(df, **kwargs)
        return effect * (1 + 0.01 * np.random.normal(size=effect.shape))


class NaiveDummy(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=DummyModel, **kwargs)
        self.identifier_method = "backdoor"
