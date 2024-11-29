from typing import List, Callable, Any

import numpy as np
import pandas as pd

from causaltune.models.wrapper import DoWhyMethods, DoWhyWrapper
from causaltune.score.scoring import Scorer

from dowhy.causal_estimators.instrumental_variable_estimator import (
    InstrumentalVariableEstimator,
)


# # Let's engage in a bit of monkey patching as we wait for this to be merged into DoWhy
# # #TODO: delete this as soon as PR #485 is merged into dowhy
# def effect(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
#     # combining them in this way allows to override method_params from kwargs
#     extra_params = {**self.method_params, **kwargs}
#     new_estimator = type(self)(
#         data=df,
#         identified_estimand=self._target_estimand,
#         treatment=self._target_estimand.treatment_variable,
#         outcome=self._target_estimand.outcome_variable,
#         test_significance=False,
#         evaluate_effect_strength=False,
#         confidence_intervals=False,
#         target_units=self._target_units,
#         effect_modifiers=self._effect_modifier_names,
#         **extra_params,
#     )
#     scalar_effect = new_estimator.estimate_effect()
#     return np.ones(len(df)) * scalar_effect.value


class DummyModel(DoWhyMethods):
    def __init__(
        self,
        propensity_modifiers: List[str],
        outcome_modifiers: List[str],
        effect_modifiers: List[str],
        treatment_name: str,
        outcome_name: str,
        control_value: Any,
    ):
        self.propensity_modifiers = propensity_modifiers
        self.outcome_modifiers = outcome_modifiers
        self.effect_modifiers = effect_modifiers
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.mean = None

    def fit(
        self,
        df: pd.DataFrame,
    ):
        # ONLY WORKS FOR BINARY TREATMENT
        self.mean, _, _ = Scorer.naive_ate(
            df[self.treatment_name], df[self.outcome_name]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            np.ones(len(X)) * self.mean * (1 + 10e-5 * np.random.normal(size=(len(X),)))
        )


class PropensityScoreWeighter(DoWhyMethods):
    def __init__(
        self,
        propensity_modifiers: List[str],
        outcome_modifiers: List[str],
        effect_modifiers: List[str],
        treatment_name: str,
        outcome_name: str,
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

        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.propensity_model = propensity_model
        self.min_ps_score = min_ps_score
        self._control_value = control_value

    def fit(self, df: pd.DataFrame):
        self._treatment_value = sorted(
            [v for v in df[self.treatment_name].unique() if v != self._control_value]
        )
        self.propensity_model.fit(
            df[self.propensity_modifiers], df[self.treatment_name]
        )

    def predict(self, X: pd.DataFrame):
        p = self.propensity_model.predict_proba(X[self.propensity_modifiers])
        p = np.clip(p, self.min_ps_score, 1 - self.min_ps_score)
        est = np.ones((len(X), len(self._treatment_value)))
        for i, v in enumerate(self._treatment_value):
            base = X[self.treatment_name] == self._control_value
            treat = X[self.treatment_name] == v
            base_outcome = weighted_average(
                X.loc[base, self.outcome_name].values, 1 / p[base, 0]
            )
            treat_outcome = weighted_average(
                X.loc[treat, self.outcome_name].values, 1 / p[treat, 0]
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
        return effect * (1 + 10e-5 * np.random.normal(size=effect.shape))


class NaiveDummy(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=DummyModel, **kwargs)
        self.identifier_method = "backdoor"


class SimpleIV(InstrumentalVariableEstimator):
    """
    Based on Wald & 2SlS Estimator from dowhy's IV estimator
    """

    identifier_method = "iv"

    def effect(self, df: pd.DataFrame, **kwargs):
        scalar_effect = self.estimate_effect().value
        # Or randomized: (1 + 0.01 * np.random.normal(size=effect.shape))
        return np.ones(len(df)) * scalar_effect
