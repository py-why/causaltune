from typing import List, Union

import pandas as pd
import numpy as np

from .wrapper import DoWhyMethods, DoWhyWrapper
from auto_causality.shap import shap_with_automl


def transformed_outcome(
    treatment: np.ndarray, outcome: np.ndarray, p: np.ndarray
) -> np.ndarray:
    return (treatment - p) * outcome / (p * (1 - p))


class TransformedOutcomeFitter(DoWhyMethods):
    def __init__(
        self,
        propensity_model,
        outcome_model,
        propensity_modifiers: List[str],
        outcome_modifiers: List[str],
        effect_modifiers: List[str],
        treatment: str,
        outcome: str,
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.propensity_modifiers = propensity_modifiers
        self.outcome_modifiers = outcome_modifiers
        self.effect_modifiers = effect_modifiers
        self.treatment = treatment
        self.outcome = outcome

    def fit(
        self,
        df: pd.DataFrame,
    ):
        self.propensity_model.fit(df[self.propensity_modifiers], df[self.treatment])
        p = self.propensity_model.predict_proba(df[self.propensity_modifiers])[:, 1]
        p = np.clip(p, 0.05, 0.95)
        ystar = transformed_outcome(
            df[self.treatment].values, df[self.outcome].values, p
        )
        # The causal graph assumption is that only effect_modifiers can affect the effect :)
        self.outcome_model.fit(X_train=df[self.effect_modifiers].values, y_train=ystar)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[self.effect_modifiers].values
        return self.outcome_model.predict(X)

    def shap_values(self, df: pd.DataFrame):
        return shap_with_automl(self.outcome_model, df)


class TransformedOutcome(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=TransformedOutcomeFitter, **kwargs)
        self.identifier_method = "backdoor"
