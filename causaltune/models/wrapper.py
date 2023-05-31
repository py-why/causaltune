from typing import List, Any, Union, Callable

import pandas as pd
import numpy as np

from dowhy.causal_estimator import CausalEstimate
from causaltune.models.monkey_patches import CausalEstimator


def remove_list(x: Any):
    if isinstance(x, str):
        return x
    else:
        return x[0]


class DoWhyMethods:
    def effect(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.predict(x)

    def const_marginal_effect(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.predict(x)


class DoWhyWrapper(CausalEstimator):
    def __init__(
        self,
        identified_estimand,
        inner_class: Callable,
        # params: dict = None,
        test_significance=False,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        **kwargs
    ):
        self.estimator_class = inner_class
        self._observed_common_causes_names = (
            identified_estimand.get_backdoor_variables().copy()
        )

        # params = {} if params is None else params
        # # this is a hack to accomodate different DoWhy versions
        # params = {**params, **kwargs}

        self._significance_test = test_significance
        self._effect_strength_eval = evaluate_effect_strength
        self._confidence_intervals = confidence_intervals

        self._target_estimand = identified_estimand

        self.method_params = kwargs
        # TODO
        self.symbolic_estimator = ""
        self.effect_intervals = None

    def fit(
        self,
        data: pd.DataFrame,
        treatment_name: str,
        outcome_name: str,
        effect_modifier_names: List[str],
        control_value=0,
    ):
        self._data = data
        self._treatment_name = remove_list(treatment_name)
        self._outcome_name = remove_list(outcome_name)
        self._effect_modifier_names = effect_modifier_names

        self.estimator = self.estimator_class(
            treatment_name=self._treatment_name,
            outcome_name=self._outcome_name,
            # TODO: feed through the propensity modifiers where available
            propensity_modifiers=effect_modifier_names
            + self._observed_common_causes_names,
            outcome_modifiers=effect_modifier_names
            + self._observed_common_causes_names,
            effect_modifiers=effect_modifier_names,
            control_value=control_value,
            **(self.method_params.get("init_params", {})),
        )

        self.estimator.fit(data, **(self.method_params.get("fit_params", {})))

    def estimate_effect(
        self,
        treatment_value=1,
        control_value=0,
        target_units="ATE",
        confidence_intervals=False,
    ):
        if isinstance(target_units, pd.DataFrame):
            data = target_units
        else:
            data = self._data

        if confidence_intervals:
            raise NotImplementedError(
                "No confidence intervals available for this estimator yet"
            )

        if isinstance(target_units, str) and target_units.lower() != "ate":
            raise NotImplementedError(
                "Only 'ate' and dataframe target units supported at the moment"
            )

        self._control_value = control_value
        self._treatment_value = treatment_value

        est = self.estimator.predict(data)

        estimate = CausalEstimate(
            estimate=np.mean(est, axis=0),
            control_value=self._control_value,
            treatment_value=self._treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            cate_estimates=est,
            effect_intervals=self.effect_intervals,
        )

        estimate.add_estimator(self)
        estimate.interpret = lambda: print("Not implemented yet...")

        return estimate

    def effect(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(data=X, columns=self._effect_modifier_names)
        return self.estimator.predict(X)

    def const_marginal_effect(self, X):
        return self.effect(self, X)

    def shap_values(self, df: pd.DataFrame):
        return self.estimator.shap_values(df[self._effect_modifier_names])
