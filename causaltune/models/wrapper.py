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
        **kwargs,
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
        effect_modifier_names: List[str] = None,
        **kwargs,
    ):
        # dowhy 0.14 calls estimator.fit(data, effect_modifier_names=..., **fit_params)
        # and no longer passes treatment/outcome names -- they come from the estimand.
        self._data = data
        self._treatment_name = remove_list(self._target_estimand.treatment_variable)
        self._outcome_name = remove_list(self._target_estimand.outcome_variable)
        self._effect_modifier_names = (
            list(effect_modifier_names) if effect_modifier_names is not None else []
        )
        control_value = self.method_params.get("control_value", 0)

        self.estimator = self.estimator_class(
            treatment_name=self._treatment_name,
            outcome_name=self._outcome_name,
            # TODO: feed through the propensity modifiers where available
            propensity_modifiers=self._effect_modifier_names
            + self._observed_common_causes_names,
            outcome_modifiers=self._effect_modifier_names
            + self._observed_common_causes_names,
            effect_modifiers=self._effect_modifier_names,
            control_value=control_value,
            **(self.method_params.get("init_params", {})),
        )

        fit_params = kwargs if kwargs else self.method_params.get("fit_params", {})
        self.estimator.fit(data, **fit_params)
        return self

    def estimate_effect(
        self,
        data: pd.DataFrame = None,
        treatment_value=1,
        control_value=0,
        target_units="ate",
        confidence_intervals=False,
        **kwargs,
    ):
        if isinstance(target_units, pd.DataFrame):
            df = target_units
        elif data is not None:
            df = data
        else:
            df = self._data

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

        est = self.estimator.predict(df)

        estimate = CausalEstimate(
            data=df,
            treatment_name=self._treatment_name,
            outcome_name=self._outcome_name,
            estimate=np.mean(est, axis=0),
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            control_value=self._control_value,
            treatment_value=self._treatment_value,
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
        return self.effect(X)

    def shap_values(self, df: pd.DataFrame):
        return self.estimator.shap_values(df[self._effect_modifier_names])
