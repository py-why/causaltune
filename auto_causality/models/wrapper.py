from typing import List, Any, Union, Callable

import pandas as pd
import numpy as np

from dowhy.causal_estimator import CausalEstimator, CausalEstimate


def remove_list(x: Any):
    if isinstance(x, str):
        return x
    else:
        return x[0]


class DoWhyWrapper(CausalEstimator):
    def __init__(
        self,
        data: pd.DataFrame,
        identified_estimand,
        treatment: str,
        outcome: str,
        effect_modifiers: List[str],
        inner_class: Callable,
        params: dict = None,
        control_value=0,
        treatment_value=1,
        test_significance=False,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        **kwargs
    ):
        self._treatment_name = remove_list(treatment)
        self._outcome_name = remove_list(outcome)
        self._effect_modifier_names = effect_modifiers

        params = {} if params is None else params
        # this is a hack to accomodate different DoWhy versions
        params = {**params, **kwargs}

        self.estimator = inner_class(
            treatment=self._treatment_name,
            outcome=self._outcome_name,
            propensity_modifiers=effect_modifiers,
            outcome_modifiers=effect_modifiers,
            **params.get("init_params", {})
        )

        self._data = data
        self._significance_test = test_significance
        self._effect_strength_eval = evaluate_effect_strength
        self._confidence_intervals = confidence_intervals
        self._control_value = control_value
        self._treatment_value = treatment_value
        self._target_estimand = identified_estimand

        self.method_params = params
        # TODO
        self.symbolic_estimator = ""
        self.effect_intervals = None

    def _estimate_effect(self):

        self.estimator.fit(self._data, **(self.method_params.get("fit_params", {})))

        est = self.estimator.predict(self._data)

        estimate = CausalEstimate(
            estimate=np.mean(est),
            control_value=self._control_value,
            treatment_value=self._treatment_value,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            cate_estimates=est,
            effect_intervals=self.effect_intervals,
            _estimator_object=self.estimator,
        )

        estimate.interpret = lambda: print("Not implemented yet...")

        return estimate

    def effect(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(data=X, columns=self._effect_modifier_names)
        return self.estimator.predict(X)

    def const_marginal_effect(self, X):
        return self.effect(self, X)
