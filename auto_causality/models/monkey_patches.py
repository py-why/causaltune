from typing import Union

import numpy as np
import pandas as pd
from sklearn import linear_model
from flaml import AutoML as FLAMLAutoML

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)


# Let's engage in a bit of monkey patching as we wait for this to be merged into DoWhy
# #TODO: delete this as soon as PR #485 is merged into dowhy
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

# TODO: raise a dowhy PR for this
def effect(
    self,
    df: pd.DataFrame,
):
    self.causal_estimator.update_input(self._treatment_value, self._control_value, df)
    return np.ones(len(df)) * self.estimate_effect().value()


# TODO: delete this once PR #486 is merged in dowhy
def _refresh_propensity_score(self):
    if self.recalculate_propensity_score is True:
        if self.propensity_score_model is None:
            self.propensity_score_model = linear_model.LogisticRegression()

        self.propensity_score_model.fit(self._observed_common_causes, self._treatment)
        self._data[
            self.propensity_score_column
        ] = self.propensity_score_model.predict_proba(self._observed_common_causes)[
            :, 1
        ]
    else:
        # check if user provides the propensity score column
        if self.propensity_score_column not in self._data.columns:
            if self.propensity_score_model is None:
                raise ValueError(
                    f"""Propensity score column {self.propensity_score_column} does not exist,
nor does a propensity_model.
Please specify the column name that has your pre-computed propensity score,
or a model to compute it."""
                )
            else:
                self._data[
                    self.propensity_score_column
                ] = self.propensity_score_model.predict_proba(
                    self._observed_common_causes
                )[
                    :, 1
                ]
        else:
            self.logger.info(
                f"INFO: Using pre-computed propensity score in column {self.propensity_score_column}"
            )


# TODO: remove once https://github.com/py-why/dowhy/pull/547 is merged in dowhy
def _estimate_effect(self):
    self._refresh_propensity_score()

    # trim propensity score weights
    self._data[self.propensity_score_column] = np.minimum(
        self.max_ps_score, self._data[self.propensity_score_column]
    )
    self._data[self.propensity_score_column] = np.maximum(
        self.min_ps_score, self._data[self.propensity_score_column]
    )

    # ips ==> (isTreated(y)/ps(y)) + ((1-isTreated(y))/(1-ps(y)))
    # nips ==> ips / (sum of ips over all units)
    # icps ==> ps(y)/(1-ps(y)) / (sum of (ps(y)/(1-ps(y))) over all control units)
    # itps ==> ps(y)/(1-ps(y)) / (sum of (ps(y)/(1-ps(y))) over all treatment units)
    ipst_sum = sum(
        self._data[self._treatment_name[0]] / self._data[self.propensity_score_column]
    )
    ipsc_sum = sum(
        (1 - self._data[self._treatment_name[0]])
        / (1 - self._data[self.propensity_score_column])
    )
    num_units = len(self._data[self._treatment_name[0]])

    # Vanilla IPS estimator
    self._data["ips_weight"] = self._data[self._treatment_name[0]] / self._data[
        self.propensity_score_column
    ] + (1 - self._data[self._treatment_name[0]]) / (
        1 - self._data[self.propensity_score_column]
    )
    self._data["tips_weight"] = self._data[self._treatment_name[0]] + (
        1 - self._data[self._treatment_name[0]]
    ) * self._data[self.propensity_score_column] / (
        1 - self._data[self.propensity_score_column]
    )
    self._data["cips_weight"] = self._data[self._treatment_name[0]] * (
        1 - self._data[self.propensity_score_column]
    ) / self._data[self.propensity_score_column] + (
        1 - self._data[self._treatment_name[0]]
    )

    # The Hajek estimator (or the self-normalized estimator)
    self._data["ips_normalized_weight"] = (
        self._data[self._treatment_name[0]]
        / self._data[self.propensity_score_column]
        / ipst_sum
        + (1 - self._data[self._treatment_name[0]])
        / (1 - self._data[self.propensity_score_column])
        / ipsc_sum
    )
    ipst_for_att_sum = sum(self._data[self._treatment_name[0]])
    ipsc_for_att_sum = sum(
        (1 - self._data[self._treatment_name[0]])
        / (1 - self._data[self.propensity_score_column])
        * self._data[self.propensity_score_column]
    )
    self._data["tips_normalized_weight"] = (
        self._data[self._treatment_name[0]] / ipst_for_att_sum
        + (1 - self._data[self._treatment_name[0]])
        * self._data[self.propensity_score_column]
        / (1 - self._data[self.propensity_score_column])
        / ipsc_for_att_sum
    )
    ipst_for_atc_sum = sum(
        self._data[self._treatment_name[0]]
        / self._data[self.propensity_score_column]
        * (1 - self._data[self.propensity_score_column])
    )
    ipsc_for_atc_sum = sum((1 - self._data[self._treatment_name[0]]))
    self._data["cips_normalized_weight"] = (
        self._data[self._treatment_name[0]]
        * (1 - self._data[self.propensity_score_column])
        / self._data[self.propensity_score_column]
        / ipst_for_atc_sum
        + (1 - self._data[self._treatment_name[0]]) / ipsc_for_atc_sum
    )

    # Stabilized weights (from Robins, Hernan, Brumback (2000))
    # Paper: Marginal Structural Models and Causal Inference in Epidemiology
    p_treatment = sum(self._data[self._treatment_name[0]]) / num_units
    self._data["ips_stabilized_weight"] = self._data[
        self._treatment_name[0]
    ] / self._data[self.propensity_score_column] * p_treatment + (
        1 - self._data[self._treatment_name[0]]
    ) / (
        1 - self._data[self.propensity_score_column]
    ) * (
        1 - p_treatment
    )
    self._data["tips_stabilized_weight"] = self._data[
        self._treatment_name[0]
    ] * p_treatment + (1 - self._data[self._treatment_name[0]]) * self._data[
        self.propensity_score_column
    ] / (
        1 - self._data[self.propensity_score_column]
    ) * (
        1 - p_treatment
    )
    self._data["cips_stabilized_weight"] = self._data[self._treatment_name[0]] * (
        1 - self._data[self.propensity_score_column]
    ) / self._data[self.propensity_score_column] * p_treatment + (
        1 - self._data[self._treatment_name[0]]
    ) * (
        1 - p_treatment
    )

    if isinstance(self._target_units, pd.DataFrame) or self._target_units == "ate":
        weighting_scheme_name = self.weighting_scheme
    elif self._target_units == "att":
        weighting_scheme_name = "t" + self.weighting_scheme
    elif self._target_units == "atc":
        weighting_scheme_name = "c" + self.weighting_scheme
    else:
        raise ValueError(f"Target units value {self._target_units} not supported")

    # Calculating the effect
    self._data["d_y"] = (
        self._data[weighting_scheme_name]
        * self._data[self._treatment_name[0]]
        * self._data[self._outcome_name]
    )
    self._data["dbar_y"] = (
        self._data[weighting_scheme_name]
        * (1 - self._data[self._treatment_name[0]])
        * self._data[self._outcome_name]
    )
    sum_dy_weights = np.sum(
        self._data[self._treatment_name[0]] * self._data[weighting_scheme_name]
    )
    sum_dbary_weights = np.sum(
        (1 - self._data[self._treatment_name[0]]) * self._data[weighting_scheme_name]
    )
    # Subtracting the weighted means
    est = (
        self._data["d_y"].sum() / sum_dy_weights
        - self._data["dbar_y"].sum() / sum_dbary_weights
    )

    # TODO - how can we add additional information into the returned estimate?
    estimate = CausalEstimate(
        estimate=est,
        control_value=self._control_value,
        treatment_value=self._treatment_value,
        target_estimand=self._target_estimand,
        realized_estimand_expr=self.symbolic_estimator,
        propensity_scores=self._data[self.propensity_score_column],
    )
    return estimate


PropensityScoreWeightingEstimator.effect = effect
PropensityScoreWeightingEstimator._estimate_effect = _estimate_effect
PropensityScoreWeightingEstimator._refresh_propensity_score = _refresh_propensity_score


# this is needed for smooth calculation of Shapley values in DomainAdaptationLearner
class AutoML(FLAMLAutoML):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def _preprocess_y(self, y: Union[pd.DataFrame, pd.Series, np.ndarray]):
        if isinstance(y, pd.DataFrame) and len(y.columns) == 1:
            return y[y.columns[0]]
        else:
            return y

    def fit(self, *args, **kwargs):
        args = list(args)
        X_train = (
            args.pop(0)
            if len(args) > 0
            else (kwargs.pop("X_train") if "X_train" in kwargs else kwargs.pop("X"))
        )
        y_train = (
            args.pop(0)
            if len(args) > 0
            else (kwargs.pop("y_train") if "y_train" in kwargs else kwargs.pop("y"))
        )

        super().fit(X_train, self._preprocess_y(y_train), *args, **kwargs)
