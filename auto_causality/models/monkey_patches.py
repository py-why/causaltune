from typing import Union

import numpy as np
import pandas as pd
from sklearn import linear_model
from flaml import AutoML as FLAMLAutoML

from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)


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
                    f"""Propensity score column {self.propensity_score_column} does not exist, nor does a propensity_model.
                Please specify the column name that has your pre-computed propensity score, or a model to compute it."""
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


PropensityScoreWeightingEstimator.effect = effect
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
            args.pop(0) if len(args) > 0 else kwargs.pop("X_train", kwargs.pop("X"))
        )
        y_train = (
            args.pop(0) if len(args) > 0 else kwargs.pop("y_train", kwargs.pop("y"))
        )

        super().fit(X_train, self._preprocess_y(y_train), *args, **kwargs)
