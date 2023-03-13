from typing import Union, Callable

import numpy as np
from numpy.distutils.misc_util import is_sequence

import pandas as pd
from flaml import AutoML as FLAMLAutoML

from dowhy.causal_estimator import CausalEstimator


def effect_stderr(self, df: pd.DataFrame, *args, **kwargs):
    """
    Inference (uncertainty) results produced by the underlying EconML estimator
    :param df: Features of the units to evaluate
    :param args: passed through to the underlying estimator
    :param kwargs: passed through to the underlying estimator
    """

    def effect_inference_fun(filtered_df, T0, T1, *args, **kwargs):
        return self.estimator.effect_inference(
            filtered_df, T0=T0, T1=T1, *args, **kwargs
        ).stderr

    return self.apply_multitreatment(df, effect_inference_fun, *args, **kwargs)


def apply_multitreatment(self, df: pd.DataFrame, fun: Callable, *args, **kwargs):
    ests = []
    assert not isinstance(self._treatment_value, str)
    assert is_sequence(self._treatment_value)

    if df is None:
        filtered_df = None
    else:
        filtered_df = df[self._effect_modifier_names].values

    for tv in self._treatment_value:
        ests.append(
            fun(
                filtered_df,
                T0=self._control_value,
                T1=tv,
                *args,
                **kwargs,
            )
        )

    if isinstance(ests[0], np.ndarray):
        est = np.stack(ests, axis=1)
    else:
        est = ests
    return est


def effect_tt(self, df: pd.DataFrame, *args, **kwargs):
    """
    Effect of treatment on the treated
    @param df: unit features and treatment values
    @param args: passed through to effect estimator
    @param kwargs: passed through to effect estimator
    @return: np.array of len(df) with effects of the actual treatment applied
    """

    eff = self.effect(df, *args, **kwargs).reshape(
        (len(df), len(self._treatment_value))
    )

    out = np.zeros(len(df))
    if is_sequence(self._treatment_value) and not isinstance(
        self._treatment_value, str
    ):
        treatment_value = self._treatment_value
    else:
        treatment_value = [self._treatment_value]

    if is_sequence(self._treatment_name) and not isinstance(self._treatment_name, str):
        treatment_name = self._treatment_name[0]
    else:
        treatment_name = self._treatment_name

    eff = np.reshape(eff, (len(df), len(treatment_value)))

    for c, col in enumerate(treatment_value):
        out[df[treatment_name] == col] = eff[df[treatment_name] == col, c]
    return pd.Series(data=out, index=df.index)


CausalEstimator.effect_tt = effect_tt


def effect(
    self,
    df: pd.DataFrame,
):
    self.update_input(self._treatment_value, self._control_value, df)
    return np.ones(len(df)) * self.estimate_effect().value


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
