from functools import partial
from typing import Union

import numpy as np

import pandas as pd
from flaml import AutoML as FLAMLAutoML

from dowhy.causal_estimator import CausalEstimator

from causaltune.utils import is_sequence


def effect_stderr(self, df: pd.DataFrame, *args, **kwargs):
    """
    Inference (uncertainty) results produced by the underlying EconML estimator.

    dowhy 0.14 provides ``apply_multitreatment`` natively on its EconML adapter;
    unlike causaltune's old copy it does *not* select the effect-modifier columns
    itself (dowhy's ``effect``/``effect_inference`` select them before calling it),
    so we select ``df[self._effect_modifier_names]`` here to keep the result
    identical.

    :param df: Features of the units to evaluate
    :param args: passed through to the underlying estimator
    :param kwargs: passed through to the underlying estimator
    """

    def effect_inference_fun(filtered_df, T0, T1, *args, **kwargs):
        return self.estimator.effect_inference(
            filtered_df, T0=T0, T1=T1, *args, **kwargs
        ).stderr

    Xdf = df[self._effect_modifier_names] if df is not None else None
    return self.apply_multitreatment(Xdf, effect_inference_fun, *args, **kwargs)


def effect_tt(self, df: pd.DataFrame, treatment_value=None, *args, **kwargs):
    """
    Effect of treatment on the treated: for each unit, the estimated effect of the
    treatment value that was actually applied to it.

    Kept as a universal fallback on ``CausalEstimator`` (dowhy 0.14's EconML adapter
    shadows it with its own native ``effect_tt``); this covers causaltune's custom
    ``DoWhyWrapper`` estimators. The signature matches dowhy's native
    ``effect_tt(df, treatment_value)`` so call sites are uniform; ``treatment_value``
    defaults to ``self._treatment_value``.

    @param df: unit features and treatment values
    @param treatment_value: the treatment value(s) to consider; defaults to
        ``self._treatment_value``
    @return: pd.Series of len(df) with effects of the actual treatment applied
    """
    if treatment_value is None:
        treatment_value = self._treatment_value

    if not (is_sequence(treatment_value) and not isinstance(treatment_value, str)):
        treatment_value = [treatment_value]

    eff = np.reshape(
        np.asarray(self.effect(df, *args, **kwargs)), (len(df), len(treatment_value))
    )

    out = np.zeros(len(df))

    if is_sequence(self._treatment_name) and not isinstance(self._treatment_name, str):
        treatment_name = self._treatment_name[0]
    else:
        treatment_name = self._treatment_name

    for c, col in enumerate(treatment_value):
        out[df[treatment_name] == col] = eff[df[treatment_name] == col, c]
    return pd.Series(data=out, index=df.index)


CausalEstimator.effect_tt = effect_tt


# this is needed for smooth calculation of Shapley values in DomainAdaptationLearner
class AutoML(FLAMLAutoML):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @property
    def predict_proba(self):
        # econml 0.16 treats any model exposing predict_proba as a classifier and
        # rejects it as a regression nuisance (e.g. DRLearner.model_regression) when
        # discrete_outcome=False. FLAML defines predict_proba regardless of task, so
        # expose it only for classification models (propensity), not regression
        # (outcome). Raising AttributeError makes hasattr(model, "predict_proba")
        # return False for regression models.
        settings = getattr(self, "_settings", None)
        task = settings.get("task") if isinstance(settings, dict) else None
        if task is not None and "regression" in str(task):
            raise AttributeError("predict_proba")
        return partial(FLAMLAutoML.predict_proba, self)

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
