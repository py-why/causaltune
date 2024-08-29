from typing import Any, Union, Sequence, Optional
import math

import numpy as np
import pandas as pd

from causaltune.memoizer import MemoizingWrapper


def clean_config(params: dict):
    # TODO: move this to formal constraints in tune?
    if "subforest_size" in params and "n_estimators" in params:
        params["n_estimators"] = params["subforest_size"] * math.ceil(
            params["n_estimators"] / params["subforest_size"]
        )

    if "min_samples_split" in params and params["min_samples_split"] > 1.5:
        params["min_samples_split"] = int(params["min_samples_split"])
    return params


def treatment_values(treatment: pd.Series, control_value: Any):
    return sorted([t for t in treatment.unique() if t != control_value])


def treatment_is_multivalue(treatment: Union[int, str, Sequence]) -> bool:
    if isinstance(treatment, str) or len(treatment) == 1:
        return False
    else:
        return True


#
# def fit_params_wrapper(parent: type):
#     class FitParamsWrapper(parent):
#         def __init__(self, *args, fit_params=None, **kwargs):
#             self.init_args = args
#             self.init_kwargs = kwargs
#             if fit_params is not None:
#                 self.fit_params = fit_params
#             else:
#                 self.fit_params = {}

#         def fit(self, *args, **kwargs):
#             # we defer the initialization to the fit() method so we can memoize the fit
#             # using all the args from both init and fit
#             used_kwargs = {**kwargs, **self.fit_params}
#             to_hash = {
#                 "class": super().__class__.__name__,
#                 "fit_args": args,
#                 "fit_kwargs": used_kwargs,
#                 "init_args": self.init_args,
#                 "init_kwargs": self.init_kwargs,
#             }
#             test_fun(to_hash)

#             super().__init__(*self.init_args, **self.init_kwargs)

#             print("calling AutoML fit method with ", used_kwargs)
#             super().fit(*args, **used_kwargs)

#     return FitParamsWrapper


AutoMLWrapper = MemoizingWrapper


# AutoMLWrapper = fit_params_wrapper(AutoML)

# class AutoMLWrapper(AutoML):
#     def __init__(self, *args, fit_params=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.init_args = args
#         self.init_kwargs = kwargs
#         if fit_params is not None:
#             self.fit_params = fit_params
#         else:
#             self.fit_params = {}
#
#     def fit(self, *args, **kwargs):
#         # we defer the initialization to the fit() method so we can memoize the fit
#         # using all the args from both init and fit
#         used_kwargs = {**kwargs, **self.fit_params}
#         print("calling AutoML fit method with ", used_kwargs)
#         super().fit(*args, **used_kwargs)


def policy_from_estimator(est, df: pd.DataFrame):
    # must be done just like this so it also works for metalearners
    X_test = df[est.estimator._effect_modifier_names]
    return est.estimator.estimator.effect(X_test) > 0


def generate_psdmat(n_dims: int = 10) -> np.ndarray:
    """generates a symmetric, positive semidefinite matrix

    Args:
        n_dims (int, optional): number of dimensions. Defaults to 10.

    Returns:
        np.ndarray: psd matrix
    """
    A = np.random.rand(n_dims, n_dims)
    A = A @ A.T

    return A


def psw_joint_weights(a: np.ndarray, b: Optional[np.ndarray] = None):
    """Generate inverse joint propensity-score weights from two vectors

    Args:
        a (np.ndarray): propensity score vector of length n_A, array of shape (n_A,) or (n_A, 1)
        b (Optional[np.ndarray], optional): propensity score vector of length n_B,
            array of shape (n_B,) or (n_B, 1).
            If None, compares a with itself. Defaults to None.Defaults to None.

    Returns:
        (np.ndarray): inverse propensity score product matrix of shape (n_A, n_B)
    """
    if b is None:
        b = a

    assert a.ndim < 3 and b.ndim < 3

    if a.ndim == 1:
        a = np.expand_dims(a, axis=1)
    if b.ndim == 1:
        b = np.expand_dims(b, axis=1)

    assert a.shape[1] == b.shape[1]

    w = np.multiply(a, b.T)
    ipw = np.reciprocal(w)

    return ipw
