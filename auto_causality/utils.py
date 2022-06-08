import math

import pandas as pd

from auto_causality.memoizer import MemoizingWrapper


def clean_config(params: dict):
    if "subforest_size" in params and "n_estimators" in params:
        params["n_estimators"] = params["subforest_size"] * math.ceil(
            params["n_estimators"] / params["subforest_size"]
        )

    if "min_samples_split" in params and params["min_samples_split"] > 1.5:
        params["min_samples_split"] = int(params["min_samples_split"])
    return params


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
#
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
#
#             super().__init__(*self.init_args, **self.init_kwargs)
#
#             print("calling AutoML fit method with ", used_kwargs)
#             super().fit(*args, **used_kwargs)
#
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
