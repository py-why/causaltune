from typing import List, Any
import math

from sklearn.preprocessing import RobustScaler
import pandas as pd

from auto_causality.memoizer import MemoizingWrapper


def clean_config(params: dict):
    if "subforest_size" in params and "n_estimators" in params:
        params["n_estimators"] = params["subforest_size"] * math.ceil(
            params["n_estimators"] / params["subforest_size"]
        )
    return params


def featurize(
    df: pd.DataFrame,
    features: List[str],
    exclude_cols: List[str],
    drop_first: bool = False,
    scale_floats: bool = False,
    prune_min_categories: int = 50,
    prune_thresh: float = 0.99,
) -> pd.DataFrame:
    # fill all the NaNs
    for col, t in zip(df.columns, df.dtypes):
        if pd.api.types.is_float_dtype(t):
            df[col] = df[col].fillna(0.0).astype("float32")
        elif pd.api.types.is_integer_dtype(t):
            df[col] = df[col].fillna(-1)
            df[col] = otherize_tail(df[col], -2, prune_thresh, prune_min_categories)
        else:
            df[col] = df[col].fillna("NA")
            df[col] = otherize_tail(
                df[col], "OTHER", prune_thresh, prune_min_categories
            ).astype("category")

    float_features = [f for f in features if pd.api.types.is_float_dtype(df.dtypes[f])]
    if scale_floats:
        float_df = pd.DataFrame(
            RobustScaler().fit_transform(df[float_features]), columns=float_features
        )
    else:
        float_df = df[float_features].reset_index(drop=True)

    # cast 0/1 int columns to float single-column dummies
    for col, t in zip(df.columns, df.dtypes):
        if pd.api.types.is_integer_dtype(t):
            if len(df[col].unique()) <= 2:
                df[col] = df[col].fillna(0.0).astype("float32")

    # for other categories, include first column dummy for easier interpretability
    cat_df = df.drop(columns=exclude_cols + float_features)
    if len(cat_df.columns):
        dummy_df = pd.get_dummies(cat_df, drop_first=drop_first).reset_index(drop=True)
    else:
        dummy_df = pd.DataFrame()

    out = pd.concat(
        [df[exclude_cols].reset_index(drop=True), float_df, dummy_df], axis=1
    )

    return out


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


def frequent_values(x: pd.Series, thresh: float = 0.99) -> set:
    # get the most frequent values, making up to the fraction thresh of total
    data = x.to_frame("value")
    data["dummy"] = True
    tmp = (
        data[["dummy", "value"]]
        .groupby("value", as_index=False)
        .count()
        .sort_values("dummy", ascending=False)
    )
    tmp["frac"] = tmp.dummy.cumsum() / tmp.dummy.sum()
    return set(tmp["value"][tmp.frac <= thresh].unique())


def otherize_tail(
    x: pd.Series, new_val: Any, thresh: float = 0.99, min_categories: int = 20
):
    uniques = x.unique()
    if len(uniques) < min_categories:
        return x
    else:
        x = x.copy()
        freq = frequent_values(x, thresh)
        x[~x.isin(freq)] = new_val
        return x
