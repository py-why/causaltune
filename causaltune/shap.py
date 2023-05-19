import pandas as pd
from dowhy.causal_model import CausalEstimate
import shap


def shap_values(estimate: CausalEstimate, df: pd.DataFrame):
    try:
        nice_df = df[estimate.estimator._effect_modifier_names]
    except AttributeError:
        # for EconML estimators
        nice_df = df[estimate.estimator._input_names["feature_names"]]
    try:
        # this will work on dowhy versions that include https://github.com/microsoft/dowhy/pull/374
        sv = estimate.estimator.shap_values(nice_df)
    except Exception as e2:
        print(e2)
        # fallback for earlier DoWhy versions
        sv = estimate.estimator.estimator.shap_values(nice_df.values)

    try:
        # try strip out the nested dict that EconML returns
        return list(list(sv.values())[0].values())[0]
    except AttributeError:
        # if it's one of causaltune models, just return as is
        return sv


def shap_with_automl(model, nice_df: pd.DataFrame):
    # is it a FLAML instance?
    if model.__class__.__name__ == "AutoML":
        return shap_with_automl(model.model.estimator, nice_df)

    # try the fast algorithm for tree models
    try:
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(nice_df)
    except Exception:
        # fall back to the slow algorithm, should work for anything
        # for some reason the generic shap.Explainer doesn't seem to do this
        explainer = shap.KernelExplainer(model.predict, nice_df)
        return explainer.shap_values(nice_df)
