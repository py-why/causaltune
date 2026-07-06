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
        # fall back to the slow algorithm, should work for anything.
        # Wrap predict in a lambda so shap's convert_to_model does not inspect
        # ``model.predict.__self__`` and try to null out the model's read-only
        # ``feature_names_in_`` property (FLAML models expose it read-only, which
        # shap >= 0.44 otherwise attempts to set, raising AttributeError).
        explainer = shap.KernelExplainer(lambda X: model.predict(X), nice_df)
        return explainer.shap_values(nice_df)
