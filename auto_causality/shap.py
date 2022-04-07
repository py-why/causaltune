import pandas as pd
from dowhy.causal_model import CausalEstimate
import shap

final_model_map = {
    "DomainAdaptationLearner": lambda x: x.final_models[0],
    "TransformedOutcomeFitter": lambda x: x.outcome_model,
    "ForestDRLearner": lambda x: x.model_final,
}


def shap_values(estimate: CausalEstimate, df: pd.DataFrame):
    nice_df = df[estimate.estimator._effect_modifier_names]
    try:
        # try special handling for FLAML outcome models
        # this makes sure we use the fast tree-based SHAP estimator
        # SHAP doesn't recognize them on its own and falls back to the slow method
        est_name = estimate.estimator.estimator.__class__.__name__
        if est_name in final_model_map:
            final_model = final_model_map[est_name](estimate.estimator.estimator)
            if final_model.__class__.__name__ == "AutoML":
                inner_model = final_model.model.estimator
                explainer = shap.TreeExplainer(inner_model)
                shap_values = explainer.shap_values(nice_df)
                return shap_values

        raise Exception

    except Exception as e:
        print(e)
        try:
            # this will work on dowhy versions that include https://github.com/microsoft/dowhy/pull/374
            shap_values = estimate.estimator.shap_values(nice_df)
        except Exception as e2:
            print(e2)
            # fallback for earlier DoWhy versions
            shap_values = estimate.estimator.estimator.shap_values(nice_df.values)
            # strip out the nested dict that EconML returns
        return list(list(shap_values.values())[0].values())[0]
