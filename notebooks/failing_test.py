# Required libraries
import re
import numpy as np
import dowhy
from dowhy import CausalModel
import dowhy.datasets
from dowhy.utils.regression import create_polynomial_function

np.random.seed(101)
data = dowhy.datasets.partially_linear_dataset(
    beta=10,
    num_common_causes=7,
    num_unobserved_common_causes=1,
    strength_unobserved_confounding=10,
    num_samples=1000,
    num_treatments=1,
    stddev_treatment_noise=10,
    stddev_outcome_noise=5,
)

# Observed data
dropped_cols = ["W0"]
user_data = data["df"].drop(dropped_cols, axis=1)
# assumed graph
user_graph = data["gml_graph"]
for col in dropped_cols:
    user_graph = user_graph.replace('node[ id "{0}" label "{0}"]'.format(col), "")
    user_graph = re.sub(
        'edge\[ source "{}" target "[vy][0]*"\]'.format(col), "", user_graph
    )

model = CausalModel(
    data=user_data,
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    graph=user_graph,
    test_significance=None,
)

model._graph.get_effect_modifiers(model._treatment, model._outcome)

# Identify effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate effect
import econml
from sklearn.ensemble import GradientBoostingRegressor

linear_dml_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.LinearDML",
    method_params={
        "init_params": {
            "model_y": GradientBoostingRegressor(),
            "model_t": GradientBoostingRegressor(),
            "linear_first_stages": False,
        },
        "fit_params": {
            "cache_values": True,
        },
    },
)

print(linear_dml_estimate)
