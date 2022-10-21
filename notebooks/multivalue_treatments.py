import os
import urllib.request
import numpy as np
import pandas as pd
from auto_causality import AutoCausality

# Generic ML imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# EconML imports
from econml.dml import LinearDML, CausalForestDML
from econml.cate_interpreter import (
    SingleTreeCateInterpreter,
    SingleTreePolicyInterpreter,
)

import matplotlib.pyplot as plt

n_points = 10000
imp = {0: 0.0, 1: 2.0, 2: 1.0}

df = pd.DataFrame(
    {
        "X": np.random.normal(size=n_points),
        "W": np.random.normal(size=n_points),
        "T": np.random.choice(np.array(list(imp.keys())), size=n_points),
    }
)
df["Y"] = df["X"] + df["T"].apply(lambda x: imp[x])
df.head()

from sklearn.model_selection import train_test_split

# Data sample
train_data, test_data = train_test_split(df, train_size=0.9)
X_test = test_data[["X"]]
treatment = "T"
outcome = "Y"
common_causes = ["W"]
effect_modifiers = ["X"]
# Define estimator inputs
Y = train_data[outcome]  # outcome of interest
T = train_data[treatment]  # intervention, or treatment
X = train_data[effect_modifiers]  # features
W = train_data[common_causes]

est = LinearDML(discrete_treatment=True)
est.fit(Y, T, X=X, W=W)
# Get treatment effect and its confidence interval
test_data["est_effect"] = est.effect(X_test, T1=1)
# test_data['effect'] =  df['T'].apply(lambda x: imp[x])
test_data["est_effect"]

from dowhy import CausalModel

causal_model = CausalModel(
    data=train_data,
    treatment=treatment,
    outcome=outcome,
    common_causes=common_causes,
    effect_modifiers=effect_modifiers,
)
identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)

est_2 = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.LinearDML",
    control_value=0,
    treatment_value=[1, 2],
    target_units="ate",  # condition used for CATE
    confidence_intervals=False,
    method_params={
        "init_params": {"discrete_treatment": True},
        "fit_params": {},
    },
)

est_test = est_2.estimator.effect(X_test)

est_1 = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.LinearDML",
    control_value=0,
    treatment_value=1,
    target_units="ate",  # condition used for CATE
    confidence_intervals=False,
    method_params={
        "init_params": {"discrete_treatment": True},
        "fit_params": {},
    },
)

ac = AutoCausality(
    components_time_budget=10,
    estimator_list=[".LinearDML"],
    metric="energy_distance",
    metrics_to_report=["energy_distance"],
)
ac.fit(train_data, treatment, outcome, common_causes, effect_modifiers)

print("Yay!")
