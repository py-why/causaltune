import os

from experiment_runner import run_batch, get_estimator_list
from experiment_plots import generate_plots

prefix = "meta"

kind = "KC"
metrics = [
    "erupt",
    # "greedy_erupt",  # regular erupt was made probabilistic,
    # "policy_risk",  # NEW
    "psw_energy_distance",
    "bite",  # NEW
    # "frobenius_norm",  # NEW
    "qini",
    "auc",
    "codec",  # NEW
]

estimators = get_estimator_list(kind, include_patterns=["SLearner", "TLearner", "XLearner"])
ptt_estimators = ["lgbm", "lrl2", "kneighbor", "rf"]  # "histgb", "catboost",
#
# use_ray = True
# run_batch(
#     identifier=f"Egor_test_{prefix}",
#     kind,
#     metrics,
#     estimators=estimators,
#     propensity_automl_estimators=ptt_estimators,
#     components_time_budget=120,
#     dataset_path=os.path.realpath("../RunDatasets"),
#     use_ray=use_ray,
# )
out_dir = os.path.realpath(os.path.join(f"../EXPERIMENT_RESULTS_Egor_test_{prefix}/Large"))
# plot results
# upper_bounds = {"MSE": 1e2, "policy_risk": 0.2}
# lower_bounds = {"erupt": 0.06, "bite": 0.75}
generate_plots(os.path.join(out_dir, kind), metrics, prefix=prefix)  # , upper_bounds, lower_bounds)
print("yay!")
