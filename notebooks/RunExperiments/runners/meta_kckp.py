import os

from experiment_runner import run_batch, get_estimator_list
from experiment_plots import generate_plots

prefix = "meta"
kind = "KCKP"
metrics = [
    "erupt",
    # "greedy_erupt",  # regular erupt was made probabilistic,
    # "policy_risk",  # NEW
    "qini",
    "auc",
    "psw_energy_distance",
    # "frobenius_norm",  # NEW
    "codec",  # NEW
    "bite",  # NEW
]

estimators = get_estimator_list(kind, include_patterns=["SLearner", "TLearner", "XLearner"])

use_ray = True
# run_batch(
#     identifier=f"Egor_test_{prefix}",
#     kind=kind,
#     metrics=metrics,
#     estimators=estimators,
#     dataset_path=os.path.realpath("../RunDatasets"),
#     use_ray=use_ray,
# )
# plot results
# upper_bounds = {"MSE": 1e2, "policy_risk": 0.2}
# lower_bounds = {"erupt": 0.06, "bite": 0.75}
out_dir = os.path.realpath(
    os.path.join(f"../EXPERIMENT_RESULTS_Egor_test_{prefix}/Large")
)  # , upper_bounds, lower_bounds)
generate_plots(os.path.join(out_dir, kind), metrics, prefix=prefix)  # , upper_bounds, lower_bounds)

print("yay!")
