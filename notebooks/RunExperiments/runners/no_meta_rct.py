import os

from experiment_runner import run_batch, get_estimator_list
from experiment_plots import generate_plots, generate_mse_plots

identifier = "no_meta"
kind = "RCT"
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

estimators = get_estimator_list(kind, exclude_patterns=["SLearner", "TLearner", "XLearner"])

if __name__ == "__main__":
    # use_ray = True
    # out_dir = run_batch(
    #     identifier,
    #     kind,
    #     metrics,
    #     estimators=estimators,
    #     dataset_path=os.path.realpath("../RunDatasets"),
    #     use_ray=use_ray,
    # )

    out_dir = os.path.realpath(os.path.join(f"../EXPERIMENT_RESULTS_{identifier}/Large"))
    # plot results
    # upper_bounds = {"MSE": 1e2, "policy_risk": 0.2}
    # lower_bounds = {"erupt": 0.06, "bite": 0.75}
    # generate_plots(os.path.join(out_dir, kind), metrics)  # , upper_bounds, lower_bounds)
    generate_mse_plots(os.path.join(out_dir, kind), metrics)
    print("yay!")
