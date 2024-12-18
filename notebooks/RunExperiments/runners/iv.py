import os

from experiment_runner import run_batch, generate_plots

identifier = "Egor_test"
kind = "IV"
metrics = ["energy_distance", "frobenius_norm", "codec"]

out_dir = run_batch(identifier, kind, metrics, dataset_path=os.path.realpath("../RunDatasets"))
# plot results
# upper_bounds = {"MSE": 1e2, "policy_risk": 0.2}
# lower_bounds = {"erupt": 0.06, "bite": 0.75}
generate_plots(os.path.join(out_dir, kind))  # , upper_bounds, lower_bounds)
print("yay!")
