import os

from experiment_runner import run_batch, get_estimator_list

use_ray = True

for kind in ["KC"]:
    metrics = [
        "erupt",
        "qini",
        "auc",
        "psw_energy_distance",
        "codec",  # NEW
        "bite",  # NEW
    ]
    estimators = get_estimator_list(kind)
    for estimator in estimators:
        short_name = estimator.split(".")[-1]
        print(f"Running {kind} with {estimator}")
        out_dir = run_batch(
            identifier=f"Estimator_{short_name}",
            kind=kind,
            metrics=metrics,
            estimators=[estimator],
            dataset_path=os.path.realpath("../RunDatasets"),
            num_trials=25,
            use_ray=use_ray,
        )
