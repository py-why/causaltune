CausalTune Experiment Runner

This script runs CausalTune experiments with customizable parameters and generates plots of the results.

Usage:
python experiment_runner.py [arguments]

Arguments:
--metrics: Metrics to use for evaluation (space-separated list)
    Default: psw_frobenius_norm
    Options: psw_frobenius_norm, energy_distance, codec, frobenius_norm, policy_risk, prob_erupt, etc.

--datasets: Datasets to use (space-separated list, prefix each with Small or Large)
    Default: Small Linear_RCT
    Example: Small Linear_RCT Large NonLinear_KC Small Linear_IV

--n_runs: Number of runs for each dataset/metric combination
    Default: 1

--test_size: Fraction of data to use for testing
    Default: 0.33

--time_budget: Total time budget for optimization (in seconds)
    Default: None (no limit)

--components_time_budget: Time budget for component optimization (in seconds)
    Default: None
    Note: If time_budget is set, components_time_budget will be None, and vice versa.
          If neither is set, time_budget defaults to 600 seconds (10 minutes).

--identifier: Additional identifier for output directory
    Default: '' (empty string)

Output:
Results are saved in a directory named:
EXPERIMENT_RESULTS_[TIMESTAMP]_[IDENTIFIER]

Each run generates a .pkl file with the naming convention:
[METRIC]_run_[RUN_NUMBER]_[SIZE]_[DATASET].pkl

Plots:
The script automatically generates the following plots in the output directory:
1. CATE_grid.pdf: A grid of scatter plots showing estimated vs. true CATEs for each metric and dataset.
2. MSE_grid.pdf: A grid of scatter plots showing MSE vs. score for each estimator, metric, and dataset.
3. MSE_legend.pdf: A legend for the MSE grid plot, showing the estimator names.

Example usage:

1. Run experiments with multiple metrics on different datasets:
python experiment_runner.py --metrics psw_frobenius_norm energy_distance codec --datasets "Small Linear_RCT" "Large NonLinear_KC" "Small Linear_IV" --n_runs 3 --time_budget 3600 --identifier multi_metric_test

This runs experiments using psw_frobenius_norm, energy_distance, and codec metrics on Small Linear_RCT, Large NonLinear_KC, and Small Linear_IV datasets, with 3 runs each, a total time budget of 1 hour, and labels the output directory with 'multi_metric_test'.

2. Run a single metric on a single dataset with component time budget:
python experiment_runner.py --metrics prob_erupt --datasets "Small Linear_KCKP" --components_time_budget 300 --identifier single_run_test

This runs an experiment using the prob_erupt metric on the Small Linear_KCKP dataset, with a component time budget of 5 minutes, and labels the output directory with 'single_run_test'.

3. Run multiple metrics on a single dataset:
python experiment_runner.py --metrics psw_frobenius_norm policy_risk codec --datasets "Large NonLinear_RCT" --n_runs 2 --time_budget 1800 --identifier multi_metric_single_dataset

This runs experiments using psw_frobenius_norm, policy_risk, and codec metrics on the Large NonLinear_RCT dataset, with 2 runs each, a total time budget of 30 minutes, and labels the output directory with 'multi_metric_single_dataset'.

4. Run a single metric on multiple datasets with different sizes:
python experiment_runner.py --metrics energy_distance --datasets "Small Linear_KC" "Large Linear_KC" "Small NonLinear_KC" "Large NonLinear_KC" --n_runs 1 --components_time_budget 600 --identifier size_comparison

This runs an experiment using the energy_distance metric on Small and Large versions of Linear_KC and NonLinear_KC datasets, with a component time budget of 10 minutes, and labels the output directory with 'size_comparison'.

Note: The script will automatically generate plots for all experiments run, adapting to the number of metrics and datasets used. These plots provide a visual representation of the results, allowing for easy comparison of different estimators and datasets.