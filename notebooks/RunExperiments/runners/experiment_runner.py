import argparse
import copy
import glob
import os
import pickle
import sys
import warnings
from datetime import datetime
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure CausalTune is in the Python path
root_path = os.path.realpath("../../../../..")  # noqa: E402
sys.path.append(os.path.join(root_path, "causaltune"))  # noqa: E402

# Import CausalTune and other custom modules after setting up the path
from causaltune import CausalTune  # noqa: E402
from causaltune.datasets import load_dataset  # noqa: E402
from causaltune.models.passthrough import passthrough_model  # noqa: E402
from causaltune.search.params import SimpleParamService  # noqa: E402
from causaltune.score.scoring import (
    metrics_to_minimize,  # noqa: E402
    supported_metrics,  # noqa: E402
)

# Configure warnings
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CausalTune experiments")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psw_frobenius_norm"],
        help="Metrics to use for evaluation",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Small Linear_RCT"],
        help="Datasets to use (format: Size Name, e.g., Small Linear_RCT)",
    )
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Maximum number of iterations"
    )

    parser.add_argument("--outcome_model", type=str, default="nested", help="Outcome model type")
    parser.add_argument(
        "--timestamp_in_dirname",
        type=bool,
        default="False",
        help="Include timestampl in out_dir name?",
    )

    parser.add_argument("--test_size", type=float, default=0.33, help="Test set size")
    parser.add_argument(
        "--time_budget", type=int, default=None, help="Time budget for optimization"
    )
    parser.add_argument(
        "--components_time_budget",
        type=int,
        default=None,
        help="Time budget for component optimization",
    )
    parser.add_argument(
        "--identifier", default="", help="Additional identifier for output directory"
    )
    return parser.parse_args()


def get_estimator_list(dataset_name):
    if "IV" in dataset_name:
        problem = "iv"
    else:
        problem = "backdoor"

    cfg = SimpleParamService(
        n_jobs=-1,
        include_experimental=False,
        multivalue=False,
    )
    estimator_list = cfg.estimator_names_from_patterns(problem, "all", 1001)
    return [est for est in estimator_list if "Dummy" not in est]


def run_experiment(args, dataset_path: str, use_ray: bool = False):
    # Process datasets
    data_sets = {}
    for dataset in args.datasets:
        parts = dataset.split()
        if len(parts) < 2:
            raise ValueError(
                f"Invalid dataset format: {dataset}. Expected format: Size Name (e.g., Small Linear_RCT)"
            )
        size = parts[0]
        name = " ".join(parts[1:])
        file_path = f"{dataset_path}/{size}/{name}.pkl"
        data_sets[f"{size} {name}"] = load_dataset(file_path)

    if args.timestamp_in_dirname:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"EXPERIMENT_RESULTS_{timestamp}_{args.identifier}"
    else:
        out_dir = f"EXPERIMENT_RESULTS_{args.identifier}"

    os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.realpath(os.path.join(out_dir, size))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loaded datasets: {list(data_sets.keys())}")

    # Set time budgets properly
    if args.time_budget is not None and args.components_time_budget is not None:
        raise ValueError("Please specify either time_budget or components_time_budget, not both.")
    elif args.time_budget is None and args.components_time_budget is None:
        args.components_time_budget = 30  # Set default components budget

    # If only time_budget is specified, derive components_time_budget from it
    if args.time_budget is not None:
        args.components_time_budget = max(30, args.time_budget / 4)  # Ensure minimum budget
        args.time_budget = None  # Use only components_time_budget

    for dataset_name, cd in data_sets.items():
        # Extract case while preserving original string checking logic
        if "KCKP" in dataset_name:
            case = "KCKP"
        elif "KC" in dataset_name:
            case = "KC"
        elif "IV" in dataset_name:
            case = "IV"
        else:
            case = "RCT"

        os.makedirs(f"{out_dir}/{case}", exist_ok=True)

        for i_run in range(1, args.n_runs + 1):
            cd_i = copy.deepcopy(cd)
            train_df, test_df = train_test_split(cd_i.data, test_size=args.test_size)
            test_df = test_df.reset_index(drop=True)
            cd_i.data = train_df

            for metric in args.metrics:
                if metric == "ate":  # this is not something to optimize
                    continue

                print(f"Optimizing {metric} for {dataset_name} (run {i_run})")
                try:
                    fn = make_filename(metric, dataset_name, i_run)
                    out_fn = os.path.join(out_dir, case, fn)
                    if os.path.isfile(out_fn):
                        print(f"File {out_fn} exists, skipping...")
                        continue

                    # Set propensity model using string checking like original version
                    if "KCKP" in dataset_name:
                        print(f"Using passthrough propensity model for {dataset_name}")
                        propensity_model = passthrough_model(
                            cd_i.propensity_modifiers, include_control=False
                        )
                    elif "KC" in dataset_name:
                        print(f"Using auto propensity model for {dataset_name}")
                        propensity_model = "auto"
                    else:
                        print(f"Using dummy propensity model for {dataset_name}")
                        propensity_model = "dummy"

                    ct = CausalTune(
                        metric=metric,
                        estimator_list=get_estimator_list(dataset_name),
                        num_samples=args.num_samples,
                        components_time_budget=args.components_time_budget,  # Use this instead
                        verbose=1,
                        components_verbose=1,
                        store_all_estimators=True,
                        propensity_model=propensity_model,
                        outcome_model=args.outcome_model,
                        use_ray=use_ray,
                    )

                    ct.fit(
                        data=cd_i,
                        treatment="treatment",
                        outcome="outcome",
                    )

                    # Compute scores and save results
                    results = compute_scores(ct, metric, test_df)

                    with open(out_fn, "wb") as f:
                        pickle.dump(results, f)
                except Exception as e:
                    print(f"Error processing {dataset_name}_{metric}_{i_run}: {e}")

    return out_dir


def compute_scores(ct, metric, test_df):
    datasets = {"train": ct.train_df, "validation": ct.test_df, "test": test_df}
    estimator_scores = {est: [] for est in ct.scores.keys() if "NewDummy" not in est}

    all_scores = []
    for trial in ct.results.trials:
        try:
            estimator_name = trial.last_result["estimator_name"]
            if "estimator" in trial.last_result and trial.last_result["estimator"]:
                estimator = trial.last_result["estimator"]
                scores = {}
                for ds_name, df in datasets.items():
                    scores[ds_name] = {}
                    est_scores = ct.scorer.make_scores(
                        estimator,
                        df,
                        metrics_to_report=ct.metrics_to_report,
                    )
                    est_scores["estimator_name"] = estimator_name

                    scores[ds_name]["CATE_estimate"] = np.squeeze(estimator.estimator.effect(df))
                    scores[ds_name]["CATE_groundtruth"] = np.squeeze(df["true_effect"])
                    est_scores["MSE"] = np.mean(
                        (scores[ds_name]["CATE_estimate"] - scores[ds_name]["CATE_groundtruth"])
                        ** 2
                    )
                    scores[ds_name]["scores"] = est_scores
                scores["optimization_score"] = trial.last_result.get("optimization_score")
                estimator_scores[estimator_name].append(copy.deepcopy(scores))
            # Will use this in the nex
            all_scores.append(scores)
        except Exception as e:
            print(f"Error processing trial: {e}")

    for k in estimator_scores.keys():
        estimator_scores[k] = sorted(
            estimator_scores[k],
            key=lambda x: x["validation"]["scores"][metric],
            reverse=metric not in metrics_to_minimize(),
        )

    # Debugging: Log final result structure
    print(f"Returning scores for metric {metric}: Best estimator: {ct.best_estimator}")

    return {
        "best_estimator": ct.best_estimator,
        "best_config": ct.best_config,
        "best_score": ct.best_score,
        "optimised_metric": metric,
        "scores_per_estimator": estimator_scores,
        "all_scores": all_scores,
    }


def extract_metrics_datasets(out_dir: str):
    metrics = set()
    datasets = set()

    for file in glob.glob(f"{out_dir}/*.pkl"):
        parts = os.path.basename(file).split("-")
        metrics.add(parts[0])
        datasets.add(parts[-1].replace(".pkl", "").replace("_", " "))

    return sorted(list(metrics)), sorted(list(datasets))


def make_filename(metric, dataset, i_run):
    return f"{metric}-run-{i_run}-{dataset.replace(' ', '_')}.pkl"


def get_all_test_scores(out_dir, dataset_name):
    size, ds_type, case = dataset_name.split(" ")
    all_scores = []
    for file in glob.glob(f"{out_dir}/*_{ds_type}_{case}.pkl"):
        with open(file, "rb") as f:
            results = pickle.load(f)
            for x in results["all_scores"]:
                all_scores.append(
                    {k: v for k, v in x["test"]["scores"].items() if k not in ["values"]}
                )
    out = pd.DataFrame(all_scores)
    return out


def generate_plots(
    out_dir: str,
    log_scale: Union[List[str], None] = None,
    upper_bounds: Union[dict, None] = None,
    lower_bounds: Union[dict, None] = None,
    font_size=0,
):
    if log_scale is None:
        log_scale = ["energy_distance", "psw_energy_distance", "frobenius_norm"]
    if upper_bounds is None:
        upper_bounds = {}  # Use an empty dictionary if None
    if lower_bounds is None:
        lower_bounds = {}  # Use an empty dictionary if None

    metrics, datasets = extract_metrics_datasets(out_dir)
    # Remove 'ate' from metrics
    metrics = [m for m in metrics if m.lower() != "ate"]

    metric_names = {
        "psw_frobenius_norm": "PSW\nFrobenius\nNorm",
        "frobenius_norm": "Frobenius\nNorm",
        "erupt": "ERUPT",
        "codec": "CODEC",
        "auc": "AUC",
        "qini": "Qini",
        "bite": "BITE",
        "policy_risk": "Policy\nRisk",
        "energy_distance": "Energy\nDistance",
        "psw_energy_distance": "PSW\nEnergy\nDistance",
        "norm_erupt": "Normalized\nERUPT",
    }

    colors = (
        [matplotlib.colors.CSS4_COLORS["black"]]
        + list(matplotlib.colors.TABLEAU_COLORS)
        + [
            matplotlib.colors.CSS4_COLORS["lime"],
            matplotlib.colors.CSS4_COLORS["yellow"],
            matplotlib.colors.CSS4_COLORS["pink"],
        ]
    )
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "*", "h", "X", "|", "_", "8"]

    # Determine the problem type from the dataset name
    problem = "iv" if any("IV" in dataset for dataset in datasets) else "backdoor"

    def plot_grid(title):
        # Use determined problem type instead of hardcoding "backdoor"
        all_metrics = [
            m
            for m in supported_metrics(problem, False, False)
            if m.lower() != "ate" and m.lower() != "norm_erupt"
        ]

        fig, axs = plt.subplots(
            len(all_metrics), len(datasets), figsize=(20, 5 * len(all_metrics)), dpi=300
        )

        if len(all_metrics) == 1 and len(datasets) == 1:
            axs = np.array([[axs]])
        elif len(all_metrics) == 1 or len(datasets) == 1:
            axs = axs.reshape(-1, 1) if len(datasets) == 1 else axs.reshape(1, -1)

        # For multiple metrics in args.metrics, use the first one that has a results file
        results_files = {}
        for dataset in datasets:
            for metric in args.metrics:
                filename = make_filename(metric, dataset, 1)
                filepath = os.path.join(out_dir, filename)
                if os.path.exists(filepath):
                    results_files[dataset] = filepath
                    break
            if dataset not in results_files:
                print(f"No results file found for dataset {dataset}")

        for j, dataset in enumerate(datasets):
            if dataset not in results_files:
                continue

            with open(results_files[dataset], "rb") as f:
                results = pickle.load(f)

            print(f"Loading results for Dataset: {dataset}")

            for i, metric in enumerate(all_metrics):
                ax = axs[i, j]

                try:
                    # Find best estimator for this metric
                    best_estimator = None
                    best_score = (
                        float("inf")
                        if metric in metrics_to_minimize()
                        else float("-inf")
                    )
                    estimator_name = None

                    for score in results["all_scores"]:
                        if "test" in score and metric in score["test"]["scores"]:
                            current_score = score["test"]["scores"][metric]
                            if metric in metrics_to_minimize():
                                if current_score < best_score:
                                    best_score = current_score
                                    best_estimator = score
                                    estimator_name = score["test"]["scores"][
                                        "estimator_name"
                                    ]
                            else:
                                if current_score > best_score:
                                    best_score = current_score
                                    best_estimator = score
                                    estimator_name = score["test"]["scores"][
                                        "estimator_name"
                                    ]

                    if best_estimator:
                        CATE_gt = np.array(
                            best_estimator["test"]["CATE_groundtruth"]
                        ).flatten()
                        CATE_est = np.array(
                            best_estimator["test"]["CATE_estimate"]
                        ).flatten()

                        # Plotting
                        ax.scatter(CATE_gt, CATE_est, s=40, alpha=0.5)
                        ax.plot(
                            [min(CATE_gt), max(CATE_gt)],
                            [min(CATE_gt), max(CATE_gt)],
                            "k-",
                            linewidth=1.0,
                        )

                        # Calculate correlation coefficient
                        corr = np.corrcoef(CATE_gt, CATE_est)[0, 1]

                        # Add correlation
                        ax.text(
                            0.05,
                            0.95,
                            f"Corr: {corr:.2f}",
                            transform=ax.transAxes,
                            verticalalignment="top",
                            fontsize=font_size + 12,
                            fontweight="bold",
                        )

                        # Add estimator name at bottom center
                        if estimator_name:
                            estimator_base = estimator_name.split(".")[-1]
                            ax.text(
                                0.5,
                                0.02,
                                estimator_base,
                                transform=ax.transAxes,
                                horizontalalignment="center",
                                color="blue",
                                fontsize=font_size + 10,
                            )

                except Exception as e:
                    print(
                        f"Error processing metric {metric} for dataset {dataset}: {e}"
                    )
                    ax.text(
                        0.5,
                        0.5,
                        "Error processing data",
                        ha="center",
                        va="center",
                        fontsize=font_size + 12,
                    )

                if j == 0:
                    # Create tight layout for ylabel
                    ax.set_ylabel(
                        metric_names.get(metric, metric),
                        fontsize=font_size + 12,
                        fontweight="bold",
                        labelpad=5,  # Reduce padding between label and plot
                    )
                if i == 0:
                    ax.set_title(
                        dataset, fontsize=font_size + 14, fontweight="bold", pad=15
                    )
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(
            f"Estimated CATEs vs. True CATEs: {title}",
            fontsize=font_size + 18,
            fontweight="bold",
        )
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0.1, 0, 1, 0.96], h_pad=1.0, w_pad=0.5)
        plt.savefig(os.path.join(out_dir, "CATE_grid.pdf"), format="pdf", bbox_inches="tight")
        plt.savefig(os.path.join(out_dir, "CATE_grid.png"), format="png", bbox_inches="tight")
        plt.close()

    def plot_mse_grid(title):
        df = get_all_test_scores(out_dir, datasets[0])
        est_names = sorted(df["estimator_name"].unique())

        # Problem type already determined at top level
        all_metrics = [c for c in df.columns if c in supported_metrics(problem, False, False)and c.lower() != "ate"
        ]

        fig, axs = plt.subplots(
            len(all_metrics), len(datasets), figsize=(20, 5 * len(all_metrics)), dpi=300
        )

        # Handle single plot cases
        if len(all_metrics) == 1 and len(datasets) == 1:
            axs = np.array([[axs]])
        elif len(all_metrics) == 1 or len(datasets) == 1:
            axs = axs.reshape(-1, 1) if len(datasets) == 1 else axs.reshape(1, -1)

        legend_elements = []
        for j, dataset in enumerate(datasets):
            df = get_all_test_scores(out_dir, dataset)
            # Apply bounds filtering
            for m, value in upper_bounds.items():
                if m in df.columns:
                    df = df[df[m] < value].copy()
            for m, value in lower_bounds.items():
                if m in df.columns:
                    df = df[df[m] > value].copy()

            for i, metric in enumerate(all_metrics):
                ax = axs[i, j]
                this_df = df[["estimator_name", metric, "MSE"]].dropna()
                this_df = this_df[~np.isinf(this_df[metric].values)]

                if len(this_df):
                    for idx, est_name in enumerate(est_names):
                        df_slice = this_df[this_df["estimator_name"] == est_name]
                        if "Dummy" not in est_name and len(df_slice):
                            marker = markers[idx % len(markers)]
                            ax.scatter(
                                df_slice["MSE"],
                                df_slice[metric],
                                color=colors[idx],
                                s=50,
                                marker=marker,
                                linewidths=0.5,
                            )
                            if metric not in metrics_to_minimize():
                                ax.invert_yaxis()

                            trimmed_est_name = est_name.split(".")[-1]
                            if i == 0 and j == 0:
                                legend_elements.append(
                                    plt.Line2D(
                                        [0],
                                        [0],
                                        color=colors[idx],
                                        marker=marker,
                                        label=trimmed_est_name,
                                        linestyle="None",
                                        markersize=6,
                                    )
                                )

                    ax.set_xscale("log")
                    if metric in log_scale:
                        ax.set_yscale("log")
                    ax.grid(True)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        fontsize=font_size + 12,
                    )

                if j == 0:
                    # Match ylabel style with plot_grid
                    ax.set_ylabel(
                        metric_names.get(metric, metric),
                        fontsize=font_size + 12,
                        fontweight="bold",
                        labelpad=5,
                    )
                if i == 0:
                    ax.set_title(
                        dataset, fontsize=font_size + 14, fontweight="bold", pad=15
                    )

        plt.suptitle(
            f"MSE vs. Scores: {title}",
            fontsize=font_size + 18,
            fontweight="bold",
        )

        # Match spacing style with plot_grid
        plt.tight_layout(rect=[0.1, 0, 1, 0.96], h_pad=1.0, w_pad=0.5)
        plt.savefig(os.path.join(out_dir, "MSE_grid.pdf"), format="pdf", bbox_inches="tight")
        plt.savefig(os.path.join(out_dir, "MSE_grid.png"), format="png", bbox_inches="tight")
        plt.close()

        # # Create separate legend
        # fig_legend, ax_legend = plt.subplots(figsize=(6, 6))
        # ax_legend.legend(handles=legend_elements, loc="center", fontsize=10)
        # ax_legend.axis("off")
        # plt.savefig(os.path.join(out_dir, "MSE_legend.pdf"), format="pdf", bbox_inches="tight")
        # plt.savefig(os.path.join(out_dir, "MSE_legend.png"), format="png", bbox_inches="tight")
        # plt.close()

    # Generate plots
    plot_grid("Experiment Results")
    plot_mse_grid("Experiment Results")


def run_batch(
    identifier: str,
    kind: str,
    metrics: List[str],
    dataset_path: str,
):
    args = parse_arguments()
    args.identifier = identifier
    args.metrics = metrics
    # run_experiment assumes we don't mix large and small datasets in the same call
    args.datasets = [f"Large Linear_{kind}", f"Large NonLinear_{kind}"]
    args.num_samples = 100
    args.timestamp_in_dirname = False
    args.outcome_model = "auto"  # or use "nested" for the old-style nested model

    # os.environ["RAY_ADDRESS"] = "ray://127.0.0.1:8265"

    use_ray = True
    if use_ray:
        import ray

        # Assuming we port-mapped already by running ray dashboard
        ray.init(
            "ray://localhost:10001", runtime_env={"pip": ["causaltune", "catboost"]}
        )  # "34.82.184.148:6379"
    out_dir = run_experiment(args, dataset_path=dataset_path, use_ray=use_ray)
    return out_dir


if __name__ == "__main__":

    args = parse_arguments()
    args.identifier = "Egor_test"
    args.metrics = supported_metrics("backdoor", False, False)
    # run_experiment assumes we don't mix large and small datasets in the same call
    args.datasets = ["Large Linear_RCT", "Large NonLinear_RCT"]
    args.num_samples = 100
    args.timestamp_in_dirname = False
    args.outcome_model = "auto"  # or use "nested" for the old-style nested model

    use_ray = True
    if use_ray:
        import ray

        ray.init()
    out_dir = run_experiment(args, dataset_path="../RunDatasets", use_ray=use_ray)

    # plot results
    upper_bounds = {"MSE": 1e2, "policy_risk": 0.2}
    lower_bounds = {"erupt": 0.06, "bite": 0.75}

    # Determine case from datasets
    if any("IV" in dataset for dataset in args.datasets):
        case = "IV"
    elif any("KC" in dataset for dataset in args.datasets):
        case = "KC"
    elif any("KCKP" in dataset for dataset in args.datasets):
        case = "KCKP"
    else:
        case = "RCT"
    # upper_bounds = {"MSE": 1e2, "policy_risk": 0.2}
    # lower_bounds = {"erupt": 0.06, "bite": 0.75}
    generate_plots(
        os.path.join(out_dir, case), font_size=8
    )  # , upper_bounds=upper_bounds, lower_bounds=lower_bounds)
