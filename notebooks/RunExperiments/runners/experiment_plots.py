import glob
import os
import pickle
from typing import Union, List, Optional

import matplotlib
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt


from causaltune.score.scoring import metrics_to_minimize, supported_metrics

metric_names = {
    "psw_frobenius_norm": "PSW\nFrobenius\nNorm",
    "frobenius_norm": "Frobenius\nNorm",
    "erupt": "ERUPT",
    "codec": "CODEC",
    "auc": "AUUC",
    "qini": "Qini",
    "bite": "BITE",
    "policy_risk": "Policy\nRisk",
    "energy_distance": "Energy\nDistance",
    "psw_energy_distance": "Energy\nDistance",
    "norm_erupt": "Normalized\nERUPT",
}


def r2_score(y, yhat):
    """
    Calculate the R-squared (coefficient of determination) for the given true vector y
    and the estimated vector yhat.

    Parameters:
        y (array-like): True values.
        yhat (array-like): Predicted/estimated values.

    Returns:
        float: R-squared value.
    """
    # Convert to numpy arrays to ensure compatibility
    y = np.array(y)
    yhat = np.array(yhat)

    # Calculate the total sum of squares (TSS)
    tss = np.sum((y - np.mean(y)) ** 2)

    # Calculate the residual sum of squares (RSS)
    rss = np.sum((y - yhat) ** 2)

    # Compute R-squared
    r2 = 1 - (rss / tss)
    return r2


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


def get_all_test_scores(out_dirs, lin_nonlin: str, kind: str):
    if isinstance(out_dirs, str):
        out_dirs = [out_dirs]
    all_scores = []
    for out_dir in out_dirs:
        for file in glob.glob(f"{out_dir}/*_{lin_nonlin}_{kind}.pkl"):
            optimized_score = file.split("/")[-1].split("-")[0]
            with open(file, "rb") as f:
                results = pickle.load(f)
                for x in results["all_scores"]:
                    tmp = {k: v for k, v in x["test"]["scores"].items() if k not in ["values"]}
                    tmp["optimized_score"] = optimized_score
                    all_scores.append(tmp)
    out = pd.DataFrame(all_scores)
    return out


def generate_plots(
    out_dir: str,
    metrics: list[str],
    log_scale: Union[List[str], None] = None,
    upper_bounds: Union[dict, None] = None,
    lower_bounds: Union[dict, None] = None,
    prefix: str = "",
    font_size=0,
):
    if log_scale is None:
        log_scale = ["energy_distance", "psw_energy_distance", "frobenius_norm"]
    if upper_bounds is None:
        upper_bounds = {}  # Use an empty dictionary if None
    if lower_bounds is None:
        lower_bounds = {}  # Use an empty dictionary if None

    kind = os.path.basename(out_dir)
    save_dir = os.path.realpath(os.path.join(out_dir, ".."))
    existing_metrics, datasets = extract_metrics_datasets(out_dir)
    # Remove 'ate' from metrics
    metrics = [m for m in metrics if m in existing_metrics]

    # colors = (
    #     [matplotlib.colors.CSS4_COLORS["black"]]
    #     + list(matplotlib.colors.TABLEAU_COLORS)
    #     + [
    #         matplotlib.colors.CSS4_COLORS["lime"],
    #         matplotlib.colors.CSS4_COLORS["yellow"],
    #         matplotlib.colors.CSS4_COLORS["pink"],
    #     ]
    # )
    # markers = ["o", "s", "D", "^", "v", "<", ">", "P", "*", "h", "X", "|", "_", "8"]
    #
    # # Determine the problem type from the dataset name
    # problem = "iv" if any("IV" in dataset for dataset in datasets) else "backdoor"

    def plot_grid(title):
        # Use determined problem type instead of hardcoding "backdoor"
        # files = os.listdir(out_dir)
        all_metrics = metrics  # sorted(list(set([f.split("-")[0] for f in files])))
        if "psw_energy_distance" in all_metrics and "energy_distance" in all_metrics:
            all_metrics.remove("energy_distance")

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
            for metric in all_metrics:
                filename = make_filename(metric, dataset, 1)
                filepath = os.path.join(out_dir, filename)
                if os.path.exists(filepath):
                    results_files[dataset] = filepath
                    break
            if dataset not in results_files:
                print(f"No results file found for dataset {dataset}")

        out_metrics = []

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
                    best_score = float("inf") if metric in metrics_to_minimize() else float("-inf")
                    estimator_name = None

                    for score in results["all_scores"]:
                        if "test" in score and metric in score["test"]["scores"]:
                            current_score = score["test"]["scores"][metric]
                            if metric in metrics_to_minimize():
                                if current_score < best_score:
                                    best_score = current_score
                                    best_estimator = score
                                    estimator_name = score["test"]["scores"]["estimator_name"]
                            else:
                                if current_score > best_score:
                                    best_score = current_score
                                    best_estimator = score
                                    estimator_name = score["test"]["scores"]["estimator_name"]

                    if best_estimator:
                        CATE_gt = np.array(best_estimator["test"]["CATE_groundtruth"]).flatten()
                        CATE_est = np.array(best_estimator["test"]["CATE_estimate"]).flatten()

                        # Plotting
                        ax.scatter(CATE_gt, CATE_est, s=40, alpha=0.5)
                        ax.plot(
                            [min(CATE_gt), max(CATE_gt)],
                            [min(CATE_gt), max(CATE_gt)],
                            "k-",
                            linewidth=1.0,
                        )

                        # Calculate correlation coefficient
                        from scipy.stats import kendalltau

                        corr = kendalltau(CATE_gt, CATE_est).correlation
                        r2 = r2_score(CATE_gt, CATE_est)
                        out_metrics.append(
                            {"dataset": dataset, "metric": metric, "corr": corr, "r2": r2}
                        )

                        # Add correlation
                        ax.text(
                            0.05,
                            0.95,
                            f"Rank corr: {corr:.2f}, R^2: {r2:.2f}",
                            transform=ax.transAxes,
                            verticalalignment="top",
                            fontsize=font_size + 18,
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
                                fontsize=font_size + 24,
                            )

                except Exception as e:
                    print(f"Error processing metric {metric} for dataset {dataset}: {e}")
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
                        fontsize=font_size + 24,
                        fontweight="bold",
                        labelpad=5,  # Reduce padding between label and plot
                    )
                if i == 0:
                    ds_kind = dataset.replace(kind, "").replace("Large", "").strip()
                    ax.set_title(ds_kind, fontsize=font_size + 14, fontweight="bold", pad=15)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(
            f"Estimated ITEs vs. True ITEs for {kind} datasets",
            fontsize=font_size + 18,
            fontweight="bold",
        )
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0.1, 0, 1, 0.96], h_pad=1.0, w_pad=0.5)
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_{kind}_ITE_grid.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_{kind}_ITE_grid.png"),
            format="png",
            bbox_inches="tight",
        )
        plt.close()
        return pd.DataFrame(out_metrics)

    # Generate plots
    results = plot_grid("Experiment Results")
    results.to_pickle(os.path.join(save_dir, f"{prefix}_{kind}_ITE_results.pkl"))


def generate_mse_plots(
    out_dirs: Union[str, List[str]],
    metrics: list[str],
    save_dir: Optional[str] = None,
    log_scale: Union[List[str], None] = None,
    upper_bounds: Union[dict, None] = None,
    lower_bounds: Union[dict, None] = None,
    prefix: str = "",
    font_size=0,
):
    """

    :param out_dirs: each dir must end with the kind of dataset (e.g. "RCT")
    :param metrics:
    :param save_dir:
    :param log_scale:
    :param upper_bounds:
    :param lower_bounds:
    :param prefix:
    :param font_size:
    :return:
    """
    if log_scale is None:
        log_scale = ["energy_distance", "psw_energy_distance", "frobenius_norm"]
    if upper_bounds is None:
        upper_bounds = {}  # Use an empty dictionary if None
    if lower_bounds is None:
        lower_bounds = {}  # Use an empty dictionary if None

    if isinstance(out_dirs, str):
        out_dirs = [out_dirs]
    out_dir = out_dirs[0]

    kind = os.path.basename(out_dir)
    save_dir = save_dir or os.path.realpath(os.path.join(out_dir, ".."))
    existing_metrics, datasets = extract_metrics_datasets(out_dir)
    # Remove 'ate' from metrics
    metrics = [m for m in metrics if m in existing_metrics]

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

    def plot_mse_grid(kind):
        # _, lin_nonlin, kind = datasets[0].split(" ")
        df = get_all_test_scores(out_dirs, "Linear", kind)
        est_names = sorted(df["estimator_name"].unique())

        # Problem type already determined at top level
        all_metrics = [
            c
            for c in df.columns
            if c in supported_metrics(problem, False, False)
            and c.lower() not in ["ate", "norm_erupt"]
        ]

        if "psw_energy_distance" in all_metrics:
            all_metrics.remove("energy_distance")

        all_metrics = [m for m in metrics if m in all_metrics]

        fig, axs = plt.subplots(
            len(all_metrics), len(datasets), figsize=(20, 5 * len(all_metrics)), dpi=300
        )

        # Handle single plot cases
        if len(all_metrics) == 1 and len(datasets) == 1:
            axs = np.array([[axs]])
        elif len(all_metrics) == 1 or len(datasets) == 1:
            axs = axs.reshape(-1, 1) if len(datasets) == 1 else axs.reshape(1, -1)

        out_metrics = []
        legend_elements = []
        for j, lin_nonlin in enumerate(["Linear", "NonLinear"]):
            df = get_all_test_scores(out_dirs, lin_nonlin, kind)
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
                    from scipy.stats import kendalltau

                    corr = kendalltau(this_df["MSE"], this_df[metric]).correlation
                    if metric not in metrics_to_minimize():
                        corr = -corr
                    out_metrics.append(
                        {"dataset": f"{kind} {lin_nonlin}", "metric": metric, "corr": corr}
                    )
                    # Add correlation
                    ax.text(
                        0.05,
                        0.95,
                        f"Rank correlation: {corr:.2f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=font_size + 18,
                        fontweight="bold",
                    )

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
                    if metric not in metrics_to_minimize():
                        ax.invert_yaxis()
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

                    # ds_kind = dataset.replace(kind, "").replace("Large", "").strip()
                    ax.set_title(
                        lin_nonlin.replace("NonLinear", "Nonlinear"),
                        fontsize=font_size + 14,
                        fontweight="bold",
                        pad=15,
                    )
        kind_map = {
            "RCT": "Randomized Controlled Trial",
            "KCKP": "Known but variable PTT",
            "KC": "Variable, estimated PTT",
        }
        plt.suptitle(
            f"MSE vs. Scores for {kind_map[kind]} datasets ",
            fontsize=font_size + 18,
            fontweight="bold",
        )

        # # Match spacing style with plot_grid
        # plt.tight_layout(rect=[0.1, 0, 1, 0.96], h_pad=1.0, w_pad=0.5)
        #
        # fig_legend, ax_legend = plt.subplots(figsize=(6, 6))
        # ax_legend.legend(handles=legend_elements, loc="center", fontsize=10)
        # ax_legend.axis("off")
        fig.legend(handles=legend_elements, loc="lower center", fontsize=18, ncol=2)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_{kind}_MSE_grid.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_{kind}_MSE_grid.png"),
            format="png",
            bbox_inches="tight",
        )
        plt.close()

        # # Create separate legend
        # fig_legend, ax_legend = plt.subplots(figsize=(6, 6))
        # ax_legend.legend(handles=legend_elements, loc="center", fontsize=10)
        # ax_legend.axis("off")
        # plt.savefig(
        #     os.path.join(save_dir, f"{prefix}_{kind}_MSE_legend.pdf"),
        #     format="pdf",
        #     bbox_inches="tight",
        # )
        # plt.savefig(
        #     os.path.join(save_dir, f"{prefix}_{kind}_MSE_legend.png"),
        #     format="png",
        #     bbox_inches="tight",
        # )
        # plt.close()

        return pd.DataFrame(out_metrics)

    results2 = plot_mse_grid(kind)
    results2.to_pickle(os.path.join(save_dir, f"{prefix}_{kind}_MSE_results.pkl"))
    return results2
