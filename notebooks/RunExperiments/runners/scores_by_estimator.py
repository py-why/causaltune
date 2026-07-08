import os.path
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from causaltune.score.scoring import metrics_to_minimize
from experiment_plots import generate_mse_plots
from experiment_plots import get_all_test_scores
from experiment_runner import get_estimator_list
from latex_utils import df_to_latex, create_latex_table, rename

metrics = [
    "erupt",
    "psw_energy_distance",
    "bite",  # NEW
    "qini",
    "auc",
    "codec",  # NEW
]


def average_clipped_array(arr, prctile: int = 50):
    # Flatten the array in case it's multi-dimensional
    flat_arr = arr.flatten()

    # Remove NaNs and Infs
    valid_arr = flat_arr[np.isfinite(flat_arr)]  # Keeps only finite values

    if valid_arr.size == 0:  # Handle edge case where all values were NaN/Inf
        return np.nan  # Return NaN to indicate no valid values

    # Calculate the absolute values
    abs_arr = np.abs(valid_arr)

    # Determine the 99th percentile value
    threshold = np.percentile(abs_arr, 99)

    # Clip the values above the threshold
    clipped_arr = valid_arr[abs_arr <= threshold]

    if clipped_arr.size == 0:  # Handle edge case where all valid values got clipped
        return np.nan  # Return NaN to indicate no values left

    return (
        np.mean(clipped_arr),
        np.std(clipped_arr),
        np.percentile(clipped_arr, 50),
        np.percentile(clipped_arr, 10),
    )


# Load data files from all the experiments, do a tab
old = False
if old:
    identifiers = [f"Egor_test _{i}" for i in ["meta", "no_meta"]]
else:
    estimators = get_estimator_list("RCT")
    identifiers = [f"Estimator_{estimator.split('.')[-1]}" for estimator in estimators]

kinds = ["RCT", "KCKP", "KC"]

missing_runs = []
dfs = []
out_dirs = []
for identifier in identifiers:
    for kind in kinds:
        out_dir = Path(f"../EXPERIMENT_RESULTS_{identifier}/Large/{kind}")
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)):
            out_dirs.append(out_dir)
        else:
            missing_runs.append(identifier.split("_")[-1])

        for lin in ["Linear", "NonLinear"]:
            dataset = " ".join([lin, kind])
            tmp = get_all_test_scores(str(out_dir), lin, kind)
            tmp["dataset"] = dataset
            dfs.append(tmp)


def contains_any(s: str, substrings: list) -> bool:
    for substring in substrings:
        if substring in s:
            return True
    return False


# patch the holes with old runs
for lin in ["Linear", "NonLinear"]:
    dirs = [f"../EXPERIMENT_RESULTS_Egor_test_{i}/Large/KC" for i in ["meta", "no_meta"]]
    patch_data = get_all_test_scores(dirs, lin, "KC")
    patch_data["dataset"] = f"{lin} KC"
    filter = patch_data["estimator_name"].apply(lambda x: contains_any(x, missing_runs))
    patch_data = patch_data[filter]
    dfs.append(patch_data)


master_df = pd.concat(dfs)

out = []

prctile = 25
for (est_name, ds), df in master_df.groupby(["estimator_name", "dataset"]):
    mse_mean, mse_std, mse_median, mse_quart = average_clipped_array(df["MSE"].values)
    out.append(
        {
            "Estimator": est_name.split(".")[-1],
            "dataset": ds,
            "Median": mse_median,
            "25\\% Quantile": mse_quart,
            "MSE_mean": mse_mean,
            "MSE_std": mse_std,
        }
    )


out_df = pd.DataFrame(out)
for c in [
    "Median",
    "25\\% Quantile",
]:

    # Step 1: Group by dataset and rescale "Median" by the mean of each group
    out_df["Rescaled"] = out_df[c] / out_df.groupby("dataset")[c].transform("mean")

    # Step 2: Compute the average of the rescaled values grouped by "Estimator"
    estimator_scores = out_df.groupby("Estimator")["Rescaled"].mean()

    # Step 3: Get the ordering of "Estimator" values by the computed average (descending order)
    ordering = estimator_scores.sort_values(ascending=True).index.tolist()
    # sort rows by pveral MSE
    out_by_estimator = (
        out_df.groupby("Estimator", as_index=False).mean().reset_index().sort_values(c)
    )

    create_latex_table(
        out_df,
        dataset_col="dataset",
        metric_col="Estimator",
        corr_col=c,
        row_order=ordering,
        filename="tex/mse_by_estimator.tex" if c == "Median" else "tex/mse_by_estimator_25.tex",
        flag_max=False,
        first_col_precision=True,
    )
out2 = []
for (score, ds), df in master_df.groupby(["optimized_score", "dataset"]):
    if score in metrics_to_minimize():
        best_score = df[score].min()
    else:
        best_score = df[score].max()
    best_MSE = df[df[score] == best_score]["MSE"].median()
    out2.append(
        {
            "optimized_score": score,
            "dataset": ds,
            "MSE": best_MSE,
        }
    )

best_MSE = pd.DataFrame(out2)
best_MSE["optimized_score"] = best_MSE["optimized_score"].apply(rename)

create_latex_table(
    best_MSE,
    dataset_col="dataset",
    metric_col="optimized_score",
    corr_col="MSE",
    row_order=[rename(m) for m in metrics],
    filename="tex/best_mse_by_score.tex",
    flag_max=False,
    first_col_precision=False,
)
# for c in ["Median", "25\\% Quantile", "MSE_mean", "MSE_std"]:
#     out_df[c] = out_df[c].apply(lambda x: f"{x:.3f}")
#
# df_to_latex(out_df[["Estimator", "25\\% Quantile", "Median"]], "tex/estimator_mse.tex")

print("yay!")

# now do the same for the ITE scores

save_dir = os.path.realpath("./plots")
results = defaultdict(list)


for pick_nice in [True, False]:
    for kind in kinds:
        out_dirs = []
        if pick_nice:
            nice_estimators = [
                "LinearDML",
                "SparseLinearDML",
                "LinearDRLearner",
                "XLearner",
                "DomainAdaptationLearner",
                "TLearner",
                "SparseLinearDRLearner",
            ]

            used_identifiers = [c for c in identifiers if any([e in c for e in nice_estimators])]
        else:
            used_identifiers = identifiers

        for identifier in used_identifiers:
            out_dir = Path(f"../EXPERIMENT_RESULTS_{identifier}/Large/{kind}")
            out_dirs.append(out_dir)
        upper_bounds = {"MSE": 1e2}
        lower_bounds = {"bite": 0.0}
        tmp_results = generate_mse_plots(
            [str(d) for d in out_dirs],
            metrics,
            save_dir=save_dir,
            prefix="nice" if pick_nice else "est_level",
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
        )
        results[pick_nice].append(tmp_results)
        print("aya")

    combined = pd.concat(results[pick_nice])
    combined["metric"] = combined["metric"].apply(rename)
    combined["dataset"] = combined["dataset"].apply(lambda x: " ".join(x.split(" ")[1::-1]))
    create_latex_table(
        combined,
        dataset_col="dataset",
        metric_col="metric",
        corr_col="corr",
        row_order=[rename(m) for m in metrics],
        filename=f"tex/ite_by_estimator{'_nice' if pick_nice else ''}.tex",
        first_col_precision=False,
    )


print("Pretty scatter generated!")
