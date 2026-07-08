import os

from latex_utils import create_latex_table, rename

from no_meta_rct import metrics

import pandas as pd

if __name__ == "__main__":

    identifier = "Egor_test_no_meta"
    data_dir = os.path.realpath(os.path.join(f"../EXPERIMENT_RESULTS_{identifier}/Large"))

    results_ite = []
    results_mse = []
    cols = []
    for kind in [
        "RCT",
        "KCKP",
        "KC",
    ]:
        results_ite.append(pd.read_pickle(os.path.join(data_dir, f"{kind}_ITE_results.pkl")))
        results_mse.append(pd.read_pickle(os.path.join(data_dir, f"{kind}_MSE_results.pkl")))
        cols += [f"Linear {kind}", f"Nonlinear {kind}"]

    results_ite = pd.concat(results_ite)
    results_mse = pd.concat(results_mse)

    results_ite["dataset"] = results_ite["dataset"].str.replace("Large ", "")
    results_mse["dataset"] = results_mse["dataset"].str.replace("Large ", "")

    results_ite["metric"] = results_ite["metric"].apply(rename)
    results_mse["metric"] = results_mse["metric"].apply(rename)

    create_latex_table(
        results_ite,
        dataset_col="dataset",
        metric_col="metric",
        corr_col="corr",
        row_order=[rename(m) for m in metrics],
        column_order=cols,
        filename=os.path.join(data_dir, "table_ite_corr.tex"),
    )

    create_latex_table(
        results_ite,
        dataset_col="dataset",
        metric_col="metric",
        corr_col="r2",
        row_order=[rename(m) for m in metrics],
        column_order=cols,
        filename=os.path.join(data_dir, "table_ite_r2.tex"),
    )

    create_latex_table(
        results_mse,
        dataset_col="dataset",
        metric_col="metric",
        corr_col="corr",
        row_order=[rename(m) for m in metrics],
        column_order=cols,
        filename=os.path.join("tex", "table_mse_corr.tex"),
    )
    print("yay!")
