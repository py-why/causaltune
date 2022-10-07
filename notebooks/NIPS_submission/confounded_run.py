import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..

from typing import Union

from sklearn.model_selection import train_test_split
from auto_causality import AutoCausality
from auto_causality.data_utils import preprocess_dataset
from auto_causality.datasets import generate_synthetic_data
from auto_causality.models.passthrough import Passthrough

# set a few params
metrics = ["norm_erupt", "qini", "energy_distance"]
n_samples = 30000
test_size = 0.33  # equal train,val,test
components_time_budget = 60 * 20  # 60 * 20
estimator_list = "all"  # ["Dummy"]  #
# [
#     "Dummy",
#     "SLearner",
#     "TLearner",
#     "XLearner",
#     "DomainAdaptationLearner",
#     "ForestDRLearner",
#     ".LinearDRLearner",
#     "TransformedOutcome",
#     ".LinearDML",
#     "CasualForestDML",
# ]
n_runs = 1
out_dir = os.path.realpath(".")
filename_out = "synthetic_observational_passthrough"

dataset = generate_synthetic_data(
    n_samples=n_samples, confounding=True, linear_confounder=False, noisy_outcomes=True
)
data_df, features_X, features_W = preprocess_dataset(
    dataset.data, treatment=dataset.treatment, targets=dataset.outcomes
)
# drop true effect:
features_X = [f for f in features_X if f != "true_effect"]
print(f"features_X: {features_X}")
print(f"features_W: {features_W}")

from collections import defaultdict


train_df, test_df = train_test_split(data_df, test_size=test_size)
test_df = test_df.reset_index(drop=True)
ac = AutoCausality(
    metric="norm_erupt",
    verbose=2,
    components_verbose=2,
    components_time_budget=components_time_budget,
    num_samples=12,
    estimator_list=estimator_list,
    store_all_estimators=False,
    propensity_model=Passthrough("random"),
)

ac.fit(
    train_df,
    treatment="treatment",
    outcome="outcome",
    common_causes=features_W,
    effect_modifiers=features_X,
)


for i_run in range(1, n_runs + 1):
    estimator_scores = defaultdict(list)
    for metric in metrics:
        if True:
            # try:

            estimator_scores = defaultdict(list)
            # compute relevant scores (skip newdummy)
            datasets = {"train": ac.train_df, "validation": ac.test_df, "test": test_df}
            # get scores on train,val,test for each trial,
            # sort trials by validation set performance
            # assign trials to estimators
            #             estimator_scores = {est: [] for est in ac.scores.keys() if "NewDummy" not in est}

            for trial in ac.results.trials:
                # estimator name:
                estimator_name = trial.last_result["estimator_name"]
                if trial.last_result.get("estimator", False):
                    estimator = trial.last_result["estimator"]
                    scores = {}
                    for ds_name, df in datasets.items():
                        scores[ds_name] = {}
                        # make scores
                        est_scores = ac.scorer.make_scores(
                            estimator,
                            df,
                            problem=ac.problem,
                            metrics_to_report=ac.metrics_to_report,
                        )

                        # add cate:
                        scores[ds_name]["CATE_estimate"] = estimator.estimator.effect(
                            df
                        )
                        # add ground truth for convenience
                        scores[ds_name]["CATE_groundtruth"] = df["true_effect"]
                        scores[ds_name][metric] = est_scores[metric]
                    estimator_scores[estimator_name].append(scores)

            # sort trials by validation performance
            for k in estimator_scores.keys():
                estimator_scores[k] = sorted(
                    estimator_scores[k],
                    key=lambda x: x["validation"][metric],
                    reverse=False if metric == "energy_distance" else True,
                )
            results = {
                "best_estimator": ac.best_estimator,
                "best_config": ac.best_config,
                "best_score": ac.best_score,
                "optimised_metric": metric,
                "scores_per_estimator": estimator_scores,
            }

            with open(
                os.path.join(out_dir, f"{filename_out}_{metric}_run_{i_run}.pkl"), "wb"
            ) as f:
                pickle.dump(results, f)
        # except Exception as e:
        #     print(e)

# f, axs = plt.subplots(1, len(metrics), figsize=(8, 2.5), dpi=300)
#
# # plot true against estimated for best estimator:
# for ax, metric in zip(axs, metrics):
#     try:
#         with open(f"{out_dir}{filename_out}_{metric}_run_1.pkl", "rb") as f:
#             results = pickle.load(f)
#         CATE_gt = results["scores_per_estimator"][results["best_estimator"]][0]["test"][
#             "CATE_groundtruth"
#         ]
#         CATE_est = results["scores_per_estimator"][results["best_estimator"]][0][
#             "test"
#         ]["CATE_estimate"]
#
#         ax.scatter(CATE_gt, CATE_est, s=20, alpha=0.1)
#         ax.plot(
#             [min(CATE_gt), max(CATE_gt)],
#             [min(CATE_gt), max(CATE_gt)],
#             "k-",
#             linewidth=0.5,
#         )
#         ax.set_xlabel("true CATE")
#         ax.set_ylabel("estimated CATE")
#         ax.set_title(f"{results['optimised_metric']}")
#         ax.set_xlim([-15, 15])
#         ax.set_ylim([-15, 15])
#         # ax.set_xticks(np.arange(-0.5,0.51,0.5))
#         # ax.set_yticks(np.arange(-0.5,0.51,0.5))
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#     except:
#         pass
# plt.tight_layout()

import colorsys


def scale_lightness(rgb, scale_l):
    # found here https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

matplotlib.use("TkAgg")
colors = (
    [matplotlib.colors.CSS4_COLORS["black"]]
    + list(matplotlib.colors.TABLEAU_COLORS)
    + [
        matplotlib.colors.CSS4_COLORS["lime"],
        matplotlib.colors.CSS4_COLORS["yellow"],
        matplotlib.colors.CSS4_COLORS["pink"],
    ]
)

plt.figure(figsize=(10, 4.5), dpi=300)
# f, axs = plt.subplots(1,len(metrics),)
filenames_out = [filename_out]
est_labels = [[], [], []]
sc = [[], [], []]
for row, base_fn in enumerate(filenames_out):
    for i, metric in enumerate(metrics):
        plt.subplot(2, len(metrics), i + 1 + len(metrics) * row)
        fn = os.path.join(out_dir, f"{filename_out}_{metric}_run_{i_run}.pkl")
        print(fn)
        with open(fn, "rb") as f:
            results = pickle.load(f)

        for (est_name, scr), col in zip(
            results["scores_per_estimator"].items(), colors
        ):
            if "Dummy" not in est_name:
                if len(scr):
                    # also plot intermediate runs:
                    #                 if len(scr) > 1:
                    #                     print(f"{est_name}: {len(scr)} intermediate runs ")
                    #                     lightness = np.linspace(1,2.8,len(scr))

                    #                     col_rgb = matplotlib.colors.ColorConverter.to_rgb(col)
                    #                     for i_run in range(1,len(scr)):
                    #                         CATE_gt = scr[i_run]["test"]["CATE_groundtruth"]
                    #                         CATE_est = scr[i_run]["test"]["CATE_estimate"]
                    #                         mse=np.mean((CATE_gt-CATE_est)**2)
                    #                         score = scr[i_run]["test"][metric]
                    #                         plt.scatter(mse,score,color=scale_lightness(col_rgb,lightness[i_run-1]),s=30,linewidths=0.5, label="_nolegend_" )
                    # get score for best estimator:
                    CATE_gt = scr[0]["test"]["CATE_groundtruth"]
                    CATE_est = scr[0]["test"]["CATE_estimate"]
                    mse = np.mean((CATE_gt - CATE_est) ** 2)
                    score = scr[0]["test"][metric]
                    print(metric, est_name, mse, score)
                    plt.scatter(mse, score, color=col, s=30, linewidths=0.5)
                    est_labels[i].append(est_name.split(".")[-1])
        if i == 1 and row == 1:
            plt.xlabel("Mean square error of pointwise estimated impact")
        if i == 0:
            plt.ylabel("test score")
        if row == 0:
            plt.title(metric)
        plt.xscale("log")
        # if row == 0:
        #     plt.xlim(10**-4.1, 10**-2.6)
        plt.grid(True)

    plt.legend(
        est_labels[0], loc="center left", bbox_to_anchor=(1.2, 0.5), frameon=False
    )
plt.tight_layout()
# plt.savefig(f"NIPS/paper_CATE_synth_scores.pdf",format="pdf")
plt.show()
print("yahoo!")
