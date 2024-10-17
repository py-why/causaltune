from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import shap
import causaltune
from causaltune.shap import shap_values
from causaltune.score.scoring import Scorer
from dowhy.causal_model import CausalEstimate


# TODO: decide on how Visualizer and CausalTune should interact; currently standalone
# TODO: make plots themselves a bit nicer and allow for more flexibility/customization
# currently not natively supporting multiple treatments


class Visualizer:
    def __init__(
        self,
        test_df: pd.DataFrame,
        treatment_col_name: str,
        outcome_col_name: str,
    ) -> None:
        assert (
            treatment_col_name and outcome_col_name in test_df.columns
        ), "Check that treatment and outcome columns are specified correctly."

        self.test_df = test_df
        self.treatment_col_name = treatment_col_name
        self.outcome_col_name = outcome_col_name
        self.t_list = [
            int(t) for t in sorted(list(self.test_df[self.treatment_col_name].unique()))
        ]

    def plot_shap(
        self,
        estimate: CausalEstimate,
        df: pd.DataFrame,
        shaps: np.ndarray = None,
        figtitle: str = "",
    ) -> None:
        """Plot Shapley values for given estimator.

        Args:
            estimate (CausalEstimate): dowhy causal estimate
            df (pandas.DataFrame): dataframe for shapley calculation
            shaps (np.ndarray): pre-computed shapley values, if None, compute shapley values. Defaults to None.
            figtitle (str): plot title

        Returns:
            None

        """
        assert not isinstance(
            estimate, causaltune.models.dummy.Dummy
        ), "Supplied model does not depends on features!"

        if shaps is None:
            sv = shap_values(estimate=estimate, df=df)
        else:
            sv = shaps

        plt.title(figtitle)
        shap.summary_plot(sv, df[estimate.estimator._input_names["feature_names"]])
        plt.show()

    def plot_metrics_by_estimator(
        self,
        scores_dict: Dict,
        figtitle: str = "",
        metrics: Tuple[str, str] = ("norm_erupt", "ate"),
        figsize: Tuple[int, int] = (7, 5),
    ) -> None:
        """Plot metrics by estimator.

        Args:
            scores_dict (Dict): scores dict from fitted CausalTune object
            figtitle (str): specifying plot title, defaults to ""
            metrics (Tuple[str, str]): specifying metrics, defaults to ('norm_erupt', 'ate')
            figsize (Tuple[int, int]): specifying plot size, defaults to (7,5)

        Returns:
            None
        """

        colors = (
            [matplotlib.colors.CSS4_COLORS["black"]]
            + list(matplotlib.colors.TABLEAU_COLORS)
            + [
                matplotlib.colors.CSS4_COLORS["lime"],
                matplotlib.colors.CSS4_COLORS["yellow"],
                matplotlib.colors.CSS4_COLORS["pink"],
            ]
        )

        plt.figure(figsize=figsize)
        plt.title(figtitle)

        m1 = metrics[1]
        m2 = metrics[0]

        for (est, scr), col in zip(scores_dict.items(), colors):
            try:
                sc = [
                    scr["scores"]["train"][m1],
                    scr["scores"]["validation"][m1],
                    scr["scores"]["test"][m1],
                ]
                crv = [
                    scr["scores"]["train"][m2],
                    scr["scores"]["validation"][m2],
                    scr["scores"]["test"][m2],
                ]
                plt.plot(sc, crv, color=col, marker="o", label=est)
                plt.scatter(sc[1:2], crv[1:2], c=col, s=70, label="_nolegend_")
                plt.scatter(sc[2:], crv[2:], c=col, s=120, label="_nolegend_")

            except Exception:
                print("Please check inputs.")
        plt.xlabel(m1)
        plt.ylabel(m2)

        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.annotate(
            "By size (ascending): train/val/test",
            (0, 0),
            (0, -40),
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
        )

        plt.grid()
        plt.show()

    def plot_group_ate(
        self, scorer: Scorer, scores_dict: Dict, estimator: str
    ) -> pd.DataFrame:
        """Plot out-of sample difference of outcomes between treated and untreated
        for the points where model predicts positive vs negative impact.

        Args:
            scorer (CausalTune.scorer): scorer from CausalTune
            scores_dict (Dict): scores dict from fitted CausalTune object
            estimator (str): name of estimator to use

        Returns:
            pandas.DataFrame: dataframe of policy groups and average effect

        """

        test_policy_df = scores_dict[estimator]["scores"]["test"]["values"]
        sts = scorer.group_ate(
            self.test_df.reset_index(), test_policy_df["norm_policy"]
        )

        colors = (
            matplotlib.colors.CSS4_COLORS["black"],
            matplotlib.colors.CSS4_COLORS["red"],
            matplotlib.colors.CSS4_COLORS["blue"],
        )

        grp = sts["policy"].unique()

        for i, (p, c) in enumerate(zip(grp, colors)):
            st = sts[sts["policy"] == p]
            plt.errorbar(
                np.array(range(len(st))) + 0.1 * i,
                st["mean"].values[0],
                yerr=st["std"].values[0],
                fmt="o",
                color=c,
            )
        plt.legend(grp)
        plt.grid(True)
        plt.xticks([])
        plt.title(estimator.split(".")[-1])
        plt.show()
