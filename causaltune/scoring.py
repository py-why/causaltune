import copy
import logging
import math
from typing import Optional, Dict, Union, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from econml.cate_interpreter import SingleTreeCateInterpreter  # noqa F401
from dowhy.causal_estimator import CausalEstimate
from dowhy import CausalModel

import sys, os

#### for debugging only ##############################
root_path = os.path.realpath("")
sys.path.insert(0, root_path + "/causaltune")
#######################################################

from causaltune.thirdparty.causalml import metrics
from causaltune.erupt import ERUPT
from causaltune.utils import treatment_values, psw_joint_weights

import dcor


class DummyEstimator:
    def __init__(
        self, cate_estimate: np.ndarray, effect_intervals: Optional[np.ndarray] = None
    ):
        self.cate_estimate = cate_estimate
        self.effect_intervals = effect_intervals

    def const_marginal_effect(self, X):
        return self.cate_estimate


def supported_metrics(problem: str, multivalue: bool, scores_only: bool) -> List[str]:
    if problem == "iv":
        metrics = ["energy_distance"]
        if not scores_only:
            metrics.append("ate")
        return metrics
    elif problem == "backdoor":
        if multivalue:
            # TODO: support other metrics for the multivalue case
            return ["energy_distance", "psw_energy_distance"]
        else:
            metrics = [
                "erupt",
                "norm_erupt",
                "qini",
                "auc",
                # "r_scorer",
                "energy_distance",
                "psw_energy_distance",
            ]
            if not scores_only:
                metrics.append("ate")
            return metrics


class Scorer:
    def __init__(
        self,
        causal_model: CausalModel,
        propensity_model: Any,
        problem: str,
        multivalue: bool,
    ):
        """
        Contains scoring logic for CausalTune.

        Access methods and attributes via `CausalTune.scorer`.

        """

        self.problem = problem
        self.multivalue = multivalue
        self.causal_model = copy.deepcopy(causal_model)

        self.identified_estimand = causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )

        if problem == "backdoor":
            print(
                "Fitting a Propensity-Weighted scoring estimator to be used in scoring tasks"
            )
            treatment_series = causal_model._data[causal_model._treatment[0]]
            # this will also fit self.propensity_model, which we'll also use in self.erupt
            self.psw_estimator = self.causal_model.estimate_effect(
                self.identified_estimand,
                method_name="backdoor.causaltune.models.MultivaluePSW",
                control_value=0,
                treatment_value=treatment_values(treatment_series, 0),
                target_units="ate",  # condition used for CATE
                confidence_intervals=False,
                method_params={
                    "init_params": {"propensity_model": propensity_model},
                },
            ).estimator

            treatment_name = self.psw_estimator._treatment_name
            if not isinstance(treatment_name, str):
                treatment_name = treatment_name[0]

            # No need to call self.erupt.fit() as propensity model is already fitted
            # self.propensity_model = est.propensity_model
            self.erupt = ERUPT(
                treatment_name=treatment_name,
                propensity_model=self.psw_estimator.estimator.propensity_model,
                X_names=self.psw_estimator._effect_modifier_names
                + self.psw_estimator._observed_common_causes_names,
            )

    def ate(self, df: pd.DataFrame) -> tuple:
        """
        Calculate the Average Treatment Effect. Provide naive std estimates in single-treatment cases.

        @param df (pandas.DataFrame): input dataframe

        @return tuple: tuple containing the ATE, standard deviation of the estimate (or None if multi-treatment),
            and sample size (or None if estimate has more than one dimension)

        """

        estimate = self.psw_estimator.estimator.effect(df).mean(axis=0)

        if len(estimate) == 1:
            # for now, let's cheat on the std estimation, take that from the naive ate
            treatment_name = self.causal_model._treatment[0]
            outcome_name = self.causal_model._outcome[0]
            naive_est = Scorer.naive_ate(df[treatment_name], df[outcome_name])
            return estimate[0], naive_est[1], naive_est[2]
        else:
            return estimate, None, None

    def resolve_metric(self, metric: str) -> str:
        """
        Check if supplied metric is supported. If not, default to 'energy_distance'.

        @param metric (str): evaluation metric

        @return str: metric/'energy_distance'

        """

        metrics = supported_metrics(self.problem, self.multivalue, scores_only=True)

        if metric not in metrics:
            logging.warning(
                f"Using energy_distance metric as {metric} is not in the list "
                f"of supported metrics for this usecase ({str(metrics)})"
            )
            return "energy_distance"
        else:
            return metric

    def resolve_reported_metrics(
        self, metrics_to_report: Union[List[str], None], scoring_metric: str
    ) -> List[str]:
        """
        Check if supplied reporting metrics are valid.

        Args:
            metrics_to_report (Union[List[str], None]): list of strings specifying the evaluation metrics to compute.
                Possible options include 'ate', 'erupt', 'norm_erupt', 'qini', 'auc', 'energy_distance' and 'psw_energy_distance'.
            scoring_metric (str): specified metric

        Returns:
            List[str]: list of valid metrics
        """

        metrics = supported_metrics(self.problem, self.multivalue, scores_only=False)
        if metrics_to_report is None:
            return metrics
        else:
            metrics_to_report = sorted(list(set(metrics_to_report + [scoring_metric])))
            for m in metrics_to_report.copy():
                if m not in metrics:
                    logging.warning(
                        f"Dropping the metric {m} for problem: {self.problem} \
                        : must be one of {metrics}"
                    )
                    metrics_to_report.remove(m)
        return metrics_to_report

    @staticmethod
    def energy_distance_score(
        estimate: CausalEstimate,
        df: pd.DataFrame,
    ) -> float:
        """
        Calculate energy distance score between treated and controls.

        For theoretical details, see Ramos-Carreño and Torrecilla (2023).

        @param estimate (dowhy.causal_estimator.CausalEstimate): causal estimate to evaluate
        @param df (pandas.DataFrame): input dataframe

        @return float: energy distance score

        """

        Y0X, _, split_test_by = Scorer._Y0_X_potential_outcomes(estimate, df)

        YX_1 = Y0X[Y0X[split_test_by] == 1]
        YX_0 = Y0X[Y0X[split_test_by] == 0]
        select_cols = estimate.estimator._effect_modifier_names + ["yhat"]

        energy_distance_score = dcor.energy_distance(
            YX_1[select_cols], YX_0[select_cols]
        )

        return energy_distance_score

    @staticmethod
    def _Y0_X_potential_outcomes(estimate: CausalEstimate, df: pd.DataFrame):
        est = estimate.estimator
        # assert est.identifier_method in ["iv", "backdoor"]
        treatment_name = (
            est._treatment_name
            if isinstance(est._treatment_name, str)
            else est._treatment_name[0]
        )
        df["dy"] = estimate.estimator.effect_tt(df)
        df["yhat"] = df[est._outcome_name] - df["dy"]

        split_test_by = (
            est.estimating_instrument_names[0]
            if est.identifier_method == "iv"
            else treatment_name
        )

        Y0X = copy.deepcopy(df)
        return Y0X, treatment_name, split_test_by

    def psw_energy_distance(
        self,
        estimate: CausalEstimate,
        df: pd.DataFrame,
        normalise_features=False,
    ) -> float:
        """
        Calculate propensity score adjusted energy distance score between treated and controls.

        Features are normalised using the sklearn.preprocessing.QuantileTransformer

        For theoretical details, see Ramos-Carreño and Torrecilla (2023).

        @param estimate (dowhy.causal_estimator.CausalEstimate): causal estimate to evaluate
        @param df (pandas.DataFrame): input dataframe
        @param normalise_features (bool): whether to normalise features with QuantileTransformer

        @return float: propensity-score weighted energy distance score

        """

        Y0X, treatment_name, split_test_by = Scorer._Y0_X_potential_outcomes(
            estimate, df
        )

        Y0X_1 = Y0X[Y0X[split_test_by] == 1]
        Y0X_0 = Y0X[Y0X[split_test_by] == 0]

        YX_1_all_psw = self.psw_estimator.estimator.propensity_model.predict_proba(
            Y0X_1[
                self.causal_model.get_effect_modifiers()
                + self.causal_model.get_common_causes()
            ]
        )
        treatment_series = Y0X_1[treatment_name]

        YX_1_psw = np.zeros(YX_1_all_psw.shape[0])
        for i in treatment_series.unique():
            YX_1_psw[treatment_series == i] = YX_1_all_psw[:, i][treatment_series == i]

        YX_0_psw = self.psw_estimator.estimator.propensity_model.predict_proba(
            Y0X_0[
                self.causal_model.get_effect_modifiers()
                + self.causal_model.get_common_causes()
            ]
        )[:, 0]

        select_cols = estimate.estimator._effect_modifier_names + ["yhat"]
        features = estimate.estimator._effect_modifier_names

        xy_psw = psw_joint_weights(YX_1_psw, YX_0_psw)
        xx_psw = psw_joint_weights(YX_0_psw)
        yy_psw = psw_joint_weights(YX_1_psw)

        xy_mean_weights = np.mean(xy_psw)
        xx_mean_weights = np.mean(xx_psw)
        yy_mean_weights = np.mean(yy_psw)

        if normalise_features:
            qt = QuantileTransformer(n_quantiles=200)
            X_quantiles = qt.fit_transform(Y0X[features])

            Y0X_transformed = pd.DataFrame(
                X_quantiles, columns=features, index=Y0X.index
            )
            Y0X_transformed.loc[:, ["yhat", split_test_by]] = Y0X[
                ["yhat", split_test_by]
            ]

            Y0X_1 = Y0X_transformed[Y0X_transformed[split_test_by] == 1]
            Y0X_0 = Y0X_transformed[Y0X_transformed[split_test_by] == 0]

        exponent = 1
        distance_xy = np.reciprocal(xy_mean_weights) * np.multiply(
            xy_psw,
            dcor.distances.pairwise_distances(
                Y0X_1[select_cols], Y0X_0[select_cols], exponent=exponent
            ),
        )
        distance_yy = np.reciprocal(yy_mean_weights) * np.multiply(
            yy_psw,
            dcor.distances.pairwise_distances(Y0X_1[select_cols], exponent=exponent),
        )
        distance_xx = np.reciprocal(xx_mean_weights) * np.multiply(
            xx_psw,
            dcor.distances.pairwise_distances(Y0X_0[select_cols], exponent=exponent),
        )
        psw_energy_distance = (
            2 * np.mean(distance_xy) - np.mean(distance_xx) - np.mean(distance_yy)
        )
        return psw_energy_distance


    @staticmethod
    def qini_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
    ) -> float:
        """
        Calculate the Qini score, defined as the area between the Qini curves of a model and random.

        @param estimate (dowhy.causal_estimator.CausalEstimate): causal estimate to evaluate
        @param df (pandas.DataFrame): input dataframe
        @param cate_estimate (np.ndarray): array with cate estimates

        @return float: Qini score

        """

        est = estimate.estimator
        new_df = pd.DataFrame()
        new_df["y"] = df[est._outcome_name]
        treatment_name = est._treatment_name
        if not isinstance(treatment_name, str):
            treatment_name = treatment_name[0]
        new_df["w"] = df[treatment_name]
        new_df["model"] = cate_estimate

        qini_score = metrics.qini_score(new_df)

        return qini_score["model"]

    @staticmethod
    def auc_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
    ) -> float:
        """
        Calculate the area under the uplift curve.

        @param estimate (dowhy.causal_estimator.CausalEstimate): causal estimate to evaluate
        @param df (pandas.DataFrame): input dataframe
        @param cate_estimate (np.ndarray): array with cate estimates

        @return float: area under the uplift curve

        """

        est = estimate.estimator
        new_df = pd.DataFrame()
        new_df["y"] = df[est._outcome_name]
        treatment_name = est._treatment_name
        if not isinstance(treatment_name, str):
            treatment_name = treatment_name[0]
        new_df["w"] = df[treatment_name]
        new_df["model"] = cate_estimate

        auc_score = metrics.auuc_score(new_df)

        return auc_score["model"]

    @staticmethod
    def real_qini_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
    ) -> float:
        # TODO  To calculate the 'real' qini score for synthetic datasets, to be done

        # est = estimate.estimator
        new_df = pd.DataFrame()

        # new_df['tau'] = [df['y_factual'] - df['y_cfactual']]
        new_df["model"] = cate_estimate

        qini_score = metrics.qini_score(new_df)

        return qini_score["model"]

    @staticmethod
    def r_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray, r_scorer
    ) -> float:
        """
        Calculate r_score.

        For details refer to Nie and Wager (2017) and Schuler et al. (2018).

        Adaption from EconML implementation.

        @param estimate (dowhy.causal_estimator.CausalEstimate): causal estimate to evaluate
        @param df (pandas.DataFrame): input dataframe
        @param cate_estimate (np.ndarray): array with cate estimates
        @param r_scorer: callable object used to compute the R-score

        @return float: r_score

        """

        # TODO
        return r_scorer.score(cate_estimate)

    @staticmethod
    def naive_ate(treatment: pd.Series, outcome: pd.Series):
        """
        Calculate simple ATE.

        @param treatment (pandas.Series): series of treatments
        @param outcome (pandas.Series): series of outcomes

        @return: tuple of simple ATE, standard deviation, and sample size

        """

        treated = (treatment == 1).sum()

        mean_ = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
        std1 = outcome[treatment == 1].std() / (math.sqrt(treated) + 1e-3)
        std2 = outcome[treatment == 0].std() / (
            math.sqrt(len(outcome) - treated) + 1e-3
        )
        std_ = math.sqrt(std1 * std1 + std2 * std2)
        return (mean_, std_, len(treatment))

    def group_ate(
        self, df: pd.DataFrame, policy: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compute the average treatment effect (ATE) for different groups specified by a policy.

        @param df (pandas.DataFrame): input dataframe, should contain columns for the treatment, outcome, and policy
        @param policy (Union[pd.DataFrame, np.ndarray]): policy column in df or an array of the policy values,
            used to group the data

        @return: pandas.DataFrame of ATE, std, and size per policy

        """

        tmp = {"all": self.ate(df)}
        for p in sorted(list(policy.unique())):
            tmp[p] = self.ate(df[policy == p])

        tmp2 = [
            {"policy": str(p), "mean": m, "std": s, "count": c}
            for p, (m, s, c) in tmp.items()
        ]

        return pd.DataFrame(tmp2)

    def make_scores(
        self,
        estimate: CausalEstimate,
        df: pd.DataFrame,
        metrics_to_report: List[str],
        r_scorer=None,
    ) -> dict:
        """Calculate various performance metrics for a given causal estimate using a given DataFrame.

        Args:
            estimate (dowhy.causal_estimator.CausalEstimate): causal estimate to evaluate
            df (pandas.DataFrame): input dataframe
            metrics_to_report (List[str]): list of strings specifying the evaluation metrics to compute.
                Possible options include 'ate', 'erupt', 'norm_erupt', 'qini', 'auc', 'energy_distance' and 'psw_energy_distance'.
            r_scorer (Optional): callable object used to compute the R-score, default is None

        Returns:
            dict: dictionary containing the evaluation metrics specified in metrics_to_report.
                The values key in the dictionary contains the input DataFrame with additional columns for
                the propensity scores, the policy, the normalized policy, and the weights, if applicable.
        """

        out = dict()
        df = df.copy().reset_index()

        est = estimate.estimator
        treatment_name = est._treatment_name
        if not isinstance(treatment_name, str):
            treatment_name = treatment_name[0]
        outcome_name = est._outcome_name

        cate_estimate = est.effect(df)

        # TODO: fix this hack with proper treatment of multivalues
        if len(cate_estimate.shape) > 1 and cate_estimate.shape[1] == 1:
            cate_estimate = cate_estimate.reshape(-1)

        # TODO: fix this, currently broken
        # covariates = est._effect_modifier_names
        # Include CATE Interpereter for both IV and CATE models
        # intrp = SingleTreeCateInterpreter(
        #     include_model_uncertainty=False, max_depth=2, min_samples_leaf=10
        # )
        # intrp.interpret(DummyEstimator(cate_estimate), df[covariates])
        # intrp.feature_names = covariates
        # out["intrp"] = intrp

        if self.problem == "backdoor":
            values = df[[treatment_name, outcome_name]]
            simple_ate = self.ate(df)[0]
            if isinstance(simple_ate, float):
                # simple_ate = simple_ate[0]
                # .reset_index(drop=True)
                values[
                    "p"
                ] = self.psw_estimator.estimator.propensity_model.predict_proba(
                    df[
                        self.causal_model.get_effect_modifiers()
                        + self.causal_model.get_common_causes()
                    ]
                )[
                    :, 1
                ]
                values["policy"] = cate_estimate > 0
                values["norm_policy"] = cate_estimate > simple_ate
                values["weights"] = self.erupt.weights(df, lambda x: cate_estimate > 0)
            else:
                pass
                # TODO: what do we do here if multiple treatments?

            if "erupt" in metrics_to_report:
                erupt_score = self.erupt.score(df, df[outcome_name], cate_estimate > 0)
                out["erupt"] = erupt_score

            if "norm_erupt" in metrics_to_report:
                norm_erupt_score = (
                    self.erupt.score(df, df[outcome_name], cate_estimate > simple_ate)
                    - simple_ate * values["norm_policy"].mean()
                )
                out["norm_erupt"] = norm_erupt_score

            if "qini" in metrics_to_report:
                out["qini"] = Scorer.qini_make_score(estimate, df, cate_estimate)

            if "auc" in metrics_to_report:
                out["auc"] = Scorer.auc_make_score(estimate, df, cate_estimate)

            if r_scorer is not None:
                out["r_score"] = Scorer.r_make_score(
                    estimate, df, cate_estimate, r_scorer
                )

            # values = values.rename(columns={treatment_name: "treated"})
            assert len(values) == len(df), "Index weirdness when adding columns!"
            values = values.copy()
            out["values"] = values

        if "ate" in metrics_to_report:
            out["ate"] = cate_estimate.mean()
            out["ate_std"] = cate_estimate.std()

        if "energy_distance" in metrics_to_report:
            out["energy_distance"] = Scorer.energy_distance_score(estimate, df)

        if "psw_energy_distance" in metrics_to_report:
            out["psw_energy_distance"] = self.psw_energy_distance(
                estimate,
                df,
            )

        del df
        return out

    @staticmethod
    def best_score_by_estimator(
        scores: Dict[str, dict], metric: str
    ) -> Dict[str, dict]:
        """
        Obtain best score for each estimator.

        @param scores (Dict[str, dict]): CausalTune.scores dictionary
        @param metric (str): metric of interest

        @return Dict[str, dict]: dictionary containing best score by estimator

        """

        for k, v in scores.items():
            if "estimator_name" not in v:
                raise ValueError(
                    f"Malformed scores dict, 'estimator_name' field missing in {k}, {v}"
                )

        estimator_names = sorted(
            list(
                set(
                    [
                        v["estimator_name"]
                        for v in scores.values()
                        if "estimator_name" in v
                    ]
                )
            )
        )
        best = {}
        for name in estimator_names:
            est_scores = [
                v
                for v in scores.values()
                if "estimator_name" in v and v["estimator_name"] == name
            ]
            best[name] = (
                min(est_scores, key=lambda x: x[metric])
                if metric in ["energy_distance", "psw_energy_distance"]
                else max(est_scores, key=lambda x: x[metric])
            )

        return best


if __name__ == "__main__":
    import pytest
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.dummy import DummyClassifier

    from dowhy import CausalModel

    from causaltune.datasets import synth_ihdp

    # from causaltune.scoring import Scorer, supported_metrics

    """Creates data to allow testing of metrics
    Args:
        rscorer (bool): determines whether the function returns the correct
        inputs for the RScoreWrapper (True) or for the metrics (False)
    Returns:
        if rscorer=True:
            input parameters for RScoreWrapper
        if rscorer=False:
            input parameters for metrics functions (such as qini_make_score
    """
    rscorer = True
    data = synth_ihdp()
    data.preprocess_dataset()

    train_df, test_df = train_test_split(data.data, train_size=0.5, random_state=123)
    causal_model = CausalModel(
        data=train_df,
        treatment=data.treatment,
        outcome=data.outcomes[0],
        common_causes=data.common_causes,
        effect_modifiers=data.effect_modifiers,
        random_state=123,
    )
    identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    estimate = causal_model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.metalearners.SLearner",
        control_value=0,
        treatment_value=1,
        target_units="ate",  # condition used for CATE
        confidence_intervals=False,
        # random_state=123,
        method_params={
            "init_params": {"overall_model": DecisionTreeRegressor(random_state=123)},
            "fit_params": {},
        },
    )

    scorer = Scorer(
        causal_model,
        DummyClassifier(strategy="prior"),
        problem="backdoor",
        multivalue=False,
    )

    # TODO: can we use scorer.psw_estimator instead of the estimator here?

    te_train = estimate.cate_estimates

    scorer.psw_energy_distance(estimate, train_df)
