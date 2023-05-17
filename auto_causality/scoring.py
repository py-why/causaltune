import copy
import logging
import math
from typing import Optional, Dict, Union, Any, List

import numpy as np
import pandas as pd

from econml.cate_interpreter import SingleTreeCateInterpreter  # noqa F401
from dowhy.causal_estimator import CausalEstimate
from dowhy import CausalModel

from auto_causality.thirdparty.causalml import metrics
from auto_causality.erupt import ERUPT
from auto_causality.utils import treatment_values

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
            return ["energy_distance"]
        else:
            metrics = [
                "erupt",
                "norm_erupt",
                "qini",
                "auc",
                # "r_scorer",
                "energy_distance",
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
                method_name="backdoor.auto_causality.models.MultivaluePSW",
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

    def ate(self, df: pd.DataFrame):
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
    ):
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

        X1 = df[df[split_test_by] == 1]
        X0 = df[df[split_test_by] == 0]
        select_cols = est._effect_modifier_names + ["yhat"]

        energy_distance_score = dcor.energy_distance(X1[select_cols], X0[select_cols])

        return energy_distance_score

    @staticmethod
    def qini_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
    ) -> float:
        est = estimate.estimator
        new_df = pd.DataFrame()
        new_df["y"] = df[est._outcome_name]
        treatment_name = est._treatment_name
        if not isinstance(treatment_name, str):
            treatment_name = treatment_name[0]
        new_df["w"] = df[treatment_name]
        new_df["model"] = cate_estimate

        qini_score = metrics.qini_score(new_df)["model"]

        return qini_score

    @staticmethod
    def auc_make_score(
        estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
    ) -> float:
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
        # TODO
        return r_scorer.score(cate_estimate)

    @staticmethod
    def naive_ate(
        treatment,
        outcome,
    ):
        treated = (treatment == 1).sum()

        mean_ = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
        std1 = outcome[treatment == 1].std() / (math.sqrt(treated) + 1e-3)
        std2 = outcome[treatment == 0].std() / (
            math.sqrt(len(outcome) - treated) + 1e-3
        )
        std_ = math.sqrt(std1 * std1 + std2 * std2)
        return (mean_, std_, len(treatment))

    def group_ate(self, df, policy: Union[pd.DataFrame, np.ndarray]):
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
        metrics_to_report,
        r_scorer=None,
    ) -> dict:
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

            values = values.rename(columns={treatment_name: "treated"})
            assert len(values) == len(df), "Index weirdness when adding columns!"
            values = values.copy()
            out["values"] = values

        if "ate" in metrics_to_report:
            out["ate"] = cate_estimate.mean()
            out["ate_std"] = cate_estimate.std()

        if "energy_distance" in metrics_to_report:
            out["energy_distance"] = Scorer.energy_distance_score(estimate, df)

        del df
        return out

    @staticmethod
    def best_score_by_estimator(
        scores: Dict[str, dict], metric: str
    ) -> Dict[str, dict]:
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
                if metric == "energy_distance"
                else max(est_scores, key=lambda x: x[metric])
            )

        return best
