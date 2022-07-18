import math
import numpy as np
import pandas as pd

from typing import Optional, Dict, Union

from econml.cate_interpreter import SingleTreeCateInterpreter
from dowhy.causal_estimator import CausalEstimate

from auto_causality.thirdparty.causalml import metrics
from auto_causality.erupt import ERUPT

import dcor


class DummyEstimator:
    def __init__(
        self, cate_estimate: np.ndarray, effect_intervals: Optional[np.ndarray] = None
    ):
        self.cate_estimate = cate_estimate
        self.effect_intervals = effect_intervals

    def const_marginal_effect(self, X):
        return self.cate_estimate


class Scorer:
    all_metrics = {
        "iv": ["energy_distance"],
        "backdoor": [
            "erupt",
            "norm_erupt",
            "qini",
            "auc",
            "ate",
            "r_scorer",
            "energy_distance",
        ],
    }

    @staticmethod
    def resolve_metric(metric, problem):
        if metric not in Scorer.all_metrics[problem]:
            if problem == "iv":
                return "energy_distance"
            elif problem == "backdoor":
                return "erupt"
        return metric

    @staticmethod
    def resolve_reported_metrics(metrics_to_report, used_metric, problem):
        if metrics_to_report is None:
            if problem == "iv":
                return [used_metric]
            elif problem == "backdoor":
                metrics_to_report = ["qini", "auc", "ate", "erupt", "norm_erupt"]
        else:
            for m in metrics_to_report:
                if m not in Scorer.all_metrics[problem]:
                    raise ValueError(
                        f"Metric to report, {m}, for problem: {problem} \
                        must be one of {Scorer.all_metrics[problem]}"
                    )
        if used_metric not in metrics_to_report:
            metrics_to_report.append(used_metric)
        return metrics_to_report

    @staticmethod
    def validate_implemented_metrics(metric):
        if metric not in Scorer.all_metrics["backdoor"]:
            raise ValueError(
                f"Metric, {metric}, must be\
                 one of {Scorer.all_metrics.values()}"
            )

    @staticmethod
    def energy_distance_score(
        estimate: CausalEstimate,
        df: pd.DataFrame,
    ) -> float:
        est = estimate.estimator
        assert est.identifier_method in ["iv", "backdoor"]

        df["dy"] = estimate.estimator.effect(df)
        df.loc[df[est._treatment_name[0]] == 0, "dy"] = 0
        df["yhat"] = df[est._outcome_name] - df["dy"]

        split_test_by = (
            est.estimating_instrument_names[0]
            if est.identifier_method == "iv"
            else est._treatment_name[0]
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

        qini_score = metrics.qini_score(new_df)

        return qini_score["model"]

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
    def ate(
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

    @staticmethod
    def group_ate(treatment, outcome, policy: Union[pd.DataFrame, np.ndarray]):

        tmp = {"all": Scorer.ate(treatment, outcome)}
        for p in sorted(list(policy.unique())):
            tmp[p] = Scorer.ate(
                treatment[policy == p],
                outcome[policy == p],
            )

        tmp2 = [
            {"policy": str(p), "mean": m, "std": s, "count": c}
            for p, (m, s, c) in tmp.items()
        ]

        return pd.DataFrame(tmp2)

    @staticmethod
    def make_scores(
        estimate: CausalEstimate,
        df: pd.DataFrame,
        propensity_model,
        problem,
        metrics_to_report,
        r_scorer=None,
    ) -> dict:

        out = dict()
        df = df.copy().reset_index()

        if problem == "backdoor":
            est = estimate.estimator
            treatment_name = est._treatment_name
            if not isinstance(treatment_name, str):
                treatment_name = treatment_name[0]
            outcome_name = est._outcome_name
            cate_estimate = est.effect(df)

            intrp = SingleTreeCateInterpreter(
                include_model_uncertainty=False, max_depth=2, min_samples_leaf=10
            )
            intrp.interpret(
                DummyEstimator(cate_estimate), df[est._effect_modifier_names]
            )
            intrp.feature_names = est._effect_modifier_names
            out["intrp"] = intrp

            erupt = ERUPT(
                treatment_name=treatment_name,
                propensity_model=propensity_model,
                X_names=est._effect_modifier_names,
            )
            # TODO: adjust for multiple categorical treatments
            erupt.fit(df)
            simple_ate = Scorer.ate(df[treatment_name], df[outcome_name])[0]
            values = df[[treatment_name, outcome_name]]  # .reset_index(drop=True)
            values["p"] = erupt.propensity_model.predict_proba(df)[:, 1]
            values["policy"] = cate_estimate > 0
            values["norm_policy"] = cate_estimate > simple_ate
            values["weights"] = erupt.weights(df, lambda x: cate_estimate > 0)

            if "ate" in metrics_to_report:
                out["ate"] = cate_estimate.mean()

            if "erupt" in metrics_to_report:
                erupt_score = erupt.score(df, df[outcome_name], cate_estimate > 0)
                out["erupt"] = erupt_score

            if "norm_erupt" in metrics_to_report:
                norm_erupt_score = (
                    erupt.score(df, df[outcome_name], cate_estimate > simple_ate)
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
                print("*****WEIRDNESS*****", k, v)

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
            best[name] = max(
                [
                    v
                    for v in scores.values()
                    if "estimator_name" in v and v["estimator_name"] == name
                ],
                key=lambda x: x[metric],
            )

        return best


# def erupt_make_scores(
#     estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
# ) -> float:
#     est = estimate.estimator
#     treatment_name = est._treatment_name
#     if not isinstance(treatment_name, str):
#         treatment_name = treatment_name[0]
#
#     # prepare the ERUPT scorer
#     erupt = ERUPT(
#         treatment_name=treatment_name,
#         propensity_model=DummyClassifier(strategy="prior"),
#         X_names=est._effect_modifier_names,
#     )
#     erupt.fit(df)
#     erupt_score = erupt.score(
#         df,
#         df[est._outcome_name],
#         cate_estimate > 0,
#     )
#     return erupt_score
