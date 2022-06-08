from typing import Optional, Dict, Union
import math

import numpy as np
import pandas as pd

from econml.cate_interpreter import SingleTreeCateInterpreter
from sklearn.dummy import DummyClassifier
from dowhy.causal_estimator import CausalEstimate


from auto_causality.thirdparty.causalml import metrics
from auto_causality.erupt import ERUPT


class DummyEstimator:
    def __init__(
        self, cate_estimate: np.ndarray, effect_intervals: Optional[np.ndarray] = None
    ):
        self.cate_estimate = cate_estimate
        self.effect_intervals = effect_intervals

    def const_marginal_effect(self, X):
        return self.cate_estimate


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


def r_make_score(
    estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray, r_scorer
) -> float:
    # TODO
    return r_scorer.score(cate_estimate)


def make_scores(
    estimate: CausalEstimate,
    df: pd.DataFrame,
    # cate_estimate: np.ndarray = None,
    propensity_model,
    r_scorer=None,
) -> dict:

    df = df.copy().reset_index()
    est = estimate.estimator

    cate_estimate = est.effect(df)

    treatment_name = est._treatment_name
    if not isinstance(treatment_name, str):
        treatment_name = treatment_name[0]

    outcome_name = est._outcome_name

    intrp = SingleTreeCateInterpreter(
        include_model_uncertainty=False, max_depth=2, min_samples_leaf=10
    )
    intrp.interpret(DummyEstimator(cate_estimate), df[est._effect_modifier_names])
    intrp.feature_names = est._effect_modifier_names

    erupt = ERUPT(
        treatment_name=treatment_name,
        propensity_model=propensity_model,
        X_names=est._effect_modifier_names,
    )
    # TODO: adjust for multiple categorical treatments
    erupt.fit(df)

    simple_ate = ate(df[treatment_name], df[outcome_name])[0]

    values = df[[treatment_name, outcome_name]]  # .reset_index(drop=True)
    values["p"] = erupt.propensity_model.predict_proba(df)[:, 1]
    values["policy"] = cate_estimate > 0
    values["norm_policy"] = cate_estimate > simple_ate
    values["weights"] = erupt.weights(df, lambda x: cate_estimate > 0)

    erupt_score = erupt.score(df, df[outcome_name], cate_estimate > 0)

    norm_erupt_score = (
        erupt.score(df, df[outcome_name], cate_estimate > simple_ate)
        - simple_ate * values["norm_policy"].mean()
    )

    values = values.rename(columns={treatment_name: "treated"})

    assert len(values) == len(df), "Index weirdness when adding columns!"

    values = values.copy()

    out = {
        "erupt": erupt_score,
        "norm_erupt": norm_erupt_score,
        "qini": qini_make_score(estimate, df, cate_estimate),
        "auc": auc_make_score(estimate, df, cate_estimate),
        "ate": cate_estimate.mean(),
        "intrp": intrp,
        "values": values,
    }

    if r_scorer is not None:
        out["r_score"] = r_make_score(estimate, df, cate_estimate, r_scorer)

    del df
    return out


def ate(
    treatment,
    outcome,
):
    treated = (treatment == 1).sum()

    mean_ = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    std1 = outcome[treatment == 1].std() / (math.sqrt(treated) + 1e-3)
    std2 = outcome[treatment == 0].std() / (math.sqrt(len(outcome) - treated) + 1e-3)
    std_ = math.sqrt(std1 * std1 + std2 * std2)
    return (mean_, std_, len(treatment))


def group_ate(treatment, outcome, policy: Union[pd.DataFrame, np.ndarray]):

    tmp = {"all": ate(treatment, outcome)}
    for p in policy.unique():
        tmp[p] = ate(
            treatment[policy == p],
            outcome[policy == p],
        )

    tmp2 = [
        {"policy": str(p), "mean": m, "std": s, "count": c}
        for p, (m, s, c) in tmp.items()
    ]

    return pd.DataFrame(tmp2)


def best_score_by_estimator(scores: Dict[str, dict], metric: str) -> Dict[str, dict]:
    for k, v in scores.items():
        if "estimator_name" not in v:
            print("*****WEIRDNESS*****", k, v)

    estimator_names = sorted(
        list(
            set([v["estimator_name"] for v in scores.values() if "estimator_name" in v])
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
