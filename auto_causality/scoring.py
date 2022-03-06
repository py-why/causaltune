from typing import Optional
import math
from causalml import metrics

import numpy as np
import pandas as pd
from econml.cate_interpreter import SingleTreeCateInterpreter
from sklearn.dummy import DummyClassifier

from auto_causality.erupt import ERUPT
from dowhy.causal_estimator import CausalEstimate

from matplotlib import pyplot as plt

 
# need this class because doing inference from scratch is super slow for some models
class DummyEstimator:
    def __init__(
        self, cate_estimate: np.ndarray, effect_intervals: Optional[np.ndarray] = None
    ):
        self.cate_estimate = cate_estimate
        self.effect_intervals = effect_intervals

    def const_marginal_effect(self, X):
        return self.cate_estimate


def erupt_make_scores(
    estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray
) -> float:
    est = estimate.estimator
    treatment_name = est._treatment_name
    if not isinstance(treatment_name, str):
        treatment_name = treatment_name[0]

    # prepare the ERUPT scorer
    erupt = ERUPT(
        treatment_name=treatment_name,
        propensity_model=DummyClassifier(strategy="prior"),
        X_names=est._effect_modifier_names,
    )
    erupt.fit(df)
    erupt_score = erupt.score(
        df,
        df[est._outcome_name],
        cate_estimate > 0,
    )
    return erupt_score


def qini_make_score(
        estimate: CausalEstimate,
        df: pd.DataFrame,
        cate_estimate: np.ndarray) -> float:
    est = estimate.estimator
    new_df = pd.DataFrame()
    new_df['y'] = df[est._outcome_name]
    #new_df['tau'] = cate_estimate
    treatment_name = est._treatment_name
    if not isinstance(treatment_name, str):
        treatment_name = treatment_name[0]
    new_df['w'] = df[treatment_name]
    new_df['model'] = cate_estimate

    qini_score = metrics.visualize.qini_score(
        new_df
    )
    

    return qini_score['model']

def real_qini_make_score( 
        estimate: CausalEstimate,
        df: pd.DataFrame,
        cate_estimate: np.ndarray) -> float:
    est = estimate.estimator
    new_df = pd.DataFrame()

    #new_df['tau'] = [df['y_factual'] - df['y_cfactual']] 
    new_df['model'] = cate_estimate

    qini_score = metrics.visualize.qini_score(
        new_df
    )
    

    return qini_score['model']

    # To calculate the 'real' qini score for synthetic datasets, to be done


def r_scorer_make_score(
        estimate: CausalEstimate,
        df: pd.DataFrame,
        cate_estimate: np.ndarray) -> float:
    est = estimate.estimator
    
        


def make_scores(
    estimate: CausalEstimate, df: pd.DataFrame, cate_estimate: np.ndarray, metric: str = "erupt"
) -> dict:

    est = estimate.estimator
    treatment_name = est._treatment_name
    if not isinstance(treatment_name, str):
        treatment_name = treatment_name[0]


    intrp = SingleTreeCateInterpreter(
        include_model_uncertainty=False, max_depth=2, min_samples_leaf=10
    )
    intrp.interpret(DummyEstimator(cate_estimate), df)
    intrp.feature_names = est._effect_modifier_names

    erupt = ERUPT(
        treatment_name=treatment_name,
        propensity_model=DummyClassifier(strategy="prior"),
        X_names=est._effect_modifier_names,
    )
    erupt.fit(df)
    values = df[[treatment_name, est._outcome_name]].reset_index(drop=True)
    values["p"] = erupt.propensity_model.predict_proba(df)[:, 1]
    values["policy"] = cate_estimate > 0
    values["weights"] = erupt.weights(df, lambda x: cate_estimate > 0)

    values = values.rename(columns={treatment_name: "treated"})
    

    assert len(values) == len(df), "Index weirdness when adding columns!"

    return {
        "erupt": erupt_make_scores(estimate, df, cate_estimate),
        'qini':qini_make_score(estimate, df, cate_estimate),
        "ate": cate_estimate.mean(),
        "intrp": intrp,
        "values": values,
    }


def ate(
    treatment,
    outcome,
):
    treated = (treatment == 1).sum()

    mean_ = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    std1 = outcome[treatment == 1].std() / (math.sqrt(treated) + 1e-3)
    std2 = outcome[treatment == 0].std() / (math.sqrt(len(outcome) - treated) + 1e-3)
    std_ = math.sqrt(std1 * std1 + std2 * std2)
    #     print(treated, mean_, std1, std2, std_)
    return (mean_, std_, len(treatment))


def group_ate(treatment, outcome, policy):
    tmp = {
        "all": ate(treatment, outcome),
        "pos": ate(
            treatment[policy == 1],
            outcome[policy == 1],
        ),
        "neg": ate(
            treatment[policy == 0],
            outcome[policy == 0],
        ),
    }
    out = {}
    for key, (mean_, std_, count_) in tmp.items():
        out[f"{key}_mean"] = mean_
        out[f"{key}_std"] = std_
        out[f"{key}_count"] = count_
    return out