import numpy as np
import pandas as pd
from econml.cate_interpreter import SingleTreeCateInterpreter
from sklearn.dummy import DummyClassifier

from auto_causality.erupt import ERUPT
from dowhy import CausalModel


# need this class because doing inference from scratch is super slow for OrthoForests
class DummyEstimator:
    def __init__(self, cate_estimate: np.ndarray):
        self.cate_estimate = cate_estimate

    def const_marginal_effect(self, X):
        return self.cate_estimate


def make_scores(
    model: CausalModel, df: pd.DataFrame, cate_estimate: np.ndarray
) -> dict:

    # prepare the ERUPT scorer
    erupt = ERUPT(
        treatment=model._treatment[0],
        propensity_model=DummyClassifier(strategy="prior"),
    )
    erupt.fit(df, model._common_causes)
    erupt_score = erupt.score(
        df,
        df[model._outcome[0]],
        lambda x: cate_estimate > 0,
    )

    intrp = SingleTreeCateInterpreter(
        include_model_uncertainty=False, max_depth=2, min_samples_leaf=10
    )
    intrp.interpret(DummyEstimator(cate_estimate), df)
    intrp.feature_names = list(df.columns)
    return {"erupt": erupt_score, "ate": cate_estimate.mean(), "intrp": intrp}
