import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
from auto_causality.datasets import synth_ihdp
from auto_causality.data_utils import preprocess_dataset
from auto_causality.scoring import Scorer
from auto_causality.r_score import RScoreWrapper
from dowhy import CausalModel
from econml.cate_interpreter import SingleTreeCateInterpreter
import pandas as pd
import numpy as np


def simple_model_run(rscorer=False):
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
    data = synth_ihdp()
    data_df = data.data
    treatment = data.treatment
    targets = data.outcomes
    data_df, features_X, features_W = preprocess_dataset(data_df, treatment, targets)
    # data_df = data_df.drop(columns = ["random"])
    outcome = targets[0] if type(targets) is list else targets
    train_df, test_df = train_test_split(data_df, train_size=0.5, random_state=123)
    causal_model = CausalModel(
        data=train_df,
        treatment=treatment,
        outcome=outcome,
        common_causes=features_W,
        effect_modifiers=features_X,
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

    scorer = Scorer(causal_model, DummyClassifier(strategy="prior"))

    # TODO: can we use scorer.psw_estimator instead of the estimator here?

    te_train = estimate.cate_estimates
    if rscorer:
        return train_df, test_df, outcome, treatment, features_W, features_X, scorer
    else:
        return estimate, train_df, te_train


class TestMetrics:
    def test_auc_score(self):
        """Tests AUC Score is within exceptable range for the test example"""
        assert Scorer.auc_make_score(*simple_model_run()) == pytest.approx(0.6, 0.05)

    # TODO: Debug wrong values: 1) QINI
    def test_qini_make_score(self):
        """Tests Qini score is within exceptable range for the test example"""
        assert Scorer.qini_make_score(*simple_model_run()) == pytest.approx(0.15, 0.05)

    # TODO: Debug wrong values: 2) R-scorer
    def test_r_make_score(self):
        """Tests RScorer output value is within exceptable range for the test
        example"""

        rscorer = RScoreWrapper(
            DecisionTreeRegressor(random_state=123),
            DummyClassifier(strategy="prior"),
            *simple_model_run(rscorer=True)[:-1]
        )
        assert Scorer.r_make_score(*simple_model_run(), rscorer.train) == pytest.approx(
            0.05, 0.1
        )

    def test_make_scores_with_rscorer(self):
        """Tests make_scores (with rscorer) produces a dictionary of the right
        structure and composition"""
        smr = simple_model_run(rscorer=True)
        scorer = smr[-1]

        rscorer = RScoreWrapper(
            DecisionTreeRegressor(random_state=123), scorer.propensity_model, *smr[:-1]
        )
        true_keys = [
            "erupt",
            "norm_erupt",
            "qini",
            "auc",
            "r_score",
            "ate",
            "intrp",
            "values",
        ]

        scores = scorer.make_scores(
            *(simple_model_run()[:2]),
            "backdoor",
            true_keys[:-2],  # Exclude non-metrics
            rscorer.train
        )
        for i in scores.keys():
            assert i in true_keys
            if i == "intrp":
                assert isinstance(scores[i], SingleTreeCateInterpreter)
            elif i == "values":
                assert isinstance(scores[i], pd.DataFrame)
            else:
                assert isinstance(scores[i], np.float64)

    def test_make_scores_without_rscorer(self):
        """Tests make_scores (without rscorer) returns 0 for 'r_score' key"""
        scorer = simple_model_run(rscorer=True)[-1]

        scores = scorer.make_scores(*(simple_model_run()[:2]), "backdoor", ["ate"])
        assert scores.get("r_score", 0) == 0


if __name__ == "__main__":
    pytest.main([__file__])
