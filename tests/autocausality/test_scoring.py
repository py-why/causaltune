import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
from auto_causality.datasets import synth_ihdp, preprocess_dataset
from auto_causality.scoring \
    import auc_make_score, r_make_score, make_scores, qini_make_score
from auto_causality.r_score import RScoreWrapper
from dowhy import CausalModel
from econml.cate_interpreter import SingleTreeCateInterpreter
import pandas as pd
import numpy as np


def simple_model_run(rscorer=False):
    '''Creates data to allow testing of metrics
    Args:
        rscorer (bool): determines whether the function returns the correct
        inputs for the RScoreWrapper (True) or for the metrics (False)
    Returns:
        if rscorer=True:
            input parameters for RScoreWrapper
        if rscorer=False:
            input parameters for metrics functions (such as qini_make_score
    '''
    data_df = synth_ihdp()
    data_df, features_X, features_W, targets, treatment = \
        preprocess_dataset(data_df)
    # data_df = data_df.drop(columns = ["random"])
    outcome = targets[0]
    train_df, test_df = train_test_split(
        data_df, train_size=0.5, random_state=123
    )
    causal_model = CausalModel(
        data=train_df,
        treatment=treatment,
        outcome=outcome,
        common_causes=features_W,
        effect_modifiers=features_X,
        random_state=123,
    )
    identified_estimand = causal_model.identify_effect(
        proceed_when_unidentifiable=True
    )
    estimate = causal_model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.metalearners.SLearner",
        control_value=0,
        treatment_value=1,
        target_units="ate",  # condition used for CATE
        confidence_intervals=False,
        # random_state=123,
        method_params={
            "init_params":
            {
                "overall_model": DecisionTreeRegressor(random_state=123)
            },
            "fit_params": {},
        },)
    te_train = estimate.cate_estimates
    if rscorer:
        return train_df, test_df, outcome, treatment, features_W, features_X
    else:
        return estimate, train_df, te_train


class TestMetrics():
    def test_auc_score(self):
        '''Tests AUC Score is within exceptable range for the test example'''
        assert auc_make_score(*simple_model_run()) == pytest.approx(0.6, 0.05)

    def test_qini_make_score(self):
        '''Tests Qini score is within exceptable range for the test example'''
        assert qini_make_score(*simple_model_run()) == \
            pytest.approx(0.15, 0.05)

    def test_r_make_score(self):
        '''Tests RScorer output value is within exceptable range for the test
        example'''
        rscorer = RScoreWrapper(
            DecisionTreeRegressor(random_state=123),
            DummyClassifier(strategy="prior"),
            *simple_model_run(rscorer=True)
        )
        assert r_make_score(*simple_model_run(), rscorer.train) == \
            pytest.approx(0.05, 0.1)

    def test_make_scores_with_rscorer(self):
        '''Tests make_scores (with rscorer) produces a dictionary of the right
        structure and composition'''
        rscorer = RScoreWrapper(
            DecisionTreeRegressor(random_state=123),
            DummyClassifier(strategy="prior"),
            *simple_model_run(rscorer=True)
        )
        scores = make_scores(*simple_model_run(), rscorer.train)
        true_keys = ["erupt", "norm_erupt", "qini", "auc", "r_score", "ate",
                     "intrp", "values"]
        for i in scores.keys():
            print(i)
            assert i in true_keys
            if i == 'intrp':
                assert isinstance(scores[i], SingleTreeCateInterpreter)
            elif i == 'values':
                assert isinstance(scores[i], pd.DataFrame)
            else:
                assert isinstance(scores[i], np.float64)

    def test_make_scores_without_rscorer(self):
        '''Tests make_scores (without rscorer) returns 0 for 'r_score' key'''
        scores = make_scores(*simple_model_run())
        assert scores['r_score'] == 0


if __name__ == "__main__":
    pytest.main([__file__])
