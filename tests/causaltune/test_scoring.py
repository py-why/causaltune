import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier

from dowhy import CausalModel

from causaltune.datasets import synth_ihdp
from causaltune.score.scoring import Scorer, supported_metrics


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
    if rscorer:
        return train_df, test_df, data, estimate, scorer
    else:
        return estimate, train_df, te_train


class TestMetrics:
    def test_resolve_metrics_default(self):
        scorer = simple_model_run(rscorer=True)[-1]
        score_metric = scorer.resolve_metric(None)
        assert score_metric == "energy_distance"

        metrics = scorer.resolve_reported_metrics(None, score_metric)
        all_metrics = supported_metrics("backdoor", False, False)

        assert not set(metrics) ^ set(all_metrics)

    def test_resolve_metrics(self):
        scorer = simple_model_run(rscorer=True)[-1]
        score_metric = scorer.resolve_metric("erupt")
        assert score_metric == "erupt"

        metrics = scorer.resolve_reported_metrics(["ate", "garbage"], score_metric)

        assert not set(metrics) ^ set(["ate", "erupt"])

    def test_auc_score(self):
        """Tests AUC Score is within exceptable range for the test example"""
        assert Scorer.auc_make_score(*simple_model_run()) == pytest.approx(0.6, 0.05)

    # TODO: Debug wrong values: 1) QINI
    def test_qini_make_score(self):
        """Tests Qini score is within exceptable range for the test example"""
        assert Scorer.qini_make_score(*simple_model_run()) == pytest.approx(22, 27)

    def test_psw_energy_distance_base_case(self):
        """Tests propensity score weighted energy distance
        equals energy distance when feature normalisation is off, dummy propensity model
        is used and there is a single treatment
        """
        _, test_df, __, estimate, scorer = simple_model_run(rscorer=True)
        assert scorer.psw_energy_distance(
            estimate, test_df, normalise_features=False
        ) == pytest.approx(scorer.energy_distance_score(estimate, test_df))

    def test_psw_energy_distance(self):
        """Test propensity score kernel weighted energy distance"""
        _, test_df, __, estimate, scorer = simple_model_run(rscorer=True)
        scorer.psw_energy_distance(estimate, test_df)

    # TODO: Fix R-scorer or purge it
    # def test_r_make_score(self):
    #     """Tests RScorer output value is within exceptable range for the test
    #     example"""
    #
    #     rscorer = RScoreWrapper(
    #         DecisionTreeRegressor(random_state=123),
    #         DummyClassifier(strategy="prior"),
    #         *simple_model_run(rscorer=True)[:-1]
    #     )
    #     assert Scorer.r_make_score(*simple_model_run(), rscorer.train) == pytest.approx(
    #         0.05, 0.1
    #     )

    # def test_make_scores_with_rscorer(self):
    #     """Tests make_scores (with rscorer) produces a dictionary of the right
    #     structure and composition"""
    #     smr = simple_model_run(rscorer=True)
    #     scorer = smr[-1]
    #
    #     rscorer = RScoreWrapper(
    #         DecisionTreeRegressor(random_state=123),
    #         scorer.psw_estimator.estimator.propensity_model,
    #         *smr[:-1]
    #     )
    #     true_keys = [
    #         "erupt",
    #         "norm_erupt",
    #         "qini",
    #         "auc",
    #         # "r_score",
    #         "ate",
    #         "ate_std",
    #         "intrp",
    #         "values",
    #     ]
    #
    #     scores = scorer.make_scores(
    #         *(simple_model_run()[:2]),
    #         "backdoor",
    #         true_keys[:-2],  # Exclude non-metrics
    #         rscorer.train
    #     )
    #
    #     # TODO: either fix the R-scorer or purge it and rewrite this test to match
    #     scores.pop("r_score")
    #
    #     for i in scores.keys():
    #         assert i in true_keys
    #         if i == "intrp":
    #             assert isinstance(scores[i], SingleTreeCateInterpreter)
    #         elif i == "values":
    #             assert isinstance(scores[i], pd.DataFrame)
    #         else:
    #             assert isinstance(scores[i], np.float64)

    def test_make_scores_without_rscorer(self):
        """Tests make_scores (without rscorer) returns 0 for 'r_score' key"""
        scorer = simple_model_run(rscorer=True)[-1]

        scores = scorer.make_scores(*(simple_model_run()[:2]), ["ate"])
        assert scores.get("r_score", 0) == 0


if __name__ == "__main__":
    pytest.main([__file__])
    # TestMetrics().test_make_scores_without_rscorer()
