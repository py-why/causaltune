import warnings
from auto_causality import AutoCausality
from auto_causality.datasets import synth_ihdp, iv_dgp_econml
from auto_causality.data_utils import preprocess_dataset

warnings.filterwarnings("ignore")


class TestEndToEnd(object):
    def test_endtoend_iv(self):
        data = iv_dgp_econml()
        treatment = data.treatment
        targets = data.outcomes
        instruments = data.instruments
        data_df, features_X, features_W = preprocess_dataset(
            data.data, treatment, targets, instruments
        )
        outcome = targets[0]
        auto_causality = AutoCausality(
            time_budget=1000,
            components_time_budget=10,
            propensity_model="auto",
            resources_per_trial={"cpu": 0.5},
            use_ray=False,
            verbose=3,
            components_verbose=2,
        )

        auto_causality.fit(
            data_df, treatment, outcome, features_W, features_X, instruments
        )

        for est_name, scores in auto_causality.scores.items():
            assert est_name in auto_causality.estimator_list
