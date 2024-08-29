import pytest
import warnings

from sklearn.linear_model import LinearRegression

from causaltune import CausalTune
from causaltune.datasets import linear_multi_dataset, generate_synthetic_data
from causaltune.search.params import SimpleParamService

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestCustomOutputModel(object):
    def test_custom_outcome_model(self):
        """tests if CATE model can be instantiated and fit to data"""

        from causaltune.shap import shap_values  # noqa F401

        data = generate_synthetic_data(n_samples=5000)
        data.preprocess_dataset()

        cfg = SimpleParamService(
            n_jobs=-1,
            include_experimental=False,
            multivalue=False,
        )
        estimator_list = cfg.estimator_names_from_patterns("backdoor", "all", 1)
        # outcome = targets[0]
        causaltune = CausalTune(
            outcome_model=LinearRegression(),
            num_samples=len(estimator_list),
            components_time_budget=10,
            estimator_list=estimator_list,  # "all",  #
            use_ray=False,
            verbose=3,
            components_verbose=2,
            resources_per_trial={"cpu": 0.5},
        )

        causaltune.fit(data)
        causaltune.effect(data.data)
        causaltune.score_dataset(data.data, "test")

        # now let's test Shapley values calculation
        for est_name, scores in causaltune.scores.items():
            # Dummy model doesn't support Shapley values
            # Orthoforest shapley calc is VERY slow
            if "Dummy" not in est_name and "Ortho" not in est_name:
                print("Calculating Shapley values for", est_name)
                shap_values(scores["estimator"], data.data[:10])

        print(f"Best estimator: {causaltune.best_estimator}")

    def test_custom_outcome_model_multivalue(self):
        data = linear_multi_dataset(10000)
        cfg = SimpleParamService(
            n_jobs=-1,
            include_experimental=False,
            multivalue=True,
        )
        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "all", data_rows=len(data)
        )

        ct = CausalTune(
            outcome_model=LinearRegression(),
            estimator_list="all",
            num_samples=len(estimator_list),
            components_time_budget=10,
        )
        ct.fit(data)
        # TODO add an effect() call and an effect_tt call
        print("yay!")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEnd().test_endtoend_iv()
