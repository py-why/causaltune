import pytest
import warnings

from auto_causality.datasets import synth_ihdp

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEndAutoMLPropensity(object):
    """tests autocausality model end-to-end
    1/ import autocausality object
    2/ preprocess data
    3/ init autocausality object
    4/ run autocausality on data
    """

    def test_endtoend(self):
        """tests if model can be instantiated and fit to data"""

        from auto_causality import AutoCausality  # noqa F401
        from auto_causality.shap import shap_values  # noqa F401

        data = synth_ihdp()
        data.preprocess_dataset()

        estimator_list = "all"

        auto_causality = AutoCausality(
            components_time_budget=10,
            estimator_list=estimator_list,
            num_samples=13,
            use_ray=False,
            verbose=4,
            components_verbose=2,
            propensity_model="auto",
            resources_per_trial={"cpu": 0.5},
        )

        auto_causality.fit(data)

        # now let's test Shapley values calculation
        for est_name, scores in auto_causality.scores.items():
            # Dummy model doesn't support Shapley values
            # Orthoforest shapley calc is VERY slow
            if "Dummy" not in est_name and "Ortho" not in est_name:

                print("Calculating Shapley values for", est_name)
                shap_values(scores["estimator"], data.data[:10])

        print(f"Best estimator: {auto_causality.best_estimator}")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEndAutoMLPropensity().test_endtoend()
