import pytest
import warnings


from auto_causality import AutoCausality
from auto_causality.datasets import synth_ihdp, linear_multi_dataset
from auto_causality.params import SimpleParamService

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEnd(object):
    """tests autocausality model end-to-end
    1/ import autocausality object
    2/ preprocess data
    3/ init autocausality object
    4/ run autocausality on data
    """

    def test_imports(self):
        """tests if AutoCausality can be imported"""

        from auto_causality import AutoCausality  # noqa F401

    def test_data_preprocessing(self):
        """tests data preprocessing routines"""
        data = synth_ihdp()  # noqa F484

    def test_init_autocausality(self):
        """tests if autocausality object can be instantiated without errors"""

        from auto_causality import AutoCausality  # noqa F401

        auto_causality = AutoCausality(time_budget=0)  # noqa F484

    def test_endtoend_cate(self):
        """tests if CATE model can be instantiated and fit to data"""

        from auto_causality.shap import shap_values  # noqa F401

        data = synth_ihdp()
        data.preprocess_dataset()

        cfg = SimpleParamService(
            propensity_model=None,
            outcome_model=None,
            n_jobs=-1,
            include_experimental=False,
            multivalue=False,
        )
        estimator_list = cfg.estimator_names_from_patterns("backdoor", "all", 1)
        # outcome = targets[0]
        auto_causality = AutoCausality(
            num_samples=len(estimator_list),
            components_time_budget=10,
            estimator_list=estimator_list,  # "all",  #
            use_ray=False,
            verbose=3,
            components_verbose=2,
            resources_per_trial={"cpu": 0.5},
        )

        auto_causality.fit(data)
        auto_causality.effect(data.data)
        auto_causality.score_dataset(data.data, "test")

        # now let's test Shapley values calculation
        for est_name, scores in auto_causality.scores.items():
            # Dummy model doesn't support Shapley values
            # Orthoforest shapley calc is VERY slow
            if "Dummy" not in est_name and "Ortho" not in est_name:

                print("Calculating Shapley values for", est_name)
                shap_values(scores["estimator"], data.data[:10])

        print(f"Best estimator: {auto_causality.best_estimator}")

    def test_endtoend_multivalue(self):
        data = linear_multi_dataset(10000)
        cfg = SimpleParamService(
            propensity_model=None,
            outcome_model=None,
            n_jobs=-1,
            include_experimental=False,
            multivalue=True,
        )
        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "all", data_rows=len(data)
        )

        ac = AutoCausality(
            estimator_list="all",
            num_samples=len(estimator_list),
            components_time_budget=10,
        )
        ac.fit(data)
        # TODO add an effect() call and an effect_tt call
        print("yay!")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEnd().test_endtoend_iv()
