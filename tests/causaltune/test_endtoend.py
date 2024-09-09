import warnings


from causaltune import CausalTune
from causaltune.datasets import generate_non_random_dataset, linear_multi_dataset
from causaltune.search.params import SimpleParamService

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEnd(object):
    """tests causaltune model end-to-end
    1/ import causaltune object
    2/ preprocess data
    3/ init causaltune object
    4/ run causaltune on data
    """

    def test_imports(self):
        """tests if CausalTune can be imported"""

        from causaltune import CausalTune  # noqa F401

    def test_data_preprocessing(self):
        """tests data preprocessing routines"""
        data = generate_non_random_dataset()  # noqa F484

    def test_init_causaltune(self):
        """tests if causaltune object can be instantiated without errors"""

        from causaltune import CausalTune  # noqa F401

        ct = CausalTune(time_budget=0)  # noqa F484

    def test_endtoend_cate(self):
        """tests if CATE model can be instantiated and fit to data"""

        from causaltune.shap import shap_values  # noqa F401

        data = generate_non_random_dataset()
        data.preprocess_dataset()

        cfg = SimpleParamService(
            n_jobs=-1,
            include_experimental=False,
            multivalue=False,
        )
        estimator_list = cfg.estimator_names_from_patterns("backdoor", "all", 1)
        # outcome = targets[0]
        ct = CausalTune(
            num_samples=len(estimator_list),
            components_time_budget=10,
            estimator_list=estimator_list,  # "all",  #
            use_ray=False,
            verbose=3,
            components_verbose=2,
            resources_per_trial={"cpu": 0.5},
            outcome_model="auto",
        )

        ct.fit(data)
        # ct.fit(data, resume=True)
        ct.effect(data.data)
        ct.score_dataset(data.data, "test")

        # now let's test Shapley ct calculation
        for est_name, scores in ct.scores.items():
            # Dummy model doesn't support Shapley values
            # Orthoforest shapley calc is VERY slow
            if "Dummy" not in est_name and "Ortho" not in est_name:
                print("Calculating Shapley values for", est_name)
                shap_values(scores["estimator"], data.data[:10])

        print(f"Best estimator: {ct.best_estimator}")

    def test_endtoend_multivalue(self):
        data = linear_multi_dataset(5000)
        cfg = SimpleParamService(
            n_jobs=-1,
            include_experimental=False,
            multivalue=True,
        )
        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "all", data_rows=len(data)
        )

        ct = CausalTune(
            estimator_list="all",
            num_samples=len(estimator_list),
            components_time_budget=10,
            outcome_model="auto",
        )
        ct.fit(data)
        # ct.fit(data, resume=True)

        # TODO add an effect() call and an effect_tt call
        print("yay!")


if __name__ == "__main__":
    TestEndToEnd().test_endtoend_cate()
    # pytest.main([__file__])
    # TestEndToEnd().test_endtoend_iv()
