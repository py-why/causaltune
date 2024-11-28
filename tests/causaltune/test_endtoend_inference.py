import pytest
import warnings

from econml.inference import BootstrapInference

from causaltune import CausalTune
from causaltune.datasets import linear_multi_dataset
from causaltune.search.params import SimpleParamService

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEndInference(object):
    """
    tests confidence interval generation
    """

    def test_endtoend_inference_nobootstrap(self):
        """tests if CATE model can be instantiated and fit to data"""
        data = linear_multi_dataset(1000, impact={0: 0.0, 1: 2.0})
        data.preprocess_dataset()

        cfg = SimpleParamService(
            n_jobs=-1,
            include_experimental=False,
            multivalue=False,
        )

        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "cheap_inference", len(data.data)
        )
        print("estimators: ", estimator_list)

        for e in estimator_list:
            print(e)
            causaltune = CausalTune(
                num_samples=4,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
                outcome_model="auto",
            )

            causaltune.fit(data)
            causaltune.effect_stderr(data.data)
            causaltune.score_dataset(data.data, "test")

    def test_endtoend_inference_bootstrap(self):
        """tests if CATE model can be instantiated and fit to data"""
        data = linear_multi_dataset(1000)
        data.preprocess_dataset()

        BootstrapInference(n_bootstrap_samples=10, n_jobs=10)
        estimator_list = ["SLearner"]

        for e in estimator_list:
            causaltune = CausalTune(
                num_samples=4,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
                outcome_model="auto",
            )

            causaltune.fit(data)
            causaltune.effect_stderr(data.data)

    def test_endtoend_multivalue_nobootstrap(self):
        data = linear_multi_dataset(1000)
        cfg = SimpleParamService(
            n_jobs=-1,
            include_experimental=False,
            multivalue=True,
        )

        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "cheap_inference", len(data.data)
        )

        for e in estimator_list:
            causaltune = CausalTune(
                num_samples=4,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
                outcome_model="auto",
            )

            causaltune.fit(data)
            causaltune.effect_stderr(data.data)
            causaltune.effect(data.data)
            scores = causaltune.score_dataset(data.data, "test")
            print(scores)
        # TODO add an effect() call and an effect_tt call
        print("yay!")

    def test_endtoend_multivalue_bootstrap(self):
        data = linear_multi_dataset(1000)

        estimator_list = ["SLearner"]

        for e in estimator_list:
            causaltune = CausalTune(
                num_samples=4,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                outcome_model="auto",
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
            )

            causaltune.fit(data)
            tmp = causaltune.effect_stderr(data.data)  # noqa F841

        # TODO add an effect() call and an effect_tt call
        print("yay!")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEndInference().test_endtoend_multivalue_bootstrap()
