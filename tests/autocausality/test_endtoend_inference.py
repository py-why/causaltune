import pytest
import warnings


from auto_causality import AutoCausality
from auto_causality.datasets import synth_ihdp, linear_multi_dataset
from auto_causality.params import SimpleParamService

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEndInference(object):
    """
    tests confidence interval generation
    """

    def test_endtoend_inference_nobootstrap(self):
        """tests if CATE model can be instantiated and fit to data"""
        data = synth_ihdp()
        data.preprocess_dataset()

        cfg = SimpleParamService(
            propensity_model=None,
            outcome_model=None,
            n_jobs=-1,
            include_experimental=False,
            multivalue=False,
        )

        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "cheap_inference", len(data.data)
        )

        for e in estimator_list:
            auto_causality = AutoCausality(
                num_samples=1,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
            )

            auto_causality.fit(data)
            auto_causality.effect_stderr(data.data)

    def test_endtoend_multivalue_nobootstrap(self):
        data = linear_multi_dataset(1000)
        cfg = SimpleParamService(
            propensity_model=None,
            outcome_model=None,
            n_jobs=-1,
            include_experimental=False,
            multivalue=True,
        )

        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "cheap_inference", len(data.data)
        )

        for e in estimator_list:
            auto_causality = AutoCausality(
                num_samples=1,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
            )

            auto_causality.fit(data)
            auto_causality.effect_stderr(data.data)

        # TODO add an effect() call and an effect_tt call
        print("yay!")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEnd().test_endtoend_iv()
