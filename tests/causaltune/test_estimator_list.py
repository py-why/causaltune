import pytest
import pandas as pd

from causaltune import CausalTune
from causaltune.search.params import SimpleParamService


class TestEstimatorListGenerator:
    """tests if estimator list is correctly generated"""

    def test_auto_list(self):
        """tests if "auto" setting yields all available estimators"""
        cfg = SimpleParamService(multivalue=False)
        auto_estimators_iv = cfg.estimator_names_from_patterns("iv", "auto")
        auto_estimators_backdoor = cfg.estimator_names_from_patterns("backdoor", "auto")
        # verify that returned estimator list includes all available estimators
        # for backdoor or iv problem(s)
        assert len(auto_estimators_backdoor) == 7
        assert len(auto_estimators_iv) == 5

    # def test_all_list(self):
    #     """tests if "auto" setting yields all available estimators"""
    #     problems = ["iv", "backdoor"]
    #     for p in problems:
    #         cfg = SimpleParamService(
    #             propensity_model=None, outcome_model=None, include_experimental=True
    #         )
    #         all_estimators = cfg.estimator_names_from_patterns(p, "all", data_rows=1)
    #         # verify that returned estimator list includes all available estimators
    #         assert len(all_estimators) == len(cfg._configs())
    #
    #         cfg = SimpleParamService(
    #             propensity_model=None, outcome_model=None, include_experimental=True
    #         )
    #         all_estimators = cfg.estimator_names_from_patterns(p, "all", data_rows=10000)
    #         # verify that returned estimator list includes all available estimators
    #         assert len(all_estimators) == len(cfg._configs()) - 2

    def test_substring_group(self):
        """tests if substring match to group of estimators works"""
        cfg = SimpleParamService(multivalue=False)

        estimator_list = cfg.estimator_names_from_patterns("backdoor", ["dml"])
        available_estimators = [e for e in cfg._configs().keys() if "dml" in e]
        # verify that returned estimator list includes all available estimators
        assert all(e in available_estimators for e in estimator_list)

        # or all econml models:
        estimator_list = cfg.estimator_names_from_patterns("backdoor", ["econml"])
        available_estimators = [e for e in cfg._configs().keys() if "econml" in e]
        # verify that returned estimator list includes all available estimators
        assert all(e in available_estimators for e in estimator_list)

    def test_substring_single(self):
        """tests if substring match to single estimators works"""
        cfg = SimpleParamService(multivalue=False)
        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", ["DomainAdaptationLearner"]
        )
        assert estimator_list == [
            "backdoor.econml.metalearners.DomainAdaptationLearner"
        ]

    def test_checkduplicates(self):
        """tests if duplicates are removed"""
        cfg = SimpleParamService(multivalue=False)
        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor",
            [
                "DomainAdaptationLearner",
                "DomainAdaptationLearner",
                "DomainAdaptationLearner",
            ],
        )
        assert len(estimator_list) == 1

    def test_invalid_choice(self):
        """tests if invalid choices are handled correctly"""
        # this should raise a ValueError
        # unavailable estimators:

        cfg = SimpleParamService(multivalue=False)

        with pytest.raises(ValueError):
            cfg.estimator_names_from_patterns(
                "backdoor", ["linear_regression", "pasta", 12]
            )

        with pytest.raises(ValueError):
            cfg.estimator_names_from_patterns("backdoor", 5)

    def test_invalid_choice_fitter(self):
        with pytest.raises(AssertionError):
            """tests if empty list is correctly handled"""
            ct = CausalTune(components_time_budget=10, outcome_model="auto")
            ct.fit(
                pd.DataFrame(
                    {"treatment": [0, 1], "outcome": [0.5, 1.5], "dummy": [0.1, 0.2]}
                ),
                treatment="treatment",
                outcome="outcome",
                common_causes=["dummy"],
                effect_modifiers=[],
                estimator_list=[],
            )


if __name__ == "__main__":
    pytest.main([__file__])
