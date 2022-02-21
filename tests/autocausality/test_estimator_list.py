import pytest
import os
import sys
from flaml import AutoML
root_path = root_path = os.path.realpath("../../..")
sys.path.append(os.path.join(root_path, "auto-causality"))
sys.path.append(os.path.join(root_path, "dowhy"))
from auto_causality import AutoCausality # noqa E402
from auto_causality.params import SimpleParamService # noqa E402


class TestEstimatorListGenerator:
    """tests if estimator list is correctly generated
    """

    def test_auto_list(self):
        """tests if "auto" setting yields all available estimators"""
        autocausality = AutoCausality(estimator_list="auto")
        estimator_list = autocausality.get_estimators()
        cfg = SimpleParamService(propensity_model=AutoML(), outcome_model=AutoML())
        available_estimators = cfg.estimators()
        # verify that returned estimator list includes all available estimators
        assert all(e in available_estimators for e in estimator_list)

    def test_empty_list(self):
        """ tests if empty list is correctly handled"""
        autocausality = AutoCausality(estimator_list=[])
        estimator_list = autocausality.get_estimators()
        cfg = SimpleParamService(propensity_model=AutoML(), outcome_model=AutoML())
        available_estimators = cfg.estimators()
        assert all(e in available_estimators for e in estimator_list)

    def test_substring_group(self):
        """ tests if substring match to group of estimators works"""
        autocausality = AutoCausality(estimator_list=["dml"])
        estimator_list = autocausality.get_estimators()
        cfg = SimpleParamService(propensity_model=AutoML(), outcome_model=AutoML())
        available_estimators = [e for e in cfg.estimators() if "dml" in e]
        # verify that returned estimator list includes all available estimators
        assert all(e in available_estimators for e in estimator_list)
        # or all econml models:
        autocausality = AutoCausality(estimator_list=["econml"])
        estimator_list = autocausality.get_estimators()
        cfg = SimpleParamService(propensity_model=AutoML(), outcome_model=AutoML())
        available_estimators = [e for e in cfg.estimators() if "econml" in e]
        # verify that returned estimator list includes all available estimators
        assert all(e in available_estimators for e in estimator_list)

    def test_substring_single(self):
        """tests if substring match to single estimators works"""
        autocausality = AutoCausality(estimator_list=["DomainAdaptationLearner"])
        estimator_list = autocausality.get_estimators()
        assert estimator_list == [
            "backdoor.econml.metalearners.DomainAdaptationLearner"
        ]

    def test_checkduplicates(self):
        """tests if duplicates are removed"""
        autocausality = AutoCausality(
            estimator_list=[
                "DomainAdaptationLearner",
                "DomainAdaptationLearner",
                "DomainAdaptationLearner",
            ]
        )
        estimator_list = autocausality.get_estimators()
        assert len(estimator_list) == 1

    def test_invalid_choice(self):
        """tests if invalid choices are handled correctly"""
        # this should revert back to defaults (all available estimators)
        # unavailable estimators:
        autocausality = AutoCausality(estimator_list=["linear_regression", "pasta", 12])
        estimator_list = autocausality.get_estimators()
        cfg = SimpleParamService(propensity_model=AutoML(), outcome_model=AutoML())
        available_estimators = cfg.estimators()
        assert all(e in available_estimators for e in estimator_list)

        # empty string
        autocausality = AutoCausality(estimator_list=[""])
        estimator_list = autocausality.get_estimators()
        cfg = SimpleParamService(propensity_model=AutoML(), outcome_model=AutoML())
        available_estimators = cfg.estimators()
        assert all(e in available_estimators for e in estimator_list)

        autocausality = AutoCausality(estimator_list="")
        estimator_list = autocausality.get_estimators()
        cfg = SimpleParamService(propensity_model=AutoML(), outcome_model=AutoML())
        available_estimators = cfg.estimators()
        assert all(e in available_estimators for e in estimator_list)

        # test wrong dtype:
        autocausality = AutoCausality(estimator_list=5)
        estimator_list = autocausality.get_estimators()
        cfg = SimpleParamService(propensity_model=AutoML(), outcome_model=AutoML())
        available_estimators = cfg.estimators()
        assert all(e in available_estimators for e in estimator_list)


if __name__ == "__main__":
    pytest.main([__file__])
