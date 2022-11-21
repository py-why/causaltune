import pytest
import warnings

from auto_causality.datasets import generate_synthetic_data
from auto_causality.models.passthrough import Passthrough

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEndPassthrough(object):
    """tests autocausality model end-to-end
    1/ import autocausality object
    2/ preprocess data
    3/ init autocausality object
    4/ run autocausality on data
    """

    def test_endtoend(self):
        """tests if model can be instantiated and fit to data"""

        from auto_causality import AutoCausality  # noqa F401

        data = generate_synthetic_data(
            n_samples=1000,
            confounding=True,
            linear_confounder=False,
            noisy_outcomes=True,
        )

        data.preprocess_dataset()

        auto_causality = AutoCausality(
            components_time_budget=10,
            estimator_list="all",
            num_samples=13,
            use_ray=False,
            verbose=4,
            components_verbose=2,
            propensity_model=Passthrough("propensity"),
            resources_per_trial={"cpu": 0.5},
        )

        auto_causality.fit(data)

        print(f"Best estimator: {auto_causality.best_estimator}")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEndAutoMLPropensity().test_endtoend()
