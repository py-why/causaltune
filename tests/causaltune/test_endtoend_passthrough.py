import pytest
import warnings

from causaltune.datasets import generate_synthetic_data, linear_multi_dataset
from causaltune.models.passthrough import passthrough_model

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEndPassthrough(object):
    """tests causaltune model end-to-end
    1/ import causaltune object
    2/ preprocess data
    3/ init causaltune object
    4/ run causaltune on data
    """

    def test_endtoend_passthrough(self):
        """tests if model can be instantiated and fit to data"""

        from causaltune import CausalTune  # noqa F401

        data = generate_synthetic_data(
            n_samples=1000,
            confounding=True,
            linear_confounder=False,
            noisy_outcomes=True,
        )

        data.preprocess_dataset()

        causaltune = CausalTune(
            components_time_budget=10,
            estimator_list=[".LinearDML"],
            num_samples=1,
            use_ray=False,
            verbose=4,
            components_verbose=2,
            propensity_model=passthrough_model("propensity"),
            resources_per_trial={"cpu": 0.5},
            outcome_model="auto",
        )

        causaltune.fit(data)

        print(f"Best estimator: {causaltune.best_estimator}")

    def test_endtoend_passthrough_multivalue(self):
        """tests if model can be instantiated and fit to data"""

        from causaltune import CausalTune  # noqa F401

        for include_control in [True, False]:
            data = linear_multi_dataset(
                10000, include_propensity=True, include_control=include_control
            )

            data.preprocess_dataset()

            causaltune = CausalTune(
                components_time_budget=10,
                estimator_list=[".LinearDML"],
                num_samples=1,
                use_ray=False,
                verbose=4,
                components_verbose=2,
                propensity_model=passthrough_model(
                    data.propensity_modifiers, include_control=include_control
                ),
                resources_per_trial={"cpu": 0.5},
                outcome_model="auto",
            )

            causaltune.fit(data)

            print(f"Best estimator: {causaltune.best_estimator}")


if __name__ == "__main__":
    pytest.main([__file__])
