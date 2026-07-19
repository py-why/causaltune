import warnings

import pytest

from causaltune import CausalTune
from causaltune.datasets import iv_dgp_econml
from causaltune.search.params import SimpleParamService

warnings.filterwarnings("ignore")

# Enumerate the IV estimators at collection time and parametrize over them, one
# per test case. The old single fit used num_samples=-1 + time_budget=1000 (a
# literal 1000s optuna budget) to be sure every IV estimator got exercised; with
# outcome_model="auto" the warm-start seeds ~3 configs/estimator, so a small
# num_samples on a single joint fit would only ever reach the first couple of
# estimators. Forcing estimator_list=[e] per case guarantees full coverage AND
# lets xdist run them concurrently.
_iv_data = iv_dgp_econml()
IV_ESTIMATORS = SimpleParamService(
    n_jobs=1, include_experimental=False, multivalue=False
).estimator_names_from_patterns("iv", "all", len(_iv_data.data))


@pytest.fixture(scope="module")
def iv_data():
    data = iv_dgp_econml()
    data.preprocess_dataset()
    return data


class TestEndToEnd(object):
    @pytest.mark.parametrize("estimator", IV_ESTIMATORS)
    def test_endtoend_iv(self, iv_data, estimator):
        causaltune = CausalTune(
            num_samples=4,
            time_budget=60,
            components_time_budget=2,
            estimator_list=[estimator],
            propensity_model="auto",
            resources_per_trial={"cpu": 0.5},
            use_ray=False,
            verbose=3,
            components_verbose=2,
            outcome_model="auto",
        )

        causaltune.fit(iv_data)

        assert estimator in causaltune.estimator_list
        for est_name, scores in causaltune.scores.items():
            assert est_name in causaltune.estimator_list
