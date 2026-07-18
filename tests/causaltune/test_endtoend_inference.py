import pytest
import warnings

from causaltune import CausalTune
from causaltune.datasets import linear_multi_dataset
from causaltune.search.params import SimpleParamService

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..

# Component-model HPO is capped hard: this is an interface/smoke test on tiny
# synthetic data, so a 1s component budget exercises every code path
# (fit -> effect_stderr -> score) without idling to fill a wall-clock budget.
# components_njobs=1 keeps each fit single-worker so xdist can run the (now
# parametrized) estimators concurrently instead of oversubscribing cores.
COMPONENTS_TIME_BUDGET = 1
NUM_SAMPLES = 4

# Enumerate the estimator lists at *collection* time so each estimator becomes
# its own test case. The old `for e in estimator_list: fit()` loop ran them
# serially inside one test (invisible to xdist); parametrizing lets xdist fan
# them out across workers. Same estimators, same surface -- just parallelizable.
_singlevalue_cfg = SimpleParamService(
    n_jobs=1, include_experimental=False, multivalue=False
)
_multivalue_cfg = SimpleParamService(
    n_jobs=1, include_experimental=False, multivalue=True
)
# linear_multi_dataset(1000) has 1000 rows; the cheap_inference pattern depends
# only on the row count, so the list matches what each fit's data would produce.
CHEAP_INFERENCE_SINGLE = _singlevalue_cfg.estimator_names_from_patterns(
    "backdoor", "cheap_inference", 1000
)
CHEAP_INFERENCE_MULTI = _multivalue_cfg.estimator_names_from_patterns(
    "backdoor", "cheap_inference", 1000
)


@pytest.fixture(scope="module")
def singlevalue_data():
    data = linear_multi_dataset(1000, impact={0: 0.0, 1: 2.0})
    data.preprocess_dataset()
    return data


@pytest.fixture(scope="module")
def multivalue_data():
    return linear_multi_dataset(1000)


def _make_ct(estimator):
    return CausalTune(
        num_samples=NUM_SAMPLES,
        components_time_budget=COMPONENTS_TIME_BUDGET,
        components_njobs=1,
        estimator_list=[estimator],
        use_ray=False,
        verbose=3,
        components_verbose=2,
        resources_per_trial={"cpu": 0.5},
        outcome_model="auto",
    )


class TestEndToEndInference(object):
    """tests confidence interval generation"""

    @pytest.mark.parametrize("estimator", CHEAP_INFERENCE_SINGLE)
    def test_endtoend_inference_nobootstrap(self, singlevalue_data, estimator):
        """CATE model fits and produces confidence intervals per estimator."""
        causaltune = _make_ct(estimator)
        causaltune.fit(singlevalue_data)
        causaltune.effect_stderr(singlevalue_data.data)
        causaltune.score_dataset(singlevalue_data.data, "test")

    def test_endtoend_inference_bootstrap(self, multivalue_data):
        causaltune = _make_ct("SLearner")
        causaltune.fit(multivalue_data)
        causaltune.effect_stderr(multivalue_data.data)

    @pytest.mark.parametrize("estimator", CHEAP_INFERENCE_MULTI)
    def test_endtoend_multivalue_nobootstrap(self, multivalue_data, estimator):
        causaltune = _make_ct(estimator)
        causaltune.fit(multivalue_data)
        causaltune.effect_stderr(multivalue_data.data)
        causaltune.effect(multivalue_data.data)
        scores = causaltune.score_dataset(multivalue_data.data, "test")
        print(scores)

    def test_endtoend_multivalue_bootstrap(self, multivalue_data):
        causaltune = _make_ct("SLearner")
        causaltune.fit(multivalue_data)
        tmp = causaltune.effect_stderr(multivalue_data.data)  # noqa F841


if __name__ == "__main__":
    pytest.main([__file__])
