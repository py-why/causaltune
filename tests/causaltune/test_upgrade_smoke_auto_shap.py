"""
End-to-end smoke tests on the upgraded stack (Codex review finding #7).

Exercises the surfaces a pure version bump can silently break but the focused
unit tests do not cover directly:

  * ``CausalTune.fit(..., outcome_model="auto")`` -> the FLAML component-model
    construction path in ``causaltune/search/component.py`` on FLAML 2.6;
  * ``.effect`` / ``.score_dataset`` scoring on the target stack;
  * the ``causaltune.shap`` adapter against econml 0.16;
  * the Ray-backed tuning path (marked slow).

Kept small (few estimators, tiny budget) so the non-Ray test runs in the
non-slow suite.
"""
import numpy as np
import pytest

from causaltune import CausalTune
from causaltune.datasets import generate_non_random_dataset
from causaltune.shap import shap_values

ESTIMATORS = ["backdoor.econml.dml.LinearDML", "backdoor.econml.metalearners.SLearner"]


def _fit(use_ray):
    data = generate_non_random_dataset(num_samples=500)
    data.preprocess_dataset()

    ct = CausalTune(
        num_samples=len(ESTIMATORS),
        components_time_budget=10,
        estimator_list=ESTIMATORS,
        use_ray=use_ray,
        verbose=1,
        components_verbose=1,
        resources_per_trial={"cpu": 0.5},
        outcome_model="auto",
    )
    ct.fit(data)
    return ct, data


def test_fit_auto_outcome_and_shap():
    ct, data = _fit(use_ray=False)

    eff = ct.effect(data.data)
    assert len(eff) == len(data.data)
    assert np.all(np.isfinite(np.asarray(eff)))

    ct.score_dataset(data.data, "smoke_test")
    assert ct.best_estimator in ESTIMATORS

    # SHAP adapter on the target stack for a fitted econml estimator
    checked = False
    for est_name, scores in ct.scores.items():
        if "Dummy" in est_name or "Ortho" in est_name:
            continue
        sv = shap_values(scores["estimator"], data.data[:5])
        assert sv is not None
        checked = True
        break
    assert checked, "no non-Dummy/Ortho estimator was available to check SHAP"


@pytest.mark.slow
def test_fit_with_ray():
    """Ray-backed tuning must work on the bumped ray (>=2.9) / Python 3.12."""
    ray = pytest.importorskip("ray")
    ct, data = _fit(use_ray=True)

    eff = ct.effect(data.data)
    assert len(eff) == len(data.data)
    assert np.all(np.isfinite(np.asarray(eff)))
    assert ct.best_estimator in ESTIMATORS

    if ray.is_initialized():
        ray.shutdown()
