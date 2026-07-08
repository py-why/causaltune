"""Tests for the pluggable HPO backends (flaml / hyperopt / optuna) that replace
the old FLAML-only ``tune.run`` surface, ported from PR #337.

These cover:
* ``SimpleParamService.parse_tuner_params`` mapping for each framework, incl.
  the ``num_samples == -1 -> None`` rule, the hyperopt ``algo`` default, exact
  key sets (no FLAML-only kwargs leaking into optuna/hyperopt), the
  unsupported-framework ``ValueError`` and the unbounded-budget guard.
* ``SimpleParamService.search_space`` now returning a ``hiertunehub.SearchSpace``
  that round-trips back to the original FLAML ``{"estimator": choice([...])}``.
* ``Scorer.best_score_by_estimator`` accepting a *list* of result dicts (and
  raising ``ValueError`` -- not ``NameError`` -- on a malformed entry).
* End-to-end ``CausalTune.fit(framework=...)`` for all three backends, asserting
  the tuner is a real ``hiertunehub`` Tuner and the public properties are
  repointed to it.
* The new default backend being optuna, and a custom optuna sampler being honored.
* FLAML-option parity on the ``framework="flaml"`` path: ``cost_attr``,
  ``low_cost_partial_config``, ``points_to_evaluate`` (from ``try_init_configs``,
  possibly ``[]``) and ``evaluated_rewards`` (``[]`` on a fresh run, rebuilt from
  the *previous* tuner's results on ``resume=True``) actually reaching
  ``flaml.tune.run``.
* Best-effort behaviour elsewhere: ``resume=True`` raises ``NotImplementedError``
  on optuna/hyperopt, while the constructor-default ``try_init_configs`` only warns.
* The packaging decision (hiertunehub + optuna hard deps; hyperopt an extra).
"""

import pathlib

import pytest

from causaltune import CausalTune
from causaltune.datasets import synth_ihdp
from causaltune.search.params import SimpleParamService
from causaltune.score.scoring import Scorer

# A single cheap estimator keeps the end-to-end fits fast. LinearDML carries a
# small tunable search space, which every backend (incl. hyperopt) can handle --
# hiertunehub 0.2.1's to_hyperopt() NameErrors on an all-parameterless choice.
CHEAP_ESTIMATORS = ["LinearDML"]


@pytest.fixture(scope="module")
def data():
    cd = synth_ihdp()
    cd.preprocess_dataset()
    return cd


def _make_ct(**overrides):
    kwargs = dict(
        metric="energy_distance",
        estimator_list=CHEAP_ESTIMATORS,
        num_samples=2,
        components_time_budget=3,
        use_ray=False,
    )
    kwargs.update(overrides)
    return CausalTune(**kwargs)


def _base_tuner_settings(**overrides):
    settings = {
        "num_samples": 5,
        "time_budget_s": 10,
        "verbose": 1,
        "resources_per_trial": {"cpu": 0.5},
        "algo": None,
    }
    settings.update(overrides)
    return settings


# --------------------------------------------------------------------------- #
# parse_tuner_params
# --------------------------------------------------------------------------- #
def test_parse_tuner_params_flaml():
    out = SimpleParamService.parse_tuner_params(_base_tuner_settings(), "flaml")
    assert out["num_samples"] == 5
    assert out["time_budget_s"] == 10
    assert out["verbose"] == 1
    assert out["resources_per_trial"] == {"cpu": 0.5}
    assert out["search_alg"] is None


def test_parse_tuner_params_optuna():
    out = SimpleParamService.parse_tuner_params(_base_tuner_settings(), "optuna")
    assert out["n_trials"] == 5
    assert out["timeout"] == 10
    assert out["sampler"] is None


def test_parse_tuner_params_optuna_keys_exact():
    # optuna must not receive FLAML-only kwargs (search_alg/resources_per_trial/verbose)
    out = SimpleParamService.parse_tuner_params(_base_tuner_settings(), "optuna")
    assert set(out) == {"n_trials", "timeout", "sampler"}


def test_parse_tuner_params_optuna_num_samples_minus_one_is_none():
    out = SimpleParamService.parse_tuner_params(
        _base_tuner_settings(num_samples=-1), "optuna"
    )
    assert out["n_trials"] is None
    # still bounded by the time budget, so no error
    assert out["timeout"] == 10


def test_parse_tuner_params_optuna_custom_sampler_passthrough():
    optuna = pytest.importorskip("optuna")
    sampler = optuna.samplers.RandomSampler(seed=0)
    out = SimpleParamService.parse_tuner_params(
        _base_tuner_settings(algo=sampler), "optuna"
    )
    assert out["sampler"] is sampler


def test_parse_tuner_params_hyperopt():
    hyperopt = pytest.importorskip("hyperopt")
    out = SimpleParamService.parse_tuner_params(_base_tuner_settings(), "hyperopt")
    assert out["max_evals"] == 5
    assert out["timeout"] == 10
    # algo defaults to TPE when the user passes None
    assert out["algo"] is hyperopt.tpe.suggest


def test_parse_tuner_params_hyperopt_keys_exact():
    pytest.importorskip("hyperopt")
    out = SimpleParamService.parse_tuner_params(_base_tuner_settings(), "hyperopt")
    assert set(out) == {"max_evals", "timeout", "verbose", "algo"}


def test_parse_tuner_params_hyperopt_num_samples_minus_one_is_none():
    pytest.importorskip("hyperopt")
    out = SimpleParamService.parse_tuner_params(
        _base_tuner_settings(num_samples=-1), "hyperopt"
    )
    assert out["max_evals"] is None


def test_parse_tuner_params_hyperopt_custom_algo_passthrough():
    hyperopt = pytest.importorskip("hyperopt")
    out = SimpleParamService.parse_tuner_params(
        _base_tuner_settings(algo=hyperopt.rand.suggest), "hyperopt"
    )
    assert out["algo"] is hyperopt.rand.suggest


def test_parse_tuner_params_unsupported_framework():
    with pytest.raises(ValueError):
        SimpleParamService.parse_tuner_params(_base_tuner_settings(), "nope")


def test_parse_tuner_params_optuna_unbounded_raises():
    # both n_trials (num_samples == -1) and timeout (time_budget_s is None) unset
    with pytest.raises(ValueError):
        SimpleParamService.parse_tuner_params(
            _base_tuner_settings(num_samples=-1, time_budget_s=None), "optuna"
        )


def test_parse_tuner_params_hyperopt_unbounded_raises():
    pytest.importorskip("hyperopt")
    with pytest.raises(ValueError):
        SimpleParamService.parse_tuner_params(
            _base_tuner_settings(num_samples=-1, time_budget_s=None), "hyperopt"
        )


def test_parse_tuner_params_flaml_unbounded_ok():
    # FLAML tolerates num_samples == -1 with no time budget; must NOT raise.
    out = SimpleParamService.parse_tuner_params(
        _base_tuner_settings(num_samples=-1, time_budget_s=None), "flaml"
    )
    assert out["num_samples"] == -1


# --------------------------------------------------------------------------- #
# search_space -> hiertunehub.SearchSpace
# --------------------------------------------------------------------------- #
def test_search_space_returns_hiertunehub_searchspace():
    SearchSpace = pytest.importorskip("hiertunehub").SearchSpace
    cfg = SimpleParamService(n_jobs=1, include_experimental=False, multivalue=False)
    est_list = cfg.estimator_names_from_patterns(
        "backdoor", ["SLearner", "TLearner"], 1000
    )
    ss = cfg.search_space(est_list, data_size=(1000, 5))
    assert isinstance(ss, SearchSpace)

    flaml_space = ss.to_flaml()
    assert "estimator" in flaml_space
    categories = flaml_space["estimator"].categories
    names = {c["estimator_name"] for c in categories}
    assert names == set(est_list)


# --------------------------------------------------------------------------- #
# Scorer.best_score_by_estimator now takes a list
# --------------------------------------------------------------------------- #
def test_best_score_by_estimator_accepts_list():
    scores = [
        {"estimator_name": "A", "energy_distance": 1.0},
        {"estimator_name": "A", "energy_distance": 0.5},
        {"estimator_name": "B", "energy_distance": 2.0},
    ]
    best = Scorer.best_score_by_estimator(scores, "energy_distance")
    assert set(best) == {"A", "B"}
    # energy_distance is minimized
    assert best["A"]["energy_distance"] == 0.5
    assert best["B"]["energy_distance"] == 2.0


def test_best_score_by_estimator_malformed_raises_valueerror():
    # Regression: the message must not reference the removed loop var `k`
    # (which used to raise NameError instead of the intended ValueError).
    with pytest.raises(ValueError):
        Scorer.best_score_by_estimator([{"energy_distance": 1.0}], "energy_distance")


# --------------------------------------------------------------------------- #
# End-to-end fits across backends
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("framework", ["flaml", "optuna", "hyperopt"])
def test_fit_end_to_end_per_backend(framework, data):
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    ct = _make_ct()
    ct.fit(data, framework=framework)

    assert ct.tuner is not None
    # the tuner must be a real hiertunehub Tuner (proves create_tuner is used,
    # not a lingering direct flaml.tune.run path)
    assert type(ct.tuner).__module__.startswith("hiertunehub")
    assert ct.tuner.framework == framework

    # public properties are repointed to the tuner (not the legacy self.results)
    assert ct.best_estimator == ct.tuner.best_result["estimator_name"]
    assert ct.best_config == ct.tuner.best_params
    assert ct.best_score == ct.tuner.best_result[ct.metric]

    assert isinstance(ct.best_estimator, str)
    assert isinstance(ct.best_config, dict)
    assert len(ct.scores) > 0

    effect = ct.effect(data.data)
    assert len(effect) == len(data.data)


def test_default_framework_is_optuna(data):
    pytest.importorskip("optuna")
    ct = _make_ct()
    ct.fit(data)  # no framework -> new default
    assert ct.tuner.framework == "optuna"


def test_optuna_custom_sampler_is_honored(data):
    optuna = pytest.importorskip("optuna")
    ct = _make_ct()
    ct.fit(data, framework="optuna", algo=optuna.samplers.RandomSampler(seed=0))
    assert isinstance(ct.tuner.study.sampler, optuna.samplers.RandomSampler)


@pytest.mark.parametrize("store_all", [False, True])
def test_model_and_effect_after_fit(store_all, data):
    ct = _make_ct(store_all_estimators=store_all)
    ct.fit(data, framework="optuna")
    assert isinstance(ct.best_estimator, str)
    # `.model` must resolve the fitted estimator from ct.scores even though the
    # objective pops the estimator object out of the result dict when not
    # store_all -- so it can't come from tuner.best_result["estimator"].
    assert ct.model is not None
    assert ct.model is ct.scores[ct.best_estimator]["estimator"].estimator
    effect = ct.effect(data.data)
    assert len(effect) == len(data.data)


@pytest.mark.parametrize("framework", ["flaml", "optuna"])
def test_failed_trial_not_selected_as_best_for_minimize(monkeypatch, data, framework):
    # A failed evaluation must not be picked as best. energy_distance is
    # minimized, so the failure sentinel must be +inf (worst), never -inf -- a
    # -inf would look optimal to a minimizing backend and poison selection.
    import numpy as np

    orig_stub = CausalTune._est_effect_stub
    state = {"n": 0}

    def flaky_stub(self, method_params):
        state["n"] += 1
        if state["n"] == 1:  # force the first trial to fail
            raise RuntimeError("boom")
        return orig_stub(self, method_params)

    monkeypatch.setattr(CausalTune, "_est_effect_stub", flaky_stub)

    ct = _make_ct()  # metric="energy_distance" (minimized), num_samples=2
    ct.fit(data, framework=framework)

    assert state["n"] >= 2  # at least one forced failure plus a real trial
    # best must be the finite (successful) trial, not the +inf failure
    assert np.isfinite(ct.best_score)
    assert ct.model is not None


# --------------------------------------------------------------------------- #
# FLAML-option parity (framework="flaml" only)
# --------------------------------------------------------------------------- #
def _spy_flaml_run(monkeypatch, sink):
    """Patch flaml.tune.run to record kwargs (into ``sink``) then call through."""
    import flaml.tune

    real_run = flaml.tune.run

    def spy(*args, **kwargs):
        # FLAML drains points_to_evaluate/evaluated_rewards in place as it runs,
        # so snapshot them (shallow copy) BEFORE handing off to the real call.
        snap = dict(kwargs)
        for key in ("points_to_evaluate", "evaluated_rewards"):
            if isinstance(snap.get(key), list):
                snap[key] = list(snap[key])
        sink.append(snap)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(flaml.tune, "run", spy)


def _estimator_level_calls(calls):
    """Filter the flaml.tune.run calls down to CausalTune's estimator-level
    tuner call. The spy also captures the many component-model AutoML calls;
    ours is the one carrying our custom ``cost_attr="evaluation_cost"``.
    """
    return [c for c in calls if c.get("cost_attr") == "evaluation_cost"]


def test_flaml_receives_parity_params(monkeypatch, data):
    calls = []
    _spy_flaml_run(monkeypatch, calls)

    ct = _make_ct(try_init_configs=True)
    ct.fit(data, framework="flaml")

    ours = _estimator_level_calls(calls)
    assert len(ours) == 1
    kw = ours[0]
    assert kw.get("cost_attr") == "evaluation_cost"
    assert kw.get("low_cost_partial_config") == {}
    # try_init_configs=True must seed points_to_evaluate
    assert len(kw.get("points_to_evaluate", [])) > 0
    # a fresh (non-resume) run has no prior rewards -> empty list, not None
    assert kw.get("evaluated_rewards") == []


def test_flaml_empty_init_configs_still_passed(monkeypatch, data):
    calls = []
    _spy_flaml_run(monkeypatch, calls)

    ct = _make_ct(try_init_configs=False)
    ct.fit(data, framework="flaml")

    ours = _estimator_level_calls(calls)
    assert len(ours) == 1
    # points_to_evaluate is an empty list, not omitted / None
    assert ours[0].get("points_to_evaluate") == []
    assert ours[0].get("evaluated_rewards") == []


def test_flaml_resume_rebuilds_from_previous_tuner(monkeypatch, data):
    calls = []
    _spy_flaml_run(monkeypatch, calls)

    # Resume is a time-budgeted "continue with more budget" flow, so drive the
    # runs by time (num_samples == -1). Use the near-instant Dummy estimator so
    # several trials actually complete within a small budget (a slow estimator
    # can consume the whole budget on a single trial, leaving nothing to resume).
    ct = CausalTune(
        metric="energy_distance",
        estimator_list=["Dummy"],
        num_samples=-1,
        components_time_budget=3,
        use_ray=False,
        try_init_configs=False,
    )
    ct.fit(data, framework="flaml", time_budget=5)

    # what the first run actually produced (captured BEFORE the resume fit)
    prev_results = ct.tuner.results
    expected_rewards = [
        r[ct.metric] for r in prev_results if ct.metric in r and "config" in r
    ]
    expected_configs = [
        r["config"] for r in prev_results if ct.metric in r and "config" in r
    ]
    assert expected_rewards  # sanity: the first run yielded resumable trials

    calls.clear()  # drop the first fit's calls; keep only the resume fit's
    ct.fit(data, framework="flaml", resume=True, time_budget=5)

    ours = _estimator_level_calls(calls)
    assert len(ours) == 1
    resumed = ours[0]
    # rewards are rebuilt exactly from the previous tuner's results...
    assert resumed.get("evaluated_rewards") == expected_rewards
    # ...and aligned to the leading points_to_evaluate configs
    assert (
        resumed.get("points_to_evaluate", [])[: len(expected_configs)]
        == expected_configs
    )


def test_resume_rebuild_skips_malformed_and_aligns_rewards():
    ct = _make_ct()
    ct.metric = "energy_distance"
    prev_results = [
        {"estimator_name": "A", "energy_distance": 1.0, "config": {"a": 1}},
        {"estimator_name": "A", "config": {"a": 2}},  # missing metric -> skip
        {"estimator_name": "A", "energy_distance": 3.0},  # missing config -> skip
    ]
    cfg, scores = ct._resume_points_and_rewards(prev_results, init_cfg=[{"a": 9}])
    # only the first (fully-formed) result contributes a reward
    assert scores == [1.0]
    # its config leads, and the unevaluated init config is appended after
    assert cfg[: len(scores)] == [{"a": 1}]
    assert {"a": 9} in cfg
    # rewards align to the leading configs; never more rewards than configs
    assert len(scores) <= len(cfg)


# --------------------------------------------------------------------------- #
# Best-effort behaviour on non-flaml backends
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_resume_raises_notimplemented_on_non_flaml(framework, data):
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    ct = _make_ct()
    with pytest.raises(NotImplementedError):
        ct.fit(data, framework=framework, resume=True)


@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_default_try_init_configs_warns_on_non_flaml(framework, data):
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    # _make_ct() leaves try_init_configs at its constructor default (True)
    ct = _make_ct()
    with pytest.warns(UserWarning, match="init config"):
        ct.fit(data, framework=framework)
    assert ct.tuner is not None


# --------------------------------------------------------------------------- #
# Packaging decision
# --------------------------------------------------------------------------- #
def test_packaging_dependencies():
    # tomllib is stdlib only from 3.11; the metadata it checks is
    # Python-independent, so just skip on 3.10 rather than pulling in tomli.
    tomllib = pytest.importorskip("tomllib")

    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())

    deps = " ".join(pyproject["project"]["dependencies"])
    assert "hiertunehub" in deps
    assert "optuna" in deps  # default backend -> hard dep

    extras = pyproject["project"].get("optional-dependencies", {})
    hyperopt_extra = " ".join(extras.get("hyperopt", []))
    assert "hyperopt" in hyperopt_extra  # optional extra, not a core dep
    assert "setuptools" in hyperopt_extra  # pkg_resources pin lives with hyperopt
