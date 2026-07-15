"""Interface-parity tests: bring optuna/hyperopt up to the FLAML option surface.

Companion to ``test_tuner_backends.py`` (which covers the base pluggable-backend
port). These exercise the *parity* additions:

* **Warm-start** (`try_init_configs`) now seeds optuna (via ``enqueue_trial``) and
  hyperopt (via seeded NEW trial docs), not just flaml. The old "only flaml"
  warning is gone.
* **Resume** works on optuna/hyperopt (routed through hiertunehub's native
  ``resume_from_results``), no longer raising ``NotImplementedError``. Budget is
  "additional" (a resume of ``num_samples=N`` runs ~N *more* trials), and
  ``model``/``effect`` still resolve after a resume.
* **verbose** is honoured on optuna (causaltune-side ``optuna.logging``).
* **parallelism**: ``resources_per_trial`` maps to optuna ``n_jobs`` but is
  clamped to 1 unless the user explicitly opts in (the objective is not
  thread-safe); opting into ``n_jobs>1`` warns.
* **framework_params passthrough** on ``fit()``/constructor: merged last (user
  wins), warns on a managed-key collision, raises on a reserved key.
* **constructor symmetry**: ``framework``/``algo``/``framework_params`` are
  settable on the constructor and overridden by ``fit()``.
"""

import warnings

import numpy as np
import pytest

import causaltune.optimiser as opt_mod
from causaltune import CausalTune
from causaltune.datasets import synth_ihdp
from causaltune.search.params import SimpleParamService

# LinearDML pattern expands to [LinearDML, SparseLinearDML] -- both carry tunable
# search spaces, so every backend (incl. hyperopt, which rejects a fully
# parameterless space) can warm-start and resume on them.
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


def _spy_create_tuner(monkeypatch, sink):
    """Record every ``create_tuner`` call's kwargs, then call through.

    CausalTune calls ``create_tuner`` exactly once per ``fit`` (the
    estimator-level tuner; component-model AutoML uses flaml directly), so
    ``sink[0]`` is our call.
    """
    real = opt_mod.create_tuner

    def spy(*args, **kwargs):
        # framework_params is drained/mutated by the backends as they run, so
        # snapshot it (shallow) before handing off.
        snap = dict(kwargs)
        fp = snap.get("framework_params")
        if isinstance(fp, dict):
            snap["framework_params"] = dict(fp)
        sink.append(snap)
        return real(*args, **kwargs)

    monkeypatch.setattr(opt_mod, "create_tuner", spy)


# --------------------------------------------------------------------------- #
# Warm-start (item B) -- optuna + hyperopt now seed points_to_evaluate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_warmstart_passes_points_to_evaluate(monkeypatch, data, framework):
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    calls = []
    _spy_create_tuner(monkeypatch, calls)

    ct = _make_ct(try_init_configs=True)
    ct.fit(data, framework=framework)

    assert len(calls) == 1
    fp = calls[0]["framework_params"]
    pts = fp.get("points_to_evaluate")
    assert pts, "try_init_configs must seed points_to_evaluate for non-flaml backends"
    # every seed is a natural config point carrying the estimator name
    assert all("estimator" in p and "estimator_name" in p["estimator"] for p in pts)
    # ...and they are exactly the default_configs() init points (deterministic
    # here since outcome_model='nested' => sample_outcome_estimators is False)
    expected = ct.cfg.default_configs(ct.estimator_list, data_size=data.data.shape)
    assert pts == expected


@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_warmstart_off_passes_no_points(monkeypatch, data, framework):
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    calls = []
    _spy_create_tuner(monkeypatch, calls)

    ct = _make_ct(try_init_configs=False)
    ct.fit(data, framework=framework)

    fp = calls[0]["framework_params"]
    assert not fp.get("points_to_evaluate")


@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_warmstart_no_longer_warns_on_non_flaml(data, framework):
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    ct = _make_ct(try_init_configs=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ct.fit(data, framework=framework)
    assert not any(
        "init config" in str(w.message).lower() for w in caught
    ), "the old 'only flaml warm-starts' warning must be gone"


@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_warmstart_end_to_end_runs(data, framework):
    """Warm-start actually drives a real fit (the backend consumes the seeds)."""
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    ct = _make_ct(try_init_configs=True)
    ct.fit(data, framework=framework)
    assert ct.tuner is not None
    assert len(ct.scores) > 0
    assert isinstance(ct.best_estimator, str)


@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_warmstart_more_seeds_than_budget(data, framework):
    """Regression: warm-start seeds one config per estimator. When num_samples is
    smaller than the estimator count, the un-run seeds must not surface as empty
    trial results (optuna leaves them WAITING; hyperopt truncates) -- otherwise
    update_summary_scores chokes on a malformed {} entry."""
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    # LinearDML pattern -> 2 estimators, but only budget for 1 trial
    ct = _make_ct(try_init_configs=True, num_samples=1)
    ct.fit(data, framework=framework)
    assert len(ct.scores) > 0
    assert isinstance(ct.best_estimator, str)
    assert all(r != {} for r in ct.tuner.results)


# --------------------------------------------------------------------------- #
# Resume (item A) -- optuna + hyperopt via hiertunehub resume_from_results
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_resume_no_longer_raises_and_grows(data, framework):
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    ct = _make_ct(try_init_configs=False, num_samples=-1, components_time_budget=3)
    ct.fit(data, framework=framework, time_budget=8)
    before = len(ct.tuner.results)
    assert before > 0

    ct.fit(data, framework=framework, resume=True, time_budget=8)
    after = len(ct.tuner.results)
    # must actually add new trials on top of the rehydrated past ones (a no-op
    # resume that only rehydrates would leave the count unchanged)
    assert after > before, "resume must retain past trials AND add new ones"
    # model/effect still resolve after a resume (guards the _best_estimators fix)
    assert ct.model is not None
    assert np.isfinite(ct.best_score)
    effect = ct.effect(data.data)
    assert len(effect) == len(data.data)


@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_resume_finite_budget_is_additional(data, framework):
    """A finite ``num_samples=N`` resume must run ~N *additional* trials, not
    zero (optuna internally subtracts len(past); hyperopt max_evals is total)."""
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    n = 2
    ct = _make_ct(try_init_configs=False, num_samples=n, components_time_budget=3)
    ct.fit(data, framework=framework)
    before = len(ct.tuner.results)

    ct.fit(data, framework=framework, resume=True)
    after = len(ct.tuner.results)
    assert (
        after - before == n
    ), f"expected {n} additional trials on resume, got {after - before}"


@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_resume_without_prior_warns_and_runs_fresh(data, framework):
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    ct = _make_ct(try_init_configs=False)
    # a no-prior resume must behave like a fresh fit -> reset _best_estimators
    ct._best_estimators["__sentinel__"] = (0.123, object())
    with pytest.warns(UserWarning, match="no previous fit"):
        ct.fit(data, framework=framework, resume=True)
    assert ct.tuner is not None
    assert len(ct.scores) > 0
    assert (
        "__sentinel__" not in ct._best_estimators
    ), "resume with no prior tuner must reset _best_estimators (fresh run)"


@pytest.mark.parametrize("framework", ["optuna", "hyperopt"])
def test_resume_does_not_mutate_previous_tuner_trials(data, framework):
    """Building the resume seed list must not mutate the previous tuner's stored
    trials (params/result), i.e. we copy before handing to hiertunehub."""
    if framework == "hyperopt":
        pytest.importorskip("hyperopt")
    ct = _make_ct(try_init_configs=False, num_samples=2)
    ct.fit(data, framework=framework)
    prev_tuner = ct.tuner
    import copy as _copy

    snapshot = _copy.deepcopy(
        [{"params": t.params, "result": dict(t.result)} for t in prev_tuner.trials]
    )

    ct.fit(data, framework=framework, resume=True)

    # every field of every previous trial must be untouched, esp. estimator_name
    for snap, t in zip(snapshot, prev_tuner.trials):
        assert snap["params"] == t.params
        # result must not be mutated either: hiertunehub adds loss/status
        # (hyperopt) or rewrites result[metric] (optuna) on the copy we hand it,
        # never on the stored trial. Compare the key set and the metric scalar
        # (full-dict == is ambiguous because some values are pandas objects).
        assert set(snap["result"]) == set(t.result), "no keys added/removed"
        assert snap["result"]["energy_distance"] == t.result["energy_distance"]
        # estimator_name is what the mutating hyperopt converter used to drop
        if "estimator" in snap["params"]:
            assert "estimator_name" in t.params["estimator"]


# energy_distance is minimized (the +inf-default branch at optimiser.py:422);
# erupt is maximized (the -inf default). The reset must be symmetric across both.
@pytest.mark.parametrize("metric", ["energy_distance", "erupt"])
def test_best_estimators_preserved_on_resume_but_reset_on_fresh(data, metric):
    ct = _make_ct(try_init_configs=False, num_samples=2, metric=metric)
    ct.fit(data, framework="optuna")
    # seed a sentinel; a real resume must preserve it, a fresh fit must clear it
    ct._best_estimators["__sentinel__"] = (0.123, object())

    ct.fit(data, framework="optuna", resume=True)
    assert (
        "__sentinel__" in ct._best_estimators
    ), "resume must preserve _best_estimators"

    ct._best_estimators["__sentinel__"] = (0.123, object())
    ct.fit(data, framework="optuna")  # fresh
    assert (
        "__sentinel__" not in ct._best_estimators
    ), f"fresh fit must reset _best_estimators (metric={metric})"


# --------------------------------------------------------------------------- #
# verbose (item C) -- optuna, causaltune-side optuna.logging
# --------------------------------------------------------------------------- #
def test_optuna_verbose_maps_to_logging_level(data):
    optuna = pytest.importorskip("optuna")
    ct = _make_ct(verbose=0)
    ct.fit(data, framework="optuna")
    assert optuna.logging.get_verbosity() == optuna.logging.WARNING

    ct = _make_ct(verbose=1)
    ct.fit(data, framework="optuna")
    assert optuna.logging.get_verbosity() == optuna.logging.INFO


def test_parse_tuner_params_optuna_verbose_not_in_params():
    # verbose is handled causaltune-side, NOT threaded into the optuna param dict
    out = SimpleParamService.parse_tuner_params(_base_tuner_settings(), "optuna")
    assert "verbose" not in out


# --------------------------------------------------------------------------- #
# parallelism (item D) -- resources_per_trial -> optuna n_jobs (clamped)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cpu", [0.5, 0.25, 1.0])
def test_parse_tuner_params_optuna_njobs_clamped_to_one(cpu):
    out = SimpleParamService.parse_tuner_params(
        _base_tuner_settings(resources_per_trial={"cpu": cpu}), "optuna"
    )
    # mapping is real but clamped to 1 by default (objective is not thread-safe)
    assert out["n_jobs"] == 1


def test_optuna_explicit_njobs_optin_warns(monkeypatch, data):
    calls = []
    _spy_create_tuner(monkeypatch, calls)
    ct = _make_ct()
    with pytest.warns(UserWarning, match="n_jobs"):
        ct.fit(data, framework="optuna", framework_params={"n_jobs": 2})
    # user opt-in wins
    assert calls[0]["framework_params"]["n_jobs"] == 2


# --------------------------------------------------------------------------- #
# framework_params passthrough
# --------------------------------------------------------------------------- #
def test_framework_params_passthrough_reaches_tuner(monkeypatch, data):
    calls = []
    _spy_create_tuner(monkeypatch, calls)
    ct = _make_ct()
    ct.fit(data, framework="optuna", framework_params={"study_name": "spam"})
    assert calls[0]["framework_params"].get("study_name") == "spam"


def test_framework_params_managed_key_warns_but_wins(monkeypatch, data):
    calls = []
    _spy_create_tuner(monkeypatch, calls)
    ct = _make_ct()
    with pytest.warns(UserWarning):
        ct.fit(data, framework="optuna", framework_params={"n_trials": 7})
    assert calls[0]["framework_params"]["n_trials"] == 7


@pytest.mark.parametrize("bad_key", ["config", "mode", "metric"])
def test_framework_params_reserved_key_raises(data, bad_key):
    # config/mode/metric collide with what hiertunehub passes flaml explicitly
    ct = _make_ct()
    with pytest.raises(ValueError, match=bad_key):
        ct.fit(data, framework="flaml", framework_params={bad_key: "boom"})


def test_framework_params_reserved_trials_raises_hyperopt(data):
    # `trials` collides with the hyperopt fmin(trials=...) hiertunehub passes
    pytest.importorskip("hyperopt")
    ct = _make_ct()
    with pytest.raises(ValueError, match="trials"):
        ct.fit(data, framework="hyperopt", framework_params={"trials": "boom"})


# --------------------------------------------------------------------------- #
# Constructor symmetry (Q1) -- framework/algo/framework_params on __init__
# --------------------------------------------------------------------------- #
def test_constructor_framework_used_by_default(data):
    ct = _make_ct(framework="flaml")
    ct.fit(data)  # no framework kwarg -> falls back to constructor value
    assert ct.tuner.framework == "flaml"


def test_fit_framework_overrides_constructor(data):
    ct = _make_ct(framework="flaml")
    ct.fit(data, framework="optuna")
    assert ct.tuner.framework == "optuna"


def test_constructor_framework_params_used(monkeypatch, data):
    calls = []
    _spy_create_tuner(monkeypatch, calls)
    ct = _make_ct(framework="optuna", framework_params={"study_name": "ctor"})
    ct.fit(data)
    assert calls[0]["framework_params"].get("study_name") == "ctor"


def test_fit_framework_params_overrides_constructor(monkeypatch, data):
    calls = []
    _spy_create_tuner(monkeypatch, calls)
    ct = _make_ct(framework="optuna", framework_params={"study_name": "ctor"})
    ct.fit(data, framework_params={"study_name": "fit"})
    assert calls[0]["framework_params"].get("study_name") == "fit"


def test_constructor_algo_used_by_default(data):
    optuna = pytest.importorskip("optuna")
    ct = _make_ct(framework="optuna", algo=optuna.samplers.RandomSampler(seed=0))
    ct.fit(data)  # no algo kwarg -> constructor sampler
    assert isinstance(ct.tuner.study.sampler, optuna.samplers.RandomSampler)


def test_fit_algo_overrides_constructor(data):
    optuna = pytest.importorskip("optuna")
    # constructor picks Random, fit overrides with a CmaEs/TPE -> fit wins
    ct = _make_ct(framework="optuna", algo=optuna.samplers.RandomSampler(seed=0))
    ct.fit(data, algo=optuna.samplers.TPESampler(seed=0))
    assert isinstance(ct.tuner.study.sampler, optuna.samplers.TPESampler)
