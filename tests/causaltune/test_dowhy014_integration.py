"""
Regression tests for the dowhy 0.14 / econml 0.16 integration.

These lock in the behaviour-preserving hybrid described in the upgrade spec:

  * the string ``method_name`` dispatch keeps working, and the resulting
    ``econml_estimator`` DeprecationWarning is suppressed by causaltune;
  * ``effect_tt(df, treatment_value)`` (the new dowhy-0.14 signature) returns a
    per-unit Series for the econml adapter (binary AND multivalue) and for a
    custom ``DoWhyWrapper`` estimator;
  * the scorer's factual-outcome path (``_Y0_X_potential_outcomes``) that
    consumes ``effect_tt`` still works after the attribute renames
    (``_treatment_name`` / ``_outcome_name`` are gone from the econml adapter);
  * ``DoWhyWrapper.estimate_effect`` builds a valid dowhy-0.14 ``CausalEstimate``;
  * the ``const_marginal_effect`` bug fix (``self.effect(X)``, not
    ``self.effect(self, X)``).
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from dowhy import CausalModel


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_data(n=400, multivalue=False, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    T = rng.randint(0, 3, n) if multivalue else rng.randint(0, 2, n)
    Y = (T > 0) * 1.5 + X[:, 0] + rng.normal(size=n) * 0.5
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["T"] = T
    df["Y"] = Y
    return df


def fit_estimate(df, method_name, method_params, treatment_value):
    cm = CausalModel(
        data=df,
        treatment="T",
        outcome="Y",
        common_causes=["x0", "x1", "x2"],
        effect_modifiers=["x0", "x1", "x2"],
    )
    identified = cm.identify_effect(proceed_when_unidentifiable=True)
    estimate = cm.estimate_effect(
        identified,
        method_name=method_name,
        control_value=0,
        treatment_value=treatment_value,
        target_units="ate",
        confidence_intervals=False,
        method_params=method_params,
    )
    return cm, estimate


SLEARNER = "backdoor.econml.metalearners.SLearner"
LINEARDML = "backdoor.econml.dml.LinearDML"
NAIVE_DUMMY = "backdoor.causaltune.models.NaiveDummy"


def slearner_params():
    return {
        "init_params": {"overall_model": DecisionTreeRegressor(random_state=0)},
        "fit_params": {},
    }


# --------------------------------------------------------------------------- #
# string dispatch + deprecation suppression
# --------------------------------------------------------------------------- #
def test_econml_string_deprecation_filter_is_narrow():
    """causaltune installs a NARROW filter (ignore, DeprecationWarning, message
    matching 'econml_estimator') — not a blanket ``ignore::DeprecationWarning``."""
    import causaltune

    with warnings.catch_warnings():
        causaltune._install_warning_filters()  # pytest resets filters per-test
        match = None
        for action, message, category, _module, _lineno in warnings.filters:
            if category is None or not issubclass(DeprecationWarning, category):
                continue
            pattern = getattr(
                message, "pattern", message if isinstance(message, str) else None
            )
            if pattern and "econml_estimator" in pattern:
                match = (action, pattern)
                break

    assert (
        match is not None
    ), "causaltune must register an econml_estimator DeprecationWarning filter"
    action, pattern = match
    assert action == "ignore"
    # must be specific, not a catch-all
    assert pattern not in (".*", "", ".*?")


def test_econml_string_deprecation_suppressed_but_others_survive():
    """Behaviourally: the econml_estimator warning is suppressed while an unrelated
    DeprecationWarning is still delivered (proves the filter is not a blanket ignore).
    """
    import causaltune

    df = make_data()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        causaltune._install_warning_filters()  # re-apply after simplefilter reset
        fit_estimate(df, LINEARDML, {"init_params": {}, "fit_params": {}}, [1])
        warnings.warn("an unrelated deprecation", DeprecationWarning)

    messages = [str(w.message) for w in rec]
    assert not any("econml_estimator" in m for m in messages), messages
    assert any("an unrelated deprecation" in m for m in messages), messages


def test_lineardml_end_to_end_finite():
    """A representative econml estimator fits via string dispatch and yields
    finite CATE estimates."""
    import causaltune  # noqa: F401  (installs the warning filter)

    df = make_data()
    _cm, estimate = fit_estimate(
        df, LINEARDML, {"init_params": {}, "fit_params": {}}, [1]
    )

    assert estimate.cate_estimates is not None
    assert np.all(np.isfinite(np.asarray(estimate.cate_estimates)))
    assert np.isfinite(np.asarray(estimate.value)).all()


# --------------------------------------------------------------------------- #
# effect_tt (new dowhy-0.14 two-arg signature)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("multivalue", [False, True])
def test_effect_tt_econml_adapter(multivalue):
    df = make_data(multivalue=multivalue)
    treatment_value = [1, 2] if multivalue else [1]
    _cm, estimate = fit_estimate(df, SLEARNER, slearner_params(), treatment_value)

    est = estimate.estimator
    tt = est.effect_tt(df, est._treatment_value)

    assert isinstance(tt, pd.Series)
    assert len(tt) == len(df)
    assert np.all(np.isfinite(tt.values))


def test_effect_tt_custom_wrapper():
    df = make_data()  # NaiveDummy/DummyModel is binary-only
    _cm, estimate = fit_estimate(
        df, NAIVE_DUMMY, {"init_params": {}, "fit_params": {}}, [1]
    )

    est = estimate.estimator
    tt = est.effect_tt(df, est._treatment_value)

    assert isinstance(tt, pd.Series)
    assert len(tt) == len(df)
    assert np.all(np.isfinite(tt.values))


# --------------------------------------------------------------------------- #
# effect_tt selects the row's OBSERVED treatment effect (deterministic)
# --------------------------------------------------------------------------- #
def _patch_constant_effect(est, per_treatment_values):
    """Force est.effect(X) to return known constant per-treatment columns."""
    import types

    def fake_effect(self, X, *args, **kwargs):
        n = len(X)
        return np.column_stack([np.full(n, v) for v in per_treatment_values])

    est.effect = types.MethodType(fake_effect, est)


def _expected_factual(df, treatment_values, per_treatment_values):
    expected = np.zeros(len(df))
    for tv, val in zip(treatment_values, per_treatment_values):
        expected[df["T"].values == tv] = val
    return expected


@pytest.mark.parametrize(
    "multivalue,treatment_values,per_treatment_values",
    [(False, [1], [10.0]), (True, [1, 2], [10.0, 20.0])],
)
def test_effect_tt_selects_observed_treatment_econml(
    multivalue, treatment_values, per_treatment_values
):
    """effect_tt must return, per row, the effect of the treatment actually applied."""
    df = make_data(multivalue=multivalue)
    _cm, estimate = fit_estimate(df, SLEARNER, slearner_params(), treatment_values)
    est = estimate.estimator

    _patch_constant_effect(est, per_treatment_values)
    tt = est.effect_tt(df, est._treatment_value)

    expected = _expected_factual(df, treatment_values, per_treatment_values)
    np.testing.assert_allclose(np.asarray(tt).ravel(), expected)


def test_effect_tt_selects_observed_treatment_custom_wrapper():
    """Same factual-selection guarantee for causaltune's own DoWhyWrapper effect_tt."""
    df = make_data()
    _cm, estimate = fit_estimate(
        df, NAIVE_DUMMY, {"init_params": {}, "fit_params": {}}, [1]
    )
    est = estimate.estimator

    _patch_constant_effect(est, [7.0])
    tt = est.effect_tt(df, est._treatment_value)

    expected = _expected_factual(df, [1], [7.0])
    np.testing.assert_allclose(np.asarray(tt).ravel(), expected)


def test_scorer_potential_outcomes_runs_and_finite():
    """Integration: the scorer factual path (which consumes effect_tt and the
    renamed treatment/outcome-name attributes) runs and yields finite values."""
    from causaltune.score.scoring import Scorer

    df = make_data()
    _cm, estimate = fit_estimate(df, SLEARNER, slearner_params(), [1])

    Y0X, _treatment_name, _split_test_by = Scorer._Y0_X_potential_outcomes(
        estimate, df.copy()
    )

    assert "dy" in Y0X.columns and "yhat" in Y0X.columns
    assert np.all(np.isfinite(Y0X["dy"].values))
    assert np.all(np.isfinite(Y0X["yhat"].values))


# --------------------------------------------------------------------------- #
# DoWhyWrapper builds a valid 0.14 CausalEstimate; full surface
# --------------------------------------------------------------------------- #
def test_dowhywrapper_estimate_and_effect_surface():
    df = make_data()
    _cm, estimate = fit_estimate(
        df, NAIVE_DUMMY, {"init_params": {}, "fit_params": {}}, [1]
    )

    # estimate_effect built a valid CausalEstimate
    assert np.isfinite(np.asarray(estimate.value)).all()

    est = estimate.estimator
    eff = est.effect(df)
    assert len(eff) == len(df)
    assert np.all(np.isfinite(np.asarray(eff)))


def test_shap_values_transformed_outcome_custom_wrapper():
    """SHAP adapter works for a custom DoWhyWrapper (TransformedOutcome) on the stack."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    from causaltune.shap import shap_values

    df = make_data(n=300)
    _cm, estimate = fit_estimate(
        df,
        "backdoor.causaltune.models.TransformedOutcome",
        {
            "init_params": {
                "outcome_model": RandomForestRegressor(n_estimators=10, random_state=0),
                "propensity_model": LogisticRegression(max_iter=500),
            },
            "fit_params": {},
        },
        [1],
    )

    sv = shap_values(estimate, df[:10])
    assert sv is not None


def test_const_marginal_effect_matches_effect():
    """Regression for the const_marginal_effect(self, X) -> self.effect(self, X) bug.

    (DummyModel.predict is randomized, so we pin ``effect`` to a deterministic
    marker and assert const_marginal_effect delegates to it with the right args.)"""
    import types

    df = make_data()
    _cm, estimate = fit_estimate(
        df, NAIVE_DUMMY, {"init_params": {}, "fit_params": {}}, [1]
    )
    est = estimate.estimator

    marker = np.arange(len(df), dtype=float)
    est.effect = types.MethodType(lambda self, X: marker, est)

    np.testing.assert_array_equal(np.asarray(est.const_marginal_effect(df)), marker)
