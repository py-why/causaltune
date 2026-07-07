"""
Regression test for the rewritten ``effect_stderr`` helper.

dowhy 0.14 provides ``apply_multitreatment`` / ``effect_inference`` natively on
its econml adapter, but has no standard-error helper.  causaltune keeps its own
``effect_stderr``; the rewrite must use the adapter's native
``apply_multitreatment`` (which expects already column-selected features) and
still return finite standard errors of the right shape.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from dowhy import CausalModel


def make_data(n=400, seed=0, extra_column=False):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    T = rng.randint(0, 2, n)
    Y = (T > 0) * 1.5 + X[:, 0] + rng.normal(size=n) * 0.5
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["T"] = T
    df["Y"] = Y
    if extra_column:
        # a column that is NOT an effect modifier; effect_stderr must ignore it,
        # otherwise a naive full-frame `.values` would feed the wrong shape to econml
        df["not_a_feature"] = rng.normal(size=n)
    return df


def _fit_lineardml(df):
    cm = CausalModel(
        data=df,
        treatment="T",
        outcome="Y",
        common_causes=["x0", "x1", "x2"],
        effect_modifiers=["x0", "x1", "x2"],
    )
    identified = cm.identify_effect(proceed_when_unidentifiable=True)
    return cm.estimate_effect(
        identified,
        method_name="backdoor.econml.dml.LinearDML",
        control_value=0,
        treatment_value=[1],
        target_units="ate",
        confidence_intervals=False,
        method_params={
            "init_params": {
                "model_y": DecisionTreeRegressor(),
                "model_t": DecisionTreeRegressor(),
            },
            "fit_params": {},
        },
    )


def test_effect_stderr_finite_shape_and_ignores_non_features():
    from causaltune.models.monkey_patches import effect_stderr

    # extra non-feature column present: a full-frame `.values` would break econml
    df = make_data(extra_column=True)
    estimate = _fit_lineardml(df)

    est = estimate.estimator
    est.__class__.effect_stderr = effect_stderr

    means = np.asarray(est.effect(df))
    stds = np.squeeze(np.asarray(est.effect_stderr(df)))

    assert stds.shape == np.squeeze(means).shape
    assert np.all(np.isfinite(stds))
    assert np.all(stds >= 0)


def test_effect_stderr_matches_native_inference():
    """The helper must equal econml's own per-unit stderr on the effect modifiers."""
    from causaltune.models.monkey_patches import effect_stderr

    df = make_data()
    estimate = _fit_lineardml(df)
    est = estimate.estimator
    est.__class__.effect_stderr = effect_stderr

    stds = np.squeeze(np.asarray(est.effect_stderr(df)))

    X = df[est._effect_modifier_names].values
    native = np.squeeze(est.estimator.effect_inference(X, T0=0, T1=1).stderr)

    np.testing.assert_allclose(stds, native, rtol=1e-6, atol=1e-8)
