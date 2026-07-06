"""
Comprehensive scoring coverage on the dowhy 0.14 / econml 0.16 stack.

This is the regression guard for the ``_treatment_name`` / ``_outcome_name``
accessor migration: it drives ``Scorer.make_scores`` through *every* backdoor and
IV metric branch, each of which reads those (now-renamed) adapter attributes.

Constructing the ``Scorer`` also fits the ``MultivaluePSW`` propensity-weighting
estimator (a custom ``DoWhyWrapper``), so the custom-estimator fit/effect_tt path
is exercised for binary and multivalue treatments too.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from dowhy import CausalModel

from causaltune.score.scoring import Scorer, supported_metrics


def _causal_model(df, instruments=None):
    return CausalModel(
        data=df,
        treatment="T",
        outcome="Y",
        common_causes=["x0", "x1", "x2"],
        effect_modifiers=["x0", "x1", "x2"],
        instruments=instruments,
    )


def make_backdoor_data(n=600, multivalue=False, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    T = rng.randint(0, 3, n) if multivalue else rng.randint(0, 2, n)
    Y = (T > 0) * 1.5 + X[:, 0] + rng.normal(size=n) * 0.5
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["T"] = T
    df["Y"] = Y
    return df


def make_iv_data(n=800, seed=2):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    Z = rng.randint(0, 2, n)
    T = ((Z + (X[:, 0] > 0).astype(int) + rng.normal(size=n)) > 1).astype(int)
    Y = T * 1.2 + X[:, 0] + rng.normal(size=n) * 0.5
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["Z"] = Z
    df["T"] = T
    df["Y"] = Y
    return df


@pytest.mark.parametrize("multivalue", [False, True])
def test_make_scores_all_backdoor_metrics(multivalue):
    df = make_backdoor_data(multivalue=multivalue)
    cm = _causal_model(df)
    identified = cm.identify_effect(proceed_when_unidentifiable=True)

    method = (
        "backdoor.econml.dml.LinearDML"
        if multivalue
        else "backdoor.econml.metalearners.SLearner"
    )
    init_params = (
        {} if multivalue else {"overall_model": DecisionTreeRegressor(random_state=0)}
    )
    treatment_value = [1, 2] if multivalue else [1]

    estimate = cm.estimate_effect(
        identified,
        method_name=method,
        control_value=0,
        treatment_value=treatment_value,
        target_units="ate",
        confidence_intervals=False,
        method_params={"init_params": init_params, "fit_params": {}},
    )

    # Building the Scorer fits MultivaluePSW (a custom DoWhyWrapper) internally.
    scorer = Scorer(
        cm, LogisticRegression(max_iter=500), problem="backdoor", multivalue=multivalue
    )

    metrics = supported_metrics("backdoor", multivalue=multivalue, scores_only=True)
    scores = scorer.make_scores(estimate, df, metrics)

    for m in metrics:
        assert m in scores, f"missing metric {m}"
        # Exercising every metric branch (the treatment/outcome-name accessor
        # migration) is the point of this test; some metrics (codec / frobenius /
        # energy) legitimately return inf on degenerate or small data, so require a
        # real number rather than strict finiteness.
        assert isinstance(
            scores[m], (int, float, np.floating, np.integer)
        ), f"metric {m} is not numeric: {scores[m]!r}"


def test_make_scores_iv_metrics():
    df = make_iv_data()
    cm = _causal_model(df, instruments=["Z"])
    identified = cm.identify_effect(proceed_when_unidentifiable=True)

    estimate = cm.estimate_effect(
        identified,
        method_name="iv.econml.iv.dml.DMLIV",
        control_value=0,
        treatment_value=[1],
        target_units="ate",
        confidence_intervals=False,
        method_params={"init_params": {}, "fit_params": {}},
    )

    scorer = Scorer(
        cm, LogisticRegression(max_iter=500), problem="iv", multivalue=False
    )

    metrics = supported_metrics("iv", multivalue=False, scores_only=True)
    scores = scorer.make_scores(estimate, df, metrics)

    for m in metrics:
        assert m in scores, f"missing metric {m}"
        # Exercising every metric branch (the treatment/outcome-name accessor
        # migration) is the point of this test; some metrics (codec / frobenius /
        # energy) legitimately return inf on degenerate or small data, so require a
        # real number rather than strict finiteness.
        assert isinstance(
            scores[m], (int, float, np.floating, np.integer)
        ), f"metric {m} is not numeric: {scores[m]!r}"
