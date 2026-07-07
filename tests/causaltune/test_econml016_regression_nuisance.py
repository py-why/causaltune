"""
Regression test for the econml 0.16 discrete-outcome nuisance check.

econml 0.16's DRLearner family raises
``AttributeError: Cannot use a classifier for model_regression when
discrete_outcome=False!`` for any regression nuisance model that exposes
``predict_proba``. FLAML estimators define ``predict_proba`` regardless of task,
so causaltune's FLAML-based *regression* outcome models must hide it (while
*classification* propensity models keep it).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dowhy import CausalModel

from causaltune.models.monkey_patches import AutoML
from causaltune.search.component import model_from_cfg


def test_flaml_regression_models_hide_predict_proba():
    # regression (outcome) models must NOT look like classifiers to econml 0.16
    assert not hasattr(AutoML(task="regression", time_budget=1), "predict_proba")
    assert not hasattr(model_from_cfg({"estimator_name": "xgboost"}), "predict_proba")
    # classification (propensity) models MUST keep predict_proba
    assert hasattr(AutoML(task="classification", time_budget=1), "predict_proba")


def test_boolean_search_params_are_bool_not_int():
    """sklearn 1.6 requires fit_intercept-style params to be real bools, not int 0/1.

    causaltune's econml estimators forward fit_cate_intercept / fit_intercept to
    sklearn nuisance models, so their search domains and defaults must be boolean.
    """
    from causaltune.search.params import SimpleParamService

    svc = SimpleParamService(multivalue=False)
    checks = {
        "backdoor.econml.dr.SparseLinearDRLearner": "fit_cate_intercept",
        "backdoor.econml.dml.SparseLinearDML": "fit_cate_intercept",
        "backdoor.econml.dml.CausalForestDML": "fit_intercept",
    }
    for estimator_name, param in checks.items():
        cfg = svc.full_config(estimator_name)
        if param in cfg.defaults:
            assert isinstance(cfg.defaults[param], bool), (estimator_name, param)
        if param in cfg.search_space:
            cats = cfg.search_space[param].categories
            assert all(isinstance(c, bool) for c in cats), (estimator_name, param, cats)


def test_shap_kernel_fallback_for_non_tree_flaml_model():
    """shap >= 0.44 KernelExplainer tried to null FLAML's read-only
    ``feature_names_in_``. Non-tree outcome models (elastic_net/lasso_lars) take
    the KernelExplainer fallback and must still produce SHAP values."""
    from causaltune.search.component import model_from_cfg
    from causaltune.shap import shap_with_automl

    rng = np.random.RandomState(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x0", "x1", "x2"])
    y = X["x0"] * 2 + rng.normal(size=n)

    for est in ("elastic_net", "lasso_lars"):
        model = model_from_cfg({"estimator_name": est})
        model.fit(X, y)
        sv = shap_with_automl(model, X[:8])
        assert np.asarray(sv).shape == (8, 3)


def test_dr_learner_fits_with_flaml_regression_outcome():
    """A DRLearner with a FLAML regression outcome model fits and predicts finite
    effects (reproduces the econml-0.16 classifier rejection if the fix regresses)."""
    rng = np.random.RandomState(0)
    n = 500
    X = rng.normal(size=(n, 3))
    T = rng.randint(0, 2, n)
    Y = T * 1.5 + X[:, 0] + rng.normal(size=n) * 0.5
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["T"] = T
    df["Y"] = Y

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
        method_name="backdoor.econml.dr.LinearDRLearner",
        control_value=0,
        treatment_value=[1],
        target_units="ate",
        confidence_intervals=False,
        method_params={
            "init_params": {
                "model_regression": model_from_cfg({"estimator_name": "xgboost"}),
                "model_propensity": LogisticRegression(max_iter=500),
            },
            "fit_params": {},
        },
    )

    assert np.all(np.isfinite(np.asarray(estimate.cate_estimates)))
