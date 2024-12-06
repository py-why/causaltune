import warnings
from typing import Tuple
import copy

import numpy as np
import pandas as pd

from flaml import tune
from flaml.automl.model import (
    KNeighborsEstimator,
    XGBoostSklearnEstimator,
    XGBoostLimitDepthEstimator,
    RandomForestEstimator,
    LGBMEstimator,
    CatBoostEstimator,
    ExtraTreesEstimator,
)
from flaml.automl.task.factory import task_factory
import flaml

from causaltune.models.regression import ElasticNetEstimator, LassoLarsEstimator


def flaml_config_to_tune_config(flaml_config: dict) -> Tuple[dict, dict, dict]:
    cfg = {}
    init_params = {}
    low_cost_init_params = {}
    for key, value in flaml_config.items():
        if isinstance(value["domain"], dict):
            raise NotImplementedError("Nested dictionaries are not supported yet")
        cfg[key] = value["domain"]
        if "init_value" in value:
            init_params[key] = value["init_value"]
        if "low_cost_init_value" in value:
            low_cost_init_params[key] = value["low_cost_init_value"]

    return cfg, init_params, low_cost_init_params


estimators = {
    "elastic_net": ElasticNetEstimator,
    "lasso_lars": LassoLarsEstimator,
    "knn": KNeighborsEstimator,
    "xgboost": XGBoostSklearnEstimator,
    "xgboost_limit_depth": XGBoostLimitDepthEstimator,
    "random_forest": RandomForestEstimator,
    "lgbm": LGBMEstimator,
    "catboost": CatBoostEstimator,
    "extra_trees": ExtraTreesEstimator,
}


def joint_config(data_size: Tuple[int, int], estimator_list=None):
    joint_cfg = []
    joint_init_params = []
    joint_low_cost_init_params = {}
    for name, cls in estimators.items():
        if estimator_list is not None and name not in estimator_list:
            continue
        task = task_factory("regression")
        cfg, init_params, low_cost_init_params = flaml_config_to_tune_config(
            cls.search_space(data_size=data_size, task=task)
        )
        cfg, init_params = tweak_config(cfg, init_params, name)
        # Test if the estimator instantiates fine
        try:
            cls(task=task, **init_params)
            cfg["estimator_name"] = name
            joint_cfg.append(cfg)
            init_params["estimator_name"] = name
            joint_init_params.append(init_params)
            joint_low_cost_init_params[name] = low_cost_init_params
        except ImportError as e:
            print(f"Error instantiating {name}: {e}")

    return tune.choice(joint_cfg), joint_init_params, joint_low_cost_init_params


def tweak_config(cfg: dict, init_params: dict, estimator_name: str):
    """
    Tweak built-in FLAML search spaces to limit the number of estimators
    :param cfg:
    :param estimator_name:
    :return:
    """
    out = copy.deepcopy(cfg)
    if "xgboost" in estimator_name or estimator_name in [
        "random_forest",
        "extra_trees",
        "lgbm",
        "catboost",
    ]:
        out["n_estimators"] = tune.lograndint(4, 1000)
        init_params["n_estimators"] = 100
    return out, init_params


def model_from_cfg(cfg: dict):
    cfg = copy.deepcopy(cfg)
    model_name = cfg.pop("estimator_name")
    estimator_class = estimators[model_name]

    # Some Econml estimators pass a weights vector as an unnamed third argument,
    # which is not supported by flaml. We need to wrap the estimator to ignore
    # TODO: expose better estimator wrappers that support weights
    class FlamlEstimatorWrapper(estimator_class):
        wrapped_class = estimator_class

        def fit(self, X, y, *args, **kwargs):
            if len(kwargs):
                warnings.warn(f"Extra args {args} {kwargs} are being ignored")
            return self.wrapped_class.fit(self, X, y)

    out = FlamlEstimatorWrapper(task=task_factory("regression"), **cfg)
    return out


def config2score(cfg: dict, X, y):
    model = model_from_cfg(cfg["estimator"])
    model.fit(X, y)
    ypred = model.predict(X)
    err = y - ypred
    return {"score": np.mean(err**2)}


def make_fake_data():

    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters for the DataFrame
    num_samples = 1000  # Number of rows (samples)
    num_features = 5  # Number of features (columns)

    # Generate random float features
    X = np.random.rand(num_samples, num_features)

    # Define the coefficients for each feature to generate the target variable
    coefficients = np.random.rand(num_features)

    # Generate the target variable y as a linear combination of the features plus some noise
    noise = np.random.normal(0, 0.1, num_samples)  # Add some Gaussian noise
    y = np.dot(X, coefficients) + noise

    # Create a DataFrame
    column_names = [f"feature_{i + 1}" for i in range(num_features)]
    df = pd.DataFrame(X, columns=column_names)

    return df, y


if __name__ == "__main__":

    # Create fake data
    X, y = make_fake_data()
    cfg, init_params, low_cost_init_params = joint_config(data_size=X.shape)
    flaml.tune.run(
        evaluation_function=lambda cfgs: config2score(cfgs, X, y),
        metric="score",
        mode="min",
        config={"estimator": cfg},
        points_to_evaluate=init_params,
        num_samples=10,
    )

    print("yay!")
