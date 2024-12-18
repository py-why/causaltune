from sklearn.linear_model import ElasticNet, LassoLars


from flaml.automl.model import SKLearnEstimator
from flaml import tune

# These models are for some reason not in the deployed version of flaml 2.2.0,
# but in the source code they are there
# So keep this file in the project for now


class ElasticNetEstimator(SKLearnEstimator):
    """The class for tuning Elastic Net regression model."""

    """Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html"""

    ITER_HP = "max_iter"

    @classmethod
    def search_space(cls, data_size, task="regression", **params):
        return {
            "alpha": {
                "domain": tune.loguniform(lower=0.0001, upper=1.0),
                "init_value": 0.1,
            },
            "l1_ratio": {
                "domain": tune.uniform(lower=0.0, upper=1.0),
                "init_value": 0.5,
            },
            "selection": {
                "domain": tune.choice(["cyclic", "random"]),
                "init_value": "cyclic",
            },
        }

    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        params["tol"] = params.get("tol", 0.0001)
        if "n_jobs" in params:
            params.pop("n_jobs")
        return params

    def __init__(self, task="regression", **config):
        super().__init__(task, **config)
        assert self._task.is_regression(), "ElasticNet for regression task only"
        self.estimator_class = ElasticNet


class LassoLarsEstimator(SKLearnEstimator):
    """The class for tuning Lasso model fit with Least Angle Regression a.k.a. Lars."""

    """Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html"""

    ITER_HP = "max_iter"

    @classmethod
    def search_space(cls, task=None, **params):
        return {
            "alpha": {
                "domain": tune.loguniform(lower=1e-4, upper=1.0),
                "init_value": 0.1,
            },
            "fit_intercept": {
                "domain": tune.choice([True, False]),
                "init_value": True,
            },
            "eps": {
                "domain": tune.loguniform(lower=1e-16, upper=1e-4),
                "init_value": 2.220446049250313e-16,
            },
        }

    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        if "n_jobs" in params:
            params.pop("n_jobs")
        return params

    def __init__(self, task="regression", **config):
        super().__init__(task, **config)
        assert self._task.is_regression(), "LassoLars for regression task only"
        self.estimator_class = LassoLars

    def predict(self, X, **kwargs):
        X = self._preprocess(X)
        return self._model.predict(X, **kwargs)
