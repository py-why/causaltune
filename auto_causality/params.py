from flaml import tune
from copy import deepcopy
from typing import Optional

from econml.inference import BootstrapInference  # noqa F401
from sklearn import linear_model


class SimpleParamService:
    def __init__(
        self,
        propensity_model,
        outcome_model,
        final_model=None,
        n_bootstrap_samples: Optional[int] = None,
        n_jobs: Optional[int] = None,
        max_depth=10,
        n_estimators=500,
        min_leaf_size=10,
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.final_model = final_model
        self.n_bootstrap_samples = n_bootstrap_samples
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_leaf_size = min_leaf_size

    def estimators(self):
        return list(self._configs().keys())

    def method_params(
        self,
        estimator: str,
    ):
        return self._configs()[estimator]

    def _configs(self):
        propensity_model = deepcopy(self.propensity_model)
        outcome_model = deepcopy(self.outcome_model)
        if self.n_bootstrap_samples is not None:
            # TODO Egor please look into this
            # bootstrap is causing recursion errors (see notes below)
            # bootstrap = BootstrapInference(
            #     n_bootstrap_samples=self.n_bootstrap_samples, n_jobs=self.n_jobs
            # )
            pass

        if self.final_model is None:
            final_model = deepcopy(self.outcome_model)
        else:
            final_model = deepcopy(self.final_model)

        configs = {
            "backdoor.propensity_score_weighting": {
                "propensity_score_model": linear_model.LogisticRegression(
                    max_iter=10000
                ),
                "search_space": {}
                # "init_params": {
                #     "propensity_score_model":
                #       linear_model.LogisticRegression(
                #         max_iter=10000
                #     )
                # },
                # "fit_params": {},
            },
            "backdoor.econml.metalearners.SLearner": {
                "init_params": {
                    "overall_model": outcome_model,
                },
                "fit_params": {},
                "search_space": {}
                # TODO Egor please look into this
                # These lines cause recursion errors
                # if self.n_bootstrap_samples is None
                # else {"inference": bootstrap},
            },
            "backdoor.econml.metalearners.TLearner": {
                "init_params": {
                    "models": outcome_model,
                },
                "fit_params": {},
                "search_space": {}
                # TODO Egor please look into this
                # These lines cause recursion errors
                # if self.n_bootstrap_samples is None
                # else {"inference": bootstrap},
            },
            "backdoor.econml.metalearners.XLearner": {
                "init_params": {
                    "propensity_model": propensity_model,
                    "models": outcome_model,
                },
                "fit_params": {},
                "search_space": {}
                # TODO Egor please look into this
                # These lines cause recursion errors
                # if self.n_bootstrap_samples is None
                # else {"inference": bootstrap},
            },
            "backdoor.econml.metalearners.DomainAdaptationLearner": {
                "init_params": {
                    "propensity_model": propensity_model,
                    "models": outcome_model,
                    "final_models": final_model,
                },
                "fit_params": {},
                "search_space": {}
                # TODO Egor please look into this
                # These lines cause recursion errors
                # if self.n_bootstrap_samples is None
                # else {"inference": bootstrap},
            },
            "backdoor.econml.dr.ForestDRLearner": {
                "init_params": {
                    "model_propensity": propensity_model,
                    "model_regression": outcome_model,
                    # "max_depth": self.max_depth,
                    # "n_estimators": self.n_estimators,
                },
                "fit_params": {},
                "search_space": {
                    "min_propensity": tune.loguniform(1e-6, 1e-1),
                    "mc_iters": tune.randint(0, 10),
                    "n_estimators": tune.randint(2, 500),
                    "max_depth": tune.randint(2, 1000),
                    "min_samples_split": tune.randint(1, 50),
                    "min_samples_leaf": tune.randint(1, 25),
                    "min_weight_fraction_leaf": tune.uniform(0, 0.5),
                    "max_features": tune.choice(["auto", "sqrt", "log2", None]),
                    "min_impurity_decrease": tune.uniform(0, 10),
                    "max_samples": tune.uniform(0, 0.5),
                    "min_balancedness_tol": tune.uniform(0, 0.5),
                    "honest": tune.choice([0, 1]),
                    "subforest_size": tune.randint(1, 10),
                },
            },
            "backdoor.econml.dr.LinearDRLearner": {
                "init_params": {
                    "model_propensity": propensity_model,
                    "model_regression": outcome_model,
                },
                "fit_params": {},
                "search_space": {
                    "fit_cate_intercept": tune.choice([0, 1]),
                    "min_propensity": tune.loguniform(1e-6, 1e-1),
                    "mc_iters": tune.randint(0, 10),
                },
            },
            "backdoor.econml.dr.SparseLinearDRLearner": {
                "init_params": {
                    "model_propensity": propensity_model,
                    "model_regression": outcome_model,
                },
                "fit_params": {},
                "search_space": {
                    "fit_cate_intercept": tune.choice([0, 1]),
                    "n_alphas": tune.lograndint(1, 1000),
                    "n_alphas_cov": tune.lograndint(1, 100),
                    "min_propensity": tune.loguniform(1e-6, 1e-1),
                    "mc_iters": tune.randint(0, 10),
                    "tol": tune.qloguniform(1e-7, 1, 1e-7),
                    "max_iter": tune.qlograndint(100, 100000, 100),
                    "mc_agg": tune.choice(["mean", "median"]),
                },
            },
            "backdoor.econml.dml.LinearDML": {
                "init_params": {
                    "model_t": propensity_model,
                    "model_y": outcome_model,
                    "discrete_treatment": True,
                    # it runs out of memory fast if the below is not set
                    "linear_first_stages": False,
                },
                "fit_params": {},
                "search_space": {
                    "fit_cate_intercept": tune.choice([0, 1]),
                    "mc_iters": tune.randint(0, 10),
                },
            },
            "backdoor.econml.dml.SparseLinearDML": {
                "init_params": {
                    "model_t": propensity_model,
                    "model_y": outcome_model,
                    "discrete_treatment": True,
                    # it runs out of memory fast if the below is not set
                    "linear_first_stages": False,
                },
                "fit_params": {},
                "search_space": {
                    "fit_cate_intercept": tune.choice([0, 1]),
                    "mc_iters": tune.randint(0, 10),
                    "n_alphas": tune.lograndint(1, 1000),
                    "n_alphas_cov": tune.lograndint(1, 100),
                    "tol": tune.qloguniform(1e-7, 1, 1e-7),
                    "max_iter": tune.qlograndint(100, 100000, 100),
                },
            },
            "backdoor.econml.dml.CausalForestDML": {
                "init_params": {
                    "model_t": propensity_model,
                    "model_y": outcome_model,
                    # "max_depth": self.max_depth,
                    # "n_estimators": self.n_estimators,
                    "discrete_treatment": True,
                    "inference": False,
                },
                "fit_params": {},
                "search_space": {
                    "mc_iters": tune.randint(0, 10),
                    "drate": tune.choice([0, 1]),
                    "n_estimators": tune.randint(2, 500),
                    "criterion": tune.choice(["mse", "het"]),
                    "max_depth": tune.randint(2, 1000),
                    "min_samples_split": tune.randint(1, 50),
                    "min_samples_leaf": tune.randint(1, 25),
                    "min_weight_fraction_leaf": tune.uniform(0, 0.5),
                    "min_var_fraction_leaf": tune.uniform(0, 1),
                    "max_features": tune.choice(["auto", "sqrt", "log2", None]),
                    "min_impurity_decrease": tune.uniform(0, 10),
                    "max_samples": tune.uniform(0, 1),
                    "min_balancedness_tol": tune.uniform(0, 0.5),
                    "honest": tune.choice([0, 1]),
                    # "inference": tune.choice([0, 1]),
                    "fit_intercept": tune.choice([0, 1]),
                    # Difficult as needs to be a factor of 'n_estimators'
                    "subforest_size": tune.randint(1, 10),
                },
            },
            "backdoor.auto_causality.models.TransformedOutcome": {
                "init_params": {
                    "propensity_model": propensity_model,
                    "outcome_model": outcome_model,
                },
                "fit_params": {},
                "search_space": {},
            },
            # leaving out DML and NonParamDML as they're base classes for the 3
            # above
            #
            # This one breaks when running, need to figure out why
            # "backdoor.econml.dr.DRLearner": {
            #     "init_params": {
            #         "model_propensity": propensity_model,
            #         "model_regression": outcome_model,
            #         "model_final": final_model,
            #     },
            #     "fit_params": {},
            # },
            "backdoor.econml.orf.DROrthoForest": {
                "init_params": {
                    "propensity_model": propensity_model,
                    "model_Y": linear_model.Ridge(
                        alpha=0.01
                    ),  # WeightedLasso(alpha=0.01),  #
                    "n_jobs": self.n_jobs,
                    # "max_depth": self.max_depth,
                    # "n_trees": self.n_estimators,
                    # "min_leaf_size": self.min_leaf_size,
                    "backend": "threading",
                },
                "fit_params": {},
                "search_space": {
                    "n_trees": tune.randint(2, 750),
                    "min_leaf_size": tune.randint(1, 50),
                    "max_depth": tune.randint(2, 1000),
                    "subsample_ratio": tune.uniform(0, 1),
                    # "bootstrap": tune.choice([0, 1]),
                    "lambda_reg": tune.uniform(0, 1),
                },
            },
            "backdoor.econml.orf.DMLOrthoForest": {
                "init_params": {
                    "model_T": propensity_model,
                    "model_Y": linear_model.Ridge(
                        alpha=0.01
                    ),  # WeightedLasso(alpha=0.01),  #
                    "discrete_treatment": True,
                    "n_jobs": self.n_jobs,
                    # "max_depth": self.max_depth,
                    # "n_trees": self.n_estimators,
                    # "min_leaf_size": self.min_leaf_size,
                    # Loky was running out of disk space for some reason
                    "backend": "threading",
                },
                "fit_params": {},
                "search_space": {
                    "n_trees": tune.randint(2, 750),
                    "min_leaf_size": tune.randint(1, 50),
                    "max_depth": tune.randint(2, 1000),
                    "subsample_ratio": tune.uniform(0, 1),
                    # "bootstrap": tune.choice([0, 1]),
                    "lambda_reg": tune.uniform(0, 1),
                },
            },
        }

        return configs
