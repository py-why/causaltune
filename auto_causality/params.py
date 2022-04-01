
import warnings
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
        requested_estimators="auto",
        blacklisted_estimators: Optional[list] = None,
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
        self.requested_estimators = requested_estimators
        self.blacklisted_estimators = blacklisted_estimators

    def estimators(self) -> list:
        """returns list of requested estimators, filtered by availability and
        whether they had beeen blacklisted (i.e. are experimental)

        Returns:
            list: list of estimator names
        """
        available_estimators = self._create_estimator_list()
        return available_estimators

    def _create_estimator_list(self) -> list:
        """Creates list of estimators via substring matching
        - Retrieves list of available estimators,
        - Returns all available estimators if provided list empty or set to 'auto'.
        - Returns only requested estimators otherwise.
        - Checks for and removes duplicates"""

        # get list of available estimators:
        all_estimators = list(self._configs().keys())
        # remove blacklisted estimators
        if not (self.blacklisted_estimators is None):
            available_estimators = [
                est
                for est in all_estimators
                if not (est in self.blacklisted_estimators)
            ]
        else:
            available_estimators = all_estimators

        # match list of requested estimators against list of available estimators
        # and remove duplicates:
        if (
            self.requested_estimators == "auto"
            or self.requested_estimators == []
        ):
            warnings.warn("Using all available estimators...")
            return available_estimators

        elif self._verify_estimator_list():
            estimators_to_use = list(
                dict.fromkeys(
                    [
                        available_estimator
                        for requested_estimator in self.requested_estimators
                        for available_estimator in available_estimators
                        if requested_estimator in available_estimator
                    ]
                )
            )
            if estimators_to_use == []:
                raise ValueError(
                    "No valid estimators in" + str(self.requested_estimators)
                )
            else:
                return estimators_to_use
        else:
            warnings.warn("invalid estimator list requested, continuing with defaults")
            return available_estimators

    def _verify_estimator_list(self):
        """verifies that provided estimator list is in correct format"""
        if not isinstance(self.requested_estimators, list):
            return False
        else:
            for e in self.requested_estimators:
                if not isinstance(e, str):
                    return False
        return True

    def method_params(
        self, estimator: str,
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
            "backdoor.auto_causality.models.Dummy": {
                "init_params": {},
                "fit_params": {},
                "search_space": {},
            },
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
                    # putting these here for now, until default values can be reconciled with search space
                    "mc_iters": None,
                    "max_depth": None,
                },
                "fit_params": {},
                "search_space": {
                    "min_propensity": tune.loguniform(1e-6, 1e-1),
                    # "mc_iters": tune.randint(0, 10),
                    "n_estimators": tune.randint(2, 500),
                    # "max_depth": tune.randint(2, 1000),
                    "min_samples_split": tune.randint(2, 20),
                    "min_samples_leaf": tune.randint(1, 25),
                    "min_weight_fraction_leaf": tune.uniform(0, 0.5),
                    "max_features": tune.choice(["auto", "sqrt", "log2", None]),
                    "min_impurity_decrease": tune.uniform(0, 10),
                    "max_samples": tune.uniform(0, 0.5),
                    "min_balancedness_tol": tune.uniform(0, 0.5),
                    "honest": tune.choice([0, 1]),
                    "subforest_size": tune.randint(2, 10),
                },
                "defaults": {
                    "min_propensity": 1e-6,
                    "n_estimators": 1000,  # this is 1000 by default?
                    "min_samples_split": 5,
                    "min_samples_leaf": 5,
                    "min_weight_fraction_leaf": 0.0,
                    "max_features": "auto",
                    "min_impurity_decrease": 0.0,
                    "max_samples": 0.45,
                    "min_balancedness_tol": 0.45,
                    "honest": True,
                    "subforest_size": 4,
                },
            },
            "backdoor.econml.dr.LinearDRLearner": {
                "init_params": {
                    "model_propensity": propensity_model,
                    "model_regression": outcome_model,
                    "mc_iters": None,
                },
                "fit_params": {},
                "search_space": {
                    "fit_cate_intercept": tune.choice([0, 1]),
                    "min_propensity": tune.loguniform(1e-6, 1e-1),
                    # "mc_iters": tune.randint(0, 10),
                },
                "defaults": {
                    "fit_cate_intercept": True,
                    "min_propensity": 1e-6,
                },
            },
            "backdoor.econml.dr.SparseLinearDRLearner": {
                "init_params": {
                    "model_propensity": propensity_model,
                    "model_regression": outcome_model,
                    "mc_iters": None,
                },
                "fit_params": {},
                "search_space": {
                    "fit_cate_intercept": tune.choice([0, 1]),
                    "n_alphas": tune.lograndint(1, 1000),
                    "n_alphas_cov": tune.lograndint(1, 100),
                    "min_propensity": tune.loguniform(1e-6, 1e-1),
                    # "mc_iters": tune.randint(0, 10),
                    "tol": tune.qloguniform(1e-7, 1, 1e-7),
                    "max_iter": tune.qlograndint(100, 100000, 100),
                    "mc_agg": tune.choice(["mean", "median"]),
                },
                "defaults": {
                    "fit_cate_intercept": True,
                    "n_alphas": 100,
                    "n_alphas_cov": 10,
                    "min_propensity": 1e-6,
                    "tol": 0.0001,
                    "max_iter": 10000,
                    "mc_agg": "mean",
                },
            },
            "backdoor.econml.dml.LinearDML": {
                "init_params": {
                    "model_t": propensity_model,
                    "model_y": outcome_model,
                    "discrete_treatment": True,
                    # it runs out of memory fast if the below is not set
                    "linear_first_stages": False,
                    "mc_iters": None,
                },
                "fit_params": {},
                "search_space": {
                    "fit_cate_intercept": tune.choice([0, 1]),
                    # "mc_iters": tune.randint(0, 10),
                    "mc_agg": tune.choice(["mean", "median"]),
                },
                "defaults": {
                    "fit_cate_intercept": True,
                    "mc_agg": "mean",
                },
            },
            "backdoor.econml.dml.SparseLinearDML": {
                "init_params": {
                    "model_t": propensity_model,
                    "model_y": outcome_model,
                    "discrete_treatment": True,
                    # it runs out of memory fast if the below is not set
                    "linear_first_stages": False,
                    "mc_iters": None,
                },
                "fit_params": {},
                "search_space": {
                    "fit_cate_intercept": tune.choice([0, 1]),
                    # "mc_iters": tune.randint(0, 10),
                    "n_alphas": tune.lograndint(1, 1000),
                    "n_alphas_cov": tune.lograndint(1, 100),
                    "tol": tune.qloguniform(1e-7, 1, 1e-7),
                    "max_iter": tune.qlograndint(100, 100000, 100),
                    "mc_agg": tune.choice(["mean", "median"]),
                },
                "defaults": {
                    "fit_cate_intercept": True,
                    "n_alphas": 100,
                    "n_alphas_cov": 10,
                    "tol": 0.0001,
                    "max_iter": 10000,
                    "mc_agg": "mean",
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
                    "mc_iters": None,
                    "max_depth": None,
                    "min_var_fraction_leaf": None,
                },
                "fit_params": {},
                "search_space": {
                    # "mc_iters": tune.randint(0, 10),
                    "drate": tune.choice([0, 1]),
                    "n_estimators": tune.randint(2, 500),
                    "criterion": tune.choice(["mse", "het"]),
                    # "max_depth": tune.randint(2, 1000),
                    "min_samples_split": tune.randint(2, 30),
                    "min_samples_leaf": tune.randint(1, 25),
                    "min_weight_fraction_leaf": tune.uniform(0, 0.5),
                    # "min_var_fraction_leaf": tune.uniform(0, 1),
                    "max_features": tune.choice(["auto", "sqrt", "log2", None]),
                    "min_impurity_decrease": tune.uniform(0, 10),
                    "max_samples": tune.uniform(0, 1),
                    "min_balancedness_tol": tune.uniform(0, 0.5),
                    "honest": tune.choice([0, 1]),
                    # "inference": tune.choice([0, 1]),
                    "fit_intercept": tune.choice([0, 1]),
                    # Difficult as needs to be a factor of 'n_estimators'
                    "subforest_size": tune.randint(2, 10),
                },
                "defaults": {
                    # "mc_iters": tune.randint(0, 10),
                    "drate": True,
                    "n_estimators": 100,
                    "criterion": "mse",
                    "min_samples_split": 10,
                    "min_samples_leaf": 5,
                    "min_weight_fraction_leaf": 0.0,
                    "max_features": "auto",
                    "min_impurity_decrease": 0.0,
                    "max_samples": 0.45,
                    "min_balancedness_tol": 0.45,
                    "honest": True,
                    # "inference": tune.choice([0, 1]),
                    "fit_intercept": True,
                    "subforest_size": 4,
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
                "defaults": {
                    "n_trees": 500,
                    "min_leaf_size": 10,
                    "max_depth": 10,
                    "subsample_ratio": 0.7,
                    "lambda_reg": 0.01,
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
                "defaults": {
                    "n_trees": 500,
                    "min_leaf_size": 10,
                    "max_depth": 10,
                    "subsample_ratio": 0.7,
                    # "bootstrap": tune.choice([0, 1]),
                    "lambda_reg": 0.01,
                },
            },
        }

        return configs
