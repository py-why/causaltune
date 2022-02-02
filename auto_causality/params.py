from copy import deepcopy
from econml.inference import BootstrapInference
from sklearn import linear_model


class SimpleParamService:
    def __init__(
        self,
        propensity_model,
        outcome_model,
        final_model=None,
        conf_intervals: bool = False,
        n_bootstrap_samples: int = 10,
        n_jobs=None,
        max_depth=10,
        n_estimators=500,
        min_leaf_size=10,
    ):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.final_model = final_model
        self.conf_intervals = conf_intervals
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
        bootstrap = BootstrapInference(
            n_bootstrap_samples=self.n_bootstrap_samples, n_jobs=self.n_jobs
        )

        if self.final_model is None:
            final_model = deepcopy(self.outcome_model)
        else:
            final_model = deepcopy(self.final_model)

        configs = {
            "backdoor.propensity_score_weighting": {
                "propensity_score_model": linear_model.LogisticRegression(
                    max_iter=10000
                )
                # "init_params": {
                #     "propensity_score_model": linear_model.LogisticRegression(
                #         max_iter=10000
                #     )
                # },
                # "fit_params": {},
            },
            "backdoor.econml.metalearners.SLearner": {
                "init_params": {
                    "overall_model": outcome_model,
                },
                "fit_params": {"inference": bootstrap} if self.conf_intervals else {},
            },
            "backdoor.econml.metalearners.TLearner": {
                "init_params": {
                    "models": outcome_model,
                },
                "fit_params": {"inference": bootstrap} if self.conf_intervals else {},
            },
            "backdoor.econml.metalearners.XLearner": {
                "init_params": {
                    "propensity_model": propensity_model,
                    "models": outcome_model,
                },
                "fit_params": {"inference": bootstrap} if self.conf_intervals else {},
            },
            "backdoor.econml.metalearners.DomainAdaptationLearner": {
                "init_params": {
                    "propensity_model": propensity_model,
                    "models": outcome_model,
                    "final_models": final_model,
                },
                "fit_params": {"inference": bootstrap} if self.conf_intervals else {},
            },
            "backdoor.econml.dr.ForestDRLearner": {
                "init_params": {
                    "model_propensity": propensity_model,
                    "model_regression": outcome_model,
                    "max_depth": self.max_depth,
                    "n_estimators": self.n_estimators,
                },
                "fit_params": {},
            },
            "backdoor.econml.dr.LinearDRLearner": {
                "init_params": {
                    "model_propensity": propensity_model,
                    "model_regression": outcome_model,
                },
                "fit_params": {},
            },
            "backdoor.econml.dr.SparseLinearDRLearner": {
                "init_params": {
                    "model_propensity": propensity_model,
                    "model_regression": outcome_model,
                },
                "fit_params": {},
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
            },
            "backdoor.econml.dml.CausalForestDML": {
                "init_params": {
                    "model_t": propensity_model,
                    "model_y": outcome_model,
                    "max_depth": self.max_depth,
                    "n_estimators": self.n_estimators,
                    "inference": self.conf_intervals,
                    "discrete_treatment": True,
                },
                "fit_params": {},
            },
            "backdoor.auto_causality.dowhy_wrapper.direct_uplift.DirectUpliftDoWhyWrapper": {
                "init_params": {
                    "propensity_model": propensity_model,
                    "outcome_model": outcome_model,
                },
                "fit_params": {},
            },
            # leaving out DML and NonParamDML as they're base classes for the 3 above
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
                    "max_depth": self.max_depth,
                    "n_trees": self.n_estimators,
                    "min_leaf_size": self.min_leaf_size,
                    "backend": "threading",
                },
                "fit_params": {},
            },
            "backdoor.econml.orf.DMLOrthoForest": {
                "init_params": {
                    "model_T": propensity_model,
                    "model_Y": linear_model.Ridge(
                        alpha=0.01
                    ),  # WeightedLasso(alpha=0.01),  #
                    "discrete_treatment": True,
                    "n_jobs": self.n_jobs,
                    "max_depth": self.max_depth,
                    "n_trees": self.n_estimators,
                    "min_leaf_size": self.min_leaf_size,
                    # Loky was running out of disk space for some reason
                    "backend": "threading",
                },
                "fit_params": {},
            },
        }

        return configs
