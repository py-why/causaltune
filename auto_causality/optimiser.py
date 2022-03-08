import warnings
import pandas as pd
from flaml import tune, AutoML
from sklearn.dummy import DummyClassifier
from auto_causality.params import SimpleParamService
from auto_causality.scoring import make_scores
from typing import List
from dowhy import CausalModel
from auto_causality.r_score import RScoreWrapper


class AutoCausality:
    """Performs AutoML to find best econML estimator.
    Optimises hyperparams of component models of each estimator
    and hyperparams of the estimators themselves. Uses the ERUPT
    metric for estimator selection.

    Example:
    ```python

    estimator_list = [".LinearDML","LinearDRLearner","metalearners"]
    auto_causality = AutoCausality(time_budget=10, estimator_list=estimator_list)

    auto_causality.fit(train_df, test_df, treatment, outcome,
    features_W, features_X)

    print(f"Best estimator: {auto_causality.best_estimator}")

    ```
    """

    def __init__(self, train_df=None, test_df=None, **settings):
        """constructor.

        Args:
            time_budget (float): a number of the time budget in seconds. -1 if no limit.
            num_samples (int): max number of iterations.
            verbose (int):  controls verbosity, higher means more messages. range (0,3). defaults to 0.
            use_ray (bool): use Ray backend (nrequires ray to be installed).
            task (str): task type, defaults to "causal_inference".
            metric (str): metric to optimise, defaults to "ERUPT".
            estimator_list (list): a list of strings for estimator names, or "auto".
               e.g. ```['dml', 'CausalForest']```
            metrics_to_log (list): additional metrics to log. TODO implement
            use_dummyclassifier (bool): use dummy classifier for propensity model or not. defaults to True.
            components_task (str): task for component models. defaults to "regression"
            components_verbose (int): verbosity of component model HPO. range (0,3). defaults to 0.
            components_pred_time_limit (float): prediction time limit for component models
            components_njobs (int): number of concurrent jobs for component model optimisation.
                defaults to -1 (all available cores).
            components_time_budget (float): time budget for HPO of component models in seconds.
                defaults to overall time budget / 2.
        """
        self._settings = settings
        settings["tuner"] = {}
        settings["tuner"]["time_budget_s"] = settings.get("time_budget", 60)
        settings["tuner"]["num_samples"] = settings.get("num_samples", 10)
        settings["tuner"]["verbose"] = settings.get("verbose", 3)
        settings["tuner"]["use_ray"] = settings.get(
            "use_ray", False
        )  # requires ray to be installed via pip install flaml[ray]
        settings["task"] = settings.get("task", "causal_inference")
        settings["metric"] = settings.get("metric", "erupt")
        settings["estimator_list"] = settings.get(
            "estimator_list", "auto"
        )  # if auto, add all estimators that we have implemented

        # params for FLAML on component models:
        settings["use_dummyclassifier"] = settings.get("use_dummyclassifier", True)
        settings["component_models"] = {}
        settings["component_models"]["task"] = settings.get(
            "components_task", "regression"
        )
        settings["component_models"]["verbose"] = settings.get("components_verbose", 0)
        settings["component_models"]["pred_time_limit"] = settings.get(
            "components_pred_time_limit", 10 / 1e6
        )
        settings["component_models"]["n_jobs"] = settings.get("components_nbjobs", -1)
        settings["component_models"]["time_budget"] = settings.get(
            "components_time_budget", int(settings["tuner"]["time_budget_s"] / 2)
        )

        # user can choose between flaml and dummy for propensity model.
        self.propensity_model = (
            DummyClassifier(strategy="prior")
            if settings["use_dummyclassifier"]
            else AutoML(**settings["component_models"])
        )
        self.outcome_model = AutoML(**settings["component_models"])

        # config with method-specific params
        self.cfg = SimpleParamService(self.propensity_model, self.outcome_model,)

        self.estimates = {}
        self.results = {}
        self.estimator_list = self._create_estimator_list()

        self.train_df = train_df or pd.DataFrame()
        self.test_df = test_df or pd.DataFrame()
        self.causal_model = None
        self.identified_estimand = None

        # trained component models for each estimator
        self.trained_estimators_dict = {}

    def get_params(self, deep=False):
        return self._settings.copy()

    def get_estimators(self, deep=False):
        return self.estimator_list.copy()

    def _create_estimator_list(self):
        """Creates list of estimators via substring matching
        - Retrieves list of available estimators,
        - Returns all available estimators is provided list empty or set to 'auto'.
        - Returns only requested estimators otherwise.
        - Checks for and removes duplicates """

        # get list of available estimators:
        available_estimators = []
        for estimator in self.cfg.estimators():
            if any(
                [
                    e in estimator
                    for e in [
                        "metalearners",
                        "CausalForestDML",
                        ".LinearDML",
                        "SparseLinearDML",
                        "ForestDRLearner",
                        "LinearDRLearner",
                        "Ortho",
                    ]
                ]
            ):
                available_estimators.append(estimator)

        # match list of requested estimators against list of available estimators
        # and remove duplicates:
        # NOTE: black formatter conflicts with flake as it moves binary operator to 2nd
        # line and hence triggers W503 ....
        if (
            self._settings["estimator_list"] == "auto"
            or self._settings["estimator_list"] == []  # noqa: W503
        ):
            warnings.warn("No estimators specified, adding all available estimators...")
            return available_estimators
        elif self._verify_estimator_list():
            estimators_to_use = list(
                dict.fromkeys(
                    [
                        available_estimator
                        for requested_estimator in self._settings["estimator_list"]
                        for available_estimator in available_estimators
                        if requested_estimator in available_estimator
                    ]
                )
            )
            if estimators_to_use == []:
                warnings.warn(
                    "requested estimators not implemented, continuing with defaults"
                )
                return available_estimators
            else:
                return estimators_to_use
        else:
            warnings.warn("invalid estimator list requested, continuing with defaults")
            return available_estimators

    def _verify_estimator_list(self):
        """verifies that provided estimator list is in correct format
        """
        if not isinstance(self._settings["estimator_list"], list):
            return False
        else:
            for e in self._settings["estimator_list"]:
                if not isinstance(e, str):
                    return False
        return True

    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        treatment: str,
        outcome: str,
        common_causes: List[str],
        effect_modifiers: List[str],
    ):
        """Performs AutoML on list of causal inference estimators
        - If estimator has a search space specified in its parameters, HPO is performed on the whole model.
        - Otherwise, only its component models are optimised

        Args:
            train_df (pd.DataFrame): Training Data
            test_df (pd.DataFrame): Test Data
            treatment (str): name of treatment variable
            outcome (str): name of outcome variable
            common_causes (List[str]): list of names of common causes
            effect_modifiers (List[str]): list of names of effect modifiers
        """

        self.train_df = train_df
        self.test_df = test_df
        self.causal_model = CausalModel(
            data=self.train_df,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes,
            effect_modifiers=effect_modifiers,
        )
        self.identified_estimand = self.causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )
        
        self.r_scorer = RScoreWrapper(self.outcome_model, self.propensity_model, train_df, test_df, outcome, 
                              treatment, common_causes, effect_modifiers)


        if self._settings["tuner"]["verbose"] > 0:
            print(f"fitting estimators: {self.estimator_list}")

        self.tune_results = (
            {}
        )  # We need to keep track of the tune results to access the best config
        for estimator in self.estimator_list:
            self.estimator = estimator
            self.estimator_cfg = self.cfg.method_params(estimator)
            if self.estimator_cfg["search_space"] == {}:
                self._estimate_effect()
                scores = self._compute_metrics()
                self.results[self.estimator] = scores["test"][
                    self._settings["metric"].lower()
                ]
            else:
                results = tune.run(
                    self._tune_with_config,
                    self.estimator_cfg["search_space"],
                    resources_per_trial={"cpu": 1, "gpu": 0.5},
                    metric=self._settings["metric"],
                    mode="max",
                    low_cost_partial_config={},
                    **self._settings["tuner"],
                )

                # log results
                best_trial = results.get_best_trial()
                if best_trial is None:
                    # if hpo didn't converge for some reason, just log None
                    self.results[self.estimator] = None
                else:
                    self.results[self.estimator] = best_trial.last_result[
                        self._settings["metric"]
                    ]
            self.tune_results[estimator] = results
            if self._settings["tuner"]["verbose"] > 0:
                print(f"... Estimator: {self.estimator}")
                for metric in ["erupt", "qini", "auc", "r_score", "ATE"]:
                    if not(best_trial.last_result[metric] is None):
                        print(f" {metric}: {best_trial.last_result[metric]:6f}")

    def _tune_with_config(self, config: dict) -> dict:
        """Performs Hyperparameter Optimisation for a
        causal inference estimator

        Args:
            config (dict): dictionary with search space for
            all tunable parameters

        Returns:
            dict: values of metrics after optimisation
        """
        # add params that are tuned by flaml:
        self.estimator_cfg["init_params"] = {
            **self.estimator_cfg["init_params"],
            **config,
        }
        # estimate effect with current config
        self._estimate_effect()

        # compute a metric and return results
        scores = self._compute_metrics()
        results = {
            "erupt": scores["test"]["erupt"],
            "qini": scores["test"]["qini"],
            "auc": scores["test"]["auc"],
            "r_score": scores["test"]["r_score"],
            "ATE": scores["test"]["ate"],
        }
        return results

    def _estimate_effect(self):
        """estimates effect with chosen estimator
        """
        if hasattr(self, "estimator"):
            self.estimates[self.estimator] = self.causal_model.estimate_effect(
                self.identified_estimand,
                method_name=self.estimator,
                control_value=0,
                treatment_value=1,
                target_units="ate",  # condition used for CATE
                confidence_intervals=False,
                method_params=self.estimator_cfg,
            )
            # Store the fitted Econml estimator
            self.trained_estimators_dict[self.estimator] = self.estimates[
                self.estimator
            ].estimator.estimator
        else:
            raise AttributeError("No estimator for causal model specified")

    def _compute_metrics(self) -> dict:
        """ computes metrics to score causal estimators"""
        try:
            te_train = self.estimates[self.estimator].cate_estimates
            X_test = self.test_df[
                self.estimates[self.estimator].estimator._effect_modifier_names
            ]
            te_test = (
                self.estimates[self.estimator]
                .estimator.estimator.effect(X_test)
                .flatten()
            )
        except Exception:
            te_train = self.estimates[self.estimator].estimator.effect(self.train_df)
            te_test = self.estimates[self.estimator].estimator.effect(self.test_df)

        scores = {
            "estimator": self.estimator,
            "train": make_scores(
                self.estimates[self.estimator], self.train_df, te_train, r_scorer=self.r_scorer.train
            ),
            "test": make_scores(
                self.estimates[self.estimator], self.test_df, te_test, r_scorer=self.r_scorer.test
            ),
        }
        return scores

    @property
    def best_estimator(self) -> str:
        """A string indicating the best estimator found
        """
        return max(self.results, key=self.results.get)

    @property
    def model(self):
        """Return the *trained* best estimator
        """
        # TODO

        return self.best_model_for_estimator(self.best_estimator)

    def best_model_for_estimator(self, estimator_name):
        """Return the best model found for a particular estimator.

        Args:
            estimator_name: a str of the estimator's name.

        Returns:
            An object storing the best model for estimator_name.
        """
        # TODO
        # Note that this returns the trained Econml estimator, whose attributes include
        # fitted  models for E[T | X, W], for E[Y | X, W], CATE model, etc.
        return self.trained_estimators_dict[estimator_name]

    @property
    def best_config(self):
        """A dictionary containing the best configuration"""
        return self.best_config_per_estimator[self.best_estimator]

    @property
    def best_config_per_estimator(self):
        """A dictionary of all estimators' best configuration."""
        return {
            estimator: self.tune_results[estimator].best_config
            for estimator in self.estimator_list
            if estimator in self.tune_results
        }

    @property
    def best_loss_per_estimator(self):
        """A dictionary of all estimators' best loss."""
        # TODO
        return None

    @property
    def best_loss(self):
        """A float of the best loss found."""
        # TODO
        return None
