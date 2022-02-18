import sys
import os
import pandas as pd
from flaml import tune, AutoML
from sklearn.dummy import DummyClassifier
from auto_causality.params import SimpleParamService
from auto_causality.scoring import make_scores
from typing import List

root_path = root_path = os.path.realpath("../../..")
sys.path.append(os.path.join(root_path, "dowhy"))
from dowhy import CausalModel  # noqa: E402


class AutoCausality:
    """Performs AutoML to find best econML estimator.
    Optimises hyperparams of component models of each estimator
    and hyperparams of the estimators themselves. Uses the ERUPT
    metric for estimator selection.

    Example:
    ```python

    estimator_list = [".LinearDML","LinearDRLearner","metalearners"]
    auto_causality = AutoCausality(time_budget=10,components_time_budget=10,estimator_list=estimator_list)

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
        settings["tuner"]["verbose"] = settings.get("verbose", 0)
        settings["tuner"]["use_ray"] = settings.get(
            "use_ray", False
        )  # requires ray to be installed...
        settings["task"] = settings.get("task", "causal_inference")
        settings["metric"] = settings.get("metric", "ERUPT")
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
        self.cfg = SimpleParamService(
            self.propensity_model,
            self.outcome_model,
        )

        self.estimates = {}
        self.results = {}
        self.estimator_list = self.create_estimator_list()

        self.train_df = train_df or pd.DataFrame()
        self.test_df = test_df or pd.DataFrame()
        self.causal_model = None
        self.identified_estimand = None

    def get_params(self, deep=False):
        return self._settings.copy()

    def create_estimator_list(self):
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
            print("No estimators specified, adding all available estimators...")
            return available_estimators
        else:
            estimators_to_use = list(
                dict.fromkeys(
                    [
                        ae
                        for re in self._settings["estimator_list"]
                        for ae in available_estimators
                        if re in ae
                    ]
                )
            )
            return estimators_to_use

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
                    metric=self._settings["metric"],
                    mode="max",
                    **self._settings["tuner"],
                )

                # log results
                self.results[self.estimator] = results.best_trial.last_result[
                    self._settings["metric"]
                ]

            print(
                f"... Estimator: {self.estimator} \t {self._settings['metric']}: {self.results[self.estimator]:6f}"
            )

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
        return {"ERUPT": scores["test"]["erupt"], "ATE": scores["test"]["ate"]}

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
                self.estimates[self.estimator], self.train_df, te_train
            ),
            "test": make_scores(self.estimates[self.estimator], self.test_df, te_test),
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

        return None

    def best_model_for_estimator(self, estimator_name):
        """Return the best model found for a particular estimator.

        Args:
            estimator_name: a str of the estimator's name.

        Returns:
            An object storing the best model for estimator_name.
        """
        # TODO
        return None

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
