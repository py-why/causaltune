from copy import deepcopy
from typing import List, Optional, Union
import traceback

import pandas as pd
import numpy as np

from flaml import tune
from flaml import AutoML as FLAMLAutoML
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from dowhy import CausalModel
from joblib import Parallel, delayed

from auto_causality.params import SimpleParamService
from auto_causality.scoring import make_scores, best_score_by_estimator
from auto_causality.r_score import RScoreWrapper
from auto_causality.utils import clean_config


# this is needed for smooth calculation of Shapley values in DomainAdaptationLearner
class AutoML(FLAMLAutoML):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


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

    def __init__(
        self,
        data_df=None,
        metric="erupt",
        metrics_to_report=None,
        time_budget=300,
        verbose=3,
        use_ray=False,
        estimator_list="auto",
        train_size=0.8,
        num_samples=-1,
        test_size=None,
        use_dummyclassifier=True,
        components_task="regression",
        components_verbose=0,
        components_pred_time_limit=10 / 1e6,
        components_njobs=-1,
        components_time_budget=20,
        try_init_configs=True,
        resources_per_trial=None,
        include_experimental_estimators=False,
    ):
        """constructor.

        Args:
            data_df (pandas.DataFrame): dataset to perform causal inference on
            metric (str): metric to optimise. Defaults to "erupt".
            metrics_to_report (list). additional metrics to compute and report.
             Defaults to ["qini","auc","ate","r_score"]
            time_budget (float): a number of the time budget in seconds. -1 if no limit.
            num_samples (int): max number of iterations.
            verbose (int):  controls verbosity, higher means more messages. range (0,3). Defaults to 0.
            use_ray (bool): use Ray backend (nrequires ray to be installed).
            estimator_list (list): a list of strings for estimator names,
             or "auto" for a recommended subset, "all" for all, or a list of substrings of estimator names
               e.g. ```['dml', 'CausalForest']```
            train_size (float): Fraction of data used for training set. Defaults to 0.5.
            test_size (float): Optional size of test dataset. Defaults to None.
            use_dummyclassifier (bool): use dummy classifier for propensity model or not. Defaults to True.
            components_task (str): task for component models. Defaults to "regression".
            components_verbose (int): verbosity of component model HPO. range (0,3). Defaults to 0.
            components_pred_time_limit (float): prediction time limit for component models
            components_njobs (int): number of concurrent jobs for component model optimisation.
                Defaults to -1 (all available cores).
            components_time_budget (float): time budget for HPO of component models in seconds.
                Defaults to overall time budget / 2.
            try_init_configs (bool): try list of good performing estimators before continuing with HPO.
                Defaults to False.
            blacklisted_estimators (list): [optional] list of estimators not to include in fitting
        """
        self._settings = {}
        self._settings["tuner"] = {}
        self._settings["tuner"]["time_budget_s"] = time_budget
        self._settings["tuner"]["num_samples"] = num_samples
        self._settings["tuner"]["verbose"] = verbose
        self._settings["tuner"][
            "use_ray"
        ] = use_ray  # requires ray to be installed via pip install flaml[ray]
        self._settings["tuner"]["resources_per_trial"] = (
            resources_per_trial if resources_per_trial is not None else {"cpu": 0.5}
        )

        self._settings["try_init_configs"] = try_init_configs

        self.metric = metric
        if metric not in ["erupt", "norm_erupt", "qini", "auc", "ate", "r_score"]:
            raise ValueError(
                f'Metric, {metric}, must be\
                 one of "erupt","norm_erupt","qini","auc","ate" or "r_score"'
            )
        self.metrics_to_report = (
            metrics_to_report
            if metrics_to_report is not None
            else [
                "qini",
                "auc",
                "ate",
                "erupt",
                "norm_erupt",
            ]  # not "r_score" by default
        )
        if self.metric not in self.metrics_to_report:
            self.metrics_to_report.append(self.metric)
        for i in self.metrics_to_report:
            if i not in ["erupt", "norm_erupt", "qini", "auc", "ate", "r_score"]:
                raise ValueError(
                    f'Metric for report, {i}, must be\
                     one of "erupt","norm_erupt","qini","auc","ate" or "r_score"'
                )

        # params for FLAML on component models:
        self._settings["use_dummyclassifier"] = use_dummyclassifier
        self._settings["component_models"] = {}
        self._settings["component_models"]["task"] = components_task
        self._settings["component_models"]["verbose"] = components_verbose
        self._settings["component_models"][
            "pred_time_limit"
        ] = components_pred_time_limit
        self._settings["component_models"]["n_jobs"] = components_njobs
        self._settings["component_models"]["time_budget"] = (
            components_time_budget
            if components_time_budget < time_budget
            else (time_budget // 2) + 1
        )
        self._settings["train_size"] = train_size
        self._settings["test_size"] = test_size

        # user can choose between flaml and dummy for propensity model.
        self.propensity_model = (
            DummyClassifier(strategy="prior")
            if self._settings["use_dummyclassifier"]
            else AutoML(**self._settings["component_models"])
        )

        self.outcome_model = AutoML(**self._settings["component_models"])

        # config with method-specific params
        self.cfg = SimpleParamService(
            self.propensity_model,
            self.outcome_model,
            n_jobs=components_njobs,
            include_experimental=include_experimental_estimators,
        )

        self.estimates = {}

        self.original_estimator_list = estimator_list

        self.data_df = data_df or pd.DataFrame()
        self.causal_model = None
        self.identified_estimand = None

        # # trained component models for each estimator
        # self.trained_estimators_dict = {}

    def get_params(self, deep=False):
        return self._settings.copy()

    def get_estimators(self, deep=False):
        return self.estimator_list.copy()

    def fit(
        self,
        data_df: pd.DataFrame,
        treatment: str,
        outcome: str,
        common_causes: List[str],
        effect_modifiers: List[str],
        estimator_list: Optional[Union[str, List[str]]] = None,
    ):
        """Performs AutoML on list of causal inference estimators
        - If estimator has a search space specified in its parameters, HPO is performed on the whole model.
        - Otherwise, only its component models are optimised

        Args:
            data_df (pandas.DataFrame): dataset for causal inference
            treatment (str): name of treatment variable
            outcome (str): name of outcome variable
            common_causes (List[str]): list of names of common causes
            effect_modifiers (List[str]): list of names of effect modifiers
        """

        assert (
            len(data_df[treatment].unique()) > 1
        ), "Treatment must take at least 2 values, eg 0 and 1!"

        self.data_df = data_df
        self.train_df, self.test_df = train_test_split(
            data_df, train_size=self._settings["train_size"]
        )

        # TODO: allow specifying an exclusion list, too
        used_estimator_list = (
            self.original_estimator_list if estimator_list is None else estimator_list
        )

        self.estimator_list = self.cfg.estimator_names_from_patterns(
            used_estimator_list, len(data_df)
        )
        if not self.estimator_list:
            raise ValueError(
                f"No valid estimators in {str(estimator_list)}, available estimators: {str(self.cfg.estimator_names)}"
            )

        if self._settings["test_size"] is not None:
            self.test_df = self.test_df.sample(self._settings["test_size"])

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

        self.r_scorer = (
            None
            if "r_scorer" not in self.metrics_to_report
            else RScoreWrapper(
                self.outcome_model,
                self.propensity_model,
                self.train_df,
                self.test_df,
                outcome,
                treatment,
                common_causes,
                effect_modifiers,
            )
        )

        # self.tune_results = (
        #     {}
        # )  # We need to keep track of the tune results to access the best config

        search_space = self.cfg.search_space(self.estimator_list)
        init_cfg = (
            self.cfg.default_configs(self.estimator_list)
            if self._settings["try_init_configs"]
            else []
        )

        self.results = tune.run(
            self._tune_with_config,
            search_space,
            metric=self.metric,
            points_to_evaluate=init_cfg,
            mode="max",
            low_cost_partial_config={},
            **self._settings["tuner"],
        )

        if self.results.get_best_trial() is None:
            raise Exception(
                "Optimization failed! Did you set large enough time_budget and components_budget?"
            )

        self.scores = best_score_by_estimator(self.results.results, self.metric)

    def _tune_with_config(self, config: dict) -> dict:
        """Performs Hyperparameter Optimisation for a
        causal inference estimator

        Args:
            config (dict): dictionary with search space for
            all tunable parameters

        Returns:
            dict: values of metrics after optimisation
        """
        # estimate effect with current config

        print(config["estimator"])

        # spawn a separate process to prevent cross-talk between tuner and automl on component models:
        estimates = Parallel(n_jobs=2)(
            delayed(self._estimate_effect)(config["estimator"]) for i in range(1)
        )[0]

        return estimates

    def _estimate_effect(self, config):
        """estimates effect with chosen estimator"""

        # add params that are tuned by flaml:
        config = clean_config(config)
        print(f"config: {config}")
        self.estimator_name = config.pop("estimator_name")
        # params_to_tune = {
        #     k: v for k, v in config.items() if (not k == "estimator_name")
        # }
        cfg = self.cfg.method_params(self.estimator_name)
        try:
            estimate = self.causal_model.estimate_effect(
                self.identified_estimand,
                method_name=self.estimator_name,
                control_value=0,
                treatment_value=1,
                target_units="ate",  # condition used for CATE
                confidence_intervals=False,
                method_params={
                    "init_params": {**deepcopy(config), **cfg["init_params"]},
                    "fit_params": {},
                },
            )
            scores = self._compute_metrics(estimate)

            return {
                self.metric: scores["validation"][self.metric],
                "estimator": estimate,
                "estimator_name": scores.pop("estimator_name"),
                "scores": scores,
                "config": config,
            }
        except Exception as e:
            print("Evaluation failed!\n", config)
            print(e)
            return {
                self.metric: -np.inf,
                "exception": e,
                "traceback": traceback.format_exc(),
            }

    def _compute_metrics(self, estimator) -> dict:
        """computes metrics to score causal estimators"""
        try:
            te_train = estimator.cate_estimates
            X_test = self.test_df[estimator.estimator._effect_modifier_names]
            te_test = estimator.estimator.estimator.effect(X_test).flatten()
        except Exception:
            te_train = estimator.estimator.effect(self.train_df)
            te_test = estimator.estimator.effect(self.test_df)

        scores = {
            "estimator_name": self.estimator_name,
            "train": make_scores(
                estimator,
                self.train_df,
                te_train,
                r_scorer=None if self.r_scorer is None else self.r_scorer.train,
            ),
            "validation": make_scores(
                estimator,
                self.test_df,
                te_test,
                r_scorer=None if self.r_scorer is None else self.r_scorer.test,
            ),
        }
        return scores

    @property
    def best_estimator(self) -> str:
        """A string indicating the best estimator found"""
        return self.results.best_result["estimator_name"]

    @property
    def model(self):
        """Return the *trained* best estimator"""
        return self.results.best_result["estimator"]

    def best_model_for_estimator(self, estimator_name):
        """Return the best model found for a particular estimator.
        estimator: self.tune_results[estimator].best_config

        Args:
            estimator_name: a str of the estimator's name.

        Returns:
            An object storing the best model for estimator_name.
        """
        # Note that this returns the trained Econml estimator, whose attributes include
        # fitted  models for E[T | X, W], for E[Y | X, W], CATE model, etc.
        return self.scores[estimator_name]["estimator"]

    @property
    def best_config(self):
        """A dictionary containing the best configuration"""
        return self.results.best_config

    @property
    def best_config_per_estimator(self):
        """A dictionary of all estimators' best configuration."""
        return {e: s["config"] for e, s in self.scores.values()}

    @property
    def best_score_per_estimator(self):
        """A dictionary of all estimators' best score."""
        return {}

    @property
    def best_score(self):
        """A float of the best score found."""
        return self.results.best_result[self.metric]
