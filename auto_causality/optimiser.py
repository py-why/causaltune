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
    params = {
        "flaml": {
            "component_params": {
                "time_budget": 10,
                "verbose": 0,
                "task": "regression",
                "n_jobs": num_cores,
                "pred_time_limit": 10 / 1e6,
            },
            "estimator_params": {
                "time_budget_s": 10,
                "num_samples": 10,
                "verbose": 0,
                "use_ray": False,
            },
        },
        "estimator_list": [
            "backdoor.econml.dml.LinearDML",
            "backdoor.econml.dr.LinearDRLearner",
        ],
        "metric": "ERUPT",
    }

    auto_causality = AutoCausality(params=params)
    auto_causality.fit(train_df, test_df, treatment,
     outcome, features_W, features_X)
    print(f"Best estimator: {auto_causality.best_estimator}")

    ```
    """

    def __init__(
        self, params=None,
    ):
        """constructor.

        Args:
            train_df (pd.DataFrame): training data
            test_df (pd.DataFrame): test data
            treatment (str): treatment variable name
            outcome (str): outcome variable name
            features_W (List[str]): common causes
            features_X (List[str]): effect modifiers
            params ([type], optional): optional parameters. Defaults to None.
        """
        self.propensity_model = DummyClassifier(strategy="prior")
        self.outcome_model = AutoML(**params["flaml"]["component_params"])

        # config with method-specific params
        self.cfg = SimpleParamService(
            self.propensity_model,
            self.outcome_model,
            n_bootstrap_samples=20,
            n_estimators=500,
            max_depth=10,
            min_leaf_size=2 * 26,
        )

        # dicts for logging
        self.estimates = {}
        self.results = {}

        self.params = params
        self.estimator_flaml_params = params["flaml"]["estimator_params"]
        self.estimator_list = params["estimator_list"]

    @property
    def best_estimator(self):
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
        # TODO
        pass

    @property
    def best_config_per_estimator(self):
        """A dictionary of all estimators' best configuration."""
        # TODO
        return None

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

        Args:
            train_df (pd.DataFrame): Training Data
            test_df (pd.DataFrame): Test Data
            treatment (str): name of treatment variable
            outcome (str): name of outcome variable
            common_causes (List[str]): list of names of common causes
            effect_modifiers (List[str]): list of names of effect modifiers
        """

        if not hasattr(self, "train_df"):
            self.train_df = train_df
        if not hasattr(self, "test_df"):
            self.test_df = test_df

        if not hasattr(self, "causal_model"):
            # dowhy causal model. TODO: move to .fit() , lazy init
            self.causal_model = CausalModel(
                data=self.train_df,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
                effect_modifiers=effect_modifiers,
            )

        if not hasattr(self, "identified_estimand"):

            self.identified_estimand = self.causal_model.identify_effect(
                proceed_when_unidentifiable=True
            )

        for estimator in self.estimator_list:
            self.estimator = estimator
            self.estimator_cfg = self.cfg.method_params(estimator)
            results = tune.run(
                self._tune_with_config,
                self.estimator_cfg["search_space"],
                metric=self.params["metric"],
                mode="max",
                **self.estimator_flaml_params,
            )
            print(f"Estimator: {self.estimator}")
            print(
                f"... {self.params['metric']}: {results.best_trial.last_result[self.params['metric']]}"
            )
            # log results
            self.results[self.estimator] = results.best_trial.last_result[
                self.params["metric"]
            ]

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
        self.estimates[self.estimator] = self.causal_model.estimate_effect(
            self.identified_estimand,
            method_name=self.estimator,
            control_value=0,
            treatment_value=1,
            target_units="ate",  # condition used for CATE
            confidence_intervals=False,
            method_params=self.estimator_cfg,
        )

        # compute a metric and return
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
        return {"ERUPT": scores["test"]["erupt"], "ATE": scores["test"]["ate"]}
