import sys
import os
from flaml import tune, AutoML
from sklearn.dummy import DummyClassifier
from auto_causality.params import SimpleParamService
from auto_causality.scoring import make_scores

root_path = root_path = os.path.realpath("../../..")
sys.path.append(os.path.join(root_path, "dowhy"))

from dowhy import CausalModel


class AutoCausality:
    """Performs AutoML to find best econML estimator. Optimises hyperparams of 
    component models of each estimator and hyperparams of the estimators themselves.
    Uses the ERUPT metric for estimator selection
    """

    def __init__(
        self, train_df, test_df, treatment, outcome, features_W, features_X, params=None
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.propensity_model = DummyClassifier(strategy="prior")
        self.outcome_model = AutoML(**params["flaml"]["component_params"])

        self.cfg = SimpleParamService(
            self.propensity_model,
            self.outcome_model,
            n_bootstrap_samples=20,
            n_estimators=500,
            max_depth=10,
            min_leaf_size=2 * (len(train_df.columns) - 2),
        )

        self.causal_model = CausalModel(
            data=train_df,
            treatment=treatment,
            outcome=outcome,
            common_causes=features_W,
            effect_modifiers=features_X,
        )
        self.identified_estimand = self.causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )

        self.estimates = {}
        self.results = {}
        self.params = params
        self.estimator_flaml_params = params["flaml"]["estimator_params"]
        self.estimator_list = params["estimator_list"]

    def fit(self):
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

    def _tune_with_config(self, config: dict):
        # start_time = time.time()
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
        except:
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

    def best_estimator(self):
        """returns best estimator
        """
        return max(self.results, key=self.results.get)

