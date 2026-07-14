import copy
import warnings
from typing import Any, List, Optional, Union
from collections import defaultdict
import time

import traceback
import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import _base
from hiertunehub import create_tuner

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from dowhy import CausalModel
from dowhy.causal_identifier import IdentifiedEstimand

from econml.inference import BootstrapInference

from causaltune.search.params import SimpleParamService
from causaltune.score.scoring import Scorer, metrics_to_minimize
from causaltune.utils import treatment_is_multivalue
from causaltune.models.monkey_patches import (
    AutoML,
    effect_stderr,
)

# from causaltune.models.monkey_patch_flaml import run

from causaltune.data_utils import CausalityDataset
from causaltune.dataset_processor import CausalityDatasetProcessor
from causaltune.models.passthrough import feature_filter


# framework_params passthrough (fit()/constructor) key classes:
# - MANAGED: collide with CausalTune's budget/parity logic -> warn, but honour.
# - RESERVED: hiertunehub passes these to the backend explicitly (positionally
#   or as fixed kwargs), so a user value would raise a duplicate-kwarg TypeError
#   deep in hiertunehub -> reject up front with a clear error.
_MANAGED_FRAMEWORK_PARAMS = {
    "num_samples",
    "n_trials",
    "max_evals",
    "time_budget_s",
    "timeout",
    "search_alg",
    "sampler",
    "algo",
    "points_to_evaluate",
    "evaluated_rewards",
    "resources_per_trial",
    "n_jobs",
    "verbose",
    "cost_attr",
    "low_cost_partial_config",
}
_RESERVED_FRAMEWORK_PARAMS = {
    "config",
    "mode",
    "metric",
    "trials",
    "objective",
    "search_space",
}


# Patched from sklearn.linear_model._base to adjust rtol and atol values
def _check_precomputed_gram_matrix(
    X, precompute, X_offset, X_scale, rtol=1e-4, atol=1e-2
):
    n_features = X.shape[1]
    f1 = n_features // 2
    f2 = min(f1 + 1, n_features - 1)

    v1 = (X[:, f1] - X_offset[f1]) * X_scale[f1]
    v2 = (X[:, f2] - X_offset[f2]) * X_scale[f2]

    expected = np.dot(v1, v2)
    actual = precompute[f1, f2]

    if not np.isclose(expected, actual, rtol=rtol, atol=atol):
        raise ValueError(
            "Gram matrix passed in via 'precompute' parameter "
            "did not pass validation when a single element was "
            "checked - please check that it was computed "
            f"properly. For element ({f1},{f2}) we computed "
            f"{expected} but the user-supplied value was "
            f"{actual}."
        )


_base._check_precomputed_gram_matrix = _check_precomputed_gram_matrix


class CausalTune:
    """Performs AutoML to find best EconML estimator.
    Optimises hyperparams of component models of each estimator
    and hyperparams of the estimators themselves. Uses the ERUPT
    metric for estimator selection.

    Example:

    .. code-block:: shell

        cd = CausalityDataset(data=df, treatment='T', outcomes=['Y'])
        cd.preprocess_dataset()

        estimator_list = [".LinearDML","LinearDRLearner","metalearners"]
        ct = CausalTune(time_budget=10, estimator_list=estimator_list)

        ct.fit(cd)

        print(f"Best estimator: {ct.best_estimator}")

    """

    def __init__(
        self,
        data_df=None,
        metric="energy_distance",
        metrics_to_report=None,
        time_budget=None,
        verbose=0,
        use_ray=False,
        estimator_list="auto",
        train_size=0.8,
        test_size=None,
        num_samples=-1,
        propensity_model="dummy",
        propensity_automl_estimators: Optional[List[str]] = None,
        outcome_model="nested",
        components_task="regression",
        components_verbose=0,
        components_pred_time_limit=10 / 1e6,
        components_njobs=-1,
        components_time_budget=None,
        try_init_configs=True,
        resources_per_trial=None,
        include_experimental_estimators=False,
        store_all_estimators: Optional[bool] = False,
        framework: str = "optuna",
        algo: Any = None,
        framework_params: Optional[dict] = None,
    ):
        """Constructor.

        Args:
            data_df (pandas.DataFrame): dataset to perform causal inference on
            metric (str): metric to optimise.
            data_df (pandas.DataFrame): dataset to perform causal inference on
            metric (str): metric to optimise.
                Defaults to "erupt" for CATE, "energy_distance" for IV
            metrics_to_report (list): additional metrics to compute and report.
                Defaults to ["qini","auc","ate","erupt", "norm_erupt"] for CATE
                or ["energy_distance"] for IV
            time_budget (float): a number of the time budget in seconds. -1 if no limit.
            num_samples (int): max number of iterations.
            verbose (int):  controls verbosity, higher means more messages. range (0,3). Defaults to 0.
            use_ray (bool): use Ray backend (requires ray to be installed).
            estimator_list (list): a list of strings for estimator names,
            time_budget (float): a number of the time budget in seconds. -1 if no limit.
            num_samples (int): max number of iterations.
            verbose (int):  controls verbosity, higher means more messages. range (0,3). Defaults to 0.
            use_ray (bool): use Ray backend (requires ray to be installed).
            estimator_list (list): a list of strings for estimator names,
             or "auto" for a recommended subset, "all" for all, or a list of substrings of estimator names
               e.g. ```['dml', 'CausalForest']```
            train_size (float): Fraction of data used for training set. Defaults to 0.8.
            test_size (float): Optional size of test dataset. Defaults to None.
            propensity_model (Union[str, Any]): 'dummy' for dummy classifier, 'auto' for AutoML, or an
            train_size (float): Fraction of data used for training set. Defaults to 0.8.
            test_size (float): Optional size of test dataset. Defaults to None.
            propensity_model (Union[str, Any]): 'dummy' for dummy classifier, 'auto' for AutoML, or an
                sklearn-style classifier
            components_task (str): task for component models. Defaults to "regression".
            components_verbose (int): verbosity of component model HPO (hyper parameter optimisation).
            components_task (str): task for component models. Defaults to "regression".
            components_verbose (int): verbosity of component model HPO (hyper parameter optimisation).
                range (0,3). Defaults to 0.
            components_pred_time_limit (float): prediction time limit for component models
            components_njobs (int): number of concurrent jobs for component model optimisation.
            components_pred_time_limit (float): prediction time limit for component models
            components_njobs (int): number of concurrent jobs for component model optimisation.
                Defaults to -1 (all available cores).
            components_time_budget (float): time budget for HPO of component models in seconds.
            components_time_budget (float): time budget for HPO of component models in seconds.
                Defaults to overall time budget / 2.
            try_init_configs (bool): try list of good performing estimators before continuing with HPO.
            try_init_configs (bool): try list of good performing estimators before continuing with HPO.
                Defaults to False.
            resources_per_trial: computational resources per trial, defaults in constructor to {"cpu": 0.5}
            include_experimental_estimators (bool): Include experimental causal estimators. Whether an estimator
            resources_per_trial: computational resources per trial, defaults in constructor to {"cpu": 0.5}
            include_experimental_estimators (bool): Include experimental causal estimators. Whether an estimator
                is experimental can be seen in SimpleParamsService in scoring.py
            store_all_estimators (Optional[bool]). store estimator objects for interim trials. Defaults to False
            store_all_estimators (Optional[bool]). store estimator objects for interim trials. Defaults to False
            framework (str): default HPO backend, one of "optuna" (default),
                "hyperopt" or "flaml". Overridable per fit(). With algo=None,
                optuna uses its default TPE sampler.
            algo (Any): default search algorithm for the backend (flaml
                search_alg / hyperopt suggest fn / optuna sampler). None uses the
                backend default. Overridable per fit().
            framework_params (Optional[dict]): default advanced backend-native
                params merged into the tuner call. Overridable per fit(). See
                fit() for the managed/reserved-key rules.

            Returns:
                None
        """
        assert (
            time_budget is not None or components_time_budget is not None
        ), "Either time_budget or components_time_budget must be specified"

        self._settings = {}
        self._settings["tuner"] = {}
        self._settings["tuner"]["time_budget_s"] = time_budget
        self._settings["tuner"]["num_samples"] = num_samples
        self._settings["tuner"]["verbose"] = verbose
        self._settings["tuner"]["resources_per_trial"] = (
            resources_per_trial if resources_per_trial is not None else {"cpu": 0.5}
        )
        self._settings["tuner"]["algo"] = None
        self._settings["try_init_configs"] = try_init_configs
        self._settings[
            "include_experimental_estimators"
        ] = include_experimental_estimators

        # params for FLAML on component models:
        self._settings["component_models"] = {}
        self._settings["component_models"]["task"] = components_task
        self._settings["component_models"]["verbose"] = components_verbose
        self._settings["component_models"][
            "pred_time_limit"
        ] = components_pred_time_limit
        self._settings["component_models"]["n_jobs"] = components_njobs
        self._settings["component_models"]["time_budget"] = components_time_budget
        self._settings["component_models"]["eval_method"] = "holdout"
        self._settings["propensity_automl_estimators"] = propensity_automl_estimators

        if 0 < train_size < 1:
            component_test_size = 1 - train_size
        else:
            # TODO: convert train_size to fraction based on data size, in fit()
            component_test_size = 0.2
        self._settings["component_models"]["split_ratio"] = component_test_size
        self._settings["train_size"] = train_size
        self._settings["test_size"] = test_size
        self._settings["store_all"] = store_all_estimators
        # HPO-backend defaults; fit() args override these when passed.
        self._settings["framework"] = framework
        self._settings["algo"] = algo
        self._settings["framework_params"] = framework_params
        self._settings["metric"] = metric
        self._settings["metrics_to_report"] = metrics_to_report
        self._settings["propensity_model"] = propensity_model
        self._settings["outcome_model"] = outcome_model

        self.tuner = None
        self._best_estimators = defaultdict(lambda: (float("-inf"), None))

        self.original_estimator_list = estimator_list
        self.data_df = data_df or pd.DataFrame()
        self.causal_model = None
        self.identified_estimand = None
        self.problem = None
        self.use_ray = use_ray

    def get_params(self, deep=False):
        return self._settings.copy()

    def get_estimators(self, deep=False):
        return self.estimator_list.copy()

    def init_propensity_model(self, propensity_model: str):
        # user can choose between flaml and dummy for propensity model.
        if propensity_model == "dummy":
            self.propensity_model = DummyClassifier(strategy="prior")
        elif propensity_model == "auto":
            automl_args = {
                **self._settings["component_models"],
                "task": "classification",
            }
            if self._settings["propensity_automl_estimators"]:
                automl_args["estimator_list"] = self._settings[
                    "propensity_automl_estimators"
                ]

            self.propensity_model = AutoML(**automl_args)
        elif hasattr(propensity_model, "fit") and hasattr(
            propensity_model, "predict_proba"
        ):
            self.propensity_model = propensity_model
        else:
            raise ValueError(
                'propensity_model valid values are "dummy", "auto", or a classifier object'
            )

    def init_outcome_model(self, outcome_model):
        # TODO: implement filtering like below, when there are propensity-only features
        # feature_filter below acts on classes not instances
        # to preserve all the extra methods through inheritance
        # if we are only supplying certain features to the propensity function,
        # make them invisible to the outcome component model
        # This is a workaround for the DoWhy/EconML data model which doesn't
        # support that out of the box

        if hasattr(outcome_model, "fit") and hasattr(outcome_model, "predict"):
            return outcome_model
        elif outcome_model == "auto":
            # Will be dynamically chosen at optimization time
            return outcome_model
        elif outcome_model == "nested":
            # The current default behavior
            return self.auto_outcome_model()
        else:
            raise ValueError(
                'outcome_model valid values are None, "auto", or an estimator object'
            )

    def auto_outcome_model(self):
        data = self.data
        propensity_only_cols = [
            p
            for p in data.propensity_modifiers
            if p not in data.common_causes + data.effect_modifiers
        ]

        if len(propensity_only_cols):
            # TODO: implement feature_filter for arbitrary outcome models
            outcome_model_class = feature_filter(
                AutoML, data.effect_modifiers + data.common_causes, first_cols=True
            )
        else:
            outcome_model_class = AutoML

        return outcome_model_class(**self._settings["component_models"])

    def fit(
        self,
        data: Union[pd.DataFrame, CausalityDataset],
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        common_causes: Optional[List[str]] = None,
        effect_modifiers: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        propensity_modifiers: Optional[List[str]] = None,
        estimator_list: Optional[Union[str, List[str]]] = None,
        resume: Optional[bool] = False,
        time_budget: Optional[int] = None,
        preprocess: bool = False,
        encoder_type: Optional[str] = None,
        encoder_outcome: Optional[str] = None,
        use_ray: Optional[bool] = None,
        framework: Optional[str] = None,
        algo: Any = None,
        framework_params: Optional[dict] = None,
    ):
        """Performs AutoML on list of causal inference estimators
        - If estimator has a search space specified in its parameters, HPO is performed on the whole model.
        - Otherwise, only its component models are optimised

        Args:
            data (pandas.DataFrame): dataset for causal inference
            treatment (str): name of treatment variable
            outcome (str): name of outcome variable
            common_causes (List[str]): list of names of common causes
            effect_modifiers (List[str]): list of names of effect modifiers
            instruments (List[str]): list of names of instrumental variables
            propensity_modifiers (List[str]): list of names of propensity modifiers
            estimator_list (Optional[Union[str, List[str]]]): subset of estimators to consider
            resume (Optional[bool]): set to True to continue previous fit
            time_budget (Optional[int]): change new time budget allocated to fit, useful for warm starts.
            preprocess (bool): preprocess CausalityDataset if needed.
            encoder_type (Optional[str]): Categorical Encoder for preprocessing
            encoder_outcome (Optional[str]): Categorical Encoder target for preprocessing: TargetEncoder, WOE.
            framework (Optional[str]): HPO backend to use, one of "optuna"
                (default), "hyperopt" or "flaml". Warm-start (try_init_configs)
                and resume are supported on all three; cost-aware search
                (cost_attr/low_cost_partial_config) remains flaml-only. None
                falls back to the value passed to the constructor.
            algo (Any): search algorithm for the chosen backend. flaml -> a
                FLAML search_alg; hyperopt -> a suggest function (defaults to
                hyperopt.tpe.suggest); optuna -> an optuna sampler (defaults to
                optuna's TPESampler). None falls back to the constructor value
                (then each backend's default).
            framework_params (Optional[dict]): advanced escape hatch of extra
                backend-native params merged into the tuner call (user wins).
                Overriding a CausalTune-managed key warns; a reserved key
                (config/mode/metric/trials/objective/search_space) raises. None
                falls back to the constructor value.

        Returns:
            None
        """
        if use_ray is not None:
            self.use_ray = use_ray

        # Resolve backend settings: an explicit fit() arg overrides the value
        # supplied to the constructor (mirroring estimator_list).
        framework = framework if framework is not None else self._settings["framework"]
        algo = algo if algo is not None else self._settings["algo"]
        user_framework_params = (
            framework_params
            if framework_params is not None
            else self._settings["framework_params"]
        )
        if user_framework_params:
            reserved = _RESERVED_FRAMEWORK_PARAMS & set(user_framework_params)
            if reserved:
                raise ValueError(
                    f"framework_params may not set reserved key(s) {sorted(reserved)}; "
                    "these are supplied to the backend by CausalTune/hiertunehub."
                )

        if outcome is None and isinstance(data, CausalityDataset):
            outcome = data.outcomes[0]

        if not isinstance(data, CausalityDataset):
            assert isinstance(data, pd.DataFrame)
            data = CausalityDataset(
                data,
                treatment,
                outcome,
                common_causes=common_causes,
                effect_modifiers=effect_modifiers,
                instruments=instruments,
                propensity_modifiers=propensity_modifiers,
            )

        if preprocess:
            data = copy.deepcopy(data)
            self.dataset_processor = CausalityDatasetProcessor()
            self.dataset_processor.fit(
                data, encoder_type=encoder_type, outcome=encoder_outcome
            )
            data = self.dataset_processor.transform(data)
        else:
            self.dataset_processor = None

        self.data = data
        treatment_values = data.treatment_values

        assert (
            len(treatment_values) > 1
        ), "Treatment must take at least 2 values, eg 0 and 1!"

        self._control_value = treatment_values[0]
        self._treatment_values = list(treatment_values[1:])

        # To be used for component model training/selection
        self.train_df, self.test_df = train_test_split(
            self.data.data, train_size=self._settings["train_size"], shuffle=True
        )

        # smuggle propensity modifiers into common causes, filter later in component models
        self.causal_model = CausalModel(
            data=self.train_df,
            treatment=data.treatment,
            outcome=outcome,
            common_causes=data.common_causes + data.propensity_modifiers,
            effect_modifiers=data.effect_modifiers,
            instruments=data.instruments,
        )

        self.init_propensity_model(self._settings["propensity_model"])

        self.identified_estimand: IdentifiedEstimand = (
            self.causal_model.identify_effect(proceed_when_unidentifiable=True)
        )

        if bool(self.identified_estimand.estimands["iv"]) and bool(data.instruments):
            self.problem = "iv"
        elif bool(self.identified_estimand.estimands["backdoor"]):
            self.problem = "backdoor"
        else:
            raise ValueError(
                "Couldn't identify the kind of problem from "
                + str(self.identified_estimand.estimands)
            )

        # This must be stateful because we need to train the treatment propensity function
        self.scorer = Scorer(
            self.causal_model,
            self.propensity_model,
            self.problem,
            treatment_is_multivalue(self._treatment_values),
        )

        self.metric = self.scorer.resolve_metric(self._settings["metric"])
        self.metrics_to_report = self.scorer.resolve_reported_metrics(
            self._settings["metrics_to_report"], self.metric
        )

        # Reset the per-estimator best cache for a fresh fit, with the correct
        # default sign for the metric direction. On a genuine resume we KEEP the
        # previous fit's fitted-estimator objects: resume does not re-run past
        # trials, yet model/effect resolve the fitted model from this cache, so a
        # past-best would otherwise be lost. (resume with no prior tuner falls
        # through to a fresh reset.)
        if not (resume and self.tuner is not None):
            worst = (
                float("inf") if self.metric in metrics_to_minimize() else float("-inf")
            )
            self._best_estimators = defaultdict(lambda: (worst, None))

        # TODO: allow specifying an exclusion list, too
        used_estimator_list = (
            self.original_estimator_list if estimator_list is None else estimator_list
        )

        assert (
            isinstance(used_estimator_list, str) or len(used_estimator_list) > 0
        ), "estimator_list must either be a str or an iterable of str"

        # config with method-specific params
        self.cfg = SimpleParamService(
            n_jobs=self._settings["component_models"]["n_jobs"],
            include_experimental=self._settings["include_experimental_estimators"],
            multivalue=treatment_is_multivalue(self._treatment_values),
            sample_outcome_estimators=self._settings["outcome_model"] == "auto",
        )

        self.estimator_list = self.cfg.estimator_names_from_patterns(
            self.problem,
            used_estimator_list,
            len(self.data),
        )

        if not self.estimator_list:
            raise ValueError(
                f"No valid estimators in {str(used_estimator_list)}, "
                f"available estimators: {str(self.cfg.estimator_names)}"
            )

        if time_budget:
            self._settings["tuner"]["time_budget_s"] = time_budget

        if self._settings["component_models"]["time_budget"] is None:
            self._settings["component_models"]["time_budget"] = self._settings["tuner"][
                "time_budget_s"
            ] / (2.5 * len(self.estimator_list))

        if (
            self._settings["tuner"]["time_budget_s"] is None
            and self._settings["tuner"]["num_samples"] == -1
        ):
            self._settings["tuner"]["time_budget_s"] = (
                2.5
                * len(self.estimator_list)
                * self._settings["component_models"]["time_budget"]
            )

        cmtb = self._settings["component_models"]["time_budget"]

        if cmtb < 300:
            warnings.warn(
                f"Component model time budget is {cmtb}. "
                f"Recommended value is at least 300 for smallish datasets, 1800 for datasets with> 100K rows"
            )

        if self._settings["test_size"] is not None:
            self.test_df = self.test_df.sample(self._settings["test_size"])

        if "r_scorer" in self.metrics_to_report:
            raise NotImplementedError(
                "R-squared scorer no longer suported, please raise an issue if you want it back"
            )
        # self.r_scorer = (
        #     None
        #     if "r_scorer" not in self.metrics_to_report
        #     else RScoreWrapper(
        #         self.outcome_model,
        #         self.propensity_model,
        #         self.train_df,
        #         self.test_df,
        #         outcome,
        #         treatment,
        #         common_causes,
        #         effect_modifiers,
        #     )
        # )

        search_space = self.cfg.search_space(
            self.estimator_list, data_size=data.data.shape
        )
        # Warm-start init configs (promising configs to try first). Computed for
        # ALL backends now: flaml consumes them via points_to_evaluate below;
        # optuna/hyperopt via points_to_evaluate on fresh runs (see below).
        init_cfg = (
            self.cfg.default_configs(self.estimator_list, data_size=data.data.shape)
            if self._settings["try_init_configs"]
            else []
        )

        if framework == "hyperopt" and not self._search_space_has_tunable_params():
            raise ValueError(
                "framework='hyperopt' needs at least one tunable hyperparameter "
                "in the search space, but all selected estimators are "
                "parameterless and outcome_model is not 'auto'. (hiertunehub's "
                "hyperopt conversion raises on a fully parameterless search.) Use "
                "framework='optuna' or 'flaml', include a parameterized "
                "estimator, or set outcome_model='auto'."
            )

        self._settings["tuner"]["algo"] = algo
        mode = "min" if self.metric in metrics_to_minimize() else "max"
        framework_params = self.cfg.parse_tuner_params(
            self._settings["tuner"], framework
        )

        # A genuine resume continues the previous in-memory tuner. (flaml has its
        # own resume path below; optuna/hyperopt route through resume_from_results.)
        resuming = (
            resume and framework in ("optuna", "hyperopt") and self.tuner is not None
        )
        if resume and framework in ("optuna", "hyperopt") and self.tuner is None:
            warnings.warn(
                "resume=True but no previous fit to resume from; running fresh.",
                UserWarning,
            )

        if framework == "flaml":
            # Full FLAML parity: cost-aware search plus warm-start / resume seeds.
            # Capture resume points/rewards from the PREVIOUS tuner before it is
            # overwritten below.
            if resume and self.tuner is not None:
                points_to_evaluate, evaluated_rewards = self._resume_points_and_rewards(
                    self.tuner.results, init_cfg
                )
            else:
                points_to_evaluate, evaluated_rewards = init_cfg, []
            framework_params.update(
                cost_attr="evaluation_cost",
                low_cost_partial_config={},
                points_to_evaluate=points_to_evaluate,
                evaluated_rewards=evaluated_rewards,
            )
        elif init_cfg and not resuming:
            # optuna / hyperopt warm-start: seed the promising configs to try
            # first. Skipped on resume, where the rehydrated past trials seed the
            # search instead.
            framework_params["points_to_evaluate"] = init_cfg

        # Resume for optuna/hyperopt: rebuild the seed list from the PREVIOUS
        # tuner's trials (copied so hiertunehub can't mutate the stored trials --
        # params deep, result shallow to avoid deep-copying fitted estimators),
        # and normalise the count budget to "N additional new trials".
        prev_tuner = self.tuner
        past_results = None
        if resuming:
            past_results = [
                {"params": copy.deepcopy(t.params), "result": dict(t.result)}
                for t in prev_tuner.trials
            ]
            framework_params = self._adjust_resume_budget(
                framework_params, framework, len(past_results)
            )

        # Advanced escape hatch: merge user backend params last (user wins),
        # warning on collisions with CausalTune-managed keys.
        if user_framework_params:
            clash = _MANAGED_FRAMEWORK_PARAMS & set(user_framework_params)
            if clash:
                warnings.warn(
                    "framework_params overrides CausalTune-managed key(s) "
                    f"{sorted(clash)}; this can change the search budget or parity "
                    "behaviour.",
                    UserWarning,
                )
            framework_params.update(user_framework_params)

        if framework == "optuna":
            if framework_params.get("n_jobs", 1) not in (None, 1):
                warnings.warn(
                    "optuna n_jobs>1 runs trials in threads, but CausalTune's "
                    "objective mutates shared instance state and is not "
                    "thread-safe; results may be corrupted. Use use_ray for safe "
                    "parallelism.",
                    UserWarning,
                )
            self._set_optuna_verbosity(self._settings["tuner"]["verbose"])

        self.tuner = create_tuner(
            self._tune_with_config,
            search_space,
            metric=self.metric,
            mode=mode,
            framework=framework,
            framework_params=framework_params,
        )
        if resuming:
            self.tuner.resume_from_results(past_results)
        else:
            self.tuner.run()

        self.update_summary_scores()

    def _resume_points_and_rewards(self, prev_results, init_cfg):
        """Rebuild FLAML warm-start seeds from a previous tuner's results.

        Mirrors the original resume semantics: for each prior trial that carries
        both the metric and its config, seed ``(config -> reward)``; then append
        any init configs not already present (without a reward, so FLAML
        evaluates them).

        Args:
            prev_results (list[dict]): ``tuner.results`` from the previous fit.
            init_cfg (list[dict]): init configs to append if not yet evaluated.

        Returns:
            tuple[list[dict], list]: ``(points_to_evaluate, evaluated_rewards)``
            with rewards aligned to the leading resumed configs.
        """
        resume_cfg = []
        resume_scores = []
        for result in prev_results:
            if self.metric not in result or "config" not in result:
                continue
            resume_scores.append(result[self.metric])
            resume_cfg.append(result["config"])
        for cfg in init_cfg:
            if cfg not in resume_cfg:
                resume_cfg.append(cfg)
        return resume_cfg, resume_scores

    @staticmethod
    def _adjust_resume_budget(
        framework_params: dict, framework: str, n_past: int
    ) -> dict:
        """Normalise a resume's count budget to "n_past + N" so the resumed run
        adds the same N *new* trials as a fresh run would.

        hiertunehub's optuna ``resume_from_results`` subtracts ``n_past`` from
        ``n_trials`` internally, and hyperopt's ``max_evals`` counts the
        rehydrated past trials toward the total. Adding ``n_past`` here makes both
        yield N new trials. Time-bounded runs (count budget is ``None``) are
        governed by the timeout and need no adjustment.

        Args:
            framework_params (dict): the computed backend params.
            framework (str): "optuna" or "hyperopt".
            n_past (int): number of past trials being rehydrated.

        Returns:
            dict: a copy of ``framework_params`` with the count budget adjusted.
        """
        fp = dict(framework_params)
        if framework == "optuna" and fp.get("n_trials") is not None:
            fp["n_trials"] += n_past
        elif framework == "hyperopt" and fp.get("max_evals") is not None:
            fp["max_evals"] += n_past
        return fp

    @staticmethod
    def _set_optuna_verbosity(verbose) -> None:
        """Map CausalTune's ``verbose`` (0..3) to optuna's logging level.

        optuna's ``study.optimize`` takes no ``verbose`` kwarg, so verbosity is
        controlled globally via ``optuna.logging`` (best-effort; prior level is
        not restored). 0 -> WARNING (silences per-trial logs), 1 -> INFO,
        >=2 -> DEBUG.
        """
        if not verbose:
            level = optuna.logging.WARNING
        elif verbose == 1:
            level = optuna.logging.INFO
        else:
            level = optuna.logging.DEBUG
        optuna.logging.set_verbosity(level)

    def _search_space_has_tunable_params(self) -> bool:
        """Whether the current search space contains any tunable hyperparameter.

        Sampling outcome estimators (outcome_model='auto') always adds tunable
        component-model params; otherwise it depends on the estimators' own
        search spaces. Used to guard the hyperopt backend, whose hiertunehub
        converter raises on a fully parameterless search space.
        """
        if self.cfg.sample_outcome_estimators:
            return True
        configs = self.cfg._configs()
        return any(
            bool(configs[est].search_space)
            for est in self.estimator_list
            if est in configs
        )

    def update_summary_scores(self):
        """Stores scores for metric of interest for each estimator

        Returns:
            None
        """
        self.scores = Scorer.best_score_by_estimator(self.tuner.results, self.metric)
        # now inject the separately saved model objects
        for est_name in self.scores:
            # Todo: Check approximate scores for OrthoIV (possibly other IV estimators)
            # assert (
            #     self._best_estimators[est_name][0] == self.scores[est_name][self.metric]
            # ), "Can't match best model to score"
            self.scores[est_name]["estimator"] = self._best_estimators[est_name][1]

    def _tune_with_config(self, config: dict) -> dict:
        """
        Performs Hyperparameter Optimisation for a causal inference estimator.

        Args:
            config (dict): Dictionary with search space for all tunable parameters.

        Returns:
            (dict): values of metrics after optimisation
        """
        from causaltune.remote import remote_exec

        if self.use_ray:
            # flaml.tune handles the interaction with Ray itself
            # estimates = self._estimate_effect(config)
            estimates = remote_exec(
                CausalTune._estimate_effect, (self, config), self.use_ray
            )
        else:
            estimates = remote_exec(
                CausalTune._estimate_effect, (self, config), self.use_ray
            )

        #     Parallel(n_jobs=2, backend="threading")(
        #     delayed(self._estimate_effect)(config) for i in range(1)
        # ))[0]

        if "exception" not in estimates:
            est_name = estimates["estimator_name"]
            current_score = estimates[self.metric]

            estimates["optimization_score"] = current_score
            estimates[
                "evaluation_cost"
            ] = 1e8  # will be overwritten for successful runs

            # Initialize best_score if this is the first estimator for this name
            if est_name not in self._best_estimators:
                self._best_estimators[est_name] = (
                    (
                        np.inf
                        if self.metric
                        in [
                            "energy_distance",
                            "psw_energy_distance",
                            "frobenius_norm",
                            "psw_frobenius_norm",
                            "codec",
                            "policy_risk",
                        ]
                        else -np.inf
                    ),
                    None,
                )

            best_score = self._best_estimators[est_name][0]

            # Determine if the current estimator performs better, handling inf values
            if self.metric in [
                "energy_distance",
                "psw_energy_distance",
                "frobenius_norm",
                "psw_frobenius_norm",
                "codec",
                "policy_risk",
            ]:
                is_better = (
                    np.isfinite(current_score) and current_score < best_score
                ) or (np.isinf(best_score) and np.isfinite(current_score))
            else:
                is_better = (
                    np.isfinite(current_score) and current_score > best_score
                ) or (np.isinf(best_score) and np.isfinite(current_score))

            # Store the estimator if we're storing all, if it's better, or if it's the first valid (non-inf) estimator
            if (
                self._settings["store_all"]
                or is_better
                or (
                    self._best_estimators[est_name][1] is None
                    and np.isfinite(current_score)
                )
            ):
                self._best_estimators[est_name] = (
                    current_score,
                    (
                        estimates["estimator"]
                        if self._settings["store_all"]
                        else estimates.pop("estimator")
                    ),
                )
            if "Dummy" not in est_name:
                estimates["evaluation_cost"] = estimates.pop("elapsed_time")

        return estimates

    def _est_effect_stub(self, method_params):
        return self.causal_model.estimate_effect(
            self.identified_estimand,
            method_name=self.estimator_name,
            control_value=self._control_value,
            treatment_value=self._treatment_values,
            target_units="ate",  # condition used for CATE
            confidence_intervals=False,
            method_params=method_params,
        )

    def _estimate_effect(self, config):
        """estimates effect with chosen estimator"""

        # Do we need an boject property for this, instead of a local var?
        self.estimator_name = config["estimator"]["estimator_name"]
        outcome_model = self.init_outcome_model(self._settings["outcome_model"])
        method_params = self.cfg.method_params(
            config, outcome_model, self.propensity_model
        )

        try:  #
            # This calls the causal model's estimate_effect method
            start_time = time.time()
            estimate = self._est_effect_stub(method_params)
            scores = {
                "estimator_name": self.estimator_name,
                "train": self._compute_metrics(
                    estimate,
                    self.train_df,
                ),
                "validation": self._compute_metrics(
                    estimate,
                    self.test_df,
                ),
            }
            elapsed_time = time.time() - start_time
            return {
                self.metric: scores["validation"][self.metric],
                "estimator": estimate,
                "estimator_name": self.estimator_name,
                "scores": scores,
                # TODO: return full config!
                "config": config,
                "elapsed_time": elapsed_time,
            }
        except Exception as e:
            print("Evaluation failed!\n", config, traceback.format_exc())
            # Use the *worst* value for the metric direction as the failure
            # sentinel, so a failed trial is never picked as best. For minimized
            # metrics (e.g. energy_distance) that is +inf; for maximized metrics
            # it is -inf. (A flat -inf would look optimal to a minimizing backend
            # such as the default optuna, poisoning best_estimator selection.)
            worst = np.inf if self.metric in metrics_to_minimize() else -np.inf
            return {
                self.metric: worst,
                "estimator_name": self.estimator_name,
                "exception": e,
                "traceback": traceback.format_exc(),
            }

    def _compute_metrics(self, estimator, df: pd.DataFrame) -> dict:
        return self.scorer.make_scores(
            estimator, df, self.metrics_to_report, r_scorer=None
        )

    def score_dataset(self, df: pd.DataFrame, dataset_name: str):
        """
        After fitting, generate scores for an additional dataset, add them to the scores dict.

        Args:
            df (pandas.DataFrame): input dataframe
            dataset_name (str): dictionary key

        Returns:
            None.
        """
        for scr in self.scores.values():
            if scr["estimator"] is None:
                warnings.warn(
                    "Skipping scoring for estimator %s" % scr["estimator_name"]
                )
            else:
                scr["scores"][dataset_name] = self._compute_metrics(
                    scr["estimator"], df
                )

    @property
    def best_estimator(self) -> str:
        """A string indicating the best estimator found

        Returns:
            None
        """
        return self.tuner.best_result["estimator_name"]

    @property
    def model(self):
        """Return the *trained* best estimator

        Returns:
            CausalEstimator
        """
        # The objective pops the fitted estimator out of the result dict in the
        # non-store_all path, so resolve it from the scores table (populated by
        # update_summary_scores) rather than from tuner.best_result.
        return self.scores[self.best_estimator]["estimator"].estimator

    def best_model_for_estimator(self, estimator_name):
        """Return the best model found for a particular estimator.
        estimator: self.tune_results[estimator].best_config

        Args:
            estimator_name (str): the estimator's name.

        Returns:
            (dowhy.causal_estimator.CausalEstimate): the best model for estimator_name.
        """
        # Note that this returns the trained Econml estimator, whose attributes include
        # fitted  models for E[T | X, W], for E[Y | X, W], CATE model, etc.
        return self.scores[estimator_name]["estimator"]

    @property
    def best_config(self):
        """
        Returns:
            (dict): the best configuration
        """
        return self.tuner.best_params

    @property
    def best_config_per_estimator(self):
        """
        Returns:
            (dict): all estimators' best configuration."""
        return {e: s["config"] for e, s in self.scores.values()}

    @property
    def best_score_per_estimator(self):
        """A dictionary of all estimators' best score."""
        return {}

    @property
    def best_score(self):
        """
        Returns:
            (float):  the best score found."""
        return self.tuner.best_result[self.metric]

    def effect(self, df, *args, **kwargs):
        """Heterogeneous Treatment Effects for data df

        Args:
            df (pd.DataFrame): data to predict treatment effect for

        Returns:
            (np.ndarray): predicted treatment effect for each datapoint

        """
        return self.model.effect(df, *args, **kwargs)

    def predict(
        self, cd: CausalityDataset, preprocess: Optional[bool] = False, *args, **kwargs
    ):
        """Heterogeneous Treatment Effects for data CausalityDataset

        Args:
            cd (CausalityDataset): data to predict treatment effect for

        Returns:
            (np.ndarray): predicted treatment effect for each datapoint

        """
        if preprocess:
            cd = copy.deepcopy(cd)
            if self.dataset_processor:
                cd = self.dataset_processor.transform(cd)
            else:
                raise ValueError("CausalityDatasetProcessor has not been trained")
        return self.model.effect(cd.data, *args, **kwargs)

    def effect_inference(self, df, *args, **kwargs):
        """Inference (uncertainty) results produced by best estimator
        Only implemented for EconML estimators so far

        Args:
            df (pd.DataFrame): data to run inference on
            args: passed through to underlying estimator
            kwargs: passed through to underlying estimator

        Returns:
            (from EconML: NormalInferenceResults):
                EconML results object for inference assuming a normal distribution.
                from EconML: NormalInferenceResults:

        """

        if "Econml" in str(type(self.model)):
            # Get a list of "Inference" objects from EconML, one per treatment.
            # dowhy 0.14's EconML adapter provides apply_multitreatment natively.
            if self.cfg._configs()[self.best_estimator].inference == "bootstrap":
                raise NotImplementedError(
                    f"Can't calculate stds for estimator \
                {self.best_estimator} \
                as boostrap inference is not supported yet"
                )
            return self.model.effect_inference(df, *args, **kwargs)
        else:
            raise NotImplementedError(
                "No pointwise error estimates for non-EconML estimators implemented yet"
            )

    def effect_stderr(self, df, n_bootstrap_samples=5, n_jobs=1, *args, **kwargs):
        """Compute standard errors for best causal estimator
            Currently implemented for EconML estimators.
            Computes analytical standard errors if available and boostraps otherwise.
        Args:
            df (pd.DataFrame): data to run inference on
            n_bootstrap_samples (int, optional): number of runs if standard errors are boostrapped. Defaults to 5.
            n_jobs (int, optional): Number of bootstrap estimates to run in parallel. Defaults to 1.

        Returns:
            (np.ndarray): standard error for each data point from df
        Args:
            df (pd.DataFrame): data to run inference on
            n_bootstrap_samples (int, optional): number of runs if standard errors are boostrapped. Defaults to 5.
            n_jobs (int, optional): Number of bootstrap estimates to run in parallel. Defaults to 1.

        Returns:
            np.ndarray: standard error for each data point from df
        """

        if "Econml" in str(type(self.model)):
            # Get a list of "Inference" objects from EconML, one per treatment
            self.model.__class__.effect_stderr = effect_stderr
            outcome_model = self.init_outcome_model(self._settings["outcome_model"])
            method_params = self.cfg.method_params(
                self.best_config, outcome_model, self.propensity_model
            )

            if self.cfg.full_config(self.best_estimator).inference == "bootstrap":
                # TODO: before bootstrapping, check whether that's already been done
                bootstrap = BootstrapInference(
                    n_bootstrap_samples=n_bootstrap_samples, n_jobs=n_jobs
                )
                method_params["fit_params"]["inference"] = bootstrap
                self.estimator_name = (
                    self.best_estimator
                )  # needed for _est_effect_stub, just in case
                self.bootstrapped_estimate = self._est_effect_stub(method_params)
                est = self.bootstrapped_estimate.estimator
            else:
                # If the estimator supports other inference methods,
                # those have already been included
                est = self.model
            return est.effect_stderr(df, *args, **kwargs)
        else:
            raise NotImplementedError(
                "No pointwise error estimates for non-EconML estimators implemented yet"
            )
