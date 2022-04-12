# import os
# from typing import List
#
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.dummy import DummyClassifier
#
# from flaml import AutoML
# from dowhy import CausalModel
#
# from auto_causality.params import SimpleParamService
# from auto_causality.utils import featurize
# from auto_causality.scoring import make_scores
# from auto_causality.datasets import synth_ihdp
#
# root_path = os.path.realpath("../..")
#
# # set this path to differ by project
# data_dir = os.path.realpath(os.path.join(root_path, "auto-causality/data/tests"))
# parent = os.path.realpath(os.path.join(data_dir, ".."))
# if not os.path.isdir(parent):
#     os.mkdir(parent)
# if not os.path.isdir(data_dir):
#     os.mkdir(data_dir)
# print(data_dir)
#
#
# def run_full(
#     data: pd.DataFrame,
#     data_dir: str,
#     treatment_name: str,
#     targets: List[str],
#     train_size=0.5,
#     test_size=None,
#     time_budget: int = 60,
#     num_cores: int = 4,
#     conf_intervals: bool = False,
#     # Only change the n_bootstrap_samples (to at least 20) if you want
#     # confidence intervals specifically for metalearner models
#     # warning: that will be VERY slow!
#     n_bootstrap_samples: int = None,
# ):
#     features = [c for c in data.columns if c not in [treatment_name] + targets]
#
#     data[treatment_name] = data[treatment_name].astype(int)
#     # this is a trick to bypass some DoWhy/EconML bugs
#     data["random"] = np.random.randint(0, 2, size=len(data))
#
#     used_df = featurize(
#         data,
#         features=features,
#         exclude_cols=[treatment_name] + targets,
#         drop_first=False,
#     )
#     used_features = [c for c in used_df.columns if c not in [treatment_name] + targets]
#
#     # Let's treat all features as effect modifiers
#     features_X = [f for f in used_features if f != "random"]
#     features_W = [f for f in used_features if f not in features_X]
#
#     if not (
#         os.path.isfile(os.path.join(data_dir, f"test_{time_budget}.csv"))
#         and os.path.isfile(
#             os.path.join(data_dir, f"train_{time_budget}.csv")
#         )  # noqa W504
#     ):
#         train_df, test_df = train_test_split(used_df, train_size=train_size)
#         if test_size is not None:
#             test_df = test_df.sample(test_size)
#         # test_df.to_csv(os.path.join(data_dir, f"test_{time_budget}.csv"))
#         # train_df.to_csv(os.path.join(data_dir, f"train_{time_budget}.csv"))
#     else:
#         pass
#         # test_df = pd.read_csv(os.path.join(data_dir, f"test_{time_budget}.csv"))
#         # train_df = pd.read_csv(os.path.join(data_dir, f"train_{time_budget}.csv"))
#
#     del data
#
#     # define model parametrization
#     propensity_model = DummyClassifier(strategy="prior")
#     outcome_model = AutoML(
#         time_budget=time_budget,
#         verbose=1,
#         task="regression",
#         n_jobs=num_cores,
#         pred_time_limit=10 / 1e6,
#     )
#
#     cfg = SimpleParamService(
#         propensity_model,
#         outcome_model,
#         n_bootstrap_samples=n_bootstrap_samples,
#         min_leaf_size=2 * len(used_features),
#     )
#
#     for outcome in targets:
#         print(outcome)
#         model = CausalModel(
#             data=train_df,
#             treatment=treatment_name,
#             outcome=outcome,
#             common_causes=features_W,
#             effect_modifiers=features_X,
#         )
#         identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
#
#         estimates = {}
#         for estimator in cfg.estimators():
#             if not any(
#                 [
#                     e in estimator
#                     for e in [
#                         "causality",
#                         "DomainAdaptationLearner",
#                         "SLearner",
#                         "TLearner",
#                         "XLearner",
#                         ".LinearDML",
#                         "SparseLinearDML",
#                         "CausalForestDML",
#                         "ForestDRLearner",
#                         ".LinearDRLearner",
#                         "DROrthoForest",  # this one doesn't work on large datasets (too slow)
#                     ]
#                 ]
#             ):
#                 continue
#
#             scores_fname = os.path.join(
#                 data_dir, f"scores_{estimator}_{outcome}_{time_budget}.zip"
#             )
#             model_fname = os.path.join(
#                 data_dir, f"model_{estimator}_{outcome}_{time_budget}.zip"
#             )
#             if os.path.isfile(scores_fname) and os.path.isfile(model_fname):
#                 continue
#
#             method_params = cfg.method_params(estimator)
#             print("fitting", estimator, method_params)
#
#             estimates[estimator] = model.estimate_effect(
#                 identified_estimand,
#                 method_name=estimator,
#                 control_value=0,
#                 treatment_value=1,
#                 target_units="ate",  # condition used for CATE
#                 confidence_intervals=conf_intervals,
#                 method_params=method_params,
#             )
#
#             estimates[estimator].interpret()
#
#             te_train = estimates[estimator].cate_estimates
#
#             try:
#                 te_test = estimates[estimator].estimator.effect(test_df)
#             except Exception:
#                 # this will no longer be necessary once https://github.com/microsoft/dowhy/pull/374 is merged
#                 X_test = test_df[estimates[estimator].estimator._effect_modifier_names]
#                 te_test = (
#                     estimates[estimator].estimator.estimator.effect(X_test).flatten()
#                 )
#
#             scores = {
#                 "estimator": estimator,
#                 "outcome": outcome,
#                 "train": make_scores(estimates[estimator], train_df, te_train),
#                 "test": make_scores(estimates[estimator], test_df, te_test),
#             }
#
#             print(f"Scores for {estimator}_{outcome}", scores)
#
#             # print("dumping...")
#             # with gzip.open(scores_fname, "wb") as f:
#             #     # dill transfers better between Python versions
#             #     dill.dump(scores, f)
#             #
#             # with gzip.open(model_fname, "wb") as f:
#             #     print("dumping", model_fname)
#             #     # Cloudpickle can serialize some things dill chokes on
#             #     cloudpickle.dump(estimates[estimator], f)
#             #     print("success dumping model file!")
#
#     print("yahoo!")
#
#
# class TestFullRun(object):
#     def test_full_run(self):
#         df = synth_ihdp()
#         run_full(df, data_dir, "treatment", ["y_factual"], num_cores=1)
#
#
# if __name__ == "__main__":
#     df = synth_ihdp()
#     run_full(
#         df,
#         data_dir,
#         "treatment",
#         ["y_factual"],
#     )
