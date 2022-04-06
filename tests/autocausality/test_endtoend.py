import pytest
import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


def import_ihdp():
    root_path = root_path = os.path.realpath("../../..")
    sys.path.append(os.path.join(root_path, "auto-causality"))
    from auto_causality.utils import featurize

    # load and prepare the data
    data = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
        header=None,
    )

    col = [
        "treatment",
        "y_factual",
        "y_cfactual",
        "mu0",
        "mu1",
    ]
    for i in range(1, 26):
        col.append("x" + str(i))
    data.columns = col
    # drop the columns we don't care about
    ignore_patterns = ["y_cfactual", "mu"]
    ignore_cols = [c for c in data.columns if any([s in c for s in ignore_patterns])]
    data = data.drop(columns=ignore_cols)

    # prepare the data
    treatment = "treatment"
    targets = ["y_factual"]  # it's good to allow multiple ones
    features = [c for c in data.columns if c not in [treatment] + targets]

    data[treatment] = data[treatment].astype(int)
    # this is a trick to bypass some DoWhy/EconML bugs
    data["random"] = np.random.randint(0, 2, size=len(data))

    used_df = featurize(
        data,
        features=features,
        exclude_cols=[treatment] + targets,
        drop_first=False,
    )
    used_features = [
        c for c in used_df.columns if c not in ignore_cols + [treatment] + targets
    ]

    # Let's treat all features as effect modifiers
    features_X = [f for f in used_features if f != "random"]
    features_W = [f for f in used_features if f not in features_X]

    return used_df, features_X, features_W, targets, treatment


class TestEndToEnd(object):
    """tests autocausality model end-to-end
    1/ import autocausality object
    2/ preprocess data
    3/ init autocausality object
    4/ run autocausality on data
    """

    def test_imports(self):
        """tests if AutoCausality can be imported"""

        from auto_causality import AutoCausality  # noqa F401

    def test_data_preprocessing(self):
        """tests data preprocessing routines"""

        data = import_ihdp()  # noqa F484

    def test_init_autocausality(self):
        """tests if autocausality object can be instantiated without errors"""

        from auto_causality import AutoCausality  # noqa F401

        auto_causality = AutoCausality()  # noqa F484

    def test_endtoend(self):
        """tests if model can be instantiated and fit to data"""

        from auto_causality import AutoCausality  # noqa F401

        data_df, features_X, features_W, targets, treatment = import_ihdp()

        estimator_list = [
            "Dummy",
            "SparseLinearDML",
            "ForestDRLearner",
            "TransformedOutcome",
            "CausalForestDML",
            ".LinearDML",
            "DomainAdaptationLearner",
            "SLearner",
            "XLearner",
            "TLearner",
            "Ortho",
        ]
        outcome = targets[0]
        auto_causality = AutoCausality(
            time_budget=180,
            components_time_budget=10,
            estimator_list=estimator_list,
            use_ray=False,
            resources_per_trial={"cpu": 0.5},
        )

        auto_causality.fit(data_df, treatment, outcome, features_W, features_X)

        print(f"Best estimator: {auto_causality.best_estimator}")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEnd().test_endtoend()
