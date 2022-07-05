import os, sys
import numpy as np
import pandas as pd
from scipy import special

from sklearn.model_selection import train_test_split

root_path = root_path = os.path.realpath("../..")
try:
    import auto_causality
except ModuleNotFoundError:
    sys.path.append(os.path.join(root_path, "auto-causality"))

from auto_causality import AutoCausality
from auto_causality.data_utils import preprocess_dataset


# Modified example (EconML Notebooks):
# OrthoIV and DRIV Examples.ipynb.
def dgp(n, p):
    X = np.random.normal(0, 1, size=(n, p))
    Z = np.random.binomial(1, 0.5, size=(n,))
    nu = np.random.uniform(0, 5, size=(n,))
    coef_Z = 0.8
    C = np.random.binomial(
        1, coef_Z * special.expit(0.4 * X[:, 0] + nu)
    )  # Compliers when recomended
    C0 = np.random.binomial(
        1, 0.006 * np.ones(X.shape[0])
    )  # Non-compliers when not recommended
    T = C * Z + C0 * (1 - Z)
    y = (
        TRUE_EFFECT * T
        + 2 * nu
        + 5 * (X[:, 3] > 0)
        + 0.1 * np.random.uniform(0, 1, size=(n,))
    )
    return y, T, Z, X


if __name__ == "__main__":

    TRUE_EFFECT = 5

    n = 1000
    p = 10
    y, T, Z, X = dgp(n, p)

    cov = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cov)
    df["y"] = y
    df["Tr"] = T
    df["Z"] = Z

    # train_df, test_df = train_test_split(df, test_size=0.2)
    treatment = "Tr"
    targets = ["y"]
    data_df, features_X, features_W = preprocess_dataset(df, treatment, targets)

    outcome = targets[0]
    train_df, test_df = train_test_split(data_df, test_size=0.2)

    ac = AutoCausality(
        time_budget=120,
        metric="iv_energy_score",
        metrics_to_report=["iv_energy_score"],
        verbose=3,
        components_verbose=2,
        components_time_budget=60,
        propensity_model="auto",
    )

    # run autocausality
    ac.fit(train_df, treatment, outcome, features_W, features_X[:-1], instruments=["Z"])

    # return best estimator
    print(f"Best estimator: {ac.best_estimator}")
    # config of best estimator:
    print(f"best config: {ac.best_config}")
    # best score:
    print(f"best score: {ac.best_score}")
