import dcor
import numpy as np
import pandas as pd
from scipy import special

from sklearn.model_selection import train_test_split
from dowhy import CausalModel


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

    TRUE_EFFECT = 10
    n = 5000
    p = 10
    y, T, Z, X = dgp(n, p)

    cov = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cov)
    df["y"] = y
    df["Tr"] = T
    df["Z"] = Z

    train_df, test_df = train_test_split(df, test_size=0.2)

    # 1: Initialize model
    model = CausalModel(
        data=train_df,
        treatment="Tr",
        outcome="y",
        effect_modifiers=cov,
        common_causes=["U"],
        instruments=["Z"],
    )

    # Step 2: Identify estimand
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Step 3: Estimate effect
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="iv.econml.iv.dml.DMLIV",
        method_params={
            "init_params": {},
            "fit_params": {},
        },
        test_significance=False,
    )

    # Step 3: Energy distance scoring
    dy = estimate.estimator.effect(test_df)
    test_df["yhat"] = dy - test_df["y"]

    t1 = test_df[test_df.Tr == 1]
    t0 = test_df[test_df.Tr == 0]
    select_cols = cov + ["yhat"]

    edist = dcor.energy_distance(t1[select_cols], t0[select_cols])
    print("Energy distance = ", edist)
