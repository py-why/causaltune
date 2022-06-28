import dcor
import warnings
import numpy as np
import pandas as pd
from scipy import special

from econml.iv.dml import OrthoIV
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


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

    est1 = OrthoIV(
        projection=False,
        discrete_treatment=True,
        discrete_instrument=True,
    )

    est1.fit(train_df.y, train_df.Tr, Z=train_df.Z, X=None, W=train_df[cov])
    print("True Treatment Effect: ", TRUE_EFFECT)
    # print("OrthoIV Summary")
    # print(est1.summary(alpha=0.05))

    # 2 models under models_y_xw, with similar effect on energy_distance
    corrected_outcome = est1.models_y_xw[0][0].predict(test_df[cov])
    test_df["coo"] = corrected_outcome

    t1 = test_df[test_df.Tr == 1]
    t0 = test_df[test_df.Tr == 0]

    cols = cov + ["coo"]

    energy_distance = dcor.energy_distance(t1[cols], t0[cols])
    print("Energy distance = ", energy_distance)
