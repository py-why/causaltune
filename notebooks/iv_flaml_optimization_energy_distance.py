import sys
import dcor
import numpy as np
import pandas as pd
from scipy import special
from sklearn.model_selection import train_test_split

from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate

sys.path.append("../")
from auto_causality import AutoCausality
from auto_causality.data_utils import preprocess_dataset
from auto_causality.scoring import Scorer


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


# Needed since ac.model.estimator doesn't include additional params -
# treatment, outcome etc. - needed from CausalEstimate instance
def energy_scorer_patch(
    estimate: CausalEstimate,
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    instrument: str,
    effect_modifiers: [],
):

    df["dy"] = estimate.estimator.effect(df[effect_modifiers])
    df.loc[df[treatment] == 0, "dy"] = 0
    df["yhat"] = df[outcome] - df["dy"]

    X1 = df[df[instrument] == 1]
    X0 = df[df[instrument] == 0]
    select_cols = effect_modifiers + ["yhat"]

    energy_distance_score = dcor.energy_distance(X1[select_cols], X0[select_cols])

    return energy_distance_score


if __name__ == "__main__":

    TRUE_EFFECT = 10

    n = 1000
    p = 10
    y, T, Z, X = dgp(n, p)

    cov = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cov)
    df["y"] = y
    df["Tr"] = T
    df["Z"] = Z

    treatment = "Tr"
    targets = ["y"]
    instruments = ["Z"]
    data_df, features_X, features_W = preprocess_dataset(
        df, treatment, targets, instruments
    )

    outcome = targets[0]
    train_df, test_df = train_test_split(data_df, test_size=0.2)
    train_df_copy, test_df_copy = train_df.copy(), test_df.copy()

    ac = AutoCausality(
        time_budget=120,
        verbose=3,
        components_verbose=2,
        components_time_budget=60,
        propensity_model="auto",
    )

    ac.fit(train_df, treatment, outcome, features_W, features_X, instruments)

    # return best estimator
    print(f"Best estimator: {ac.best_estimator}")
    # config of best estimator:
    print(f"best config: {ac.best_config}")
    # best score:
    print(f"best score: {ac.best_score}")

    # Comparing best model searched to base IV model configuration
    model = CausalModel(
        data=train_df,
        treatment="Tr",
        outcome="y",
        effect_modifiers=cov,
        common_causes=["random"],
        instruments=["Z"],
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="iv.econml.iv.dml.DMLIV",
        method_params={
            "init_params": {},
            "fit_params": {},
        },
        test_significance=False,
    )

    Xtest = test_df[cov]
    print()
    print("True Treatment Effect: ", TRUE_EFFECT)
    print(
        "(Baseline Estimator) Treatment Effect: ",
        estimate.estimator.effect(Xtest).mean(),
    )
    print(
        "(AutoCausality Estimator) Treatment Effect: ",
        ac.model.estimator.estimator.effect(Xtest).mean(),
    )

    print("Energy distance scores")
    base_estimator_edist = Scorer.energy_distance_score(estimate, test_df)
    ac_estimator_edist = energy_scorer_patch(
        ac.model.estimator, test_df, treatment, outcome, instruments[0], cov
    )
    print("(Baseline Estimator) Energy distance score: ", base_estimator_edist)
    print("(AutoCausality Estimator) Energy distance score: ", ac_estimator_edist)
