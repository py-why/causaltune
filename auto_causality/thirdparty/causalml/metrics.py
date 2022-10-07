import numpy as np
import pandas as pd

# code adapted from https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py

RANDOM_COL = "Random"


def get_cumlift(
    df, outcome_col="y", treatment_col="w", treatment_effect_col="tau", random_seed=42
):
    """Get average uplifts of model estimates in cumulative population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the mean of the true treatment effect in each of cumulative population.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each of cumulative population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): average uplifts of model estimates in cumulative population
    """

    assert (
        (outcome_col in df.columns)
        and (treatment_col in df.columns)  # noqa W503
        or treatment_effect_col in df.columns  # noqa W503
    )

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = "__random_{}__".format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [
        x
        for x in df.columns
        if x not in [outcome_col, treatment_col, treatment_effect_col]
    ]

    lift = []
    for i, col in enumerate(model_names):
        sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)
        sorted_df.index = sorted_df.index + 1

        if treatment_effect_col in sorted_df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population.
            lift.append(sorted_df[treatment_effect_col].cumsum() / sorted_df.index)
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population.
            sorted_df["cumsum_tr"] = sorted_df[treatment_col].cumsum()
            sorted_df["cumsum_ct"] = sorted_df.index.values - sorted_df["cumsum_tr"]
            sorted_df["cumsum_y_tr"] = (
                sorted_df[outcome_col] * sorted_df[treatment_col]
            ).cumsum()
            sorted_df["cumsum_y_ct"] = (
                sorted_df[outcome_col] * (1 - sorted_df[treatment_col])
            ).cumsum()

            lift.append(
                sorted_df["cumsum_y_tr"] / sorted_df["cumsum_tr"]
                - sorted_df["cumsum_y_ct"] / sorted_df["cumsum_ct"]
            )

    lift = pd.concat(lift, join="inner", axis=1)
    lift.loc[0] = np.zeros((lift.shape[1],))
    lift = lift.sort_index().interpolate()

    lift.columns = model_names
    lift[RANDOM_COL] = lift[random_cols].mean(axis=1)
    lift.drop(random_cols, axis=1, inplace=True)

    return lift


def get_cumgain(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
):
    """Get cumulative gains of model estimates in population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """

    lift = get_cumlift(
        df, outcome_col, treatment_col, treatment_effect_col, random_seed
    )

    # cumulative gain = cumulative lift x (# of population)
    gain = lift.mul(lift.index.values, axis=0)

    if normalize:
        gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)

    return gain


def get_qini(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
):
    """Get Qini of model estimates in population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """
    assert (
        (outcome_col in df.columns)
        and (treatment_col in df.columns)
        or treatment_effect_col in df.columns
    )

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = "__random_{}__".format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [
        x
        for x in df.columns
        if x not in [outcome_col, treatment_col, treatment_effect_col]
    ]

    qini = []
    for i, col in enumerate(model_names):
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df["cumsum_tr"] = df[treatment_col].cumsum()

        if treatment_effect_col in df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population.
            l_ = df[treatment_effect_col].cumsum() / df.index * df["cumsum_tr"]
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population.
            df["cumsum_ct"] = df.index.values - df["cumsum_tr"]
            df["cumsum_y_tr"] = (df[outcome_col] * df[treatment_col]).cumsum()
            df["cumsum_y_ct"] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

            l_ = (
                df["cumsum_y_tr"]
                - df["cumsum_y_ct"] * df["cumsum_tr"] / df["cumsum_ct"]
            )

        qini.append(l_)

    qini = pd.concat(qini, join="inner", axis=1)
    qini.loc[0] = np.zeros((qini.shape[1],))
    qini = qini.sort_index().interpolate()

    qini.columns = model_names
    qini[RANDOM_COL] = qini[random_cols].mean(axis=1)
    qini.drop(random_cols, axis=1, inplace=True)

    # this is just here to set a conditional breakpoint
    out = (qini.sum(axis=0) - qini[RANDOM_COL].sum()) / qini.shape[0]

    # what is the biggest theoretical value that the qini above can reach?
    trt = (
        df[df[treatment_col] == 1][outcome_col]
        .sort_values(ascending=False)
        .cumsum()
        .mean()
    )
    non_trt = (
        df[df[treatment_col] == 0][outcome_col]
        .sort_values(ascending=True)
        .cumsum()
        .mean()
    )

    if normalize:
        # qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)
        qini = qini / (trt - non_trt)

    return qini


def auuc_score(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=True,
    *args,
    **kwarg
):
    """Calculate the AUUC (Area Under the Uplift Curve) score.

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the AUUC score
    """

    cumgain = get_cumgain(
        df, outcome_col, treatment_col, treatment_effect_col, normalize
    )

    return cumgain.sum() / cumgain.shape[0]


def qini_score(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    *args,
    **kwarg
):
    """Calculate the Qini score: the area between the Qini curves of a model and random.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the Qini score
    """

    qini = get_qini(df, outcome_col, treatment_col, treatment_effect_col, normalize)

    out = (qini.sum(axis=0) - qini[RANDOM_COL].sum()) / qini.shape[0]
    return out


def get_simple_iptw(W, propensity_score):
    IPTW = (W / propensity_score) + (1 - W) / (1 - propensity_score)

    return IPTW


def get_std_diffs(X, W, weight=None, weighted=False, numeric_threshold=5):
    """Calculate the inverse probability of treatment weighted standardized
    differences in covariate means between the treatment and the control.
    If weighting is set to 'False', calculate unweighted standardized
    differences. Accepts only continuous and binary numerical variables.
    """
    cont_cols, prop_cols = _get_numeric_vars(X, threshold=numeric_threshold)
    cols = cont_cols + prop_cols

    if len(cols) == 0:
        raise ValueError(
            "No variable passed the test for continuous or binary variables."
        )

    treat = W == 1
    contr = W == 0

    X_1 = X.loc[treat, cols]
    X_0 = X.loc[contr, cols]

    cont_index = np.array([col in cont_cols for col in cols])
    prop_index = np.array([col in prop_cols for col in cols])

    std_diffs_cont = np.empty(sum(cont_index))
    std_diffs_prop = np.empty(sum(prop_index))

    if weighted:
        assert (
            weight is not None
        ), 'weight should be provided when weighting is set to "True"'

        weight_1 = weight[treat]
        weight_0 = weight[contr]

        X_1_mean, X_1_var = np.apply_along_axis(
            lambda x: _get_wmean_wvar(x, weight_1), 0, X_1
        )
        X_0_mean, X_0_var = np.apply_along_axis(
            lambda x: _get_wmean_wvar(x, weight_0), 0, X_0
        )

    elif not weighted:
        X_1_mean, X_1_var = np.apply_along_axis(lambda x: _get_mean_var(x), 0, X_1)
        X_0_mean, X_0_var = np.apply_along_axis(lambda x: _get_mean_var(x), 0, X_0)

    X_1_mean_cont, X_1_var_cont = X_1_mean[cont_index], X_1_var[cont_index]
    X_0_mean_cont, X_0_var_cont = X_0_mean[cont_index], X_0_var[cont_index]

    std_diffs_cont = (X_1_mean_cont - X_0_mean_cont) / np.sqrt(
        (X_1_var_cont + X_0_var_cont) / 2
    )

    X_1_mean_prop = X_1_mean[prop_index]
    X_0_mean_prop = X_0_mean[prop_index]

    std_diffs_prop = (X_1_mean_prop - X_0_mean_prop) / np.sqrt(
        ((X_1_mean_prop * (1 - X_1_mean_prop)) + (X_0_mean_prop * (1 - X_0_mean_prop)))
        / 2
    )

    std_diffs = np.concatenate([std_diffs_cont, std_diffs_prop], axis=0)
    std_diffs_df = pd.DataFrame(std_diffs, index=cols)

    return std_diffs_df


def _get_numeric_vars(X, threshold=5):
    """Attempt to determine which variables are numeric and which
    are categorical. The threshold for a 'continuous' variable
    is set to 5 by default.
    """

    cont = [
        (not hasattr(X.iloc[:, i], "cat")) and (X.iloc[:, i].nunique() >= threshold)
        for i in range(X.shape[1])
    ]

    prop = [X.iloc[:, i].nunique() == 2 for i in range(X.shape[1])]

    cont_cols = list(X.loc[:, cont].columns)
    prop_cols = list(X.loc[:, prop].columns)

    return cont_cols, prop_cols


def _get_mean_var(X):
    """Calculate the mean and variance of a variable."""
    mean = X.mean()
    var = X.var()

    return [mean, var]


def _get_wmean_wvar(X, weight):
    """
    Calculate the weighted mean of a variable given an arbitrary
    sample weight. Formulas from:

    Austin, Peter C., and Elizabeth A. Stuart. 2015. Moving towards Best
    Practice When Using Inverse Probability of Treatment Weighting (IPTW)
    Using the Propensity Score to Estimate Causal Treatment Effects in
    Observational Studies.
    Statistics in Medicine 34 (28): 3661 79. https://doi.org/10.1002/sim.6607.
    """
    weighted_mean = np.sum(weight * X) / np.sum(weight)
    weighted_var = (
        np.sum(weight) / (np.power(np.sum(weight), 2) - np.sum(np.power(weight, 2)))
    ) * (np.sum(weight * np.power((X - weighted_mean), 2)))

    return [weighted_mean, weighted_var]
