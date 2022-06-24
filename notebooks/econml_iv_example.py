import numpy as np
import pandas as pd
from dowhy import CausalModel

# from lightgbm import LGBMRegressor

from flaml import AutoML


if __name__ == "__main__":

    n_points = 1000
    education_abilty = 1
    education_voucher = 2
    income_abilty = 2
    income_education = 4

    # confounder
    ability = np.random.normal(0, 3, size=n_points)

    # instrument
    voucher = np.random.normal(2, 1, size=n_points)

    # treatment
    education = (
        np.random.normal(5, 1, size=n_points)
        + education_abilty * ability
        + education_voucher * voucher
    )

    # outcome
    income = (
        np.random.normal(10, 3, size=n_points)
        + income_abilty * ability
        + income_education * education
    )

    # build dataset (exclude confounder `ability` which we assume to be unobserved)
    data = np.stack([education, income, voucher, np.random.normal(size=n_points)]).T
    df = pd.DataFrame(data, columns=["education", "income", "voucher", "random"])

    # Step 1: Model
    model = CausalModel(
        data=df,
        treatment="education",
        outcome="income",
        common_causes=["random"],
        instruments=["voucher"],
    )

    # Step 2: Identify
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("Identified estimand = ", identified_estimand)

    # Step 3: Estimate
    # Choose the second estimand: using IV
    cfg_regressor = {"time_budget": 10, "task": "regression"}

    # estimate = model.estimate_effect(identified_estimand,
    #     method_name="iv.instrumental_variable", test_significance=True)

    # * Works *
    # OrthoIV
    # DMLIV

    # * Fails *
    # NonParamDMLIV

    # Here's a working example
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="iv.econml.iv.dml.DMLIV",
        method_params={
            "init_params": {
                "model_y_xw": AutoML(**cfg_regressor),  # LGBMRegressor(),  #
                "model_t_xw": AutoML(**cfg_regressor),  # LGBMRegressor(),  #
                "model_t_xwz": AutoML(**cfg_regressor),  # LGBMRegressor(),  #
                "model_final": AutoML(**cfg_regressor),  # LGBMRegressor(),  #
            },
            "fit_params": {},
        },
        # test_significance=True, #TODO: why enabled?
    )
    print(estimate)
