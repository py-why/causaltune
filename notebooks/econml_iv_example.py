import numpy as np
import pandas as pd
from dowhy import CausalModel
from lightgbm import LGBMRegressor

# from flaml import AutoML


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
    data = np.stack([education, income, voucher]).T
    df = pd.DataFrame(data, columns=["education", "income", "voucher"])

    # Step 1: Model
    model = CausalModel(
        data=df,
        treatment="education",
        outcome="income",
        common_causes=["U"],
        instruments=["voucher"],
    )

    # Step 2: Identify
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("Identified estimand = ", identified_estimand)

    # Step 3: Estimate
    # Choose the second estimand: using IV
    # cfg_regressor = {"time_budget": 1, "task": "regression", "estimator_list": ["rf"]}

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
                "model_y_xw": LGBMRegressor(),  # AutoML(**cfg_regressor),
                "model_t_xw": LGBMRegressor(),
                "model_t_xwz": LGBMRegressor(),
                "model_final": LGBMRegressor(),
            },
            "fit_params": {},
        },
        test_significance=True,
    )
    print(estimate)
