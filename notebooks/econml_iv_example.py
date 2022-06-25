import numpy as np
import pandas as pd
from dowhy import CausalModel

# from lightgbm import LGBMRegressor

from flaml import AutoML


if __name__ == "__main__":

    n_points = 1000
    education_ability = 1
    education_voucher = 2
    income_ability = 2
    income_education = 4

    # confounder
    ability = np.random.normal(0, 3, size=n_points)

    # instrument
    voucher = np.random.normal(2, 1, size=n_points)

    # treatment
    education = (
        np.random.normal(5, 1, size=n_points)
        + education_ability * ability
        + education_voucher * voucher
    )

    # outcome
    income = (
        np.random.normal(10, 3, size=n_points)
        + income_ability * ability
        + income_education * education
    )

    # build dataset (exclude confounder `ability` which we assume to be unobserved)
    # Todo: Should confounder be observed? Since it already influences the data above
    # Todo: Also exlcuding it yields closer ATE estimates. True causal effect is 4
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

    # * Fails *
    # NonParamDMLIV --

    # Tested working configurations
    OrthoIVParams = {
        "model_y_xw": AutoML(**cfg_regressor),
        "model_t_xw": AutoML(**cfg_regressor),
        "model_z_xw": AutoML(**cfg_regressor),
    }

    # DMLIVParams = {
    #     "model_y_xw": AutoML(**cfg_regressor),  # LGBMRegressor(),  #
    #     "model_t_xw": AutoML(**cfg_regressor),  # LGBMRegressor(),  #
    #     "model_t_xwz": AutoML(**cfg_regressor),  # LGBMRegressor(),  #
    #     "model_final": AutoML(**cfg_regressor),  # LGBMRegressor(),  #
    # },
    #
    # DRIVParams = {
    #     "model_y_xw": AutoML(**cfg_regressor),
    #     "model_t_xw": AutoML(**cfg_regressor),
    #     "model_tz_xw": AutoML(**cfg_regressor),
    #     "model_z_xw": AutoML(**cfg_regressor),
    #     # "discrete_treatment": True,
    #     "fit_cate_intercept": True
    # }

    # Here's a working example
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="iv.econml.iv.dml.OrthoIV",
        method_params={
            "init_params": OrthoIVParams,
            "fit_params": {},
        },
        test_significance=False,  # TODO: why enabled? From doWhy example. AutoML run is interminable if True. Will investigate
    )
    print(estimate)
