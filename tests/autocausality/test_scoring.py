import pytest
from sklearn.model_selection import train_test_split
from auto_causality.datasets import synth_ihdp, preprocess_dataset
from dowhy import CausalModel
from econml.metalearners import SLearner
from sklearn.linear_model import LinearRegression


def simple_model_run():
    data_df = synth_ihdp()
    data_df, features_X, features_W, targets, treatment = preprocess_dataset(data_df)
    outcome = targets[0]
    train_df, test_df = train_test_split(
        data_df, train_size=0.5
    )
    causal_model = CausalModel(
        data=train_df,
        treatment=treatment,
        outcome=outcome,
        common_causes=features_W,
        effect_modifiers=features_X,
    )
    identified_estimand = causal_model.identify_effect(
        proceed_when_unidentifiable=True
    )
    estimate = causal_model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.metalearners.SLearner",
        control_value=0,
        treatment_value=1,
        target_units="ate",  # condition used for CATE
        confidence_intervals=False,)
    te_train = estimate.cate_estimates
    return estimate, train_df ,te_train

simple_model_run()



def test_auc_score():
    return None

# qini_make_score
# auc_make_score
# real_qini_make_score
# r_make_score
# make_scores
# ate
# group_ate
# best_score_by_estimator

if __name__ == "__main__":
    pytest.main([__file__])