import numpy as np
import pandas as pd

from causaltune.score.erupt_core import erupt, erupt_with_std
from causaltune.score.erupt import ERUPT, DummyPropensity
from causaltune.score.erupt_old import ERUPTOld


def evaluate_erupt_old(
    actual_propensity: pd.Series,
    actual_treatment: pd.Series,
    actual_outcome: pd.Series,
    hypothetical_policy: pd.Series,
) -> float:
    # define a dummy propensity model that will simply return the propensities when calling predict_proba
    propensity_model = DummyPropensity(p=actual_propensity, treatment=actual_treatment)

    # obtain ERUPT value under hypothetical policy
    e = ERUPTOld(treatment_name="T", propensity_model=propensity_model)
    return e.score(
        df=pd.DataFrame({"T": np.array(actual_treatment)}),
        outcome=actual_outcome,
        policy=hypothetical_policy,
    )


def evaluate_erupt_new(
    actual_propensity: pd.Series,
    actual_treatment: pd.Series,
    actual_outcome: pd.Series,
    hypothetical_policy: pd.Series,
) -> float:
    # define a dummy propensity model that will simply return the propensities when calling predict_proba
    propensity_model = DummyPropensity(p=actual_propensity, treatment=actual_treatment)

    # obtain ERUPT value under hypothetical policy
    e = ERUPT(treatment_name="T", propensity_model=propensity_model)
    return e.score(
        df=pd.DataFrame({"T": np.array(actual_treatment)}),
        outcome=actual_outcome,
        policy=hypothetical_policy,
    )


def make_dataset(n: int = 10000):
    # Let's create a dataset with a single feature
    df = pd.DataFrame({"X": np.random.uniform(size=n)})

    # Now let's create a response-to-treatment function that correlates with the feature
    def outcome(x: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        return 2 * np.random.uniform(size=len(x)) + x * (treatment == 1)

    # Let's consider a fully random treatment
    df["T1"] = np.random.randint(0, 2, size=n)
    # and simulate the corresponing experiment outcomes
    df["Y1"] = outcome(df["X"], df["T1"])

    # Let's consider another experiment on the same population, but with
    # treatment assignment that's biased by the feature, because we believe that
    # customers with higher values of the feature will be more responsive to the treatment

    df["p"] = 0.5 + 0.5 * df["X"]  # probability of binary treatment being applied
    df["T2"] = (np.random.rand(len(df)) < df["p"]).astype(
        int
    )  # sample with that propensity

    # We really only need the ex ante probability of the treatment that actually was applied
    # This will work exactly the same way in a multi-treatment case
    df["p_of_actual"] = df["p"] * df["T2"] + (1 - df["p"]) * (1 - df["T2"])

    # Now let's evaluate the outcome for this experiment

    df["Y2"] = outcome(df["X"], df["T2"])
    return df


def test_random_from_biased():
    df = make_dataset()
    est1 = evaluate_erupt_old(
        actual_propensity=df["p_of_actual"],
        actual_treatment=df["T2"],
        actual_outcome=df["Y2"],
        hypothetical_policy=df["T1"],
    )

    est1a = evaluate_erupt_new(
        actual_propensity=df["p_of_actual"],
        actual_treatment=df["T2"],
        actual_outcome=df["Y2"],
        hypothetical_policy=df["T1"],
    )

    est2 = erupt(
        actual_propensity=df["p_of_actual"].values,
        actual_treatment=df["T2"].values,
        actual_outcome=df["Y2"].values,
        hypothetical_policy=df["T1"].values,
    )

    est2a, std = erupt_with_std(
        actual_propensity=df["p_of_actual"].values,
        actual_treatment=df["T2"].values,
        actual_outcome=df["Y2"].values,
        hypothetical_policy=df["T1"].values,
    )

    est3 = df["Y1"].mean()
    assert np.isclose(est1, est2, atol=1e-3)
    assert np.isclose(est1a, est2, atol=1e-3)
    assert np.isclose(est2a, est3, atol=4 * std)
    assert np.isclose(est2, est3, atol=5e-2)


def test_random_from_biased_probabilistic():
    df = make_dataset()
    est1 = evaluate_erupt_old(
        actual_propensity=df["p_of_actual"],
        actual_treatment=df["T2"],
        actual_outcome=df["Y2"],
        hypothetical_policy=df["T1"],
    )

    est2 = erupt(
        actual_propensity=df["p_of_actual"].values,
        actual_treatment=df["T2"].values,
        actual_outcome=df["Y2"].values,
        hypothetical_policy=0.5 * np.ones((len(df), 2)),
    )

    est2a, std = erupt_with_std(
        actual_propensity=df["p_of_actual"].values,
        actual_treatment=df["T2"].values,
        actual_outcome=df["Y2"].values,
        hypothetical_policy=0.5 * np.ones((len(df), 2)),
    )
    est3 = df["Y1"].mean()
    assert np.isclose(est1, est2, atol=5e-2)
    assert np.isclose(est2a, est3, atol=4 * std)
    assert np.isclose(est2, est3, atol=5e-2)


def test_biased_from_random():
    df = make_dataset()
    est1 = evaluate_erupt_old(
        actual_propensity=0.5 * pd.Series(np.ones(len(df))),
        actual_treatment=df["T1"],
        actual_outcome=df["Y1"],
        hypothetical_policy=df["T2"],
    )

    est1a = evaluate_erupt_new(
        actual_propensity=0.5 * pd.Series(np.ones(len(df))),
        actual_treatment=df["T1"],
        actual_outcome=df["Y1"],
        hypothetical_policy=df["T2"],
    )

    est2 = erupt(
        actual_propensity=0.5 * np.ones(len(df)),
        actual_treatment=df["T1"].values,
        actual_outcome=df["Y1"].values,
        hypothetical_policy=df["T2"].values,
    )

    est2a, std = erupt_with_std(
        actual_propensity=0.5 * np.ones(len(df)),
        actual_treatment=df["T1"].values,
        actual_outcome=df["Y1"].values,
        hypothetical_policy=df["T2"].values,
    )

    est3 = df["Y2"].mean()
    assert np.isclose(est1, est2, atol=1e-3)
    assert np.isclose(est1a, est2, atol=1e-3)
    assert np.isclose(est2a, est3, atol=4 * std)
    assert np.isclose(est2, est3, atol=5e-2)
