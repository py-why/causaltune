import numpy as np


def erupt(
    actual_propensity: np.ndarray,
    actual_treatment: np.ndarray,
    actual_outcome: np.ndarray,
    hypothetical_policy: np.ndarray,
    clip: float = 0.05,
    remove_tiny: bool = True,
) -> float:
    """
    ERUPT score
    :param actual_propensity: The ex-ante probability of selecting the particular treatment in the experiment data
    :param actual_treatment: The treatment selected in the experiment data
    :param actual_outcome: The outcome observed in the experiment data
    :param hypothetical_policy: The hypothetical policy to evaluate,
    either as a vector of ints or matrix of probabilities
    :return:
    """
    assert (
        len(actual_propensity)
        == len(actual_treatment)
        == len(actual_outcome)
        == len(hypothetical_policy)
    )
    assert (
        len(actual_propensity.shape)
        == len(actual_treatment.shape)
        == len(actual_outcome.shape)
        == 1
    )
    assert len(hypothetical_policy.shape) in (1, 2)

    n = len(actual_propensity)
    treatments = np.unique(actual_treatment)
    assert (
        len(treatments) == max(treatments) + 1
    ), "The treatments must be integers from 0 to N-1, every treatment must be present in the sample"
    assert min(treatments) == 0, "The treatments must be integers from 0 to N-1"

    if len(hypothetical_policy.shape) == 1:
        # Convert to matrix of probabilities
        policy = np.zeros((len(hypothetical_policy), len(treatments)))
        policy[
            np.arange(len(hypothetical_policy)), hypothetical_policy.astype(np.int32)
        ] = 1.0
    else:
        policy = hypothetical_policy

    # Calculate the ERUPT score
    # Clip propensity scores to avoid division by zero or extreme weights
    min_clip = max(1e-6, clip)  # Ensure minimum clip is not too small
    propensity = np.clip(actual_propensity, min_clip, 1 - min_clip)

    weight = 1 / (propensity + 1e-6)
    # Handle extreme weights
    if remove_tiny:
        weight[weight > 1 / clip] = 0.0
    else:
        weight[weight > 1 / clip] = 1 / clip

    new_policy = policy[np.arange(len(actual_treatment)), actual_treatment]

    new_weight = weight * new_policy
    new_weight = n * new_weight / np.sum(new_weight)

    estimate = np.sum(new_weight * actual_outcome) / np.sum(new_weight)

    return estimate
