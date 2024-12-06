from typing import Tuple

import numpy as np


def shuffle_and_split(N, k):
    """
    Shuffle the numbers from 0 to N-1 and split them into k approximately equal parts.

    :param N: int, the range of numbers
    :param k: int, the number of parts to split into
    :return:  k np.ndarrays, each containing integers
    """
    # Create an array of numbers from 1 to N
    numbers = np.arange(0, N)

    # Shuffle the array
    np.random.shuffle(numbers)

    # Split the array into k approximately equal parts
    split_arrays = np.array_split(numbers, k)

    return split_arrays


def erupt_with_std(
    actual_propensity: np.ndarray,
    actual_treatment: np.ndarray,
    actual_outcome: np.ndarray,
    hypothetical_policy: np.ndarray,
    num_splits: int = 5,
    resamples: int = 100,
    clip: float = 0.05,
    remove_tiny: bool = True,
) -> Tuple[float, float]:
    std = 0.0
    mean = 0.0
    for _ in range(resamples):
        splits = shuffle_and_split(len(actual_propensity), num_splits)
        means = [
            erupt(
                actual_propensity[s],
                actual_treatment[s],
                actual_outcome[s],
                hypothetical_policy[s],
                clip,
                remove_tiny,
            )
            for s in splits
        ]
        mean += np.mean(means)
        std += np.std(means) / np.sqrt(num_splits)  # Standard error of the mean
    # 1.5 is an empirical factor to make the confidence interval wider
    return mean / resamples, 1.5 * std / resamples


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


if __name__ == "__main__":
    out = shuffle_and_split(1001, 5)
    print("yay!")
