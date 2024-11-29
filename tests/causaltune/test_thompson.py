import numpy as np
import pytest

from causaltune.score.thompson import (
    calculate_probabilities_per_row as thompson_policy,
)  # Replace with the actual module name

num_samples = 10000


def test_equal_means_and_stds():
    """
    Test case 1: All means and stds equal to 1.0, the policy should have equal entries
    """
    means = np.ones((5, 4))
    stds = np.ones((5, 4))
    policy = thompson_policy(means, stds, num_samples=num_samples)

    expected_value = 1 / means.shape[1]
    assert np.allclose(
        policy, expected_value, atol=100 / num_samples
    ), "Policy entries should be equal when all means and stds are identical."


def test_means_smaller_in_first_column():
    """
    Test case 2: First column means are smaller by 10x std, almost zero policy in first column
    """
    stds = np.ones((5, 4))
    means = np.ones((5, 4))
    means[:, 0] -= 10 * stds[:, 0]  # First column means smaller by 10x std

    policy = thompson_policy(means, stds, num_samples=num_samples)

    assert np.allclose(
        policy[:, 0], 0.0, atol=100 / num_samples
    ), "First column policy should be almost zero."
    assert np.allclose(
        policy[:, 1:], 1 / (means.shape[1] - 1), atol=100 / num_samples
    ), "Remaining columns should have equal probabilities."


def test_means_larger_in_first_column():
    """
    Test case 3: First column means are larger by 10x std, policy should be ones in first column
    """
    stds = np.ones((5, 4))
    means = np.ones((5, 4))
    means[:, 0] += 10 * stds[:, 0]  # First column means larger by 10x std

    policy = thompson_policy(means, stds, num_samples=num_samples)

    assert np.allclose(
        policy[:, 0], 1.0, atol=100 / num_samples
    ), "First column policy should be almost one."
    assert np.allclose(
        policy[:, 1:], 0.0, atol=100 / num_samples
    ), "Remaining columns should have probabilities close to zero."


if __name__ == "__main__":
    pytest.main()
