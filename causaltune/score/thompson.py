import numpy as np
from scipy.stats import norm


def thompson_policy(means: np.ndarray, stds: np.ndarray, num_samples: int = 10000):
    """
    Thompson sampling policy
    :param means: np.ndarray
    :param stds: np.ndarray
    :param num_samples: int
    :return: np.ndarray
    """
    assert len(means.shape) == 2
    assert len(stds.shape) == 2
    assert means.shape == stds.shape

    policy = np.zeros_like(means)
    for _ in range(num_samples):
        # Step 1: Create moo, a matrix of random normal samples
        moo = np.random.normal(loc=means, scale=stds)

        # In each row, add 1.0 to the column with the largest element of moo
        max_indices = np.argmax(
            moo, axis=1
        )  # Get the column index of the largest element in each row
        policy[np.arange(moo.shape[0]), max_indices] += 1.0

    policy /= num_samples
    return policy


# This does the same thing analytically, but is slooow and wrong :)
def calculate_probabilities_per_row(means, stds, *args, **kwargs):
    """
    Analytical approximation of the probability of each column being the largest, applied per row.
    :param means: np.ndarray of means (m x n)
    :param stds: np.ndarray of standard deviations (m x n)
    :return: np.ndarray of probabilities (m x n)
    """
    assert means.shape == stds.shape, "Means and stds must have the same shape."
    m, n = means.shape  # Number of rows and columns
    probabilities = np.zeros_like(means)

    # Iterate over each row
    for row_idx in range(m):
        row_means = means[row_idx, :]
        row_stds = stds[row_idx, :]
        row_probs = np.zeros(n)

        for i in range(n):

            def integrand(x):
                # Probability density for column i
                term_i = norm.pdf(x, loc=row_means[i], scale=row_stds[i])
                # CDF for all other columns
                term_others = np.prod(
                    [
                        norm.cdf(x, loc=row_means[j], scale=row_stds[j])
                        for j in range(n)
                        if j != i
                    ]
                )
                return term_i * term_others

            # Numerical integration over all x
            x_vals = np.linspace(-10, 10, 1000)
            integrand_vals = np.array([integrand(x) for x in x_vals])
            row_probs[i] = np.trapz(integrand_vals, x=x_vals)

        probabilities[row_idx, :] = row_probs / np.sum(
            row_probs
        )  # Normalize to ensure probabilities sum to 1

    return probabilities
