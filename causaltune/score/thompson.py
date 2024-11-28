import numpy as np
from scipy.stats import norm
from causaltune.models.monkey_patches import effect_stderr


def prepend_zero_column(mat: np.ndarray, fill: float):
    """
    Append a column of zeros to a matrix
    :param mat: np.ndarray
    :return: np.ndarray
    """
    if len(mat.shape) == 1:
        mat = mat[:, np.newaxis]

    return np.concatenate((fill * np.ones((mat.shape[0], 1)), mat), axis=1)


def extract_means_stds(est, df, treatment_name, num_treatments):
    # TODO: test this properly for the multivalue case
    means = np.squeeze(est.effect(df))
    means = prepend_zero_column(means, 0.0)

    if "Econml" in str(type(est)):
        est.__class__.effect_stderr = effect_stderr
        try:
            stds = np.squeeze(est.effect_stderr(df))
        except Exception:
            stds = None
    else:
        stds = None

    if stds is None:
        # Ignore the first column, it just describes the control group
        typical_std = 0.1 * max(1e-6, np.mean(np.abs(means[1:]))) + np.std(means[:, 1:])
        stds = typical_std * np.ones_like(means[:, 1:]) * 0.5
    stds = prepend_zero_column(stds, 1e-6 * np.median(np.abs(means)))

    return means, stds


def thompson_policy(
    means: np.ndarray, stds: np.ndarray, num_samples: int = 10000, clip=1e-3
):
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

    if clip:
        stds = np.clip(stds, clip, None)

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
