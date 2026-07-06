"""
Regression tests for the numpy>=2 migration.

These guard the hard breaks that numpy 2 introduces:
  * ``numpy.distutils`` (and its ``is_sequence`` helper) is removed;
  * ``np.trapz`` is removed in favour of ``np.trapezoid``.

They must pass on the upgraded stack (numpy 2.x) with no reliance on any
removed numpy internals.
"""
import importlib

import numpy as np
import pandas as pd


def test_is_sequence_helper_semantics():
    """A faithful reimplementation of numpy.distutils.misc_util.is_sequence.

    Sequences (list/tuple/ndarray/Series/dict) are True; strings and scalars
    are False.  In particular ``is_sequence("y")`` must be False, matching the
    historical numpy.distutils behaviour that ``data_utils`` relies on.
    """
    from causaltune.utils import is_sequence

    assert is_sequence([1, 2, 3]) is True
    assert is_sequence((1, 2)) is True
    assert is_sequence(np.array([1, 2, 3])) is True
    assert is_sequence(pd.Series([1, 2, 3])) is True

    assert is_sequence("abc") is False
    assert is_sequence(5) is False
    assert is_sequence(3.14) is False
    assert is_sequence(None) is False


def test_modules_import_without_numpy_distutils():
    """The two modules that used numpy.distutils must import cleanly on numpy 2."""
    for mod in ("causaltune.data_utils", "causaltune.models.monkey_patches"):
        m = importlib.import_module(mod)
        importlib.reload(m)  # ensure a fresh import path, no cached success


def test_thompson_probabilities_use_trapezoid():
    """calculate_probabilities_per_row relied on np.trapz (removed in numpy 2).

    It must now work via np.trapezoid and return finite per-row probabilities
    that sum to 1.
    """
    from causaltune.score.thompson import calculate_probabilities_per_row

    means = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, -1.0]])
    stds = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])

    probs = calculate_probabilities_per_row(means, stds)

    assert probs.shape == means.shape
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(means.shape[0]), atol=1e-6)
