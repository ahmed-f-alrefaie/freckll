"""Test utility functions."""


def test_n_largest_index():
    import numpy as np

    from freckll.utils import n_largest_index

    test = np.array([1, 2, 3, 4, 5])
    n = 3

    result = n_largest_index(test, n)

    assert np.all(result == np.array([4, 3, 2]))
