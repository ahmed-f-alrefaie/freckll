"""Utility functions for Freckll."""

import numpy as np
import numpy.typing as npt


def n_largest_index(array: npt.NDArray, n: int, axis: int = 0) -> npt.NDArray[np.integer]:
    """Return the indices of the n largest elements along the given axis."""
    return np.argsort(array, axis=axis)[-1 : -n - 1 : -1]
