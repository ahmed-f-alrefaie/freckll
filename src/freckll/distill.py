"""Contains distillation algorithms for solving chemical kinetics."""

import typing as t

import numba as nb
import numpy as np

from .types import FreckllArray


class UnsupportedDimension(Exception):
    """Raised when the dimension of the array is not supported."""

    def __init__(self) -> None:
        super().__init__("Unsupported dimension. Only 1D, 2D and 3D allowed.")


@nb.njit(nogil=True)
def distill_jit_1d(array: FreckllArray) -> FreckllArray:
    """Distill the given 1D array.
    Uses algorithm 4.3 from Kahan.

    """
    for i in range(1, array.shape[0]):
        x = array[i] + array[i - 1]
        z = x - array[i]
        array[i - 1] = (array[i] - (x - z)) + (array[i - 1] - z)
        array[i] = x
    return array


@nb.njit(nogil=True)
def distill_jit_3d(array: FreckllArray) -> FreckllArray:
    for i in range(1, array.shape[0]):
        for j in range(0, array.shape[1]):
            for k in range(0, array.shape[2]):
                x = array[i, j, k] + array[i - 1, j, k]
                z = x - array[i, j, k]
                array[i - 1, j, k] = (array[i, j, k] - (x - z)) + (array[i - 1, j, k] - z)
                array[i, j, k] = x
    return array


@nb.njit(nogil=True)
def distill_jit_2d(array: FreckllArray) -> FreckllArray:
    """Distill the given 2D array.
    Uses algorithm 4.3 from Kahan.

    """
    for i in range(1, array.shape[0]):
        for j in range(0, array.shape[1]):
            x = array[i, j] + array[i - 1, j]
            z = x - array[i, j]
            array[i - 1, j] = (array[i, j] - (x - z)) + (array[i - 1, j] - z)
            array[i, j] = x
    return array


def distill_1d(array: FreckllArray, k: int) -> FreckllArray:
    """Distill K-times the given array."""
    for _ in range(k):
        array = distill_jit_1d(array)
    return array


def distill_2d(array: FreckllArray, k: int, axis: int = 0) -> FreckllArray:
    """Distill K-times the given array."""
    for _ in range(k):
        array = distill_jit_2d(array)
    return array


def distill_3d(array: FreckllArray, k: int, axis: int = 0) -> FreckllArray:
    """Distill K-times the given array."""
    for _ in range(k):
        array = distill_jit_3d(array)
    return array


def ksum(array: FreckllArray, k: int = 2, inplace: bool = False) -> FreckllArray:
    """Sum the array K-times."""
    result = array.copy() if not inplace else array
    if array.ndim == 1:
        result = distill_1d(result, k)
    elif array.ndim == 2:
        result = distill_2d(result, k)
    elif array.ndim == 3:
        result = distill_3d(result, k)
    else:
        raise UnsupportedDimension
    return t.cast(FreckllArray, np.sum(result, axis=0))
