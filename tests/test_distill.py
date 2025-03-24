"""Tests distillation algorithm."""

import numpy as np
import pytest


def test_distill_hit_1d():
    """Test 1D distillation."""
    from freckll.distill import distill_jit_1d

    test = np.array([1e-16, 1, -1e-16])
    assert test.sum() != 1
    distill_jit_1d(test)
    assert test.sum() == 1


def test_distill_jit_2d():
    """Test 2D distillation."""
    from freckll.distill import distill_jit_2d

    test = np.array([1e-16, 1, -1e-16, 1e16, -1e16, 1e16, -1e16])
    test = np.stack([test, test],axis=1)

    assert test.sum() != 2
    distill_jit_2d(test)
    assert test.sum() == 2


def test_distill():
    """Test distillation."""
    from freckll.distill import distill_1d, distill_2d

    test = np.array([1e-16, 1, -1e-16])
    assert test.sum() != 1
    distill_1d(test, 3)
    assert test.sum() == 1

    test = np.array([1e-16, 1, -1e-16, 1e16, -1e16, 1e16, -1e16])
    test = np.vstack([test, test]).T
    assert test.sum() != 2
    distill_2d(test, 3)
    assert test.sum() == 2

    test = np.array([1e-16, 1, -1e-16, 1e16, -1e16, 1e16, -1e16])
    test = np.vstack([test, test])
    test = test.T
    distill_2d(test, 3, axis=0)
    assert test.sum() == 2


def test_ksum():
    from freckll.distill import ksum

    test = np.array([1e-16, 1, -1e-16])
    assert test.sum() != 1
    assert ksum(test, 3) == 1
    # Make sure array is not changed
    np.testing.assert_equal(test, np.array([1e-16, 1, -1e-16]))

    test = np.array([1e-16, 1, -1e-16, 1e16, -1e16, 1e16, -1e16])
    test = np.vstack([test, test]).T
    assert test.sum() != 2
    result = ksum(test, 3)
    np.testing.assert_allclose(result, np.array([1, 1]))
    # Make sure array is not changed
    np.testing.assert_equal(
        test,
        np.array([
            [1e-16, 1, -1e-16, 1e16, -1e16, 1e16, -1e16],
            [1e-16, 1, -1e-16, 1e16, -1e16, 1e16, -1e16],
        ]).T,
    )

    test = np.array([1e-16, 1, -1e-16, 1e16, -1e16, 1e16, -1e16])
    test = np.vstack([test, test])
    test = test.T
    result = ksum(test, 3)
    np.testing.assert_allclose(result, np.array([1, 1]))
    # Make sure array is not changed
    np.testing.assert_equal(
        test,
        np.array([
            [1e-16, 1e-16],
            [1, 1],
            [-1e-16, -1e-16],
            [1e16, 1e16],
            [-1e16, -1e16],
            [1e16, 1e16],
            [-1e16, -1e16],
        ]),
    )


def test_ksum_raise_error():
    # Check if the function raises an error for 3D arrays
    from freckll.distill import UnsupportedDimension, ksum

    test = np.random.rand(2, 2, 2, 2)
    with pytest.raises(UnsupportedDimension):
        ksum(test, 3)
