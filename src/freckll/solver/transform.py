"""Transform functions for solving the ODE."""

import numpy as np
from scipy import sparse

from ..types import FreckllArray


class Transform:
    """Base class for transform functions."""

    def transform(self, y: FreckllArray) -> FreckllArray:
        """Transform the input value."""
        raise NotImplementedError("Transform function not implemented.")

    def inverse_transform(self, u: FreckllArray) -> FreckllArray:
        """Inverse transform the input value."""
        raise NotImplementedError("Inverse transform function not implemented.")

    def derivative(self, y: FreckllArray) -> FreckllArray:
        """Derivative of the transform function."""
        raise NotImplementedError("Derivative function not implemented.")

    def second_derivative(self, y: FreckllArray) -> FreckllArray:
        """Second derivative of the transform function."""
        raise NotImplementedError("Second derivative function not implemented.")

    def transform_jacobian(self, jacobian: sparse.spmatrix, y: FreckllArray, f: FreckllArray) -> sparse.spmatrix:
        """Transform the Jacobian matrix."""

        s_prime = self.derivative(y)
        s_dprime = self.second_derivative(y)

        jac_u = jacobian.multiply(s_prime[:, None]).multiply(1 / s_prime)

        jac_u = jac_u + sparse.diags(s_dprime * f * y, 0, shape=jacobian.shape)

        return jac_u


class UnityTransform(Transform):
    def transform(self, y: FreckllArray) -> FreckllArray:
        """Transform the input value."""
        return y

    def inverse_transform(self, u: FreckllArray) -> FreckllArray:
        """Inverse transform the input value."""
        return u

    def derivative(self, y: FreckllArray) -> FreckllArray:
        """Derivative of the transform function."""
        return np.ones_like(y)

    def second_derivative(self, y: FreckllArray) -> FreckllArray:
        """Second derivative of the transform function."""
        return np.zeros_like(y)


class LogTransform(Transform):
    def transform(self, y: FreckllArray) -> FreckllArray:
        """Transform the input value."""
        return np.log(y)

    def inverse_transform(self, u: FreckllArray) -> FreckllArray:
        """Inverse transform the input value."""
        return np.exp(u)

    def derivative(self, y: FreckllArray) -> FreckllArray:
        """Derivative of the transform function."""
        return 1 / (y)

    def second_derivative(self, y: FreckllArray) -> FreckllArray:
        """Second derivative of the transform function."""
        return -1 / (y**2)


class Log10Transform(Transform):
    def transform(self, y: FreckllArray) -> FreckllArray:
        """Transform the input value."""
        return np.log10(y)

    def inverse_transform(self, u: FreckllArray) -> FreckllArray:
        """Inverse transform the input value."""
        return 10**u

    def derivative(self, y: FreckllArray) -> FreckllArray:
        """Derivative of the transform function."""
        return 1 / (y * np.log(10))

    def second_derivative(self, y: FreckllArray) -> FreckllArray:
        """Second derivative of the transform function."""
        return -1 / (y**2 * np.log(10))


class LogitTransform(Transform):
    def transform(self, y: FreckllArray) -> FreckllArray:
        """Transform the input value."""
        from scipy.special import logit

        return logit(y)

    def inverse_transform(self, u: FreckllArray) -> FreckllArray:
        """Inverse transform the input value."""
        from scipy.special import expit

        return expit(u)

    def derivative(self, y: FreckllArray) -> FreckllArray:
        """Derivative of the transform function."""
        return 1 / (y * (1 - y))

    def second_derivative(self, y: FreckllArray) -> FreckllArray:
        """Second derivative of the transform function."""
        return -1 / (y**2 * (1 - y) ** 2)
