"""NASA polynomials calculation and loading"""

from dataclasses import dataclass

import numpy as np

from .species import SpeciesFormula
from .types import FreckllArray


@dataclass
class NasaCoeffs:
    """Dataclass to store NASA coefficients."""
    species: SpeciesFormula
    x1: float
    x2: float
    x3: float
    a_coeff: FreckllArray
    b_coeff: FreckllArray

    def __call__(self, temperature: FreckllArray) -> tuple[FreckllArray, FreckllArray]:
        r"""Calculate coefficients for a given temperature.

        Nasa polynomials have the form:

        \begin{split}
            \frac{H}{RT} &= a_0 + a_1 \frac{T}{2} + a_2 \frac{T^2}{3} + a_3 \frac{T^3}{4} + a_4 \frac{T^4}{5} + \frac{a_5}{T}
            \frac{S}{RT} &= a_0 \ln(T) + a_1 T + a_2 \frac{T^2}{2} + a_3 \frac{T^3}{3} + a_4 \frac{T^4}{4} + a_6
        \end{split}

        Args:
            temperature: The temperature at which to calculate the coefficients.

        """
        coeffs = np.zeros(shape=(self.b_coeff.shape[0], temperature.shape[0]))

        temperature_greater = temperature > self.x3
        temperature_less = ~temperature_greater

        coeffs[:, temperature_greater] = self.a_coeff[:, None]
        coeffs[:, temperature_less] = self.b_coeff[:, None]

        h_coeffs: FreckllArray = (
            coeffs[0]
            + coeffs[1] * temperature / 2
            + coeffs[2] * temperature**2 / 3
            + coeffs[3] * temperature**3 / 4
            + coeffs[4] * temperature**4 / 5
            + coeffs[5] / temperature
        )
        s_coeffs: FreckllArray = (
            coeffs[0] * np.log(temperature)
            + coeffs[1] * temperature
            + coeffs[2] * temperature**2 / 2
            + coeffs[3] * temperature**3 / 3
            + coeffs[4] * temperature**4 / 4
            + coeffs[6]
        )

        return (h_coeffs, s_coeffs)
