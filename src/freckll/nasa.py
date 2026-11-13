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

        NASA7 polynomials have the form:

        \begin{split}
            \frac{H}{RT} &= a_0 + a_1 \frac{T}{2} + a_2 \frac{T^2}{3} + a_3 \frac{T^3}{4} + a_4 \frac{T^4}{5} + \frac{a_5}{T}
            \frac{S}{R} &= a_0 \ln(T) + a_1 T + a_2 \frac{T^2}{2} + a_3 \frac{T^3}{3} + a_4 \frac{T^4}{4} + a_6
        \end{split}

        NASA9 polynomials have the form:

        \begin{split}
            \frac{H}{RT} &= - a_0 \frac{1}{T^2} + a_1 \frac{\ln(T)}{T} + a_2 + a_3 \frac{T}{2} + a_4 \frac{T^2}{3} + a_5 \frac{T^3}{4} + a_6 \frac{T^4}{5} + \frac{a_7}{T}
            \frac{S}{R} &= - a_0 \frac{1}{2 T^2} - a_1 \frac{1}{T} + a_2 \ln(T) + a_3 T + a_4 \frac{T^2}{2} + a_5 \frac{T^3}{3} + a_6 \frac{T^4}{4} + a_8
        \end{split}

        Args:
            temperature: The temperature at which to calculate the coefficients.

        """
        coeffs = np.zeros(shape=(self.b_coeff.shape[0], temperature.shape[0]))

        temperature_greater = temperature > self.x3
        temperature_less = ~temperature_greater

        coeffs[:, temperature_greater] = self.a_coeff[:, None]
        coeffs[:, temperature_less] = self.b_coeff[:, None]

        if coeffs.shape[0] == 7:
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
        elif coeffs.shape[0] == 9:
            h_coeffs: FreckllArray = (
                - coeffs[0] / temperature**2
                + coeffs[1] * np.log(temperature) / temperature
                + coeffs[2]
                + coeffs[3] * temperature / 2
                + coeffs[4] * temperature**2 / 3
                + coeffs[5] * temperature**3 / 4
                + coeffs[6] * temperature**4 / 5
                + coeffs[7] / temperature
            )
            s_coeffs: FreckllArray = (
                - 0.5 * coeffs[0] / temperature**2
                - coeffs[1] / temperature
                + coeffs[2] * np.log(temperature)
                + coeffs[3] * temperature
                + coeffs[4] * temperature**2 / 2
                + coeffs[5] * temperature**3 / 3
                + coeffs[6] * temperature**4 / 4
                + coeffs[8]
            )
        else:
            raise NotImplementedError(f"Unknown NASA format with {coeffs.shape[0]} coefficients : only NASA7 and NASA9 are supported")

        return (h_coeffs, s_coeffs)
