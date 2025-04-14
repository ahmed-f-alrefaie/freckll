"""Utility functions for Freckll."""

import numpy as np
import numpy.typing as npt
from astropy import units as u
from scipy import sparse

from .reactions.photo import StarSpectra


def n_largest_index(array: npt.NDArray, n: int, axis: int = 0) -> npt.NDArray[np.integer]:
    """Return the indices of the n largest elements along the given axis."""
    return np.argsort(array, axis=axis)[-1 : -n - 1 : -1]


def convert_to_banded(mat: sparse.sparray, band: int) -> npt.NDArray[np.float64]:
    import numpy as np
    from scipy.sparse import find

    lower_band = band  # mat[:, 0].indices[-1] or mat[0].indices[-1]
    # lower_band = find(mat[:,0])[0][-1]

    upper_band = lower_band

    ab = np.zeros(shape=(upper_band + lower_band + 1, mat.shape[0]))
    # diag_index = np.arange(0,mat.shape[0])
    # diagonals = [(kth_diag_indices(mat,x),mat.diagonal(x)) for x in range(-lower_band,upper_band)]
    row, col, values = find(mat)
    ab[upper_band + row - col, col] = values
    # for indices,vals in diagonals:
    #     row,col = indices
    #     ab[upper_band+row-col,col] = mat[row,col]

    return ab


def convert_to_banded_lsoda(mat: sparse.sparray, band: int):
    import numpy as np
    from scipy.sparse import find

    lower_band = band
    upper_band = lower_band

    ab = np.zeros(shape=(upper_band + lower_band * 2 + 1, mat.shape[0]))
    row, col, values = find(mat)
    ab[upper_band + row - col, col] = values

    return ab


def blackbody(spectral: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """Compute the blackbody spectrum.

    Args:
        spectral: The spectral range.
        temperature: The temperature of the blackbody.
    Returns:
        The blackbody spectrum.


    """
    from astropy.constants import c, h, k_B

    spectral = spectral.to(u.nm, equivalencies=u.spectral())
    temperature = temperature.to(u.K)

    return (2 * h * c**2) / (spectral**5) * (1 / (np.exp((h * c) / (spectral * k_B * temperature)) - 1))


def rescale_star_spectrum(
    star_spectrum: StarSpectra,
    current_radius: u.Quantity,
    new_radius: u.Quantity,
    current_temperature: u.Quantity,
    new_temperature: u.Quantity,
) -> StarSpectra:
    """Rescale the star spectrum to a new temperature.

    Args:
        star_spectrum: The star spectrum to rescale.
        current_temperature: The current temperature of the star spectrum.
        new_temperature: The new temperature to rescale to.

    Returns:
        The rescaled star spectrum.
    """

    current_bb = blackbody(star_spectrum.wavelength, current_temperature).decompose().value
    new_bb = blackbody(star_spectrum.wavelength, new_temperature).decompose().value
    # fix zeros
    # current_bb[current_bb < 1e-50] = 1e-50
    # new_bb[new_bb < 1e-50] = 1e-50

    current_bb = np.maximum(current_bb, 1e-50)
    new_bb = np.maximum(new_bb, 1e-50)

    ratio = (new_bb / current_bb) * (new_radius / current_radius) ** 2

    return StarSpectra(
        wavelength=star_spectrum.wavelength,
        flux=star_spectrum.flux * ratio,
        reference_distance=star_spectrum.reference_distance,
    )


def interpolate_pressure(
    pressure: u.Quantity,
    data: u.Quantity,
    new_pressure: u.Quantity,
) -> u.Quantity:
    from scipy.interpolate import interp1d

    pressure = pressure.to(new_pressure.unit)
    data_f = interp1d(np.log10(pressure.value), data.value)
    data = data_f(np.log10(new_pressure.value)) << data.unit

    return data
