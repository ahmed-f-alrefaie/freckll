"""Computation of diffusion coefficients."""

from .species import SpeciesFormula, SpeciesDict
from .types import FreckllArray
import numpy as np
import typing as t
from astropy import units as u
_diffusion_volumes = SpeciesDict(
    {
        SpeciesFormula("C"): 15.9,
        SpeciesFormula("H"): 2.31,
        SpeciesFormula("O"): 6.11,
        SpeciesFormula("N"): 4.54,
        SpeciesFormula("S"): 22.9,
        SpeciesFormula("F"): 14.7,
        SpeciesFormula("Cl"): 21.0,
        SpeciesFormula("Br"): 21.9,
        SpeciesFormula("I"): 29.8,
        SpeciesFormula("He"): 2.67,
        SpeciesFormula("Ne"): 5.98,
        SpeciesFormula("Ar"): 16.2,
        SpeciesFormula("Kr"): 24.5,
        SpeciesFormula("Xe"): 32.7,
        SpeciesFormula("H2"): 6.12,
        SpeciesFormula("D2"): 6.84,
        SpeciesFormula("N2"): 18.5,
        SpeciesFormula("O2"): 16.3,
        SpeciesFormula("CO"): 18.0,
        SpeciesFormula("CO2"): 26.9,
        SpeciesFormula("NH3"): 20.7,
        SpeciesFormula("H2O"): 13.1,
        SpeciesFormula("SF6"): 71.3,
        SpeciesFormula("SO2"): 41.8,
        SpeciesFormula("Cl2"): 38.4,
        SpeciesFormula("Br2"): 69.0,
        SpeciesFormula("N2O"): 35.9,
    }
)


def diffusion_volume(species: SpeciesFormula) -> float:
    """Compute the diffusion volume of a species.

    Args:
        species: The species to compute the diffusion volume

    Returns:
        The diffusion volume of the species.
    """
    if species in _diffusion_volumes:
        return _diffusion_volumes[species]

    # Compute it from constituent atoms
    volume = 0.0
    composition = species.composition().asdict()
    for key, (count, _, _) in composition.items():
        volume += count * _diffusion_volumes.get(key, 0.0)

    return volume


def molecular_diffusion(
    species: list[SpeciesFormula],
    number_density: u.Quantity,
    temperature: u.Quantity,
    pressure: u.Quantity,
) -> u.Quantity:
    """Compute the molecular diffusion term for a species.

    Args:
        species: The species to compute the molecular diffusion term.
        number_density: The number density of the atmosphere.
        temperature: The temperature of the species.
        pressure: The pressure of the species.

    Returns:
        The molecular diffusion term along atmosphere.

    """
    from .utils import n_largest_index

    y = (number_density / np.sum(number_density, axis=0)).decompose().value
    sigma = np.array([s.diffusion_volume for s in species])
    mole_masses = np.array([s.monoisotopic_mass for s in species])

    index_1, index_2 = n_largest_index(y, 2, axis=0)

    mass_over_one = 1 / np.maximum(mole_masses, 1.0)

    mass_ab_one = 2.0 / (mass_over_one[:, None] + mass_over_one[None, index_1])
    mass_ab_two = 2.0 / (mass_over_one[:, None] + mass_over_one[None, index_2])


    pressure_bar = pressure.to(u.bar).value
    temperature = temperature.to(u.K).value
    diff_1 = (0.00143 * temperature[None, :] ** 1.75) / (
        pressure_bar[None,]
        * np.sqrt(mass_ab_one)
        * (sigma[:, None] ** (1 / 3) + sigma[None, index_1] ** (1 / 3)) ** 2
    )

    diff_2 = (0.00143 * temperature[None, :] ** 1.75) / (
        pressure_bar[None,]
        * np.sqrt(mass_ab_two)
        * (sigma[:, None] ** (1 / 3) + sigma[None, index_2] ** (1 / 3)) ** 2
    )

    layer_idx = np.arange(number_density.shape[1], dtype=np.int64)

    y_diff_1 = y[index_1,layer_idx]
    y_diff_2 = y[index_2,layer_idx]

    diff_mol = 1.0 / (y_diff_1[None, :] / diff_1 + y_diff_2[None, :] / diff_2)
    diff_mol[index_2, layer_idx] = diff_1[index_2, layer_idx]
    diff_mol[index_1, layer_idx] = 0.0

    return diff_mol << u.cm**2 / u.s
