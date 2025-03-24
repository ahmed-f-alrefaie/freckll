"""Common functions and equations for reactions."""

import numpy as np

from ..constants import AVO, K_BOLTZMANN, RA
from ..nasa import NasaCoeffs
from ..species import SpeciesDict, SpeciesFormula
from ..types import FreckllArray, FreckllArrayInt


class UniBiReactionSupported(Exception):
    def __init__(self, reactants: list[SpeciesFormula]) -> None:
        super().__init__(
            f"Reaction {reactants} is not supported. Up to 2 supported only"
        )


H2 = SpeciesFormula("H2")


def collision_rate_limit(
    reactants: list[SpeciesFormula],
    k_rate: FreckllArray,
    k_inf: FreckllArray,
    m_concentration: FreckllArray,
    temperature: FreckllArray,
) -> FreckllArray:
    """Limits the reaction rate to the collision rate.

    Args:
        reactants: The reactants in the reaction.
        k_rate: The rate constant of the reaction.
        k_inf: high-pressure limit of the rate constant.
        m_concentration: The concentration of the reactants.
        temperature: The temperature of the reaction.


    """
    uni_reaction = len(reactants) == 1

    spec1 = reactants[0]
    try:
        spec2 = reactants[1]
    except IndexError:
        spec2 = H2

    #
    mass1 = spec1.monoisotopic_mass / AVO * 1e-3
    mass2 = spec2.monoisotopic_mass / AVO * 1e-3

    reduced_mass = (mass1 * mass2) / (mass1 + mass2)

    eff_xsec = 4.0 * np.pi * (RA * RA)

    avg_speed = np.sqrt((8 * K_BOLTZMANN * temperature) / (np.pi * reduced_mass))

    k_coll = eff_xsec * avg_speed

    update_mask = k_rate >= k_coll

    if uni_reaction:
        k_kinf = np.divide(k_rate * m_concentration, k_inf, where=k_inf != 0)
        k_rate_coll = k_rate / k_coll
        kinf_zero = k_inf == 0
        k_rate_exceeds_kinf = k_kinf > 1.0
        k_rate_condition = k_rate_coll > k_kinf
        k_rate = np.where(
            update_mask
            & (kinf_zero | k_rate_exceeds_kinf)
            & (~kinf_zero)
            & k_rate_condition,
            k_coll,
            k_rate,
        )
    else:
        k_rate = np.where(update_mask, k_coll, k_rate)

    return k_rate


def collision_rate_array(
    reduced_masses: FreckllArray,
    num_species: FreckllArrayInt,
    k_rate: FreckllArray,
    k_inf: FreckllArray,
    m_concentration: FreckllArray,
    temperature: FreckllArray,
) -> FreckllArray:
    """Limits the reaction rate to the collision rate.

    Args:
        reduced_masses: The reduced masses of the reactants.
        num_species: The number of species in the reaction.
        k_rate: The rate constant of the reaction.
        k_inf: high-pressure limit of the rate constant.
        m_concentration: The concentration of the reactants.
        temperature: The temperature of the reaction.


    """
    eff_xsec = 4.0 * np.pi * (RA * RA)

    avg_speed = np.sqrt(
        (8 * K_BOLTZMANN * temperature[None, :]) / (np.pi * reduced_masses[:, None])
    )

    k_coll = eff_xsec * avg_speed

    update_mask = k_rate >= k_coll

    uni_reactions = num_species[:, None] == 1
    if np.any(uni_reactions):
        k_kinf = np.divide(k_rate * m_concentration, k_inf, where=k_inf != 0)
        k_rate_coll = k_rate / k_coll
        kinf_zero = k_inf == 0
        k_rate_exceeds_kinf = k_kinf > 1.0
        k_rate_condition = k_rate_coll > k_kinf
        k_rate = np.where(
            update_mask
            & uni_reactions
            & (kinf_zero | k_rate_exceeds_kinf)
            & (~kinf_zero)
            & k_rate_condition,
            k_coll,
            k_rate,
        )

    k_rate = np.where(update_mask & ~uni_reactions, k_coll, k_rate)

    return k_rate


def compile_thermodynamic_properties(
    species: list[SpeciesFormula],
    nasa_coeffs: SpeciesDict[NasaCoeffs],
    temperature: FreckllArray,
) -> FreckllArray:
    """Compiles the thermodynamic properties of the species in the reaction.

    Resultant array will be of shape (Nspecies,2, Nlayers)

    Where the second axis is the enthalpy and entropy.

    Args:
        species: The species in the network.
        nasa_coeffs: The NASA polynomial coefficients of the species.
        temperature: The temperature of the reaction.

    Returns:
        The thermodynamic properties of the species.
    """
    thermo_properties = np.empty(
        shape=(len(species), 2, temperature.shape[0]), dtype=temperature.dtype
    )

    for idx, spec in enumerate(species):
        if spec.state != "gas":
            continue
        nasa = nasa_coeffs[spec]
        h, s = nasa(temperature)
        thermo_properties[idx, 0] = h
        thermo_properties[idx, 1] = s

    return thermo_properties


def invert_reaction(
    thermo_inv_reactants: FreckllArray,
    thermo_inv_products: FreckllArray,
    k0: FreckllArray,
    k_inf: FreckllArray,
    temperature: FreckllArray,
) -> tuple[FreckllArray, FreckllArray, FreckllArray]:
    r"""Reverses the reaction.

    Args:
        thermo_products: The thermodynamic properties of the products.
        thermo_reactants: The thermodynamic properties of the reactants.
        k0: The low-pressure rate constant.
        k_inf: The high-pressure rate constant.
        temperature: The temperature of the reaction.

    Returns:
        The inverted rate constants $k_0$, $k_\infty$ and the equilibrium constant $K$.


    """
    from ..constants import ATM_BAR, AVO

    r_si = 8.3144598

    sum_reactants = np.sum(thermo_inv_reactants, axis=0)
    sum_products = np.sum(thermo_inv_products, axis=0)

    delta_h = sum_reactants[0] - sum_products[0]
    delta_s = sum_reactants[1] - sum_products[1]
    exp_dh = np.exp(delta_s - delta_h)

    d_stoic = thermo_inv_reactants.shape[0] - thermo_inv_products.shape[0]

    k_factor = (ATM_BAR * AVO) / (r_si * temperature * 10)

    k_equil = exp_dh * k_factor**d_stoic

    k0_inv = k0 / k_equil

    k_inf_inv = k_inf / k_equil

    return k0_inv, k_inf_inv, k_equil
