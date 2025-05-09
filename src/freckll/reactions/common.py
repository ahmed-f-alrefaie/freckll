"""Common functions and equations for reactions."""

import typing as t

import numpy as np

from ..constants import AVO, K_BOLTZMANN, RA
from ..nasa import NasaCoeffs
from ..species import SpeciesDict, SpeciesFormula
from ..types import FreckllArray, FreckllArrayInt


class UniBiReactionSupported(Exception):
    def __init__(self, reactants: list[SpeciesFormula]) -> None:
        super().__init__(f"Reaction {reactants} is not supported. Up to 2 supported only")


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
    kboltz = K_BOLTZMANN * 1e4
    spec1 = reactants[0]
    try:
        spec2 = reactants[1]
    except IndexError:
        spec2 = H2

    #
    mass1 = spec1.monoisotopic_mass / AVO * 1e-3
    mass2 = spec2.monoisotopic_mass / AVO * 1e-3

    reduced_mass = (mass1 + mass2) / 2

    eff_xsec = 4.0 * np.pi * (RA * RA)

    avg_speed = np.sqrt((8 * kboltz * temperature) / (np.pi * reduced_mass))

    k_coll = eff_xsec * avg_speed

    update_mask = k_rate >= k_coll

    if uni_reaction:
        k_kinf = np.divide(k_rate * m_concentration, k_inf, where=k_inf != 0)
        k_rate_coll = k_rate / k_coll
        kinf_zero = k_inf == 0
        k_rate_exceeds_kinf = k_kinf > 1.0
        k_rate_condition = k_rate_coll > k_kinf
        k_rate = np.where(
            update_mask & (kinf_zero | k_rate_exceeds_kinf) & (~kinf_zero) & k_rate_condition,
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

    avg_speed = np.sqrt((8 * K_BOLTZMANN * temperature[None, :]) / (np.pi * reduced_masses[:, None]))

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
            update_mask & uni_reactions & (kinf_zero | k_rate_exceeds_kinf) & (~kinf_zero) & k_rate_condition,
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
    thermo_properties = np.empty(shape=(len(species), 2, temperature.shape[0]), dtype=temperature.dtype)

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
        thermo_inv_reactants: The thermodynamic properties of the reactants.
        thermo_inv_products: The thermodynamic properties of the products.
        k0: The rate constant of the reaction.
        k_inf: The high-pressure limit of the rate constant.
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


def arrhenius_constant(
    a: float | FreckllArray,
    n: float | FreckllArray,
    er: float | FreckllArray,
    temperature: FreckllArray,
) -> FreckllArray:
    r"""Computes Arrhenius rate constants for low and high pressure limits.

    Formula is as follows:

    $$
    k = A T^n \exp(-E_r/T)
    $$

    where:
    - $k$ is the rate constant
    - $A$ is the pre-exponential factor
    - $T$ is the temperature
    - $n$ is the temperature exponent
    - $E_r$ is the activation energy
    - $k$ is the rate constant

    Args:
        a: The pre-exponential factor.
        n: The temperature exponent
        er: The activation energy.

    Returns:
        k: The rate constant of the reaction.

    """
    return t.cast(FreckllArray, (a * temperature**n) * np.exp(-er / temperature))


def plog_interpolate(
    log_points: FreckllArray,
    a_points: FreckllArray,
    n_points: FreckllArray,
    er_points: FreckllArray,
    pressures: FreckllArray,
    temperature: FreckllArray,
) -> tuple[FreckllArray, FreckllArray, FreckllArray]:
    r"""Interpolates the Arrhenius parameters for the given pressures.

    This is for the Pressure-Dependent Arrhenius (PLOG) rate constants.

    The interpolation is given by:

    $$
    \log{k(T, P)} = \log{k_1(T)} + (\log{k_2(T)} - \log{k_1(T)}) \frac{\log{P}- \log{P_1}}{\log{P_2} - \log{P_1}}
    $$


    Args:
        plog_points: The pressures at which the Arrhenius parameters are defined.
        a_points: The pre-exponential factors at the given pressures.
        n_points: The temperature exponents at the given pressures.
        er_points: The activation energies at the given pressures.
        pressures: The pressures at which to interpolate the Arrhenius parameters.

    Returns:
        A tuple of the interpolated pre-exponential factor, temperature exponent, and activation energy.

    """

    log_pressures = np.log10(pressures)

    p_1 = np.searchsorted(log_points, log_pressures)
    p_1 = np.clip(p_1, 0, len(log_points) - 2)
    p_2 = p_1 + 1

    k_1 = np.log10(
        arrhenius_constant(
            a_points[p_1],
            n_points[p_1],
            er_points[p_1],
            temperature,
        )
    )

    k_2 = np.log10(
        arrhenius_constant(
            a_points[p_2],
            n_points[p_2],
            er_points[p_2],
            temperature,
        )
    )

    k_interp = k_1 + (k_2 - k_1) * (log_pressures - log_points[p_1]) / (log_points[p_2] - log_points[p_1])

    # k = arrhenius_constant(
    #     a_points[:, None],
    #     n_points[:, None],
    #     er_points[:, None],
    #     temperature[None, :],
    # )

    # log_k = np.log10(k)

    return 10**k_interp


def check_balance(balance, threshold=1000):
    is_good = True

    if balance > threshold:
        print("Send spending alert")
        is_good = False

    else:
        print("Balance is normal")
        is_good = True

    return is_good
