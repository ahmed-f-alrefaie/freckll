"""Reaction equations."""

import typing as t

import numpy as np

from ..species import SpeciesFormula
from ..types import FreckllArray, ReactionFunction
from .common import collision_rate_limit, invert_reaction
from .falloff import FalloffFunction


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


def k0kinf_reaction(
    k0_coeffs: list[float],
    kinf_coeffs: list[float],
    falloff_coeffs: list[float],
    efficiency: FreckllArray,
    invert: bool,
    falloff_function: FalloffFunction,
    reactants: list[SpeciesFormula],
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> list[FreckllArray]:
    r"""Calculates the effective pressure-dependent rate constant.

    This the Arrheniues rate with the low-pressure and high pressure
    limits
    The low pressure limit is:
    $$
    k_0 = A_0 T^{n_0} \exp(-E_{r0}/T)
    $$
    The high pressure limit is:
    $$
    k_\infty = A_i T^{n_i} \exp(-E_{ri}/T)
    $$

    Additionally you can also provide a falloff function to calculate
    the falloff term.

    Args:
        k0_coeffs: The low-pressure rate constant coefficients.
        kinf_coeffs: The high-pressure rate constant coefficients.
        falloff_coeffs: The falloff coefficients.
        efficiency: The efficiency of the reaction.
        invert: Whether to invert the reaction.
        falloff_function: The falloff function to use.
        reactants: The reactants of the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        k_rate: The rate constant of the reaction.
        inv_k_rate: The inverted rate constant of the reaction.


    """
    # k0 terms
    a0, n0, er0, _, _ = k0_coeffs
    # kinf terms
    ai, ni, eri, _, _ = kinf_coeffs

    # Falloff terms

    unimolecular = len(reactants) == 1 or len(products) == 1

    k0 = arrhenius_constant(a0, n0, er0, temperature)
    kinf = arrhenius_constant(ai, ni, eri, temperature)

    def _react(
        concentration: FreckllArray,
        efficiency: FreckllArray = efficiency,
        k0: FreckllArray = k0,
        kinf: FreckllArray = kinf,
        falloff_function: FalloffFunction = falloff_function,
        reactants: list[SpeciesFormula] = reactants,
        products: list[SpeciesFormula] = products,
        temperature: FreckllArray = temperature,
        falloff_coeffs: list[float] = falloff_coeffs,
        invert: bool = invert,
        unimolecular: bool = unimolecular,
    ) -> list[FreckllArray]:
        m: FreckllArray = (efficiency[:, None] * concentration).sum(axis=0)

        falloff = falloff_function(
            k0,
            kinf,
            m,
            temperature,
            *falloff_coeffs,
        )

        k_rate = k0 * falloff / (1 + k0 * m / kinf)

        if unimolecular:
            k_rate *= m

        # Limit the collision rate
        k_rate = collision_rate_limit(reactants, k_rate, kinf, m, temperature)

        if invert:
            k0inv, kinfinv, keq = invert_reaction(thermo_products, thermo_reactants, k0, kinf, temperature)

            k0inv_coll = collision_rate_limit(products, k0inv, kinfinv, m, temperature)
            falloff = falloff_function(k0inv_coll, kinfinv, m, temperature, *falloff_coeffs)

            inv_k_rate = k0inv_coll * falloff / (1 + k0inv_coll * m / kinfinv)

            if unimolecular:
                inv_k_rate *= m

            check_rate = k0inv != k0inv_coll

            k_rate[check_rate] = inv_k_rate[check_rate] * keq[check_rate]

            return [k_rate, inv_k_rate]

        return [k_rate]

    return _react


def decomposition_k0kinf_reaction(
    k0_coeffs: list[float],
    kinf_coeffs: list[float],
    falloff_coeffs: list[float],
    efficiency: FreckllArray,
    invert: bool,
    falloff_function: FalloffFunction,
    reactants: list[SpeciesFormula],
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> ReactionFunction:
    r"""Decomposition reaction rate constant.


    This the Arrheniues rate with the low-pressure and high pressure
    limits
    The low pressure limit is:
    $$
    k_0 = A_0 T^{n_0} \exp(-E_{r0}/T)
    $$
    The high pressure limit is:
    $$
    k_\infty = A_i T^{n_i} \exp(-E_{ri}/T)
    $$

    Additionally you can also provide a falloff function to calculate
    the falloff term.

    Args:
        k0_coeffs: The low-pressure rate constant coefficients.
        kinf_coeffs: The high-pressure rate constant coefficients.
        falloff_coeffs: The falloff coefficients.
        efficiency: The efficiency of the reaction.
        invert: Whether to invert the reaction.
        falloff_function: The falloff function to use.
        reactants: The reactants of the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        A function that takes the concentration of the species and returns the rate

    """
    # k0 terms
    a0, n0, er0, _, _ = k0_coeffs
    # kinf terms
    ai, ni, eri, _, _ = kinf_coeffs

    # Falloff terms
    unimolecular = len(reactants) == 1 or len(products) == 1

    k0 = arrhenius_constant(a0, n0, er0, temperature)
    kinf = arrhenius_constant(ai, ni, eri, temperature)

    def _react(
        concentration: FreckllArray,
        efficiency: FreckllArray = efficiency,
        k0: FreckllArray = k0,
        kinf: FreckllArray = kinf,
        falloff_function: FalloffFunction = falloff_function,
        reactants: list[SpeciesFormula] = reactants,
        products: list[SpeciesFormula] = products,
        temperature: FreckllArray = temperature,
        falloff_coeffs: list[float] = falloff_coeffs,
        invert: bool = invert,
        unimolecular: bool = unimolecular,
    ):
        m: FreckllArray = (efficiency[:, None] * concentration).sum(axis=0)

        k0 = collision_rate_limit(reactants, k0, kinf, m, temperature)

        falloff = falloff_function(
            k0,
            kinf,
            m,
            temperature,
            *falloff_coeffs,
        )

        k_rate = k0 * falloff / (1 + k0 * m / kinf)

        if unimolecular:
            k_rate *= m

        # Limit the collision rate

        if invert:
            k0inv, kinfinv, keq = invert_reaction(thermo_products, thermo_reactants, k0, kinf, temperature)
            # falloff = falloff_function(k0inv, kinfinv, m, temperature, *falloff_coeffs)

            inv_k_rate = k0inv * falloff / (1 + k0inv * m / kinfinv)
            if unimolecular:
                inv_k_rate *= m

            inv_k_rate_coll = collision_rate_limit(products, inv_k_rate, kinfinv, m, temperature)

            check_rate = inv_k_rate != inv_k_rate_coll

            k_rate[check_rate] = inv_k_rate_coll[check_rate] * keq[check_rate]

            return [k_rate, inv_k_rate_coll]

        return [k_rate]

    return _react


def decomposition_reaction(
    k0_coeffs: list[float],
    invert: bool,
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> ReactionFunction:
    """Decomposition reaction rate constant.

    Args:
        k0_coeffs: The low-pressure rate constant coefficients.
        invert: Whether to invert the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        k_rate: The rate constant of the reaction.
        inv_k_rate: The inverted rate constant of the reaction. (if ``invert`` is True)

    """
    a0, n0, er0, _, _ = k0_coeffs

    k0 = arrhenius_constant(a0, n0, er0, temperature)

    def _react(
        concentration: FreckllArray,
        k0: FreckllArray = k0,
        products: list[SpeciesFormula] = products,
        temperature: FreckllArray = temperature,
        thermo_reactants: FreckllArray = thermo_reactants,
        thermo_products: FreckllArray = thermo_products,
        invert: bool = invert,
    ) -> list[FreckllArray]:
        k_rate = np.copy(k0)
        m = np.sum(concentration, axis=0)
        if invert:
            k0inv, _, keq = invert_reaction(thermo_products, thermo_reactants, k0, np.zeros_like(k0), temperature)
            inv_k_rate = k0inv
            if len(products) > 1:
                inv_k_rate_coll = collision_rate_limit(products, inv_k_rate, np.zeros_like(k0), m, temperature)

                check_rate = k0inv != inv_k_rate_coll

                k_rate[check_rate] = inv_k_rate_coll[check_rate] * keq[check_rate]
                inv_k_rate = inv_k_rate_coll
            return [k_rate, inv_k_rate]

        return [k_rate]

    return _react


def corps_reaction(
    k0_coeffs: list[float],
    invert: bool,
    reactants: list[SpeciesFormula],
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> ReactionFunction:
    """Many body reaction rate constant.


    Args:
        k0_coeffs: The low-pressure rate constant coefficients.
        invert: Whether to invert the reaction.
        reactants: The reactants of the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        A function that takes the concentration of the species and returns the rate

    """
    a0, n0, er0, _, _ = k0_coeffs

    k0 = arrhenius_constant(a0, n0, er0, temperature)

    def _react(
        concentration: FreckllArray,
        k0: FreckllArray = k0,
        reactants: list[SpeciesFormula] = reactants,
        products: list[SpeciesFormula] = products,
        temperature: FreckllArray = temperature,
        thermo_reactants: FreckllArray = thermo_reactants,
        thermo_products: FreckllArray = thermo_products,
        invert: bool = invert,
    ) -> list[FreckllArray]:
        k_rate = np.copy(k0)
        m = np.sum(concentration, axis=0)

        if len(reactants) < 3:
            k_rate = collision_rate_limit(reactants, k_rate, np.zeros_like(k0), m, temperature)

        if invert:
            k0inv, _, keq = invert_reaction(thermo_products, thermo_reactants, k0, np.zeros_like(k0), temperature)

            inv_k_rate = k0inv

            if len(products) > 1:
                inv_k_rate_coll = collision_rate_limit(products, inv_k_rate, np.zeros_like(k0), m, temperature)

                check_rate = k0inv != inv_k_rate_coll

                k_rate[check_rate] = inv_k_rate_coll[check_rate] * keq[check_rate]

                inv_k_rate = inv_k_rate_coll
            return [k_rate, inv_k_rate]

        return [k_rate]

    return _react


def de_excitation_reaction(
    k0_coeffs: list[float],
    efficiency: FreckllArray,
    invert: bool,
    reactants: list[SpeciesFormula],
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> ReactionFunction:
    """De-excitation reaction rate constant.

    Args:
        k0_coeffs: The low-pressure rate constant coefficients.
        efficiency: The efficiency of the reaction.
        invert: Whether to invert the reaction.
        reactants: The reactants of the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        A function that takes the concentration of the species and returns the rate

    """
    a0, n0, er0, _, _ = k0_coeffs

    k0 = arrhenius_constant(a0, n0, er0, temperature)

    def _react(
        concentration: FreckllArray,
        efficiency: FreckllArray = efficiency,
        reactants: list[SpeciesFormula] = reactants,
        k0: FreckllArray = k0,
        temperature: FreckllArray = temperature,
        thermo_reactants: FreckllArray = thermo_reactants,
        thermo_products: FreckllArray = thermo_products,
    ) -> FreckllArray:
        m = np.sum(concentration * efficiency[:, None], axis=0)

        k0 = collision_rate_limit(reactants, k0, np.zeros_like(k0), m, temperature)

        k_rate = k0 * m
        if invert:
            k0inv, _, keq = invert_reaction(thermo_products, thermo_reactants, k0, np.zeros_like(k0), temperature)

            inv_k_rate = k0inv * m
            k0inv_coll = collision_rate_limit(products, k0inv, np.zeros_like(k0), m, temperature)
            inv_k_rate_coll = k0inv_coll * m
            check_rate = inv_k_rate != inv_k_rate_coll

            k_rate[check_rate] = inv_k_rate_coll[check_rate] * keq[check_rate]
            inv_k_rate = inv_k_rate_coll

            return [k_rate, inv_k_rate]

        return [k_rate]

    return _react


def k0_reaction(
    k0_coeffs: list[float],
    efficiency: FreckllArray,
    invert: bool,
    reactants: list[SpeciesFormula],
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> ReactionFunction:
    """Reaction rate constant.

    Args:
        k0_coeffs: The low-pressure rate constant coefficients.
        efficiency: The efficiency of the reaction.
        invert: Whether to invert the reaction.
        reactants: The reactants of the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        A function that takes the concentration of the species and returns the rate

    """
    a0, n0, er0, _, _ = k0_coeffs

    k0 = arrhenius_constant(a0, n0, er0, temperature)

    def _react(
        concentration: FreckllArray,
        k0: FreckllArray = k0,
        efficiency: FreckllArray = efficiency,
        reactants: list[SpeciesFormula] = reactants,
        products: list[SpeciesFormula] = products,
        temperature: FreckllArray = temperature,
        thermo_reactants: FreckllArray = thermo_reactants,
        thermo_products: FreckllArray = thermo_products,
        invert: bool = invert,
    ):
        m = np.sum(concentration * efficiency[:, None], axis=0)
        k_rate = k0 * m
        k_rate = collision_rate_limit(reactants, k_rate, np.zeros_like(k0), m, temperature)

        if invert:
            k0inv, _, keq = invert_reaction(thermo_products, thermo_reactants, k0, np.zeros_like(k0), temperature)

            k0inv_coll = collision_rate_limit(products, k0inv, np.zeros_like(k0), m, temperature)

            inv_k_rate = k0inv_coll * m

            check_rate = k0inv != k0inv_coll

            k_rate[check_rate] = inv_k_rate[check_rate] * keq[check_rate]
            return [k_rate, inv_k_rate]

        return [k_rate]

    return _react


def decomposition_k0_reaction(
    k0_coeffs: list[float],
    efficiency: FreckllArray,
    invert: bool,
    reactants: list[SpeciesFormula],
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> ReactionFunction:
    """Decomposition reaction rate constant.

    Args:
        k0_coeffs: The low-pressure rate constant coefficients.
        efficiency: The efficiency of the reaction.
        invert: Whether to invert the reaction.
        reactants: The reactants of the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        A function that takes the concentration of the species and returns the rate

    """
    a0, n0, er0, _, _ = k0_coeffs

    k0 = arrhenius_constant(a0, n0, er0, temperature)

    def _react(
        concentration: FreckllArray,
        k0: FreckllArray = k0,
        efficiency: FreckllArray = efficiency,
        reactants: list[SpeciesFormula] = reactants,
        products: list[SpeciesFormula] = products,
        temperature: FreckllArray = temperature,
        thermo_reactants: FreckllArray = thermo_reactants,
        thermo_products: FreckllArray = thermo_products,
        invert: bool = invert,
    ) -> list[FreckllArray]:
        m = np.sum(concentration * efficiency[:, None], axis=0)
        k0 = collision_rate_limit(reactants, k0, np.zeros_like(k0), m, temperature)
        k_rate = k0 * m

        if invert:
            k0inv, _, keq = invert_reaction(thermo_products, thermo_reactants, k0, np.zeros_like(k0), temperature)

            inv_k_rate = k0inv * m

            inv_k_rate_coll = collision_rate_limit(products, inv_k_rate, np.zeros_like(k0), m, temperature)

            check_rate = inv_k_rate != inv_k_rate_coll

            k_rate[check_rate] = inv_k_rate_coll[check_rate] * keq[check_rate]
            inv_k_rate = inv_k_rate_coll
            return [k_rate, inv_k_rate]

        return [k_rate]

    return _react


def manybody_plog_reaction(
    plog_coeffs: list[float],
    invert: bool,
    reactants: list[SpeciesFormula],
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> ReactionFunction:
    """Many body reaction rate constant.

    Args:
        plog_coeffs: The PLOG rate constant coefficients.
        invert: Whether to invert the reaction.
        reactants: The reactants of the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        A function that takes the concentration of the species and returns the rate

    """
    from scipy.interpolate import RectBivariateSpline

    from freckll.constants import ATM_BAR

    BAR_ATM = 1.0 / ATM_BAR

    p0_plog, a0_plog, n0_plog, er0_plog = np.split(np.array(plog_coeffs), 4)
    p0_plog = np.log10(p0_plog)
    p_log = np.log10(pressure * BAR_ATM * 1e-3)

    temperature_test = np.linspace(100, 5000, 5000)

    _, a0 = np.meshgrid(temperature_test, a0_plog, indexing="ij")
    _, p0 = np.meshgrid(temperature_test, p0_plog, indexing="ij")
    _, n0 = np.meshgrid(temperature_test, n0_plog, indexing="ij")
    _temperature, er0 = np.meshgrid(temperature_test, er0_plog, indexing="ij")

    k0_log = arrhenius_constant(a0, n0, er0, _temperature)

    f = RectBivariateSpline(temperature_test, p0_plog, k0_log)
    k0 = np.array([f(t, p)[0] for t, p in zip(temperature, p_log)])[:, 0]

    def _react(concentration: FreckllArray) -> list[FreckllArray]:
        m = np.sum(concentration, axis=0)

        krate = k0
        if len(reactants) < 3:
            krate = collision_rate_limit(reactants, k0, np.zeros_like(k0), m, temperature)

        if invert:
            k0inv, _, keq = invert_reaction(thermo_products, thermo_reactants, k0, np.zeros_like(k0), temperature)

            inv_k_rate = k0inv
            if len(products) > 1:
                inv_k_rate_coll = collision_rate_limit(products, inv_k_rate, np.zeros_like(k0), m, temperature)

                check_rate = inv_k_rate != inv_k_rate_coll

                krate[check_rate] = inv_k_rate_coll[check_rate] * keq[check_rate]
                inv_k_rate = inv_k_rate_coll
            return [krate, inv_k_rate]

        return [krate]

    return _react


def decomposition_plog(
    plog_coeffs: list[float],
    invert: bool,
    products: list[SpeciesFormula],
    temperature: FreckllArray,
    pressure: FreckllArray,
    thermo_reactants: FreckllArray,
    thermo_products: FreckllArray,
) -> ReactionFunction:
    """Decomposition reaction rate constant.

    Args:
        plog_coeffs: The PLOG rate constant coefficients.
        invert: Whether to invert the reaction.
        products: The products of the reaction.
        temperature: The temperature of the species.
        pressure: The pressure of the species.
        thermo_reactants: The thermodynamic properties of the reactants.
        thermo_products: The thermodynamic properties of the products.

    Returns:
        A function that takes the concentration of the species and returns the rate
    """
    from scipy.interpolate import RectBivariateSpline

    from freckll.constants import ATM_BAR

    BAR_ATM = 1.0 / ATM_BAR

    p0_plog, a0_plog, n0_plog, er0_plog = np.split(np.array(plog_coeffs), 4)
    p0_plog = np.log10(p0_plog)
    p_log = np.log10(pressure * BAR_ATM * 1e-3)

    temperature_test = np.linspace(100, 5000, 5000)

    _, a0 = np.meshgrid(temperature_test, a0_plog, indexing="ij")
    _, p0 = np.meshgrid(temperature_test, p0_plog, indexing="ij")
    _, n0 = np.meshgrid(temperature_test, n0_plog, indexing="ij")
    _temperature, er0 = np.meshgrid(temperature_test, er0_plog, indexing="ij")

    k0_log = arrhenius_constant(a0, n0, er0, _temperature)

    f = RectBivariateSpline(temperature_test, p0_plog, k0_log)
    k0 = np.array([f(t, p)[0] for t, p in zip(temperature, p_log)])[:, 0]

    def _react(concentration: FreckllArray) -> list[FreckllArray]:
        m = np.sum(concentration, axis=0)

        krate = k0

        if invert:
            k0inv, _, keq = invert_reaction(thermo_products, thermo_reactants, k0, np.zeros_like(k0), temperature)

            inv_k_rate = k0inv
            if len(products) > 1:
                inv_k_rate_coll = collision_rate_limit(products, inv_k_rate, np.zeros_like(k0), m, temperature)

                check_rate = inv_k_rate != inv_k_rate_coll

                krate[check_rate] = inv_k_rate_coll[check_rate] * keq[check_rate]
                inv_k_rate = inv_k_rate_coll
            return [krate, inv_k_rate]

        return [krate]

    return _react
