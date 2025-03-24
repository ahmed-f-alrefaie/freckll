"""Falloff functions for reactions."""

import typing as t

import numpy as np

from ..types import FreckllArray


class FalloffFunction(t.Protocol):
    """Defines a calloff function protocol."""

    def __call__(
        self,
        k0: FreckllArray,
        kinf: FreckllArray,
        m: FreckllArray,
        temperature: FreckllArray,
        *args: float,
    ) -> FreckllArray: ...


def troe_falloff_term(
    k0: FreckllArray,
    kinf: FreckllArray,
    m: FreckllArray,
    temperature: FreckllArray,
    a_troe: float,
    t3: float,
    t1: float,
    t2: float,
) -> FreckllArray:
    r"""Troe Falloff function.

    Computes the center factor:
    $$
    F_{c} = (1-\alpha)\exp{\frac{-T}{T_3}} + \alpha\exp{\frac{-T}{T_1}} + \exp{\frac{-T_2}{T}}
    $$
    The constants:
    $$
    c = -0.4 - 0.67\log_{10}(F_c)
    $$
    $$
    N = 0.75 - 1.27\log_{10}(F_c)
    $$

    The logarithmic falloff factor is:
    $$
    \log_{10}(F) = \frac{\log_{10}(F_c)}{1 + (\log_{10}(\frac{k_0}{k_\infty}) + \frac{c}{N - d\log_{10}(\frac{k_0}{k_\infty}) + c})^2}
    $$

    Finally we return the factor as:
    $$
    F = 10^{\log_{10}(F)}
    $$


    Args:
        k0: The low-pressure rate constant.
        kinf: The high-pressure rate constant.
        m: The concentration of the reactants.
        temperature: The temperature of the reaction.
        a_troe: The Troe falloff parameter.
        t3: The third Troe falloff parameter.
        t1: The first Troe falloff parameter.
        t2: The second Troe falloff parameter.

    Returns:
        falloff: The falloff term.


    """
    # Falloff terms
    d = 0.14
    if t3 == 0.0 or t1 == 0.0:
        return 1.0

    f_cent = (
        (1 - a_troe) * np.exp(-temperature / t3)
        + a_troe * np.exp(-temperature / t1)
        + np.exp(-t2 / temperature)
    )
    log_fcent = np.log10(f_cent)

    c = -0.4 - 0.67 * log_fcent
    nf = 0.75 - 1.27 * log_fcent
    log_k0kinfm = np.log10(k0 * m / kinf)
    log_falloff = log_fcent / (
        1 + (log_k0kinfm + c / (nf - d * (log_k0kinfm + c))) ** 2
    )

    return t.cast(FreckllArray, 10.0**log_falloff)


def sri_falloff(
    k0: FreckllArray,
    kinf: FreckllArray,
    m: FreckllArray,
    temperature: FreckllArray,
    a_sri: float,
    b_sri: float,
    c_sri: float,
    d_sri: float,
    e_sri: float,
) -> FreckllArray:
    r"""Stanford Research Institute falloff function.

    Computes the Stan falloff function, first $X_{sri}$ is computed as:

    $$
    X_{SRI} = \frac{1}{1 + [\log_{10}(\frac{k_0 M}{k_\infty})]^2}
    $$

    The center fall-off factor is:
    $$
    F_{c} = d_{SRI}(a_{SRI} \exp(-\frac{b_{SRI}}{T}) + \exp(-\frac{T}{c_{SRI}}))^{X_{SRI}}\times T^{e_{SRI}}
    $$

    Args:
        k0: The low-pressure rate constant.
        kinf: The high-pressure rate constant.
        m: The concentration of the reactants.
        temperature: The temperature of the reaction.
        a_sri: The SRI falloff parameter.
        b_sri: The SRI falloff parameter.
        c_sri: The SRI falloff parameter.
        d_sri: The SRI falloff parameter.
        e_sri: The SRI falloff parameter.
    """

    x_sri = 1 / (1 + np.log10(k0 * m / kinf) ** 2)

    f_cent = (
        d_sri * (a_sri * np.exp(-b_sri / temperature) + np.exp(-temperature / c_sri))
    ) ** x_sri * temperature**e_sri

    return t.cast(FreckllArray, f_cent)


def no_falloff(
    k0: FreckllArray,
    kinf: FreckllArray,
    m: FreckllArray,
    temperature: FreckllArray,
    *args: float,
) -> FreckllArray:
    """No falloff function."""
    return 1.0
