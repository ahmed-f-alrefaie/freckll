"""Handles computing the kinetics of a reaction."""

import numpy as np
from astropy import constants as const
from astropy import units as u

from .species import SpeciesFormula
from .types import FreckllArray


class AltitudeSolveError(Exception):
    pass


def gravity_at_height(mass: u.Quantity, radius: u.Quantity, altitude: u.Quantity) -> u.Quantity:
    r"""Compute the gravity at a given altitude.

    The gravity at a given altitude is given by:

    $$
        g = \frac{Gm}{r^2}
    $$

    Where $G$ is the gravitational constant, $m$ is the mass of the planet,
    and $r$ is the radius of the planet.

    Args:
        mass: The mass of the planet.
        radius: The radius of the planet.
        altitude: The altitude at which to compute the gravity.

    """
    return const.G * mass / (radius + altitude) ** 2


def air_density(temperature: u.Quantity, pressure: u.Quantity) -> u.Quantity:
    r"""Compute the density of the atmosphere.

    The *air* density of the atmosphere is given by:

    $$
        \rho = \frac{P}{k_BT}
    $$

    Where $P$ is the pressure, $k_B$ is the Boltzmann constant, and $T$ is the temperature.

    """
    return pressure / (const.k_B * temperature)


def scaleheight(temperature: u.Quantity, gravity: u.Quantity, mass: u.Quantity) -> u.Quantity:
    r"""Compute the scale height of the atmosphere.

    The scale height is given by:

    $$
        H = \frac{k_BT}{mg}
    $$

    Where $k_B$ is the Boltzmann constant, $T$ is the temperature,
    $m$ is the molar mass, and $g$ is the gravity.

    Args:
        temperature: The temperature.
        gravity: The gravity at a given altitude.
        mass: Mass .

    """
    return const.k_B * temperature / (mass * gravity)


def solve_altitude_profile(
    temperature: u.Quantity, mu: u.Quantity, pressures: u.Quantity, planet_mass: u.Quantity, planet_radius: u.Quantity
) -> u.Quantity:
    r"""Solve altitude corresponding to given pressure levels.

    Solves the hydrostatic equilibrium equation to compute the altitude corresponding to the given pressure levels.

    $$
    \frac{dz}{dP} = -\frac{1}{\rho g}
    $$


    Args:
        temperature: Temperature profile as a function of pressure.
        mu: Mean molecular weight profile as a function of pressure.
        pressures: Pressure levels.
        planet_mass: Mass of the planet.
        planet_radius: Radius of the planet.

    Returns:
        Altitude profile corresponding to the given pressure
    """
    from astropy import constants as const
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    G = const.G.value

    density = (air_density(temperature, pressures) * mu).to(u.kg / u.m**3).value

    planet_mass = planet_mass.to(u.kg).value
    planet_radius = planet_radius.to(u.m).value
    pressures = pressures.to(u.Pa).value
    # Ensure pressure is in decreasing order for interpolation
    sort_idx = np.argsort(pressures)[::-1]
    pressures_sorted = pressures[sort_idx]
    density_sorted = density[sort_idx]

    # Create interpolators for T(P) and mu(P)
    rho_interp = interp1d(pressures_sorted, density_sorted, kind="linear", copy=False, fill_value="extrapolate")

    # Define the ODE function dz/dP
    def dzdP(P, z):
        rho = rho_interp(P)

        g = G * planet_mass / (planet_radius + z) ** 2
        return -1.0 / (rho * g)

    P_surface = pressures_sorted[0]

    # Integrate from P_surface down to the minimum pressure in the data
    P_min = pressures_sorted.min()
    P_span = (P_surface, P_min)
    initial_z = [0.0]  # Starting altitude at surface

    # Solve the ODE
    sol = solve_ivp(dzdP, P_span, initial_z, dense_output=True)
    if sol.success is False:
        raise AltitudeSolveError
    # Generate altitude at the original pressure points (interpolate if necessary)
    # Reverse pressures to increasing order for interpolation
    P_eval = np.sort(pressures)
    z_eval = sol.sol(P_eval)[0]

    # Ensure altitudes are in the same order as input pressures
    return z_eval[np.argsort(sort_idx)][::-1] << u.m


def deltaz_terms(
    altitude: u.Quantity,
) -> tuple[FreckllArray, FreckllArray, FreckllArray, FreckllArray, FreckllArray, FreckllArray]:
    r"""Compute the delta z terms.
    Computes the delta z terms for the finite difference scheme.
    The delta z terms are given by:
    $$
        \Delta z = z_{i+1} - z_i
    $$
    $$
        \Delta z_{+} = z_{i+1} - z_i
    $$
    $$
        \Delta z_{-} = z_i - z_{i-1}
    $$

    and the inverse delta z terms are given by:
    $$
        \frac{1}{\Delta z} = \frac{1}{z_{i+1} - z_i}
    $$
    $$
        \frac{1}{\Delta z_{+}} = \frac{1}{z_{i+1} - z_i}
    $$
    $$
        \frac{1}{\Delta z_{-}} = \frac{1}{z_i - z_{i-1}}
    $$


    Args:
        altitude: The altitude

    Returns:
        delta_z_plus: The delta z plus term.
        delta_z_minus: The delta z minus term.
        inv_dz: The inverse delta z term.
        inv_dz_minus: The inverse delta z minus term.
        inv_dz_plus: The inverse delta z plus term.


    """

    delta_z = np.zeros_like(altitude)
    delta_z_p = np.zeros_like(altitude)
    delta_z_m = np.zeros_like(altitude)

    delta_z_p[:-1] = altitude[1:] - altitude[:-1]
    delta_z_m[1:] = altitude[1:] - altitude[:-1]

    delta_z[:-1] = altitude[1:] - altitude[:-1]
    delta_z[-1] = altitude[-1] - altitude[-2]
    delta_z[0] = altitude[1] - altitude[0]

    inv_dz = 1.0 / (0.5 * delta_z_p + 0.5 * delta_z_m)
    inv_dz[0] = 1.0 / (delta_z[0])
    inv_dz[-1] = 1.0 / (delta_z[-1])

    inv_dz_m = 1.0 / (delta_z_m)
    inv_dz_p = 1.0 / (delta_z_p)

    return delta_z, delta_z_p, delta_z_m, inv_dz, inv_dz_p, inv_dz_m


def diffusive_terms(
    planet_radius: u.Quantity,
    planet_mass: u.Quantity,
    altitude: u.Quantity,
    mu: u.Quantity,
    temperature: u.Quantity,
    masses: u.Quantity,
    delta_z: u.Quantity,
    delta_z_plus: u.Quantity,
    delta_z_minus: u.Quantity,
    inv_dz_plus: u.Quantity,
    inv_dz_minus: u.Quantity,
    alpha: float = 0.0,
) -> tuple[u.Quantity, u.Quantity]:
    r"""Compute the diffusive term.

    Computes the staggered gridpoints for the diffusive term.

    We use the following finite difference scheme:

    $$
    \frac{\partial}{\partial z}\left(\frac{1}{H}\frac{\partial y}{\partial z}\right)
    $$

    Args:
        planet_radius: Radius of planet in kilometers.
        planet_mass: Mass of planet in kg.
        altitude: The altitude in km.
        mu: The mean molecular weight in kg.
        temperature: The temperature in K.
        masses: The molar masses in kg.
        delta_z: The delta z term.
        delta_z_plus: The delta z plus term.
        delta_z_minus: The delta z minus term.
        inv_dz_plus: The inverse delta z plus term.
        inv_dz_minus: The inverse delta z minus term.
        alpha: The alpha parameter to include temperature term.

    """

    # cm/m2
    central_g = gravity_at_height(planet_mass, planet_radius, altitude)
    plus_g = gravity_at_height(planet_mass, planet_radius, altitude + delta_z_plus)
    minus_g = gravity_at_height(planet_mass, planet_radius, altitude - delta_z_minus)

    # total scaleheight
    h_total = scaleheight(temperature, central_g, mu)
    h_mass = scaleheight(temperature, central_g, masses[:, None])

    h_total_plus = np.zeros_like(h_total) << h_total.unit
    h_total_minus = np.zeros_like(h_total) << h_total.unit
    h_mass_plus = np.zeros_like(h_mass) << h_mass.unit
    h_mass_minus = np.zeros_like(h_mass) << h_mass.unit

    h_total_plus[:-1] = scaleheight(
        temperature[1:],
        plus_g[:-1],
        mu[1:],
    )

    h_total_minus[1:] = scaleheight(
        temperature[:-1],
        minus_g[1:],
        mu[:-1],
    )

    h_mass_plus[..., :-1] = scaleheight(
        temperature[1:],
        plus_g[:-1],
        masses[:, None],
    )

    h_mass_minus[..., 1:] = scaleheight(
        temperature[:-1],
        minus_g[1:],
        masses[:, None],
    )

    # ----- Diffusive flux -----

    # temperature terms
    temperature_factor = (temperature[1:] - temperature[:-1]) / (temperature[1:] + temperature[:-1])
    t_diffusion_plus = 2.0 * alpha * inv_dz_plus[:-1] * temperature_factor
    t_diffusion_minus = 2.0 * alpha * inv_dz_minus[1:] * temperature_factor

    #    diffusion_plus[:, :-1] = (
    #         2.0 / (hip[:, :-1] + hi[:, :-1]) - 2.0 / (hap[:-1] + ha[:-1])
    #     ) + 2.0 * alpha * inv_dz_p[:-1] * (T[1:] - T[:-1]) / (T[1:] + T[:-1])

    diffusion_plus = 2 / (h_mass_plus + h_mass) - 2 / (h_total_plus + h_total)
    diffusion_minus = 2 / (h_mass_minus + h_mass) - 2 / (h_total_minus + h_total)

    diffusion_plus[..., :-1] += t_diffusion_plus
    diffusion_minus[..., 1:] += t_diffusion_minus

    diffusion_plus[..., -1] = 1 / h_mass[..., -1] - 1 / h_total[-1]
    diffusion_minus[..., 0] = 1 / h_mass[..., 0] - 1 / h_total[0]

    # Upwind scheme

    return (
        diffusion_plus,
        diffusion_minus,
    )


def finite_difference_terms(
    altitude: u.Quantity,
    radius: float,
    inv_dz: u.Quantity,
    inv_dz_plus: u.Quantity,
    inv_dz_minus: u.Quantity,
) -> tuple[u.Quantity, u.Quantity]:
    r"""Compute finite difference terms.


    Computes the finite difference terms for the diffusion flux.
    The finite difference terms are given by:

    \begin{align}
       c^{+} &= \frac{(1 + \frac{0.5 \Delta z}{R + z})^2}{\Delta z}\\
       c^{-} &= -\frac{(1 - \frac{0.5 \Delta z}{R + z})^2}{\Delta z}
    \end{align}


    Args:
        altitude: Altitude in km.
        radius: Radius of the planet in km.
        inv_dz: The inverse delta z term in cm^-1.
        inv_dz_plus: The inverse delta z plus term in cm^-1.
        inv_dz_minus: The inverse delta z minus term in cm^-1.
    """
    fd_plus = (1 + 0.5 / ((radius + altitude) * inv_dz_plus)) ** 2 * inv_dz
    fd_minus = -((1 - 0.5 / ((radius + altitude) * inv_dz_minus)) ** 2) * inv_dz

    # Handle boundaries
    fd_minus[0] = -((1 - 0.5 / ((radius + altitude[0]) * inv_dz[0])) ** 2) * inv_dz[0]
    fd_plus[-1] = (1 + 0.5 / ((radius + altitude[-1]) * inv_dz[-1])) ** 2 * inv_dz[-1]

    return fd_plus, fd_minus


def general_plus_minus(
    array: u.Quantity,
) -> tuple[u.Quantity, u.Quantity]:
    r"""Compute the plus and minus terms.

    Computes general plus minus terms

    Generally defined as:
    $$
       a_{+} = \frac{1}{2} \left( a_{i+1} + a_{i} \right)
    $$
    $$
       a_{-} = \frac{1}{2} \left( a_{i} + a_{i-1} \right)
    $$
    Where $a$ is the array.


    Args:
        array: The array to compute the plus and minus terms.

    Returns:
        plus: The plus term.
        minus: The minus term.
    """

    sum_arr = array[..., :-1] + array[..., 1:]

    plus = np.zeros_like(array) << array.unit
    minus = np.zeros_like(array) << array.unit

    plus[..., :-1] = sum_arr
    minus[..., 1:] = sum_arr
    plus[..., -1] = sum_arr[..., -1]
    minus[..., 0] = sum_arr[..., 0]

    return 0.5 * plus, 0.5 * minus


def vmr_terms(
    vmr: FreckllArray, inv_dz_plus: u.Quantity, inv_dz_minus: u.Quantity
) -> tuple[FreckllArray, FreckllArray]:
    r"""Compute the VMR terms.

    this is the finite difference term for the VMR.

    $$
    \frac{dy}{dz}
    $$

    Args:
        vmr: The volume mixing ratio.
        inv_dz_plus: The inverse delta z plus term.
        inv_dz_minus: The inverse delta z minus term.

    """
    vmr_diff = np.diff(vmr, axis=-1)
    dy_plus = np.zeros(vmr.shape) << inv_dz_plus.unit
    dy_minus = np.zeros(vmr.shape) << inv_dz_minus.unit
    dy_plus[:, :-1] = (vmr_diff) * inv_dz_plus[:-1]
    dy_minus[:, 1:] = (vmr_diff) * inv_dz_minus[1:]

    return dy_plus, dy_minus


def diffusion_flux(
    vmr: FreckllArray,
    density: u.Quantity,
    planet_radius: u.Quantity,
    planet_mass: u.Quantity,
    altitude: u.Quantity,
    temperature: u.Quantity,
    mu: u.Quantity,
    masses: u.Quantity,
    molecular_diffusion: u.Quantity,
    kzz: u.Quantity,
) -> u.Quantity:
    r"""Compute the diffusion flux using finite difference.

    This is the term $\frac{d \phi}{dz}$ term in the full kinetic equation where $\phi$ is the diffusion flux:

    $$
    \phi_i = -(D_i + K_{zz})n_t \frac{dy_i}{dz} - D_in_t(\frac{1}{H_0} - \frac{1}{H_i}-\frac{\alpha_i}{T}\frac{d T}{dz})
    $$

    Where

    - $D_i$ is the diffusion coefficient for species $i$,
    - $K_{zz}$ is the eddy diffusion coefficient,
    - $n_t$ is the total number density,
    - $H_0$ is the scale height of the atmosphere,
    - $H_i$ is the scale height of species $i$,
    - $\alpha_i$ is the thermal diffusion coefficient for species $i$,
    - $T$ is the temperature,
    - $y_i$ is the volume mixing ratio of species $i$,


    Args:
        vmr: The volume mixing ratio.
        density: The density of the atmosphere.
        planet_radius: The radius of the planet.
        planet_mass: The mass of the planet.
        altitude: The altitude in km.
        temperature: The temperature in K.
        mu: The mean molecular weight in kg.
        masses: The molar masses in kg.
        molecular_diffusion: The molecular diffusion coefficient.
        kzz: The eddy diffusion coefficient.
    Returns:
        The diffusion flux. $\frac{d \phi}{dz}$



    """
    # Compute the delta z terms
    delta_z, delta_z_plus, delta_z_minus, inv_dz, inv_dz_plus, inv_dz_minus = deltaz_terms(altitude)

    # Compute the diffusive terms
    diffusion_plus, diffusion_minus = diffusive_terms(
        planet_radius,
        planet_mass,
        altitude,
        mu,
        temperature,
        masses,
        delta_z,
        delta_z_plus,
        delta_z_minus,
        inv_dz_plus,
        inv_dz_minus,
    )

    # Compute the finite difference terms
    fd_plus, fd_minus = finite_difference_terms(
        altitude,
        planet_radius,
        inv_dz,
        inv_dz_plus,
        inv_dz_minus,
    )

    # Compute the VMR terms
    dy_plus, dy_minus = vmr_terms(vmr, inv_dz_plus, inv_dz_minus)

    # Compute the general plus and minus terms
    dens_plus, dens_minus = general_plus_minus(density)
    mdiff_plus, mdiff_minus = general_plus_minus(molecular_diffusion)
    kzz_plus, kzz_minus = general_plus_minus(kzz)

    diff_flux = np.zeros(vmr.shape) << (1 / (u.cm**3 * u.s))

    diff_flux[:, 1:-1] += (
        dens_plus[1:-1]
        * (
            mdiff_plus[:, 1:-1] * ((vmr[:, 2:] + vmr[:, 1:-1]) * 0.5 * diffusion_plus[:, 1:-1] + dy_plus[:, 1:-1])
            + kzz_plus[1:-1] * dy_plus[:, 1:-1]
        )
        * fd_plus[1:-1]
        + dens_minus[1:-1]
        * (
            mdiff_minus[:, 1:-1] * ((vmr[:, 1:-1] + vmr[:, :-2]) * 0.5 * diffusion_minus[:, 1:-1] + dy_minus[:, 1:-1])
            + kzz_minus[1:-1] * dy_minus[:, 1:-1]
        )
        * fd_minus[1:-1]
    )
    diff_flux[:, 0] += (
        dens_plus[0]
        * (
            mdiff_plus[:, 0] * ((vmr[:, 1] + vmr[:, 0]) * 0.5 * diffusion_plus[:, 0] + dy_plus[:, 0])
            + kzz_plus[0] * dy_plus[:, 0]
        )
        * fd_plus[0]
    )
    diff_flux[:, -1] += (
        fd_minus[-1]
        * dens_minus[-1]
        * (
            mdiff_minus[:, -1] * ((vmr[:, -1] + vmr[:, -2]) * 0.5 * diffusion_minus[:, -1] + dy_minus[:, -1])
            + kzz_minus[-1] * dy_minus[:, -1]
        )
    )

    return diff_flux


def alpha_term(species: list[SpeciesFormula], vmr: FreckllArray) -> FreckllArray:
    r"""Compute the thermal diffusion coefficient.

    The alpha term is given by:

    $$
        \alpha = \frac{1}{\sum_i \frac{y_i}{\mu_i}}
    $$

    Where $y_i$ is the volume mixing ratio of species $i$ and $\mu_i$ is the mean molecular weight of species $i$.

    Args:
        species: The list of species.
        vmr: The volume mixing ratio.

    Returns:
        The alpha term.
    """
    alpha = np.full_like(vmr, 0.25)

    if "H" in species:
        index = species.index("H")
        alpha[index] = -0.1 * (1 - vmr[index])

    if "He" in species:
        index = species.index("He")
        alpha[index] = 0.145 * (1 - vmr[index])

    if "H2" in species:
        index = species.index("H2")
        alpha[index] = -0.38

    return alpha
