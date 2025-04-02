"""Handles computing the kinetics of a reaction."""

import numpy as np
from astropy import constants as const
from astropy import units as u

from .types import FreckllArray


def gravity_at_height(mass: u.Quantity, radius: u.Quantity, altitude: u.Quantity) -> u.Quantity:
    r"""Compute the gravity at a given altitude.

    The gravity at a given altitude is given by:

    ..math::
        g = \frac{Gm}{r^2}

    Where $G$ is the gravitational constant, $m$ is the mass of the planet,
    and $r$ is the radius of the planet.

    Args:
        mass: The mass of the planet.
        radius: The radius of the planet.
        altitude: The altitude at which to compute the gravity.

    """
    return const.G * mass / (radius + altitude) ** 2


def air_density(temperature: u.Quantity, pressure: u.Quantity) -> FreckllArray:
    r"""Compute the density of the atmosphere.

    The *air* density of the atmosphere is given by:

    ..math::
        \rho = \frac{P}{kT}

    Where $P$ is the pressure, $k$ is the Boltzmann constant, and $T$ is the temperature.

    """
    return pressure / (const.k_B * temperature)


def scaleheight(temperature: u.Quantity, gravity: u.Quantity, mass: u.Quantity) -> u.Quantity:
    r"""Compute the scale height of the atmosphere.

    The scale height is given by:

    ..math::
        H = \frac{kT}{mg}

    Where $k$ is the Boltzmann constant, $T$ is the temperature,
    $m$ is the molar mass, and $g$ is the gravity.

    Args:
        temperature: The temperature.
        gravity: The gravity at a given altitude.
        mass: Mass .

    """
    return const.k_B * temperature / (mass * gravity)


def solve_altitude_profile(
    temperature: u.Quantity, mu: u.Quantity, pressures: u.Quantity, planet_mass: u.Quantity, planet_radius: u.Quantity
):
    r"""Solve altitude corresponding to given pressure levels.

    Solves the hydrostatic equilibrium equation to compute the altitude corresponding to the given pressure levels.

    $$
    \frac{dz}{dP} = -\frac{1}{\rho g}
    $$


    Args:
        temperature: Temperature profile as a function of pressure.
        mu_profile: Mean molecular weight profile as a function of pressure.
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
    # Generate altitude at the original pressure points (interpolate if necessary)
    # Reverse pressures to increasing order for interpolation
    P_eval = np.sort(pressures)
    z_eval = sol.sol(P_eval)[0]

    # Ensure altitudes are in the same order as input pressures
    return z_eval[np.argsort(sort_idx)][::-1] << u.m


def deltaz_terms(
    altitude: u.Quantity,
) -> tuple[FreckllArray, FreckllArray, FreckllArray, FreckllArray, FreckllArray, FreckllArray]:
    """Compute the delta z terms.

    Returns everything in $cm$.

    Args:
        altitude: The altitude in km

    Returns:
        delta_z_plus: The delta z plus term.
        delta_z_minus: The delta z minus term.
        inv_dz: The inverse delta z term.
        inv_dz_minus: The inverse delta z minus term.
        inv_dz_plus: The inverse delta z plus term.


    """
    # Compute delta z terms

    altitude_cm = altitude
    delta_z = np.append(np.diff(altitude_cm), altitude_cm[-1] - altitude_cm[-2])

    delta_z_plus = np.diff(altitude_cm, append=0.0 << altitude_cm.unit)
    delta_z_minus = np.diff(altitude_cm, prepend=0.0 << altitude_cm.unit)

    inv_dz = 2 / (delta_z_plus + delta_z_minus)
    inv_dz[0] = 1 / delta_z[0]
    inv_dz[-1] = 1 / delta_z[-1]

    inv_dz_minus = 1 / delta_z_minus[1:]
    inv_dz_plus = 1 / delta_z_plus[:-1]

    return delta_z, delta_z_plus, delta_z_minus, inv_dz, inv_dz_plus, inv_dz_minus


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
        delta_z_plus: The delta z plus term.
        delta_z_minus: The delta z minus term.
        inv_dz_plus: The inverse delta z plus term.
        inv_dz_minus: The inverse delta z minus term.
        alpha: The alpha parameter to include temperature term.

    """

    # cm/m2
    central_g: FreckllArray = gravity_at_height(planet_mass, planet_radius, altitude)
    plus_g: FreckllArray = gravity_at_height(planet_mass, planet_radius, altitude + delta_z_plus)
    minus_g: FreckllArray = gravity_at_height(planet_mass, planet_radius, altitude - delta_z_minus)

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
    t_diffusion_plus = 2.0 * alpha * inv_dz_plus * temperature_factor
    t_diffusion_minus = 2.0 * alpha * inv_dz_minus * temperature_factor

    #    dip[:, :-1] = (
    #         2.0 / (hip[:, :-1] + hi[:, :-1]) - 2.0 / (hap[:-1] + ha[:-1])
    #     ) + 2.0 * alpha * inv_dz_p[:-1] * (T[1:] - T[:-1]) / (T[1:] + T[:-1])

    diffusion_plus = 2 / (h_mass_plus + h_mass) - 2 / (h_total_plus + h_total)
    diffusion_minus = 2 / (h_mass_minus + h_mass) - 2 / (h_total_minus + h_total)

    diffusion_plus[..., :-1] += t_diffusion_plus
    diffusion_minus[..., 1:] += t_diffusion_minus

    diffusion_plus[..., -1] = 1 / h_mass[..., -1] - 1 / h_total[-1]
    diffusion_minus[..., 0] = 1 / h_mass[..., 0] - 1 / h_total[0]

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
    """Compute finite difference terms.

    Args:
        altitude: Altitude in km.
        radius: Radius of the planet in km.
        delta_z_plus: The delta z plus term in cm.
        delta_z_minus: The delta z minus term in cm.
        inv_dz: The inverse delta z term in cm^-1.
        inv_dz_plus: The inverse delta z plus term in cm^-1.
        inv_dz_minus: The inverse delta z minus term in cm^-1.
    """
    altitude_cm = altitude
    radius_cm = radius
    fd_plus = np.zeros_like(inv_dz) << inv_dz.unit
    fd_minus = np.zeros_like(inv_dz) << inv_dz.unit

    fd_plus[:-1] = (1 + 0.5 / ((radius_cm + altitude_cm[:-1]) * inv_dz_plus)) ** 2 * inv_dz[:-1]

    fd_minus[1:] = -((1 - 0.5 / ((radius_cm + altitude_cm[1:]) * inv_dz_minus)) ** 2) * inv_dz[1:]
    # Handle boundaries
    fd_minus[0] = -((1 - 0.5 / ((radius_cm + altitude_cm[0]) * inv_dz[0])) ** 2) * inv_dz[0]
    fd_plus[-1] = (1 + 0.5 / ((radius_cm + altitude_cm[-1]) * inv_dz[-1])) ** 2 * inv_dz[-1]

    return fd_plus, fd_minus


def general_plus_minus(
    array: u.Quantity,
) -> tuple[u.Quantity, u.Quantity]:
    """Compute the plus and minus terms.

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

    dy_plus = (vmr_diff) * inv_dz_plus
    dy_minus = (vmr_diff) * inv_dz_minus

    return dy_plus, dy_minus


def diffusion_flux(
    vmr: FreckllArray,
    density: FreckllArray,
    planet_radius: float,
    planet_mass: float,
    altitude: FreckllArray,
    temperature: FreckllArray,
    mu: FreckllArray,
    masses: FreckllArray,
    molecular_diffusion: FreckllArray,
    kzz: FreckllArray,
) -> FreckllArray:
    r"""Compute the diffusion flux using finite difference.

    This is the term:

    $$
    \frac{d \pi}{dz}
    $$
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

    plus_term = dens_plus[1:-1] * (
        mdiff_plus[..., 1:-1] * ((vmr[..., 2:] + vmr[:, 1:-1]) * 0.5 * diffusion_plus[..., 1:-1] + dy_plus[..., 1:])
        + kzz_plus[..., 1:-1] * dy_plus[..., 1:]
    )

    minus_term = dens_minus[1:-1] * (
        mdiff_minus[..., 1:-1]
        * ((vmr[..., 1:-1] + vmr[..., :-2]) * 0.5 * diffusion_minus[..., 1:-1] + dy_minus[..., :-1])
        + kzz_minus[..., 1:-1] * dy_minus[..., :-1]
    )

    diffusive_flux = np.zeros(vmr.shape) << (plus_term.unit * fd_plus.unit)

    diffusive_flux[..., 1:-1] = plus_term * fd_plus[..., 1:-1] + minus_term * fd_minus[..., 1:-1]

    # Handle boundaries
    diffusive_flux[..., 0] = (
        dens_plus[..., 0]
        * (
            mdiff_plus[..., 0] * ((vmr[..., 1] + vmr[..., 0]) * 0.5 * diffusion_plus[..., 0] + dy_plus[..., 0])
            + kzz_plus[..., 0] * dy_plus[..., 0]
        )
        * fd_plus[..., 0]
    )
    diffusive_flux[..., -1] = (
        dens_minus[..., -1]
        * (
            mdiff_minus[..., -1] * ((vmr[..., -1] + vmr[..., -2]) * 0.5 * diffusion_minus[..., -1] + dy_minus[..., -1])
            + kzz_minus[..., -1] * dy_minus[..., -1]
        )
        * fd_minus[..., -1]
    )

    return diffusive_flux
