"""Test kinetic related functions."""

import numpy as np
import pytest
from astropy import units as u

def compute_coeffs(fm, altitude, Ro, go, density, masses, mu, T, diffusion, Kzz):
    import warnings

    import numpy as np

    from freckll.constants import K_BOLTZMANN

    k_boltz_si = K_BOLTZMANN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k_b = k_boltz_si * 1.0e4
        alpha = 0.0

        delta_z = np.zeros_like(T)
        delta_z_p = np.zeros_like(T)
        delta_z_m = np.zeros_like(T)

        delta_z_p[:-1] = altitude[1:] - altitude[:-1]
        delta_z_m[1:] = altitude[1:] - altitude[:-1]

        delta_z[:-1] = altitude[1:] - altitude[:-1]
        delta_z[-1] = altitude[-1] - altitude[-2]
        delta_z[0] = altitude[1] - altitude[0]

        inv_dz = 1.0 / ((0.5 * delta_z_p + 0.5 * delta_z_m) * 1e5)
        inv_dz[0] = 1.0 / (delta_z[0] * 1.0e5)
        inv_dz[-1] = 1.0 / (delta_z[-1] * 1.0e5)

        inv_dz_m = 1.0 / (delta_z_m * 1.0e5)
        inv_dz_p = 1.0 / (delta_z_p * 1.0e5)

        c_plus = (1.0 + 1.0 / ((Ro + altitude) * 1.0e5 * inv_dz_p) / 2) ** 2 * inv_dz
        c_moins = (
            -((1.0 - 1.0 / ((Ro + altitude) * 1.0e5 * inv_dz_m) / 2) ** 2) * inv_dz
        )

        # Level = 0

        c_plus[0] = (
            1.0 + 1.0 / ((Ro + altitude[0]) * 1.0e5 * inv_dz[0]) / 2
        ) ** 2 * inv_dz[0]
        c_moins[0] = (
            -((1.0 - 1.0 / ((Ro + altitude[0]) * 1.0e5 * inv_dz[0]) / 2) ** 2)
            * inv_dz[0]
        )

        # Level = max

        c_plus[-1] = (
            1.0 + 1.0 / ((Ro + altitude[-1]) * 1.0e5 * inv_dz[-1]) / 2
        ) ** 2 * inv_dz[-1]
        c_moins[-1] = (
            -((1.0 - 1.0 / ((Ro + altitude[-1]) * 1.0e5 * inv_dz[-1]) / 2) ** 2)
            * inv_dz[-1]
        )

        cm, cp = np.zeros_like(T), np.zeros_like(T)

        cm[1:] = (density[1:] + density[:-1]) * 0.5
        cp[:-1] = (density[:-1] + density[1:]) * 0.5

        # Edge cases

        cm[0] = (density[1] + density[0]) * 0.5
        cp[-1] = (density[-1] + density[-2]) * 0.5

        ha = k_b * T / (go * 100.0 * (Ro**2 / (Ro + altitude) ** 2) * mu * 1.0e-3)
        hi = (
            k_b
            * T
            / (go * 100.0 * (Ro**2 / (Ro + altitude) ** 2) * masses[:, None] * 1.0e-3)
        )

        hap = np.zeros_like(ha)
        hip = np.zeros_like(hi)
        dip = np.zeros_like(hi)
        dp = np.zeros_like(hi)
        dyp = np.zeros_like(hi)
        kp = np.zeros_like(ha)

        hip[:, :-1] = (
            k_b
            * T[1:]
            / (
                go
                * 100.0
                * (Ro**2 / (Ro + (altitude[:-1] + delta_z_p[:-1])) ** 2)
                * masses[:, None]
                * 1.0e-3
            )
        )
        hap[:-1] = (
            k_b
            * T[1:]
            / (
                go
                * 100.0
                * (Ro**2 / (Ro + (altitude[:-1] + delta_z_p[:-1])) ** 2)
                * mu[1:]
                * 1.0e-3
            )
        )

        dip[:, :-1] = (
            2.0 / (hip[:, :-1] + hi[:, :-1]) - 2.0 / (hap[:-1] + ha[:-1])
        ) + 2.0 * alpha * inv_dz_p[:-1] * (T[1:] - T[:-1]) / (T[1:] + T[:-1])

        ham = np.zeros_like(ha)
        him = np.zeros_like(hi)
        dim = np.zeros_like(hi)
        dm = np.zeros_like(hi)
        dym = np.zeros_like(hi)
        km = np.zeros_like(ha)

        him[:, 1:] = (
            k_b
            * T[:-1]
            / (
                go
                * 100.0
                * (Ro**2 / (Ro + (altitude[1:] - delta_z_m[1:])) ** 2)
                * masses[:, None]
                * 1.0e-3
            )
        )
        ham[1:] = (
            k_b
            * T[:-1]
            / (
                go
                * 100.0
                * (Ro**2 / (Ro + (altitude[1:] - delta_z_m[1:])) ** 2)
                * mu[:-1]
                * 1.0e-3
            )
        )

        him[:, -1] = (
            k_b
            * T[-2]
            / (
                go
                * 100.0
                * (Ro**2 / (Ro + (altitude[-1] - delta_z[-1])) ** 2)
                * masses[:, None]
                * 1.0e-3
            )
        ).ravel()
        ham[-1] = (
            k_b
            * T[-2]
            / (
                go
                * 100.0
                * (Ro**2 / (Ro + (altitude[-1] - delta_z[-1])) ** 2)
                * mu[-2]
                * 1.0e-3
            )
        )

        dim[:, 1:] = (
            2.0 / (hi[:, 1:] + him[:, 1:]) - 2.0 / (ha[1:] + ham[1:])
        ) + 2.0 * alpha * inv_dz_m[1:] * (T[1:] - T[:-1]) / (T[1:] + T[:-1])

        dim[:, 0] = 1 / hi[:, 0] - 1 / ha[0]

        dm[:, 1:] = (diffusion[:, :-1] + diffusion[:, 1:]) * 0.5
        dp[:, :-1] = (diffusion[:, :-1] + diffusion[:, 1:]) * 0.5
        dm[:, 0] = (diffusion[:, 0] + diffusion[:, 1]) * 0.5
        dp[:, -1] = (diffusion[:, -1] + diffusion[:, -2]) * 0.5

        dym[:, 1:] = (fm[:, 1:] - fm[:, :-1]) * inv_dz_m[1:]
        dyp[:, :-1] = (fm[:, 1:] - fm[:, :-1]) * inv_dz_p[:-1]
        km[1:] = (Kzz[:-1] + Kzz[1:]) * 0.5
        kp[:-1] = (Kzz[:-1] + Kzz[1:]) * 0.5

        kp[-1] = km[-1]
        km[0] = kp[0]
        # print('HIM', him)
        # print('HAM', ham)
        # print('mu', mu)
    return (
        delta_z_p,
        delta_z_m,
        inv_dz,
        inv_dz_p,
        inv_dz_m,
        cp,
        cm,
        c_plus,
        c_moins,
        dip,
        dim,
        dp,
        dm,
        dyp,
        dym,
        kp,
        km,
    )


@pytest.fixture
def coeff_inputs():
    from freckll.kinetics import gravity_at_height

    NMOLS = 20
    NLAYERS = 10
    fm = np.random.rand(NMOLS, NLAYERS)
    altitude = np.linspace(1, 1000, NLAYERS)
    altitude += np.random.rand(NLAYERS)
    Ro = 6371.0
    planet_mass = 5.972e24
    go = gravity_at_height(planet_mass << u.kg, Ro << u.km, 0.0 << u.km).to(u.cm/u.s**2).value
    density = np.random.rand(NLAYERS)
    masses = np.random.rand(NMOLS)
    mu = np.random.rand(NLAYERS)
    T = np.linspace(1000, 2000, NLAYERS)
    diffusion = np.random.rand(NMOLS, NLAYERS)
    Kzz = np.random.rand(NLAYERS)
    return fm, altitude, Ro, go, density, masses, mu, T, diffusion, Kzz, planet_mass


def test_deltaz_pychegp(coeff_inputs):
    """Test to make sure the delta z calculation is produces the same as pychegp."""
    import numpy as np

    from freckll.kinetics import deltaz_terms

    altitude = coeff_inputs[1]
    dz, dz_p, dz_m, inv_dz, inv_dz_p, indz_m = deltaz_terms(altitude << u.km)

    chegp = compute_coeffs(*coeff_inputs[:-1])[:5]

    assert inv_dz_p.shape == indz_m.shape

    np.testing.assert_allclose(dz_p[:-1].value, chegp[0][:-1])
    np.testing.assert_allclose(dz_m[1:].value, chegp[1][1:])
    np.testing.assert_allclose(inv_dz.to(1/u.cm).value, chegp[2])
    np.testing.assert_allclose(inv_dz_p.to(1/u.cm).value, chegp[3][:-1])
    np.testing.assert_allclose(indz_m.to(1/u.cm).value, chegp[4][1:])


def test_fd_terms(coeff_inputs):
    """Test to make sure the fd terms are the same
    as the chegp implementation."""
    from freckll.kinetics import deltaz_terms, finite_difference_terms

    altitude = coeff_inputs[1]
    planet_radius = coeff_inputs[2]
    dz, dz_p, dz_m, inv_dz, inv_dz_p, indz_m = deltaz_terms(altitude << u.km)

    fd_plus, fd_minus = finite_difference_terms(
        altitude << u.km,
        planet_radius << u.km,
        inv_dz,
        inv_dz_p,
        indz_m,
    )

    chegp = compute_coeffs(*coeff_inputs[:-1])

    np.testing.assert_allclose(fd_plus.to(1/u.mm).value, chegp[7])
    np.testing.assert_allclose(fd_minus, chegp[8])


def test_diffusive_terms(coeff_inputs):
    """Test to make sure the diffusion terms are the same
    as the chegp implementation."""
    from freckll.kinetics import deltaz_terms, diffusive_terms
    from freckll.constants import AMU

    # Test dip and dim
    chegp = compute_coeffs(*coeff_inputs[:-1])

    dip = chegp[9]
    dim = chegp[10]

    planet_radius = coeff_inputs[2]
    planet_mass = coeff_inputs[-1]

    altitude = coeff_inputs[1]
    mu = coeff_inputs[6]
    T = coeff_inputs[7]
    masses = coeff_inputs[5]

    delta_z, delta_z_p, delta_z_m, inv_dz, inv_dz_p, inv_dz_m = deltaz_terms(altitude)

    diffusion_plus, diffusion_minus = diffusive_terms(
        planet_radius,
        planet_mass,
        altitude,
        mu,
        T,
        masses,
        delta_z,
        delta_z_p,
        delta_z_m,
        inv_dz_p,
        inv_dz_m,
    )

    np.testing.assert_allclose(diffusion_plus[..., :-1], dip[..., :-1])
    np.testing.assert_allclose(diffusion_minus[..., 1:], dim[..., 1:])


def test_density_mdiffuse_kzz_terms(coeff_inputs):
    from freckll.kinetics import general_plus_minus

    chegp = compute_coeffs(*coeff_inputs[:-1])

    density = coeff_inputs[4]
    molecular_diffusion = coeff_inputs[8]
    Kzz = coeff_inputs[9]

    dens_plus, dens_minus = general_plus_minus(density)
    mdiff_plus, mdiff_minus = general_plus_minus(molecular_diffusion)
    kzz_plus, kzz_minus = general_plus_minus(Kzz)

    np.testing.assert_allclose(dens_plus, chegp[5])
    np.testing.assert_allclose(dens_minus, chegp[6])
    np.testing.assert_allclose(mdiff_plus, chegp[11])
    np.testing.assert_allclose(mdiff_minus, chegp[12])
    np.testing.assert_allclose(kzz_plus, chegp[15])
    np.testing.assert_allclose(kzz_minus, chegp[16])


def test_vmr_terms(coeff_inputs):
    from freckll.kinetics import deltaz_terms, vmr_terms

    chegp = compute_coeffs(*coeff_inputs[:-1])

    dyp = chegp[13]
    dym = chegp[14]

    altitude = coeff_inputs[1]
    fm = coeff_inputs[0]

    _, _, _, _, inv_dz_p, inv_dz_m = deltaz_terms(altitude)

    vmr_plus, vmr_minus = vmr_terms(fm, inv_dz_p, inv_dz_m)

    np.testing.assert_allclose(vmr_plus, dyp[..., :-1])
    np.testing.assert_allclose(vmr_minus, dym[..., 1:])


def test_diffusive_flux(coeff_inputs):
    from freckll.kinetics import diffusion_flux
    from freckll.constants import K_BOLTZMANN, AMU

    chegp = compute_coeffs(*coeff_inputs[:-1])
    cp, cm, c_plus, c_moins, dip, dim, dp, dm, dyp, dym, kp, km = chegp[5:]

    fm = coeff_inputs[0]

    expected = np.zeros_like(coeff_inputs[0])

    expected[:, 1:-1] += (
        cp[1:-1]
        * (
            dp[:, 1:-1]
            * ((fm[:, 2:] + fm[:, 1:-1]) * 0.5 * dip[:, 1:-1] + dyp[:, 1:-1])
            + kp[1:-1] * dyp[:, 1:-1]
        )
        * c_plus[1:-1]
        + cm[1:-1]
        * (
            dm[:, 1:-1]
            * ((fm[:, 1:-1] + fm[:, :-2]) * 0.5 * dim[:, 1:-1] + dym[:, 1:-1])
            + km[1:-1] * dym[:, 1:-1]
        )
        * c_moins[1:-1]
    )
    expected[:, 0] += (
        cp[0]
        * (
            dp[:, 0] * ((fm[:, 1] + fm[:, 0]) * 0.5 * dip[:, 0] + dyp[:, 0])
            + kp[0] * dyp[:, 0]
        )
        * c_plus[0]
    )
    expected[:, -1] += (
        c_moins[-1]
        * cm[-1]
        * (
            dm[:, -1] * ((fm[:, -1] + fm[:, -2]) * 0.5 * dim[:, -1] + dym[:, -1])
            + km[-1] * dym[:, -1]
        )
    )

    # vmr: FreckllArray,
    # density: FreckllArray,
    # planet_radius: FreckllArray,
    # planet_mass: FreckllArray,
    # altitude: FreckllArray,
    # temperature: FreckllArray,
    # mu: FreckllArray,
    # masses: FreckllArray,
    # molecular_diffusion: FreckllArray,
    # kzz: FreckllArray,

    vmr = coeff_inputs[0]
    density = coeff_inputs[4]
    planet_radius = coeff_inputs[2]
    planet_mass = coeff_inputs[-1]
    altitude = coeff_inputs[1]
    temperature = coeff_inputs[7]
    mu = coeff_inputs[6]
    masses = coeff_inputs[5]
    molecular_diffusion = coeff_inputs[8]
    kzz = coeff_inputs[9]

    diff_flux = diffusion_flux(
        vmr,
        density,
        planet_radius,
        planet_mass,
        altitude,
        temperature,
        mu,
        masses,
        molecular_diffusion,
        kzz,
    )

    np.testing.assert_allclose(diff_flux, expected)
