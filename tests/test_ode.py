import numpy as np
import pytest

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


def build_jacobian_levels(coeffs, density):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (
            inv_dz,
            inv_dz_m,
            inv_dz_p,
            cp,
            cm,
            c_plus,
            c_moins,
            dip,
            dim,
            dm,
            dp,
            dyp,
            dym,
            km,
            kp,
        ) = coeffs

        pd_same = (
            cp * (dp * (0.5 * dip - inv_dz_p) - inv_dz_p * kp) * c_plus
            + cm * (dm * (0.5 * dim + inv_dz_m) + inv_dz_m * km) * c_moins
        )
        pd_p = cm * (dm * (0.5 * dim - inv_dz_m) - inv_dz_m * km) * c_moins
        pd_m = cp * (dp * (0.5 * dip + inv_dz_p) + inv_dz_p * kp) * c_plus

        pd_same[:, 0] = (
            cp[0]
            * (dp[:, 0] * (0.5 * dip[:, 0] - inv_dz[0]) - inv_dz[0] * kp[0])
            * c_plus[0]
        )

        pd_m[:, 0] = (
            cp[0]
            * (dp[:, 0] * (0.5 * dip[:, 0] + inv_dz[0]) + inv_dz[0] * kp[0])
            * c_plus[0]
        )

        pd_same[:, -1] = (
            cm[-1]
            * (dm[:, -1] * (0.5 * dim[:, -1] + inv_dz[-1]) + inv_dz[-1] * km[-1])
            * c_moins[-1]
        )

        pd_p[:, -1] = (
            cm[-1]
            * (dm[:, -1] * (0.5 * dim[:, -1] - inv_dz[-1]) - inv_dz[-1] * km[-1])
            * c_moins[-1]
        )
        # pd_p[:,-1] =  pd_same[:, -1]
        pd_p[:, 0] = 0.0
        # pd_m[:, 0] = pd_same[:, 0]
        pd_m[:, -1] = 0.0

        # pd_m[:,:-1]/=density[1:]
    return pd_same / density, pd_p / density, pd_m / density


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
    go = gravity_at_height(planet_mass, Ro * 1000, 0.0)
    density = np.random.rand(NLAYERS)
    masses = np.random.rand(NMOLS)
    mu = np.random.rand(NLAYERS)
    T = np.linspace(1000, 2000, NLAYERS)
    diffusion = np.random.rand(NMOLS, NLAYERS)
    Kzz = np.random.rand(NLAYERS)
    return fm, altitude, Ro, go, density, masses, mu, T, diffusion, Kzz, planet_mass


def test_species_layer_idx():
    from freckll.ode import species_layer_index
    import numpy as np

    spec_idx = 1
    num_species = 10
    num_layers = 5

    result = species_layer_index(spec_idx, num_species, num_layers)
    array = np.zeros(10 * 5)

    array[result] = 1
    array = array.reshape(10, 5)

    np.testing.assert_array_equal(array[1], np.ones(5))
