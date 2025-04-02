import warnings

import numpy as np
from astropy import units as u

from .types import FreckllArray

# Do products


# Rewrite
def compute_coeffs(fm, altitude, Ro, go, density, masses, mu, T, diffusion, Kzz):
    import numpy as np

    k_boltz_si = 1.380662e-23
    with warnings.catch_warnings():
        G = 6.674e-11
        warnings.simplefilter("ignore")
        k_b = k_boltz_si * 1.0e4
        alpha = 0.0

        g_alt = go
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
        c_moins = -((1.0 - 1.0 / ((Ro + altitude) * 1.0e5 * inv_dz_m) / 2) ** 2) * inv_dz

        # Level = 0

        c_plus[0] = (1.0 + 1.0 / ((Ro + altitude[0]) * 1.0e5 * inv_dz[0]) / 2) ** 2 * inv_dz[0]
        c_moins[0] = -((1.0 - 1.0 / ((Ro + altitude[0]) * 1.0e5 * inv_dz[0]) / 2) ** 2) * inv_dz[0]

        # Level = max

        c_plus[-1] = (1.0 + 1.0 / ((Ro + altitude[-1]) * 1.0e5 * inv_dz[-1]) / 2) ** 2 * inv_dz[-1]
        c_moins[-1] = -((1.0 - 1.0 / ((Ro + altitude[-1]) * 1.0e5 * inv_dz[-1]) / 2) ** 2) * inv_dz[-1]

        cm, cp = np.zeros_like(T), np.zeros_like(T)

        cm[1:] = (density[1:] + density[:-1]) * 0.5
        cp[:-1] = (density[:-1] + density[1:]) * 0.5

        # Edge cases

        cp[0] = (density[1] + density[0]) * 0.5
        cm[-1] = (density[-1] + density[-2]) * 0.5

        ha = k_b * T / (g_alt * 100.0 * (Ro**2 / (Ro + altitude) ** 2) * mu * 1.0e-3)
        hi = k_b * T / (g_alt * 100.0 * (Ro**2 / (Ro + altitude) ** 2) * masses[:, None] * 1.0e-3)

        hap = np.zeros_like(ha)
        hip = np.zeros_like(hi)
        dip = np.zeros_like(hi)
        dp = np.zeros_like(hi)
        dyp = np.zeros_like(hi)
        kp = np.zeros_like(ha)

        hip[:, :-1] = (
            k_b
            * T[1:]
            / (g_alt * 100.0 * (Ro**2 / (Ro + (altitude[:-1] + delta_z_p[:-1])) ** 2) * masses[:, None] * 1.0e-3)
        )
        hap[:-1] = (
            k_b * T[1:] / (g_alt * 100.0 * (Ro**2 / (Ro + (altitude[:-1] + delta_z_p[:-1])) ** 2) * mu[1:] * 1.0e-3)
        )

        dip[:, :-1] = (2.0 / (hip[:, :-1] + hi[:, :-1]) - 2.0 / (hap[:-1] + ha[:-1])) + 2.0 * alpha * inv_dz_p[:-1] * (
            T[1:] - T[:-1]
        ) / (T[1:] + T[:-1])

        ham = np.zeros_like(ha)
        him = np.zeros_like(hi)
        dim = np.zeros_like(hi)
        dm = np.zeros_like(hi)
        dym = np.zeros_like(hi)
        km = np.zeros_like(ha)

        him[:, 1:] = (
            k_b
            * T[:-1]
            / (g_alt * 100.0 * (Ro**2 / (Ro + (altitude[1:] - delta_z_m[1:])) ** 2) * masses[:, None] * 1.0e-3)
        )
        ham[1:] = (
            k_b * T[:-1] / (g_alt * 100.0 * (Ro**2 / (Ro + (altitude[1:] - delta_z_m[1:])) ** 2) * mu[:-1] * 1.0e-3)
        )

        him[:, -1] = (
            k_b
            * T[-2]
            / (g_alt * 100.0 * (Ro**2 / (Ro + (altitude[-1] - delta_z[-1])) ** 2) * masses[:, None] * 1.0e-3)
        ).ravel()
        ham[-1] = k_b * T[-2] / (g_alt * 100.0 * (Ro**2 / (Ro + (altitude[-1] - delta_z[-1])) ** 2) * mu[-2] * 1.0e-3)

        dim[:, 1:] = (2.0 / (hi[:, 1:] + him[:, 1:]) - 2.0 / (ha[1:] + ham[1:])) + 2.0 * alpha * inv_dz_m[1:] * (
            T[1:] - T[:-1]
        ) / (T[1:] + T[:-1])

        dim[:, 0] = 1 / hi[:, 0] - 1 / ha[0]

        dm[:, 1:] = (diffusion[:, :-1] + diffusion[:, 1:]) * 0.5
        dp[:, :-1] = (diffusion[:, :-1] + diffusion[:, 1:]) * 0.5

        dym[:, 1:] = (fm[:, 1:] - fm[:, :-1]) * inv_dz_m[1:]
        dyp[:, :-1] = (fm[:, 1:] - fm[:, :-1]) * inv_dz_p[:-1]
        km[1:] = (Kzz[:-1] + Kzz[1:]) * 0.5
        kp[:-1] = (Kzz[:-1] + Kzz[1:]) * 0.5
        # print('HIM', him)
        # print('HAM', ham)
        # print('mu', mu)
    return (
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
    )


def build_jacobian_levels(coeffs, density):
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

        pd_same_p = cp * (dp * (0.5 * dip - inv_dz_p) - inv_dz_p * kp) * c_plus

        pd_same_m = cm * (dm * (0.5 * dim + inv_dz_m) + inv_dz_m * km) * c_moins

        pd_same = pd_same_p + pd_same_m
        pd_p = cm * (dm * (0.5 * dim - inv_dz_m) - inv_dz_m * km) * c_moins
        pd_m = cp * (dp * (0.5 * dip + inv_dz_p) + inv_dz_p * kp) * c_plus

        pd_same[:, 0] = cp[0] * (dp[:, 0] * (0.5 * dip[:, 0] - inv_dz[0]) - inv_dz[0] * kp[0]) * c_plus[0]

        pd_m[:, 0] = cp[0] * (dp[:, 0] * (0.5 * dip[:, 0] + inv_dz[0]) + inv_dz[0] * kp[0]) * c_plus[0]

        pd_same[:, -1] = cm[-1] * (dm[:, -1] * (0.5 * dim[:, -1] + inv_dz[-1]) + inv_dz[-1] * km[-1]) * c_moins[-1]

        pd_p[:, -1] = cm[-1] * (dm[:, -1] * (0.5 * dim[:, -1] - inv_dz[-1]) - inv_dz[-1] * km[-1]) * c_moins[-1]
        # pd_p[:,-1] =  pd_same[:, -1]
        pd_p[:, 0] = 0.0
        # pd_m[:, 0] = pd_same[:, 0]
        pd_m[:, -1] = 0.0

        # pd_m[:,:-1]/=density[1:]
    return pd_same / density, pd_p / density, pd_m / density


def compute_index(species_id, layer_id, num_spec, num_layers):
    return species_id + (num_layers - layer_id - 1) * num_spec
    # return layer_id + species_id*num_layers


def construct_level_matrix_sparse(pd_same, pd_p, pd_m):
    from scipy.sparse import csc_matrix

    num_species, num_layers = pd_same.shape
    same_layer = np.arange(0, num_layers)
    plus_one = np.arange(1, num_layers)
    minus_one = np.arange(0, num_layers - 1)

    neq = num_species * num_layers

    rows = []
    cols = []
    data = []
    for x in range(num_species):
        species_index = compute_index(x, same_layer, num_species, num_layers)
        rows.append(species_index)
        cols.append(species_index)
        data.append(pd_same[x, :])

        plus_index = compute_index(x, plus_one, num_species, num_layers)
        minus_index = compute_index(x, minus_one, num_species, num_layers)
        rows.append(species_index[1:])
        cols.append(minus_index)
        data.append(pd_m[x, :-1])

        rows.append(species_index[:-1])
        cols.append(plus_index)
        data.append(pd_p[x, 1:])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    return csc_matrix((data, (cols, rows)), shape=(neq, neq))


# [35, 5]


def convert_fm_to_y(fm):
    num_species, num_layers = fm.shape
    species_idx = np.arange(0, num_species)
    layer_idx = np.arange(0, num_layers)
    X, Y = np.meshgrid(species_idx, layer_idx)

    y = np.empty(shape=(num_species * num_layers))

    y[compute_index(X, Y, num_species, num_layers)] = fm[X, Y]
    return y


def convert_y_to_fm(y, num_species, num_layers):
    species_idx = np.arange(0, num_species)
    layer_idx = np.arange(0, num_layers)
    X, Y = np.meshgrid(species_idx, layer_idx)

    fm = np.empty(shape=(num_species, num_layers))

    fm[X, Y] = y[compute_index(X, Y, num_species, num_layers)]
    return fm


def dndt(
    fm,
    density,
    masses,
    altitude,
    Ro,
    go,
    T,
    diffusion,
    Kzz,
):
    mu = np.sum(fm * masses[:, None], axis=0)
    coeffs = compute_coeffs(fm, altitude, Ro, go, density, masses, mu, T, diffusion, Kzz)

    cp, cm, c_plus, c_moins, dip, dim, dm, dp, dyp, dym, km, kp = coeffs[3:]

    func = np.zeros_like(fm)

    func[:, 1:-1] += (
        cp[1:-1]
        * (dp[:, 1:-1] * ((fm[:, 2:] + fm[:, 1:-1]) * 0.5 * dip[:, 1:-1] + dyp[:, 1:-1]) + kp[1:-1] * dyp[:, 1:-1])
        * c_plus[1:-1]
        + cm[1:-1]
        * (dm[:, 1:-1] * ((fm[:, 1:-1] + fm[:, :-2]) * 0.5 * dim[:, 1:-1] + dym[:, 1:-1]) + km[1:-1] * dym[:, 1:-1])
        * c_moins[1:-1]
    )
    func[:, 0] += (
        cp[0] * (dp[:, 0] * ((fm[:, 1] + fm[:, 0]) * 0.5 * dip[:, 0] + dyp[:, 0]) + kp[0] * dyp[:, 0]) * c_plus[0]
    )
    func[:, -1] += (
        c_moins[-1]
        * cm[-1]
        * (dm[:, -1] * ((fm[:, -1] + fm[:, -2]) * 0.5 * dim[:, -1] + dym[:, -1]) + km[-1] * dym[:, -1])
    )
    # Handle layer
    final = func / density
    return final


def convert_to_banded(A):
    N = np.shape(A)[0]
    D = A[0].nnz
    ab = np.zeros((D, N))
    for i in np.arange(1, D):
        ab[i, :] = np.concatenate(
            (
                A.diagonal(k=i),
                np.zeros(
                    i,
                ),
            ),
            axis=None,
        )
    ab[0, :] = A.diagonal(k=0)
    return ab


def compute_jacobian_sparse(
    vmr: FreckllArray,
    altitude: u.Quantity,
    Ro: u.Quantity,
    planet_mass: u.Quantity,
    density: u.Quantity,
    masses: u.Quantity,
    mu: u.Quantity,
    T: u.Quantity,
    diffusion: u.Quantity,
    Kzz: u.Quantity,
):
    from .kinetics import gravity_at_height

    go = gravity_at_height(planet_mass, Ro, 0 << u.km).to(u.cm / u.s**2).value
    altitude = altitude.to(u.km).value
    Ro = Ro.to(u.km).value
    planet_mass = planet_mass.to(u.kg).value
    density = density.to(1 / u.cm**3).value
    masses = masses.to(u.g).value
    mu = mu.to(u.g).value
    T = T.to(u.K).value
    diffusion = diffusion.to(u.cm**2 / u.s).value
    Kzz = Kzz.to(u.cm**2 / u.s).value

    # Compute the coefficients
    coeffs = compute_coeffs(vmr, altitude, Ro, go, density, masses, mu, T, diffusion, Kzz)

    # Compute the Jacobian
    pd_same, pd_p, pd_m = build_jacobian_levels(coeffs, density)

    J = construct_level_matrix_sparse(pd_same, pd_p, pd_m)

    return J


def compute_dndt_vertical(
    vmr: FreckllArray,
    altitude: u.Quantity,
    Ro: u.Quantity,
    planet_mass: u.Quantity,
    density: u.Quantity,
    masses: u.Quantity,
    mu: u.Quantity,
    T: u.Quantity,
    diffusion: u.Quantity,
    Kzz: u.Quantity,
):
    from .kinetics import gravity_at_height

    go = gravity_at_height(planet_mass, Ro, 0 << u.km).to(u.cm / u.s**2).value
    altitude = altitude.to(u.km).value
    Ro = Ro.to(u.km).value
    planet_mass = planet_mass.to(u.kg).value
    density = density.to(1 / u.cm**3).value
    masses = masses.to(u.kg).value
    mu = mu.to(u.kg).value
    T = T.to(u.K).value
    diffusion = diffusion.to(u.cm**2 / u.s).value
    Kzz = Kzz.to(u.cm**2 / u.s).value

    return dndt(
        vmr,
        density,
        masses,
        altitude,
        Ro,
        go,
        T,
        diffusion,
        Kzz,
    )
