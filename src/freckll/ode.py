"""Module to construct the ODE system for the Freckll model."""

import numpy as np
from astropy import units as u
from scipy import sparse

from .distill import ksum
from .reactions.data import Reaction
from .species import SpeciesDict, SpeciesFormula
from .types import FreckllArray, FreckllArrayInt


def construct_reaction_terms(
    production_reactions: SpeciesDict[list[Reaction]],
    loss_reactions: SpeciesDict[list[Reaction]],
    species: list[SpeciesFormula],
    num_layers: int,
    k: int = 4,
) -> FreckllArray:
    """Construct all of the reaction terms."""

    reaction_terms = np.zeros((len(species), num_layers), dtype=np.float64)

    for spec_idx, spec in enumerate(species):
        production_spec = production_reactions.get(spec, [])
        loss_spec = loss_reactions.get(spec, [])
        productions = [p.dens_krate for p in production_spec]
        losses = [-ls.dens_krate for ls in loss_spec]

        all_reactions = productions + losses

        if not all_reactions:
            continue

        reaction_terms[spec_idx] = ksum(np.array(all_reactions), k=k)

    # reaction_terms = np.zeros(
    #     (len(reactions), num_species, num_layers), dtype=np.float64
    # )
    # for idx, r in enumerate(reactions):
    #     reaction_terms[idx, r.product_indices] = r.dens_krate
    #     reaction_terms[idx, r.reactants_indices] -= r.dens_krate

    return reaction_terms


def compute_index(species_id, layer_id, num_spec, num_layers):
    return species_id + (num_layers - layer_id - 1) * num_spec


def convert_fm_to_y(fm: FreckllArray) -> FreckllArray:
    num_species, num_layers = fm.shape
    species_idx = np.arange(0, num_species)
    layer_idx = np.arange(0, num_layers)
    X, Y = np.meshgrid(species_idx, layer_idx)

    y = np.zeros(shape=(num_species * num_layers))

    y[compute_index(X, Y, num_species, num_layers)] = fm[X, Y]
    return y


def convert_y_to_fm(y, num_species, num_layers):
    species_idx = np.arange(0, num_species)
    layer_idx = np.arange(0, num_layers)
    X, Y = np.meshgrid(species_idx, layer_idx)

    fm = np.empty(shape=(num_species, num_layers))

    fm[X, Y] = y[compute_index(X, Y, num_species, num_layers)]
    return fm


def construct_jacobian_reaction_terms(
    loss_reactions: SpeciesDict[list[Reaction]],
    species: list[SpeciesFormula],
    number_density: FreckllArray,
    k: int = 4,
) -> tuple[list[FreckllArrayInt], list[FreckllArrayInt], FreckllArray]:
    """Construct the Jacobian for the reaction terms.

    Args:
        loss_reactions: The loss reactions for the species.
        species: The list of species.
        number_density: The number density of the species.
        k: K-sum number (higher is more accurate)

    """
    from collections import defaultdict

    # Construct the reaction terms
    # df/dn dR/dn =

    rows = []
    cols = []
    data = []

    num_species = len(species)

    atmos_shape = number_density.shape

    num_layers = atmos_shape[1]

    layer_idx = np.arange(num_layers)

    for spec_idx, spec in enumerate(species):
        spec_density = number_density[spec_idx]
        all_reactions = loss_reactions.get(spec, [])
        if not all_reactions:
            continue

        chem_dict = defaultdict(list)
        for _react_idx, r in enumerate(all_reactions):
            for p in r.product_indices:
                chem_dict[p].append(r.dens_krate)
            for p in r.reactants_indices:
                chem_dict[p].append(-r.dens_krate)

        if not chem_dict:
            continue

        row_idx = compute_index(spec_idx, layer_idx, num_species, num_layers)
        for p, v in chem_dict.items():
            reaction_term = ksum(np.array(v), k=k) / spec_density
            col_idx = compute_index(p, layer_idx, num_species, num_layers)
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(reaction_term)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)


def construct_jacobian_vertical_terms(
    density: u.Quantity,
    planet_radius: float,
    planet_mass: float,
    altitude: u.Quantity,
    temperature: u.Quantity,
    mu: u.Quantity,
    masses: u.Quantity,
    molecular_diffusion: u.Quantity,
    kzz: u.Quantity,
):
    """Construct the Jacobian for the vertical terms."""
    from freckll import kinetics

    atmos_shape = masses.shape + mu.shape
    with np.errstate(all="ignore"):
        delta_z, delta_z_plus, delta_z_minus, inv_dz, inv_dz_plus, inv_dz_minus = kinetics.deltaz_terms(altitude)

        fd_plus, fd_minus = kinetics.finite_difference_terms(
            altitude,
            planet_radius,
            inv_dz,
            inv_dz_plus,
            inv_dz_minus,
        )

        diffusion_plus, diffusion_minus = kinetics.diffusive_terms(
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

        dens_plus, dens_minus = kinetics.general_plus_minus(density)
        mdiff_plus, mdiff_minus = kinetics.general_plus_minus(molecular_diffusion)
        kzz_plus, kzz_minus = kinetics.general_plus_minus(kzz)

        pd_same_p = dens_plus * (mdiff_plus * (0.5 * diffusion_plus - inv_dz_plus) - inv_dz_plus * kzz_plus) * fd_plus

        pd_same_m = (
            dens_minus * (mdiff_minus * (0.5 * diffusion_minus + inv_dz_minus) + inv_dz_minus * kzz_minus) * fd_minus
        )

        pd_same = pd_same_p + pd_same_m
        pd_p = dens_minus * (mdiff_minus * (0.5 * diffusion_minus - inv_dz_minus) - inv_dz_minus * kzz_minus) * fd_minus
        pd_m = dens_plus * (mdiff_plus * (0.5 * diffusion_plus + inv_dz_plus) + inv_dz_plus * kzz_plus) * fd_plus

        pd_same[:, 0] = (
            dens_plus[0]
            * (mdiff_plus[:, 0] * (0.5 * diffusion_plus[:, 0] - inv_dz[0]) - inv_dz[0] * kzz_plus[0])
            * fd_plus[0]
        )

        pd_m[:, 0] = (
            dens_plus[0]
            * (mdiff_plus[:, 0] * (0.5 * diffusion_plus[:, 0] + inv_dz[0]) + inv_dz[0] * kzz_plus[0])
            * fd_plus[0]
        )

        pd_same[:, -1] = (
            dens_minus[-1]
            * (mdiff_minus[:, -1] * (0.5 * diffusion_minus[:, -1] + inv_dz[-1]) + inv_dz[-1] * kzz_minus[-1])
            * fd_minus[-1]
        )

        pd_p[:, -1] = (
            dens_minus[-1]
            * (mdiff_minus[:, -1] * (0.5 * diffusion_minus[:, -1] - inv_dz[-1]) - inv_dz[-1] * kzz_minus[-1])
            * fd_minus[-1]
        )
        # pd_p[:,-1] =  pd_same[:, -1]
        # pd_p[:, 0] = 0.0
        # pd_m[:, 0] = pd_same[:, 0]
        # pd_m[:, -1] = 0.0

        # Now its time to construct the Jacobian
        pd_same /= density
        pd_p /= density
        pd_m /= density

    pd_same = pd_same.decompose().value
    pd_p = pd_p.decompose().value
    pd_m = pd_m.decompose().value

    num_species, num_layers = pd_same.shape

    rows = []
    columns = []

    data = []

    same_layer = np.arange(0, num_layers)
    plus_one = np.arange(1, num_layers)
    minus_one = np.arange(0, num_layers - 1)

    for x in range(num_species):
        spec_index = compute_index(x, same_layer, num_species, num_layers)
        rows.append(spec_index)
        columns.append(spec_index)
        data.append(pd_same[x])

        plus_index = compute_index(x, plus_one, num_species, num_layers)
        minus_index = compute_index(x, minus_one, num_species, num_layers)

        rows.append(spec_index[1:])
        columns.append(minus_index)
        data.append(pd_m[x, :-1])

        rows.append(spec_index[:-1])
        columns.append(plus_index)
        data.append(pd_p[x, 1:])

    return np.concatenate(rows), np.concatenate(columns), np.concatenate(data)


def construct_jacobian_vertical_terms_sparse(
    density: u.Quantity,
    planet_radius: u.Quantity,
    planet_mass: u.Quantity,
    altitude: u.Quantity,
    temperature: u.Quantity,
    mu: u.Quantity,
    masses: u.Quantity,
    molecular_diffusion: u.Quantity,
    kzz: u.Quantity,
):
    rows, columns, data = construct_jacobian_vertical_terms(
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

    neq = mu.size * masses.size

    return sparse.csc_matrix((data, (columns, rows)), shape=(neq, neq))


def construct_jacobian_reaction_terms_sparse(
    loss_reactions: SpeciesDict[list[Reaction]],
    species: list[SpeciesFormula],
    number_density: FreckllArray,
    k: int = 4,
) -> sparse.csc_matrix:
    rows, columns, data = construct_jacobian_reaction_terms(loss_reactions, species, number_density, k)
    neq = number_density.size
    return sparse.csc_matrix((data, (columns, rows)), shape=(neq, neq))


def construct_vertical_jacobian(
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

    This is the term:

    $$
    \frac{d \pi}{dz}
    $$
    """
    from scipy.sparse import csc_matrix

    from freckll.kinetics import deltaz_terms, diffusive_terms, finite_difference_terms, general_plus_minus

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

    # Compute the general plus and minus terms
    dens_plus, dens_minus = general_plus_minus(density)
    mdiff_plus, mdiff_minus = general_plus_minus(molecular_diffusion)
    kzz_plus, kzz_minus = general_plus_minus(kzz)
    pd_same_p = dens_plus * (mdiff_plus * (0.5 * diffusion_plus - inv_dz_plus) - inv_dz_plus * kzz_plus) * fd_plus

    pd_same_m = (
        dens_minus * (mdiff_minus * (0.5 * diffusion_minus + inv_dz_minus) + inv_dz_minus * kzz_minus) * fd_minus
    )

    pd_same = pd_same_p + pd_same_m
    pd_p = dens_minus * (mdiff_minus * (0.5 * diffusion_minus - inv_dz_minus) - inv_dz_minus * kzz_minus) * fd_minus
    pd_m = dens_plus * (mdiff_plus * (0.5 * diffusion_plus + inv_dz_plus) + inv_dz_plus * kzz_plus) * fd_plus

    pd_same[:, 0] = (
        dens_plus[0]
        * (mdiff_plus[:, 0] * (0.5 * diffusion_plus[:, 0] - inv_dz[0]) - inv_dz[0] * kzz_plus[0])
        * fd_plus[0]
    )

    pd_m[:, 0] = (
        dens_plus[0]
        * (mdiff_plus[:, 0] * (0.5 * diffusion_plus[:, 0] + inv_dz[0]) + inv_dz[0] * kzz_plus[0])
        * fd_plus[0]
    )

    pd_same[:, -1] = (
        dens_minus[-1]
        * (mdiff_minus[:, -1] * (0.5 * diffusion_minus[:, -1] + inv_dz[-1]) + inv_dz[-1] * kzz_minus[-1])
        * fd_minus[-1]
    )

    pd_p[:, -1] = (
        dens_minus[-1]
        * (mdiff_minus[:, -1] * (0.5 * diffusion_minus[:, -1] - inv_dz[-1]) - inv_dz[-1] * kzz_minus[-1])
        * fd_minus[-1]
    )
    # pd_p[:,-1] =  pd_same[:, -1]

    # pd_m[:,:-1]/=density[1:]
    pd_same = (pd_same / density).to(1 / u.s).value

    pd_p = (pd_p / density).to(1 / u.s).value

    pd_m = (pd_m / density).to(1 / u.s).value

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
