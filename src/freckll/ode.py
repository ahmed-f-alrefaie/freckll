"""Module to construct the ODE system for the Freckll model."""

import numpy as np
from astropy import units as u
from scipy import sparse

from .distill import ksum
from .network import ChemicalNetwork
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
        losses = [-l.dens_krate for l in loss_spec]

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
    from collections import defaultdict

    # Construct the reaction terms
    # df/dn dR/dn =

    rows = []
    cols = []
    data = []

    n_equation = number_density.size
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
        for react_idx, r in enumerate(all_reactions):
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

    atmos_size = np.prod(atmos_shape)

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

    vt_same_layer = np.zeros(atmos_shape, dtype=np.float64) << 1 / (u.cm**3 * u.s)

    # Same layer
    vt_same_layer_p = (
        dens_plus[..., :-1]
        * (mdiff_plus[..., :-1] * (0.5 * diffusion_plus[..., :-1] - inv_dz_plus) - inv_dz_plus * kzz_plus[..., :-1])
        * fd_plus[..., :-1]
    )

    vt_same_layer_m = (
        dens_minus[..., 1:]
        * (mdiff_minus[..., 1:] * (0.5 * diffusion_minus[..., 1:] + inv_dz_minus) + inv_dz_minus * kzz_minus[..., 1:])
        * fd_minus[..., 1:]
    )

    vt_same_layer[..., :-1] = vt_same_layer_p
    vt_same_layer[..., 1:] += vt_same_layer_m

    vt_same_layer[..., 0] = (
        dens_plus[..., 0]
        * (mdiff_plus[..., 0] * (0.5 * diffusion_plus[..., 0] - inv_dz[..., 0]) - inv_dz[..., 0] * kzz_plus[..., 0])
        * fd_plus[..., 0]
    )

    vt_same_layer[..., -1] = (
        dens_minus[..., -1]
        * (
            mdiff_minus[..., -1] * (0.5 * diffusion_minus[..., -1] - inv_dz_minus[..., -1])
            - inv_dz_minus[..., -1] * kzz_minus[..., -1]
        )
        * fd_minus[..., -1]
    )

    # +1 layer
    vt_plus_layer = dens_minus * (mdiff_minus * (0.5 * diffusion_minus - inv_dz) - inv_dz * kzz_minus) * fd_minus

    vt_plus_layer[..., -1] = (
        dens_minus[..., -1]
        * (
            mdiff_minus[..., -1] * (0.5 * diffusion_minus[..., -1] - inv_dz_minus[..., -1])
            - inv_dz_minus[..., -1] * kzz_minus[..., -1]
        )
        * fd_minus[..., -1]
    )

    # -1 layer
    vt_minus_layer = dens_plus * (mdiff_plus * (0.5 * diffusion_plus + inv_dz) + inv_dz * kzz_plus) * fd_plus

    vt_minus_layer[..., 0] = (
        dens_plus[..., 0]
        * (mdiff_plus[..., 0] * (0.5 * diffusion_plus[..., 0] + inv_dz[..., 0]) + inv_dz[..., 0] * kzz_plus[..., 0])
        * fd_plus[..., 0]
    )

    vt_minus_layer[:, -1] = 0.0

    # Now its time to construct the Jacobian
    vt_same_layer /= density
    vt_plus_layer /= density
    vt_minus_layer /= density
    vt_minus_layer = vt_minus_layer[..., :-1]
    vt_plus_layer = vt_plus_layer[..., 1:]

    vt_same_layer = vt_same_layer.decompose().value
    vt_plus_layer = vt_plus_layer.decompose().value
    vt_minus_layer = vt_minus_layer.decompose().value

    num_species, num_layers = vt_same_layer.shape

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
        data.append(vt_same_layer[x])

        plus_index = compute_index(x, plus_one, num_species, num_layers)
        minus_index = compute_index(x, minus_one, num_species, num_layers)

        rows.append(spec_index[1:])
        columns.append(minus_index)
        data.append(vt_minus_layer[x])

        rows.append(spec_index[:-1])
        columns.append(plus_index)
        data.append(vt_plus_layer[x])

    return np.concatenate(rows), np.concatenate(columns), np.concatenate(data)


def construct_jacobian_vertical_terms_sparse(
    density: FreckllArray,
    planet_radius: float,
    planet_mass: float,
    altitude: FreckllArray,
    temperature: FreckllArray,
    mu: FreckllArray,
    masses: FreckllArray,
    molecular_diffusion: FreckllArray,
    kzz: FreckllArray,
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


class KineticSolver:
    def __init__(self, network: ChemicalNetwork) -> None:
        """Initialize the solver."""
        self.network = network
