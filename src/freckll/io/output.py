"""Output functions for freckll."""

import pathlib

import h5py
from astropy import units as u
from astropy.io.typing import PathLike

from freckll import __version__

from ..solver import Solution
from ..solver.solver import PlanetOutputData, StellarFluxData


def write_quantity(name: str, array: u.Quantity, group: h5py.Group) -> h5py.Dataset:
    """Write a quantity to a h5py dataset.

    Args:
        array: The quantity to write.
        dataset: The dataset to write to.

    """
    dataset = group.create_dataset(
        name,
        shape=array.shape,
        dtype=array.dtype,
        data=array.value,
    )
    dataset.attrs["unit"] = array.unit.to_string()


def write_planet(planet: PlanetOutputData, group: h5py.Group):
    """Write the planet data to a h5py group.

    Args:
        planet: The planet data to write.
        group: The group to write to.

    """
    write_quantity(
        "radius",
        planet["radius"],
        group,
    )
    write_quantity(
        "mass",
        planet["mass"],
        group,
    )
    if "distance" in planet:
        write_quantity(
            "distance",
            planet["distance"],
            group,
        )
    if "albedo" in planet:
        group.create_dataset(
            "albedo",
            data=planet.get("albedo", 0),
        )


def write_stellar_flux(star: StellarFluxData, group: h5py.Group):
    """Write the stellar flux data to a h5py group.

    Args:
        star: The stellar flux data to write.
        group: The group to write to.

    """

    write_quantity(
        "wavelength",
        star["wavelength"],
        group,
    )
    write_quantity(
        "incident_flux",
        star["incident_flux"],
        group,
    )
    write_quantity(
        "incident_angle",
        star["incident_angle"],
        group,
    )


def write_solution_h5py(
    solution: Solution,
    filename: PathLike,
    overwrite: bool = False,
) -> None:
    """Output the solution to a h5py file.

    Args:
        solution: The solution object to output.
        filename: The name of the file to output to.

    """
    import datetime

    import h5py

    file_path = pathlib.Path(filename)
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File {file_path} already exists.")

    # Add freckll version to the file

    with h5py.File(filename, "w") as f:
        f.attrs["freckll_version"] = __version__
        f.attrs["created"] = datetime.datetime.now().isoformat()

        # Write the solution to the file
        write_quantity(
            "pressure",
            solution["pressure"],
            f,
        )

        write_quantity(
            "temperature",
            solution["temperature"],
            f,
        )
        write_quantity(
            "kzz",
            solution["kzz"],
            f,
        )

        write_quantity(
            "density",
            solution["density"],
            f,
        )

        write_quantity(
            "masses",
            solution["masses"],
            f,
        )

        f.create_dataset(
            "inital_vmr",
            shape=solution["initial_vmr"].shape,
            dtype=solution["initial_vmr"].dtype,
            data=solution["initial_vmr"],
        )

        g = f.create_group("solution")

        g.create_dataset(
            "vmrs",
            shape=solution["vmr"].shape,
            dtype=solution["vmr"].dtype,
            data=solution["vmr"],
        )

        g.create_dataset(
            "times",
            shape=solution["times"].shape,
            dtype=solution["times"].dtype,
            data=solution["times"],
        )

        write_planet(
            solution["planet"],
            f.create_group("planet"),
        )
        if "stellar_flux" in solution:
            write_stellar_flux(
                solution["stellar_flux"],
                f.create_group("stellar_flux"),
            )

        species_group = f.create_group("species")

        species_input = [s.input_formula for s in solution["species"]]

        species_formula = [s.formula for s in solution["species"]]

        species_state = [s.state.value for s in solution["species"]]

        species_group["species_input"] = species_input
        species_group["species_formula"] = species_formula
        species_group["species_state"] = species_state


def read_h5py_quantity(
    group: h5py.Group,
    name: str,
) -> u.Quantity:
    """Read a quantity from a h5py dataset.

    Args:
        group: The group to read from.
        name: The name of the dataset to read.

    Returns:
        The quantity read from the dataset.
    """
    return u.Quantity(group[name][()], unit=group[name].attrs["unit"])


def read_h5py_planet(
    group: h5py.Group,
) -> PlanetOutputData:
    """Read the planet data from a h5py group.

    Args:
        group: The group to read from.

    Returns:
        The planet data read from the group.
    """
    data = {
        "radius": read_h5py_quantity(group, "radius"),
        "mass": read_h5py_quantity(group, "mass"),
    }

    if "distance" in group:
        data["distance"] = read_h5py_quantity(group, "distance")
    if "albedo" in group:
        data["albedo"] = group["albedo"][()]
    return data


def read_h5py_stellar_flux(
    group: h5py.Group,
) -> StellarFluxData:
    """Read the stellar flux data from a h5py group.

    Args:
        group: The group to read from.

    Returns:
        The stellar flux data read from the group.
    """
    data = {
        "wavelength": read_h5py_quantity(group, "wavelength"),
        "incident_flux": read_h5py_quantity(group, "incident_flux"),
        "incident_angle": read_h5py_quantity(group, "incident_angle"),
    }
    return data


def read_h5py_solution(
    filename: PathLike,
) -> Solution:
    from ..species import SpeciesFormula

    file_path = pathlib.Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with h5py.File(filename, "r") as f:
        solution = {
            "pressure": read_h5py_quantity(f, "pressure"),
            "temperature": read_h5py_quantity(f, "temperature"),
            "kzz": read_h5py_quantity(f, "kzz"),
            "initial_vmr": f["inital_vmr"][()],
            "vmr": f["solution"]["vmrs"][()],
            "times": f["solution"]["times"][()],
            "masses": read_h5py_quantity(f, "masses"),
            "density": read_h5py_quantity(f, "density"),
        }

        species_group = f["species"]
        species_input = species_group["species_input"].asstr()
        species_formula = species_group["species_formula"].asstr()
        species_state = species_group["species_state"].asstr()

        species = [
            SpeciesFormula(
                formula=form_in,
                true_formula=true_form,
                state=state,
            )
            for form_in, true_form, state in zip(species_input, species_formula, species_state)
        ]
        solution["species"] = species

        solution["planet"] = read_h5py_planet(f["planet"])
        if "stellar_flux" in f:
            solution["stellar_flux"] = read_h5py_stellar_flux(f["stellar_flux"])

    return solution
