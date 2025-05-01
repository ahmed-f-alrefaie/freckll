import typing as t
from dataclasses import dataclass
from typing import Callable

from astropy import units as u
from astropy.io.typing import PathLike

from ..log import setup_log
from ..network import ChemicalNetwork, PhotoChemistry
from ..reactions.photo import StarSpectra
from ..solver import Solution, Solver
from ..species import SpeciesFormula
from ..types import FreckllArray

T = t.TypeVar("T")

_log = setup_log(__name__)


def _dispatch(
    data: dict | str,
    format_key: str = "format",
    dispatcher_map: t.Optional[dict[str, Callable[..., T]]] = None,
    **kwargs,
) -> T:
    """Dispatch a function based on the format key in the data dictionary.

    Used to load in data from an input file. The function will look for the format key in the
    data dictionary and call the corresponding function from the dispatcher map. If the format

    Args:
        data: The data to dispatch.
        format_key: The key to look for in the data dictionary.
        dispatcher_map: A map of format keys to functions.
        **kwargs: Additional arguments to pass to the function.

    """

    dispatcher_map = dispatcher_map or {}
    if isinstance(data, str):
        data = {"format": data}

    data = data.copy()
    if format_key not in data:
        raise ValueError(f"Missing format key '{format_key}' in data.")  # noqa: TRY003
    format_value = data.pop(format_key)
    if format_value not in dispatcher_map:
        raise ValueError(f"Unknown format '{format_value}' in data.")  # noqa: TRY003

    kwargs = {**kwargs, **data}

    return dispatcher_map[format_value](**kwargs)


@dataclass
class PlanetData:
    radius: u.Quantity
    mass: u.Quantity
    distance: u.Quantity = None
    albedo: int = 0


@dataclass
class StarData:
    """Schema for the star data."""

    spectra: StarSpectra
    incident_angle: u.Quantity = 45.0 * u.deg


@dataclass
class AtmosphereData:
    """Schema for the atmosphere data."""

    pressure: u.Quantity
    temperature: u.Quantity
    kzz: u.Quantity


def dispatch_planet(
    planet_data: dict | str,
) -> PlanetData:
    """Convert input data to a PlanetData object."""
    return PlanetData(**planet_data)


def dispatch_star(
    star_data: dict | str,
) -> StarData:
    """Convert input data to a StarData object."""
    from .loader import Stars, default_stellar_spectra_loader, rescale_stellar_spectra, star_spectra_loader

    _dispatcher_map = {k: lambda x=k: default_stellar_spectra_loader(x) for k in t.get_args(Stars)}
    _dispatcher_map["from-file"] = star_spectra_loader
    _dispatcher_map["rescale"] = rescale_stellar_spectra

    return StarData(
        spectra=_dispatch(star_data["spectrum"], dispatcher_map=_dispatcher_map),
        incident_angle=star_data.get("incident_angle", 45.0 * u.deg),
    )


def dispatch_network(network_data: dict | str) -> ChemicalNetwork:
    """Convert input data to a NetworkData object."""
    from ..venot import VenotChemicalNetwork
    from .loader import Networks, default_network_loader

    _dispatcher_map = {k: lambda x=k: default_network_loader(x) for k in t.get_args(Networks)}
    _dispatcher_map["venot"] = VenotChemicalNetwork

    return _dispatch(network_data, dispatcher_map=_dispatcher_map)


def dispatch_photochemistry(
    photochemistry_data: dict | str | None, species_list: list[SpeciesFormula]
) -> PhotoChemistry | None:
    from ..venot import VenotPhotoChemistry
    from .loader import default_photonetwork_loader

    if photochemistry_data is None:
        return None

    _dispatcher_map = {
        "venot": VenotPhotoChemistry,
        "venot-methanol-2023-photo": default_photonetwork_loader,
    }

    return _dispatch(photochemistry_data, dispatcher_map=_dispatcher_map, species_list=species_list)


def dispatch_thermochemistry(
    thermochemistry_data: dict | str, species: list[SpeciesFormula], temperature: u.Quantity, pressure: u.Quantity
) -> FreckllArray:
    """Convert input data to a ThermochemistryData object."""
    from .loader import ace_equil_chemistry_loader

    _dispatcher_map = {"ace": ace_equil_chemistry_loader}
    return _dispatch(
        thermochemistry_data,
        dispatcher_map=_dispatcher_map,
        species=species,
        temperature=temperature,
        pressure=pressure,
    )


def dispatch_atmosphere(atmosphere_data: dict | str) -> AtmosphereData:
    """Convert input data to a ThermochemistryData object."""
    import numpy as np
    from scipy.interpolate import interp1d

    from .loader import kzz_profile_loader, tp_profile_loader

    _tp_dispatcher_map = {
        "from-file": tp_profile_loader,
    }
    _kzz_dispatcher_map = {
        "from-file": kzz_profile_loader,
    }

    pressure, temperature = _dispatch(
        atmosphere_data["tp_profile"],
        dispatcher_map=_tp_dispatcher_map,
    )
    kzz_data = atmosphere_data["kzz"]
    if isinstance(kzz_data, u.Quantity):
        kzz = kzz_data * np.ones(pressure.shape)
        kzz_pressure = pressure
    else:
        kzz_pressure, kzz = _dispatch(
            kzz_data,
            dispatcher_map=_kzz_dispatcher_map,
        )

    # Interpolate kzz to the pressure grid
    if atmosphere_data.get("interpolate_kzz", True):
        kzz_pressure = kzz_pressure.to(pressure.unit)
        kzz_f = interp1d(np.log10(kzz_pressure.value), kzz.value)
        kzz = kzz_f(np.log10(pressure.value)) << kzz.unit

    return AtmosphereData(
        pressure=pressure,
        temperature=temperature,
        kzz=kzz,
    )


def dispatch_solver(
    solver_data: dict | str, network: ChemicalNetwork, photochem: t.Optional[PhotoChemistry] = None
) -> tuple[type[Solver], dict]:
    """Convert input data to a SolverData object."""
    from ..solver import BDF, LSODA, Rosenbrock, Vode
    from ..solver.transform import Log10Transform, LogitTransform, LogTransform, UnityTransform

    _map = {
        "rosenbrock": Rosenbrock,
        "vode": Vode,
        "lsoda": LSODA,
        "bdf": BDF,
    }

    choice = solver_data.pop("method", "rosenbrock")

    if choice not in _map:
        raise ValueError(f"Unknown solver '{choice}' in data.")  # noqa: TRY003

    _transform_map = {
        "log10": Log10Transform(),
        "logit": LogitTransform(),
        "log": LogTransform(),
        "unity": UnityTransform(),
    }

    transform = solver_data.pop("transform", "unity")

    return _map[choice](network, photochem), {
        **solver_data,
        "transform": _transform_map[transform],
    }


def dispatch_input(
    input_data: dict | str,
) -> tuple[Solver, dict]:
    """Convert input data to a FreckllInput object."""
    planet_data = dispatch_planet(input_data["planet"])
    star_data = None
    if "star" in input_data:
        star_data = dispatch_star(input_data["star"])

    network_data = dispatch_network(input_data["network"])
    photochem_data = None
    if star_data is not None and "photochemistry" in input_data:
        photochem_data = dispatch_photochemistry(input_data.get("photochemistry"), species_list=network_data.species)
        photochem_data.set_spectra(
            star_data.spectra, planet_data.distance, star_data.incident_angle, planet_data.albedo
        )
    elif "photochemistry" in input_data:
        _log.warning("No star data provided. Skipping photochemistry.")
    atmosphere_data = dispatch_atmosphere(input_data["atmosphere"])

    thermochemistry = dispatch_thermochemistry(
        input_data["thermochemistry"],
        species=network_data.species,
        temperature=atmosphere_data.temperature,
        pressure=atmosphere_data.pressure,
    )

    solver, kwargs = dispatch_solver(input_data["solver"], network_data, photochem_data)

    kwargs["vmr"] = thermochemistry

    solver.set_system_parameters(
        temperature=atmosphere_data.temperature,
        pressure=atmosphere_data.pressure,
        kzz=atmosphere_data.kzz,
        planet_radius=planet_data.radius,
        planet_mass=planet_data.mass,
    )

    return solver, kwargs


def load_freckll_input(input_path: PathLike) -> tuple[Solver, dict]:
    """Load the input data from a file.

    Args:
        input_path: The path to the input file.

    Returns:
        A tuple of the solver and the kwargs.

    """
    from .yaml import load_yaml_from_file

    input_data = load_yaml_from_file(input_path)

    return dispatch_input(input_data)


def load_and_run_input(
    input_path: PathLike,
) -> Solution:
    """Load and run the input data.

    Args:
        input_data: The input data to load.
        **kwargs: Additional arguments to pass to the dispatcher.

    """
    solver, kwargs = load_freckll_input(input_path)

    result = solver.solve(
        **kwargs,  # type: ignore[call-arg]
    )

    return result
