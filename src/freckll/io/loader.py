"""Parse FRECKLL input file."""

import pathlib
import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.io.typing import PathLike

from ..log import setup_log
from ..reactions.photo import StarSpectra
from ..species import SpeciesFormula
from ..venot import VenotChemicalNetwork, VenotPhotoChemistry

Networks = t.Literal["velliet-2024", "venot-methanol-2020", "venot-methanol-2020-reduced"]
Photonetworks = t.Literal["velliet-2024-photo", "venot-methanol-2020-photo"]

Stars = t.Literal[
    "55cnc", "adleo", "gj436", "gj3470", "hd128167", "hd189733", "hd209458", "sun", "wasp12", "wasp39", "wasp43"
]

NetworkFormats = t.Literal["venot"]
PhotchemFormats = t.Literal["venot"]

_log = setup_log(__name__)


class AceFailureError(Exception):
    """Raised when the ACE chemistry model fails to run."""

    pass


def generic_csv_loader(
    filename: PathLike,
    columns: list[int],
    column_units: list[u.Unit | None],
    skiprows: t.Optional[int] = 0,
    delimiter: t.Optional[str] = None,
    comments: t.Optional[str] = None,
) -> tuple[u.Quantity | npt.NDArray[np.floating], ...]:
    """Load a csv file

    Args:
        filename: Path to the csv file.
        columns: List of columns to load. 0 being first
        column_units: List of units for each column.
        skiprows: Number of rows to skip at the beginning of the file.
        delimiter: Delimiter used in the file.
        comments: Comment character in the file.
    Returns:
        Tuple of quantities for each column.

    """
    filename = pathlib.Path(filename)
    if not filename.exists():
        _log.error(f"File {filename} does not exist.")
        raise FileNotFoundError

    if not filename.is_file():
        _log.error(f"File {filename} is not a file.")
        raise FileNotFoundError

    columns = [int(c) for c in columns]

    with open(filename) as f:
        res = np.loadtxt(
            f,
            skiprows=int(skiprows),
            delimiter=delimiter,
            usecols=columns,
            comments=comments,
            unpack=True,
        )

    # Convert to quantities
    quantities = []
    for data, quantity in zip(res, column_units):
        if quantity is None:
            quantities.append(data)
        else:
            quantities.append(data * quantity)
    return tuple(quantities)


def tp_profile_loader(
    *,
    filename: PathLike,
    temperature_column: int,
    pressure_column: int,
    temperature_unit: u.Unit,
    pressure_unit: u.Unit,
    skiprows: t.Optional[int] = 0,
    delimiter: t.Optional[str] = None,
    comments: t.Optional[str] = None,
    start: t.Literal["top", "bottom"] = "bottom",
) -> tuple[u.Quantity, u.Quantity]:
    """Load a temperature-pressure profile from a csv file.

    Args:
        filename: Path to the csv file.
        temperature_column: Column index for temperature. 0 being first
        pressure_column: Column index for pressure. 0 being first
        temperature_unit: Unit for temperature.
        skiprows: Number of rows to skip at the beginning of the file.
        delimiter: Delimiter used in the file.
        comments: Comment character in the file.
        start: Whether the profile starts at the top or bottom of the atmosphere.
    Returns:
        Tuple of temperature and pressure quantities.
    """
    pressure, temperature = generic_csv_loader(
        filename,
        [pressure_column, temperature_column],
        [pressure_unit, temperature_unit],
        skiprows=skiprows,
        delimiter=delimiter,
        comments=comments,
    )

    # Reverse the profile if it starts at the top
    if start == "top":
        pressure = pressure[::-1]
        temperature = temperature[::-1]

    return pressure, temperature


def kzz_profile_loader(
    *,
    filename: PathLike,
    kzz_column: int,
    pressure_column: int,
    kzz_unit: u.Unit,
    pressure_unit: u.Unit,
    skiprows: t.Optional[int] = 0,
    delimiter: t.Optional[str] = None,
    comments: t.Optional[str] = None,
    start: t.Literal["top", "bottom"] = "bottom",
) -> tuple[u.Quantity, u.Quantity]:
    """Load a kzz profile from a csv file.

    Args:
        filename: Path to the csv file.
        kzz_column: Column index for kzz. 0 being first
        pressure_column: Column index for pressure. 0 being first
        kzz_unit: Unit for kzz.
        skiprows: Number of rows to skip at the beginning of the file.
        delimiter: Delimiter used in the file.
        comments: Comment character in the file.
    Returns:
        Tuple of kzz and pressure quantities.


    """
    pressure, kzz = generic_csv_loader(
        filename,
        [pressure_column, kzz_column],
        [pressure_unit, kzz_unit],
        skiprows=skiprows,
        delimiter=delimiter,
        comments=comments,
    )

    # Reverse the profile if it starts at the top
    if start == "top":
        pressure = pressure[::-1]
        kzz = kzz[::-1]
    return pressure, kzz


def star_spectra_loader(
    *,
    filename: PathLike,
    flux_column: int,
    spectral_column: int,
    flux_unit: u.Unit,
    spectral_unit: u.Unit,
    reference_distance: u.Quantity,
    skiprows: t.Optional[int] = 0,
    delimiter: t.Optional[str] = None,
    comments: t.Optional[str] = None,
) -> StarSpectra:
    """Load a kzz profile from a csv file.

    Args:
        filename: Path to the csv file.
        flux_column: Column index for flux. 0 being first
        spectral_column: Column index for spectral. 0 being first
        flux_unit: Unit for flux.
        spectral_unit: Unit for spectral.
        reference_distance: Reference distance for the flux.
        skiprows: Number of rows to skip at the beginning of the file.
        delimiter: Delimiter used in the file.
        comments: Comment character in the file.
    Returns:
        Tuple of kzz and pressure quantities.


    """
    wav, flux = generic_csv_loader(
        filename,
        [spectral_column, flux_column],
        [spectral_unit, flux_unit],
        skiprows=skiprows,
        delimiter=delimiter,
        comments=comments,
    )

    return StarSpectra(
        wav,
        flux,
        reference_distance=reference_distance,
    )


def default_reduced_network_loader() -> VenotChemicalNetwork:
    """Load the default reduced network."""
    import importlib.resources
    import pathlib

    reduced_network_path = importlib.resources.files("freckll.data") / "Venot2020_reduced_TAUREX"
    reduced_network_path = reduced_network_path.resolve()
    reduced_network_path = pathlib.Path(reduced_network_path)

    return VenotChemicalNetwork(
        reduced_network_path,
    )


def default_full_network_loader() -> VenotChemicalNetwork:
    """Load the default full network."""
    import importlib.resources
    import pathlib

    full_network_path = importlib.resources.files("freckll.data") / "Venot2020_Taurex"
    full_network_path = full_network_path.resolve()
    full_network_path = pathlib.Path(full_network_path)

    return VenotChemicalNetwork(
        full_network_path,
    )


def default_latest_network_loader() -> VenotChemicalNetwork:
    """Load the default full network."""
    import importlib.resources
    import pathlib

    full_network_path = importlib.resources.files("freckll.data") / "V23_FRECKLL" / "network"
    full_network_path = full_network_path.resolve()
    full_network_path = pathlib.Path(full_network_path)

    return VenotChemicalNetwork(full_network_path, composes_name="new_composes.dat", coeffs_name="new_coeff_nasa.dat")


def default_venot_photonetwork_loader(species_list: list[SpeciesFormula]) -> VenotPhotoChemistry:
    """Load the default photo network."""
    import importlib.resources

    photo_file = importlib.resources.files("freckll.data") / "Venot2020_Taurex" / "photodissociations.dat"
    photo_file = photo_file.resolve()
    section_path = importlib.resources.files("freckll.data") / "Sections"

    section_path = section_path.resolve()

    return VenotPhotoChemistry(
        species_list,
        photodissociation_file=photo_file,
        cross_section_path=section_path,
    )


def default_latest_photonetwork_loader(species_list: list[SpeciesFormula]) -> VenotPhotoChemistry:
    """Load the default photo network."""
    import importlib.resources

    photo_file = importlib.resources.files("freckll.data") / "V23_FRECKLL" / "network" / "photodissociations.dat"
    photo_file = photo_file.resolve()
    section_path = importlib.resources.files("freckll.data") / "Sections"

    section_path = section_path.resolve()

    return VenotPhotoChemistry(
        species_list,
        photodissociation_file=photo_file,
        cross_section_path=section_path,
    )


def default_photonetwork_loader(network: Photonetworks, species_list: list[SpeciesFormula]) -> VenotPhotoChemistry:
    """Load the default photo network.

    Args:
        network: The network to load. Can be "venot-methanol-2020-photo" or "venot-methanol-2020-reduced-photo".

    Returns:
        The loaded network.

    """
    if network == "velliet-2024-photo":
        return default_latest_photonetwork_loader(species_list)
    elif network == "venot-methanol-2020-photo":
        return default_venot_photonetwork_loader(species_list)
    else:
        _log.error(f"Unknown network '{network}'")
        raise ValueError


def default_network_loader(network: Networks) -> VenotChemicalNetwork:
    """Load the default network.

    Args:
        network: The network to load. Can be "venot-methanol-2020" or "venot-methanol-2020-reduced".

    Returns:
        The loaded network.

    """
    if network == "velliet-2024":
        return default_latest_network_loader()
    elif network == "venot-methanol-2020":
        return default_full_network_loader()
    elif network == "venot-methanol-2020-reduced":
        return default_reduced_network_loader()
    else:
        _log.error(f"Unknown network '{network}'")
        raise ValueError


def default_stellar_spectra_loader(
    star: Stars,
) -> StarSpectra:
    """Load the default stellar spectra.

    Args:
        star: The star to load. Can be "55cnc", "adleo", "gj436", "gj3470", "hd128167", "hd189733", "hd209458", "sun", "wasp12", "wasp39", "wasp43".

    Returns:
        The loaded stellar spectra.
    """

    import importlib.resources
    import pathlib

    if star not in t.get_args(Stars):
        _log.error(f"Unknown star '{star}'")
        raise ValueError

    star_path = importlib.resources.files("freckll.data") / "Stars" / f"stellarflux_{star.lower()}.dat"
    star_path = star_path.resolve()
    star_path = pathlib.Path(star_path)

    with open(star_path) as f:
        wav, flux = np.loadtxt(
            f,
            unpack=True,
        )

    return StarSpectra(
        wav << u.nm,
        flux << u.photon / u.cm**2 / u.s / u.nm,
        reference_distance=1.0 * u.AU,
    )


def rescale_stellar_spectra(*, from_star: Stars, temperature: u.Quantity, radius: u.Quantity) -> StarSpectra:
    from ..utils import rescale_star_spectrum

    _star_temperature_map = {
        "55cnc": 5200 << u.K,
        "adleo": 3477 << u.K,
        "gj436": 3350 << u.K,
        "gj3470": 3552 << u.K,
        "hd128167": 6723 << u.K,
        "hd189733": 5050 << u.K,
        "hd209458": 6117 << u.K,
        "sun": 5778 << u.K,
        "wasp12": 6300 << u.K,
        "wasp39": 5372 << u.K,
        "wasp43": 4400 << u.K,
    }

    _star_radius_map = {
        "55cnc": 0.96 << u.R_sun,
        "adleo": 0.4233 << u.R_sun,
        "gj436": 0.432 << u.R_sun,
        "gj3470": 0.547 << u.R_sun,
        "hd128167": 1.431 << u.R_sun,
        "hd189733": 0.805 << u.R_sun,
        "hd209458": 1.16 << u.R_sun,
        "sun": 1.0 << u.R_sun,
        "wasp12": 1.657 << u.R_sun,
        "wasp39": 1.01 << u.R_sun,
        "wasp43": 0.6 << u.R_sun,
    }

    star = default_stellar_spectra_loader(from_star)

    star = rescale_star_spectrum(
        star, _star_radius_map[from_star], radius, _star_temperature_map[from_star], temperature
    )

    return star


def ace_equil_chemistry_loader(
    *,
    species: list[SpeciesFormula],
    temperature: u.Quantity,
    pressure: u.Quantity,
    therm_file: t.Optional[pathlib.Path] = None,
    elements: t.Sequence[str] = ("H", "He", "C", "N", "O"),
    abundances: t.Sequence[float] = (
        12,
        10.93,
        8.39,
        7.86,
        8.73,
    ),
    **kwargs: t.Any,
) -> npt.NDArray[np.floating]:
    """Loads and runs the ACE chemistry model."""
    import importlib.resources

    import acepython

    from ..ace import equil_chemistry_ace

    therm_file = therm_file or importlib.resources.files("freckll.data") / "new_nasa.therm"

    for x in [
        therm_file,
        importlib.resources.files("freckll.data") / "new_nasa.therm",
        importlib.resources.files("freckll.data") / "NASA.therm",
    ]:
        try:
            return equil_chemistry_ace(
                composition=species,
                therm_file=x,
                elements=elements,
                abundances=abundances,
                temperature=temperature,
                pressure=pressure,
            )
        except acepython.ace.AceError:
            pass

    _log.error("ACE chemistry model failed to run.")
    raise AceFailureError()
