import pathlib
import typing as t
from contextlib import contextmanager

import numpy as np
from acepython import run_ace
from astropy import units as u

from .species import SpeciesFormula
from .types import FreckllArrayInt


@contextmanager
def create_composes(
    composition: list[SpeciesFormula],
    use_input_formula: bool = True,
    elements: t.Optional[t.Sequence[str]] = ("C", "H", "O", "N"),
) -> t.Iterator[tuple[str, FreckllArrayInt]]:
    """Creates a composition file for ACE."""
    import tempfile

    composition_index = []
    filetemp = tempfile.NamedTemporaryFile(mode="w", delete=False)
    try:
        for idx, s in enumerate(composition):
            if s.state != "gas":
                continue
            composition_index.append(idx)
            formula = s.input_formula if use_input_formula else s.formula
            comp = s.composition().asdict()

            composition_str = " ".join([f"{comp.get(e, [0])[0]:2d}" for e in elements])

            filetemp.write(f"{idx + 1:3d}  {formula:12} {s.monoisotopic_mass:6.3f} {composition_str}\n")
        filetemp.close()
        yield filetemp.name, np.array(composition_index)
    finally:
        pathlib.Path(filetemp.name).unlink()


def equil_chemistry_ace(
    temperature: u.Quantity,
    pressure: u.Quantity,
    composition: list[SpeciesFormula],
    therm_file: pathlib.Path,
    elements: t.Sequence[str] = ("H", "He", "C", "N", "O"),
    abundances: t.Sequence[float] = (
        12,
        10.93,
        8.39,
        7.86,
        8.73,
    ),
    use_input_formula: bool = True,
    pressure_unit: str = "bar",
    composes_elements: t.Sequence[str] = ("C", "H", "O", "N"),
) -> tuple[list[str], list[float], float]:
    """Computes the ACE profile from a chemical network.

    Args:
        temperature: The temperature profile.
        pressure: The pressure profile.
        composition: The composition of the network.
        use_input_formula: Whether to use the input formula.
        elements: The elements in the network.

    Returns:
        The species, mixing ratios and mean molecular weight.

    """

    vmr = np.full(shape=(len(composition), len(temperature)), fill_value=1e-50)

    with create_composes(composition, use_input_formula, composes_elements) as (
        specfile,
        indices,
    ):
        species, mix_profile, mmw = run_ace(
            temperature,
            pressure,
            elements=elements,
            abundances=abundances,
            specfile=specfile,
            thermfile=therm_file,
        )
        vmr[indices] = mix_profile

    return vmr, mmw
