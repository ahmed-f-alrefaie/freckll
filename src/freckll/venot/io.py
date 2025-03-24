"""Parsing Olivia Venot chemical network data."""

import pathlib
import typing as t

import numpy as np
import numpy.typing as npt

from ..nasa import NasaCoeffs
from ..reactions.data import ReactionCall
from ..species import SpeciesDict, SpeciesFormula, SpeciesState


def _parse_nasa_lines(
    line_1: str,
    line_2: str,
    line_3: str,
    decode_species: t.Callable[[str], SpeciesFormula] = SpeciesFormula,
) -> NasaCoeffs:
    """Parse NASA polynomial coefficients from three lines."""
    species, x1, x2, x3 = line_1.split()
    a_coeff = np.array([float(s) for s in line_2.split()])
    b_coeff = np.array([float(s) for s in line_3.split()])
    return NasaCoeffs(
        decode_species(species), float(x1), float(x2), float(x3), a_coeff, b_coeff
    )


def load_nasa_coeffs(file_path: pathlib.Path | str) -> SpeciesDict[NasaCoeffs]:
    """Load NASA polynomial coefficients from a file."""
    nasa_coeffs = SpeciesDict[NasaCoeffs]()

    file_path = pathlib.Path(file_path)

    with open(file_path) as file:
        while True:
            line_1 = file.readline().strip()
            if not line_1:
                break
            line_2 = file.readline().strip()
            line_3 = file.readline().strip()
            nasa = _parse_nasa_lines(line_1, line_2, line_3, _decode_species)
            nasa_coeffs[nasa.species] = nasa

    return nasa_coeffs


# Some species have multiple isomers/states, so we need to map the species to the correct isomer/state
_species_mapping: dict[str, SpeciesFormula] = {
    "CH2CHO": SpeciesFormula("CH3CO", isomer_id="H2", input_formula="CH2CHO"),
    "C4H8Y": SpeciesFormula("C4H8", isomer_id="Y", input_formula="C4H8Y"),
    "C2H3CHOZ": SpeciesFormula("C2H3CHO", isomer_id="Z", input_formula="C2H3CHOZ"),
    "N4S": SpeciesFormula("N", isomer_id="4S", input_formula="N4S"),
    "N2D": SpeciesFormula("N", isomer_id="2D", input_formula="N2D"),
    "H2Oc": SpeciesFormula("H2O", state=SpeciesState.LIQUID, input_formula="H2Oc"),
    "CH4c": SpeciesFormula("CH4", state=SpeciesState.LIQUID, input_formula="CH4c"),
    "NH3c": SpeciesFormula("NH3", state=SpeciesState.LIQUID, input_formula="NH3c"),
    "O3P": SpeciesFormula("O", isomer_id="3P", input_formula="O3P"),
    "O1D": SpeciesFormula("O", isomer_id="1D", input_formula="O1D"),
    "toluene": SpeciesFormula("C7H8", input_formula="toluene"),
    "C4H7T": SpeciesFormula("C4H7", isomer_id="T", input_formula="C4H7T"),
    "1cC8H9": SpeciesFormula("C8H9", isomer_id="1c", input_formula="1cC8H9"),
}


def _decode_species(s: str) -> SpeciesFormula:
    """Decode a species formula from a string.

    Args:
        s: The species formula string.

    Returns:
        The decoded species formula

    """
    # Try to figure out the variant.
    # If the first letter is lowercase or a number, it's a variant. Use it as an identifier

    if s not in _species_mapping:
        if s[0].islower() or (s[0].isdigit() and s[1].isupper()):
            return SpeciesFormula(s[1:], isomer_id=s[0], input_formula=s)
        if s[0].isdigit() and s[1].isdigit():
            return SpeciesFormula(s[2:], isomer_id=int(s[:2]), input_formula=s)

    spec = _species_mapping.get(s)

    if spec is not None:
        return spec

    return SpeciesFormula(s, input_formula=s)


def _parse_reaction_line(
    line: str,
) -> tuple[list[SpeciesFormula], list[SpeciesFormula], npt.NDArray[np.floating]]:
    """Parse a reaction line from the CHEGP format.

    Args:
        line: The reaction line.

    Returns:
        The reactants, products, and coefficients.


    """
    values = [line[x : x + 11].strip() for x in range(1, len(line), 11)]

    reactants = [_decode_species(x.strip()) for x in values[:5] if x and x != "HV"]
    products = [_decode_species(x.strip()) for x in values[5:10] if x and x != "HV"]
    coeffs = np.array([float(x.replace("d", "e")) for x in line[112:].split()])
    return reactants, products, coeffs


def load_efficiencies(
    file_path: pathlib.Path | str, composition: list[SpeciesFormula]
) -> npt.NDArray[np.integer]:
    """Load the efficiencies from a file.

    Args:
        file_path: The file path.

    Returns:
        The efficiencies.

    """
    file_path = pathlib.Path(file_path)

    efficiencies = []

    with open(file_path) as file:
        for line in file:
            if not line:
                break
            efficiencies.append(_decode_species(line.strip()))
    # Build a species
    efficiencies = [e for e in efficiencies if e in composition]

    # Build species index
    efficiencies = np.array([composition.index(e) for e in efficiencies])

    return efficiencies


def load_composition(file_path: pathlib.Path | str) -> list[SpeciesFormula]:
    """Load the composition from a file.

    Args:
        file_path: The file path.

    Returns:
        The composition.

    """
    file_path = pathlib.Path(file_path)

    composition = []

    with open(file_path) as file:
        while True:
            line = file.readline().strip()
            if not line:
                break
            _, species, _, _, _, _, _ = line.split()
            composition.append(_decode_species(species.strip()))

    return composition


def parse_reaction_file(
    file_path: pathlib.Path | str,
) -> list[tuple[list[SpeciesFormula], list[SpeciesFormula], list[float]]]:
    """Parse a reaction file.

    Args:
        file_path: The file path.

    Returns:
        The reactions.

    """
    file_path = pathlib.Path(file_path)

    reactions = []

    with open(file_path) as file:
        for line in file:
            if not line:
                break

            reactants, products, coeffs = _parse_reaction_line(line)
            reactions.append((reactants, products, coeffs))

    return reactions


def build_efficienies(
    composition: list[SpeciesFormula],
    efficiencies: list[float],
    efficiency_index: npt.NDArray[np.integer],
    fill_value: float = 1.0,
) -> npt.NDArray[np.integer]:
    """Build the efficiencies array.

    Args:
        composition: The composition.
        efficiencies: The efficiencies.
        fill_value: The fill value.

    Returns:
        The efficiencies array.

    """
    efficiencies_array = np.full(len(composition), fill_value)
    efficiencies = np.array(efficiencies)

    num_efficiencies = min(len(efficiencies), len(efficiency_index))
    efficiencies_array[efficiency_index[:num_efficiencies]] = efficiencies[
        :num_efficiencies
    ]

    return efficiencies_array


def _handle_k0_reactions(
    composition: list[SpeciesFormula],
    reaction_data: list[tuple[list[SpeciesFormula], list[SpeciesFormula], list[float]]],
    efficiency_indices: npt.NDArray[np.integer],
    inverted: bool,
    file_path: pathlib.Path,
) -> list[ReactionCall]:
    """Handle k0 reactions."""
    from functools import partial

    import freckll.reactions.falloff as falloff
    import freckll.reactions.reactions as react

    stem_name = file_path.stem

    reaction_calls = []

    if "kinf" in stem_name:
        fall_function = (
            falloff.sri_falloff if "SRI" in stem_name else falloff.troe_falloff_term
        )

        react_function = (
            react.decomposition_k0kinf_reaction
            if "decompo" in stem_name
            else react.k0kinf_reaction
        )
        efficiency_fill = 0.0 if "sansM" in stem_name else 1.0
        for r, p, c in reaction_data:
            k0_coeffs = c[:5]
            kinf_coeffs = c[5:10]
            coeffs = c[10:]
            falloff_coeffs = coeffs[:5] if "SRI" in stem_name else coeffs[:4]
            efficiency_coeff = coeffs[5:] if "SRI" in stem_name else coeffs[4:]

            efficiencies = build_efficienies(
                composition,
                efficiency_coeff,
                efficiency_indices,
                fill_value=efficiency_fill,
            )
            reaction_calls.append(
                ReactionCall(
                    composition,
                    r,
                    p,
                    [
                        "k0",
                        "kinf",
                        "falloff",
                        "SRI" if "SRI" in stem_name else "Troe",
                        "decomposition" if "decompo" in stem_name else "reaction",
                    ],
                    inverted,
                    partial(
                        react_function,
                        k0_coeffs,
                        kinf_coeffs,
                        falloff_coeffs,
                        efficiencies,
                        inverted,
                        fall_function,
                        r,
                        p,
                    ),
                )
            )

    else:
        react_function = (
            react.decomposition_k0_reaction
            if "decompo" in stem_name
            else react.k0_reaction
        )
        for r, p, c in reaction_data:
            k0_coeffs = c[:5]
            efficiency_coeff = c[5:]
            efficiencies = build_efficienies(
                composition,
                efficiency_coeff,
                efficiency_indices,
                fill_value=1.0,
            )
            reaction_calls.append(
                ReactionCall(
                    composition,
                    r,
                    p,
                    ["k0", "reaction"],
                    inverted,
                    partial(react_function, k0_coeffs, efficiencies, inverted, r, p),
                )
            )

    return reaction_calls


def _handle_plog_reactions(
    composition: list[SpeciesFormula],
    reaction_data: list[tuple[list[SpeciesFormula], list[SpeciesFormula], list[float]]],
    inverted: bool,
    file_path: pathlib.Path,
) -> list[ReactionCall]:
    """Handle plog reactions."""
    from functools import partial

    import freckll.reactions.reactions as react

    reaction_function = (
        react.decomposition_plog
        if "decompo" in file_path.stem
        else react.manybody_plog_reaction
    )
    reaction_calls = []
    for r, p, c in reaction_data:
        *plog_coeffs, _, _ = c
        reaction_calls.append(
            ReactionCall(
                composition,
                r,
                p,
                [
                    "plog",
                    "decomposition" if "decompo" in file_path.stem else "many body",
                ],
                inverted,
                (
                    partial(reaction_function, plog_coeffs, inverted, p)
                    if "decompo" in file_path.stem
                    else partial(reaction_function, plog_coeffs, inverted, r, p)
                ),
            )
        )
    return reaction_calls


def _construct_reaction_call(  # noqa: C901
    composition: list[SpeciesFormula],
    file_path: pathlib.Path,
    efficiency_indices: npt.NDArray[np.integer],
) -> list[ReactionCall]:
    """Given a filepath, construct the correct_reaction call."""
    from functools import partial

    import freckll.reactions.reactions as react

    stem_name = file_path.stem
    if stem_name in ("composes", "coeff_NASA", "efficacites", "photodissociations"):
        return []

    reaction_data = parse_reaction_file(file_path)
    if len(reaction_data) == 0:
        return []
    inverted = "irrev" not in stem_name

    if "k0" in stem_name:
        return _handle_k0_reactions(
            composition,
            reaction_data,
            efficiency_indices,
            inverted,
            file_path,
        )
    if "desexcitation" in stem_name:
        reaction_calls = []
        for r, p, c in reaction_data:
            efficiencies = build_efficienies(
                composition, c[5:], efficiency_indices, fill_value=1.0
            )
            reaction_calls.append(
                ReactionCall(
                    composition,
                    r,
                    p,
                    ["desexcitation"],
                    inverted,
                    partial(
                        react.de_excitation_reaction,
                        c[:5],
                        efficiencies,
                        inverted,
                        r,
                        p,
                    ),
                )
            )
        return reaction_calls
    if "plog" in stem_name:
        return _handle_plog_reactions(composition, reaction_data, inverted, file_path)
    if "reaction_2_Corps" in stem_name:
        reaction_calls = []
        for r, p, c in reaction_data:
            reaction_calls.append(
                ReactionCall(
                    composition,
                    r,
                    p,
                    ["corps", "reaction", "many body"],
                    inverted,
                    partial(react.corps_reaction, c, inverted, r, p),
                )
            )
        return reaction_calls
    if "decompo" in stem_name:
        reaction_calls = []
        for r, p, c in reaction_data:
            reaction_calls.append(
                ReactionCall(
                    composition,
                    r,
                    p,
                    ["decomposition"],
                    inverted,
                    partial(react.decomposition_reaction, c, inverted, p),
                )
            )
        return reaction_calls
    return []


def load_reactions(
    composition: list[SpeciesFormula],
    directory: pathlib.Path | str,
    efficiency_indices: npt.NDArray[np.integer],
) -> list[ReactionCall]:
    """Construct a reaction call from a directory.

    Args:
        directory: The directory.

    Returns:
        The reaction call.

    """
    directory = pathlib.Path(directory)

    # Glob .dat files
    reaction_files = directory.glob("*.dat")

    # Reaction and its type is based on the file name
    reaction_calls = []
    for r in reaction_files:
        reaction_calls.extend(
            _construct_reaction_call(composition, r, efficiency_indices)
        )

    return reaction_calls


def infer_composition(
    directory: pathlib.Path | str,
) -> list[SpeciesFormula]:
    """Infer the composition from the reactions.

    Loop through and load in all of the reactions and determine
    the species in the network from all of the reactants and products.

    Args:
        directory: The directory.



    """
    directory = pathlib.Path(directory)

    # Glob .dat files
    reaction_files = directory.glob("*.dat")

    composition: set[SpeciesFormula] = set()

    for r in reaction_files:
        stem_name = r.stem
        if stem_name in ("composes", "coeff_NASA", "efficacites"):
            continue
        data = parse_reaction_file(r)
        if len(data) == 0:
            continue
        reactants, products, _ = list(zip(*data))
        # flatten reactants
        reactants = [item for sublist in reactants for item in sublist]
        # flatten products
        products = [item for sublist in products for item in sublist]
        composition |= set(reactants) | set(products)

    return list(composition)
