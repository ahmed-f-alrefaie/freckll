"""Load cross-sections and quantum yield data for a molecule."""

import pathlib

from astropy.io.typing import PathLike

from ..reactions.photo import CrossSection, PhotoMolecule, PhotoReactionCall, QuantumYield
from ..species import SpeciesDict, SpeciesFormula


def determine_photodisocciation_reactions(photod_file: PathLike) -> bool:
    """Determine whether photodissociation reactions are available.

    Args:
        photod_file: The path to the photodissociation reaction file.

    Returns:
        bool: True if photodissociation reactions are available, False otherwise.
    """
    photo_path = pathlib.Path(photo_path)
    if not photo_path.exists():
        return False
    # Count the number of reactions in the file

    with open(photod_file) as file:
        lines = file.readlines()
        num_reactions = sum(1 for line in lines if line.strip() and not line.startswith("#"))

    return num_reactions > 0


def determine_cross_sections(section_path: PathLike) -> bool:
    """Determine whether cross-section data is available.

    Args:
        section_path: The path to the cross-section data file.

    Returns:
        bool: True if cross-section data is available, False otherwise.
    """
    section_path = pathlib.Path(section_path)
    if not section_path.exists():
        return False
    if not section_path.is_dir():
        return False

    # Count the number of files in the directory
    num_sections = len(list(section_path.glob("se*.dat")))
    return num_sections > 0


def load_cross_section(molecule: SpeciesFormula, cross_section_file: PathLike) -> CrossSection:
    """Load cross-section data from ``se`` file.

    Args:
        molecule: The name of the molecule.
        cross_section_file: The path to the cross-section data file.

    Returns:
        CrossSection: The loaded cross-section data.
    """
    import numpy as np
    from astropy import units as u

    with open(cross_section_file) as file:
        wav, xsec = np.loadtxt(file, unpack=True)

    return CrossSection(
        molecule,
        wav << u.nm,
        xsec << u.cm**2,
    )


def load_quantum_yield(molecule: SpeciesFormula, branch_id: int, quantum_yield_file: PathLike) -> QuantumYield:
    """Load quantum yield data from ``qy`` file.

    Args:
        molecule: The name of the molecule.
        quantum_yield_file: The path to the quantum yield data file.

    Returns:
        QuantumYield: The loaded quantum yield data.


    """
    import numpy as np
    from astropy import units as u

    with open(quantum_yield_file) as file:
        wav, qy = np.loadtxt(file, unpack=True)

    return QuantumYield(
        molecule,
        branch_id,
        wav << u.nm,
        qy,
    )


def load_all_cross_sections(
    section_path: PathLike,
    molecule_list: list[SpeciesFormula],
) -> SpeciesDict[CrossSection]:
    """Load all cross-sections from a directory.

    Args:
        section_path: The path to the cross-section data directory.
        molecule_list: List of molecules to load.

    Returns:
        SpeciesDict[CrossSection]: Dictionary of loaded cross-sections.
    """

    section_path = pathlib.Path(section_path)
    cross_section_files = section_path.glob("se*.dat")

    cross_sections = SpeciesDict[CrossSection]()
    for file in cross_section_files:
        molecule = file.stem[2:]
        if molecule in molecule_list:
            cross_sections[molecule] = load_cross_section(molecule, file)

    return cross_sections


def load_all_quantum_yields(
    section_path: PathLike,
    molecule_list: list[SpeciesFormula],
) -> SpeciesDict[list[QuantumYield]]:
    """Load all quantum yields from a directory.

    Args:
        section_path: The path to the quantum yield data directory.
        molecule_list: List of molecules to load.

    Returns:
        SpeciesDict[QuantumYield]: Dictionary of loaded quantum yields.
    """

    section_path = pathlib.Path(section_path)
    quantum_yield_files = section_path.glob("qy*.dat")

    quantum_yields = SpeciesDict[QuantumYield]()
    for file in quantum_yield_files:
        filename = file.stem[2:]

        molecule, branch_id = filename.split("_")
        branch_id = int(branch_id)
        if molecule in molecule_list:
            if molecule not in quantum_yields:
                quantum_yields[molecule] = []
            quantum_yields[molecule].append(load_quantum_yield(molecule, branch_id, file))

    return quantum_yields


def construct_photomolecules(
    cross_sections: SpeciesDict[CrossSection],
    quantum_yields: SpeciesDict[list[QuantumYield]],
) -> SpeciesDict[PhotoMolecule]:
    """Construct a list of PhotoMolecule objects from cross-sections and quantum yields.

    Args:
        cross_sections: Dictionary of cross-sections.
        quantum_yields: Dictionary of quantum yields.

    Returns:
        list[PhotoMolecule]: List of constructed PhotoMolecule objects.
    """
    photomolecules = SpeciesDict[PhotoMolecule]()
    for molecule, cross_section in cross_sections.items():
        photo_molecule = PhotoMolecule(molecule, cross_section)
        if molecule in quantum_yields:
            for qy in quantum_yields[molecule]:
                photo_molecule.add_quantum_yield(qy.branch_id, qy)
        photomolecules[photo_molecule.molecule] = photo_molecule

    return photomolecules


def load_photolysis_reactions(
    species_list: list[SpeciesFormula],
    photomolecules: SpeciesDict[PhotoMolecule],
    photodissociation_file: PathLike,
) -> list[PhotoReactionCall]:
    """Load photolysis reactions from a file.

    Args:
        photomolecules: Dictionary of PhotoMolecule objects.
        photodissociation_file: The path to the photodissociation reaction file.

    Returns:
        list[PhotoReactionCall]: List of loaded photolysis reactions.
    """
    from .io import _parse_reaction_line

    with open(photodissociation_file) as file:
        lines = file.readlines()

    photo_reactions = []
    for line in lines:
        reactants, products, coeffs = _parse_reaction_line(line)
        if not reactants or not products:
            continue
        reactant = reactants[0]
        if reactant not in photomolecules:
            raise ValueError(f"Reactant {reactant} not found in photomolecules.")

        if len(reactants) > 1:
            raise ValueError(f"Multiple reactants not supported: {reactants}")
        branch_id = int(coeffs[0])
        reaction_call = PhotoReactionCall(
            species_list=species_list,
            reactant=photomolecules[reactant],
            products=products,
            branch_id=branch_id,
            tags=["photolysis", "photodissociation"],
        )

        photo_reactions.append(reaction_call)

    return photo_reactions
