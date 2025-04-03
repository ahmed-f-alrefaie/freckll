"""Chemical netowrk from Olivia Venot"""

import pathlib
import typing as t

from astropy.io.typing import PathLike

from ..network import ChemicalNetwork, PhotoChemistry
from ..reactions.photo import PhotoMolecule, StarSpectra
from ..species import SpeciesFormula


class VenotChemicalNetwork(ChemicalNetwork):
    """A chemical network from Olivia Venot."""

    def __init__(self, network_path: pathlib.Path) -> None:
        """Initialize the network.

        Args:
            network_path: The path to the network.

        """
        from .io import infer_composition, load_composition, load_efficiencies, load_nasa_coeffs, load_reactions

        network_path = pathlib.Path(network_path)
        if not network_path.is_dir():
            raise ValueError(f"{network_path} is not a directory")
        composes_file = network_path / "composes.dat"
        nasa_file = network_path / "coeff_NASA.dat"
        efficiencies = network_path / "efficacites.dat"

        if not composes_file.exists():
            print("Inferring network from directory structure.")
            composition = infer_composition(network_path)
        else:
            composition = load_composition(composes_file)

        efficiencies = load_efficiencies(efficiencies, composition)
        nasa_coeffs = load_nasa_coeffs(nasa_file)
        reactions = load_reactions(composition, network_path, efficiencies)

        super().__init__(composition, nasa_coeffs, reactions)


class VenotPhotoChemistry(PhotoChemistry):
    """Loads photochemistry data."""

    def __init__(
        self,
        species_list: list[SpeciesFormula],
        photodissociation_file: PathLike,
        photomolecules: t.Optional[list[PhotoMolecule]] = None,
        cross_section_path: t.Optional[PathLike] = None,
        star_spectra: t.Optional[StarSpectra] = None,
    ) -> None:
        """Initialize the photochemistry.

        Args:
            photo_dissociation_file: The path to the photodissociation data.
            photomolecules: A list of photomolecules, previously loaded.
            cross_section_path: The path to the cross-section data (if not passing photomolecules).
            star_spectra_path: The path to the star spectra data.

        """
        from .photo import (
            construct_photomolecules,
            load_all_cross_sections,
            load_all_quantum_yields,
            load_photolysis_reactions,
        )

        if photomolecules is None:
            cross_sections = load_all_cross_sections(cross_section_path, species_list)
            quantum_yields = load_all_quantum_yields(cross_section_path, species_list)
            photomolecules = construct_photomolecules(cross_sections, quantum_yields)

        photo_reactions = load_photolysis_reactions(
            species_list,
            photomolecules,
            photodissociation_file,
        )

        super().__init__(species_list, photo_reactions, star_spectra)
