"""Chemical netowrk from Olivia Venot"""

import pathlib
import typing as t

from astropy.io.typing import PathLike

from ..network import ChemicalNetwork, PhotoChemistry
from ..reactions.photo import PhotoMolecule
from ..species import SpeciesFormula


class VenotChemicalNetwork(ChemicalNetwork):
    """A chemical network from Olivia Venot."""

    def __init__(
        self,
        network_path: pathlib.Path,
        composes_name: str = "composes.dat",
        coeffs_name: str = "coeff_NASA.dat",
        efficacies_name: str = "efficacites.dat",
    ) -> None:
        """Initialize the network.

        Args:
            network_path: The path to the network.

        """
        from .io import load_composition, load_efficiencies, load_nasa_coeffs, load_reactions

        network_path = pathlib.Path(network_path)
        if not network_path.is_dir():
            self.error(f"{network_path} is not a directory")
            raise ValueError
        composes_file = network_path / composes_name
        nasa_file = network_path / coeffs_name
        effi_file = network_path / efficacies_name

        if not composes_file.exists():
            self.error(f"{composes_file} does not exist")
            raise FileNotFoundError
        else:
            composition, decoder = load_composition(composes_file)

        efficiencies = load_efficiencies(effi_file, composition, decoder)
        nasa_coeffs = load_nasa_coeffs(nasa_file, decoder, species=composition)

        reactions = load_reactions(
            composition,
            network_path,
            efficiencies,
            decode_species=decoder,
            ignore_files=[composes_file.stem, nasa_file.stem, effi_file.stem],
        )

        super().__init__(composition, nasa_coeffs, reactions)


class VenotPhotoChemistry(PhotoChemistry):
    """Loads photochemistry data."""

    def __init__(
        self,
        species_list: list[SpeciesFormula],
        photodissociation_file: PathLike,
        photomolecules: t.Optional[list[PhotoMolecule]] = None,
        cross_section_path: t.Optional[PathLike] = None,
    ) -> None:
        """Initialize the photochemistry.

        Args:
            species_list: The list of species. Must come from a chemical network.
            photo_dissociation_file: The path to the photodissociation data.
            photomolecules: A list of photomolecules, previously loaded.
            cross_section_path: The path to the cross-section data (if not passing photomolecules).

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

        super().__init__(species_list, photo_reactions)
