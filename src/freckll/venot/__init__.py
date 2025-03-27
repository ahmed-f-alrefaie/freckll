"""Chemical netowrk from Olivia Venot"""

from ..network import ChemicalNetwork
import pathlib



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
