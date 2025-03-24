from ..reactions.data import Reaction, ReactionCall
from ..reactions.common import compile_thermodynamic_properties
from ..species import SpeciesDict, SpeciesFormula
from ..nasa import NasaCoeffs
from ..types import FreckllArray


def map_production_loss(
    reactions: list[Reaction],
) -> tuple[SpeciesDict[list[Reaction]], SpeciesDict[list[Reaction]]]:
    """Map production and loss reactions.

    Args:
        reactions: The reactions to map.

    Returns:
        The production and loss reactions mapped to each species.

    """
    production = SpeciesDict[list[Reaction]]()
    loss = SpeciesDict[list[Reaction]]()
    for reaction in reactions:
        for species in reaction.products:
            if species not in production:
                production[species] = []
            production[species].append(reaction)
        for species in reaction.reactants:
            if species not in loss:
                loss[species] = []
            loss[species].append(reaction)
    return production, loss


class ChemicalNetwork:
    """A chemical network."""

    def __init__(
        self,
        composition: list[SpeciesFormula],
        nasa_coeffs: SpeciesDict[NasaCoeffs],
        reaction_calls: list[ReactionCall],
    ) -> None:
        """Initialize the network.

        Args:
            reaction_calls: The reaction calls.

        """
        self.reaction_calls = reaction_calls
        self.nasa_coeffs = nasa_coeffs
        self.composition = composition

    @property
    def species(self) -> list[SpeciesFormula]:
        """The species in the network."""
        return self.composition

    def compile_reactions(
        self, temperature: FreckllArray, pressure: FreckllArray
    ) -> None:
        """Compile the reactions.

        Args:
            temperature: The temperature.
            pressure: The pressure.

        """
        from ..kinetics import density

        thermo_properties = compile_thermodynamic_properties(
            self.species, self.nasa_coeffs, temperature
        )

        self.compiled_density = density(temperature, pressure)

        for reaction_call in self.reaction_calls:
            reaction_call.compile(temperature, pressure, thermo_properties)

    def compute_reactions(
        self,
        vmr: FreckllArray,
        temperature: FreckllArray = None,
        pressure: FreckllArray = None,
        with_production_loss: bool = False,
    ) -> (
        tuple[
            list[Reaction],
            tuple[SpeciesDict[list[Reaction]], SpeciesDict[list[Reaction]]],
        ]
        | list[Reaction]
    ):
        """Compute the reactions.

        Args:
            temperature: The temperature.
            pressure: The pressure.

        Returns:
            A list of reactions, the production reactions and the loss reactions.

        """
        if temperature is not None and pressure is not None:
            self.compile_reactions(temperature, pressure)

        concentration = vmr * self.compiled_density
        reactions = []
        for reaction_call in self.reaction_calls:
            reactions.extend(reaction_call(concentration))
        if not with_production_loss:
            return reactions
        return reactions, map_production_loss(reactions)
