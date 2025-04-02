import typing as t

import numpy as np
from astropy import units as u

from ..nasa import NasaCoeffs
from ..reactions.common import compile_thermodynamic_properties
from ..reactions.data import Reaction, ReactionCall
from ..reactions.photo import PhotoReactionCall, StarSpectra
from ..species import SpeciesDict, SpeciesFormula
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

    def compile_reactions(self, temperature: u.Quantity, pressure: u.Quantity) -> None:
        """Compile the reactions.

        Args:
            temperature: The temperature.
            pressure: The pressure.

        """
        from ..kinetics import air_density

        thermo_properties = compile_thermodynamic_properties(self.species, self.nasa_coeffs, temperature.to(u.K).value)

        self.compiled_density = air_density(temperature, pressure).to(1 / u.cm**3).value

        pressure = pressure.to(u.mbar).value
        temperature = temperature.to(u.K).value

        for reaction_call in self.reaction_calls:
            reaction_call.compile(temperature, pressure, thermo_properties)

    def compute_reactions(
        self,
        vmr: FreckllArray,
        temperature: FreckllArray = None,
        pressure: FreckllArray = None,
        with_production_loss: bool = False,
        **kwargs: t.Any,
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


class PhotoChemistry:
    def __init__(
        self,
        species_list: list[SpeciesFormula],
        photo_reaction_calls: list[PhotoReactionCall],
        spectra: t.Optional[StarSpectra] = None,
    ) -> None:
        """Initialize the network.

        Args:
            reaction_calls: The reaction calls.
            cross_sections: The cross sections.

        """
        from ..reactions.photo import rayleigh_species

        self.photo_reaction_calls = photo_reaction_calls
        self.working_reaction_calls = photo_reaction_calls
        self.spectra = spectra

        self.available_rayleigh = [c for c in rayleigh_species if c in species_list]

        self.species_index = {c.molecule: species_list.index(c) for c in self.photo_reaction_calls}

        for c in self.available_rayleigh:
            self.species_index[c] = species_list.index(c)

        if spectra is not None:
            self.set_spectra(spectra)

    def set_spectra(self, spectra: StarSpectra) -> None:
        """Set the stellar spectra.

        Args:
            spectra: The spectra to set.

        """
        from ..reactions.photo import CrossSection, rayleigh_species

        self.cross_sections = SpeciesDict[CrossSection]()
        self.spectra = spectra
        for c in self.photo_reaction_calls:
            self.cross_sections[c.reactant] = c.cross_section.interp_to(self.spectra.wavelength)

        for r in self.available_rayleigh:
            c = rayleigh_species[r]
            rayleigh = c(self.spectra.wavelength)
            if r in self.cross_sections:
                cross_section = self.cross_sections[r]
                self.cross_sections[r] = cross_section + rayleigh
            else:
                self.cross_sections[r] = rayleigh
        self.working_reaction_calls = [c.interpolate_to(self.spectra.wavelength) for c in self.photo_reaction_calls]

        self.cross_section_array = (
            np.array([c.cross_section.to(u.cm**2).value for c in self.cross_sections.values()]) << u.cm**2
        )

        self.cross_section_indices = np.array(
            [self.species_index[c] for c in self.cross_sections.keys()], dtype=np.int64
        )

    def compile_chemistry(
        self, distance: u.Quantity, incident_angle: u.Quantity = 45 * u.deg, albedo: float = 0.0
    ) -> None:
        """Compile the photochemistry.

        Args:
            distance: The planet-distance to the star.
            incident_angle: The angle of incidence.


        """

        self.incident_angle = incident_angle
        self.incident_flux = self.spectra.incident_flux(distance)
        self.albedo = albedo

    def compute_reactions(self, number_density: u.Quantity, altitude: u.Quantity) -> list[Reaction]:
        """Compute the reactions.

        Args:
            number_density: The number density.
            altitude: The altitude.

        Returns:
            A list of reactions.

        """
        from ..reactions.photo import optical_depth, radiative_transfer

        tau = optical_depth(altitude, number_density, self.cross_section_array, self.cross_section_indices)

        flux = radiative_transfer(self.incident_flux, tau, self.incident_angle, self.albedo)

        reactions = []

        for reaction_call in self.working_reaction_calls:
            reactions.append(reaction_call(flux, number_density))

        return reactions
