"""Handles storing reactions."""

import typing as t
from dataclasses import dataclass

import numpy as np

from freckll.species import SpeciesFormula

from ..types import FreckllArray, FreckllArrayInt, ReactionFunction


@dataclass
class Reaction:
    reactants: list[SpeciesFormula]
    products: list[SpeciesFormula]
    reactants_indices: FreckllArrayInt
    product_indices: FreckllArrayInt
    reaction_rate: FreckllArray
    tags: list[str]
    dens_krate: FreckllArray = None

    def calculate_density_krate(self, number_density: FreckllArray):
        """Calculate the density krate."""
        reactants_dens = np.prod(number_density[self.reactants_indices], axis=0)
        self.dens_krate = self.reaction_rate * reactants_dens

    def __repr__(self):
        reactants = " + ".join([str(x) for x in self.reactants])
        products = " + ".join([str(x) for x in self.products])
        return f"{reactants} -> {products} ({self.reaction_rate.mean():.2e}) ({self.tags})"


class ReactionCall:
    """A function that builds a standardized reaction computation."""

    def __init__(
        self,
        species_list: list[SpeciesFormula],
        reactants: list[SpeciesFormula],
        products: list[SpeciesFormula],
        tags: list[str],
        inverted: bool,
        reaction_function: t.Callable[..., ReactionFunction],
    ) -> None:
        """Initialize the reaction call.

        Args:
            species_list: The species list.
            reactants: The reactants.
            products: The products.
            tags: The tags.
            inverted: Whether the reaction is inverted.
            reaction_function: The reaction function.



        """
        self.reactants = reactants
        self.products = products
        self.inverted = inverted
        self.reaction_function = reaction_function
        self.tags = tags
        self.reactants_indices = np.array([species_list.index(x) for x in reactants])
        self.product_indices = [species_list.index(x) for x in products]

    def compile(
        self,
        temperature: FreckllArray,
        pressure: FreckllArray,
        thermo_properties: FreckllArray,
    ) -> None:
        """Compile the reaction."""
        thermo_reactants = thermo_properties[self.reactants_indices]
        thermo_products = thermo_properties[self.product_indices]

        self.reaction_call = self.reaction_function(temperature, pressure, thermo_reactants, thermo_products)

    def __call__(
        self,
        concentration: FreckllArray,
    ) -> list[Reaction]:
        """Call the reaction.

        A normalised reaction call.

        Args:
            concentration: The concentration.

        Returns:
            list[Reaction]: The reaction.


        """
        reaction_rate = self.reaction_call(concentration)
        reactions = [
            Reaction(
                self.reactants,
                self.products,
                self.reactants_indices,
                self.product_indices,
                reaction_rate[0],
                self.tags,
            )
        ]
        if self.inverted:
            reactions.append(
                Reaction(
                    self.products,
                    self.reactants,
                    self.product_indices,
                    self.reactants_indices,
                    reaction_rate[1],
                    [*self.tags, "inverted"],
                )
            )
        for r in reactions:
            r.calculate_density_krate(concentration)
        return reactions

    def __repr__(self) -> str:
        return f"ReactionCall({self.reactants}{'->' if not self.inverted else '<->'}{self.products}, {self.tags})"
