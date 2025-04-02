"""Test diffusion computation."""

import pytest


@pytest.mark.parametrize(
    "species_str, expected",
    [("H2O", 13.1), ("H2", 6.12), ("H2O2", 16.84)],
)
def test_diffusion_volume(species_str: str, expected: float) -> None:
    from freckll.diffusion import diffusion_volume
    from freckll.species import SpeciesFormula

    species = SpeciesFormula(species_str)

    volume = diffusion_volume(species)

    assert volume == expected
