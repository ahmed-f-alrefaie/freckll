"""Tests for species handling."""


def test_species_formula_init():
    """Tests species formula."""
    from freckll.species import SpeciesFormula

    species = SpeciesFormula("H2O")
    assert species.formula == "H2O"
    assert species.state == "gas"

    species = SpeciesFormula("H2O", state="liquid")
    assert species.formula == "H2O"
    assert species.state == "liquid"


    species = SpeciesFormula("H2O", state="solid")
    assert species.formula == "H2O"
    assert species.state == "solid"


    species = SpeciesFormula("H2O[s]")
    assert species.formula == "H2O"
    assert species.state == "solid"


    species = SpeciesFormula("H2O[l]")
    assert species.formula == "H2O"
    assert species.state == "liquid"


    species = SpeciesFormula("H2O[c]")
    assert species.formula == "H2O"
    assert species.state == "liquid"


    species = SpeciesFormula("O2H", true_formula="H2O")
    assert species.formula == "H2O"
    assert species.state == "gas"
    assert species.input_formula == "O2H"


def test_species_states_different():
    """Test species states are different."""
    from freckll.species import SpeciesFormula

    species1 = SpeciesFormula("H2O")
    species2 = SpeciesFormula("H2O", state="liquid")
    species3 = SpeciesFormula("H2O", state="solid")
    species4 = SpeciesFormula("H2O[s]")
    species5 = SpeciesFormula("H2O[l]")
    species6 = SpeciesFormula("H2O[c]")

    assert species1 != species2
    assert species1 != species3
    assert species1 != species4
    assert species1 != species5
    assert species1 != species6

    assert species2 != species3
    assert species2 != species4
    assert species2 == species5
    assert species2 == species6

    assert species3 == species4
    assert species3 != species5
    assert species3 != species6

    assert species4 != species5
    assert species4 != species6

    assert species5 == species6


def test_species_not_same():
    """Test species not the same."""
    from freckll.species import SpeciesFormula

    species1 = SpeciesFormula("CH3CO")
    species2 = SpeciesFormula("CH2CHO")

    assert species1 != species2

def test_species_similarity():
    """Test species similarity."""
    from freckll.species import SpeciesFormula

    species1 = SpeciesFormula("H2O")
    species2 = SpeciesFormula("OH2")

    assert species1 != species2


# def test_isomer_id():
#     """Test isomer id."""
#     from freckll.species import SpeciesFormula

#     species1 = SpeciesFormula("H2O", isomer_id=1)
#     species2 = SpeciesFormula("H2O", isomer_id=1)

#     assert species1 == species2
#     assert hash(species1) == hash(species2)
#     species1 = SpeciesFormula("H2O", isomer_id="B")
#     species2 = SpeciesFormula("H2O", isomer_id="Z")
#     species3 = SpeciesFormula("H2O")

#     assert species1 != species2
#     assert hash(species1) != hash(species2)
#     assert species3 != species1
#     assert hash(species3) != hash(species1)
#     assert species3 != species2
#     assert hash(species3) != hash(species2)

#     species1 = SpeciesFormula("H2O", isomer_id="B")
#     species2 = SpeciesFormula("H2O", isomer_id="B")

#     assert species1 == species2

#     assert hash(species1) == hash(species2)
