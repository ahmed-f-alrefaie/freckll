import pytest


def test_taurex3_detect():
    """Tests taurex3 to see if it runs."""
    pytest.importorskip("taurex")

    from taurex.parameter.classfactory import ClassFactory

    from freckll.taurex import FreckllChemistryFile, FreckllChemistryInput

    cf = ClassFactory()

    assert FreckllChemistryInput in cf.chemistryKlasses
    assert FreckllChemistryFile in cf.chemistryKlasses


def test_taurex3_full():
    """Tests taurex3 to see if it runs."""
    pytest.importorskip("taurex")

    import numpy as np
    from taurex.planet import Planet
    from taurex.stellar import BlackbodyStar

    from freckll.taurex import FreckllChemistryInput

    star = BlackbodyStar(temperature=5778, radius=1)
    planet = Planet(planet_mass=1, planet_radius=1)

    temperature = np.linspace(3000, 1000, 100)
    pressure = np.logspace(6, -2, 100)

    tau = FreckllChemistryInput()
    tau.set_star_planet(star, planet)
    tau.initialize_chemistry(temperature_profile=temperature, pressure_profile=pressure)
    assert tau.photochemistry is not None
    assert tau.mixProfile.shape[-1] == pressure.shape[-1]
    assert tau.muProfile.shape[-1] == pressure.shape[-1]


def test_taurex3_reduced():
    """Tests taurex3 to see if it runs."""
    pytest.importorskip("taurex")

    import numpy as np
    from taurex.planet import Planet
    from taurex.stellar import BlackbodyStar

    from freckll.taurex import FreckllChemistryInput

    star = BlackbodyStar(temperature=5778, radius=1)
    planet = Planet(planet_mass=1, planet_radius=1)

    temperature = np.linspace(3000, 1000, 100)
    pressure = np.logspace(6, -2, 100)

    tau = FreckllChemistryInput(network="venot-methanol-2020-reduced")
    assert tau.photochemistry is None
    tau.set_star_planet(star, planet)
    tau.initialize_chemistry(temperature_profile=temperature, pressure_profile=pressure)

    assert tau.mixProfile.shape[-1] == pressure.shape[-1]
    assert tau.muProfile.shape[-1] == pressure.shape[-1]


def test_taurex3_c_ratio():
    """Tests taurex3 to see if it runs."""
    pytest.importorskip("taurex")

    from freckll.taurex import FreckllChemistryInput

    tau = FreckllChemistryInput()
    assert "C_O_ratio" in tau.fitting_parameters()
