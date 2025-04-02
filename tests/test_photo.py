"""Test photodisocciation module."""

import numpy as np
from astropy import units as u

from freckll.reactions.photo import CrossSection, PhotoMolecule, QuantumYield


def test_photomolecule_no_yield():
    """Test the PhotoMolecule class."""
    # Create a mock cross-section
    molecule = "H2O"
    wavelengths = [200, 210, 220] * u.nm
    cross_section_values = [1.0, 2.0, 3.0] * u.cm**2 / u.mol
    cross_section = CrossSection(molecule, wavelengths, cross_section_values)

    # Create a mock molecule

    photo_molecule = PhotoMolecule(molecule, cross_section)

    # Check if the molecule and cross-section are set correctly
    assert photo_molecule.molecule == molecule
    assert photo_molecule.cross_section == cross_section

    # Check if the quantum yields dictionary is initialized correctly
    np.testing.assert_allclose(photo_molecule.get_quantum_yield(0).qy, 1.0)


def test_photomolecule_with_yield():
    """Test the PhotoMolecule class with quantum yields."""
    # Create a mock cross-section
    molecule = "H2O"
    wavelengths = [200, 210, 220] * u.nm
    cross_section_values = [1.0, 2.0, 3.0] * u.cm**2
    cross_section = CrossSection(molecule, wavelengths, cross_section_values)

    # Create a mock quantum yield
    branch_id = 1
    qy_values = np.array([0.5, 0.6, 0.7])
    quantum_yield = QuantumYield(molecule, branch_id, wavelengths, qy_values)

    # Create a PhotoMolecule and add the quantum yield
    photo_molecule = PhotoMolecule(molecule, cross_section)
    photo_molecule.add_quantum_yield(branch_id, quantum_yield)

    # Check if the quantum yield is set correctly
    np.testing.assert_allclose(photo_molecule.get_quantum_yield(branch_id).qy, [0.5, 0.6, 0.7])


def test_photomolecule_interpolation():
    """Test the interpolation of cross-section and quantum yields."""
    # Create a mock cross-section
    molecule = "H2O"
    wavelengths = [200, 210, 220] * u.nm
    cross_section_values = [1.0, 2.0, 3.0] * u.cm**2
    cross_section = CrossSection(molecule, wavelengths, cross_section_values)

    # Create a mock quantum yield
    branch_id = 1
    qy_values = np.array([0.5, 0.6, 0.7])
    quantum_yield = QuantumYield(molecule, branch_id, wavelengths, qy_values)

    # Create a PhotoMolecule and add the quantum yield
    photo_molecule = PhotoMolecule(molecule, cross_section)
    photo_molecule.add_quantum_yield(branch_id, quantum_yield)

    # Interpolate to a new wavelength
    new_wavelengths = [205, 215] * u.nm
    interpolated_photo_molecule = photo_molecule.interp_to(new_wavelengths)

    # Check if the interpolated values are correct
    np.testing.assert_allclose(interpolated_photo_molecule.cross_section.wavelength.value, [205.0, 215.0])


def test_photomolecule_reaction_rate_unit():
    molecule = "H2O"
    wavelengths = [200, 210, 220] * u.nm
    cross_section_values = [1.0, 2.0, 3.0] * u.cm**2
    cross_section = CrossSection(molecule, wavelengths, cross_section_values)

    # Create a mock quantum yield
    branch_id = 1
    qy_values = np.array([0.5, 0.6, 0.7])
    quantum_yield = QuantumYield(molecule, branch_id, wavelengths, qy_values)

    # Create a PhotoMolecule and add the quantum yield
    photo_molecule = PhotoMolecule(molecule, cross_section)
    photo_molecule.add_quantum_yield(branch_id, quantum_yield)

    # Define a mock flux
    flux = np.array([1.0, 2.0, 3.0]) * u.W / u.m**2 / u.nm
    # Calculate the reaction rate
    reaction_rate = photo_molecule.reaction_rate(branch_id, flux)


def test_rayleigh():
    from freckll.reactions.photo import rayleigh_species

    wavelength = np.array([200, 210, 220]) * u.nm

    check_cross = rayleigh_species["N2"](wavelength)

    check_cross.to(u.cm**2)
