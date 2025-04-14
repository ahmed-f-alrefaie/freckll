"""Test for io related functions"""

import numpy as np
import pytest
from astropy import units as u


def test_generic_csv_loader(tmpdir):
    """Test the generic csv loader."""
    from freckll.io.loader import generic_csv_loader

    filename = tmpdir / "test.csv"

    filename.write_text(
        """# This is a comment
        # This is another comment
        0.0, 1.0, 2.0, 3.0
        4.0, 5.0, 6.0, 7.0
        8.0, 9.0, 10.0, 11.0
        12.0, 13.0, 14.0, 15.0""",
        encoding="utf-8",
    )

    res_1, res_2, res_3 = generic_csv_loader(
        filename,
        [0, 1, 2],
        [u.m, u.s, None],
        skiprows=2,
        delimiter=",",
        comments="#",
    )

    assert isinstance(res_1, u.Quantity)

    assert isinstance(res_2, u.Quantity)

    assert isinstance(res_3, np.ndarray)

    res_1.to(u.km)

    res_2.to(u.h)

    np.testing.assert_array_equal(res_1.value, np.array([0.0, 4.0, 8.0, 12.0]))
    np.testing.assert_array_equal(res_2.value, np.array([1.0, 5.0, 9.0, 13.0]))
    np.testing.assert_array_equal(res_3, np.array([2.0, 6.0, 10.0, 14.0]))


def test_generic_csv_loader_swap_order(tmpdir):
    """Test the generic csv loader."""
    from freckll.io.loader import generic_csv_loader

    filename = tmpdir / "test.csv"

    filename.write_text(
        """# This is a comment
        # This is another comment
        0.0, 1.0, 2.0, 3.0
        4.0, 5.0, 6.0, 7.0
        8.0, 9.0, 10.0, 11.0
        12.0, 13.0, 14.0, 15.0""",
        encoding="utf-8",
    )

    res_2, res_1, res_3 = generic_csv_loader(
        filename,
        [1, 0, 2],
        [u.s, u.m, None],
        skiprows=2,
        delimiter=",",
        comments="#",
    )

    assert isinstance(res_1, u.Quantity)

    assert isinstance(res_2, u.Quantity)

    assert isinstance(res_3, np.ndarray)

    res_1.to(u.km)

    res_2.to(u.h)

    np.testing.assert_array_equal(res_1.value, np.array([0.0, 4.0, 8.0, 12.0]))
    np.testing.assert_array_equal(res_2.value, np.array([1.0, 5.0, 9.0, 13.0]))
    np.testing.assert_array_equal(res_3, np.array([2.0, 6.0, 10.0, 14.0]))


@pytest.mark.parametrize(
    "star",
    [
        ("55cnc"),
        ("adleo"),
        ("gj436"),
        ("gj3470"),
        ("hd128167"),
        ("hd189733"),
        ("hd209458"),
        ("sun"),
        ("wasp12"),
        ("wasp39"),
        ("wasp43"),
    ],
)
def test_load_default_stellar_spectra(star):
    from freckll.io.loader import load_default_stellar_spectra
    from freckll.reactions.photo import StarSpectra

    spectra = load_default_stellar_spectra(star)
    assert spectra is not None
    assert spectra.wavelength.unit == u.nm
    assert spectra.flux.unit == u.photon / (u.cm**2 * u.s * u.nm)
    assert isinstance(spectra, StarSpectra)
