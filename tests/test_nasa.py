"""Test NASA polynomial related functions."""

import numpy as np


def test_nasa_coeffs():
    """Test NASA polynomial coefficients."""
    from freckll.nasa import NasaCoeffs

    a_coeff = np.array([
        7.87651000e00,
        1.42400000e-02,
        -4.81648000e-06,
        7.62396000e-10,
        -4.65514000e-14,
        -4.39851000e04,
        -9.71105000e00,
    ])

    b_coeff = np.array([
        2.19569000e-01,
        3.53131000e-02,
        -2.24276000e-05,
        3.22888000e-09,
        1.67243000e-12,
        -4.19548000e04,
        2.96617000e01,
    ])

    assert b_coeff.shape == a_coeff.shape

    nasa = NasaCoeffs("CH3COOOH", 300, 5000, 1000, a_coeff, b_coeff)
    temperature_below = np.array([200, 400, 600, 800])
    temperature_above = np.array([1200, 1400, 1600, 1800])

    h_coeffs_below, s_coeffs_below = nasa(temperature_below)
    h_coeffs_above, s_coeffs_above = nasa(temperature_above)

    # MAke sure they're not the same
    assert not np.allclose(h_coeffs_below, h_coeffs_above)
    assert not np.allclose(s_coeffs_below, s_coeffs_above)

    # Concantenate the temperatures
    temperature = np.concatenate([temperature_below, temperature_above])
    h_coeffs, s_coeffs = nasa(temperature)
    # Now check the coefficients are being correctly calculated on the right portion.
    np.testing.assert_allclose(h_coeffs_below, h_coeffs[:4])
    np.testing.assert_allclose(s_coeffs_below, s_coeffs[:4])
    np.testing.assert_allclose(h_coeffs_above, h_coeffs[4:])
    np.testing.assert_allclose(s_coeffs_above, s_coeffs[4:])
