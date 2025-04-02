"""Test Venot related netowrk functions."""

import numpy as np
import pytest

TEST_DATA = """1  CH3COOOH     76.054  2  4  3  0
  2  C4H9O        79.116  4  9  1  0
  3  C3H7O        59.089  3  7  1  0
  4  NO3          62.007  0  0  3  1
  5  CH3COOO      75.046  2  3  3  0
  6  C2H5OO       61.062  2  5  2  0
  7  C2H4OOH      61.062  2  5  2  0
  8  HONO2        63.015  0  1  3  1
  9  C2H5OOH      62.070  2  6  2  0
 10  CH3ONO       61.042  1  3  2  1
 11  C3H8CO       72.108  4  8  1  0
 12  CH3NO2       61.042  1  3  2  1
 13  1C4H9        57.116  4  9  0  0
 14  2C4H9        57.116  4  9  0  0
 15  C4H10        58.124  4 10  0  0
 16  C3H7OH       60.097  3  8  1  0
 17  CH3OO        47.035  1  3  2  0
 18  C4H8Y        56.108  4  8  0  0
 19  CH3OOH       48.043  1  4  2  0
 20  HNO2         47.015  0  1  2  1
 21  CH3OCO       59.046  2  3  2  0
 22  C2H5CHO      58.081  3  6  1  0
 23  C2H6CO       58.081  3  6  1  0
 24  C2H5O        45.062  2  5  1  0
 25  CH3NO        45.042  1  3  1  1
 26  2C2H4OH      45.062  2  5  1  0
 27  NO2          46.007  0  0  2  1
 28  2C3H7        43.089  3  7  0  0
 29  1C3H7        43.089  3  7  0  0
 30  1C2H4OH      45.062  2  5  1  0
 31  HONO         47.015  0  1  2  1
 32  C3H8         44.097  3  8  0  0
 33  HCNN         41.033  1  1  0  2
 34  cC2H4O       44.054  2  4  1  0
 35  HCNO         43.026  1  1  1  1
 36  C2H5OH       46.070  2  6  1  0
 37  N2O          44.014  0  0  1  2
 38  C2H3CHOZ     56.065  3  4  1  0
 39  OOH          33.008  0  1  2  0
 40  CH2CHO       43.046  2  3  1  0
 41  H2O2         34.016  0  2  2  0
 42  CH3CO        43.046  2  3  1  0
 43  NCO          42.018  1  0  1  1
 44  CH3O         31.035  1  3  1  0
 45  O2           32.000  0  0  2  0
 46  CH3CHO       44.054  2  4  1  0
 47  HNO          31.015  0  1  1  1
 48  C            12.011  1  0  0  0
 49  CHCO         41.030  2  1  1  0
 50  CO2H         45.019  1  1  2  0
 51  HOCN         43.026  1  1  1  1
 52  C2H5         29.062  2  5  0  0
 53  C2H          25.030  2  1  0  0
 54  CH2OH        31.035  1  3  1  0
 55  CH           13.019  1  1  0  0
 56  C2H6         30.070  2  6  0  0
 57  C2H3         27.046  2  3  0  0
 58  CH2CO        42.038  2  2  1  0
 59  NNH          29.022  0  1  0  2
 60  H2CN         28.034  1  2  0  1
 61  CH3OH        32.043  1  4  1  0
 62  N4S          14.007  0  0  0  1
 63  N2D          14.007  0  0  0  1
 64  CN           26.018  1  0  0  1
 65  1CH2         14.027  1  2  0  0
 66  HNCO         43.026  1  1  1  1
 67  NO           30.007  0  0  1  1
 68  O3P          16.000  0  0  1  0
 69  O1D          16.000  0  0  1  0
 70  C2H4         28.054  2  4  0  0
 71  NH           15.015  0  1  0  1
 72  3CH2         14.027  1  2  0  0
 73  HCO          29.019  1  1  1  0
 74  C2H2         26.038  2  2  0  0
 75  H2CO         30.027  1  2  1  0
 76  NH2          16.023  0  2  0  1
 77  CO2          44.011  1  0  2  0
 78  OH           17.008  0  1  1  0
 79  CH3          15.035  1  3  0  0
 80  HCN          27.026  1  1  0  1
 81  NH3          17.031  0  3  0  1
 82  CH4          16.043  1  4  0  0
 83  N2           28.014  0  0  0  2
 84  CO           28.011  1  0  1  0
 85  H2O          18.016  0  2  1  0
 86  H             1.008  0  1  0  0
 87  He            4.003  0  0  0  0
 88  H2            2.016  0  2  0  0
 89  N2O4         92.014  0  0  4  2
 90  N2O3         76.014  0  0  3  2
 91  N2H2         30.030  0  2  0  2
 92  N2H3         31.038  0  3  0  2
 93  N2H4         32.046  0  4  0  2
 94  HNNO         45.022  0  1  1  2
 95  HNOH         32.023  0  2  1  1
 96  HNO3         63.015  0  1  3  1
 97  NH2OH        33.031  0  3  1  1
 98  H2NO         32.023  0  2  1  1
 99  CNN          40.025  1  0  0  2
100  H2CNO        44.034  1  2  1  1
101  C2N2         52.036  2  0  0  2
102  HCNH         28.034  1  2  0  1
103  HNC          27.026  1  1  0  1
104  HON          31.015  0  1  1  1
105  NCN          40.025  1  0  0  2
106  HCOH         30.027  1  2  1  0
107  HOCHO        46.027  1  2  2  0
108  HOCH2O       47.035  1  3  2  0
109  H2Oc         18.016  0  2  1  0
110  CH4c         16.043  1  4  0  0
111  NH3c         17.031  0  3  0  1"""

expected = []
for line in TEST_DATA.split("\n"):
    _, species, mass, num_c, num_h, num_o, num_n = line.split()
    mass = float(mass)
    expected.append((species, mass, int(num_c), int(num_h), int(num_o), int(num_n)))


@pytest.mark.parametrize("species, expected_mass, num_c, num_h, num_o, num_n", expected)
def test_decode_species_num_atoms(species, expected_mass, num_c, num_h, num_o, num_n):
    """Test that the number of atoms in a species is decoded correctly."""
    from freckll.venot.io import _decode_species

    decoded = _decode_species(species)

    composition = decoded.composition().asdict()

    assert composition.get("C", [0])[0] == num_c
    assert composition.get("H", [0])[0] == num_h
    assert composition.get("O", [0])[0] == num_o
    assert composition.get("N", [0])[0] == num_n

    if species in ("H2Oc", "NH3c", "CH4c"):
        assert decoded.state == "liquid"


@pytest.mark.parametrize("species, expected_mass, num_c, num_h, num_o, num_n", expected)
def test_decode_species_mass(species, expected_mass, num_c, num_h, num_o, num_n):
    from freckll.venot.io import _decode_species

    decoded = _decode_species(species)
    if species == "C4H9O":
        expected_mass = 73.06
    assert decoded.monoisotopic_mass == pytest.approx(expected_mass, rel=1e-3)


def test_parse_nasa_lines():
    """Test parsing NASA polynomial coefficients."""
    from freckll.venot.io import _parse_nasa_lines

    line1 = "CH3COOOH             300  5000  1000"
    line2 = (
        "7.87651000e+00  1.42400000e-02 -4.81648000e-06  7.62396000e-10"
        " -4.65514000e-14 -4.39851000e+04 -9.71105000e+00"
    )
    line3 = (
        "2.19569000e-01  3.53131000e-02 -2.24276000e-05  3.22888000e-09"
        "  1.67243000e-12 -4.19548000e+04  2.96617000e+01"
    )

    a_truth = np.array([
        7.87651000e00,
        1.42400000e-02,
        -4.81648000e-06,
        7.62396000e-10,
        -4.65514000e-14,
        -4.39851000e04,
        -9.71105000e00,
    ])

    b_truth = np.array([
        2.19569000e-01,
        3.53131000e-02,
        -2.24276000e-05,
        3.22888000e-09,
        1.67243000e-12,
        -4.19548000e04,
        2.96617000e01,
    ])

    nasa = _parse_nasa_lines(line1, line2, line3)

    assert nasa.species == "CH3COOOH"
    assert nasa.x1 == 300
    assert nasa.x2 == 5000
    assert nasa.x3 == 1000
    assert nasa.a_coeff.shape == (7,)
    assert nasa.b_coeff.shape == (7,)

    np.testing.assert_allclose(nasa.a_coeff, a_truth, rtol=1e-5)
    np.testing.assert_allclose(nasa.b_coeff, b_truth, rtol=1e-5)


def test_read_nasa_file(tmp_path):
    from freckll.species import SpeciesFormula
    from freckll.venot.io import load_nasa_coeffs

    test_data = """CH3COOOH             300  5000  1000
 7.87651000e+00  1.42400000e-02 -4.81648000e-06  7.62396000e-10 -4.65514000e-14 -4.39851000e+04 -9.71105000e+00
 2.19569000e-01  3.53131000e-02 -2.24276000e-05  3.22888000e-09  1.67243000e-12 -4.19548000e+04  2.96617000e+01
C4H9O                300  5000  1000
 1.08161110e+01  2.04849930e-02 -6.12612710e-06  8.93178750e-10 -5.17843620e-14 -1.92811310e+04 -4.12957710e+01
-1.95327960e+00  5.21634150e-02 -3.10300130e-05  6.12449510e-09  7.11754390e-13 -1.55101920e+04  2.57509210e+01
C3H7O                300  5000  1000
 7.93919420e+00  1.59497130e-02 -4.72068500e-06  6.84053480e-10 -3.95399350e-14 -1.54257270e+04 -2.76003010e+01
-1.84931550e+00  4.10015100e-02 -2.54725640e-05  5.89854340e-09  2.34561960e-13 -1.26042650e+04  2.35337830e+01"""

    nasa_file = tmp_path / "nasa.dat"
    nasa_file.write_text(test_data)

    nasa_coeffs = load_nasa_coeffs(nasa_file)

    assert len(nasa_coeffs) == 3
    assert SpeciesFormula("CH3COOOH") in nasa_coeffs
    assert SpeciesFormula("C4H9O") in nasa_coeffs
    assert SpeciesFormula("C3H7O") in nasa_coeffs

    assert nasa_coeffs["CH3COOOH"].species == "CH3COOOH"
    assert nasa_coeffs["C4H9O"].species == "C4H9O"
    assert nasa_coeffs["C3H7O"].species == "C3H7O"
    assert nasa_coeffs["CH3COOOH"].x1 == 300
    assert nasa_coeffs["CH3COOOH"].x2 == 5000
    assert nasa_coeffs["CH3COOOH"].x3 == 1000

    assert nasa_coeffs["C4H9O"].x1 == 300
    assert nasa_coeffs["C4H9O"].x2 == 5000
    assert nasa_coeffs["C4H9O"].x3 == 1000

    assert nasa_coeffs["C3H7O"].x1 == 300
    assert nasa_coeffs["C3H7O"].x2 == 5000
    assert nasa_coeffs["C3H7O"].x3 == 1000

    np.testing.assert_allclose(
        nasa_coeffs["CH3COOOH"].a_coeff,
        np.array([
            7.87651000e00,
            1.42400000e-02,
            -4.81648000e-06,
            7.62396000e-10,
            -4.65514000e-14,
            -4.39851000e04,
            -9.71105000e00,
        ]),
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        nasa_coeffs["CH3COOOH"].b_coeff,
        np.array([
            2.19569000e-01,
            3.53131000e-02,
            -2.24276000e-05,
            3.22888000e-09,
            1.67243000e-12,
            -4.19548000e04,
            2.96617000e01,
        ]),
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        nasa_coeffs["C4H9O"].a_coeff,
        np.array([
            1.08161110e01,
            2.04849930e-02,
            -6.12612710e-06,
            8.93178750e-10,
            -5.17843620e-14,
            -1.92811310e04,
            -4.12957710e01,
        ]),
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        nasa_coeffs["C4H9O"].b_coeff,
        np.array([
            -1.95327960e00,
            5.21634150e-02,
            -3.10300130e-05,
            6.12449510e-09,
            7.11754390e-13,
            -1.55101920e04,
            2.57509210e01,
        ]),
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        nasa_coeffs["C3H7O"].a_coeff,
        np.array([
            7.93919420e00,
            1.59497130e-02,
            -4.72068500e-06,
            6.84053480e-10,
            -3.95399350e-14,
            -1.54257270e04,
            -2.76003010e01,
        ]),
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        nasa_coeffs["C3H7O"].b_coeff,
        np.array([
            -1.84931550e00,
            4.10015100e-02,
            -2.54725640e-05,
            5.89854340e-09,
            2.34561960e-13,
            -1.26042650e04,
            2.35337830e01,
        ]),
        rtol=1e-5,
    )


def test_load_composes_file(tmp_path):
    from freckll.species import SpeciesFormula
    from freckll.venot.io import load_composition

    composes_file = tmp_path / "composes.dat"
    composes_file.write_text(TEST_DATA)

    species = load_composition(composes_file)
    # Check first is okay
    assert species[0] == SpeciesFormula("CH3COOOH")
    # Check last is condensed phase
    assert species[-1].formula == "H3N"
    assert species[-1].state == "liquid"
    # Check isomer is okay
    assert species[37] == SpeciesFormula("C2H3CHOZ", true_formula="C3H4O")
    assert len(species) == 111


def test_parse_reaction_line():
    from freckll.species import SpeciesFormula

    REACTION_LINES = """ HCNO                                                   HCN        O3P                                          4.200E+31 -6.120E+00  3.077E+04  1.100E+00  0.000E+00
 HCNH                                                   HCN        H                                            6.100E+28 -5.690E+00  1.220E+04  1.100E+00  0.000E+00
"""
    from freckll.venot.io import _parse_reaction_line

    reactants, products, coeffs = _parse_reaction_line(REACTION_LINES.split("\n")[0])

    assert reactants == [SpeciesFormula("HCNO")]
    assert products == [SpeciesFormula("HCN"), SpeciesFormula("O3P", true_formula="O")]
    np.testing.assert_allclose(coeffs, [4.200e31, -6.120e00, 3.077e04, 1.100e00, 0.000e00])

    reactants, products, coeffs = _parse_reaction_line(REACTION_LINES.split("\n")[1])

    assert reactants == [SpeciesFormula("HCNH")]
    assert products == [SpeciesFormula("HCN"), SpeciesFormula("H")]
    np.testing.assert_allclose(coeffs, [6.100e28, -5.690e00, 1.220e04, 1.100e00, 0.000e00])

    REACTION_LINE = """ CH2CHO                                                 CH3CO                                                   1.000E+13  0.000E+00  2.363E+04  1.260E+00  0.000E+00"""

    reactants, products, coeffs = _parse_reaction_line(REACTION_LINE)

    assert reactants == [SpeciesFormula("CH2CHO", true_formula="C2H3O")]
    assert products == [SpeciesFormula("CH3CO")]
    np.testing.assert_allclose(coeffs, [1.000e13, 0.000e00, 2.363e04, 1.260e00, 0.000e00])


def test_build_efficiencies():
    from freckll.venot.io import build_efficienies

    effi_coeffs = np.array([
        0.4,
        0.75,
        1.5,
        6.5,
        3.0,
        1.0,
        3.0,
        0.35,
        0.4,
        0.35,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    effi_index = np.array([6, 34, 27, 35, 32, 38, 11, -1, 33, 37, 21, -1, -1, -1, -1, 20, 31, -1, -1, -1, -1, -1])

    effi_species = np.zeros(47)

    efficiency = build_efficienies(effi_species, effi_coeffs, effi_index)

    expected = np.array([
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.4,
        1.0,
        1.0,
        1.0,
        1.0,
        3.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.5,
        1.0,
        1.0,
        1.0,
        1.0,
        3.0,
        0.4,
        0.75,
        6.5,
        1.0,
        0.35,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    np.testing.assert_allclose(efficiency, expected)
