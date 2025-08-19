"""Module for the TauREx plugin"""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.io.typing import PathLike
from taurex.chemistry import AutoChemistry
from taurex.core import fitparam
from taurex.data.profiles.temperature.temparray import TemperatureArray
from taurex.planet import Planet
from taurex.pressure import ArrayPressureProfile
from taurex.stellar import Star

import freckll.io.loader as freckll_loader
from freckll.io import dispatcher as freckll_dispatcher
from freckll.reactions.photo import StarSpectra
from freckll.solver import Rosenbrock

DEFAULT_ELEMENTS = ("C", "N", "O")
DEFAULT_ABUNDANCES = (8.39, 7.86, 8.73)


DEFAULT_EQUIL_CONFIG = {
    "format": "ace",
    "abundances": ("H", "He", *DEFAULT_ELEMENTS),
    "elements": (12.0, 10.92, *DEFAULT_ABUNDANCES),
}

DEFAULT_SOLVER_CONFIG = {
    "method": "rosenbrock",
    "t_span": (0.0, 1e10),
    "max_iter": 100,
    "nevals": 200,
    "dn_crit": 1e-3,
    "dndt_crit": 1e-6,
    "max_solve_time_hour": 99.0,
    "enable_diffusion": False,
    "rtol": 1e-2,
    "atol": 1e-15,
    "maxiter": 1000,
}


def resolve_star(star_string):
    """Resolves the star into the appropriate SED method."""
    freckll_loader.star_spectra_loader()


class BaseFreckllChemistry(AutoChemistry):
    """Chemistry class for the Disequilibrium."""

    def __init__(
        self,
        network_config: t.Union[str, dict[str, t.Any]] = "venot-methanol-2020",
        photochemistry_config: t.Optional[t.Union[str, dict[str, t.Any]]] = None,
        initial_equilibrium_config: t.Optional[dict] = None,
        star_config: t.Optional[t.Union[dict[str, t.Any]]] = None,
        solver_config: t.Optional[dict[str, t.Any]] = None,
        kzz: t.Union[u.Quantity, dict] = 1e10 * u.cm**2 / u.s,
        ratio_element: str = "O",
        metallicity: float = 1.0,
        override_star_params_taurex: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """Initialize the FreckllChemistry.


        Currently only Rosenbrock solver is supported.


        Args:
            network_config: The chemical network configuration.
            photochemistry_config: The photochemistry configuration.
            initial_equilibrium_config: Initial equilibrium configuration.
            star_config: Star configuration.
            solver_config: Solver configuration.
            kzz: Eddy diffusion coefficient.
            ratio_element: Element to use for the C/O ratio.
            metallicity: Metallicity of the atmosphere.
            override_star_params_taurex: Whether to override the star parameters in TauREx.
            **kwargs: Additional keyword arguments.

        """
        super().__init__("FreckllChemistry")

        initial_equilibrium_config = initial_equilibrium_config or DEFAULT_EQUIL_CONFIG.copy()

        self.ratio_element = ratio_element

        self._elements = initial_equilibrium_config["elements"]
        self._abundances = initial_equilibrium_config["abundances"]

        self.h_abundance = self._abundances[self._elements.index("H")]
        self.he_h_ratio = 10 ** (self._abundances[self._elements.index("He")] - self.h_abundance)
        self._metallicity = metallicity

        metal_elements, metal_abundances = zip(*[
            (ele, abu) for ele, abu in zip(self._elements, self._abundances) if ele not in ["H", "He", ratio_element]
        ])

        self.metal_elements = metal_elements
        self.metal_abundances = metal_abundances

        self.ratio_abundance = self._abundances[self._elements.index(self.ratio_element)]

        self._ratios: np.ndarray = 10 ** (np.array(self.metal_abundances) - self.ratio_abundance)

        network = freckll_dispatcher.dispatch_network(network_config)

        self.network = network
        self.photochemistry = None
        if photochemistry_config:
            self.photochemistry = freckll_dispatcher.dispatch_photochemistry(
                photochemistry_config,
                species_list=network.species,
            )

        self.mu_profile = None
        self.mix_profile = None
        self.species = [s.true_formula for s in self.network.species]
        self._only_gases_mask = np.array([s.state == "gas" for s in self.network.species])

        self.solver_args = solver_config or DEFAULT_SOLVER_CONFIG.copy()

        self.solver = Rosenbrock(
            self.network,
            self.photochemistry,
        )

        self.star = None
        self.planet = None

        self.determine_active_inactive()
        self.add_ratio_params()
        for key, value in kwargs.items():
            if key in self._ratio_setters:
                self.info(f"Setting {key} to {value}")
                self._ratio_setters[key](self, value)

        self.kzz_arr = kzz
        self._kzz = np.max(kzz)
        self.star_config = star_config
        self.override_star_params_taurex = override_star_params_taurex

    def add_ratio_params(self):
        self._ratio_setters = {}
        for idx, element in enumerate(self.metal_elements):
            if element == self.ratio_element:
                continue
            param_name = f"{element}_{self.ratio_element}_ratio"
            param_tex = f"{element}/{self.ratio_element}"

            def read_mol(self, idx=idx):
                return self._ratios[idx]

            def write_mol(self, value, idx=idx):
                self._ratios[idx] = value

            read_mol.__doc__ = f"Equilibrium {element}/{self.ratio_element} ratio."
            write_mol.__doc__ = f"Equilibrium {element}/{self.ratio_element} ratio."
            fget = read_mol
            fset = write_mol

            bounds = [1.0e-12, 0.1]

            default_fit = False
            self._ratio_setters[f"{element}_ratio"] = fset
            self.add_fittable_param(param_name, param_tex, fget, fset, "log", default_fit, bounds)

    def generate_elements_abundances(self):
        """Generates elements and abundances to pass into ace."""
        import math

        ratios = np.log10(self._ratios)
        ratio_abund = math.log10(self._metallicity * (10 ** (self.ratio_abundance - 12))) + 12

        metals = ratio_abund + ratios

        complete = np.array([
            self.h_abundance,
            self.h_abundance + math.log10(self.he_h_ratio),
            ratio_abund,
            *list(metals),
        ])

        return ["H", "He", self.ratio_element, *list(self.metal_elements)], complete

    @property
    def gases(self) -> list[str]:
        """Finds species in the ACE database."""
        return [s.true_formula for s in self.network.species if s.state == "gas"]

    @property
    def liquids(self) -> list[str]:
        """Finds species in the ACE database."""
        return [s.true_formula for s in self.network.species if s.state == "liquid"]

    @property
    def solids(self) -> list[str]:
        """Finds species in the ACE database."""
        return [s.true_formula for s in self.network.species if s.state == "solid"]

    @property
    def species_masses(self) -> u.Quantity[u.u]:
        """Returns the molecular weights of the species in the network."""
        return np.array([s.monoisotopic_mass for s in self.network.species]) << u.u

    def set_star_planet(self, star: Star, planet: Planet) -> None:
        self.star_taurex = star
        self.planet = planet
        self.star = None
        if self.star_config:
            if (
                isinstance(self.star_config, dict)
                and self.star_config["spectrum"]["format"] == "rescale"
                and self.override_star_params_taurex
            ):
                self.star_config["spectrum"]["radius"] = self.star_taurex.radius << u.m
                self.star_config["spectrum"]["temperature"] = self.star_taurex.temperature << u.K
            self.incident_angle = self.star_config.get("incident_angle", 45.0 * u.deg)

            self.star = freckll_dispatcher.dispatch_star(
                self.star_config,
            )

    def initialize_chemistry(
        self,
        nlayers=100,
        temperature_profile=None,
        pressure_profile=None,
        altitude_profile=None,
    ):
        """Initializes the chemistry.

        Args:
            nlayers: Number of layers.
            temperature_profile: Temperature profile.
            pressure_profile: Pressure profile.
            altitude_profile: Altitude profile. (Deprecated)

        """
        from freckll.io.loader import ace_equil_chemistry_loader

        elements, abundances = self.generate_elements_abundances()
        star = self.star
        if self.star_config is None and self.photochemistry is not None:
            # Since this will come before the TauREx star is initialized we can initialize it here.
            wl = np.arange(1, 901) << u.nm

            wn = wl.to(u.k, equivalencies=u.spectral()).value

            self.star_taurex.initialize(wn)

            # Generate a star spectra from TauREx
            sed = (self.star_taurex.sed) << u.W / (u.m**2 * u.micron)
            star = freckll_dispatcher.StarData(
                spectra=StarSpectra(
                    wavelength=wl,
                    spectra=sed[::-1],
                    reference_distance=self.star_taurex.radius << u.m,
                    incident_angle=self.incident_angle,
                ),
                incident_angle=self.incident_angle,
            )

            # Scale the star spectra to the star radius

        temperature = temperature_profile << u.K
        pressure = pressure_profile << u.Pa

        vmr = ace_equil_chemistry_loader(
            species=self.network.species,
            temperature=temperature,
            pressure=pressure,
            elements=elements,
            abundances=abundances,
        )
        self.network.compile_reactions(temperature, pressure)
        if self.photochemistry:
            self.photochemistry.set_spectra(
                spectra=star.spectra,
                distance=self.planet.get_planet_semimajoraxis("AU") << u.AU,
                incident_angle=self.incident_angle,
            )
            self.photochemistry.compile_chemistry(
                temperature=temperature,
                pressure=pressure,
            )
        kzz_value = self._kzz * np.ones(pressure.shape) if self.kzz_arr.ndim == 0 else self.kzz_arr

        self.solver.set_system_parameters(
            temperature=temperature,
            pressure=pressure,
            kzz=kzz_value,
            planet_radius=self.planet.get_planet_radius("Rjup") << u.R_jup,
            planet_mass=self.planet.get_planet_mass("Mjup") << u.M_jup,
        )

        self.result = self.solver.solve(
            vmr,
            **self.solver_args,
        )
        self.final_vmr = self.result["vmr"][-1]
        self.mu_profile = (
            np.sum(self.final_vmr[self._only_gases_mask] * self.species_masses[self._only_gases_mask, None], axis=0)
            .to(u.kg)
            .value
        )

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:
        """Mixing profile (VMR)."""
        return self.final_vmr[self._only_gases_mask]

    @property
    def muProfile(self) -> npt.NDArray[np.float64]:
        """Mean molecular weight profile in kg"""
        return self.mu_profile

    @fitparam(
        param_name="metallicity",
        param_latex="Z",
        default_bounds=[0.2, 2.0],
        default_fit=False,
    )
    def metallicity(self):
        """Metallicity of the atmosphere."""
        return self._metallicity

    @metallicity.setter
    def metallicity(self, value):
        """Metallicity of the atmosphere."""
        self._metallicity = value

    @fitparam(
        param_name="Kzz",
        param_latex=r"K_{zz}",
        default_bounds=[1e8, 1e12],
        default_fit=False,
    )
    def Kzz(self) -> float:
        """Kzz value for the atmosphere."""
        # If its not scalar then raise an error
        if not np.isscalar(self.kzz_arr):
            raise ValueError("Kzz must be a scalar value to use fitting property.")  # noqa: TRY003
        return self._kzz.to(u.cm**2 / u.s).value

    @Kzz.setter
    def Kzz(self, value: float):
        """Kzz value for the atmosphere."""
        self._kzz = value << u.cm**2 / u.s

    BIBTEX_ENTRIES = (
        r"""
@ARTICLE{2024ApJ...967..132A,
       author = {{Al-Refaie}, Ahmed Faris and {Venot}, Olivia and {Changeat}, Quentin and {Edwards}, Billy},
        title = "{FRECKLL: Full and Reduced Exoplanet Chemical Kinetics DistiLLed}",
      journal = {\apj},
     keywords = {Astrochemistry, Exoplanet atmospheres, Exoplanet atmospheric dynamics, Chemical kinetics, Chemical reaction network models, Chemical abundances, Exoplanet atmospheric structure, 75, 487, 2307, 2233, 2237, 224, 2310, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Physics - Chemical Physics},
         year = 2024,
        month = jun,
       volume = {967},
       number = {2},
          eid = {132},
        pages = {132},
          doi = {10.3847/1538-4357/ad3dee},
archivePrefix = {arXiv},
       eprint = {2209.11203},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...967..132A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    )


class FreckllChemistry(BaseFreckllChemistry):
    pass


def build_star_config(
    string: str,
) -> t.Union[dict[str, t.Any], str]:
    """Builds the star configuration from a string."""

    if string in t.get_args(freckll_loader.Stars):
        return {"spectrum": string}
    elif string.startswith("rescale-"):
        return {
            "spectrum": {
                "format": "rescale",
                "from_star": string.split("rescale-")[1],
            }
        }
    elif string.startswith("from-taurex"):
        return None


class FreckllChemistryInput(BaseFreckllChemistry):
    """Chemistry class for the Disequilibrium."""

    def __init__(
        self,
        network: t.Union[PathLike, freckll_loader.Networks] = "venot-methanol-2020",
        photochemistry: t.Optional[t.Union[PathLike, freckll_loader.Photonetworks, t.Literal["auto"]]] = "auto",
        elements: tuple[str] = DEFAULT_ELEMENTS,
        abundances: tuple[float] = DEFAULT_ABUNDANCES,
        ratio_element: str = "O",
        h_abundance: float = 12.0,
        h_he_ratio: float = 0.083,
        metallicity: float = 1.0,
        star_method: str = "rescale-sun",
        solve_method: str = "rosenbrock",
        t_span: tuple[float, float] = (0.0, 1e10),
        max_iter: int = 100,
        nevals: int = 200,
        dn_crit: float = 1e-3,
        dndt_crit: float = 1e-6,
        max_solve_time_hour: float = 1,
        enable_diffusion: bool = False,
        rtol: float = 1e-2,
        atol: float = 1e-15,
        maxiter: int = 1000,
        **kwargs: t.Any,
    ) -> None:
        """Initialize the FreckllChemistry.

        Args:


        """
        network_config = network
        if photochemistry == "auto" and network_config == "venot-methanol-2020":
            photochemistry = "venot-methanol-2020-photo"
        elif photochemistry == "auto" and network_config == "venot-methanol-2020-reduced":
            photochemistry = None
        photochemistry_config = photochemistry

        he_dex = np.log10(h_he_ratio) + h_abundance

        elements = elements or DEFAULT_ELEMENTS
        abundances = abundances or DEFAULT_ABUNDANCES

        elements = ("H", "He", *tuple(elements))
        abundances = (h_abundance, he_dex, *tuple(abundances))

        equil_config = {
            "format": "ace",
            "elements": elements,
            "abundances": abundances,
        }

        solver_config = {
            "method": solve_method,
            "t_span": t_span,
            "max_iter": max_iter,
            "nevals": nevals,
            "dn_crit": dn_crit,
            "dndt_crit": dndt_crit,
            "max_solve_time_hour": max_solve_time_hour,
            "enable_diffusion": enable_diffusion,
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
        }

        star_config = build_star_config(star_method)

        super().__init__(
            network_config=network_config,
            photochemistry_config=photochemistry_config,
            initial_equilibrium_config=equil_config,
            star_config=star_config,
            solver_config=solver_config,
            kzz=kwargs.get("kzz", 1e10 * u.cm**2 / u.s),
            ratio_element=ratio_element,
            metallicity=metallicity,
            incident_angle=kwargs.get("incident_angle", 45.0 * u.deg),
            override_star_params_taurex=True,  # Should be True for input file format
            **kwargs,
        )

    @classmethod
    def input_keywords(cls):
        return [
            "freckll",
        ]


class FreckllChemistryFile(BaseFreckllChemistry):
    """Chemistry class for the Disequilibrium."""

    def __init__(self, input_file: PathLike, ratio_element: str, metallicity: float, kzz: u.Quantity) -> None:
        """Initialize the FreckllChemistryFile."""
        from freckll.io.yaml import load_yaml

        input_data = load_yaml(input_file)

        if kzz is None:
            atmosphere_data = freckll_dispatcher.dispatch_atmosphere(
                input_data["atmosphere"],
            )
            kzz = atmosphere_data.kzz

        super().__init__(
            network_config=input_data["network"],
            photochemistry_config=input_data.get("photochemistry"),
            initial_equilibrium_config=input_data.get("thermochemistry", {}),
            star_config=input_data.get("star"),
            solver_config=input_data.get("solver", {}),
            kzz=kzz,
            ratio_element=ratio_element,
            metallicity=metallicity,
            incident_angle=input_data.get("incident_angle", 45.0 * u.deg),
            override_star_params_taurex=True,  # Should be True for input file format
        )

    @classmethod
    def input_keywords(cls):
        return [
            "freckll-file",
        ]


def create_taurex_objects_from_file(
    input_file: PathLike,
    ratio_element: str = "O",
    metallicity: float = 1.0,
) -> tuple[FreckllChemistryFile, Planet, TemperatureArray, ArrayPressureProfile]:
    """Create all consistent TauREx objects from a YAML input file.

    Args:
        input_file: The path to the input file.
        ratio_element: The element to use for the C/O ratio.
        metallicity: The metallicity of the atmosphere.
    Returns:
        A tuple of the FreckllChemistryFile, Planet, TemperatureArray, and ArrayPressureProfile objects.


    """
    from freckll.io.yaml import load_yaml_from_file

    input_data = load_yaml_from_file(input_file)
    atmosphere_data = freckll_dispatcher.dispatch_atmosphere(
        input_data["atmosphere"],
    )

    kzz = atmosphere_data.kzz

    pressure_profile = ArrayPressureProfile(
        atmosphere_data.pressure.to(u.Pa).value,
    )
    temperature_profile = TemperatureArray(
        atmosphere_data.temperature.to(u.K).value,
        pressure_profile=atmosphere_data.pressure.to(u.Pa).value,
    )

    planet = freckll_dispatcher.dispatch_planet(input_data["planet"])

    taurex_planet = Planet(
        planet_mass=planet.mass.to(u.Mjup).value,
        planet_radius=planet.radius.to(u.Rjup).value,
        planet_sma=planet.distance.to(u.AU).value,
    )

    freckll_chemistry = FreckllChemistryFile(
        input_file=input_file,
        ratio_element=ratio_element,
        metallicity=metallicity,
        kzz=kzz,
    )

    return (
        freckll_chemistry,
        taurex_planet,
        temperature_profile,
        pressure_profile,
    )
