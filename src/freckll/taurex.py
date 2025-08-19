"""Module for the TauREx plugin"""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.io.typing import PathLike
from taurex.chemistry import AutoChemistry

import freckll.io.loader as freckll_loader
from freckll.solver import Rosenbrock

DEFAULT_ELEMENTS = ("C", "N", "O")
DEFAULT_ABUNDANCES = (8.39, 7.86, 8.73)


class FreckllChemistry(AutoChemistry):
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
        solve_method: str = "rosenbrock",
        t_span: tuple[float, float] = (0.0, 1e10),
        max_iter: int = 100,
        nevals: int = 200,
        dn_crit: float = 1e-3,
        dndt_crit: float = 1e-6,
        max_solve_time: str = "1 hour",
        enable_diffusion: bool = False,
        rtol: float = 1e-2,
        atol: float = 1e-15,
        maxiter: int = 1000,
        **kwargs: t.Any,
    ) -> None:
        """Initialize the FreckllChemistry.


        Currently only Rosenbrock solver is supported.


        Args:
            network: The chemical network to use.
            photochemistry: The photochemistry data to use.
            thermochemistry: The thermochemistry data to use.
            elements: The elements for the thermochemistry.
            abundances: The initial abundances for the thermochemistry.
            ratio_element: The element to use for the ratio (default is "O").
            h_abundance: The abundance of hydrogen (default is 12.0).
            h_he_ratio: The ratio of hydrogen to helium (default is 0.083).
            metallicity: The metallicity of the system (default is 1.0).
            solve_method: The method to use for solving the chemistry (default is "rosenbrock").
            t_span: The time span for the simulation (default is (0.0, 1e10)).
            max_iter: The maximum number of iterations for the solver (default is 100).
            nevals: The number of evaluations to perform (default is 200).
            dn_crit: The criteria for convergence in number density (default is 1e-3).
            dndt_crit: The criteria for convergence in number density change (default is 1e-6).
            max_solve_time: The maximum time allowed for the solver (default is "1 hour").
            enable_diffusion: Whether to enable diffusion in the solver (default is False).
            rtol: The relative tolerance for the solver (default is 1e-2).
            atol: The absolute tolerance for the solver (default is 1e-15).
            maxiter: The maximum number of iterations for the solver (default is 1000).
            kwargs: ratio values. e,g (C_ratio=0.5) for C/<ratio_element> ratio.

        """
        super().__init__("FreckllChemistry")

        self.ratio_element = ratio_element

        self._elements = elements
        self._abundances = abundances

        self.h_abundance = h_abundance
        self.he_h_ratio = h_he_ratio
        self._metallicity = metallicity

        metal_elements, metal_abundances = zip(*[
            (ele, abu) for ele, abu in zip(self._elements, self._abundances) if ele not in ["H", "He", ratio_element]
        ])

        self.metal_elements = metal_elements
        self.metal_abundances = metal_abundances

        self.ratio_abundance = self._abundances[self._elements.index(self.ratio_element)]

        self._ratios: np.ndarray = 10 ** (np.array(self.metal_abundances) - self.ratio_abundance)

        self.species = self._species()
        self.determine_active_inactive()
        self.add_ratio_params()
        for key, value in kwargs.items():
            if key in self._ratio_setters:
                self.info(f"Setting {key} to {value}")
                self._ratio_setters[key](self, value)

        self.network = freckll_loader.default_network_loader(network)
        self.photochemistry = None
        if photochemistry == "auto" and "network" == "venot-methanol-2020":
            self.photochemistry = freckll_loader.default_photonetwork_loader(self.network.species)

        self.mu_profile = None
        self.mix_profile = None

        self._only_gases_mask = np.array([s.state == "gas" for s in self.network.species])

        self.solver_args = {
            "t_span": t_span,
            "max_iter": max_iter,
            "nevals": nevals,
            "dn_crit": dn_crit,
            "dndt_crit": dndt_crit,
            "max_solve_time": max_solve_time,
            "enable_diffusion": enable_diffusion,
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
        }

        self.solver = Rosenbrock(
            self.network,
            self.photochemistry,
        )

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
            *ratio_abund,
            *list(metals),
        ])

        return ["H", "He", *self.ratio_element, *list(self.metal_elements)], complete

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

        temperature = temperature_profile << u.K
        pressure = pressure_profile << u.Pa

        vmr = ace_equil_chemistry_loader(
            self.network.species,
            temperature,
            pressure,
            elements=elements,
            abundances=abundances,
        )
        self.network.compile_reactions(temperature, pressure)

        self.result = self.solver.solve(
            vmr,
            **self.solver_args,
        )
        self.final_vmr = self.result["vmr"]
        self.mu_profile = (
            np.sum(self.final_vmr[self._only_gases_mask] * self.species_masses[self._only_gases_mask], axis=0)
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
        return self.result["mu_profile"][self._only_gases_mask]
