"""Handles solving of ODE."""
from astropy import units as u

from ..types import FreckllArray

from typing import TypedDict, Optional, Callable
from ..network import ChemicalNetwork, PhotoChemistry
from scipy.sparse import spmatrix
from ..log import Loggable
from ..reactions.photo import StarSpectra
from ..species import SpeciesFormula
import numpy as np
from functools import partial
from collections import deque

class StellarFluxData(TypedDict):
    wavelength: u.Quantity
    incident_flux: u.Quantity
    incident_angle: u.Quantity

class PlanetOutputData(TypedDict):
    """TypedDict for planet output data."""
    
    radius: u.Quantity
    mass: u.Quantity
    distance: Optional[u.Quantity] = None
    albedo: Optional[float] = None


class Solution(TypedDict):
    """TypedDict for solver output."""
    
    success: bool
    initial_vmr: FreckllArray
    num_jacobian_evals: int
    num_dndt_evals: int
    temperature: u.Quantity
    pressure: u.Quantity
    density: u.Quantity
    vmr: FreckllArray
    kzz: u.Quantity
    planet: PlanetOutputData
    times: FreckllArray
    species: list[SpeciesFormula]
    masses: u.Quantity
    vmr: FreckllArray
    stellar_flux: Optional[StellarFluxData] = None


class SolverOutput(TypedDict):
    """TypedDict for solver output."""
    
    num_jac_evals: int
    num_dndt_evals: int
    success: bool
    times: FreckllArray
    y: FreckllArray

DyCallable=Callable[[float, FreckllArray], FreckllArray]
JacCallable=Callable[[float, FreckllArray], spmatrix]


def convergence_test(
    ys: list[FreckllArray],
    t: list[float],
    y0: FreckllArray,
    logger: Loggable,
    atol: float = 1e-25,
    df_criteria: float = 1e-3,
    dfdt_criteria: float = 1e-8,
) -> bool:
    """Test for convergence of the solution.

    Args:
        ys: The solution.
        t: The time points.
        y0: The initial conditions.
        logger: The logger.
        atol: The absolute tolerance.
        df_criteria: The criteria for convergence.
        dfdt_criteria: The criteria for convergence.

    Returns:
        True if the solution converged, False otherwise.

    """
    if len(ys) < 2:
        return False
    
    current_y = np.copy(ys[-1])
    previous_y = np.copy(ys[-2])

    #dy = np.abs(current_y - previous_y)/ys[-1]
    dy = np.abs(current_y - previous_y)/current_y

    dy[current_y < atol] = 0

    dy = np.amax(np.abs(dy))
    dydt = dy/(t[-1] - t[-2])

    logger.info("Convergence test: dy: %4.2E, dydt: %4.2E", dy, dydt)

    return dy < df_criteria and dydt < dfdt_criteria
    



def output_step(t,y, logger:Loggable=None) -> bool:
    """Output the current step of the solver.

    Args:
        t: The time.
        y: The solution.
        logger: The logger.
    """
    import math
    final = y
    logger.info("ydot sum/min/max: %4.2E / %4.2E / %4.2E Time/log(Time): %4.2E / %4.2E",
                final.sum(), final.min(), final.max(), t,math.log10(t))
    



def dndt(t: float, y: FreckllArray,
            density: Optional[u.Quantity] = None,
            temperature: Optional[u.Quantity] = None,
            pressure: Optional[u.Quantity] = None,
            kzz: Optional[u.Quantity] = None,
            planet_radius: Optional[u.Quantity] = None,
            planet_mass: Optional[u.Quantity] = None,
            masses: Optional[u.Quantity] = None,
            network: Optional[ChemicalNetwork] = None,
            photochemistry: Optional[Optional[PhotoChemistry]] = None,
            vmr_shape: Optional[tuple[int, ...]] = None,
            ) -> FreckllArray:
    """Calculate the rate of change of the VMR."""
    from ..diffusion import molecular_diffusion
    from ..kinetics import solve_altitude_profile

    from ..ode import convert_y_to_fm, convert_fm_to_y, construct_reaction_terms
    from ..chegp import compute_dndt_vertical
    from ..network import map_production_loss
    from ..constants import AMU
    from ..distill import ksum

    vmr = convert_y_to_fm(y, *vmr_shape)

    density = density.to(u.cm**-3)
    masses = masses
#
    nlayers = density.size

    vmr = convert_y_to_fm(y, len(network.species), nlayers)
    mu = ksum(masses[:, None].value * vmr, k=4) << masses.unit

    altitude = solve_altitude_profile(
        temperature, mu, pressure, planet_mass, planet_radius,
    ).to(u.km)


    number_density = density*vmr

    mole_diffusion = molecular_diffusion(
        network.species, number_density, temperature, pressure
    )

    vert_term = compute_dndt_vertical(
        vmr,
        altitude,
        planet_radius,
        planet_mass,
        density,
        masses,
        mu,
        temperature,
        mole_diffusion,
        kzz
    )

    
    reactions = network.compute_reactions(vmr, with_production_loss=False)
    if photochemistry is not None:
        reactions = reactions + photochemistry.compute_reactions(
            number_density, altitude
        )
    products, loss = map_production_loss(reactions)
    reaction_term =construct_reaction_terms(products, loss, network.species , density.size, k=4)
    final = (reaction_term) / density.value + vert_term

    return convert_fm_to_y(final)

def jac(t: float, y: FreckllArray,
    density: Optional[u.Quantity] = None,
    temperature: Optional[u.Quantity] = None,
    pressure: Optional[u.Quantity] = None,
    kzz: Optional[u.Quantity] = None,
    planet_radius: Optional[u.Quantity] = None,
    planet_mass: Optional[u.Quantity] = None,
    masses: Optional[u.Quantity] = None,
    network: Optional[ChemicalNetwork] = None,
    photochemistry: Optional[Optional[PhotoChemistry]] = None,
    vmr_shape: Optional[tuple[int, ...]] = None,
    ) -> FreckllArray:
    from freckll.ode import construct_jacobian_reaction_terms_sparse, convert_y_to_fm
    from freckll.constants import AMU
    from ..network import map_production_loss
    from freckll.kinetics import solve_altitude_profile
    from freckll.chegp import compute_jacobian_sparse
    from freckll.distill import ksum
    from freckll.diffusion import molecular_diffusion

    density = density.to(u.cm**-3)
    masses = masses
    # altitude = altitude << u.km
    nlayers = density.size
    kzz = kzz.to(u.cm**2 / u.s)
    vmr = convert_y_to_fm(y, len(network.species), nlayers)
    mu = ksum(masses[:, None].value * vmr, k=4) << masses.unit
    number_density = density*vmr
    altitude = solve_altitude_profile(
        temperature, mu, pressure, planet_mass, planet_radius,
    ).to(u.km)

    mole_diffusion = molecular_diffusion(
        network.species, number_density, temperature, pressure
    )


    reactions = network.compute_reactions(vmr, with_production_loss=False)
    if photochemistry is not None:
        reactions = reactions + photochemistry.compute_reactions(
            number_density, altitude
        )

    _, loss = map_production_loss(reactions)
    react_mat = construct_jacobian_reaction_terms_sparse(loss, network.species, number_density.value)

    vert_mat = compute_jacobian_sparse(
        vmr,
        altitude,
        planet_radius,
        planet_mass,
        density,
        masses,
        mu,
        temperature,
        mole_diffusion,
        kzz)

    return vert_mat + react_mat

class Solver(Loggable):

    def __init__(self,
                 network: ChemicalNetwork,
                 photochemistry: Optional[PhotoChemistry] = None):
        
        """Initialize the solver."""
        super().__init__()
        self.set_network(network, photochemistry)

        self.temperature: Optional[u.Quantity] = None
        self.pressure: Optional[u.Quantity] = None
        self.kzz: Optional[u.Quantity] = None
        self.planet_radius: Optional[u.Quantity] = None
        self.planet_mass: Optional[u.Quantity] = None

    def set_network(self,
                    network: ChemicalNetwork,
                    photochemistry: Optional[PhotoChemistry] = None,
                    ) -> None:
        """Set the chemical network and photochemistry.
    
        Args:
            network: The chemical network.
            photochemistry: The photochemistry, if present.

        """
        self.network = network
        self.photochemistry = photochemistry


    def _run_solver(self, 
                    f: DyCallable,
                    jac: JacCallable,
                    y0: FreckllArray,
                    t0: float,
                    t1: float,
                    num_species: int,
                    **kwargs: dict) -> SolverOutput:
        """Run the solver."""
        raise NotImplementedError("Solver not implemented.")

    def set_system_parameters(self,
        *,
        temperature: Optional[u.Quantity] = None, 
        pressure: Optional[u.Quantity] = None, 
        kzz: Optional[u.Quantity] = None,   
        planet_radius: Optional[u.Quantity] = None,
        planet_mass: Optional[u.Quantity] = None) -> None:
        """Set the system parameters.
        
        Args:
            temperature: The temperature.
            pressure: The pressure.
            kzz: The eddy diffusion coefficient.
            planet_radius: The planet radius.
            planet_mass: The planet mass.
        """
        self.temperature = temperature if temperature is not None else self.temperature
        self.pressure = pressure if pressure is not None else self.pressure
        self.kzz = kzz if kzz is not None else self.kzz
        self.planet_radius = planet_radius if planet_radius is not None else self.planet_radius
        self.planet_mass = planet_mass if planet_mass is not None else self.planet_mass
        self.debug("Setting system parameters: temperature: %s, pressure: %s, kzz: %s", temperature, pressure, kzz)
        self.debug("Planet radius: %s, planet mass: %s", planet_radius, planet_mass)
        self.info("System parameters set. Compiling reactions.")
        self.network.compile_reactions(self.temperature, self.pressure)
        if self.photochemistry is not None:
            self.info("Compiling photochemistry.")
            self.photochemistry.compile_chemistry(
                self.temperature,
                self.pressure,
            )


    def solve(self,
              vmr: FreckllArray,
              t_span: tuple[float, float],
              atol: float = 1e-25,
              rtol: float = 1e-2,
              df_criteria: float = 1e-3,
              dfdt_criteria: float = 1e-8,
              **solver_kwargs,
              ) -> Solution:
        """Solve the ODE.
    
    
        Args:
            vmr: The initial VMR.
            t_span: The time span.
            atol: The absolute tolerance.
            rtol: The relative tolerance.
            df_criteria: The criteria for convergence.
            dfdt_criteria: The criteria for convergence.
            **solver_kwargs: Additional arguments for the solver.
        
        Returns:
            SolverOutput: The solver output.

        """
        from ..ode import convert_fm_to_y, convert_y_to_fm
        from ..kinetics import air_density

        density = air_density(self.temperature, self.pressure)

        f = partial(
            dndt,
            density=density,
            temperature=self.temperature,
            pressure=self.pressure,
            kzz=self.kzz,
            planet_radius=self.planet_radius,
            planet_mass=self.planet_mass,
            masses=self.network.masses,
            network=self.network,
            photochemistry=self.photochemistry,
            vmr_shape=vmr.shape,

        )

        jacobian = partial(
            jac,
            density=density,
            temperature=self.temperature,
            pressure=self.pressure,
            kzz=self.kzz,
            planet_radius=self.planet_radius,
            planet_mass=self.planet_mass,
            masses=self.network.masses,
            network=self.network,
            photochemistry=self.photochemistry,
            vmr_shape=vmr.shape,
        )

        self.debug("Solving with %s solver", self.__class__.__name__)
        y0 = convert_fm_to_y(vmr)

        num_species = vmr.shape[0]
        solver_output = self._run_solver(
            f,
            jacobian,
            y0,
            t_span[0],
            t_span[1],
            num_species,
            atol=atol,
            rtol=rtol,
            df_criteria=df_criteria,
            dfdt_criteria=dfdt_criteria,
            **solver_kwargs
        )

        self.info("Number of Jacobian evaluations: %d", solver_output["num_jac_evals"])
        self.info("Number of dndt evaluations: %d", solver_output["num_dndt_evals"])
        self.info("Success: %s", solver_output["success"])

        if solver_output["success"]:
            self.info("Solver completed successfully.")

        vmrs = np.array([
            convert_y_to_fm(ys, *vmr.shape)
            for ys in solver_output["y"] ]
        )


        planet_data = {
            "radius": self.planet_radius,
            "mass": self.planet_mass,
        }
        stellar_data = {}
        if self.photochemistry is not None and self.photochemistry.spectra is not None:
            stellar_data = {
                "stellar_flux": {
                "wavelength": self.photochemistry.spectra.wavelength,
                "incident_flux": self.photochemistry.incident_flux,
                "incident_angle": self.photochemistry.incident_angle,
                }
            }
            planet_data["distance"] = self.photochemistry.planet_distance
            planet_data["albedo"] = self.photochemistry.albedo



        result = {
            "initial_vmr": vmr,
            "success": solver_output["success"],
            "num_jacobian_evals": solver_output["num_jac_evals"],
            "num_dndt_evals": solver_output["num_dndt_evals"],
            "temperature": self.temperature,
            "pressure": self.pressure,
            "kzz": self.kzz,
            "planet": planet_data,
            "times": np.array(solver_output["times"]),
            "vmr": vmrs,
            "masses": self.network.masses,
            "density": density,
            "species": self.network.species,
            **stellar_data,
        }

        return result
        






