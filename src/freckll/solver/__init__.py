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


class Solution(TypedDict):
    """TypedDict for solver output."""
    
    success: bool
    initial_vmr: FreckllArray
    num_jacobian_evals: int
    num_dndt_evals: int
    temperature: u.Quantity
    pressure: u.Quantity
    vmr: FreckllArray
    kzz: u.Quantity
    times: FreckllArray
    vmr: FreckllArray


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


    def solve(self,
              vmr: FreckllArray,
              temperature: u.Quantity, 
              pressure: u.Quantity, 
              kzz: u.Quantity,   
              planet_radius: u.Quantity,
              planet_mass: u.Quantity,   

              star_spectra: Optional[StarSpectra] = None,
              distance: Optional[u.Quantity] = None,
              incident_angle: Optional[u.Quantity] = 45 << u.deg,
              albedo: Optional[float] = 0.0,

              t0: float = 0.0,
              t1: float = 1e10,
              atol: float = 1e-25,
              rtol: float = 1e-3,
              df_criteria: float = 1e-3,
              dfdt_criteria: float = 1e-8,

              **solver_kwargs: dict,
              ) -> Solution:
        """Solve the ODE.
        
        Args:
            vmr: The initial VMR.
            temperature: The temperature.
            pressure: The pressure.
            kzz: The eddy diffusion coefficient.
            solver_kwargs: Additional arguments for the solver.
        
        Returns:
            SolverOutput: The solver output.

        """
        from ..ode import convert_fm_to_y, convert_y_to_fm
        from ..kinetics import air_density
        self.network.compile_reactions(temperature, pressure)
        if self.photochemistry is not None:
            self.photochemistry.compile_chemistry(
                temperature,
                pressure,
            )
        density = air_density(temperature, pressure)

        f = partial(
            dndt,
            density=density,
            temperature=temperature,
            pressure=pressure,
            kzz=kzz,
            planet_radius=planet_radius,
            planet_mass=planet_mass,
            masses=self.network.masses,
            network=self.network,
            photochemistry=self.photochemistry,
            vmr_shape=vmr.shape,

        )

        jacobian = partial(
            jac,
            density=density,
            temperature=temperature,
            pressure=pressure,
            kzz=kzz,
            planet_radius=planet_radius,
            planet_mass=planet_mass,
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
            t0,
            t1,
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

        return {
            "initial_vmr": vmr,
            "success": solver_output["success"],
            "num_jacobian_evals": solver_output["num_jac_evals"],
            "num_dndt_evals": solver_output["num_dndt_evals"],
            "temperature": temperature,
            "pressure": pressure,
            "kzz": kzz,
            "times": solver_output["times"],
            "vmr": vmrs,
        }
        






