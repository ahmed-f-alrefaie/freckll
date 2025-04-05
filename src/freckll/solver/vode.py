from .solver import Solver, DyCallable, JacCallable, SolverOutput, output_step, convergence_test
import numpy as np
from ..types import FreckllArray
from collections import deque
from typing import Optional
from astropy import units as u
import time
class Vode(Solver):
    """VODE solver for Freckll."""
    def _run_solver(self, f: DyCallable, jac: JacCallable, y0: FreckllArray, t0: float, t1: float, 
                    num_species: int,
                    atol: float = 1e-25,
                    rtol: float = 1e-3,
                    df_criteria: float = 1e-3,
                    dfdt_criteria: float = 1e-8,
                    nevals:int=200,
                    max_solve_time: Optional[u.Quantity] = None,
                    max_retries:int=10,
                    **kwargs)-> SolverOutput:
        """Run the VODE solver.
        
        Args:
            f: The function to solve.
            jac: The Jacobian function.
            y0: The initial conditions.
            t0: The initial time.
            t1: The final time.
            num_species: The number of species in the system.
            atol: Absolute tolerance for the solver.
            rtol: Relative tolerance for the solver.
            df_criteria: Criteria for convergence.
            dfdt_criteria: Criteria for convergence.
            nevals: Number of evaluations to perform.
            max_retries: Maximum number of retries for the solver.
        
        Returns:
            SolverOutput: The output of the solver.
        
        """
        import math
        from scipy.integrate import ode
        from ..utils import convert_to_banded

        band = num_species + 2
        banded_jac = lambda t,x : convert_to_banded(jac(t,x), band)
        # Set the solver options

        options = {
            "method": "BDF",
            "atol": atol,
            "rtol": rtol,
            "uband": band,
            "lband": band,
        }

        start_t = math.log10(max(t0, 1e-6))
        end_t = math.log10(t1)
        t_eval = np.logspace(start_t, end_t, nevals)

        # Run the solver
        soln = ode(f,banded_jac).set_integrator("vode", **options)
        soln.set_initial_value(y0, t0)
        time_idx = 1

        ys = [y0]
        ts = [t0]

        if max_solve_time is not None:
            max_solve_time = max_solve_time.to(u.s).value

        start_time = time.time()
        retries = 0
        current_t = t_eval[time_idx]
        while soln.t < t1:
            soln.integrate(current_t)
            if soln.successful():
                ys.append(soln.y)
                ts.append(soln.t)

                output_step(soln.t, soln.y, self)
                if convergence_test(ys, ts, y0, self, atol, df_criteria, dfdt_criteria):
                    self.info("ODE solver converged.")
                    break
                time_idx += 1
                if time_idx >= len(t_eval):
                    break
                current_t = t_eval[time_idx]
            else:
                current_t = soln.t
                current_y = np.maximum(soln.y, 1e-50)
                soln = ode(f,banded_jac).set_integrator("vode", **options)
                soln.set_initial_value(current_y, current_t)
                max_retries += 1
                if retries >= max_retries:
                    self.warning("ODE solver failed to converge. Exiting.")
                    break
                self.info("ODE solver failed to converge. Resuming with last known state.")
                continue
            current_time = time.time() - start_time
            if max_solve_time is not None and current_time > max_solve_time:
                self.info("Maximum solve time reached")
                break


        return {
            "num_dndt_evals": 0,
            "num_jac_evals": 0,
            "success": True,
            "times": np.array(ts),
            "y": np.array(ys),
        }