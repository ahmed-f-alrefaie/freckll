from . import Solver, DyCallable, JacCallable, SolverOutput
import numpy as np
from ..types import FreckllArray

class BDF(Solver):

    def _run_solver(self, f: DyCallable, jac: JacCallable, y0: FreckllArray, t0: float, t1: float, 
                    num_species: int,
              atol: float = 1e-25,
                rtol: float = 1e-3,
              df_criteria: float = 1e-3,
                dfdt_criteria: float = 1e-8,
                    nsteps:int=50,
                    **kwargs)-> SolverOutput:
        import math
        from scipy.integrate import solve_ivp

        # Set the solver options

        options = {
            "method": "BDF",
            "atol": atol,
            "rtol": rtol,
            "jac": jac,
        }

        start_t = math.log10(max(t0, 1e-20))
        end_t = math.log10(t1)
        t_eval = np.logspace(start_t, end_t, nsteps)

        # Run the solver
        sol = solve_ivp(
            fun=f,
            t_span=(t0, t1),
            y0=y0,
            t_eval=t_eval,
            **options,
         
        )
        
        return {
            "num_dndt_evals": sol.nfev,
            "num_jac_evals": sol.njev,
            "success": sol.success,
            "times": sol.t,
            "y": sol.y,
        }
