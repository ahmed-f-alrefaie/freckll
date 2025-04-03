from . import Solver, DyCallable, JacCallable, SolverOutput, output_step, convergence_test
import numpy as np
from ..types import FreckllArray
from collections import deque
class Vode(Solver):

    def _run_solver(self, f: DyCallable, jac: JacCallable, y0: FreckllArray, t0: float, t1: float, 
                    num_species: int,
                    atol: float = 1e-25,
                    rtol: float = 1e-3,
                    df_criteria: float = 1e-3,
                    dfdt_criteria: float = 1e-8,
                    nsteps:int=200,

                    **kwargs)-> SolverOutput:
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
        t_eval = np.logspace(start_t, end_t, nsteps)

        # Run the solver
        soln = ode(f,banded_jac).set_integrator("vode", **options)
        soln.set_initial_value(y0, t0)
        time_idx = 1

        ys = [y0]
        ts = [t0]

        track_y = deque(maxlen=10)
        track_t = deque(maxlen=10)


        last_y = np.copy(y0)

        # def solout(t, y):
        #     nonlocal time_idx
        #     ys.append(y)
        #     ts.append(t)
        #     time_idx += 1
        #     return 0



                    
        #soln.set_solout(lambda t, y: output_step(t, y, self))

        while soln.t < t1:
            soln.integrate(t_eval[time_idx])
            if soln.successful():
                ys.append(soln.y)
                ts.append(soln.t)

                output_step(soln.t, soln.y, self)
                convergence_test(ys, ts, y0, self, atol, df_criteria, dfdt_criteria)
                time_idx += 1
            else:
                self.info("ODE solver failed to converge. Resuming with last known state.")
                continue
        

        return {
            "num_dndt_evals": 0,
            "num_jac_evals": 0,
            "success": True,
            "times": np.array(ts),
            "y": np.array(ys),
        }