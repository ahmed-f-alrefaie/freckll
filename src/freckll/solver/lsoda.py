import numpy as np

from ..types import FreckllArray
from .solver import DyCallable, JacCallable, Solver, SolverOutput, output_step
from .transform import Transform


class LSODA(Solver):
    def _run_solver(
        self,
        f: DyCallable,
        jac: JacCallable,
        y0: FreckllArray,
        t0: float,
        t1: float,
        num_species: int,
        transform: Transform,
        atol: float = 1e-25,
        rtol: float = 1e-3,
        df_criteria: float = 1e-3,
        dfdt_criteria: float = 1e-8,
        nevals: int = 50,
        **kwargs,
    ) -> SolverOutput:
        import math

        from scipy.integrate import solve_ivp

        from ..utils import convert_to_banded_lsoda

        band = num_species + 2
        banded_jac = lambda t, x, jac=jac, band=band: convert_to_banded_lsoda(jac(t, x), band)

        def dydt(t, y):
            # Call the function to compute dy/dt
            output_step(t, y, self)
            return f(t, y)

        # Set the solver options

        options = {
            "method": "LSODA",
            "atol": atol,
            "rtol": rtol,
            "jac": banded_jac,
            "lband": band,
            "uband": band,
        }

        start_t = math.log10(max(t0, 1e-20))
        end_t = math.log10(t1)
        t_eval = np.logspace(start_t, end_t, nevals)
        y0_transform = transform.transform(y0)
        # Run the solver
        sol = solve_ivp(
            fun=dydt,
            t_span=(t0, t1),
            y0=y0_transform,
            t_eval=t_eval,
            **options,
        )

        return {
            "num_dndt_evals": sol.nfev,
            "num_jac_evals": sol.njev,
            "success": sol.success,
            "times": sol.t,
            "y": transform.inverse_transform(sol.y),
        }
