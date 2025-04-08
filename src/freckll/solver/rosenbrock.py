from .solver import Solver, DyCallable, JacCallable, SolverOutput, convergence_test, output_step
import numpy as np
from ..types import FreckllArray
from typing import Optional
from astropy import units as u
import time
from collections import deque
from ..kinetics import AltitudeSolveError
from ..ode import convert_y_to_fm, convert_fm_to_y
from ..distill import ksum
def update_timestep(timestep: float, rtol: float, error: float):
    """Updates the timestep based on the error and the desired tolerance.
    
    Args:
        timestep (float): The current timestep.
        rtol (float): The relative tolerance.
        error (float): The estimated error.
    
    
    """
    return 0.9*timestep*(rtol/error)**0.5

def step_second_order_rosenbrock(f: DyCallable, jac: JacCallable, y: FreckllArray, t:float, h: float) -> FreckllArray:
    """Single step of the second-order Rosenbrock method.
    
    Implementation based on VULCAN paper Tsai et al 2017 ApJS 228 Eq. A9
    
    """
    import math
    import warnings
    from scipy import sparse
    from scipy.sparse import linalg as spla
    I = sparse.eye(len(y), format="csc")
    gamma = 1 + 1/math.sqrt(2)
    lhs = (I - gamma*h*jac(0, y))
    # Supress RuntimeWarning
    with np.errstate(all="ignore"):
        rhs = f(t, y)

        k1 = spla.spsolve(lhs, rhs, use_umfpack=True)
        # k1 = spla.lgmres(lhs, rhs, M=M, atol=1e-25)[0]

        new_f = f(t, y + h*k1) - 2*k1

        if np.any(np.isnan(new_f)):
            return None
        
        k2 = spla.spsolve(lhs, new_f, use_umfpack=True)
        # k2 = spla.lgmres(
        #    lhs, 
        #    new_f,
        #    M=M, atol=1e-25, 
        # )[0]



        y_new = y + (1.5*k1 + 0.5*k2) * h
        
        y1 = y + k1 * h
        error = np.abs(y_new - y1)

    return y_new, error




class Rosenbrock(Solver):
    r"""Second-order Rosenbrock solver for stiff ODEs.
    
    Implementation based on VULCAN paper Tsai et al 2017 ApJS 228 (Appendix A)

    We solve first for $k_1$ and then for $k_2$ using the implicit method.

    Where $k_1$ is defined as:

    $$(I - \gamma h J) k_1 = f(t, y)$$

    And $k_2$ is defined as:

    $$(I - \gamma h J) k_2 = f(t, y + h k_1) - 2 k_1$$

    Where $\gamma = 1 + \frac{1}{\sqrt{2}}$.

    The new value of y is then given by:

    $$y_{new} = y + (1.5 k_1 + 0.5 k_2) h$$
    
    The error is estimated as:
    
    $$error = |y_{new} - y + k_1 h|$$
    
    The timestep is updated using the formula:
    
    $$h_{new} = 0.9 h \left( \frac{rtol}{error} \right)^{0.5}$$




    
    """
    def _run_solver(self, f: DyCallable, jac: JacCallable, y0: FreckllArray, t0: float, t1: float, 
                    num_species: int,
                    atol: float = 1e-25,
                    rtol: float = 1e-3,
                    df_criteria: float = 1e-3,
                    dfdt_criteria: float = 1e-8,
                    initial_step: float = 1e-13,
                    timestep_reject_factor: float = 0.1,
                    minimum_step: float = 1e-16,
                    tiny: float = 1e-50,
                    nevals: int = 200,
                    strict: bool = False,
                    maxiter: Optional[int] = 100,
                    max_solve_time: Optional[u.Quantity] = None,
                    **kwargs)-> SolverOutput:
        """Solve the ODE using the Rosenbrock method.
        
        Args:
            f: The function to integrate.
            jac: The Jacobian of the function.
            y0: The initial conditions.
            t0: The initial time.
            t1: The final time.
            num_species: The number of species in the system.
            atol: The absolute tolerance.
            rtol: The relative tolerance.
            df_criteria: The criteria for convergence.
            dfdt_criteria: The criteria for convergence.
            initial_step: The initial step size.
            timestep_reject_factor: The factor to reduce the timestep by on rejection.
            minimum_step: The minimum step size.
            tiny: A small value to avoid division by zero.
            neval: The number of time evaluations to store.
            strict: If True, reject negative values.
            maxiter: The maximum number of iterations.
            max_solve_time: The maximum time to spend solving the ODE.
        """
        import math

        h = initial_step
        y = np.copy(y0)

        f_eval = 0
        jac_eval = 0



        ys = [y]
        ts = [t0]

        track_y = deque(maxlen=3)
        track_t = deque(maxlen=3)
        track_y.append(y)
        track_t.append(t0)
    

        t = t0
        iterations = 0
        success = False

        if max_solve_time is not None:
            max_solve_time = max_solve_time.to(u.s).value
        
        start_time = time.time()

        start_t = math.log10(max(t0, 1e-6))
        end_t = math.log10(t1)
        t_evals = np.logspace(start_t, end_t, nevals)
        t_eval_index = 0
        while t < t1:


            # Step the solver
            f_eval += 2
            jac_eval += 1
            iterations += 1
            current_time = time.time() - start_time
            # Break if the maximum number of iterations is reached
            if maxiter and iterations > maxiter:
                ys.append(y)
                self.info("Maximum iterations reached")
                break

            if max_solve_time is not None and current_time > max_solve_time:
                self.info("Maximum solve time reached")
                ys.append(y)
                success = False
                break

            try:
                result = step_second_order_rosenbrock(f, jac, y,t, h)
            except AltitudeSolveError:
                # If we get an altitude solve error, we need to reject the step
                self.info("Altitude solve error")
                self.info("Y values: %s %s", y.min(), y.max())
                h = h * timestep_reject_factor
                continue
            
            # Reject
 




            # Reject the step if the result is None (due to NaN or Inf)
            if result is None:
                h = h * timestep_reject_factor
                continue

            # Check if the step is valid
            y_new, error = result

    

            if not strict:
                y_new = np.maximum(y_new, tiny)
                # fm = convert_y_to_fm(y_new, num_species, self.kzz.size)

                # fm = fm/ksum(fm, k=4)

                # y_new = convert_fm_to_y(fm)

            # If we are under strict conditions then reject the step if valeus are negative.
            test_f = f(t, y_new)

    
            # Check for NaN or Inf in the new values
            if np.any(np.isnan(y_new) | np.isinf(y_new) | np.isnan(test_f) | (y_new < 0)):
                self.info("Reducing step size")
                # Reject the step and reduce the timestep
                h = h * timestep_reject_factor
                # If the new step is too small, 
                if h < minimum_step:
                    self.info("Minimum step size reached")
                    break

                h = max(h, minimum_step)
                continue
            
            if h == 0:
                self.info("Zero step size")
                ys.append(y)
                break
            
            # Accept the step
            track_y.append(y_new)
            y = np.copy(y_new)

            
            t += h
            track_t.append(t)
                


            # Determine the new timestep
            error[y_new < atol] = 0
            error[y_new < 0] = 0
            delta = np.amax(error[y_new > 0])

            h = update_timestep(h, rtol, delta)
            output_step(t, test_f, self)

            while t_eval_index < len(t_evals) and t >= t_evals[t_eval_index]:
                # We need to interpolate if we've stepped past the t_eval point
                if len(ts) >= 2:  # Need at least two points to interpolate
                    t_prev = track_t[-2]
                    y_prev = track_y[-2]
                    # Linear interpolation between previous and current point
                    alpha = (t_evals[t_eval_index] - t_prev) / (t - t_prev)
                    y_interp = y_prev + alpha * (y - y_prev)
                    ys.append(y_interp)
                    ts.append(t_evals[t_eval_index])
                else:
                    # If this is the first step, just use the current value
                    ys.append(y)
                    ts.append(t_evals[t_eval_index])
                t_eval_index += 1
            if t >= t1:
                self.info("Reached the end of the integration range")
                success = True
                break

            

            if convergence_test(track_y, track_t, y0, self, atol, df_criteria, dfdt_criteria):
                self.info("Converged to the solution")
                success = True
                break
        if t != ts[-1]:
            ys.append(y)
            ts.append(t)

        extra = {

        }

        if not success:
            extra["dndt"] = f(t, y)
            extra["jac"] = jac(t, y)


        return {
            "num_dndt_evals": f_eval,
            "num_jac_evals": jac_eval,
            "success": success,
            "times": ts,
            "y": ys,
            **extra,
        }





            



