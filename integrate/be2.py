#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from scipy import linalg as LA

from math import sqrt
from copy import copy


def ej(j, n):
    e = np.zeros(n, dtype=float)
    e[j] = 1.0
    return e


def be2_adaptive(u0, f, *args, **kwargs):
    """
      Integrate an initial value problem using backward Euler with Richardson's
      extrapolation to both (1) estimate an adaptive time-step and (2) increase
      the order of the method of O(k^2) so that it matches the order of the
      explicit RK2 method implemented below.
    """
    t0 = kwargs.pop('t0', 0.0)
    te = kwargs.pop('te', 1.0)
    rtol = kwargs.pop('rtol', 1e-4)
    atol = kwargs.pop('atol', 1e-4)
    a_min = kwargs.pop('a_min', 0.1)
    a_max = kwargs.pop('a_max', 2.0)
    safety = kwargs.pop('safety', 0.9)
    k_min = kwargs.pop('k_min', 1e-8)
    stepmax = kwargs.pop('stepmax', 10000)
    n_eqn = u0.size if isinstance(u0, np.ndarray) else 1  # Number of equations

    # Error messages
    msg = {0: 'Successful', 2: 'Step size dropped below kmin!', 3: 'Required more than stepmax steps!'}

    # Norm function to use
    norm = lambda x: np.max(np.abs(x))

    # Error code
    idid = 0

    # Time step setup
    k = kwargs.pop('k0', 1e-4 * (te - t0))

    # Current integrator state
    v = np.asarray(u0) # This always holds the current ODE value
    t = t0             # Current time

    # Logical variable to indicate if we can quit
    done = False

    # Some stats
    rejected_steps = 0
    r_newton_steps = 0
    accepted_steps = 0

    # Storage for output
    vs = []
    vs.append((t, np.copy(v)))  # Store initial condition

    # Newton method pars
    chi = 1e-1
    delta = sqrt(np.finfo(float).eps)

    # Finally integrate
    while not done:
        if t + k >= te:
            k = te - t
            done = True
        else: # Make sure that the last time step isn't too small.
            k = min(k, 0.5 * (te - t))

        # Now compute new steps starting from (t, v)
        # Use a forward difference to approximate the Jacobian column-wise
        # The delta is chosen so that it's roughly at the point of lowest error.
        df = np.zeros((n_eqn, n_eqn), dtype=float)
        fv = f(t, v)
        for i in range(n_eqn):
            ei = ej(i, n_eqn)
            df[:, i] = (f(t, v + ei * delta) - fv) / delta

        # Code Newton method
        def newton_iter(v0, df, t, k):
            """
              Implement the nonlinear update function.

              1) Use simplified Newton i.e. the Jacobian from the first step is re-used.
                 Compute LU decomposition of Jacobian once at the beginning
                 of the loop.

              2) This reduces the convergence from quadratic to linear, hence
              monitor convergence speed via theta.

                || v^{n+1} - v^{n} || <= Theta || v^{n} - v^{n-1} ||

              If Theta > 1 we don't have convergence quit and try again with smaller step size.

              3) Loop termination condition based on distance between adjacent
              elements. Note that we must modify the convergence criteria on the right.
              See chapter 5 in the notes for the details.

              || v^{n+1} - v^{n} || <= chi atol (1 - theta) / theta

              This gives us a bound of chi atol for the actual absolute error.
              The chi in [1e-3, 1e-1] to make sure the iterate computed via
              Newton is at least as accurate as the error tolerance on the itegrator.
            """
            # Setup simplified Newton
            v_new = np.copy(v0)
            done = False

            # Assemble the matrix and invert it: Can we do this better?
            T = np.eye(n_eqn) - k * df
            lu, piv = LA.lu_factor(T)

            # Do first iterate
            dv_new = LA.lu_solve((lu, piv), (v0 + k * f(t, v_new) - v_new))
            dv = dv_new
            v_new += dv_new

            # Quit if we are done now
            done = norm(dv) == 0.0

            while not done:
                # dv_new = (v0 + k * f(t, v_new) - v_new) / (1.0 - k * df)
                dv_new = LA.lu_solve((lu, piv), (v0 + k * f(t, v_new) - v_new))
                theta = norm(dv_new) / norm(dv)
                if theta > 1.0:  # Again disgusting!
                    break

                done = norm(dv_new) == 0.0 or norm(dv_new) <= chi * atol * (1 - theta) / theta
                dv = dv_new
                v_new += dv_new

            return v_new, theta

        # This Newton error handling is disgusting! FIX IT!
        # Call first Newton iteration
        vf_new, theta = newton_iter(v,      df, t + k, k)      # Full step
        if theta > 1.:
            r_newton_steps += 1
            k = k * a_min
            continue

        vh_new, theta = newton_iter(v,      df, t + 0.5 * k, 0.5 * k) # Half-step
        if theta > 1.:
            r_newton_steps += 1
            k = k * a_min
            continue

        vh_new, theta = newton_iter(vh_new, df, t + k, 0.5 * k) # Half-step
        if theta > 1.:
            r_newton_steps += 1
            k = k * a_min
            continue

        # Estimate the error
        err = max(np.finfo(float).eps, norm(vf_new - vh_new) / (atol + max(norm(vh_new), norm(v)) * rtol))

        # Generate the new time-step
        knew = k * min(a_max, max(a_min, safety * sqrt(1./err)))

        # Make sure that new time-step is acceptable
        if k < k_min:
            idid = 2
            break

        if rejected_steps + accepted_steps > stepmax:
            idid = 3
            break

        if err <= 1.0: # Current error was small enough so update state
            # Now do the Richardson extrapolation step
            # v_new = vh_new + (vh_new - vf_new) / (2**p - 1)  p = 1 here
            v = 2. * vh_new - vf_new
            t = t + k

            # Store accepted step
            vs.append((t, np.copy(v)))

            # Update stats
            accepted_steps += 1
        else: # Error was too large -> repeat the step with new step-size
            done = False

            # Update stats
            rejected_steps += 1

        # Always use new step-size for next step
        k = knew

    print('Second order backward Euler: {0:s}'.format(msg[idid]))
    print('Accepted steps = ', accepted_steps,
          ' Rejected steps = ', rejected_steps,
          ' Rejected Newton steps = ', r_newton_steps)
    return np.array(vs, dtype=[('t', 'float'), ('u', 'f8', (n_eqn, ))])

