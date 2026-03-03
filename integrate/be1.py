#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

from math import sqrt


def be1_adaptive(u0, f, *args, **kwargs):
    """
      Integrate a scalar initial value problem using backward Euler with
      Richardson extrapolation.

      This is the scalar variant: the simplified Newton iteration uses scalar
      division (no LU factorization). For systems, use be2_adaptive instead.

      Richardson extrapolation both (1) estimates an adaptive time-step and
      (2) increases the order from O(k) to O(k^2).
    """
    t0 = kwargs.pop('t0', 0.0)
    te = kwargs.pop('te', 1.0)
    rtol = kwargs.pop('rtol', 1e-4)
    atol = kwargs.pop('atol', 1e-4)
    a_min = kwargs.pop('a_min', 0.1)
    a_max = kwargs.pop('a_max', 5.0)
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
    accepted_steps = 0

    # Storage for output
    vs = []
    vs.append((t, np.copy(v)))  # Store initial condition

    # Newton method pars
    chi = 1e-3
    delta = sqrt(np.finfo(float).eps)

    # Finally integrate
    while not done:
        if t + k >= te:
            k = te - t
            done = True
        else: # Make sure that the last time step isn't too small.
            k = min(k, 0.5 * (te - t))

        # Now compute new steps starting from (t, v)
        df = (f(t, v + delta) - f(t, v)) / delta

        # Code Newton method
        def newton_iter(v0, df, t, k):
            # Setup simplified Newton
            v_new = np.copy(v0)
            dv = 1.0
            done = False
            while not done:
                dv_new = (v0 + k * f(t, v_new) - v_new) / (1.0 - k * df)
                theta = abs(dv_new) / abs(dv)
                done = dv_new == 0.0 or abs(dv_new) <= chi * atol * (1 - theta) / theta

                dv = dv_new
                v_new += dv_new

            return v_new

        # Call first Newton iteration
        vf_new = newton_iter(v, df, t + k, k)      # Full step
        vh_new = newton_iter(v, df, t + 0.5 * k, 0.5 * k) # Half-step
        vh_new = newton_iter(vh_new, df, t + k, 0.5 * k) # Half-step

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

    print('Scalar backward Euler: {0:s}'.format(msg[idid]))
    print('Accepted steps = ', accepted_steps, ' Rejected steps = ', rejected_steps)
    return np.array(vs, dtype=[('t', 'float'), ('u', 'f8', (n_eqn, ))])
