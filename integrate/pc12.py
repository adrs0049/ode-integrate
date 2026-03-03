#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
"""Adaptive predictor-corrector (PECE) integrator using Euler/trapezoidal pair."""
import numpy as np

from math import sqrt


def pc_adaptive(u0, f, *args, **kwargs):
    """
      Adaptive PECE integrator: the predictor is a forward Euler step,
      the corrector applies the trapezoidal rule. Their difference
      estimates the local error and drives step-size control.
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
    vs.append((t, v))  # Store initial condition

    # Finally integrate
    while not done:
        if t + k >= te:
            k = te - t
            done = True
        else: # Make sure that the last time step isn't too small.
            k = min(k, 0.5 * (te - t))

        # Now compute new steps starting from (t, v)
        r1 = f(t, v)

        # Compute Euler step i.e. the predictor
        w = v + k * r1

        # Apply the corrector i.e. trapezoidal rule
        vnew = v + 0.5 * k * (r1 + f(t + k, w))

        # Estimate the error
        err = max(np.finfo(float).eps, norm(w - vnew) / (atol + max(norm(vnew), norm(v)) * rtol))

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
            v = vnew
            t = t + k

            # Store accepted step
            vs.append((t, v))

            # Update stats
            accepted_steps += 1
        else: # Error was too large -> repeat the step with new step-size
            done = False

            # Update stats
            rejected_steps += 1

        # Always use new step-size for next step
        k = knew

    print('Adaptive PECE: {0:s}'.format(msg[idid]))
    print('Accepted steps = ', accepted_steps, ' Rejected steps = ', rejected_steps)
    return np.array(vs, dtype=[('t', 'float'), ('u', 'f8', (n_eqn, ))])
