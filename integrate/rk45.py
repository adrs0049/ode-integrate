#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np


def rk5_adaptive(u0, f, *args, **kwargs):
    t0 = kwargs.pop('t0', 0.0)
    te = kwargs.pop('te', 1.0)
    rtol = kwargs.pop('rtol', 1e-4)
    atol = kwargs.pop('atol', 1e-4)
    a_min = kwargs.pop('a_min', 0.1)
    a_max = kwargs.pop('a_max', 5.0)
    safety = kwargs.pop('safety', 0.9)
    k_min = kwargs.pop('k_min', 1e-8)
    k_max = kwargs.pop('k_max', 10)
    stepmax = kwargs.pop('stepmax', 10000)
    n_eqn = u0.size if isinstance(u0, np.ndarray) else 1  # Number of equations
    verbose = kwargs.pop('verbose', False)

    # Dummy solout
    def solout(*args):
        return False

    # overwrite solout
    solout = kwargs.pop('solout', solout)

    # Error messages
    msg = {0: 'Success!', 2: 'Step size dropped below kmin!', 3: 'Required more than stepmax steps!'}

    # Error code
    idid = 0

    # Norm function to use
    norm = lambda x: np.max(np.abs(x))

    # Time step setup
    k = kwargs.pop('k0', 1e-4 * (te - t0))

    # Current integrator state
    v = np.asarray(u0) # This always holds the current ODE value
    t = t0             # Current time

    # Logical variable to indicate if we can quit
    done = False

    # Some stats
    fevals         = 0
    rejected_steps = 0
    accepted_steps = 0

    # PI-parameters
    alpha = 0.8
    beta  = -0.4
    perr  = np.finfo(float).eps

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
        # RK2: Compute the two slopes
        r1 = k * f(t, v)
        r2 = k * f(t + (2. / 9.) * k, v + (2. / 9.) * r1)
        r3 = k * f(t + (1. / 3.) * k, v + (r1 + 3.0 * r2) / 12.)
        r4 = k * f(t + (3. / 4.) * k, v + (69. * r1 - 243.0 * r2 + 270.0 * r3) / 128.)
        r5 = k * f(t + k,             v - (17. / 12.) * r1 + (27. / 4.) * r2 - (27. / 5.) * r3 + (16. / 15.) * r4)
        r6 = k * f(t + (5. / 6.) * k, v + (65. / 432.) * r1 - (5. / 16.) * r2 + (13. / 16.) * r3 + (4. / 27.) * r4 + (5. / 144.) * r5)

        # Count number of function evaluations
        fevals += 6

        # Compute RK3 step
        vnew = v + (47. / 450.) * r1 + (12. / 25) * r3 + (32. / 225) * r4 + (1. / 30.) * r5 + (6. / 25.) * r6

        # Compute RK2 step
        w  = v + (1. / 9.) * r1 + (9. / 20.) * r3 + (16. / 45.) * r4 + (1. / 12.) * r5

        # Estimate the error: TODO I need to do better here -> MOVE To vectorized version!
        err = max(np.finfo(float).eps, norm((w - vnew) / (atol + np.maximum(np.abs(vnew), np.abs(v)) * rtol)))

        # Generate the new time-step
        knew = min(k_max, k * min(a_max, max(a_min, safety * np.power(1./err, 1./5))))
        #knew = k * (1./err)**(alpha / 5.0) * (1./perr)**(beta / 5.0)

        # Make sure that new time-step is acceptable
        if k < k_min:
            idid = 2
            break

        if accepted_steps + rejected_steps > stepmax:
            idid = 3
            break

        if err <= 1.0: # Current error was small enough so update state
            # Call solout
            done = done or solout(v, vnew, t, t + k)

            # Update step information
            perr = err
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

    if verbose:
        print('Explicit RK5: {0:s}'.format(msg[idid]))
        print('Accepted steps = ', accepted_steps, ' Rejected steps = ', rejected_steps)
        print('Function evluations = {:d}.'.format(fevals))
    return np.array(vs, dtype=[('t', 'float'), ('u', 'f8', (n_eqn, ))])

