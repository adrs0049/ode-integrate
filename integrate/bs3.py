#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
"""
Bogacki-Shampine 3(2) adaptive Runge-Kutta method with FSAL property
and integrated stiffness detection.

Butcher tableau:
    0   |
    1/2 | 1/2
    3/4 | 0    3/4
    1   | 2/9  1/3  4/9
    ----|--------------------
    y3  | 2/9  1/3  4/9  0     (3rd order, primary)
    y^  | 7/24 1/4  1/3  1/8   (2nd order, embedded)

FSAL: k4 = f(t_n + h, y_{n+1}) is reused as k1 of the next step,
giving 3 new f-evaluations per accepted step instead of 4.

Error estimate coefficients (y3 - y^):
    e1 = -5/72, e2 = 1/12, e3 = 1/9, e4 = -1/8

Stiffness detection uses:
    1. Spectral radius estimation from stage differences
    2. Step rejection ratio over a sliding window
    3. Persistence counter to avoid thrashing
"""
import numpy as np
from collections import deque


def bs3_adaptive(u0, f, *args, **kwargs):
    t0 = kwargs.pop('t0', 0.0)
    te = kwargs.pop('te', 1.0)
    rtol = kwargs.pop('rtol', 1e-4)
    atol = kwargs.pop('atol', 1e-4)
    a_min = kwargs.pop('a_min', 0.1)
    a_max = kwargs.pop('a_max', 5.0)
    safety = kwargs.pop('safety', 0.9)
    k_min = kwargs.pop('k_min', 1e-8)
    stepmax = kwargs.pop('stepmax', 10000)
    verbose = kwargs.pop('verbose', False)
    detect_stiffness = kwargs.pop('detect_stiffness', True)
    n_eqn = u0.size if isinstance(u0, np.ndarray) else 1

    # Dummy solout
    def solout(*args):
        return False
    solout = kwargs.pop('solout', solout)

    # Error messages
    msg = {0: 'Success!', 2: 'Step size dropped below kmin!',
           3: 'Required more than stepmax steps!'}

    # Error code
    idid = 0

    # Norm function
    norm = lambda x: np.max(np.abs(x))

    # Time step setup
    k = kwargs.pop('k0', 1e-4 * (te - t0))

    # Current integrator state
    v = np.asarray(u0, dtype=float)
    t = t0

    done = False

    # Stats
    fevals = 0
    rejected_steps = 0
    accepted_steps = 0

    # Storage for output
    vs = []
    vs.append((t, np.copy(v)))

    # --- FSAL: compute first stage ---
    k1 = f(t, v)
    fevals += 1

    # --- Stiffness detection state ---
    stiff_counter = 0              # persistence counter
    stiff_threshold = 15           # switch threshold
    stiff_detected = False
    bs3_stability_boundary = 2.5   # approx |z| boundary of BS3 stability region
    # Sliding window for rejection ratio
    window_size = 20
    step_outcomes = deque(maxlen=window_size)  # True=accepted, False=rejected
    # Diagnostics
    spectral_history = []

    # Finally integrate
    while not done:
        if t + k >= te:
            k = te - t
            done = True
        else:
            k = min(k, 0.5 * (te - t))

        # --- BS3 stages (FSAL: k1 is already computed) ---
        k2 = f(t + 0.5 * k, v + 0.5 * k * k1)
        k3 = f(t + 0.75 * k, v + 0.75 * k * k2)

        # 3rd order solution
        vnew = v + k * (2.0/9.0 * k1 + 1.0/3.0 * k2 + 4.0/9.0 * k3)

        # 4th stage (FSAL: this becomes k1 of next step if accepted)
        k4 = f(t + k, vnew)
        fevals += 3  # k2, k3, k4 are the new evaluations

        # Error estimate: y3 - y^ = e1*k1 + e2*k2 + e3*k3 + e4*k4
        err_vec = k * (-5.0/72.0 * k1 + 1.0/12.0 * k2
                       + 1.0/9.0 * k3 - 1.0/8.0 * k4)

        # Scaled error norm
        sc = atol + np.maximum(np.abs(vnew), np.abs(v)) * rtol
        err = max(np.finfo(float).eps, norm(err_vec / sc))

        # New step size (order p^ = 2 for the embedded formula)
        knew = k * min(a_max, max(a_min, safety * np.power(1.0/err, 1.0/3.0)))

        # Check termination conditions
        if k < k_min and not done:
            idid = 2
            break

        if accepted_steps + rejected_steps > stepmax:
            idid = 3
            break

        if err <= 1.0:
            # --- Stiffness detection on accepted steps ---
            if detect_stiffness:
                dy = vnew - v
                dy_norm = norm(dy)
                if dy_norm > 0:
                    # Estimate spectral radius from the difference between
                    # the last and first stage evaluations, normalized by
                    # the step size. For a linear problem y' = Jy,
                    # k4 - k1 ~ J * dy, so |lam| ~ ||k4 - k1|| / ||dy||
                    lam_est = norm(k4 - k1) / dy_norm
                    h_lam = k * lam_est
                    spectral_history.append((t + k, lam_est, h_lam))

                    # Rejection ratio over sliding window
                    rej_ratio = (step_outcomes.count(False) /
                                 max(1, len(step_outcomes)))

                    # Stiffness indicator: h*|lam| is near the stability
                    # boundary. When the adaptive controller is working, it
                    # keeps h just below the boundary so we use 0.8 * boundary
                    # as threshold. A high rejection ratio is supplementary.
                    is_stiff_now = (h_lam > 0.8 * bs3_stability_boundary
                                    or (h_lam > 0.5 * bs3_stability_boundary
                                        and rej_ratio > 0.3))

                    if is_stiff_now:
                        stiff_counter = min(stiff_counter + 1,
                                            stiff_threshold + 5)
                    else:
                        stiff_counter = max(stiff_counter - 1, 0)

                    stiff_detected = stiff_counter >= stiff_threshold

                step_outcomes.append(True)

            # Call solout (don't override done=True from final step)
            done = done or solout(v, vnew, t, t + k)

            # FSAL: reuse k4 as k1 for next step
            k1 = k4

            # Update state
            v = vnew
            t = t + k
            vs.append((t, np.copy(v)))
            accepted_steps += 1

        else:
            # Step rejected
            done = False
            rejected_steps += 1

            if detect_stiffness:
                step_outcomes.append(False)

            # On rejection, k1 stays the same (we retry from same point)

        k = knew

    if verbose:
        print('BS3 (Bogacki-Shampine): {0:s}'.format(msg[idid]))
        print('Accepted steps =', accepted_steps,
              ' Rejected steps =', rejected_steps)
        print('Function evaluations = {:d}.'.format(fevals))
        if detect_stiffness:
            print('Stiffness detected:', stiff_detected,
                  ' (counter={:d})'.format(stiff_counter))

    result = np.array(vs, dtype=[('t', 'float'), ('u', 'f8', (n_eqn,))])

    diagnostics = {
        'accepted_steps': accepted_steps,
        'rejected_steps': rejected_steps,
        'fevals': fevals,
        'stiff_detected': stiff_detected,
        'stiff_counter': stiff_counter,
        'spectral_history': spectral_history,
        'idid': idid,
        'msg': msg[idid],
    }

    return result, diagnostics
