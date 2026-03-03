#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
"""
Auto-switching integrator: BS3 <-> Rosenbrock 2(3).

Starts in explicit mode (BS3/Bogacki-Shampine) and monitors for stiffness.
When stiffness is detected, switches to the L-stable Rosenbrock 2(3) method
(Shampine modified Rosenbrock triple, as in MATLAB ode23s).
When the Rosenbrock method detects the problem has become non-stiff again,
it switches back to BS3.

Both methods use order-2 advancing formulas with order-3 error estimates,
so the switch is seamless with no transient step-size adjustment needed.

Stiffness detection (in BS3 mode):
  - Spectral radius estimate from stage differences
  - Persistence counter to avoid thrashing (threshold = 15)

Non-stiffness detection (in Rosenbrock mode):
  - Periodically estimate spectral radius
  - If h * |lam_est| is well within BS3 stability region, switch back
  - Persistence counter for the reverse direction
"""
import numpy as np
from math import sqrt
from scipy import linalg as LA
from collections import deque


# BS3 Butcher tableau coefficients
_bs3_e1 = -5.0/72.0
_bs3_e2 = 1.0/12.0
_bs3_e3 = 1.0/9.0
_bs3_e4 = -1.0/8.0

# Rosenbrock 2(3) parameters (Shampine modified Rosenbrock triple)
_d = 1.0 / (2.0 + sqrt(2.0))     # ≈ 0.29289322
_e32 = 6.0 + sqrt(2.0)           # ≈ 7.41421356

# BS3 stability boundary (radius along negative real axis)
_bs3_boundary = 2.5


def _fd_jacobian(f, t, y, n, delta_base):
    """Finite-difference Jacobian approximation, column by column."""
    fv = f(t, y)
    J = np.empty((n, n), dtype=float)
    for i in range(n):
        delta = delta_base * max(1.0, abs(y[i]))
        e = np.zeros(n)
        e[i] = delta
        J[:, i] = (f(t, y + e) - fv) / delta
    return J, fv


def ode23s(u0, f, *args, **kwargs):
    t0 = kwargs.pop('t0', 0.0)
    te = kwargs.pop('te', 1.0)
    rtol = kwargs.pop('rtol', 1e-4)
    atol = kwargs.pop('atol', 1e-4)
    a_min = kwargs.pop('a_min', 0.1)
    a_max_explicit = kwargs.pop('a_max', 5.0)
    a_max_implicit = kwargs.pop('a_max_implicit', 2.0)
    safety = kwargs.pop('safety', 0.9)
    k_min = kwargs.pop('k_min', 1e-8)
    stepmax = kwargs.pop('stepmax', 10000)
    verbose = kwargs.pop('verbose', False)
    jac = kwargs.pop('jac', None)
    stiff_threshold = kwargs.pop('stiff_threshold', 15)
    nonstiff_threshold = kwargs.pop('nonstiff_threshold', 10)
    n_eqn = u0.size if isinstance(u0, np.ndarray) else 1

    msg = {0: 'Success!', 2: 'Step size dropped below kmin!',
           3: 'Required more than stepmax steps!'}
    idid = 0

    norm = lambda x: np.max(np.abs(x))

    k = kwargs.pop('k0', 1e-4 * (te - t0))

    v = np.asarray(u0, dtype=float)
    t = t0
    done = False

    # Stats
    fevals = 0
    jevals = 0
    lu_solves = 0
    rejected_steps = 0
    accepted_steps = 0

    delta_base = sqrt(np.finfo(float).eps)

    vs = []
    vs.append((t, np.copy(v)))

    # --- Mode tracking ---
    MODE_EXPLICIT = 0
    MODE_IMPLICIT = 1
    mode = MODE_EXPLICIT

    # Stiffness detection state
    stiff_counter = 0
    nonstiff_counter = 0
    step_outcomes = deque(maxlen=20)
    steps_since_switch = 0   # cooldown counter

    # FSAL state for BS3
    k1_fsal = f(t, v)
    fevals += 1

    # Rosenbrock state
    J = None
    h_old = 0.0
    F0_fsal = None  # FSAL state for Rosenbrock (F2 from prev step)

    # Diagnostics: record switch events
    switch_events = []  # list of (t, from_mode, to_mode)
    mode_history = []   # list of (t, mode) for each step

    while not done:
        if t + k >= te:
            k = te - t
            done = True
        else:
            k = min(k, 0.5 * (te - t))

        if k < k_min and not done:
            idid = 2
            break

        if accepted_steps + rejected_steps > stepmax:
            idid = 3
            break

        if mode == MODE_EXPLICIT:
            # ===== BS3 step =====
            a_max = a_max_explicit
            k1 = k1_fsal

            k2 = f(t + 0.5 * k, v + 0.5 * k * k1)
            k3 = f(t + 0.75 * k, v + 0.75 * k * k2)
            vnew = v + k * (2.0/9.0 * k1 + 1.0/3.0 * k2 + 4.0/9.0 * k3)
            k4 = f(t + k, vnew)
            fevals += 3

            err_vec = k * (_bs3_e1 * k1 + _bs3_e2 * k2
                           + _bs3_e3 * k3 + _bs3_e4 * k4)
            sc = atol + np.maximum(np.abs(vnew), np.abs(v)) * rtol
            err = max(np.finfo(float).eps, norm(err_vec / sc))
            knew = k * min(a_max, max(a_min,
                           safety * np.power(1.0/err, 1.0/3.0)))

            if err <= 1.0:
                # Stiffness detection
                dy = vnew - v
                dy_norm = norm(dy)
                if dy_norm > 0:
                    lam_est = norm(k4 - k1) / dy_norm
                    h_lam = k * lam_est
                    rej_ratio = (step_outcomes.count(False)
                                 / max(1, len(step_outcomes)))
                    is_stiff = (h_lam > 0.8 * _bs3_boundary
                                or (h_lam > 0.5 * _bs3_boundary
                                    and rej_ratio > 0.3))
                    if is_stiff:
                        stiff_counter = min(stiff_counter + 1,
                                            stiff_threshold + 5)
                    else:
                        stiff_counter = max(stiff_counter - 1, 0)

                step_outcomes.append(True)
                k1_fsal = k4
                v = vnew
                t = t + k
                vs.append((t, np.copy(v)))
                mode_history.append((t, mode))
                accepted_steps += 1
                steps_since_switch += 1

                # Check for switch to implicit
                if (stiff_counter >= stiff_threshold
                        and steps_since_switch >= stiff_threshold):
                    mode = MODE_IMPLICIT
                    stiff_counter = 0
                    nonstiff_counter = 0
                    steps_since_switch = 0
                    J = None  # force fresh Jacobian
                    F0_fsal = k1_fsal  # BS3's FSAL k4 = f(t,v) at switch
                    switch_events.append((t, MODE_EXPLICIT, MODE_IMPLICIT))
            else:
                done = False
                rejected_steps += 1
                step_outcomes.append(False)

            k = knew

        else:
            # ===== Rosenbrock 2(3) step (Shampine triple) =====
            a_max = a_max_implicit

            # Compute or reuse Jacobian
            need_jac = (J is None
                        or abs(k - h_old) > 0.2 * abs(h_old))
            if need_jac:
                if jac is not None:
                    J = jac(t, v)
                else:
                    J, _ = _fd_jacobian(f, t, v, n_eqn, delta_base)
                    fevals += n_eqn  # perturbation evals
                jevals += 1

            # F0 via FSAL or fresh evaluation
            if F0_fsal is not None:
                F0 = F0_fsal
            else:
                F0 = f(t, v)
                fevals += 1

            W = np.eye(n_eqn) - k * _d * J
            lu, piv = LA.lu_factor(W)

            # f_t approximation
            dt_delta = delta_base * max(1.0, abs(t))
            F0t = f(t + dt_delta, v)
            fevals += 1
            T = (F0t - F0) / dt_delta

            # Stage 1: W*k1 = F0 + h*d*T
            rhs1 = F0 + k * _d * T
            s1 = LA.lu_solve((lu, piv), rhs1)
            lu_solves += 1

            # Stage 2: k2 = W^{-1}(F1 - k1) + k1
            F1 = f(t + 0.5 * k, v + 0.5 * k * s1)
            fevals += 1
            z = LA.lu_solve((lu, piv), F1 - s1)
            lu_solves += 1
            s2 = z + s1

            # Solution (order 2, L-stable): y + h*k2
            vnew = v + k * s2

            # Stage 3 (for error estimate)
            F2 = f(t + k, vnew)
            fevals += 1
            rhs3 = F2 - _e32 * (s2 - F1) - 2.0 * (s1 - F0) + k * _d * T
            s3 = LA.lu_solve((lu, piv), rhs3)
            lu_solves += 1

            # Error estimate (order 3): (h/6)*(k1 - 2*k2 + k3)
            err_vec = (k / 6.0) * (s1 - 2.0 * s2 + s3)

            sc = atol + np.maximum(np.abs(vnew), np.abs(v)) * rtol
            err = max(np.finfo(float).eps, norm(err_vec / sc))
            knew = k * min(a_max, max(a_min,
                           safety * np.power(1.0 / err, 1.0 / 3.0)))

            if err <= 1.0:
                h_old = k
                v = vnew
                t = t + k
                vs.append((t, np.copy(v)))
                mode_history.append((t, mode))
                accepted_steps += 1

                # FSAL: reuse F2 as F0 for next step
                F0_fsal = F2

                steps_since_switch += 1

                # Periodically check if problem is no longer stiff.
                # Use the Jacobian (which we already have) to estimate
                # spectral radius via the infinity norm (upper bound).
                if (J is not None
                        and steps_since_switch >= nonstiff_threshold
                        and accepted_steps % 10 == 0):
                    # ||J||_inf is an upper bound on |lam_max|
                    lam_est = np.max(np.sum(np.abs(J), axis=1))
                    h_lam = k * lam_est
                    if h_lam < 0.3 * _bs3_boundary:
                        nonstiff_counter += 1
                    else:
                        nonstiff_counter = max(nonstiff_counter - 1, 0)

                    if nonstiff_counter >= nonstiff_threshold:
                        mode = MODE_EXPLICIT
                        nonstiff_counter = 0
                        stiff_counter = 0
                        steps_since_switch = 0
                        k1_fsal = f(t, v)
                        fevals += 1
                        F0_fsal = None
                        switch_events.append(
                            (t, MODE_IMPLICIT, MODE_EXPLICIT))
            else:
                done = False
                rejected_steps += 1
                J = None
                F0_fsal = None  # F0 still valid (haven't moved)

            k = knew

    if verbose:
        mode_names = {MODE_EXPLICIT: 'BS3', MODE_IMPLICIT: 'Rosenbrock'}
        print('Auto-switch BS3/Rosenbrock: {0:s}'.format(msg[idid]))
        print('Accepted steps =', accepted_steps,
              ' Rejected steps =', rejected_steps)
        print('Function evaluations = {:d}.'.format(fevals))
        if jevals > 0:
            print('Jacobian evaluations = {:d}.'.format(jevals))
            print('LU solves = {:d}.'.format(lu_solves))
        print('Switch events:')
        for (st, fm, tm) in switch_events:
            print(f'  t={st:.6f}: {mode_names[fm]} -> {mode_names[tm]}')
        if not switch_events:
            print('  (none)')

    result = np.array(vs, dtype=[('t', 'float'), ('u', 'f8', (n_eqn,))])

    diagnostics = {
        'accepted_steps': accepted_steps,
        'rejected_steps': rejected_steps,
        'fevals': fevals,
        'jevals': jevals,
        'lu_solves': lu_solves,
        'switch_events': switch_events,
        'mode_history': mode_history,
        'idid': idid,
        'msg': msg[idid],
    }

    return result, diagnostics
