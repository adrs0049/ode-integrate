#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
"""
Modified Rosenbrock triple — order 2(3), L-stable.

Based on the "modified Rosenbrock triple" from:
    Shampine & Reichelt (1997), "The MATLAB ODE Suite",
    SIAM J. Sci. Comput., 18(1), 1-22.  (Section 4.1)

This is the method underlying MATLAB's ode23s.  The integration advances
with an L-stable order-2 formula; a third stage provides an A-stable
order-3 embedded estimate used only for step-size control.

Parameters:
    d   = 1/(2 + sqrt(2))   ≈ 0.2929   (ensures L-stability)
    e32 = 6 + sqrt(2)       ≈ 7.4142

Stage equations (W = I - h*d*J,  T ≈ df/dt):

    F0 = f(t_n, y_n)                                   [FSAL from previous]
    k1 = W \\ (F0 + h*d*T)
    F1 = f(t_n + h/2, y_n + h*k1/2)
    k2 = W \\ (F1 - k1) + k1
    y_{n+1} = y_n + h*k2                               [order 2, L-stable]
    F2 = f(t_{n+1}, y_{n+1})                            [FSAL → F0 next step]
    k3 = W \\ [F2 - e32*(k2 - F1) - 2*(k1 - F0) + h*d*T]
    err ≈ (h/6)*(k1 - 2*k2 + k3)                       [order 3 estimate]

Cost per accepted step:
  - 1 Jacobian evaluation (or reuse)
  - 1 LU factorization of W (n × n)
  - 3 forward/backward solves
  - 2 new f-evaluations (FSAL saves one) + 1 f_t probe
"""
import numpy as np
from math import sqrt
from scipy import linalg as LA


# Method parameters
_d = 1.0 / (2.0 + sqrt(2.0))     # ≈ 0.29289322
_e32 = 6.0 + sqrt(2.0)           # ≈ 7.41421356


def _fd_jacobian(f, t, y, n, delta_base):
    """Finite-difference Jacobian approximation, column by column."""
    fv = f(t, y)
    J = np.empty((n, n), dtype=float)
    for i in range(n):
        delta = delta_base * max(1.0, abs(y[i]))
        e = np.zeros(n)
        e[i] = delta
        J[:, i] = (f(t, y + e) - fv) / delta
    return J, fv  # return fv to avoid recomputing


def ros2_adaptive(u0, f, *args, **kwargs):
    t0 = kwargs.pop('t0', 0.0)
    te = kwargs.pop('te', 1.0)
    rtol = kwargs.pop('rtol', 1e-4)
    atol = kwargs.pop('atol', 1e-4)
    a_min = kwargs.pop('a_min', 0.1)
    a_max = kwargs.pop('a_max', 2.0)
    safety = kwargs.pop('safety', 0.9)
    k_min = kwargs.pop('k_min', 1e-8)
    stepmax = kwargs.pop('stepmax', 10000)
    verbose = kwargs.pop('verbose', False)
    jac = kwargs.pop('jac', None)  # optional analytical Jacobian
    n_eqn = u0.size if isinstance(u0, np.ndarray) else 1

    # Dummy solout
    def solout(*args):
        return False
    solout = kwargs.pop('solout', solout)

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

    # Jacobian state for reuse
    J = None
    h_old = 0.0

    # FSAL state: F0 = f(t, v)
    F0 = f(t, v)
    fevals += 1

    while not done:
        if t + k >= te:
            k = te - t
            done = True
        else:
            k = min(k, 0.5 * (te - t))

        # --- Compute or reuse Jacobian ---
        need_jac = (J is None
                    or abs(k - h_old) > 0.2 * abs(h_old))

        if need_jac:
            if jac is not None:
                J = jac(t, v)
            else:
                J, F0 = _fd_jacobian(f, t, v, n_eqn, delta_base)
                fevals += n_eqn  # n perturbation evals (base eval updates F0)
            jevals += 1

        # --- Form and factorize W = I - h*d*J ---
        W = np.eye(n_eqn) - k * _d * J
        lu, piv = LA.lu_factor(W)

        # --- Approximate df/dt via finite difference ---
        dt_delta = delta_base * max(1.0, abs(t))
        F0t = f(t + dt_delta, v)
        fevals += 1
        T = (F0t - F0) / dt_delta

        # --- Stage 1: W*k1 = F0 + h*d*T ---
        rhs1 = F0 + k * _d * T
        k1 = LA.lu_solve((lu, piv), rhs1)
        lu_solves += 1

        # --- Stage 2: k2 = W^{-1}(F1 - k1) + k1 ---
        F1 = f(t + 0.5 * k, v + 0.5 * k * k1)
        fevals += 1
        z = LA.lu_solve((lu, piv), F1 - k1)
        lu_solves += 1
        k2 = z + k1

        # --- Solution (order 2, L-stable): y + h*k2 ---
        vnew = v + k * k2

        # --- Stage 3 (for error estimate only) ---
        F2 = f(t + k, vnew)
        fevals += 1
        rhs3 = F2 - _e32 * (k2 - F1) - 2.0 * (k1 - F0) + k * _d * T
        k3 = LA.lu_solve((lu, piv), rhs3)
        lu_solves += 1

        # --- Error estimate (order 3): (h/6)*(k1 - 2*k2 + k3) ---
        err_vec = (k / 6.0) * (k1 - 2.0 * k2 + k3)

        sc = atol + np.maximum(np.abs(vnew), np.abs(v)) * rtol
        err = max(np.finfo(float).eps, norm(err_vec / sc))

        # Step-size: 2(3) pair, exponent = 1/(p+1) = 1/3
        knew = k * min(a_max, max(a_min,
                       safety * np.power(1.0 / err, 1.0 / 3.0)))

        if k < k_min and not done:
            idid = 2
            break

        if accepted_steps + rejected_steps > stepmax:
            idid = 3
            break

        if err <= 1.0:
            done = done or solout(v, vnew, t, t + k)

            h_old = k
            v = vnew
            t = t + k
            vs.append((t, np.copy(v)))
            accepted_steps += 1

            # FSAL: reuse F2 as F0 for next step
            F0 = F2
        else:
            done = False
            rejected_steps += 1
            J = None  # force Jacobian recomputation after rejection
            # F0 stays valid (we haven't moved)

        k = knew

    if verbose:
        print('Rosenbrock 2(3) (L-stable, Shampine): {0:s}'.format(msg[idid]))
        print('Accepted steps =', accepted_steps,
              ' Rejected steps =', rejected_steps)
        print('Function evaluations = {:d}.'.format(fevals))
        print('Jacobian evaluations = {:d}.'.format(jevals))
        print('LU solves = {:d}.'.format(lu_solves))

    result = np.array(vs, dtype=[('t', 'float'), ('u', 'f8', (n_eqn,))])

    diagnostics = {
        'accepted_steps': accepted_steps,
        'rejected_steps': rejected_steps,
        'fevals': fevals,
        'jevals': jevals,
        'lu_solves': lu_solves,
        'idid': idid,
        'msg': msg[idid],
    }

    return result, diagnostics
