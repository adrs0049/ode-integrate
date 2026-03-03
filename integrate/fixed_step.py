#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
"""Fixed-step ODE integrators for teaching and comparison."""
import numpy as np


def fwd_euler(u0, f, k, numsteps, t0=0.0):
    """Forward Euler method with fixed step size."""
    vs = np.empty(numsteps + 1, dtype=float)
    vs[0] = u0

    v = u0
    for i in range(numsteps):
        t = t0 + k * i
        v = v + k * f(t, v)
        vs[i + 1] = v

    return vs


def rk2(u0, f, k, numsteps, t0=0.0):
    """Explicit RK2 (midpoint) method with fixed step size."""
    vs = np.empty(numsteps + 1, dtype=float)
    vs[0] = u0

    # Method constants (midpoint rule)
    a = 0.0
    b = 1.0
    alpha = 0.5
    beta = 0.5

    v = u0
    for i in range(numsteps):
        t = t0 + k * i
        r1 = f(t, v)
        r2 = f(t + alpha * k, v + beta * k * r1)
        v = v + k * (a * r1 + b * r2)
        vs[i + 1] = v

    return vs


def adams_bashforth(u0, f, k, numsteps, t0=0.0):
    """Adams-Bashforth 2-step method. Uses one Euler step to start."""
    vs = np.empty(numsteps + 1, dtype=float)
    vs[0] = u0

    # 1. Do one step of Euler to start
    vs[1] = u0 + k * f(t0, u0)

    for i in range(1, numsteps):
        t_i = t0 + i * k
        t_im1 = t0 + (i - 1) * k
        vs[i + 1] = vs[i] + 1.5 * k * f(t_i, vs[i]) - 0.5 * k * f(t_im1, vs[i - 1])

    return vs


def pece(u0, f, k, numsteps, t0=0.0):
    """Predictor-corrector (PECE) using Euler predictor and trapezoidal corrector."""
    vs = np.empty(numsteps + 1, dtype=float)
    vs[0] = u0

    v = u0
    for i in range(numsteps):
        t = t0 + k * i
        # 1. Predictor (forward Euler)
        ftv = f(t, v)
        y_star = v + k * ftv

        # 2. Corrector (trapezoidal rule)
        t_new = t + k
        v = v + 0.5 * k * (ftv + f(t_new, y_star))

        # 3. Store result
        vs[i + 1] = v

    return vs


def adams_pece(u0, f, k, numsteps, t0=0.0):
    """Adams predictor-corrector (PECE). Uses one PECE Euler step to start."""
    vs = np.empty(numsteps + 1, dtype=float)
    vs[0] = u0

    # 1. Do one step of PECE Euler to start
    y_star = u0 + k * f(t0, u0)
    vs[1] = vs[0] + 0.5 * k * (f(t0, u0) + f(t0 + k, y_star))

    for i in range(1, numsteps):
        t_i = t0 + i * k
        t_im1 = t0 + (i - 1) * k
        t_ip1 = t0 + (i + 1) * k
        y_star = vs[i] + 1.5 * k * f(t_i, vs[i]) - 0.5 * k * f(t_im1, vs[i - 1])
        vs[i + 1] = vs[i] + 0.5 * k * (f(t_ip1, y_star) + f(t_i, vs[i]))

    return vs
