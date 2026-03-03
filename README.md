# ode-integrate

Adaptive ODE integrators for teaching. Written in pure Python (NumPy/SciPy) so
students can read, modify, and understand every line.

## Installation

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from integrate import rk2_adaptive, be2_adaptive

def f(t, u):
    return -2.0 * u

u0 = 1.0
data = rk2_adaptive(u0, f, te=5.0, k0=0.1, atol=1e-4, rtol=1e-4)

ts = data['t']
us = data['u'].flatten()
```

All adaptive integrators share the same interface:

```python
data = method(u0, f, te=..., k0=..., atol=..., rtol=...)
```

and return a structured NumPy array with fields `'t'` (time) and `'u'`
(solution).

## Available methods

### Fixed-step methods

Simple implementations for comparison and teaching. These return a plain NumPy
array of solution values.

| Function | Method | Order |
|---|---|---|
| `fwd_euler(u0, f, k, numsteps)` | Forward Euler | 1 |
| `rk2(u0, f, k, numsteps)` | Explicit RK2 (midpoint) | 2 |
| `pece(u0, f, k, numsteps)` | Euler predictor, trapezoidal corrector | 2 |
| `adams_bashforth(u0, f, k, numsteps)` | Adams-Bashforth 2-step | 2 |
| `adams_pece(u0, f, k, numsteps)` | Adams predictor-corrector (PECE) | 2 |

### Adaptive explicit methods

| Function | Method | Order | Error estimator |
|---|---|---|---|
| `rk2_adaptive` | RK1(2) midpoint | 2 | Embedded Euler/RK2 pair |
| `rk3_adaptive` | RK2(3) | 3 | Embedded 2nd/3rd order pair |
| `bs3_adaptive` | Bogacki-Shampine 3(2) | 3 | FSAL with stiffness detection |
| `rk5_adaptive` | RK4(5) | 5 | Embedded 4th/5th order pair |
| `pc_adaptive` | PECE (Euler/trapezoidal) | 2 | Predictor-corrector difference |

### Adaptive implicit methods

| Function | Method | Order | Notes |
|---|---|---|---|
| `be1_adaptive` | Backward Euler + Richardson (scalar) | 2 | Simplified Newton, scalar division |
| `be2_adaptive` | Backward Euler + Richardson (systems) | 2 | Simplified Newton, LU factorization |
| `ros2_adaptive` | Rosenbrock 2(3) | 2 | L-stable, Jacobian-based |

### Auto-switching

| Function | Description |
|---|---|
| `ode23s` | Switches between BS3 (explicit) and Rosenbrock (implicit) based on stiffness detection |

## Common parameters

All adaptive methods accept:

| Parameter | Default | Description |
|---|---|---|
| `t0` | `0.0` | Start time |
| `te` | `1.0` | End time |
| `k0` | `1e-4 * (te - t0)` | Initial step size |
| `atol` | `1e-4` | Absolute error tolerance |
| `rtol` | `1e-4` | Relative error tolerance |
| `k_min` | `1e-8` | Minimum step size before failure |
| `stepmax` | `10000` | Maximum number of steps |
| `safety` | `0.9` | Safety factor for step-size control |
| `a_min` | `0.1` | Minimum step-size ratio |
| `a_max` | varies | Maximum step-size ratio |

## License

MIT
