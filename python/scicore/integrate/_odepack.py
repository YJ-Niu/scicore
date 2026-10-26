"""Stub replacement for the deleted C ``_odepack`` extension.

Provides the ODEPACK solver interface used by ``scipy.integrate._odepack_py``.
This is a stub: real ODE solvers should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['odeint', 'ODEintWarning', 'lsoda', 'zvode', 'dvode']


class ODEintWarning(Warning):
    pass


def odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, ml=None, mu=None,
           full_output=0, rtol=None, atol=None, tcrit=None, h0=0.0,
           hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
           mxords=5, printmessg=0, tfirst=False):
    """Pure-Python adaptive Runge-Kutta integrator (RK4 step doubling).

    Replaces the ODEPACK `odeint` stub.  Returns ``[Y, ier]``
    (``[Y, info, ier]`` when ``full_output`` is truthy), the convention
    expected by ``_odepack_py.odeint``.
    """
    if rtol is None:
        rtol = 1.49e-8
    if atol is None:
        atol = 1e-6

    tl = [float(v) for v in np.asarray(t).tolist()]
    y = [float(v) for v in y0]
    n = len(y)

    def f(t_val, yv):
        if tfirst:
            out = func(t_val, yv, *args)
        else:
            out = func(yv, t_val, *args)
        return [float(v) for v in out]

    def rk4(t_val, yv, h):
        k1 = f(t_val, yv)
        k2 = f(t_val + 0.5 * h, [yv[i] + 0.5 * h * k1[i] for i in range(n)])
        k3 = f(t_val + 0.5 * h, [yv[i] + 0.5 * h * k2[i] for i in range(n)])
        k4 = f(t_val + h, [yv[i] + h * k3[i] for i in range(n)])
        return [yv[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
                for i in range(n)]

    rows = [y[:]]
    for span in range(len(tl) - 1):
        t0, t1 = tl[span], tl[span + 1]
        if t1 == t0:
            rows.append(y[:])
            continue
        target = t1 - t0
        h = h0 if h0 != 0.0 else target / 10.0
        if abs(h) >= abs(target):
            h = target
        t_val = t0
        steps = 0
        limit = mxstep if mxstep > 0 else 100000
        while abs(t1 - t_val) > 1e-15 * max(1.0, abs(t1)):
            remaining = t1 - t_val
            if abs(h) > abs(remaining):
                h = remaining
            y1 = rk4(t_val, y, h)
            ya = rk4(t_val, y, 0.5 * h)
            y2 = rk4(t_val + 0.5 * h, ya, 0.5 * h)
            # Step doubling: y2 - y1 overestimates the local error by ~15x
            err = max(
                abs(y2[i] - y1[i])
                / (15.0 * (atol + rtol * max(abs(y[i]), abs(y2[i]))))
                for i in range(n))
            if err <= 1.0:
                t_val += h
                y = y2
                h *= 5.0 if err == 0.0 else 0.9 * err ** (-0.2)
            else:
                h *= 0.9 * err ** (-0.2)
            if hmax > 0.0:
                h = min(h, hmax) if h > 0 else max(h, -hmax)
            if hmin > 0.0 and abs(h) < hmin:
                h = hmin if h > 0 else -hmin
            steps += 1
            if steps >= limit:
                break
        rows.append(y[:])

    Y = np.array(rows)
    ier = 0
    if full_output:
        info = {'t': np.asarray(tl), 'y': Y, 'h': 0.0, 'message': ''}
        return [Y, info, ier]
    return [Y, ier]


def lsoda(*args, **kwargs):
    """Stub for lsoda."""
    raise NotImplementedError("lsoda not yet ported to Rust/rsnumpy")


def zvode(*args, **kwargs):
    """Stub for zvode."""
    raise NotImplementedError("zvode not yet ported to Rust/rsnumpy")


def dvode(*args, **kwargs):
    """Stub for dvode."""
    raise NotImplementedError("dvode not yet ported to Rust/rsnumpy")
