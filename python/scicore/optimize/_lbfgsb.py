"""Pure-Python replacement for the deleted C ``_lbfgsb`` extension.

Implements the L-BFGS-B ``setulb`` kernel (limited-memory BFGS with box
constraints and a Wolfe line search) in pure Python, speaking the task
protocol used by ``scicore.optimize._lbfgsb_py.boundmin``:

- ``task[0] == 3`` (NEW_X): ``x`` holds a point at which the driver must
  compute ``f`` and ``g`` before the next call;
- ``task[0] == 1`` (NEW_POINT): a step was accepted, a new iteration
  begins (the ``f``/``g`` passed in are valid at the current ``x``);
- ``task[0] == 4`` (DONE): converged, ``task[1]`` carries the code;
- ``task[0] == 5`` (ERROR/STOP): terminated, ``task[1]`` carries the code;
  a call that finds ``task[0] == 5`` (driver-requested halt) returns at
  once without changing state.

Between calls the driver re-evaluates ``f`` and ``g``, so the solver state
(L-BFGS memory, line search progress) is kept per workspace, keyed on
``id(wa)``.  The driver reads the s/y update buffers from ``wa`` (most
recent updates in the leading rows) and the total update count from
``isave[30]``.
"""

from __future__ import annotations

import math

import rsnumpy as np

__all__ = ['setulb']

_STATES = {}


def _vec(a):
    return [float(v) for v in np.asarray(a).ravel().tolist()]


def _scalar(v):
    return float(np.asarray(v).ravel().tolist()[0])


def _dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _project(x, nbd, low, up, n):
    """Clip ``x`` into the box described by ``nbd`` (0 free, 1 lower,
    2 both bounds, 3 upper)."""
    out = list(x)
    for k in range(n):
        if nbd[k] in (1, 2) and out[k] < low[k]:
            out[k] = low[k]
        if nbd[k] in (2, 3) and out[k] > up[k]:
            out[k] = up[k]
    return out


def _trial(st, p, base, alpha):
    """Candidate point ``base + alpha * p``, clipped into the box."""
    n = st['n']
    out = []
    for k in range(n):
        v = base[k] + alpha * p[k]
        if st['nbd'][k] in (1, 2) and v < st['low'][k]:
            v = st['low'][k]
        if st['nbd'][k] in (2, 3) and v > st['up'][k]:
            v = st['up'][k]
        out.append(v)
    return out


def _write_x(x, values):
    for k in range(len(values)):
        x[k] = values[k]


def _pgnorm(st, g):
    """Infinity norm of the projected gradient (bound-aware).

    ``nbd`` follows the boundmin convention: 0 free, 1 lower, 2 both
    bounds, 3 upper (a fixed variable is a box with equal bounds).
    """
    n = st['n']
    m = 0.0
    for k in range(n):
        at_lower = (st['nbd'][k] in (1, 2)
                    and st['x'][k] <= st['low'][k] + 1e-14)
        at_upper = (st['nbd'][k] in (2, 3)
                    and st['x'][k] >= st['up'][k] - 1e-14)
        if (at_lower and g[k] >= 0.0) or (at_upper and g[k] <= 0.0):
            continue
        a = abs(g[k])
        if a > m:
            m = a
    return m


def _direction(st, g):
    """L-BFGS search direction: two-loop recursion, ``p = -H g`` with the
    components pointing out of the box zeroed (simplified L1 handling)."""
    n = st['n']
    pairs = st['pairs']
    rho = st['rho']
    cnt = len(pairs)
    q = list(g)
    alpha = [0.0] * cnt
    for i in range(cnt - 1, -1, -1):
        s_i, y_i = pairs[i]
        ai = rho[i] * _dot(s_i, q)
        alpha[i] = ai
        for k in range(n):
            q[k] -= ai * y_i[k]
    r = list(q)
    for i in range(cnt):
        s_i, y_i = pairs[i]
        bi = rho[i] * _dot(y_i, r)
        for k in range(n):
            r[k] += (alpha[i] - bi) * s_i[k]
    p = [-st['gamma'] * v for v in r]
    for k in range(n):
        at_lower = (st['nbd'][k] in (1, 2)
                    and st['x'][k] <= st['low'][k] + 1e-14)
        at_upper = (st['nbd'][k] in (2, 3)
                    and st['x'][k] >= st['up'][k] - 1e-14)
        if (at_lower and p[k] < 0.0) or (at_upper and p[k] > 0.0):
            p[k] = 0.0
    return p


def _store_memory(wa, m, st):
    """Write the most recent s/y pairs (newest first) into the ``wa``
    buffers the driver reshapes to (m, n)."""
    n = st['n']
    pairs = st['pairs'][-m:]
    for i in range(len(pairs)):
        s_i, y_i = pairs[len(pairs) - 1 - i]
        for k in range(n):
            wa[i * n + k] = s_i[k]
            wa[m * n + i * n + k] = y_i[k]


def setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol, wa, iwa,
           task, lsave, isave, dsave, maxls, ln_task):
    """One step of the L-BFGS-B ``setulb`` task protocol (see module
    docstring).  ``x`` is updated in place; ``task`` carries the status."""
    key = id(wa)
    st = _STATES.get(key)
    if st is None:
        x0 = _vec(x)
        n = len(x0)
        low = _vec(low_bnd)
        up = _vec(upper_bnd)
        nb = _vec(nbd)
        st = {
            'm': m, 'n': n, 'low': low, 'up': up, 'nbd': nb,
            'x': _project(x0, nb, low, up, n),
            'pairs': [], 'rho': [], 'gamma': 1.0,
            'total_updates': 0, 'phase': 'iter', 'ls': None,
        }
        _STATES[key] = st
    try:
        task0 = int(np.asarray(task[0]).ravel().tolist()[0])
        if task0 == 5:
            # Driver-requested halt: keep the status, do no work.
            return 0
        if task0 == 0:
            # First call: present the projected start and request f, g.
            _write_x(x, st['x'])
            task[0] = 3
            task[1] = 0
            return 0
        # task0 in (1, 3): f and g are valid at the current x.
        f_val = _scalar(f)
        g_val = _vec(g)
        if math.isnan(f_val) or any(math.isnan(v) for v in g_val):
            task[0] = 5
            task[1] = 601
            return 0
        if st['phase'] == 'iter':
            st['x'] = _vec(x)
            if _pgnorm(st, g_val) <= pgtol:
                task[0] = 4
                task[1] = 401
                return 0
            p = _direction(st, g_val)
            if max(abs(v) for v in p) <= 1e-14:
                task[0] = 4
                task[1] = 401
                return 0
            # Largest step that stays inside the box.
            amax = 1.0
            for k in range(st['n']):
                pk = p[k]
                if pk < 0.0 and st['nbd'][k] in (1, 2):
                    amax = min(amax, (st['low'][k] - st['x'][k]) / pk)
                elif pk > 0.0 and st['nbd'][k] in (2, 3):
                    amax = min(amax, (st['up'][k] - st['x'][k]) / pk)
            if amax <= 0.0:
                task[0] = 4
                task[1] = 401
                return 0
            st['ls'] = {
                'base_x': list(st['x']), 'base_g': list(g_val),
                'base_f': f_val, 'p': p,
                'alpha': min(1.0, amax), 'amax': amax, 'count': 0,
            }
            st['phase'] = 'ls'
            _write_x(x, _trial(st, p, st['ls']['base_x'], st['ls']['alpha']))
            task[0] = 3
            task[1] = 0
            return 0
        # phase == 'ls': evaluate the candidate against the Wolfe tests.
        ls = st['ls']
        ls['count'] += 1
        trial = _trial(st, ls['p'], ls['base_x'], ls['alpha'])
        gbp = _dot(ls['base_g'], ls['p'])
        armijo = f_val <= ls['base_f'] + 1e-4 * ls['alpha'] * gbp
        curl = _dot(g_val, ls['p']) >= 0.9 * gbp
        if armijo and (curl or ls['alpha'] >= 0.9999 * ls['amax']):
            s_new = [trial[k] - ls['base_x'][k] for k in range(st['n'])]
            y_new = [g_val[k] - ls['base_g'][k] for k in range(st['n'])]
            sy = _dot(s_new, y_new)
            st['x'] = trial
            st['phase'] = 'iter'
            st['ls'] = None
            if sy > 1e-16:
                st['pairs'].append((s_new, y_new))
                st['rho'].append(1.0 / sy)
                st['gamma'] = sy / max(_dot(y_new, y_new), 1e-300)
                st['total_updates'] += 1
                isave[30] = st['total_updates']
                if len(st['pairs']) > 10 * st['m']:
                    st['pairs'] = st['pairs'][-st['m']:]
                    st['rho'] = st['rho'][-st['m']:]
                _store_memory(wa, st['m'], st)
            task[0] = 1
            task[1] = 0
            return 0
        if ls['count'] >= maxls or ls['alpha'] < 1e-12:
            task[0] = 5
            task[1] = 601
            return 0
        ls['alpha'] *= 0.5
        _write_x(x, _trial(st, ls['p'], ls['base_x'], ls['alpha']))
        task[0] = 3
        task[1] = 0
        return 0
    finally:
        if int(np.asarray(task[0]).ravel().tolist()[0]) in (4, 5):
            _STATES.pop(key, None)
