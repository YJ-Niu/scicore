"""Pure-Python replacement for the deleted C ``_fitpack`` extension.

Provides the FITPACK spline routines used by ``_fitpack_impl`` /
``_fitpack2``:

* Fitting: ``curfit`` delegates to the pure-Python smoothing spline fitter
  (``_fitpack_repro.make_splrep``); ``regrid`` to the pure-Python bivariate
  fitter (``_regrid._regrid``).
* Evaluation: ``splev`` / ``splder`` / ``bispev`` / ``parder`` /
  ``bispeu`` / ``pardeu`` are hand-rolled de Boor evaluators in pure Python
  (no rsnumpy arithmetic, so no backend quirks).
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['splev', 'splder', 'bispev', 'bispeu', 'parder', 'pardeu',
           'curfit', 'regrid']


# ---------------------------------------------------------------------------
# B-spline evaluation primitives (pure Python floats)
# ---------------------------------------------------------------------------

def _knots_list(t):
    return [float(v) for v in np.asarray(t).tolist()]


def _coeffs_list(c):
    return [float(v) for v in np.asarray(c).ravel().tolist()]


def _points(v):
    return [float(q) for q in np.asarray(v).ravel().tolist()]


def _find_span(t, k, n, x):
    """Index i of the knot span [t[i], t[i+1]) containing x."""
    if x <= t[k]:
        return k
    if x >= t[n]:
        return n - 1
    lo, hi = k, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x < t[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def _basis(t, k, i, x):
    """Non-vanishing B-spline basis values at span i.

    Returns ``[(idx, value), ...]`` for ``N_{i-k+j, k}(x)``, j = 0..k, via
    the Cox-de Boor recursion with the 0/0 := 0 convention.  ``i`` is the
    knot span with ``t[i] <= x < t[i+1]``; at the right endpoint of a
    clamped vector only the last basis function is non-zero.
    """
    if x == t[-1]:
        return [(i - k + j, 1.0 if j == k else 0.0) for j in range(k + 1)]
    V = [[0.0] * (k + 1) for _ in range(k + 1)]
    for q in range(k + 1):
        p0 = i - k + q
        if t[p0] <= x < t[p0 + 1]:
            V[q][0] = 1.0
    for deg in range(1, k + 1):
        for q in range(k + 1):
            p0 = i - k + q
            d1 = t[p0 + deg] - t[p0]
            term1 = ((x - t[p0]) * V[q][deg - 1]) / d1 if d1 != 0.0 else 0.0
            if q < k:
                d2 = t[p0 + deg + 1] - t[p0 + 1]
                term2 = ((t[p0 + deg + 1] - x) * V[q + 1][deg - 1]) / d2 \
                    if d2 != 0.0 else 0.0
            else:
                term2 = 0.0
            V[q][deg] = term1 + term2
    return [(i - k + q, V[q][k]) for q in range(k + 1)]


def _eval1d(t, c, k, x, ext=0):
    """Evaluate a 1-D B-spline (with local-polynomial extension) at x."""
    n = len(c)
    tlo, thi = t[k], t[-k - 1]
    if x < tlo or x > thi:
        if ext == 1:
            return 0.0
        if ext == 2:
            raise ValueError(f"{x} lies outside the domain of the spline")
        if ext == 3:
            x = tlo if x < tlo else thi
    s = 0.0
    for idx, val in _basis(t, k, _find_span(t, k, n, x), x):
        if 0 <= idx < n:
            s += c[idx] * val
    return s


def _solve_square(N, b):
    """Solve N c = b (N square) via Gaussian elimination, partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(N)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            raise ValueError("singular collocation matrix in curfit")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for cc in range(col, n + 1):
                M[r][cc] -= f * M[col][cc]
    c = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * c[j]
        c[i] = s / M[i][i]
    return c


def _deriv1d(t, c, k, d):
    """Differentiate a 1-D B-spline d times; returns (t, c, k)."""
    for _ in range(d):
        c = [k * (c[i] - c[i - 1]) / (t[i + k] - t[i])
             for i in range(1, len(c))]
        t = t[1:-1]
        k -= 1
    return t, c, k


def _c2d(c_flat, ncx, ncy):
    """FITPACK flat order: ``c_flat[j * ncx + i] = c2[i, j]``."""
    return [[c_flat[j * ncx + i] for i in range(ncx)] for j in range(ncy)]


def _eval2d_der(tx, ty, c_flat, kx, ky, dx, dy, x, y):
    """Tensor-product B-spline evaluation with partial derivatives."""
    ncx = len(tx) - kx - 1
    ncy = len(ty) - ky - 1
    c2 = _c2d(c_flat, ncx, ncy)
    for _ in range(dx):
        c2 = [[kx * (row[i] - row[i - 1]) / (tx[i + kx] - tx[i])
               for i in range(1, len(row))] for row in c2]
        tx = tx[1:-1]
        kx -= 1
    for _ in range(dy):
        old = c2
        c2 = [[0.0] * len(old[0]) for _ in range(len(old) - 1)]
        for j in range(1, len(old)):
            for i in range(len(old[0])):
                c2[j - 1][i] = (ky * (old[j][i] - old[j - 1][i])
                                / (ty[j + ky] - ty[j]))
        ty = ty[1:-1]
        ky -= 1
    s = 0.0
    for ix, ax in _basis(tx, kx, _find_span(tx, kx, len(c2[0]), x), x):
        if not (0 <= ix < len(c2[0])):
            continue
        for jy, by in _basis(ty, ky, _find_span(ty, ky, len(c2), y), y):
            if 0 <= jy < len(c2):
                s += c2[jy][ix] * ax * by
    return s


# ---------------------------------------------------------------------------
# 1-D evaluation (FITPACK SPLEV / SPLDER)
# ---------------------------------------------------------------------------

def splev(t, c, k, x, ext=0):
    t = _knots_list(t)
    c = _coeffs_list(c)
    k = int(k)
    y = [_eval1d(t, c, k, q, int(ext)) for q in _points(x)]
    return np.array(y, dtype=float), 0


def splder(t, c, k, der, x, ext=0):
    t = _knots_list(t)
    c = _coeffs_list(c)
    k = int(k)
    if int(der) > 0:
        t, c, k = _deriv1d(t, c, k, int(der))
    y = [_eval1d(t, c, k, q, int(ext)) for q in _points(x)]
    return np.array(y, dtype=float), 0


# ---------------------------------------------------------------------------
# 2-D evaluation (FITPACK BISPEV / PARDER and scattered variants)
# ---------------------------------------------------------------------------

def _bis_eval(tx, ty, c, kx, ky, dx, dy, x, y, grid):
    tx = _knots_list(tx)
    ty = _knots_list(ty)
    c_flat = _coeffs_list(c)
    kx, ky, dx, dy = int(kx), int(ky), int(dx), int(dy)
    xpts = _points(x)
    ypts = _points(y)
    if grid:
        z = [_eval2d_der(tx, ty, c_flat, kx, ky, dx, dy, xv, yv)
             for xv in xpts for yv in ypts]
        return np.array(z, dtype=float).reshape((len(xpts), len(ypts))), 0
    z = [_eval2d_der(tx, ty, c_flat, kx, ky, dx, dy, xv, yv)
         for xv, yv in zip(xpts, ypts)]
    return np.array(z, dtype=float), 0


def bispev(tx, ty, c, kx, ky, x, y):
    return _bis_eval(tx, ty, c, kx, ky, 0, 0, x, y, True)


def parder(tx, ty, c, kx, ky, dx, dy, x, y):
    return _bis_eval(tx, ty, c, kx, ky, dx, dy, x, y, True)


def bispeu(tx, ty, c, kx, ky, x, y):
    return _bis_eval(tx, ty, c, kx, ky, 0, 0, x, y, False)


def pardeu(tx, ty, c, kx, ky, dx, dy, x, y):
    return _bis_eval(tx, ty, c, kx, ky, dx, dy, x, y, False)


# ---------------------------------------------------------------------------
# Fitting (delegated to the pure-Python fitters)
# ---------------------------------------------------------------------------

def curfit(task, x, y, w, xb, xe, k, s, t, wrk, iwrk):
    """Pure-Python replacement for FITPACK CURF11.

    Mirrors the (task, x, y, w, xb, xe, k, s, t, wrk, iwrk) call convention
    used by ``_fitpack2._curfit`` and returns ``(n, t, c, fp, ier)``.
    """
    from . import _fitpack_repro

    k = int(k)
    s = 0.0 if s is None else float(s)
    xf = _points(x)
    yf = _points(y)

    if s == 0.0 and len(xf) == k + 1:
        # FITPACK rule: with exactly k+1 data points the interpolant is a
        # polynomial — use the minimal clamped knot vector and solve the
        # square collocation system directly.
        xb_f = float(np.asarray(xb).tolist()) if xb is not None else min(xf)
        xe_f = float(np.asarray(xe).tolist()) if xe is not None else max(xf)
        t_full = [xb_f] * (k + 1) + [xe_f] * (k + 1)
        m = len(xf)
        N = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for idx, val in _basis(t_full, k, _find_span(t_full, k, m, xf[i]),
                                   xf[i]):
                if 0 <= idx < m:
                    N[i][idx] = val
        c_full = _solve_square(N, yf)
        nk = k
    else:
        # `_curfit` converts w=None to ones before the call; make_splrep
        # rejects weights for s==0, and equal weights are equivalent to None.
        w_eff = None if s == 0.0 else w
        spl = _fitpack_repro.make_splrep(x, y, w=w_eff, xb=xb, xe=xe, k=k, s=s)
        tt = spl.t
        if isinstance(tt, (tuple, list)):
            tt = tt[0]
        t_full = _knots_list(tt)
        c_full = _coeffs_list(spl.c)
        nk = int(spl.k)

    nest = len(t) if t is not None and len(t) > 0 else len(c_full) + nk + 1
    t_out = list(t_full) + [t_full[-1]] * max(0, nest - len(t_full))
    c_out = list(c_full) + [0.0] * max(0, nest - nk - 1 - len(c_full))

    wf = None if w is None else _points(w)
    if wf is None:
        fp = sum((yf[i] - _eval1d(t_full, c_full, nk, xf[i], 0)) ** 2
                 for i in range(len(xf)))
    else:
        fp = sum((wf[i] * (yf[i] - _eval1d(t_full, c_full, nk, xf[i], 0))) ** 2
                 for i in range(len(xf)))

    return (len(t_out), np.array(t_out, dtype=float),
            np.array(c_out, dtype=float), fp, 0)


def regrid(iopt, x, y, z, xb, xe, yb, ye, kx, ky, s, nxest, nyest, maxit,
           wrk, iwrk):
    """Pure-Python replacement for FITPACK REGRID.

    Mirrors the call convention used by ``_fitpack2._regrid_smth`` (z is a
    flattened (len(x), len(y)) grid) and returns
    ``(nx, tx, ny, ty, c, fp, ier)`` with c in FITPACK flat order.
    """
    from . import _regrid as _regrid_mod

    kx, ky = int(kx), int(ky)
    s = 0.0 if s is None else float(s)
    mx, my = len(x), len(y)
    z2 = np.asarray(z, dtype=float).reshape(mx, my)
    spl = _regrid_mod._regrid(x, y, z2, bbox=[xb, xe, yb, ye],
                              kx=kx, ky=ky, s=s, maxit=maxit)
    tx = _knots_list(spl.t[0])
    ty = _knots_list(spl.t[1])
    c2l = np.asarray(spl.c, dtype=float).tolist()
    ncx = len(c2l[0])
    ncy = len(c2l)
    c_flat = [c2l[j][i] for j in range(ncy) for i in range(ncx)]

    xf = _points(x)
    yf = _points(y)
    zf = np.asarray(z2, dtype=float).tolist()
    fp = 0.0
    for i in range(mx):
        for j in range(my):
            d = (zf[i][j] - _eval2d_der(tx, ty, c_flat, kx, ky, 0, 0,
                                        xf[i], yf[j]))
            fp += d * d

    return (len(tx), np.array(tx, dtype=float), len(ty),
            np.array(ty, dtype=float), np.array(c_flat, dtype=float), fp, 0)
