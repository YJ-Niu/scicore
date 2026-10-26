"""Python replacement for the deleted Cython ``_ppoly`` extension.

Provides piecewise polynomial (PPoly) routines used by ``scicore.interpolate``.
``evaluate`` implements the full C calling contract (see its docstring); the
remaining routines are still stubs pending the Rust port.
"""

from __future__ import annotations

import bisect

import rsnumpy as np

__all__ = ['evaluate', 'fix_continuity', 'integrate', 'real_roots',
           'evaluate_bernstein']


def evaluate(c, knots, points, der=0, extrapolate=True, out=None):
    """Evaluate a piecewise polynomial at the query points (C contract).

    Mirrors the deleted Cython ``_ppoly.evaluate``:

        evaluate(c, knots, points, der, extrapolate, out) -> out

    Parameters
    ----------
    c : (K, N, ...) array_like
        Coefficients in the port's PPoly layout: ``c[k, i]`` is the
        coefficient of power ``K - 1 - k`` of piece ``i`` (descending
        powers, degree-major, as built by ``CubicHermiteSpline`` et al.);
        trailing dimensions select the number of simultaneous outputs.
    knots : (N+1,) array_like
        Ascending breakpoints.
    points : (n,) array_like
        Query points (1-D, float64).
    der : int, optional
        Non-negative derivative order.
    extrapolate : bool, optional
        When False, query points outside ``[knots[0], knots[-1]]`` yield
        NaN.  ('periodic' is mapped onto the interval by the caller before
        the call, as in ``_PPolyBase.__call__``.)
    out : (n, ...) ndarray or None
        Buffer to fill in place.

    Returns
    -------
    out : (n, ...) ndarray
        Values of the ``der``-th derivative at each query point.
    """

    def _poch(n, m):
        r = 1.0
        for q in range(1, m + 1):
            r *= (n - m + q)
        return r

    cl = [row for row in np.asarray(c).tolist()]
    K = len(cl)
    N = len(cl[0])
    multi = isinstance(cl[0][0], list)
    n_out = len(cl[0][0]) if multi else 1
    knot_l = [float(v) for v in np.asarray(knots).ravel().tolist()]
    pt_l = [float(v) for v in np.asarray(points).ravel().tolist()]

    lo, hi = knot_l[0], knot_l[-1]
    rows = []
    for xq in pt_l:
        if (not extrapolate) and (xq < lo or xq > hi):
            rows.append([float('nan')] * n_out)
            continue
        i = bisect.bisect_right(knot_l, xq) - 1
        if i < 0:
            i = 0
        if i > N - 1:
            i = N - 1
        t = xq - knot_l[i]
        # Power p (p = 0 .. K-1-der) of the `der`-th derivative comes from
        # c[K-1-p-der][i] * (p+der)!/p!.  Horner from the top power down.
        if multi:
            res = None
            for p in range(K - 1 - der, -1, -1):
                coef = cl[K - 1 - p - der][i]
                if der:
                    scale = _poch(p + der, der)
                    coef = [v * scale for v in coef]
                res = (coef if res is None
                       else [a * t + b for a, b in zip(res, coef)])
            rows.append(res)
        else:
            res = 0.0
            for p in range(K - 1 - der, -1, -1):
                coef = float(cl[K - 1 - p - der][i])
                if der:
                    coef *= _poch(p + der, der)
                res = res * t + coef
            rows.append([res])

    if out is None:
        return np.array(rows, dtype=np.float64)
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            out[i, j] = rows[i][j]
    return out


def fix_continuity(c, *args, **kwargs):
    """Stub for PPoly fix_continuity."""
    return c.copy()


def integrate(c, *args, **kwargs):
    """Stub for PPoly integrate."""
    return np.zeros_like(c)


def real_roots(c, *args, **kwargs):
    """Stub for PPoly real_roots."""
    return np.array([])


def evaluate_bernstein(c, x, *args, **kwargs):
    """Stub for PPoly evaluate_bernstein."""
    return np.zeros_like(x)
