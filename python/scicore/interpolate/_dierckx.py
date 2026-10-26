"""Pure-Python replacement for the deleted Cython ``_dierckx`` extension.

Implements the FITPACK/Dierckx banded-matrix helpers used by
``scicore.interpolate._regrid``:

* ``data_matrix`` — B-spline collocation matrix in packed banded form
  (de Boor recurrence, per-row weight applied).
* ``qr_reduce``   — in-place Householder QR reduction of the (packed
  banded) system, applied to all right-hand-side columns of ``B``.  The
  full reduced matrix is kept in a module cache (keyed by the id of the
  packed array) because QR fill-in exceeds the banded storage width for
  systems larger than the minimal knot count.
* ``fpback``      — back-substitution on the QR-reduced system, returning
  the least-squares solution ``(nc, nrhs)`` plus residual norms.
"""

from __future__ import annotations

import math

import rsnumpy as np

__all__ = ['qr_reduce', 'fpback', 'data_matrix']

# Dense reduced systems handed between qr_reduce and the matching fpback
# call.  The packed banded storage cannot hold QR fill-in, and rsnumpy
# 2-D in-place setitem is unreliable, so the reduced state is carried in
# this LIFO (call sequence is strictly qr_reduce / fpback alternating).
_PENDING = []


def _span(t, k, n, x):
    """Index of the knot span [t[i], t[i+1]) containing x (clamped)."""
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


def _basis_values(t, k, span, x):
    """The k+1 non-vanishing basis values N_{span-k+j, k}(x) at a span.

    Cox-de Boor recursion with the 0/0 := 0 convention; at the right
    endpoint of a clamped vector only the last basis function is
    non-zero.
    """
    if x == t[-1]:
        return [1.0 if j == k else 0.0 for j in range(k + 1)]
    V = [[0.0] * (k + 1) for _ in range(k + 1)]
    for q in range(k + 1):
        p0 = span - k + q
        if t[p0] <= x < t[p0 + 1]:
            V[q][0] = 1.0
    for deg in range(1, k + 1):
        for q in range(k + 1):
            p0 = span - k + q
            d1 = t[p0 + deg] - t[p0]
            term1 = ((x - t[p0]) * V[q][deg - 1]) / d1 if d1 != 0.0 else 0.0
            if q < k:
                d2 = t[p0 + deg + 1] - t[p0 + 1]
                term2 = ((t[p0 + deg + 1] - x) * V[q + 1][deg - 1]) / d2 \
                    if d2 != 0.0 else 0.0
            else:
                term2 = 0.0
            V[q][deg] = term1 + term2
    return [V[q][k] for q in range(k + 1)]


def data_matrix(u, t, k, w=None):
    """B-spline collocation matrix in packed banded form.

    Returns ``(A, offset, nc)``: ``A`` is an ``(m, k+1)`` array holding the
    non-zero entries of each row (consecutive columns starting at
    ``offset[i]``); ``nc`` is the total column count (knot count - k - 1).
    Each row is scaled by the corresponding weight (1.0 if ``w is None``).
    """
    ul = [float(v) for v in np.asarray(u).tolist()]
    tl = [float(v) for v in np.asarray(t).tolist()]
    k = int(k)
    n = len(tl) - k - 1
    m = len(ul)
    if w is None:
        wl = [1.0] * m
    else:
        wl = [float(v) for v in np.asarray(w).tolist()]
    A = [[0.0] * (k + 1) for _ in range(m)]
    offset = [0] * m
    for i in range(m):
        xi = ul[i]
        span = _span(tl, k, n, xi)
        offset[i] = span - k
        for j, v in enumerate(_basis_values(tl, k, span, xi)):
            A[i][j] = v * wl[i]
    return (np.array(A, dtype=float), np.array(offset, dtype=int), n)


def qr_reduce(A, offset, nc, B):
    """In-place Householder QR reduction of a packed banded system.

    ``A`` is an ``(m, nz)`` packed array (row i holds the first ``nz``
    entries of row i of the dense matrix, starting at column
    ``offset[i]``); ``B`` is the ``(m, nrhs)`` right-hand side, reduced in
    place.  After the call the top-left block of the (conceptual) dense
    ``A`` is upper triangular and ``B`` has had the same reflectors
    applied.
    """
    offset = [int(v) for v in np.asarray(offset).tolist()]
    m = len(offset)
    nz = int(A.shape[1])
    ncol = int(nc)
    Ad = [[0.0] * ncol for _ in range(m)]
    for i in range(m):
        off = offset[i]
        nel = min(ncol - off, nz)
        for j in range(nel):
            Ad[i][off + j] = float(A[i, j])
    nrhs = int(B.shape[1])
    Bd = [[float(B[i, j]) for j in range(nrhs)] for i in range(m)]

    r = min(m, ncol)
    for j in range(r):
        col = [Ad[i][j] for i in range(j, m)]
        nrm = math.sqrt(sum(v * v for v in col))
        if nrm == 0.0:
            continue
        v = list(col)
        v[0] = v[0] + math.copysign(nrm, v[0] if v[0] != 0.0 else 1.0)
        vv = sum(vi * vi for vi in v)
        if vv == 0.0:
            continue
        for jj in range(j, ncol):
            d = sum(v[ii] * Ad[j + ii][jj] for ii in range(m - j)) / vv
            for ii in range(m - j):
                Ad[j + ii][jj] -= 2.0 * v[ii] * d
        for cc in range(nrhs):
            d = sum(v[ii] * Bd[j + ii][cc] for ii in range(m - j)) / vv
            for ii in range(m - j):
                Bd[j + ii][cc] -= 2.0 * v[ii] * d

    # Hand the reduced system to the matching fpback call (no write-back
    # into the packed array: rsnumpy 2-D setitem is unreliable and the
    # callers consume only fpback's returned solution).
    _PENDING.append((Ad, Bd))


def fpback(A, nc, x, B, t, k, w, B2, flag):
    """Back-substitution after ``qr_reduce``.

    Returns ``(X, rnorm, fp)`` where ``X`` is the ``(nc, nrhs)``
    least-squares solution, ``rnorm`` its residual norm and ``fp`` the
    residual sum of squares.  The reduced system is taken from the
    matching ``qr_reduce`` call (``_PENDING``); if that is unavailable
    the packed storage is re-densified using row offsets derived from
    ``(x, t, k)`` (data rows) and unit advance (trailing penalty rows),
    with the given ``B`` used as the right-hand side.
    """
    nc = int(nc)
    R = Bd = None
    if _PENDING:
        R, Bd = _PENDING.pop()
    if R is None:
        m = int(A.shape[0])
        nz = int(A.shape[1])
        tl = [float(v) for v in np.asarray(t).tolist()]
        k = int(k)
        xl = [float(v) for v in np.asarray(x).ravel().tolist()]
        offset = [_span(tl, k, nc, xi) - k for xi in xl]
        offset += list(range(m - len(xl)))
        R = [[0.0] * nc for _ in range(m)]
        for i in range(m):
            off = offset[i]
            nel = min(nc - off, nz)
            for j in range(nel):
                R[i][off + j] = float(A[i, j])
        Bd = [[float(B[i, j]) for j in range(int(B.shape[1]))]
              for i in range(m)]
    m = len(R)
    nrhs = len(Bd[0])
    r = min(m, nc)
    X = [[0.0] * nrhs for _ in range(nc)]
    for i in range(r - 1, -1, -1):
        piv = R[i][i]
        if abs(piv) < 1e-300:
            raise ValueError("singular QR-reduced system in fpback")
        for cc in range(nrhs):
            s = Bd[i][cc]
            for j in range(i + 1, nc):
                s -= R[i][j] * X[j][cc]
            X[i][cc] = s / piv
    fp = 0.0
    for i in range(r, m):
        for cc in range(nrhs):
            fp += Bd[i][cc] ** 2
    return np.array(X, dtype=float), math.sqrt(fp), fp
