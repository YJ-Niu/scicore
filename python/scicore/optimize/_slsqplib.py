"""Pure-Python replacement for the C ``_slsqplib`` extension.

Implements the SLSQP driver used by ``scipy.optimize._slsqp_py``.  The
algorithm is a line-faithful port of scipy's ``slsqp.c`` (the C
reimplementation of Kraft's Fortran SLSQP): reverse-communication
``slsqp`` main driver (QP subproblem via ``lsq``/``lsei``/``lsi``/``ldp``,
inexact line search with the L1 merit test, BFGS updates on the packed
LDL' factors via ``ldl_update``).  All LAPACK/BLAS helpers (householder
QR/RQ, triangular solves, a small one-sided Jacobi SVD for the
rank-deficient least-squares step) are implemented here so that the
module has no native dependencies.

The internal buffer layout of the C code is intentionally NOT replicated:
only the top-level persistent regions of ``buffer`` (the packed BFGS
factors, ``x0``, ``mu``, ``s``, ``u``, ``v``) are shared across
reverse-communication calls; transient subproblem workspaces are plain
Python lists.
"""

from __future__ import annotations

import math

import rsnumpy as np

__all__ = ['slsqp', 'nnls']

_EPSMACH = 2.220446049250313e-16


# ----------------------------------------------------------------------
# small vector / matrix kernels (column-major C routines as list math)
# ----------------------------------------------------------------------

def _ddot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _dnrm2(a):
    s = 0.0
    for v in a:
        s += v * v
    return math.sqrt(s)


def _qr(a, b, ma, n):
    """Householder QR of column-major (ma x n) ``a`` (ma >= n), in place.

    Upper triangle of ``a`` becomes R, subdiagonal entries hold the
    reflectors anchored with v[0] = 1 (tau paired with that anchor).
    Applies Q.T to the single rhs ``b`` (ma).  Returns tau.
    """
    tau = []
    for j in range(n):
        norm = 0.0
        for i in range(j, ma):
            norm += a[i + j * ma] * a[i + j * ma]
        norm = math.sqrt(norm)
        if norm == 0.0 or norm == abs(a[j + j * ma]):
            # Column already aligned with the first basis vector (or
            # zero): no reflector needed, R diagonal stays as-is.
            tau.append(0.0)
            continue
        x0 = a[j + j * ma]
        sign = 1.0 if x0 >= 0.0 else -1.0
        beta = -sign * norm
        scale = x0 - beta
        u = [0.0] * (ma - j)
        for i in range(j + 1, ma):
            u[i - j] = a[i + j * ma] / scale
        u[0] = 1.0
        alpha = (abs(x0) + norm) / norm
        # Apply to the trailing columns of a ...
        for c in range(j, n):
            base = j + c * ma
            s = 0.0
            for tt in range(ma - j):
                s += u[tt] * a[base + tt]
            if s != 0.0:
                f = alpha * s
                for tt in range(ma - j):
                    a[base + tt] -= f * u[tt]
        a[j + j * ma] = beta
        for i in range(j + 1, ma):
            a[i + j * ma] = u[i - j]
        tau.append(alpha)
    # Apply Q.T to b: reflectors in factorization order (j ascending),
    # each anchored with v[0] = 1.
    for j in range(n):
        if tau[j] == 0.0:
            continue
        s = b[j]
        for i in range(j + 1, ma):
            s += a[i + j * ma] * b[i]
        if s != 0.0:
            f = tau[j] * s
            b[j] -= f
            for i in range(j + 1, ma):
                b[i] -= f * a[i + j * ma]
    return tau


def _rq(e, n, me):
    """Householder RQ of column-major (me x n) ``e`` (n >= me), in place.

    On exit ``e`` holds E @ Q.T = [0 | R]: the leading (me x (n-me))
    block is zero and R is the trailing (me x me) upper-triangular
    block, with R(i, i) at ``e[i + (n - me + i) * me]``.  Each step
    (bottom row first) applies a Householder to the segment
    columns [0, n - jj) of every row, mapping the active row segment
    to [0, ..., 0, ±norm] (norm at column n - jj - 1).  Returns the
    reflectors in application order (first applied first): each is
    (i0, i1, w, coef) with H = I - coef * outer(w, w) acting on
    columns [i0, i1).
    """
    reflectors = []
    for jj in range(me):
        j = me - 1 - jj
        i1 = n - jj
        i0 = 0
        k = i1 - i0
        norm = 0.0
        for j2 in range(i0, i1):
            norm += e[j + j2 * me] * e[j + j2 * me]
        norm = math.sqrt(norm)
        if norm == 0.0:
            continue
        x0 = e[j + (i1 - 1) * me]
        beta = norm if x0 == 0.0 else -math.copysign(1.0, x0) * norm
        w = [0.0] * k
        for j2 in range(i0, i1):
            w[j2 - i0] = e[j + j2 * me]
        w[k - 1] -= beta
        wtow = 0.0
        for tt in range(k):
            wtow += w[tt] * w[tt]
        coef = 2.0 / wtow
        # Apply H = I - coef w w^T to every row's segment [i0, i1)
        # (column-major: row r is strided by me).
        for r in range(me):
            s = 0.0
            for tt in range(k):
                s += w[tt] * e[r + (i0 + tt) * me]
            if s != 0.0:
                f = coef * s
                for tt in range(k):
                    e[r + (i0 + tt) * me] -= f * w[tt]
        reflectors.append((i0, i1, w, coef))
    return reflectors


def _rq_apply_t(mat, reflectors, rows, n):
    """Apply Q.T from the right to the ``rows`` rows of a (rows x n)
    column-major ``mat``: mat <- mat @ Q.T (reflectors in stored
    order)."""
    for i0, i1, w, coef in reflectors:
        k = i1 - i0
        for r in range(rows):
            s = 0.0
            for tt in range(k):
                s += w[tt] * mat[r + (i0 + tt) * rows]
            if s != 0.0:
                f = coef * s
                for tt in range(k):
                    mat[r + (i0 + tt) * rows] -= f * w[tt]


def _rq_apply_t_vec(x, reflectors, n):
    """Apply Q.T from the left to the length-n vector ``x``:
    x <- Q.T @ x (last-stored reflector applied first)."""
    for i0, i1, w, coef in reversed(reflectors):
        k = i1 - i0
        s = 0.0
        for tt in range(k):
            s += w[tt] * x[i0 + tt]
        if s != 0.0:
            f = coef * s
            for tt in range(k):
                x[i0 + tt] -= f * w[tt]


def _dtrsm_right_upper(g, r, mg, n):
    """G <- G R^{-1}; R upper triangular (n x n) row-major, G (mg x n)
    column-major.  Returns the result column-major.  Row-wise forward
    substitution of R^T z = g_i^T (equivalently z^T = g_i R^{-1})."""
    out = [0.0] * (mg * n)
    for i in range(mg):
        for j in range(n):
            s = 0.0
            for k in range(j):
                s += r[k * n + j] * out[i + k * mg]
            out[i + j * mg] = (g[i + j * mg] - s) / r[j * n + j]
    return out


def _dtrsv_upper(x, r, n):
    """Solve R x = x in place; R upper triangular (n x n) row-major."""
    for i in range(n - 1, -1, -1):
        s = x[i]
        for k in range(i + 1, n):
            s -= r[i * n + k] * x[k]
        x[i] = s / r[i * n + i]
    return x


def _svd(a, m, n):
    """One-sided Jacobi SVD of row-major (m x n) ``a``, m >= n.

    Returns s (length n), U (m x n row-major), V (n x n row-major) with
    A ~ U diag(s) V.T.  A is not modified.
    """
    w = [row[:] for row in a]
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            V[i][j] = 1.0 if i == j else 0.0
    converged = True
    for _ in range(60):
        converged = True
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = 0.0
                for i in range(m):
                    apq = w[i][p] * w[i][q] + apq
                app = 0.0
                aqq = 0.0
                for i in range(m):
                    app = w[i][p] * w[i][p] + app
                    aqq = w[i][q] * w[i][q] + aqq
                if abs(apq) < 1e-30 * math.sqrt(app * aqq + 1.0):
                    continue
                tau_v = (aqq - app) / (2.0 * apq)
                if tau_v >= 0.0:
                    t = 1.0 / (tau_v + math.sqrt(1.0 + tau_v * tau_v))
                else:
                    t = -2.0 / (tau_v - math.sqrt(1.0 + tau_v * tau_v))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s_ = c * t
                for i in range(m):
                    wp = w[i][p]
                    wq = w[i][q]
                    w[i][p] = c * wp - s_ * wq
                    w[i][q] = s_ * wp + c * wq
                for i in range(n):
                    vp = V[i][p]
                    V[i][q] = V[i][q]
                    V[i][p] = c * vp - s_ * V[i][q]
                    V[i][q] = s_ * vp + c * V[i][q]
                converged = False
        if converged:
            break
    s = [0.0] * n
    for j in range(n):
        nrm = 0.0
        for i in range(m):
            nrm += w[i][j] * w[i][j]
        nrm = math.sqrt(nrm)
        s[j] = nrm
        if nrm > 0.0:
            for i in range(m):
                w[i][j] /= nrm
    order = sorted(range(n), key=lambda j: -s[j])
    s = [s[j] for j in order]
    U = [[w[i][j] for j in order] for i in range(m)]
    V = [[V[i][j] for j in order] for i in range(n)]
    return s, U, V


def _dgelsy(a2, wb, ma, nvars, rcond, jw):
    """Least squares via SVD (the dgelsy role in lsei).

    ``a2`` (ma x nvars, row-major) is overwritten, ``wb`` (ma) becomes the
    solution (nvars).  ``jw`` holds the (trivial) column permutation.
    Returns the numerical rank.
    """
    if nvars == 0:
        return 0
    for i in range(nvars):
        jw[i] = i
    a2_2d = [a2[i * nvars:(i + 1) * nvars] for i in range(ma)]
    s, U, V = _svd(a2_2d, ma, nvars)
    rank = 0
    if s[0] > 0.0:
        thr = rcond * s[0]
        rank = 0
        for j in range(nvars):
            if s[j] > thr:
                rank += 1
    sol = [0.0] * nvars
    for j in range(rank):
        c = 0.0
        for i in range(ma):
            c = U[i][j] * wb[i] + c
        if s[j] > 0.0:
            c /= s[j]
        t = 0.0
        for i in range(nvars):
            t = V[i][j] * c + t
        sol[j] = t
    for i in range(nvars):
        wb[i] = sol[jw[i]]
    for i in range(nvars, ma):
        wb[i] = 0.0
    return rank


# ----------------------------------------------------------------------
# LDL' rank-one update (Fletcher-Powell composite-t method)
# ----------------------------------------------------------------------

def _pstart(n, j):
    """Start offset of column j in a packed n x n lower triangle."""
    return j * (2 * n - j + 1) // 2


def _ldl_update(a, z, sigma, n):
    """Update packed LDL' factors ``a`` by rank-one sigma z z'."""
    if sigma == 0.0:
        return
    t = 1.0 / sigma
    w = [0.0] * n
    ij = 0
    if sigma <= 0.0:
        # Negative update
        for i in range(n):
            w[i] = z[i]
        for i in range(n):
            v = w[i]
            t = t + v * v / a[ij]
            for j in range(i + 1, n):
                ij += 1
                w[j] = w[j] - v * a[ij]
            ij += 1
        if t >= 0.0:
            t = _EPSMACH / sigma
        ij = 0
        for i in range(n):
            j = n - i - 1
            ij = _pstart(n, j)
            u = w[j]
            w[j] = t
            t = t - u * u / a[ij]
    # Positive update
    ij = 0
    for i in range(n):
        v = z[i]
        delta = v / a[ij]
        tp = w[i] if sigma < 0.0 else t + delta * v
        alpha = tp / t
        a[ij] = alpha * a[ij]
        if i == n - 1:
            return
        beta = delta / tp
        if alpha <= 4.0:
            for j in range(i + 1, n):
                ij += 1
                z[j] = z[j] - v * a[ij]
                a[ij] = a[ij] + beta * z[j]
        else:
            gamma = t / tp
            for j in range(i + 1, n):
                ij += 1
                u = a[ij]
                a[ij] = gamma * u + beta * z[j]
                z[j] = z[j] - v * u
        ij += 1
        t = tp


# ----------------------------------------------------------------------
# QP subproblem chain: ldp (least distance) <- lsi (ineq LS) <-
# lsei (equality + ineq LS) <- lsq (the SQP subproblem)
# ----------------------------------------------------------------------

def _ldp(g, h, n):
    """Least distance problem: min 0.5|x|^2 s.t. G x >= h.

    ``g`` (m x n column-major), ``h`` (m).  Returns (mode, x, xnorm,
    mult) with x (n), mult (m).
    """
    m = len(h)
    x = [0.0] * n
    if n <= 0:
        return 2, x, 0.0, []
    if m == 0:
        return 1, x, 0.0, []
    # Dual data: A = [G^T ; h^T] ((n+1) x m column-major), b = [0..0, 1]
    a = [0.0] * (m * (n + 1))
    for j in range(m):
        for i in range(n):
            a[i + j * (n + 1)] = g[j + i * m]
        a[n + j * (n + 1)] = h[j]
    b = [0.0] * (n + 1)
    b[n] = 1.0
    a2d = [[a[i + j * (n + 1)] for j in range(m)] for i in range(n + 1)]
    y_arr, rnorm, info = nnls(a2d, b, 3 * m)
    if info != 0:
        return 3, x, rnorm, []
    if rnorm <= 0.0:
        return 4, x, rnorm, []
    y_l = [float(v) for v in np.asarray(y_arr).tolist()]
    fac = 1.0 - _ddot(h, y_l)
    if not (1.0 + fac - 1.0 > 0.0):
        return 4, x, rnorm, []
    fac = 1.0 / fac
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += g[j + i * m] * y_l[j]
        x[i] = fac * s
    xnorm = _dnrm2(x)
    mult = [fac * yj for yj in y_l]
    return 1, x, xnorm, mult


def _lsi(a, b, g, h, ma, mg, n):
    """Inequality-constrained least squares: min |Ax - b| s.t. Gx >= h.

    ``a`` (ma x n column-major, overwritten), ``b`` (ma, overwritten),
    ``g`` (mg x n column-major, overwritten), ``h`` (mg, overwritten).
    Returns (mode, x, xnorm, mult) with x (n), mult (mg).
    """
    x = [0.0] * n
    tmp_int = ma if ma < n else n
    _qr(a, b, ma, tmp_int)
    # Rank check on the diagonal of R.
    for i in range(tmp_int):
        if not (abs(a[i + i * ma]) >= _EPSMACH):
            return 5, x, 0.0, []
    # G <- G R^{-1} (R is the upper n x n of a).
    r = [a[i + j * ma] for i in range(n) for j in range(n)]
    g2 = _dtrsm_right_upper(g, r, mg, n)
    # h -= G b
    for i in range(mg):
        s = 0.0
        for j in range(n):
            s += g2[i + j * mg] * b[j]
        h[i] -= s
    mode, x, xnorm, mult = _ldp(g2, h, n)
    if mode != 1:
        return mode, x, xnorm, mult
    # Convert back to the original problem.
    for i in range(n):
        x[i] += b[i]
    _dtrsv_upper(x, r, n)
    tail = b[n:]
    if tail:
        xnorm = math.hypot(xnorm, _dnrm2(tail))
    return 1, x, xnorm, mult


def _lsei(a, b, e, f, g, h, ma, me, mg, n):
    """Equality + inequality constrained least squares:
    min |Ax - b| s.t. Ex = f, Gx >= h.

    All matrices are column-major; ``a`` (ma x n), ``e`` (me x n),
    ``g`` (mg x n) are overwritten, ``b`` (ma), ``f`` (me), ``h`` (mg)
    are overwritten.  Returns (mode, x, xnorm, gmults, emults).
    """
    x = [0.0] * n
    if me > n:
        return 2, x, 0.0, [], []
    nvars = n - me
    reflectors = []
    xnorm = 0.0
    gmults = [0.0] * mg

    def _original_basis(mode):
        # Residual: b <- A x - b
        for i in range(ma):
            s = 0.0
            for j in range(n):
                s += a[i + j * ma] * x[j]
            b[i] = s - b[i]
        # f <- A1^T b - G1^T gmults (columns nvars..n-1)
        for j in range(me):
            s = 0.0
            for i in range(ma):
                s += a[i + (nvars + j) * ma] * b[i]
            for i in range(mg):
                s -= g[i + (nvars + j) * mg] * gmults[i]
            f[j] = s
        # x <- Q^T x
        if me > 0:
            _rq_apply_t_vec(x, reflectors, n)
        # R^T emults = f (R = upper-right me x me block of e)
        emults = [0.0] * me
        for j in range(me):
            s = f[j]
            for k in range(j):
                s -= e[k + (nvars + j) * me] * emults[k]
            emults[j] = s / e[j + (nvars + j) * me]
        return mode, x, xnorm, gmults, emults

    # RQ decomposition of E; apply Q.T to A and G from the right.
    if me > 0:
        reflectors = _rq(e, n, me)
        _rq_apply_t(a, reflectors, ma, n)
        _rq_apply_t(g, reflectors, mg, n)
        # Rank check on the diagonal of R.
        for i in range(me):
            if not (abs(e[i + (nvars + i) * me]) >= _EPSMACH):
                return 6, x, 0.0, [], []
        # Solve R xe = f (R upper triangular: back-substitute bottom-up).
        xe = f[:]
        for j in range(me - 1, -1, -1):
            s = xe[j]
            for k in range(j + 1, me):
                s -= e[j + (nvars + k) * me] * xe[k]
            xe[j] = s / e[j + (nvars + j) * me]
        for i in range(me):
            x[nvars + i] = xe[i]
    if me == n:
        return _original_basis(1)

    # Modified rhs: wb = b - A1 xe
    wb = b[:]
    for i in range(ma):
        s = 0.0
        for j in range(me):
            s += a[i + (nvars + j) * ma] * x[nvars + j]
        wb[i] -= s
    # Transformed A2 and G2 (column-major)
    a2 = [a[i + j * ma] for j in range(nvars) for i in range(ma)]
    g2 = [g[i + j * mg] for j in range(nvars) for i in range(mg)]
    if mg == 0:
        # No inequality constraints: least squares directly (dgelsy).
        a2row = [a2[i + j * ma] for i in range(ma) for j in range(nvars)]
        wb_orig = wb[:]
        krank = _dgelsy(a2row, wb, ma, nvars, math.sqrt(_EPSMACH),
                        [0] * nvars)
        for i in range(nvars):
            x[i] = wb[i]
        # Residual with the original A (a2row is overwritten).
        res = wb_orig[:]
        for i in range(ma):
            s = 0.0
            for j in range(nvars):
                s += a[i + j * ma] * x[j]
            res[i] = s - res[i]
        xnorm = _dnrm2(res)
        if krank < nvars:
            return 7, x, xnorm, gmults, []
        return _original_basis(1)
    # Modify h, and solve the inequality constrained least squares.
    for i in range(mg):
        s = 0.0
        for j in range(me):
            s += g[i + (nvars + j) * mg] * x[nvars + j]
        h[i] -= s
    mode, xr, xnorm, gmults = _lsi(a2, wb, g2, h, ma, mg, nvars)
    for i in range(nvars):
        x[i] = xr[i]
    # If no equality constraints this was an LSI problem all along.
    if me == 0:
        return mode, x, xnorm, gmults, []
    xnorm = math.hypot(xnorm, _dnrm2(x[nvars:]))
    if mode != 1:
        return mode, x, xnorm, gmults, []
    return _original_basis(1)


def _lsq(C, d, xl, xu, bfgs, gradx, m, meq, n, augment, aug_weight):
    """The SQP subproblem:

        min 0.5 s' B s + g' s   s.t.   C s + d = 0   (rows < meq),
                                         C s + d >= 0 (rows >= meq),
                                         xl + s within the finite bounds

    ``C`` (m x n column-major), ``d`` (m), ``xl``/``xu`` (n, NaN =
    unbounded), ``bfgs`` packed LDL' factors of B (n*(n+1)/2),
    ``gradx`` (n).  Returns (mode, s, mult): s has n+1 entries when
    ``augment`` else n; mult (m).
    """
    orign = n
    mineq = m - meq
    ld = n + 1 if augment else n
    # Recover A = sqrt(diag) L^T and b = -L^{-1} g / sqrt(diag) from the
    # packed BFGS factor.
    wA = [0.0] * (ld * ld)
    cursor = 0
    for j in range(n):
        diag = math.sqrt(bfgs[cursor])
        cursor += 1
        wA[j + j * ld] = diag
        for i in range(j + 1, n):
            wA[j + i * ld] = bfgs[cursor] * diag
            cursor += 1
    w = gradx[:]
    for i in range(n):
        s = w[i]
        for j in range(i):
            s -= bfgs[_pstart(n, j) + (i - j)] * w[j]
        w[i] = s
    cursor = 0
    for i in range(n):
        w[i] /= -math.sqrt(bfgs[cursor])
        cursor += n - i
    wb = w[:] + ([0.0] if augment else [])
    if augment:
        wA[ld * ld - 1] = aug_weight
        xl_use = xl + [0.0]
        xu_use = xu + [1.0]
        n = n + 1
    else:
        xl_use = xl
        xu_use = xu
    # Equality constraint data (E, f)
    if meq > 0:
        wE = [0.0] * (meq * n)
        for j in range(n - 1):
            for i in range(meq):
                wE[i + j * meq] = C[i + j * m]
        if augment:
            for i in range(meq):
                wE[i + (n - 1) * meq] = -d[i]
        else:
            for i in range(meq):
                wE[i + (n - 1) * meq] = C[i + (n - 1) * m]
        wf = [-d[i] for i in range(meq)]
    else:
        wE = []
        wf = []
    # Inequality + bounds data (G, h); NaN bounds are skipped.
    nancount = 0
    wh = [0.0] * (mineq + 2 * n)
    nrow = mineq
    if m > meq:
        for i in range(mineq):
            wh[i] = -d[meq + i]
    for i in range(n):
        if math.isnan(xl_use[i]):
            nancount += 1
        else:
            wh[nrow] = xl_use[i]
            nrow += 1
    for i in range(n):
        if math.isnan(xu_use[i]):
            nancount += 1
        else:
            wh[nrow] = -xu_use[i]
            nrow += 1
    n_wG_rows = mineq + 2 * n - nancount
    wG = [0.0] * (n_wG_rows * ld)
    wh = wh[:n_wG_rows]
    if m > meq:
        for j in range(orign):
            for i in range(mineq):
                wG[i + j * n_wG_rows] = C[meq + i + j * m]
    if augment:
        for i in range(mineq):
            wG[i + orign * n_wG_rows] = max(-d[meq + i], 0.0)
    nrow = mineq
    for i in range(orign):
        if not math.isnan(xl_use[i]):
            wG[nrow + i * n_wG_rows] = 1.0
            nrow += 1
    for i in range(orign):
        if not math.isnan(xu_use[i]):
            wG[nrow + i * n_wG_rows] = -1.0
            nrow += 1
    mode, x, _xnorm, gmults, emults = _lsei(
        wA, wb, wE, wf, wG, wh, ld, meq, n_wG_rows, n)
    mult = [0.0] * m
    if mode == 1:
        for i in range(meq):
            mult[i] = emults[i]
        for i in range(mineq):
            mult[meq + i] = gmults[i]
    # Clamp the solution to the finite bound interval
    for i in range(n):
        if not math.isnan(xl_use[i]) and x[i] < xl_use[i]:
            x[i] = xl_use[i]
        elif not math.isnan(xu_use[i]) and x[i] > xu_use[i]:
            x[i] = xu_use[i]
    return mode, x, mult


# ----------------------------------------------------------------------
# main reverse-communication driver (C ``__slsqp_body``)
# ----------------------------------------------------------------------

def slsqp(state_dict, fx, g, C, d, x, mult, xl, xu, buffer, indices):
    """One reverse-communication step of the SLSQP solver.

    Port of scipy's C ``__slsqp_body``.  ``state_dict`` is updated in
    place; its ``mode`` selects the re-entry point: 0 on the first
    call, 1 after an objective/constraint evaluation, -1 after a
    gradient/Jacobian evaluation.  On return ``mode`` is 1 (objective
    and constraints of the trial point must be evaluated), -1
    (gradients must be evaluated) or a termination code (0 success,
    2..9 — see the orchestrator's ``exit_modes``).

    ``C`` is (m x n), the first ``meq`` rows being the equality
    constraint normals; ``d`` (m) the constraint values; ``x`` (n) the
    current point (the trial point on return); ``mult`` (m + 2n + 2)
    the multipliers; ``xl``/``xu`` (n) the bounds, NaN meaning
    unbounded; ``buffer`` the persistent work area (packed BFGS
    factors, x0, mu, s, u, v); ``indices`` is accepted for interface
    compatibility.
    """
    n = state_dict['n']
    m = state_dict['m']
    meq = state_dict['meq']
    n1 = n + 1
    n2 = n * n1 // 2

    def _flat(arr):
        out = []

        def _into(t):
            if isinstance(t, (list, tuple)):
                for e in t:
                    _into(e)
            else:
                out.append(float(t))
        _into(arr.tolist())
        return out

    x_l = _flat(x)
    d_l = _flat(d) if m else []
    g_l = _flat(g)
    mult_l = _flat(mult)
    xl_l = _flat(xl)
    xu_l = _flat(xu)
    buf = _flat(buffer)
    Crow = C.tolist() if m else []
    Ccol = [float(Crow[i][j]) for j in range(n) for i in range(m)] if m else []
    bfgs = buf[0:n2]
    x0 = buf[n2:n2 + n]
    mu = buf[n2 + n:n2 + n + m]
    s = buf[n2 + n + m:n2 + 2 * n + m]
    u = buf[n2 + 2 * n + m:n2 + 2 * n + m + n1]
    v = buf[n2 + 2 * n + m + n1:n2 + 2 * n + m + 2 * n1]
    badlin = 0

    def _flush():
        off = 0
        for val in bfgs:
            buffer[off] = val
            off += 1
        for val in x0:
            buffer[off] = val
            off += 1
        for val in mu:
            buffer[off] = val
            off += 1
        for val in s:
            buffer[off] = val
            off += 1
        for val in u:
            buffer[off] = val
            off += 1
        for val in v:
            buffer[off] = val
            off += 1
        for i, val in enumerate(x_l):
            x[i] = val
        for i, val in enumerate(mult_l):
            mult[i] = val

    def _label255():
        h3 = 0.0
        for j in range(m):
            hj = d_l[j] if j < meq else 0.0
            h3 += max(-d_l[j], hj)
        if ((abs(fx - state_dict['f0']) < state_dict['tol'])
                or (_dnrm2(s) < state_dict['tol'])) \
                and (h3 < state_dict['tol']) and (not badlin) and (fx == fx):
            state_dict['mode'] = 0
        else:
            state_dict['mode'] = 8
        _flush()

    def _reset_bfgs():
        state_dict['reset'] += 1
        if state_dict['reset'] > 5:
            _label255()
            return False
        for i in range(n2):
            bfgs[i] = 0.0
        j = 0
        for i in range(n):
            bfgs[j] = 1.0
            j += n - i
        return True

    def _iter_start():
        """The ITER_START body.  Returns 'reset' (back to RESET_BFGS)
        or 'return' (finished with the mode set, line-search exit
        included)."""
        nonlocal badlin
        state_dict['mode'] = 9
        if state_dict['iter'] >= state_dict['itermax']:
            return 'return'
        state_dict['iter'] += 1
        for i in range(n):
            u[i] = -x_l[i] + xl_l[i]
            v[i] = -x_l[i] + xu_l[i]
        state_dict['h4'] = 1.0
        qmode, qs, qmult = _lsq(Ccol, d_l, u, v, bfgs, g_l,
                                m, meq, n, 0, 0.0)
        state_dict['mode'] = qmode
        if qmode == 1:
            for i in range(m):
                mult_l[i] = qmult[i]
            for i in range(2 * n):
                mult_l[m + i] = float('nan')
        badlin = 0
        if (state_dict['mode'] == 6) and (n == meq):
            state_dict['mode'] = 4
        if state_dict['mode'] == 4:
            badlin = 1
            for i in range(n):
                s[i] = 0.0
            state_dict['h3'] = 0.0
            rho = 100.0
            state_dict['inconsistent'] = 0
            while True:
                qmode, qs, qmult = _lsq(Ccol, d_l, u, v, bfgs,
                                        g_l, m, meq, n, 1, rho)
                state_dict['mode'] = qmode
                if qmode == 1:
                    for i in range(m):
                        mult_l[i] = qmult[i]
                    for i in range(2 * n):
                        mult_l[m + i] = float('nan')
                    # Copy the (augmented) step into the s buffer (the C
                    # lsq writes the solution in place).
                    for i in range(n):
                        s[i] = qs[i]
                state_dict['h4'] = 1.0 - qs[n]
                if state_dict['mode'] == 4:
                    rho *= 10.0
                    state_dict['inconsistent'] += 1
                    if state_dict['inconsistent'] > 5:
                        return 'return'
                    continue
                elif state_dict['mode'] != 1:
                    return 'return'
                break
        elif state_dict['mode'] != 1:
            return 'return'
        # Copy the QP step into the s buffer (the C lsq writes the
        # solution in place; _lsq returns it as a fresh list).
        for i in range(n):
            s[i] = qs[i]
        # v <- g - C' mult: the Lagrangian gradient at x_old.  The BFGS
        # update at MODEM1 then forms u = g_new - C' mult - v =
        # g_new - g_old (the multipliers cancel).  Writing into the v
        # buffer (as the C code does) is essential: it also keeps v
        # finite for unbounded variables.
        for i in range(n):
            sj = 0.0
            for j in range(m):
                sj += Ccol[i + j * m] * qmult[j]
            v[i] = g_l[i] - sj
        state_dict['f0'] = fx
        for i in range(n):
            x0[i] = x_l[i]
        state_dict['gs'] = _ddot(g_l, qs)
        h1 = abs(state_dict['gs'])
        h2 = 0.0
        for j in range(m):
            hj = d_l[j] if j < meq else 0.0
            h2 += max(-d_l[j], hj)
            hm = abs(qmult[j])
            mu[j] = max(hm, (mu[j] + hm) / 2.0)
            h1 += hm * abs(d_l[j])
        # Check convergence
        state_dict['mode'] = 0
        if (h1 < state_dict['acc']) and (h2 < state_dict['acc']) \
                and (not badlin) and (fx == fx):
            return 'return'
        h1 = 0.0
        for j in range(m):
            hj = d_l[j] if j < meq else 0.0
            h1 += mu[j] * max(-d_l[j], hj)
        state_dict['t0'] = fx + h1
        state_dict['h3'] = state_dict['gs'] - h1 * state_dict['h4']
        state_dict['mode'] = 8
        if state_dict['h3'] >= 0.0:
            return 'reset'
        # Line search with the L1 test function
        state_dict['line'] = 1
        state_dict['alpha'] = 1.0
        state_dict['h3'] = state_dict['alpha'] * state_dict['h3']
        for i in range(n):
            s[i] *= state_dict['alpha']
        for i in range(n):
            x_l[i] = x0[i] + s[i]
        state_dict['mode'] = 1
        return 'return'

    mode = state_dict['mode']
    if mode == 0:
        # Inexact line search is always used (exact search is broken in
        # the original Fortran code).
        state_dict['exact'] = 0
        state_dict['acc'] = abs(state_dict['acc'])
        state_dict['tol'] = 10.0 * state_dict['acc']
        state_dict['iter'] = 0
        state_dict['reset'] = 0
        for i in range(n):
            s[i] = 0.0
        for i in range(m):
            mu[i] = 0.0
        while True:  # RESET_BFGS
            if not _reset_bfgs():
                return
            action = _iter_start()
            _flush()
            if action == 'return':
                return
    elif mode == 1:
        t = fx
        for j in range(m):
            hj = d_l[j] if j < meq else 0.0
            t += mu[j] * max(-d_l[j], hj)
        h1 = t - state_dict['t0']
        if not ((h1 <= state_dict['h3'] / 10.0)
                or (state_dict['line'] > 10)):
            state_dict['alpha'] = max(
                state_dict['h3'] / (2.0 * (state_dict['h3'] - h1)), 0.1)
            state_dict['h3'] = state_dict['alpha'] * state_dict['h3']
            for i in range(n):
                s[i] *= state_dict['alpha']
            for i in range(n):
                x_l[i] = x0[i] + s[i]
            state_dict['mode'] = 1
            _flush()
            return
        # Check convergence
        h3 = 0.0
        for j in range(m):
            hj = d_l[j] if j < meq else 0.0
            h3 += max(-d_l[j], hj)
        if ((abs(fx - state_dict['f0']) < state_dict['acc'])
                or (_dnrm2(s) < state_dict['acc'])) \
                and (h3 < state_dict['acc']) and (fx == fx):
            state_dict['mode'] = 0
        else:
            state_dict['mode'] = -1
        _flush()
        return
    elif mode == -1:
        # BFGS update of the LDL' factors: u <- g - C' mult - v
        for i in range(n):
            u[i] = g_l[i]
            sj = 0.0
            for j in range(m):
                sj += Ccol[i + j * m] * mult_l[j]
            u[i] -= sj
            u[i] -= v[i]
        # w <- B s via the LDL' factors
        w = s[:]
        for i in range(n):
            s2 = w[i]
            for j in range(i):
                s2 += bfgs[_pstart(n, j) + (i - j)] * w[j]
            w[i] = s2
        j = 0
        for i in range(n):
            w[i] *= bfgs[j]
            j += n - i
        for i in range(n - 1, -1, -1):
            s2 = w[i]
            for j in range(i):
                s2 += bfgs[_pstart(n, j) + (i - j)] * w[j]
            w[i] = s2
        h1 = _ddot(s, u)
        h2 = _ddot(s, w)
        h3 = 0.2 * h2
        if h1 < h3:
            h4 = (h2 - h3) / (h2 - h1)
            h1 = h3
            tmp = 1.0 - h4
            for i in range(n):
                u[i] = h4 * u[i] + tmp * w[i]
        # Test for a singular update, and reset the hessian if so.
        if (h1 == 0.0) or (h2 == 0.0):
            if not _reset_bfgs():
                return
        else:
            _ldl_update(bfgs, u, 1.0 / h1, n)
            _ldl_update(bfgs, w, -1.0 / h2, n)
        while True:  # ITER_START
            action = _iter_start()
            _flush()
            if action == 'return':
                return
            if not _reset_bfgs():
                return


# ----------------------------------------------------------------------
# NNLS (Lawson-Hanson active set) — also used by the least-distance
# problem in the QP chain
# ----------------------------------------------------------------------

def _dlarfgp(alpha, xvec):
    """Householder reflector (scipy's dlarfgp convention): H = I - tau v v^T
    with v = [1; xs] (v[0] implicit) maps [alpha; x] to [beta; 0] with
    beta = +hypot(alpha, |x|).  Returns (beta, tau, xs)."""
    xnorm2 = 0.0
    for t in xvec:
        xnorm2 += t * t
    if not xvec or xnorm2 == 0.0:
        return alpha, 0.0, list(xvec)
    xnorm = math.sqrt(xnorm2)
    beta = math.hypot(alpha, xnorm)
    acoef = -(alpha + beta) / xnorm2
    xs = [t * acoef for t in xvec]
    if alpha > 0.0:
        tau = xnorm2 / (beta * (beta + alpha))
    else:
        tau = (beta - alpha) / beta
    return beta, tau, xs


def _dlarf_apply(v_tail, tau, target, start, length):
    """Apply H = I - tau v v^T (v = [1] + v_tail) to
    ``target[start : start + length]`` in place (single-column dlarf)."""
    if tau == 0.0:
        return
    s = target[start]
    for t in range(length - 1):
        s += v_tail[t] * target[start + 1 + t]
    if s == 0.0:
        return
    f = tau * s
    target[start] -= f
    for t in range(length - 1):
        target[start + 1 + t] -= f * v_tail[t]


def _dlartgp(f, g):
    """Givens rotation [cs sn; -sn cs] [f; g] = [r; 0] with r >= 0
    (scipy's dlartgp convention).  Returns (cs, sn, r)."""
    if f == 0.0 and g == 0.0:
        return 1.0, 0.0, 0.0
    if g == 0.0:
        return (1.0 if f >= 0.0 else -1.0), 0.0, abs(f)
    if f == 0.0:
        return 0.0, (1.0 if g >= 0.0 else -1.0), abs(g)
    if abs(f) < abs(g):
        t = f / g
        c = t / math.sqrt(1.0 + t * t)
        cs, sn, r = c, 1.0 / c, g * c
    else:
        t = g / f
        s = t / math.sqrt(1.0 + t * t)
        cs, sn, r = 1.0 / s, s, f * s
    if r < 0.0:
        r, cs, sn = -r, -cs, -sn
    return cs, sn, r


def nnls(A, b, maxiter=None):
    """Non-negative least squares (Lawson-Hanson active set method).

    Faithful port of scipy's C ``__nnls`` (nnls.c): Householder column
    acceptance (dlarfgp/dlarf), Givens-rotated active-set removal
    (dlartgp), in-place permuted triangular solves.  Takes row-major
    (m x n) ``A``; returns ``(x, rnorm, info)`` with info 0 on
    convergence, 2 on bad dimensions, 3 when ``maxiter`` is exceeded.
    """
    A_l = [list(row) for row in np.asarray(A).tolist()]
    b_l = _flat_scalar(b)
    m = len(A_l)
    n = len(A_l[0]) if m else 0
    if m <= 0 or n <= 0:
        return np.zeros(n), 0.0, 2
    if not maxiter:
        maxiter = 3 * m
    # column-major copy of A
    a = [A_l[i][j] for j in range(n) for i in range(m)]
    x = [0.0] * n
    w = [0.0] * n
    zz = [0.0] * m
    indices = list(range(n))
    indz = 0
    iteration = 0
    info = 1
    done = False
    while (not done) and indz < (m if m < n else n):
        # Dual vector for the columns still in set Z.
        for i in range(indz, n):
            j = indices[i]
            s = 0.0
            for t in range(m - indz):
                s += a[indz + t + j * m] * b_l[indz + t]
            w[j] = s
        while True:
            # Candidate with the largest dual coefficient.
            wmax = 0.0
            izmax = 0
            for k in range(indz, n):
                j = indices[k]
                if w[j] > wmax:
                    wmax = w[j]
                    izmax = k
            if wmax <= 0.0:
                done = True
                break
            iz = izmax
            j = indices[iz]
            # Householder reflector for column j below the pivot row.
            beta, tau, xs = _dlarfgp(
                a[indz + j * m], a[indz + 1 + j * m: m + j * m])
            unorm = 0.0
            for t in range(indz):
                unorm += a[t + j * m] * a[t + j * m]
            unorm = math.sqrt(unorm)
            spacing = (math.nextafter(unorm, 2.0 * unorm) - unorm) \
                if unorm > 0.0 else 0.0
            if abs(beta) > 100.0 * spacing:
                for i in range(m):
                    zz[i] = b_l[i]
                pivot2 = a[indz + j * m]
                a[indz + j * m] = 1.0
                _dlarf_apply(xs, tau, zz, indz, m - indz)
                if zz[indz] / beta > 0.0:
                    break
                a[indz + j * m] = pivot2
            # Reject this candidate.
            w[j] = 0.0
        if done:
            break
        # Move j from Z into the active set, keeping A orthonormal.
        for i in range(m):
            b_l[i] = zz[i]
        indices[iz] = indices[indz]
        indices[indz] = j
        indz += 1
        if indz < n:
            for k in range(indz, n):
                jj = indices[k]
                _dlarf_apply(xs, tau, a, indz - 1 + jj * m, m - indz + 1)
        a[indz - 1 + j * m] = beta
        if indz < m:
            for i in range(indz, m):
                a[j * m + i] = 0.0
        w[j] = 0.0
        # Solve the permuted triangular system (result in zz).
        jj = 0
        for k in range(indz):
            ip = indz - 1 - k
            if k != 0:
                for i in range(ip + 1):
                    zz[i] -= a[i + jj * m] * zz[ip + 1]
            jj = indices[ip]
            zz[ip] /= a[ip + jj * m]
        # Inner loop: feasibility of the active-set solution.
        while not done:
            iteration += 1
            if iteration >= maxiter:
                info = 3
                done = True
                break
            alpha = 2.0
            jj = 0
            for ip in range(indz):
                k = indices[ip]
                if zz[ip] <= 0.0:
                    T = -x[k] / (zz[ip] - x[k])
                    if alpha > T:
                        alpha = T
                        jj = ip
            if alpha == 2.0:
                break
            for ip in range(indz):
                k = indices[ip]
                x[k] = x[k] + alpha * (zz[ip] - x[k])
            i = indices[jj]
            while True:
                x[i] = 0.0
                if jj != indz - 1:
                    jj += 1
                    for j2 in range(jj, indz):
                        ii = indices[j2]
                        indices[j2 - 1] = ii
                        cc, ss, r = _dlartgp(a[j2 - 1 + ii * m],
                                              a[j2 + ii * m])
                        a[j2 + ii * m] = 0.0
                        a[j2 - 1 + ii * m] = r
                        for k2 in range(n):
                            if k2 != ii:
                                tmp = a[j2 - 1 + k2 * m]
                                a[j2 - 1 + k2 * m] = (cc * tmp
                                                      + ss * a[j2 + k2 * m])
                                a[j2 + k2 * m] = (-ss * tmp
                                                  + cc * a[j2 + k2 * m])
                        tmp = b_l[j2 - 1]
                        b_l[j2 - 1] = cc * tmp + ss * b_l[j2]
                        b_l[j2] = -ss * tmp + cc * b_l[j2]
                indz -= 1
                indices[indz] = i
                # Numerical noise: remove further zero components.
                nobreak = 0
                for jj2 in range(indz):
                    i2 = indices[jj2]
                    if x[i2] <= 0.0:
                        break
                    if jj2 == indz - 1:
                        nobreak = 1
                if nobreak:
                    break
                if indz <= 0:
                    # C leaves this case to undefined behaviour; stop
                    # safely with the zero solution.
                    break
                jj, i = jj2, i2
            if done:
                break
            # Re-solve the reduced system.
            for i in range(m):
                zz[i] = b_l[i]
            jj = 0
            for k in range(indz):
                ip = indz - 1 - k
                if k != 0:
                    for i in range(ip + 1):
                        zz[i] -= a[i + jj * m] * zz[ip + 1]
                jj = indices[ip]
                zz[ip] /= a[ip + jj * m]
        # Commit the active-set solution to x.
        for k in range(indz):
            x[indices[k]] = zz[k]
    if indz < m:
        rnorm = math.sqrt(sum(t * t for t in b_l[indz:]))
    else:
        rnorm = 0.0
    return np.array(x), rnorm, 0 if info == 1 else info


def _flat_scalar(b):
    """Flatten an array-like rhs to a list of floats."""
    out = []

    def _into(t):
        if isinstance(t, (list, tuple)):
            for e in t:
                _into(e)
        else:
            out.append(float(t))
    _into(np.asarray(b).tolist())
    return out
