"""Pure-Python replacement for the deleted C ``_minpack`` extension.

Implements the MINPACK least-squares kernels (``_hybrd``, ``_hybrj``,
``_lmdif``, ``_lmder``) in pure Python with the C calling conventions used
by ``_minpack_py``.  The solvers follow the MINPACK designs (modified Powell
hybrid method and Levenberg-Marquardt, forward-difference Jacobians,
Householder QR) implemented over plain Python lists, and return
``(x, infodict, ier)`` exactly like the deleted C extension.
"""

from __future__ import annotations

import math

import rsnumpy as np

_MACHINE_EPS = 2.220446049250313e-16


def _vec(y):
    return [float(v) for v in np.asarray(y).ravel().tolist()]


def _eval_f(fcn, x, args):
    out = fcn(np.array(x), *args)
    if isinstance(out, tuple):
        out = out[0]
    return _vec(out)


def _eval_jac(Dfun, x, args, col_deriv):
    out = Dfun(np.array(x), *args)
    J = [[float(v) for v in row] for row in np.asarray(out).tolist()]
    if col_deriv:
        J = [list(col) for col in zip(*J)]
    return J


def _num_jacobian(fcn, x, fvec, args, epsfcn, m, n):
    """Forward-difference Jacobian (MINPACK diford/difor convention)."""
    eps = math.sqrt(max(epsfcn, _MACHINE_EPS))
    J = [[0.0] * n for _ in range(m)]
    nfev = 0
    for j in range(n):
        xj = x[j]
        h = eps * xj
        if h == 0.0:
            h = eps
        x[j] = xj + h
        f = _eval_f(fcn, x, args)
        x[j] = xj
        nfev += 1
        for i in range(m):
            J[i][j] = (f[i] - fvec[i]) / h
    return J, nfev


def _qr_transform(a, f, m, n, pivot):
    """In-place Householder QR of ``a`` (m x n) and matching transform of ``f``.

    On return the upper triangle of ``a`` holds R and ``f`` holds Q'f.
    With ``pivot``, rows are permuted for column pivoting and the permutation
    is returned in ``ipvt``.
    """
    ipvt = list(range(n))
    for j in range(n):
        if pivot:
            prow = j
            for i in range(j + 1, m):
                if abs(a[i][j]) > abs(a[prow][j]):
                    prow = i
            if prow != j:
                a[j], a[prow] = a[prow], a[j]
                f[j], f[prow] = f[prow], f[j]
                ipvt[j], ipvt[prow] = ipvt[prow], ipvt[j]
        x = [a[i][j] for i in range(j, m)]
        nrm = math.sqrt(sum(v * v for v in x))
        if nrm == 0.0:
            continue
        sigma = -math.copysign(1.0, x[0]) * nrm
        v = list(x)
        v[0] -= sigma
        vnrm = math.sqrt(sum(vi * vi for vi in v))
        if vnrm == 0.0:
            continue
        for k in range(len(v)):
            v[k] /= vnrm
        dot = sum(v[k] * f[j + k] for k in range(len(v)))
        for i in range(j, m):
            f[i] -= 2.0 * v[i - j] * dot
        for c in range(j, n):
            dot = sum(v[k] * a[j + k][c] for k in range(len(v)))
            for i in range(j, m):
                a[i][c] -= 2.0 * v[i - j] * dot
    return ipvt


def _upper_trisolve(R, b, n):
    """Solve R y = b for upper-triangular R (n x n); None if singular."""
    y = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for k in range(i + 1, n):
            s -= R[i][k] * y[k]
        if R[i][i] == 0.0:
            return None
        y[i] = s / R[i][i]
    return y


def _trans_trisolve(R, b, n):
    """Solve R' y = b for upper-triangular R (n x n); None if singular."""
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= R[k][i] * y[k]
        if R[i][i] == 0.0:
            return None
        y[i] = s / R[i][i]
    return y


def _cholesky_solve(A, b, n):
    """Solve A x = b for symmetric positive definite A; None if not PD."""
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    return None
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    z = [0.0] * n
    for i in range(n):
        s = b[i] - sum(L[i][k] * z[k] for k in range(i))
        z[i] = s / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = z[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))
        x[i] = s / L[i][i]
    return x


def _jtp(J, fvec, m, n):
    """Return (J'J, J'f) for J (m x n)."""
    JtJ = [[0.0] * n for _ in range(n)]
    Jtf = [0.0] * n
    for i in range(m):
        Ji = J[i]
        fi = fvec[i]
        for k in range(n):
            Jtf[k] += Ji[k] * fi
            for l in range(n):
                JtJ[k][l] += Ji[k] * Ji[l]
    return JtJ, Jtf


def _step_scale(delta, scale, n):
    ds = 0.0
    for k in range(n):
        sk = scale[k] if scale[k] > 0.0 else 1.0
        d = abs(delta[k]) / sk
        if d > ds:
            ds = d
    return ds


def _lm_core(fcn, x, args, m, n, ftol, xtol, gtol, maxfev, epsfcn, factor,
             scale, Dfun, col_deriv):
    """Levenberg-Marquardt iteration (MINPACK lmdif design).

    Returns (x, fvec, nfev, ier, J, R, ipvt, qtf, rank) with the MINPACK
    lmdif ``ier`` codes (1 ftol, 2 xtol, 3 both, 4 gtol, 5 maxfev, 0 improper).
    """
    fvec = _eval_f(fcn, x, args)
    nfev = 1
    fnorm = math.sqrt(sum(v * v for v in fvec))
    par = factor * 0.001
    iterct = 0
    ier = 0
    J = [[0.0] * n for _ in range(m)]
    R = [[0.0] * n for _ in range(n)]
    ipvt = list(range(n))
    qtf = list(fvec)
    rank = n
    while True:
        if iterct >= maxfev:
            ier = 5
            break
        iterct += 1
        if Dfun is None:
            J, nf = _num_jacobian(fcn, x, fvec, args, epsfcn, m, n)
            nfev += nf
        else:
            J = _eval_jac(Dfun, x, args, col_deriv)
        a = [row[:] for row in J]
        f = list(fvec)
        ipvt = _qr_transform(a, f, m, n, pivot=False)
        R = [a[i][:] for i in range(n)]
        qtf = f
        rank = 0
        for j in range(n):
            if R[j][j] != 0.0:
                rank = j + 1
        if fnorm <= math.sqrt(max(epsfcn, _MACHINE_EPS)):
            ier = 1
            break
        gt = _trans_trisolve(R, qtf[:n], n)
        gnorm = math.sqrt(sum(g * g for g in gt)) if gt is not None else 0.0
        if gnorm == 0.0:
            ier = 4
            break
        JtJ, Jtf = _jtp(J, fvec, m, n)
        A = [row[:] for row in JtJ]
        for k in range(n):
            A[k][k] += par * (JtJ[k][k] if JtJ[k][k] > 0.0 else 1.0)
        b = [-Jtf[k] for k in range(n)]
        delta = _cholesky_solve(A, b, n)
        if delta is None:
            par *= 10.0
            if par > 1e16:
                ier = 0
                break
            continue
        dscale = _step_scale(delta, scale, n)
        if dscale <= xtol:
            ier = 2
            break
        x_new = [x[k] + delta[k] for k in range(n)]
        f_new = _eval_f(fcn, x_new, args)
        nfev += 1
        fnorm_new = math.sqrt(sum(v * v for v in f_new))
        if fnorm_new < fnorm:
            old_fnorm = fnorm
            x, fvec, fnorm = x_new, f_new, fnorm_new
            ftol_hit = (fnorm_new == 0.0) or (
                abs(old_fnorm * old_fnorm - fnorm_new * fnorm_new)
                <= ftol * old_fnorm * old_fnorm)
            xtol_hit = dscale <= xtol
            if ftol_hit or xtol_hit:
                ier = 3 if (ftol_hit and xtol_hit) else (1 if ftol_hit else 2)
                break
            par = max(par / 10.0, 1e-12)
        else:
            par *= 10.0
            if par > 1e16:
                ier = 0
                break
    return x, fvec, nfev, ier, J, R, ipvt, qtf, rank


def _hybr_core(fcn, x, args, m, n, xtol, maxfev, epsfcn, factor, scale,
               Dfun, col_deriv):
    """Damped trust-region hybrid iteration (MINPACK hybrd design).

    Returns (x, fvec, nfev, status, J, R, ipvt, qtf, rank) with the scipy C
    status codes used by fsolve/_root_hybr (1 converged, 2 maxfev, 0 improper).
    """
    fvec = _eval_f(fcn, x, args)
    nfev = 1
    fnorm = math.sqrt(sum(v * v for v in fvec))
    wnorm = math.sqrt(sum((scale[k] * x[k]) ** 2 for k in range(n)))
    par = factor * (wnorm if wnorm > 0.0 else 1.0)
    damp = 1e-3
    iterct = 0
    status = 0
    J = [[0.0] * n for _ in range(m)]
    R = [[0.0] * n for _ in range(n)]
    ipvt = list(range(n))
    qtf = list(fvec)
    rank = n
    while True:
        if iterct >= maxfev:
            status = 2
            break
        iterct += 1
        if Dfun is None:
            J, nf = _num_jacobian(fcn, x, fvec, args, epsfcn, m, n)
            nfev += nf
        else:
            J = _eval_jac(Dfun, x, args, col_deriv)
        a = [row[:] for row in J]
        f = list(fvec)
        ipvt = _qr_transform(a, f, m, n, pivot=True)
        R = [a[i][:] for i in range(n)]
        qtf = f
        rank = 0
        for j in range(n):
            if R[j][j] != 0.0:
                rank = j + 1
        gt = _trans_trisolve(R, qtf[:n], n)
        gnorm = math.sqrt(sum(g * g for g in gt)) if gt is not None else 0.0
        if gnorm == 0.0:
            status = 1
            break
        JtJ, Jtf = _jtp(J, fvec, m, n)
        A = [row[:] for row in JtJ]
        for k in range(n):
            A[k][k] += damp * (JtJ[k][k] if JtJ[k][k] > 0.0 else 1e-30)
        b = [-Jtf[k] for k in range(n)]
        delta = _cholesky_solve(A, b, n)
        if delta is None:
            damp *= 10.0
            if damp > 1e16:
                status = 0
                break
            continue
        dscale = _step_scale(delta, scale, n)
        if dscale > par:
            ratio = par / dscale
            for k in range(n):
                delta[k] *= ratio
            dscale = par
        if dscale <= xtol:
            status = 1
            break
        x_new = [x[k] + delta[k] for k in range(n)]
        f_new = _eval_f(fcn, x_new, args)
        nfev += 1
        fnorm_new = math.sqrt(sum(v * v for v in f_new))
        if fnorm_new < fnorm:
            x, fvec, fnorm = x_new, f_new, fnorm_new
            if fnorm_new == 0.0 or fnorm_new <= xtol:
                status = 1
                break
            damp = max(damp / 10.0, 1e-12)
            par = max(par, min(2.0 * par, 1.5 * dscale))
        else:
            damp *= 10.0
            par = max(0.5 * par, 1e-12)
            if damp > 1e16:
                status = 0
                break
    return x, fvec, nfev, status, J, R, ipvt, qtf, rank


def _improper(n, m, x, fvec, nfev):
    return (
        np.array(x),
        {'fvec': np.array(fvec), 'nfev': nfev,
         'fjac': np.zeros((m, n)), 'ipvt': np.arange(n),
         'r': np.zeros((n, n)), 'qtf': np.array(fvec), 'rank': 0},
        0,
    )


def _hybrd(fcn, x, args, mode, xtol, maxfev, ml, mu, epsfcn, factor, diag):
    """C-convention hybrd kernel (numerical Jacobian).

    Returns ``(x, infodict, status)`` as the deleted C extension did.
    """
    x0 = _vec(x)
    n = len(x0)
    fvec = _eval_f(fcn, x0, args)
    m = len(fvec)
    if n > m or maxfev <= 0:
        return _improper(n, m, x0, fvec, 1)
    scale = _vec(diag) if diag is not None else [1.0] * n
    (x, fvec, nfev, status, J, R, ipvt, qtf, rank) = _hybr_core(
        fcn, x0, args, m, n, xtol, maxfev, epsfcn, factor, scale,
        None, False)
    return (
        np.array(x),
        {'fvec': np.array(fvec), 'nfev': nfev, 'fjac': np.array(J),
         'ipvt': np.array(ipvt), 'r': np.array(R), 'qtf': np.array(qtf),
         'rank': rank},
        status,
    )


def _hybrj(fcn, fprime, x, args, mode, col_deriv, xtol, maxfev, factor, diag):
    """C-convention hybrj kernel (analytic Jacobian)."""
    x0 = _vec(x)
    n = len(x0)
    fvec = _eval_f(fcn, x0, args)
    m = len(fvec)
    if n > m or maxfev <= 0:
        return _improper(n, m, x0, fvec, 1)
    scale = _vec(diag) if diag is not None else [1.0] * n
    (x, fvec, nfev, status, J, R, ipvt, qtf, rank) = _hybr_core(
        fcn, x0, args, m, n, xtol, maxfev, epsfcn=None, factor=factor,
        scale=scale, Dfun=fprime, col_deriv=col_deriv)
    return (
        np.array(x),
        {'fvec': np.array(fvec), 'nfev': nfev, 'fjac': np.array(J),
         'ipvt': np.array(ipvt), 'r': np.array(R), 'qtf': np.array(qtf),
         'rank': rank},
        status,
    )


def _lmdif(func, x0, args, full_output, ftol, xtol, gtol, maxfev, epsfcn,
           factor, diag):
    """C-convention lmdif kernel (numerical Jacobian).

    Returns ``(x, infodict, ier)`` as the deleted C extension did.
    """
    x0v = _vec(x0)
    n = len(x0v)
    fvec = _eval_f(func, x0v, args)
    m = len(fvec)
    if n > m:
        return _improper(n, m, x0v, fvec, 1)
    if maxfev <= 0:
        maxfev = 200 * (n + 1)
    scale = _vec(diag) if diag is not None else [1.0] * n
    (x, fvec, nfev, ier, J, R, ipvt, qtf, rank) = _lm_core(
        func, x0v, args, m, n, ftol, xtol, gtol, maxfev, epsfcn, factor,
        scale, None, False)
    return (
        np.array(x),
        {'fvec': np.array(fvec), 'nfev': nfev, 'fjac': np.array(J),
         'ipvt': np.array(ipvt), 'r': np.array(R), 'qtf': np.array(qtf),
         'rank': rank},
        ier,
    )


def _lmder(func, Dfun, x0, args, full_output, col_deriv, ftol, xtol, gtol,
           maxfev, factor, diag):
    """C-convention lmder kernel (analytic Jacobian)."""
    x0v = _vec(x0)
    n = len(x0v)
    fvec = _eval_f(func, x0v, args)
    m = len(fvec)
    if n > m:
        return _improper(n, m, x0v, fvec, 1)
    if maxfev <= 0:
        maxfev = 100 * (n + 1)
    scale = _vec(diag) if diag is not None else [1.0] * n
    (x, fvec, nfev, ier, J, R, ipvt, qtf, rank) = _lm_core(
        func, x0v, args, m, n, ftol, xtol, gtol, maxfev,
        _MACHINE_EPS, factor, scale, Dfun, col_deriv)
    return (
        np.array(x),
        {'fvec': np.array(fvec), 'nfev': nfev, 'fjac': np.array(J),
         'ipvt': np.array(ipvt), 'r': np.array(R), 'qtf': np.array(qtf),
         'rank': rank},
        ier,
    )
