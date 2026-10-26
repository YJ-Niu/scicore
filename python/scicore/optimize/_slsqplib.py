"""Stub replacement for the deleted C ``_slsqplib`` extension.

Provides the SLSQP solver interface used by ``scipy.optimize._slsqp``.
This is a stub: real SLSQP should be ported to Rust under ``rust/src/``.
"""

from __future__ import annotations

import math

import rsnumpy as np

__all__ = ['slsqp', 'nnls']


def slsqp(func, x0, args=(), jac=None, bounds=None, constraints=(),
          maxiter=100, ftol=1e-6, disp=False, **kwargs):
    """Stub for SLSQP."""
    raise NotImplementedError("_slsqplib not yet ported to Rust/rsnumpy")


def _solve_spd(A, b, n):
    """Solve A x = b for symmetric positive definite A via Cholesky.

    Returns None when A is not positive definite.
    """
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


def nnls(A, b, maxiter=None):
    """Non-negative least squares (Lawson-Hanson active set method).

    Pure-Python replacement for the C ``nnls`` in ``_slsqplib``.  Returns
    ``(x, rnorm, info)`` where ``info`` is 0 on convergence and 3 when the
    maximum iteration count is reached.
    """
    m, n = A.shape
    A_l = [list(row) for row in np.asarray(A).tolist()]
    b_l = [float(v) for v in np.asarray(b).ravel().tolist()]
    if n == 0:
        return np.zeros(0), float(np.linalg.norm(b)), 0
    if not maxiter:
        maxiter = 3 * n
    x = [0.0] * n
    P = set(range(n))  # passive (x_k = 0)
    Z = []             # active
    converged = False
    for _ in range(maxiter):
        r = [b_l[i] - sum(A_l[i][k] * x[k] for k in range(n)) for i in range(m)]
        g = [sum(A_l[i][k] * r[i] for i in range(m)) for k in range(n)]
        if not P:
            converged = True
            break
        j = max(P, key=lambda k: g[k])
        if g[j] <= 0.0:
            converged = True
            break
        P.discard(j)
        Z.append(j)
        # Equality-constrained least squares on the active set:
        # solve (A_Z' A_Z) d = A_Z' b.
        mZ = len(Z)
        M = [[0.0] * mZ for _ in range(mZ)]
        rhs = [0.0] * mZ
        for ia in range(mZ):
            ka = Z[ia]
            for ib in range(mZ):
                kb = Z[ib]
                s = 0.0
                for i in range(m):
                    s += A_l[i][ka] * A_l[i][kb]
                M[ia][ib] = s
            s = 0.0
            for i in range(m):
                s += A_l[i][ka] * b_l[i]
            rhs[ia] = s
        d = _solve_spd(M, rhs, mZ)
        if d is None:
            for k in range(mZ):
                M[k][k] += 1e-12
            d = _solve_spd(M, rhs, mZ)
        if d is None:
            break
        x_new = [0.0] * n
        for ia in range(mZ):
            x_new[Z[ia]] = d[ia]
        # Backtrack if the equality solution leaves the nonnegative orthant:
        # move the most violated component back to the passive set.
        tmin = 1.0
        kmin = -1
        for ia in range(mZ):
            ka = Z[ia]
            if x_new[ka] < x[ka]:
                denom = x[ka] - x_new[ka]
                if denom > 0.0:
                    ratio = x[ka] / denom
                    if ratio < tmin:
                        tmin = ratio
                        kmin = ka
        if kmin < 0:
            x = x_new
            continue
        for k in range(n):
            x[k] = x[k] + tmin * (x_new[k] - x[k])
        x[kmin] = 0.0
        if kmin in Z:
            Z.remove(kmin)
        P.add(kmin)
    res = [b_l[i] - sum(A_l[i][k] * x[k] for k in range(n)) for i in range(m)]
    rnorm = math.sqrt(sum(v * v for v in res))
    info = 0 if converged else 3
    return np.array(x), rnorm, info
