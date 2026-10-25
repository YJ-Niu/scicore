"""Stub replacement for the deleted C ``_superlu`` extension.

Provides SuperLU sparse direct solver interface used by
``scipy.sparse.linalg._dsolve.linsolve``.
This is a stub: real sparse direct solve should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['gssv', 'gstrf', 'gstrs', 'SuperLU']


def gssv(N, nnz, a, ia, ja, b, ldb, options=None):
    """Solve ``A x = b`` (scipy-compatible ``gssv`` wrapper).

    Pure-Python LU with partial pivoting on a dense assembly of the sparse
    matrix.  ``ldb`` selects the storage of ``a/ia/ja`` (0 = CSR, 1 = CSC);
    ``b`` is 1-D (single right-hand side) or 2-D with shape ``(N, nrhs)``
    (C order).  Returns ``(x, info)``: ``x`` has shape ``(N, nrhs)``, and
    ``info`` is 0 on success or the 1-based pivot index of a zero pivot.
    ``options`` is accepted for API compatibility (column permutations are
    not implemented).
    """
    a = [float(v) for v in a]
    ia = [int(v) for v in ia]
    ja = [int(v) for v in ja]
    A = [[0.0] * N for _ in range(N)]
    if ldb == 0:  # CSR
        for i in range(N):
            for k in range(ia[i], ia[i + 1]):
                A[i][ja[k]] = a[k]
    else:  # CSC
        for j in range(N):
            for k in range(ia[j], ia[j + 1]):
                A[ja[k]][j] = a[k]

    b_arr = np.asarray(b)
    if b_arr.ndim == 1:
        cols = [[float(b_arr[i]) for i in range(N)]]
    else:
        b_flat = [float(v) for v in b_arr.ravel()]
        nrhs2 = int(b_arr.shape[1])
        cols = [[b_flat[i + j * N] for i in range(N)] for j in range(nrhs2)]

    sols = []
    info = 0
    for col in cols:
        M = [row[:] for row in A]
        rhs = list(col)
        singular = 0
        for p in range(N):
            piv = p
            best = abs(M[p][p])
            for r in range(p + 1, N):
                if abs(M[r][p]) > best:
                    best = abs(M[r][p])
                    piv = r
            if best == 0.0:
                singular = p + 1
                break
            if piv != p:
                M[p], M[piv] = M[piv], M[p]
                rhs[p], rhs[piv] = rhs[piv], rhs[p]
            for r in range(p + 1, N):
                f = M[r][p] / M[p][p]
                M[r][p] = f
                for c in range(p + 1, N):
                    M[r][c] -= f * M[p][c]
                rhs[r] -= f * rhs[p]
        if singular:
            info = max(info, singular)
            sols.append([float("nan")] * N)
            continue
        sol = [0.0] * N
        for r in range(N - 1, -1, -1):
            s = rhs[r]
            for c in range(r + 1, N):
                s -= M[r][c] * sol[c]
            sol[r] = s / M[r][r]
        sols.append(sol)

    nrhs = len(sols)
    rows = [[sols[j][i] for j in range(nrhs)] for i in range(N)]
    x = np.array(rows)
    return x, info


def gstrf(M, N, nnz, a, ia, ja, perm_c, perm_r, etree, info, L, U, work):
    """Stub for SuperLU gstrf (factorize sparse matrix)."""
    info[0] = 0


def gstrs(trans, L, U, perm_c, perm_r, nrhs, b, ldb, info, work):
    """Stub for SuperLU gstrs (solve factored system)."""
    info[0] = 0


class SuperLU:
    """Stub SuperLU object."""

    def __init__(self, A):
        self.A = A
        self.L = None
        self.U = None
        self.perm_c = None
        self.perm_r = None

    def solve(self, b):
        """Stub solve."""
        return np.zeros_like(b)

    def solve_transpose(self, b):
        """Stub solve_transpose."""
        return np.zeros_like(b)
