"""Pure-Python eigenvalue solver for general real square matrices.

Backend for ``scicore.linalg.eig`` when the matrix is larger than 3x3:
the rsnumpy core ``eig`` only supports 2x2/3x3 and does not return
eigenvectors.  Implements the classical explicit QR iteration with
Wilkinson shifts (Trefethen & Bau, "Numerical Linear Algebra"), with the
termination test for complex conjugate pairs that the plain real QR
iteration needs:

1. balance: diagonal similarity scaling to reduce element spread;
2. explicit QR steps (Householder factorisation) with Wilkinson shifts
   from the trailing 2x2 corner of the active leading block;
3. deflation when the active block's subdiagonal element is small, and
   acceptance of a trailing 2x2 block whose eigenvalues form a complex
   conjugate pair;
4. eigenvalues from the converged 1x1 / 2x2 blocks;
5. eigenvectors by SVD null space: the eigenvector for ``lambda`` is the
   right singular vector of ``(A - lambda I)`` for the smallest singular
   value (real block ``[[A-aI, -bI], [bI, A-aI]]`` for complex lambda).

All numerics are plain Python floats/complexes; inputs and outputs are
nested lists.
"""

from __future__ import annotations

import math

import rsnumpy as np
from rsnumpy.linalg import LinAlgError

__all__ = ['eigenvalues', 'eigenvectors', 'eigenvector_matrix']


def _matmul(P, Q):
    """Product of two square nested-list matrices."""
    n = len(P)
    return [[sum(P[r][t] * Q[t][c] for t in range(n)) for c in range(n)]
            for r in range(n)]


def _householder_qr(M):
    """Full Householder QR factorisation of a square matrix (lists).

    Returns ``(Q, R)`` with ``M = Q @ R``, ``Q`` orthogonal.
    """
    n = len(M)
    R = [row[:] for row in M]
    Q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for k in range(n):
        x = [R[i][k] for i in range(k, n)]
        nrm = math.sqrt(sum(t * t for t in x))
        if nrm == 0.0:
            continue
        x[0] += math.copysign(nrm, x[0])
        nrmx = math.sqrt(sum(t * t for t in x))
        if nrmx == 0.0:
            continue
        v = [t / nrmx for t in x]
        # R <- P R: only rows k..n-1 change (v is supported there); columns
        # < k are already zero by the QR invariant, so the trailing block
        # update suffices.
        w = [sum(v[i - k] * R[i][j] for i in range(k, n)) for j in range(k, n)]
        for i in range(k, n):
            for j in range(k, n):
                R[i][j] -= 2.0 * v[i - k] * w[j - k]
        # Q <- Q P = Q - 2 (Q v) v^T:  ALL rows change (Q v has entries in
        # every row), columns k..n-1.
        Qv = [sum(v[i2 - k] * Q[i][i2] for i2 in range(k, n)) for i in range(n)]
        for i in range(n):
            for j in range(k, n):
                Q[i][j] -= 2.0 * Qv[i] * v[j - k]
    return Q, R


def _qr_step(H, sigma):
    """One shifted explicit QR step: H <- RQ + sigma*I for H - sigma*I = QR."""
    n = len(H)
    M = [[H[r][s] - (sigma if r == s else 0.0) for s in range(n)] for r in range(n)]
    Q, R = _householder_qr(M)
    RQ = _matmul(R, Q)
    return [[RQ[r][s] + (sigma if r == s else 0.0) for s in range(n)]
            for r in range(n)]


def _balance(A):
    """Diagonal similarity scaling (row/column off-diagonal equilibration).

    Returns ``(H, scale)`` with ``H = D^-1 A D`` and ``D = diag(scale)``.
    """
    n = len(A)
    H = [row[:] for row in A]
    scale = [1.0] * n
    for i in range(n):
        for _ in range(200):
            r = sum(abs(H[i][j]) for j in range(n) if j != i)
            c = sum(abs(H[j][i]) for j in range(n) if j != i)
            if r == 0.0 or c == 0.0:
                break
            s = math.sqrt(r / c)
            if 0.5 < s < 2.0:
                break
            scale[i] *= s
            for j in range(n):
                if j == i:
                    continue
                H[i][j] /= s
                H[j][i] *= s
    return H, scale


def eigenvalues(A):
    """All eigenvalues of a real n x n matrix, as a list of complex.

    The ordering is not guaranteed (LAPACK-like): blocks are emitted from
    the trailing corner backwards.
    """
    n = len(A)
    if n == 1:
        return [complex(A[0][0], 0.0)]
    H, _scale = _balance(A)
    w = []
    j = n - 1
    steps = 0
    max_steps = 1000 * n
    while j >= 1:
        a = H[j - 1][j - 1]
        b = H[j - 1][j]
        c = H[j][j - 1]
        d = H[j][j]
        tol = 1e-14 * (abs(a) + abs(d))
        if abs(c) <= tol:
            w.append(complex(d, 0.0))
            j -= 1
            continue
        e = (a - d) / 2.0
        disc = e * e + b * c
        if disc < 0.0:
            # Complex conjugate pair: accept the 2x2 as a real Schur block.
            re = (a + d) / 2.0
            im = math.sqrt(-disc)
            w.append(complex(re, im))
            w.append(complex(re, -im))
            j -= 2
            continue
        # Wilkinson shift: eigenvalue of [a b; c d] closest to d.
        sigma = d + math.copysign(1.0, e if e != 0.0 else 1.0) * math.hypot(e, c)
        H = _qr_step(H, sigma)
        steps += 1
        if steps > max_steps:
            raise LinAlgError("QR iteration did not converge")
    # The loop leaves j == 0 (one leading 1x1 block remains) or j == -1
    # (a trailing 2x2 pair consumed the last two rows).
    if j == 0:
        w.append(complex(H[0][0], 0.0))
    return w


def _eigenvector(A, lam):
    """Unit eigenvector of (real) A for eigenvalue lam (complex scalar).

    Computed as the right singular vector for the smallest singular value
    of (A - lam I); for complex lam the real block
    [[A-aI, -bI], [bI, A-aI]] is used so everything stays real.
    """
    n = len(A)
    a = lam.real
    b = lam.imag
    if n == 1:
        v = [complex(1.0, 0.0)]
    elif b == 0.0:
        B = [[A[r][s] - (a if r == s else 0.0) for s in range(n)]
             for r in range(n)]
        _u, _s, vh = np.linalg.svd(B, full_matrices=False)
        v = [float(x) for x in np.asarray(vh).tolist()[n - 1]]
    else:
        B = []
        for r in range(2 * n):
            row = []
            for s in range(2 * n):
                if r < n:
                    if s < n:
                        val = A[r][s] - (a if r == s else 0.0)
                    else:
                        val = -b if (s - n) == r else 0.0
                else:
                    if s < n:
                        val = b if (r - n) == s else 0.0
                    else:
                        val = (A[r - n][s - n]
                               - (a if (r - n) == (s - n) else 0.0))
                row.append(val)
            B.append(row)
        _u, _s, vh = np.linalg.svd(B, full_matrices=False)
        last = [float(x) for x in np.asarray(vh).tolist()[2 * n - 1]]
        # Null vectors (x, y) of the block [[A-aI, -bI], [bI, A-aI]]
        # satisfy A (x - i y) = (a + b i) (x - i y).
        v = [complex(last[s], -last[n + s]) for s in range(n)]
    nrm = math.sqrt(sum(abs(t) ** 2 for t in v))
    if nrm == 0.0:
        v = [complex(0.0, 0.0) for _ in range(n)]
        v[0] = complex(1.0, 0.0)
        return v
    v = [t / nrm for t in v]
    # Convention: largest-magnitude component real and positive.
    idx = max(range(n), key=lambda i: abs(v[i]))
    if b == 0.0:
        if v[idx].real < 0.0:
            v = [-t for t in v]
    else:
        phase = v[idx] / abs(v[idx])
        v = [t / phase for t in v]
    return v


def eigenvectors(A, w_list):
    """Eigenvector for each eigenvalue in ``w_list`` (list of vectors)."""
    return [_eigenvector(A, w) for w in w_list]


def eigenvector_matrix(A, w_list):
    """Assemble ``eigenvectors(A, w_list)`` into a nested list.

    Row ``r``, column ``c`` is the r-th component of eigenvector c.
    """
    vecs = eigenvectors(A, w_list)
    n = len(vecs[0])
    has_cplx = any(x.imag != 0.0 for vec in vecs for x in vec)
    if has_cplx:
        return [[complex(vecs[c][r]) for c in range(n)] for r in range(n)]
    return [[float(vecs[c][r]) for c in range(n)] for r in range(n)]
