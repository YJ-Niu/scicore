"""Pure Python replacement for the deleted Cython
``_decomp_interpolative`` extension — the "id" (interpolative
decomposition) backend consumed by :mod:`scicore.linalg.interpolative`.

Naming follows the idd/idz C library: ``idd*``/``idz*`` select the
real/complex dense routines, the ``p``/``r`` position selects the
eps-based (numerical rank) vs rank-based (explicit ``k``) interface,
``i``/``a`` the deterministic/randomized pivot selection, and the
``...rid``/``...rsvd`` names the LinearOperator-input routines.

Core algorithm (deterministic variants): greedy rank-revealing column
selection by residual maximization (the pivoting decisions of a
column-pivoted QR), with the projection computed as
``pinv(U) A = (U^H U)^(-1) U^H A``.  The small k x k solve uses
Gaussian elimination with partial pivoting in pure Python, because
rsnumpy's ``solve`` does not raise on singular input.  Randomized
variants pick each pivot uniformly among the columns whose residual
energy is within 90% of the current maximum.  LinearOperator inputs
are densified with n matvecs (``A @ I``), adequate for the sizes this
port is validated at.
"""

from __future__ import annotations

import math

import rsnumpy as np

__all__ = [
    'iddp_id', 'iddp_aid', 'iddr_id', 'iddr_aid',
    'idzp_id', 'idzp_aid', 'idzr_id', 'idzr_aid',
    'iddp_rid', 'iddr_rid', 'idzp_rid', 'idzr_rid',
    'idd_reconid', 'idz_reconid',
    'idd_reconint', 'idz_reconint',
    'idd_copycols', 'idz_copycols',
    'idd_id2svd', 'idz_id2svd',
    'idd_estrank', 'idz_estrank',
    'idd_findrank', 'idz_findrank',
    'iddp_svd', 'iddp_asvd', 'iddp_rsvd',
    'iddr_svd', 'iddr_asvd', 'iddr_rsvd',
    'idzp_svd', 'idzp_asvd', 'idzp_rsvd',
    'idzr_svd', 'idzr_asvd', 'idzr_rsvd',
    'idd_snorm', 'idz_snorm',
    'idd_diffsnorm', 'idz_diffsnorm',
]

_EPS = 2.220446049250313e-16  # float64 machine epsilon


# ----------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------

def _rows_from_array(A, is_complex):
    A = np.asarray(A)
    if is_complex:
        return [[complex(t) for t in row] for row in A.tolist()]
    return [[float(t) for t in row] for row in A.tolist()]


def _array_from_rows(rows, is_complex):
    if is_complex:
        re = np.array([[t.real for t in row] for row in rows],
                      dtype=np.float64)
        im = np.array([[t.imag for t in row] for row in rows],
                      dtype=np.float64)
        return re + 1j * im
    return np.array(rows, dtype=np.float64)


def _solve_small(G, rhs, k):
    """Solve G X = rhs (G: k x k, rhs: k x n), real or complex."""
    n = len(rhs[0]) if k else 0
    a = [[G[i][j] for j in range(k)] + [rhs[i][j] for j in range(n)]
         for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) <= 1e-300:
            raise np.linalg.LinAlgError(
                'singular system in interpolative decomposition')
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
        inv = 1.0 / a[col][col]
        for j in range(col, k + n):
            a[col][j] *= inv
        for r in range(k):
            if r == col or a[r][col] == 0:
                continue
            f = a[r][col]
            for j in range(col, k + n):
                a[r][j] -= f * a[col][j]
    return [[a[i][k + j] for j in range(n)] for i in range(k)]


def _select_pivots(cols, m, n, k, rand, rng):
    """Greedy (optionally randomized) rank-revealing column selection."""
    energy = [sum(t * t.conjugate() for t in cols[j]).real for j in range(n)]
    basis = []
    pivots = []
    for _ in range(k):
        emax = max(energy)
        if emax <= 1e-30:
            used = set(pivots)
            p = next(j for j in range(n) if j not in used)
        elif rand:
            cands = [j for j in range(n) if energy[j] >= 0.9 * emax]
            p = cands[int(rng.integers(0, len(cands)))]
        else:
            p = max(range(n), key=lambda j: energy[j])
        pivots.append(p)
        v = list(cols[p])
        for u in basis:
            c = sum(tu.conjugate() * tv for tu, tv in zip(u, v))
            v = [vv - c * uu for vv, uu in zip(v, u)]
        nrm2 = sum(t * t.conjugate() for t in v).real
        if nrm2 > 1e-28:
            nrm = math.sqrt(nrm2)
            uh = [t / nrm for t in v]
        else:
            uh = [0.0] * m
        basis.append(uh)
        for j in range(n):
            proj = sum(tu.conjugate() * tj for tu, tj in zip(uh, cols[j]))
            energy[j] -= (proj * proj.conjugate()).real
    return pivots


def _id_core(A_rows, m, n, k, rand, rng):
    """Return (pivots, proj): B = A[:, pivots], A = B @ proj (exact for
    k = rank(A)), with the rows of proj at the pivot positions
    forming the identity."""
    cols = [[A_rows[i][j] for i in range(m)] for j in range(n)]
    pivots = _select_pivots(cols, m, n, k, rand, rng)

    def dotcv(x, y):
        return sum(tu.conjugate() * tv for tu, tv in zip(x, y))

    Ucols = [[A_rows[i][p] for i in range(m)] for p in pivots]
    G = [[dotcv(Ucols[a], Ucols[b]) for b in range(k)] for a in range(k)]
    Bmat = [[dotcv(Ucols[a], cols[j]) for j in range(n)] for a in range(k)]
    proj = _solve_small(G, Bmat, k)
    return pivots, proj


def _num_rank(A_rows, m, n, eps, is_complex):
    if m == 0 or n == 0:
        return 0
    M = _array_from_rows(A_rows, is_complex)
    s = [float(t) for t in
         np.asarray(np.linalg.svd(M, compute_uv=False)).tolist()]
    if not s:
        return 0
    tol = eps * s[0]
    return sum(1 for t in s if t > tol)


def _dense_id(A_rows, m, n, eps, k, rand, rng, is_complex):
    if k is None:
        k = _num_rank(A_rows, m, n, eps, is_complex)
    else:
        # rank-based request: cap k by the numerical rank at machine
        # precision (protects the k x k solve from singular input)
        rank = _num_rank(A_rows, m, n, max(m, n) * _EPS, is_complex)
        k = int(min(k, rank, m, n))
    if k == 0:
        return 0, [], []
    pivots, proj = _id_core(A_rows, m, n, k, rand, rng)
    return k, pivots, proj


def _id_arrays(k, pivots, proj, m, n, is_complex):
    idx = (np.array(pivots, dtype=np.int64) if pivots
           else np.zeros(0, dtype=np.int64))
    if k == 0:
        proj_arr = np.array([[] for _ in range(n)]).T
    else:
        proj_arr = _array_from_rows(proj, is_complex)
    return idx, proj_arr


def _dense_id_call(A, eps, k, rand, rng, is_complex):
    A = np.asarray(A)
    m, n = A.shape
    A_rows = _rows_from_array(A, is_complex)
    kk, pivots, proj = _dense_id(A_rows, m, n, eps, k, rand, rng,
                                 is_complex)
    idx, proj_arr = _id_arrays(kk, pivots, proj, m, n, is_complex)
    return kk, idx, proj_arr


def _lo_rows(A, is_complex):
    m, n = A.shape
    cols = []
    for j in range(n):
        # A @ e_j (e_j in the n-dimensional domain) is column j of A
        e = np.array([1.0 if j == i else 0.0 for i in range(n)])
        c = np.asarray(A.dot(e))
        cols.append(list(c.tolist()))
    A_rows = [[cols[j][i] for j in range(n)] for i in range(m)]
    if is_complex:
        A_rows = [[complex(t) for t in row] for row in A_rows]
    else:
        A_rows = [[float(t) for t in row] for row in A_rows]
    return A_rows, m, n


def _lo_id_call(A, eps, k, rng, is_complex):
    A_rows, m, n = _lo_rows(A, is_complex)
    kk, pivots, proj = _dense_id(A_rows, m, n, eps, k, True, rng,
                                 is_complex)
    idx, proj_arr = _id_arrays(kk, pivots, proj, m, n, is_complex)
    B_rows = ([[A_rows[i][p] for p in pivots] for i in range(m)]
              if pivots else [[] for _ in range(m)])
    return kk, idx, proj_arr, B_rows


# ----------------------------------------------------------------------
# public backend API
# ----------------------------------------------------------------------

# -- interp_decomp family (dense) --

def iddp_id(A, eps):
    k, idx, proj = _dense_id_call(A, eps, None, False, None, False)
    return k, idx, proj


def iddp_aid(A, eps, rng=None):
    k, idx, proj = _dense_id_call(A, eps, None, True, rng, False)
    return k, idx, proj


def iddr_id(A, k):
    k, idx, proj = _dense_id_call(A, None, k, False, None, False)
    return idx, proj


def iddr_aid(A, k, rng=None):
    k, idx, proj = _dense_id_call(A, None, k, True, rng, False)
    return idx, proj


def idzp_id(A, eps):
    k, idx, proj = _dense_id_call(A, eps, None, False, None, True)
    return k, idx, proj


def idzp_aid(A, eps, rng=None):
    k, idx, proj = _dense_id_call(A, eps, None, True, rng, True)
    return k, idx, proj


def idzr_id(A, k):
    k, idx, proj = _dense_id_call(A, None, k, False, None, True)
    return idx, proj


def idzr_aid(A, k, rng=None):
    k, idx, proj = _dense_id_call(A, None, k, True, rng, True)
    return idx, proj


# -- interp_decomp family (LinearOperator) --

def iddp_rid(A, eps, rng=None):
    k, idx, proj, _B = _lo_id_call(A, eps, None, rng, False)
    return k, idx, proj


def iddr_rid(A, k, rng=None):
    k, idx, proj, _B = _lo_id_call(A, None, k, rng, False)
    return idx, proj


def idzp_rid(A, eps, rng=None):
    k, idx, proj, _B = _lo_id_call(A, eps, None, rng, True)
    return k, idx, proj


def idzr_rid(A, k, rng=None):
    k, idx, proj, _B = _lo_id_call(A, None, k, rng, True)
    return idx, proj


# -- reconstruction / skeleton utilities --

def idd_reconid(B, idx, proj):
    return np.dot(B, proj)


def idz_reconid(B, idx, proj):
    return np.dot(B, proj)


def idd_reconint(idx, proj):
    return proj


def idz_reconint(idx, proj):
    return proj


def idd_copycols(A, k, idx):
    return A[:, idx]


def idz_copycols(A, k, idx):
    return A[:, idx]


# -- SVD conversions --

def _id2svd_core(B, proj, k):
    R = np.dot(B, proj)
    m, n = R.shape
    if k == 0:
        return (np.array([[] for _ in range(m)]),
                np.zeros(0, dtype=np.float64),
                np.array([[] for _ in range(n)]).T)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]


def _svd_dense(A, eps, k, rand, rng, is_complex, transpose_V):
    kk, idx, proj = _dense_id_call(A, eps, k, rand, rng, is_complex)
    Aa = np.asarray(A)
    B = Aa[:, idx] if kk else np.array([[] for _ in range(Aa.shape[0])])
    U, s, Vt = _id2svd_core(B, proj, kk)
    return U, s, Vt.T if transpose_V else Vt


def iddp_svd(A, eps):
    return _svd_dense(A, eps, None, False, None, False, False)


def iddp_asvd(A, eps, rng=None):
    return _svd_dense(A, eps, None, True, rng, False, True)


def iddr_svd(A, k):
    return _svd_dense(A, None, k, False, None, False, False)


def iddr_asvd(A, k, rng=None):
    return _svd_dense(A, None, k, True, rng, False, True)


def idzp_svd(A, eps):
    return _svd_dense(A, eps, None, False, None, True, False)


def idzp_asvd(A, eps, rng=None):
    return _svd_dense(A, eps, None, True, rng, True, True)


def idzr_svd(A, k):
    return _svd_dense(A, None, k, False, None, True, False)


def idzr_asvd(A, k, rng=None):
    return _svd_dense(A, None, k, True, rng, True, True)


def _svd_lo(A, eps, k, rng, is_complex):
    kk, _idx, proj, B_rows = _lo_id_call(A, eps, k, rng, is_complex)
    B = _array_from_rows(B_rows, is_complex)
    U, s, Vt = _id2svd_core(B, proj, kk)
    return U, s, Vt.T


# -- rank estimation --

def idd_estrank(A, eps, rng=None):
    A = np.asarray(A)
    return _num_rank(_rows_from_array(A, False), A.shape[0], A.shape[1],
                     eps, False), 0.0


def idz_estrank(A, eps, rng=None):
    A = np.asarray(A)
    return _num_rank(_rows_from_array(A, True), A.shape[0], A.shape[1],
                     eps, True), 0.0


def idd_findrank(A, eps, rng=None):
    A_rows, m, n = _lo_rows(A, False)
    return _num_rank(A_rows, m, n, eps, False), 0.0


def idz_findrank(A, eps, rng=None):
    A_rows, m, n = _lo_rows(A, True)
    return _num_rank(A_rows, m, n, eps, True), 0.0


# -- spectral norm estimation (power iteration) --

def _random_vector(n, rng, is_complex):
    a = [float(t) for t in
         np.asarray(rng.uniform(-1.0, 1.0, size=(n,))).tolist()]
    if not is_complex:
        return np.array(a, dtype=np.float64)
    b = [float(t) for t in
         np.asarray(rng.uniform(-1.0, 1.0, size=(n,))).tolist()]
    return (np.array(a, dtype=np.float64)
            + 1j * np.array(b, dtype=np.float64))


def _power_norm(apply, adjoint, n, its, rng, is_complex):
    """Spectral norm via power iteration on the domain of ``apply``.

    x <- adjoint(apply(x)) / ||.|| alternates between the domain (n)
    and range (m) spaces, so it is valid for rectangular operators;
    the Rayleigh-Ritz estimate ||apply(x)|| of the largest singular
    value is returned.
    """
    rng = np.random.default_rng(rng)
    x = _random_vector(n, rng, is_complex)
    last = 0.0
    for _ in range(its):
        y = np.asarray(apply(x))
        nrm = float(np.linalg.norm(y))
        if nrm == 0.0:
            return 0.0
        z = np.asarray(adjoint(y))
        nrm_z = float(np.linalg.norm(z))
        if nrm_z == 0.0:
            return 0.0
        x = z / nrm_z
        last = nrm
    return last


def idd_snorm(A, its, rng=None):
    return _power_norm(A.dot, A.rdot, A.shape[1], its, rng, False)


def idz_snorm(A, its, rng=None):
    return _power_norm(A.dot, A.rdot, A.shape[1], its, rng, True)


def idd_diffsnorm(A, B, its, rng=None):
    def diff(x):
        return A.dot(x) - B.dot(x)

    def diff_adj(y):
        return A.rdot(y) - B.rdot(y)
    return _power_norm(diff, diff_adj, A.shape[1], its, rng, False)


def idz_diffsnorm(A, B, its, rng=None):
    def diff(x):
        return A.dot(x) - B.dot(x)

    def diff_adj(y):
        return A.rdot(y) - B.rdot(y)
    return _power_norm(diff, diff_adj, A.shape[1], its, rng, True)
