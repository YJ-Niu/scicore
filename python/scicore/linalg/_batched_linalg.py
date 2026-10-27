"""Batched dense linear algebra helpers used across `scipy.linalg`.

Replaces the C++ extension built from ``scipy/linalg/src/_batched_linalg_module.cc``
and its ``_linalg_*.hh`` headers.

Only `_bandwidth` is implemented so far.  It is the sole symbol `scipy.linalg`
needs at *import* time (`_misc` pulls it in for `scipy.linalg.bandwidth`), and
it is also the only one that is self-contained: the remaining 11 routines
(`_cholesky`, `_qr`, `_svd`, `_eig`, `_eigh`, `_lu`, `_det`, `_solve`, `_inv`,
`_lstsq`) each bind a different LAPACK driver and belong with the `_flapack`
port, where the row-major to column-major handling already lives.

They raise `NotImplementedError` here rather than silently going missing, so a
caller either yields to a pure-Python fallback upstream or fails loudly.
"""

import rsnumpy as np

__all__ = ['_bandwidth']


def _bandwidth(a):
    """Compute the lower and upper bandwidth of an array.

    Scans the trailing two axes for the outermost nonzero entries below and
    above the main diagonal, as the C++ ``_linalg_bandwidth`` did.

    Parameters
    ----------
    a : array_like, at least 2-D

    Returns
    -------
    (lower, upper) : tuple of int
        ``lower`` is the largest ``i - j`` with ``a[..., i, j] != 0`` and
        ``upper`` the largest ``j - i``, both zero for a diagonal matrix and
        counted across every stacked slice for batched input.
    """
    arr = np.asarray(a)
    if arr.ndim < 2:
        raise ValueError('Input array must have at least 2 dimensions.')

    rows, cols = arr.shape[-2], arr.shape[-1]
    lower = 0
    upper = 0

    # Iterate the trailing slices; `tolist` gives nested lists to walk.
    flat = arr.reshape(-1, rows, cols).tolist()
    for matrix in flat:
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if value != 0:
                    if i - j > lower:
                        lower = i - j
                    if j - i > upper:
                        upper = j - i
    return lower, upper


# The `qr` mode flags defined by `scipy.linalg._decomp_qr`.
_FULL, _R, _RAW, _ECONOMIC = 1, 11, 21, 31


def _qr(a1, overwrite_a, modeFlag, pivoting):
    """QR factorisation backing `scipy.linalg.qr`.

    Delegates to `rsnumpy.linalg.qr`, which factors with Householder
    reflections.  That keeps this free of another LAPACK binding, at the cost of
    not exposing the elementary reflectors that `mode='raw'` needs.

    Parameters
    ----------
    a1 : rsnumpy.ndarray
        The `(M, N)` matrix, already normalised and possibly copied upstream.
    overwrite_a : bool
        Accepted for signature compatibility; the factorisation reads `a1` but
        never mutates it, so the flag cannot be honoured.
    modeFlag : int
        One of `_FULL` (1), `_R` (11), `_RAW` (21), `_ECONOMIC` (31).
    pivoting : bool
        When true a column-pivoted factorisation is requested.

    Returns
    -------
    (Q, R, tau, jpvt, err_lst) : tuple
        `Q` and `R` follow the shapes the mode demands, `tau` is `None` because
        no reflector output is produced, `jpvt` is `None` without pivoting, and
        `err_lst` collects factorisation failures to surface upstream.
    """
    if pivoting:
        raise NotImplementedError(
            'scipy.linalg.qr(pivoting=True) needs the column-pivoted driver '
            'geqp3, which is not ported yet.')

    arr = np.asarray(a1)
    if arr.ndim != 2:
        # `_apply_over_batch` strips batch dimensions before reaching here, so
        # anything else is a caller mistake rather than a missing feature.
        raise ValueError('expected a 2-D matrix, got %d dimensions'
                         % arr.ndim)

    if modeFlag == _RAW:
        raise NotImplementedError(
            'scipy.linalg.qr(mode="raw") needs the packed reflectors (tau), '
            'which the Householder-based rsnumpy factorisation does not expose.')

    if modeFlag not in (_FULL, _R, _ECONOMIC):
        raise ValueError('unknown qr mode flag %r' % (modeFlag,))

    # `_FULL` and `_R` want the (M, M)/(M, N) pair; `_ECONOMIC` wants the thin
    # (M, k)/(k, N) one for k = min(M, N).
    mode = 'complete' if modeFlag in (_FULL, _R) else 'reduced'

    try:
        q, r = np.linalg.qr(arr, mode=mode)
    except Exception as exc:  # factorisation of an empty/degenerate input
        # rsnumpy raises ahead of `qr` for empty input; let that through as-is.
        raise exc

    return np.asarray(q), np.asarray(r), None, None, []


_UNIMPLEMENTED = (
    'scipy.linalg._batched_linalg.{name} needs LAPACK driver {driver}; it is '
    'part of the pending scipy.linalg._flapack port.'
)

_DRIVERS = {
    '_cholesky': 'potrf',
    '_svd': 'gesdd',
    '_eig': 'geev',
    '_eigh': 'syevd',
    '_lu': 'getrf',
    '_det': 'getrf',
    '_solve': 'getrs',
    '_inv': 'getri',
    '_lstsq': 'gelsd',
}


def _not_implemented(name):
    def stub(*args, **kwargs):
        raise NotImplementedError(_UNIMPLEMENTED.format(name=name,
                                                        driver=_DRIVERS[name]))

    stub.__name__ = name
    stub.__qualname__ = name
    return stub


from scicore.linalg.blas import _py_jacobi_eigh  # noqa: E402


def _eigh(a1, overwrite_a, eigvals_only, vals_range, lower, vl, vu, il, iu,
          driver_index, b1=None, overwrite_b=False, type=1):
    """Pure-rsnumpy standard symmetric eigenproblem (cyclic Jacobi).

    Stacked (..., N, N) input and the generalized problem (b1 is not None)
    remain unimplemented, matching the pending _flapack port.
    """
    if b1 is not None:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_eigh',
                                                        driver='syevd'))
    a1 = np.asarray(a1, dtype=np.float64)
    if a1.ndim != 2 or a1.shape[0] != a1.shape[1]:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_eigh',
                                                        driver='syevd'))
    n = a1.shape[0]
    M = [[float(a1[i, j]) for j in range(n)] for i in range(n)]
    w, v = _py_jacobi_eigh(M, compute_v=not eigvals_only)
    if vals_range == 1:
        keep = [i for i, x in enumerate(w) if vl <= x <= vu]
    elif vals_range == 2:
        keep = list(range(il, iu + 1))
    else:
        keep = list(range(n))
    w = [w[i] for i in keep]
    m = len(w)
    if eigvals_only:
        return np.array(w), None, m, []
    Z = [[v[i][keep[j]] for j in range(m)] for i in range(n)]
    return np.array(w), np.array(Z), m, []


def _solve(a1, b1, structure, lower, transposed, overwrite_a, overwrite_b):
    """Pure-rsnumpy general linear solve (delegates to rsnumpy.linalg.solve).

    Stacked (..., M, M) input and structured matrices (any ``structure``
    other than general) remain unimplemented, matching the pending
    _flapack port.
    """
    if structure not in (None, -1, 0):
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_solve',
                                                        driver='getrs'))
    a1 = np.asarray(a1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    if a1.ndim != 2 or a1.shape[0] != a1.shape[1]:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_solve',
                                                        driver='getrs'))
    if b1.ndim == 1:
        b1 = b1.reshape(-1, 1)
    a = a1.T if transposed else a1
    return np.linalg.solve(a, b1), []


def _inv(a1, structure, overwrite_a, lower):
    """Pure-rsnumpy matrix inverse (delegates to rsnumpy.linalg.inv).

    Stacked input and structured matrices remain unimplemented.
    """
    if structure not in (None, -1, 0):
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_inv',
                                                        driver='getri'))
    a1 = np.asarray(a1, dtype=np.float64)
    if a1.ndim != 2 or a1.shape[0] != a1.shape[1]:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_inv',
                                                        driver='getri'))
    return np.linalg.inv(a1), []


def _det(a1, overwrite_a):
    """Pure-rsnumpy determinant (delegates to rsnumpy.linalg.det).

    rsnumpy returns a Python scalar for 2-D input, so the result is wrapped
    in an ndarray for the upstream dtype/ndim handling.
    """
    a1 = np.asarray(a1, dtype=np.float64)
    if a1.ndim < 2 or a1.shape[0] != a1.shape[1]:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_det',
                                                        driver='getrf'))
    return np.asarray(np.linalg.det(a1), dtype=np.float64)


def _lu(a1, permute_l, overwrite_a):
    """LU factorisation via the Rust _flapack dgetrf, with P/L/U
    reconstruction in Python.

    Returns ``(P_vec, L, U)`` where ``P_vec[i]`` is the column holding the
    one in row ``i`` of the permutation matrix (i.e. ``P = eye[P_vec, :]``),
    ``L`` is the unit-lower factor ``(M, K)`` and ``U`` the upper factor
    ``(K, N)``.  With ``permute_l`` the second return is ``P @ L`` instead.
    Stacked input remains unimplemented.
    """
    from scicore.linalg import _flapack
    from scicore.linalg._misc import LinAlgError

    if _flapack is None:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_lu',
                                                        driver='getrf'))
    a1 = np.asarray(a1, dtype=np.float64)
    if a1.ndim != 2:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_lu',
                                                        driver='getrf'))
    m, n = a1.shape
    k = min(m, n)
    lu, piv, info = _flapack.dgetrf(a1)
    if info != 0:
        raise LinAlgError('matrix is singular in LU factorisation')
    lu_l = lu.tolist()
    U = [[lu_l[i][j] if i <= j else 0.0 for j in range(n)] for i in range(k)]
    L = [[1.0 if i == j else (lu_l[i][j] if i > j and j < k else 0.0)
          for j in range(k)] for i in range(m)]
    # 0-based pivots: pivot i records the row swapped with row i; applying
    # the same swaps to the identity yields the permutation vector.
    pvec = list(range(m))
    for i, p in enumerate(piv.tolist()):
        pvec[i], pvec[int(p)] = pvec[int(p)], pvec[i]
    if permute_l:
        PL = [[L[pvec[i]][j] for j in range(k)] for i in range(m)]
        return None, np.array(PL, dtype=np.float64), np.array(U, dtype=np.float64)
    return (np.array(pvec, dtype=np.int32),
            np.array(L, dtype=np.float64),
            np.array(U, dtype=np.float64))


def _eig(a1, left, right, overwrite_a, flag, b1=None):
    """Pure-Python general eigenvalue problem (QR iteration backend).

    The rsnumpy core ``eig`` only supports 2x2/3x3 and does not return
    eigenvectors, so all sizes go through ``_eig_qr``: explicit QR
    iteration with Wilkinson shifts for the eigenvalues, SVD null-space
    for the eigenvectors.  Left eigenvectors are the right eigenvectors
    of the transpose (the backend is real-valued).  The generalized
    problem (``b1`` given) and stacked input remain unimplemented.
    """
    from scicore.linalg import _eig_qr

    if b1 is not None:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_eig',
                                                        driver='ggev'))
    a1 = np.asarray(a1, dtype=np.float64)
    if a1.ndim != 2 or a1.shape[0] != a1.shape[1]:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_eig',
                                                        driver='geev'))
    A = [[float(x) for x in row] for row in a1.tolist()]
    n = len(A)
    w_list = _eig_qr.eigenvalues(A)
    all_real = all(w.imag == 0.0 for w in w_list)
    if right:
        vr_mat = _eig_qr.eigenvector_matrix(A, w_list)
    else:
        vr_mat = None
    if left:
        # scipy convention: a.H @ vl[:, i] = conj(w[i]) @ vl[:, i], i.e. the
        # left eigenvector for w[i] is the right eigenvector of the transpose
        # for the conjugate eigenvalue.
        AT = [[A[s][r] for s in range(n)] for r in range(n)]
        vl_mat = _eig_qr.eigenvector_matrix(
            AT, [w.conjugate() for w in w_list])
    else:
        vl_mat = None
    if all_real:
        w_arr = np.array([float(w.real) for w in w_list], dtype=np.float64)
    else:
        w_arr = np.array([complex(w) for w in w_list])
    return (w_arr, None,
            np.array(vl_mat) if vl_mat is not None else None,
            np.array(vr_mat) if vr_mat is not None else None,
            [])


def _as_svd_singular_values(s):
    """Return the singular values as a 1-D float64 array.

    rsnumpy's svd fast path (2x2 input) returns them as a diagonal matrix;
    the general path returns them 1-D.  ``np.diag`` on the 2-D form is not
    reliable in rsnumpy, so the diagonal is extracted by hand.
    """
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 2:
        n = s.shape[0]
        s = np.array([float(s.tolist()[i][i]) for i in range(n)],
                     dtype=np.float64)
    return s


def _svd(a1, lapack_driver, compute_uv, full_matrices, overwrite_a):
    """Pure-rsnumpy SVD (delegates to rsnumpy.linalg.svd, which already
    returns ``Vh`` like numpy; only the singular-value shape is normalised
    to 1-D).  Stacked input remains unimplemented.
    """
    a1 = np.asarray(a1, dtype=np.float64)
    if a1.ndim != 2:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_svd',
                                                        driver='gesdd'))
    if not compute_uv:
        s = np.linalg.svd(a1, compute_uv=False, full_matrices=full_matrices)
        return (_as_svd_singular_values(s), [])
    u, s, vh = np.linalg.svd(a1, full_matrices=full_matrices)
    return np.asarray(u), _as_svd_singular_values(s), np.asarray(vh), []


def _cholesky(a1, lower, overwrite_a, clean):
    """Pure-rsnumpy Cholesky (delegates to rsnumpy.linalg.cholesky, which
    returns the lower factor; the upper triangle is its transpose).
    Stacked input remains unimplemented.
    """
    a1 = np.asarray(a1, dtype=np.float64)
    if a1.ndim != 2 or a1.shape[0] != a1.shape[1]:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_cholesky',
                                                        driver='potrf'))
    c = np.linalg.cholesky(a1)
    if not lower:
        c = c.T
    return np.asarray(c), []


def _lstsq(a1, b1, cond, driver, overwrite_a, overwrite_b):
    """Pure-rsnumpy least squares (delegates to rsnumpy.linalg.lstsq).

    The extra residual rows that LAPACK's gelsd appends for over-determined
    problems are recomputed as ``b - a @ x``.  Stacked input remains
    unimplemented.
    """
    a1 = np.asarray(a1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    if a1.ndim != 2:
        raise NotImplementedError(_UNIMPLEMENTED.format(name='_lstsq',
                                                        driver='gelsd'))
    if b1.ndim == 1:
        b1 = b1.reshape(-1, 1)
    m, n = a1.shape
    x, _residuals, rank, s = np.linalg.lstsq(a1, b1, rcond=cond)
    x = np.asarray(x)
    s = _as_svd_singular_values(s)
    rank = int(rank)
    if rank < min(m, n):
        # rsnumpy's lstsq reports rank/s correctly but does not return the
        # minimum-norm solution for rank-deficient systems; fall back to
        # the SVD pseudo-inverse (zeroing singular values below rcond*s_max).
        u, s2, vh = np.linalg.svd(a1, full_matrices=True)
        s2 = _as_svd_singular_values(s2)
        sv = [float(v) for v in s2.tolist()]
        tol = float(cond) * sv[0] if sv else 0.0
        ut_b_l = (u.T @ b1).tolist()
        coef = []
        for i in range(len(sv)):
            val = ut_b_l[i]
            if isinstance(val, list):
                val = val[0]
            coef.append(float(val) / sv[i] if sv[i] > tol else 0.0)
        x = np.asarray(vh.T @ np.array(coef, dtype=np.float64))
    if m > n:
        # Append the residual rows b - a@x, recomputed in plain Python:
        # rsnumpy broadcasts (m,1) - (m,) to (m,m), so the mixed shapes
        # must not be combined through rsnumpy arithmetic.
        x_l = x.tolist()
        if not (x_l and isinstance(x_l[0], list)):
            x_l = [[float(v)] for v in x_l]
        nrhs = len(x_l[0])
        a_l = a1.tolist()
        b_rows = []
        for row in b1.tolist():
            if isinstance(row, list):
                b_rows.append([float(v) for v in row])
            else:
                b_rows.append([float(row)])
        res_rows = []
        for i in range(m):
            res_row = []
            for c in range(nrhs):
                s = 0.0
                for k in range(n):
                    s += a_l[i][k] * x_l[k][c]
                res_row.append(b_rows[i][c] - s)
            res_rows.append(res_row)
        x = np.array(x_l + res_rows, dtype=np.float64)
    return x, rank, s, []
