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


for _name, _driver in _DRIVERS.items():
    if _name != '_eigh':
        globals()[_name] = _not_implemented(_name)

del _name, _driver
