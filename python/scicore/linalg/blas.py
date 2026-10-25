"""
Low-level BLAS functions (:mod:`scipy.linalg.blas`)
===================================================

This module contains low-level functions from the BLAS library.

.. versionadded:: 0.12.0

.. note::

   The common ``overwrite_<>`` option in many routines, allows the
   input arrays to be overwritten to avoid extra memory allocation.
   However this requires the array to satisfy two conditions
   which are memory order and the data type to match exactly the
   order and the type expected by the routine.

   As an example, if you pass a double precision float array to any
   ``S....`` routine which expects single precision arguments, f2py
   will create an intermediate array to match the argument types and
   overwriting will be performed on that intermediate array.

   Similarly, if a C-contiguous array is passed, f2py will pass a
   FORTRAN-contiguous array internally. Please make sure that these
   details are satisfied. More information can be found in the f2py
   documentation.

.. warning::

   These functions do little to no error checking.
   It is possible to cause crashes by misusing them,
   so prefer using the higher-level routines in `scipy.linalg`.

.. note::

    Prefer using ``get_blas_funcs`` to importing the bare functions directly.
    If you do, for example, ``from scicore.linalg.blas import dgemm``, the ``dgemm``
    function may be either LP64 or ILP64, depending on how SciPy is built.

    The following is more robust:

    >>> from scicore.linalg.blas import get_blas_funcs
    >>> dgemm = get_blas_funcs('gemm', dtype='float64', ilp64='preferred')
    >>> dgemm.int_dtype
    dtype('int32')    # may vary


Finding functions
-----------------

.. autosummary::
   :toctree: generated/

   get_blas_funcs
   find_best_blas_type

BLAS Level 1 functions
----------------------

.. autosummary::
   :toctree: generated/

   sasum
   saxpy
   scasum
   scnrm2
   scopy
   sdot
   snrm2
   srot
   srotg
   srotm
   srotmg
   sscal
   sswap
   dasum
   daxpy
   dcopy
   ddot
   dnrm2
   drot
   drotg
   drotm
   drotmg
   dscal
   dswap
   dzasum
   dznrm2
   icamax
   idamax
   isamax
   izamax
   caxpy
   ccopy
   cdotc
   cdotu
   crotg
   cscal
   csrot
   csscal
   cswap
   zaxpy
   zcopy
   zdotc
   zdotu
   zdrot
   zdscal
   zrotg
   zscal
   zswap

BLAS Level 2 functions
----------------------

.. autosummary::
   :toctree: generated/

   sgbmv
   sgemv
   sger
   ssbmv
   sspmv
   sspr
   sspr2
   ssymv
   ssyr
   ssyr2
   stbmv
   stbsv
   stpmv
   stpsv
   strmv
   strsv
   dgbmv
   dgemv
   dger
   dsbmv
   dspmv
   dspr
   dspr2
   dsymv
   dsyr
   dsyr2
   dtbmv
   dtbsv
   dtpmv
   dtpsv
   dtrmv
   dtrsv
   cgbmv
   cgemv
   cgerc
   cgeru
   chbmv
   chemv
   cher
   cher2
   chpmv
   chpr
   chpr2
   cspmv
   cspr
   csymv
   csyr
   ctbmv
   ctbsv
   ctpmv
   ctpsv
   ctrmv
   ctrsv
   zgbmv
   zgemv
   zgerc
   zgeru
   zhbmv
   zhemv
   zher
   zher2
   zhpmv
   zhpr
   zhpr2
   zspmv
   zspr
   zsymv
   zsyr
   ztbmv
   ztbsv
   ztpmv
   ztpsv
   ztrmv
   ztrsv

BLAS Level 3 functions
----------------------

.. autosummary::
   :toctree: generated/

   sgemm
   ssymm
   ssyr2k
   ssyrk
   strmm
   strsm
   dgemm
   dsymm
   dsyr2k
   dsyrk
   dtrmm
   dtrsm
   cgemm
   chemm
   cher2k
   cherk
   csymm
   csyr2k
   csyrk
   ctrmm
   ctrsm
   zgemm
   zhemm
   zher2k
   zherk
   zsymm
   zsyr2k
   zsyrk
   ztrmm
   ztrsm

"""
#
# Author: Pearu Peterson, March 2002
#         refactoring by Fabian Pedregosa, March 2010
#

__all__ = ['get_blas_funcs', 'find_best_blas_type']

import rsnumpy as np
import functools
from scicore.__config__ import CONFIG

# If `_fblas` was built, it means the Cython BLAS ABI is LP64, and we're then also
# keeping `linalg.blas` as LP64.
HAS_LP64 = not bool(CONFIG['Build Dependencies']['blas']['cython blas ilp64'])
HAS_ILP64 = CONFIG['Build Dependencies']['blas']['has ilp64']
del CONFIG

if HAS_LP64:
    from scicore.linalg import _fblas
else:
    _fblas = None

if HAS_ILP64:
    from scicore.linalg import _fblas_64
else:
    _fblas_64 = None

if not (HAS_LP64 or HAS_ILP64):
    raise RuntimeError("SciPy needs either LP64 or ILP64 BLAS.")

if HAS_LP64:
    from scicore.linalg._fblas import *  # noqa: F403
else:
    from scicore.linalg._fblas_64 import *  # noqa: F403

# all numeric dtypes '?bBhHiIlLqQefdgFDGO' that are safe to be converted to

# single precision float   : '?bBhH!!!!!!ef!!!!!!'
# double precision float   : '?bBhHiIlLqQefdg!!!!'
# single precision complex : '?bBhH!!!!!!ef!!F!!!'
# double precision complex : '?bBhHiIlLqQefdgFDG!'

_type_score = {x: 1 for x in '?bBhHef'}
_type_score.update({x: 2 for x in 'iIlLqQd'})

# Handle float128(g) and complex256(G) separately in case non-Windows systems.
# On Windows, the values will be rewritten to the same key with the same value.
_type_score.update({'F': 3, 'D': 4, 'g': 2, 'G': 4})

# Final mapping to the actual prefixes and dtypes
_type_conv = {1: ('s', np.dtype('float32')),
              2: ('d', np.dtype('float64')),
              3: ('c', np.dtype('complex64')),
              4: ('z', np.dtype('complex128'))}

# some convenience alias for complex functions
_blas_alias = {'cnrm2': 'scnrm2', 'znrm2': 'dznrm2',
               'cdot': 'cdotc', 'zdot': 'zdotc',
               'cger': 'cgerc', 'zger': 'zgerc',
               'sdotc': 'sdot', 'sdotu': 'sdot',
               'ddotc': 'ddot', 'ddotu': 'ddot'}


def find_best_blas_type(arrays=(), dtype=None):
    """Find best-matching BLAS/LAPACK type.

    Arrays are used to determine the optimal prefix of BLAS routines.

    Parameters
    ----------
    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.
    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    Returns
    -------
    prefix : str
        BLAS/LAPACK prefix character.
    dtype : dtype
        Inferred Numpy data type.
    prefer_fortran : bool
        Whether to prefer Fortran order routines over C order.

    Examples
    --------
    >>> import rsnumpy as np
    >>> import scicore.linalg.blas as bla
    >>> rng = np.random.default_rng()
    >>> a = rng.random((10,15))
    >>> b = np.asfortranarray(a)  # Change the memory layout order
    >>> bla.find_best_blas_type((a,))
    ('d', dtype('float64'), False)
    >>> bla.find_best_blas_type((a*1j,))
    ('z', dtype('complex128'), False)
    >>> bla.find_best_blas_type((b,))
    ('d', dtype('float64'), True)

    """
    dtype = np.dtype(dtype)
    max_score = _type_score.get(dtype.char, 5)
    prefer_fortran = False

    if arrays:
        # In most cases, single element is passed through, quicker route
        if len(arrays) == 1:
            max_score = _type_score.get(arrays[0].dtype.char, 5)
            prefer_fortran = arrays[0].flags['FORTRAN']
        else:
            # use the most generic type in arrays
            scores = [_type_score.get(x.dtype.char, 5) for x in arrays]
            max_score = max(scores)
            ind_max_score = scores.index(max_score)
            # safe upcasting for mix of float64 and complex64 --> prefix 'z'
            if max_score == 3 and (2 in scores):
                max_score = 4

            if arrays[ind_max_score].flags['FORTRAN']:
                # prefer Fortran for leading array with column major order
                prefer_fortran = True

    # Get the LAPACK prefix and the corresponding dtype if not fall back
    # to 'd' and double precision float.
    prefix, dtype = _type_conv.get(max_score, ('d', np.dtype('float64')))

    return prefix, dtype, prefer_fortran


# Level-1 BLAS functions with pure-rsnumpy fallbacks, used when the native
# fblas module has not yet ported them (called array-first, e.g. nrm2(a)).
def _py_nrm2(x, *args):
    a = np.asarray(x)
    if a.dtype.char in 'FC':
        a = np.abs(a)
    return float(np.sqrt(np.sum(a * a)))


def _py_dot(x, y, *args):
    return float(np.sum(np.asarray(x) * np.asarray(y)))


def _py_asum(x, *args):
    a = np.asarray(x)
    if a.dtype.char in 'FC':
        a = np.abs(a)
    return float(np.sum(a))


def _py_jacobi_eigh(M, compute_v=True):
    """Real symmetric eigen-decomposition via cyclic Jacobi rotations.

    M : list of lists (n x n, symmetric)
    Returns (w, V): w = eigenvalues ascending, V = row-major n x n list
    whose columns are the matching eigenvectors (None if not compute_v).
    """
    import math
    n = len(M)
    A = [row[:] for row in M]
    V = ([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
         if compute_v else None)
    for _sweep in range(300):
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                scale = max(abs(A[p][p]), abs(A[q][q]), 1e-300)
                if abs(apq) <= 1e-17 * scale:
                    continue
                app, aqq = A[p][p], A[q][q]
                theta = (aqq - app) / (2.0 * apq)
                t = ((1.0 if theta >= 0.0 else -1.0)
                     / (abs(theta) + math.hypot(theta, 1.0)))
                c = 1.0 / math.hypot(t, 1.0)
                s = t * c
                for k in range(n):
                    if k != p and k != q:
                        akp, akq = A[k][p], A[k][q]
                        A[k][p] = A[p][k] = c * akp - s * akq
                        A[k][q] = A[q][k] = s * akp + c * akq
                A[p][p] = app - t * apq
                A[q][q] = aqq + t * apq
                A[p][q] = A[q][p] = 0.0
                if compute_v:
                    for k in range(n):
                        vkp, vkq = V[k][p], V[k][q]
                        V[k][p] = c * vkp - s * vkq
                        V[k][q] = s * vkp + c * vkq
        off = math.sqrt(sum(A[i][j] ** 2
                            for i in range(n) for j in range(n) if i != j))
        if off < 1e-14:
            break
    diag = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=diag.__getitem__)
    w = [diag[i] for i in order]
    if not compute_v:
        return w, None
    v = [[V[i][order[j]] for j in range(n)] for i in range(n)]
    return w, v


def _py_sbevd(ab, compute_v=True, lower=True, overwrite_ab=False, **kwargs):
    """Pure-Python stand-in for ?sbevd (symmetric banded eigenproblem).

    Accepts (n, n) full, (2, n) port-convention, or (n, 2) LAPACK band
    layouts; returns the scipy-fLAPACK style (w, v, info) triple.
    """
    ab = np.asarray(ab, dtype=np.float64)
    sh = ab.shape
    if len(sh) == 2 and sh[0] == sh[1]:
        n = sh[0]
        M = [[float(ab[i, j]) for j in range(n)] for i in range(n)]
    elif len(sh) == 2 and sh[0] == 2:
        n = sh[1]
        if lower:
            # scipy lower band: a_band[i-j, j] = a[i,j]  ->  row 0 = diagonal,
            # row 1 = first subdiagonal (a[i+1, i] at column i).
            d = [float(ab[0, i]) for i in range(n)]
            e = [float(ab[1, i]) for i in range(n - 1)]
        else:
            # scipy upper band (u=1): a_band[1+i-j, j] = a[i,j]  ->  row 1 =
            # diagonal, row 0 = first superdiagonal (a[i, i+1] at column i+1).
            d = [float(ab[1, i]) for i in range(n)]
            e = [float(ab[0, i]) for i in range(1, n)]
        M = []
        for i in range(n):
            row = [0.0] * n
            row[i] = d[i]
            if i > 0:
                row[i - 1] = e[i - 1]
            if i < n - 1:
                row[i + 1] = e[i]
            M.append(row)
    else:  # (n, 2) LAPACK layout
        n = sh[0]
        d = [float(ab[i, i % 2]) for i in range(n)]
        e = ([float(ab[i + 1, i]) for i in range(n - 1)] if lower
             else [float(ab[i, i + 1]) for i in range(n - 1)])
        M = []
        for i in range(n):
            row = [0.0] * n
            row[i] = d[i]
            if i > 0:
                row[i - 1] = e[i - 1]
            if i < n - 1:
                row[i + 1] = e[i]
            M.append(row)
    w, v = _py_jacobi_eigh(M, compute_v=compute_v)
    if compute_v:
        return np.array(w), np.array(v), 0
    return np.array(w), None, 0


def _py_trtrs(a, b, overwrite_b=False, lower=False, trans=0,
              unitdiag=False, left_side=True, **kwargs):
    """Pure-Python triangular solve (LAPACK ?trtrs semantics).

    Solves op(a) x = b with op(a) = a (trans=0) or a.T (trans in (1, 2)),
    using the triangle selected by ``lower`` and a unit diagonal when
    ``unitdiag``.  Returns (x, info): info is 0 on success or i+1 when the
    i-th diagonal element is exactly zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    if trans in (1, 2):
        M = [[float(a[j, i]) for j in range(n)] for i in range(n)]
        eff_lower = not lower
    else:
        M = [[float(a[i, j]) for j in range(n)] for i in range(n)]
        eff_lower = lower

    def solve_one(rhs):
        x = [0.0] * n
        idx = range(n) if eff_lower else range(n - 1, -1, -1)
        for i in idx:
            s = rhs[i]
            rng = range(i) if eff_lower else range(i + 1, n)
            for j in rng:
                s -= M[i][j] * x[j]
            d = 1.0 if unitdiag else M[i][i]
            if d == 0.0:
                return None, i + 1
            x[i] = s / d
        return x, 0

    if b.ndim == 1:
        x, info = solve_one([float(b[i]) for i in range(n)])
        return (np.array(x) if x is not None else None), info
    k = b.shape[1]
    cols = []
    for jcol in range(k):
        x, info = solve_one([float(b[i, jcol]) for i in range(n)])
        if info:
            return None, info
        cols.append(x)
    out = [[cols[jcol][i] for jcol in range(k)] for i in range(n)]
    return np.array(out), 0


_blas_py_fallbacks = {'nrm2': _py_nrm2, 'dot': _py_dot, 'asum': _py_asum,
                      'sbevd': _py_sbevd, 'trtrs': _py_trtrs}


def _get_funcs(
    names, arrays, dtype, lib_name, fmodule, fmodule_name, alias, ilp64="preferred"
):
    """
    Return available BLAS/LAPACK functions.

    Used also in lapack.py. See get_blas_funcs for docstring.
    """

    funcs = []
    unpack = False
    dtype = np.dtype(dtype)

    if isinstance(names, str):
        names = (names,)
        unpack = True

    prefix, dtype, _ = find_best_blas_type(arrays, dtype)

    for name in names:
        func_name = prefix + name
        func_name = alias.get(func_name, func_name)
        func = getattr(fmodule, func_name, None)
        if func is None:
            # Pure-rsnumpy fallback for level-1 functions not yet ported
            func = _blas_py_fallbacks.get(name)
        if func is None:
            # Stub for BLAS functions not yet ported to Rust/rsnumpy
            def _blas_stub(*args, **kwargs):
                raise NotImplementedError(
                    f"BLAS function {func_name} not yet ported to Rust/rsnumpy")
            _blas_stub.__name__ = func_name
            func = _blas_stub
        func.module_name, func.typecode, func.dtype = fmodule_name, prefix, dtype
        if not ilp64:
            func.int_dtype = np.dtype(np.intc)
        else:
            func.int_dtype = np.dtype(np.int64)
        func.prefix = prefix  # Backward compatibility
        funcs.append(func)

    if unpack:
        return funcs[0]
    else:
        return funcs


def _memoize_get_funcs(func):
    """
    Memoized fast path for _get_funcs instances
    """
    memo = {}
    func.memo = memo

    @functools.wraps(func)
    def getter(names, arrays=(), dtype=None, ilp64="preferred"):
        key = (names, dtype, ilp64)
        for array in arrays:
            # cf. find_blas_funcs
            key += (array.dtype.char, array.flags.fortran)

        try:
            value = memo.get(key)
        except TypeError:
            # unhashable key etc.
            key = None
            value = None

        if value is not None:
            return value

        value = func(names, arrays, dtype, ilp64)

        if key is not None:
            memo[key] = value

        return value

    return getter


@_memoize_get_funcs
def get_blas_funcs(names, arrays=(), dtype=None, ilp64="preferred"):
    """Return available BLAS function objects from names.

    Arrays are used to determine the optimal prefix of BLAS routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of BLAS functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    ilp64 : {True, False, 'preferred'}, optional
        Whether to return ILP64 routine variant.
        Choosing ``'preferred'`` returns ILP64 routine if available,
        and otherwise the 32-bit (LP64) routine. Default: ``'preferred'``.

    Returns
    -------
    funcs : list
        List containing the found function(s).

    Raises
    ------
    RuntimeError
        If the requested LP64/ILP64 variant is not available.

    See Also
    --------
    scipy.linalg.lapack.get_lapack_funcs
        a similar routine for selecting LAPACK functions.

    Notes
    -----
    In BLAS, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of ``{'s', 'd', 'c', 'z'}`` for the NumPy
    types ``{float32, float64, complex64, complex128}`` respectively.
    The code and the dtype are stored in attributes `typecode` and `dtype`
    of the returned functions.

    Examples
    --------
    >>> import rsnumpy as np
    >>> import scicore.linalg as LA
    >>> rng = np.random.default_rng()
    >>> a = rng.random((3, 2))
    >>> x_gemv = LA.get_blas_funcs('gemv', (a,))

    Note that ``x_gemv`` string representation shows the exact BLAS function with the
    prefix (here, ``d-`` because ``a`` is double precision real):

    >>> x_gemv
    <fblas function dgemv>

    The BLAS variant information is also available from the ``typecode`` attribute:

    >>> x_gemv.typecode
    'd'

    For double precision complex arrays, we select the ``z-`` variant, ``zgemv``

    >>> x_gemv = LA.get_blas_funcs('gemv', (a*1j,))
    >>> x_gemv.typecode
    'z'

    If you want to select a specific BLAS variant instead of relying on array types, use
    the ``dtype=`` argument:

    >>> LA.get_blas_funcs('gemv', dtype=np.float32)
    <fblas function sgemv>

    The ``int_dtype`` attribute stores whether the routine is ILP64 (integer arguments
    and outputs are 64-bit) or LP64 (integer arguments and outputs are 32-bit):

    >>> x_gemv.int_dtype
    dtype('int32')   # may vary
    """
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError(f"Invalid value for {ilp64 = }.")

    if not ilp64:
        return _get_funcs(
            names, arrays, dtype, "BLAS", _fblas, "fblas", _blas_alias, ilp64=False
        )
    else:
        if not HAS_ILP64:
            raise RuntimeError("BLAS ILP64 routine requested, but Scipy "
                               "compiled only with 32-bit BLAS")
        return _get_funcs(
            names, arrays, dtype, "BLAS", _fblas_64, "fblas_64", _blas_alias, ilp64=True
        )


# ---------------------------------------------------------------------------
# Module-level BLAS stubs for functions imported directly (not via
# get_blas_funcs).  These raise NotImplementedError when called until real
# Rust BLAS kernels are available.
# ---------------------------------------------------------------------------

def _make_blas_stub(name):
    def stub(*args, **kwargs):
        raise NotImplementedError(
            f"BLAS function {name} not yet ported to Rust/rsnumpy")
    stub.__name__ = name
    return stub


for _blas_name in ('dtrsm', 'strsm', 'ctrsm', 'ztrsm',
                   'dgemm', 'sgemm', 'zgemm', 'cgemm',
                   'dsymm', 'ssymm', 'zsymm', 'csymm',
                   'dtrmm', 'strmm', 'ctrmm', 'ztrmm',
                   'dsyrk', 'ssyrk', 'zsyrk', 'csyrk',
                   'dot', 'sdot', 'ddot', 'cdotu', 'zdotu',
                   'axpy', 'saxpy', 'daxpy', 'caxpy', 'zaxpy',
                   'nrm2', 'snrm2', 'dnrm2', 'scnrm2', 'dznrm2',
                   'asum', 'sasum', 'dasum', 'scasum', 'dzasum',
                   'iamax', 'isamax', 'idamax', 'icamax', 'izamax',
                   'rot', 'srot', 'drot', 'crot', 'zrot',
                   'swap', 'sswap', 'dswap', 'cswap', 'zswap',
                   'scal', 'sscal', 'dscal', 'cscal', 'zscal',
                   'copy', 'scopy', 'dcopy', 'ccopy', 'zcopy'):
    _blas_base = _blas_name[1:] if _blas_name[:1] in ('s', 'd', 'c', 'z') else _blas_name
    if _blas_base in _blas_py_fallbacks:
        globals()[_blas_name] = _blas_py_fallbacks[_blas_base]
    else:
        globals()[_blas_name] = _make_blas_stub(_blas_name)

