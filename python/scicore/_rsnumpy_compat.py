"""Compatibility layer between :mod:`rsnumpy` and the NumPy surface SciPy uses.

The SciPy source tree has been ported from ``numpy`` to ``rsnumpy``.  rsnumpy
1.2.4 covers the vast majority of the NumPy 2.x Python API, but not everything
that SciPy relies on.  This module fills the remaining gaps and registers the
results on the :mod:`rsnumpy` package so that plain

    from rsnumpy.testing import assert_allclose
    import rsnumpy.testing as npt

work exactly like they did with NumPy.

Gaps handled here
-----------------
* ``rsnumpy.testing`` -- not shipped by rsnumpy at all, yet imported by ~800
  SciPy modules (``assert_allclose``, ``assert_equal``, ``assert_``, ...).
* ``rsnumpy.ufunc`` / ``rsnumpy.fromfile`` / ``rsnumpy.memmap`` /
  ``rsnumpy.float128`` / ``rsnumpy.float96`` / ``rsnumpy.matrix_power``.
* Arrays produced by NumPy-based C extensions are not instances of
  ``rsnumpy.ndarray``; the ``asarray`` family below converts them instead of
  raising ``TypeError``.

This module must be imported before any other SciPy module; ``scipy/__init__``
and ``scipy/conftest.py`` both do so.
"""

import sys
import types
import unittest
import warnings

import rsnumpy

__all__ = ['install']


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# NumPy's `dtype.char` alphabet, see numpy.dtype.char
_TYPE_CHAR = {
    'bool': '?',
    'int8': 'b', 'uint8': 'B',
    'int16': 'h', 'uint16': 'H',
    'int32': 'i', 'uint32': 'I',
    'int64': 'l', 'uint64': 'L',
    'float16': 'e', 'float32': 'f',
    'float64': 'd', 'float128': 'g',
    'complex64': 'F', 'complex128': 'D', 'complex256': 'G',
}

# Reverse of `_TYPE_CHAR`, for callers that pass a character to a NumPy API.
_TYPE_NAME = {char: name for name, char in _TYPE_CHAR.items()}

# `a.flags[<key>]` lookup table, given NumPy's spelling.
_FLAG_ALIASES = {
    'C': lambda f: f.C_CONTIGUOUS,
    'C_CONTIGUOUS': lambda f: f.C_CONTIGUOUS,
    'CONTIGUOUS': lambda f: f.C_CONTIGUOUS,
    'F': lambda f: f.F_CONTIGUOUS,
    'F_CONTIGUOUS': lambda f: f.F_CONTIGUOUS,
    'FORTRAN': lambda f: f.F_CONTIGUOUS,
    'FORTRAN_CONTIGUOUS': lambda f: f.F_CONTIGUOUS,
    'W': lambda f: f.WRITEABLE,
    'WRITEABLE': lambda f: f.WRITEABLE,
    'A': lambda f: f.ALIGNED,
    'ALIGNED': lambda f: f.ALIGNED,
    'O': lambda f: f.OWNDATA,
    'OWNDATA': lambda f: f.OWNDATA,
    'U': lambda f: f.UPDATEIFCOPY,
    'UPDATEIFCOPY': lambda f: f.UPDATEIFCOPY,
    'B': lambda f: f.ALIGNED and f.WRITEABLE,
    'BEHAVED': lambda f: f.ALIGNED and f.WRITEABLE,
}

# the pristine constructor - `install()` must not wrap it, otherwise `_adopt`
# would recurse through the wrapper.
_RS_ARRAY = rsnumpy.array


def _is_foreign_array(obj):
    """True for array-likes that rsnumpy refuses to adopt (e.g. NumPy arrays)."""
    if isinstance(obj, rsnumpy.ndarray):
        return False
    return hasattr(obj, '__array_interface__') or hasattr(obj, 'tolist')


def _adopt(obj):
    """Convert a foreign array-like into an ``rsnumpy.ndarray``."""
    try:
        return _RS_ARRAY(obj)
    except Exception:
        pass
    try:
        return _RS_ARRAY(obj.tolist())
    except Exception:
        pass
    return obj


def _to_rs(x):
    return _adopt(x) if _is_foreign_array(x) else x


def _as(x):
    """``rsnumpy.asarray`` that also accepts foreign (NumPy) arrays."""
    return _to_rs(rsnumpy.asarray(_to_rs(x)))


# ---------------------------------------------------------------------------
# rsnumpy.testing
# ---------------------------------------------------------------------------

def _shape_msg(actual, desired, header):
    return (f"\n{header}\n(shapes {actual.shape}, {desired.shape} mismatch)"
            f"\n ACTUAL: {actual}\n DESIRED: {desired}")


def _mismatch_msg(actual, desired, mask, err_msg, header):
    try:
        n_bad = int(rsnumpy.count_nonzero(mask))
        n_tot = int(rsnumpy.size(mask))
        pct = 100.0 * n_bad / n_tot if n_tot else 0.0
        with rsnumpy.errstate(all='ignore'):
            absdiff = rsnumpy.abs(_diff(actual, desired))
        max_abs = float(rsnumpy.nanmax(absdiff)) if absdiff.size else 0.0
        with rsnumpy.errstate(all='ignore'):
            denom = rsnumpy.abs(desired)
            denom = rsnumpy.where(denom == 0, 1, denom)
            reldiff = absdiff / denom
        max_rel = float(rsnumpy.nanmax(reldiff)) if reldiff.size else 0.0
        extra = (f"\nMismatch: {pct:.1f}%"
                 f"\nMax absolute difference: {max_abs:g}"
                 f"\nMax relative difference: {max_rel:g}")
    except Exception:  # pragma: no cover - message formatting is best effort
        extra = ""
    return (f"\n{header}{extra}"
            f"\n ACTUAL: {actual}\n DESIRED: {desired}"
            f"{': ' + err_msg if err_msg else ''}")


def _diff(a, b):
    try:
        return a - b
    except Exception:
        return rsnumpy.asarray(a, dtype=rsnumpy.float64) - \
            rsnumpy.asarray(b, dtype=rsnumpy.float64)


def _nan_safe_isnan(x):
    try:
        return rsnumpy.isnan(x)
    except Exception:
        return rsnumpy.zeros(x.shape, dtype=bool)


def _broadcast_pair(actual, desired):
    """按 NumPy 语义广播两组值；不可广播时返回 ``None``。

    NumPy 的比较断言是先广播再逐元素比，因此
    ``assert_allclose(a[mask], 0.0)`` 这类“数组 vs 标量”的写法完全合法。
    """
    if actual.shape == desired.shape:
        return actual, desired
    try:
        return rsnumpy.broadcast_arrays(actual, desired)
    except Exception:
        return None


def assert_allclose(actual, desired, rtol=1e-07, atol=0,
                    equal_nan=True, err_msg='', verbose=True):
    """Like :func:`numpy.testing.assert_allclose`."""
    actual = _as(actual)
    desired = _as(desired)
    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"
    if actual.shape != desired.shape:
        broadcast = _broadcast_pair(actual, desired)
        if broadcast is None:
            raise AssertionError(_shape_msg(actual, desired, header))
        actual, desired = broadcast
    if actual.size == 0:
        return
    with rsnumpy.errstate(invalid='ignore', divide='ignore'):
        diff = rsnumpy.abs(_diff(actual, desired))
        tol = atol + rtol * rsnumpy.abs(desired)
        close = diff <= tol
        if equal_nan:
            close = close | (_nan_safe_isnan(actual) & _nan_safe_isnan(desired))
    if not bool(rsnumpy.all(close)):
        raise AssertionError(
            _mismatch_msg(actual, desired, ~close, err_msg, header))


def assert_array_equal(actual, desired, err_msg='', verbose=True, strict=False):
    """Like :func:`numpy.testing.assert_array_equal` (NaNs compare equal)."""
    actual = _as(actual)
    desired = _as(desired)
    header = "Arrays are not equal"
    if actual.shape != desired.shape:
        broadcast = _broadcast_pair(actual, desired)
        if broadcast is None:
            raise AssertionError(_shape_msg(actual, desired, header))
        actual, desired = broadcast
    with rsnumpy.errstate(invalid='ignore'):
        eq = actual == desired
    if not strict:
        try:
            eq = eq | (_nan_safe_isnan(actual) & _nan_safe_isnan(desired))
        except Exception:
            pass
    if not bool(rsnumpy.all(eq)):
        raise AssertionError(
            _mismatch_msg(actual, desired, ~eq, err_msg, header))


def assert_equal(actual, desired, err_msg='', verbose=True):
    """Like :func:`numpy.testing.assert_equal`."""
    _assert_equal_rec(actual, desired, err_msg)


def _assert_equal_rec(actual, desired, err_msg, path=''):
    if isinstance(actual, dict) and isinstance(desired, dict):
        if set(actual) != set(desired):
            raise AssertionError(
                f"{err_msg}\nKeys are not equal at {path or 'top level'}")
        for k in actual:
            _assert_equal_rec(actual[k], desired[k], err_msg, f'{path}[{k!r}]')
        return
    if isinstance(actual, (list, tuple)) and isinstance(desired, (list, tuple)):
        if type(actual) is not type(desired) and not (
                isinstance(actual, tuple) and isinstance(desired, tuple)):
            raise AssertionError(f"{err_msg}\nTypes differ at {path}")
        if len(actual) != len(desired):
            raise AssertionError(
                f"{err_msg}\nLengths differ at {path}: "
                f"{len(actual)} != {len(desired)}")
        for i, (a, d) in enumerate(zip(actual, desired)):
            _assert_equal_rec(a, d, err_msg, f'{path}[{i}]')
        return
    if isinstance(actual, str) or isinstance(desired, str):
        if actual != desired:
            raise AssertionError(
                f"{err_msg}\nItems are not equal at {path}:\n"
                f" ACTUAL: {actual!r}\n DESIRED: {desired!r}")
        return

    try:
        a = _as(actual)
        d = _as(desired)
        if a.shape != d.shape:
            raise AssertionError(
                f"{err_msg}\nShapes differ at {path}: {a.shape} != {d.shape}")
        with rsnumpy.errstate(invalid='ignore'):
            eq = a == d
        eq = eq | (_nan_safe_isnan(a) & _nan_safe_isnan(d))
        ok = bool(rsnumpy.all(eq))
    except AssertionError:
        raise
    except Exception:
        ok = actual == desired

    if not ok:
        raise AssertionError(
            f"{err_msg}\nItems are not equal at {path}:\n"
            f" ACTUAL: {actual}\n DESIRED: {desired}")


def assert_(val, msg=None):
    """Like :func:`numpy.testing.assert_`."""
    if not val:
        raise AssertionError(msg or 'Condition is not true')


def assert_array_almost_equal(actual, desired, decimal=6, err_msg='',
                              verbose=True):
    """Like :func:`numpy.testing.assert_array_almost_equal`."""
    actual = _as(actual)
    desired = _as(desired)
    header = f"Arrays are not almost equal to {decimal} decimals"
    if actual.shape != desired.shape:
        raise AssertionError(_shape_msg(actual, desired, header))
    if actual.size == 0:
        return
    with rsnumpy.errstate(invalid='ignore'):
        close = rsnumpy.abs(_diff(actual, desired)) < 1.5 * 10.0 ** (-decimal)
    if not bool(rsnumpy.all(close)):
        raise AssertionError(
            _mismatch_msg(actual, desired, ~close, err_msg, header))


def assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True):
    """Like :func:`numpy.testing.assert_almost_equal`."""
    header = f"Items are not almost equal to {decimal} decimals"
    try:
        a = float(actual)
        d = float(desired)
        if a == d:
            return
        if not abs(a - d) < 1.5 * 10.0 ** (-decimal):
            raise AssertionError(
                f"\n{header}\n ACTUAL: {a}\n DESIRED: {d}{err_msg}")
    except AssertionError:
        raise
    except (TypeError, ValueError):
        assert_array_almost_equal(actual, desired, decimal, err_msg)


def assert_array_less(x, y, err_msg='', verbose=True, strict=False):
    """Like :func:`numpy.testing.assert_array_less`."""
    x = _as(x)
    y = _as(y)
    header = "Arrays are not less-ordered"
    if x.shape != y.shape:
        raise AssertionError(_shape_msg(x, y, header))
    if x.size == 0:
        return
    with rsnumpy.errstate(invalid='ignore'):
        ok = x < y
    if not bool(rsnumpy.all(ok)):
        raise AssertionError(_mismatch_msg(x, y, ~ok, err_msg, header))


def assert_array_almost_equal_nulp(x, y, nulp=1):
    """Like :func:`numpy.testing.assert_array_almost_equal_nulp`."""
    x = _as(x)
    y = _as(y)
    if x.shape != y.shape:
        raise AssertionError(f"Shapes differ: {x.shape} != {y.shape}")
    if x.size == 0:
        return
    with rsnumpy.errstate(invalid='ignore'):
        spacing = rsnumpy.spacing(rsnumpy.maximum(rsnumpy.abs(x),
                                                  rsnumpy.abs(y)))
        ok = rsnumpy.abs(x - y) <= nulp * spacing
        ok = ok | ((x == y) & _nan_safe_isnan(x))
    if not bool(rsnumpy.all(ok)):
        raise AssertionError(
            f"\nArrays are not equal to {nulp} ULP\n ACTUAL: {x}\n DESIRED: {y}")


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """Like :func:`numpy.testing.assert_array_max_ulp`."""
    a = _as(a if dtype is None else rsnumpy.asarray(a, dtype=dtype))
    b = _as(b if dtype is None else rsnumpy.asarray(b, dtype=dtype))
    if a.shape != b.shape:
        raise AssertionError(f"Shapes differ: {a.shape} != {b.shape}")
    with rsnumpy.errstate(invalid='ignore'):
        spacing = rsnumpy.spacing(rsnumpy.maximum(rsnumpy.abs(a),
                                                  rsnumpy.abs(b)))
        diff = rsnumpy.abs(a - b) / spacing
    maxdiff = float(rsnumpy.nanmax(diff)) if diff.size else 0.0
    if maxdiff > maxulp:
        raise AssertionError(
            f"\nMax ULP difference {maxdiff:g} > allowed {maxulp:g}")
    return maxdiff


def assert_approx_equal(actual, desired, significant=7, err_msg='',
                        verbose=True):
    """Like :func:`numpy.testing.assert_approx_equal`."""
    actual = float(actual)
    desired = float(desired)
    if actual == desired:
        return
    magnitude = max(abs(actual), abs(desired))
    err = abs(actual - desired) if magnitude == 0 else \
        abs(actual - desired) / magnitude
    if err >= 10.0 ** (-significant):
        raise AssertionError(
            f"\nItems are not equal to {significant} significant digits:\n"
            f" ACTUAL: {actual}\n DESIRED: {desired}{err_msg}")


def assert_string_equal(actual, desired):
    """Like :func:`numpy.testing.assert_string_equal`."""
    if not isinstance(actual, str):
        raise AssertionError(
            f"`actual` is not a string, but {type(actual).__name__}")
    if not isinstance(desired, str):
        raise AssertionError(
            f"`desired` is not a string, but {type(desired).__name__}")
    if actual != desired:
        raise AssertionError(f"\n ACTUAL: {actual!r}\n DESIRED: {desired!r}")


def assert_no_warnings(func, *args, **kwargs):
    """Like :func:`numpy.testing.assert_no_warnings`."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        func(*args, **kwargs)
    if record:
        raise AssertionError(
            "No warnings allowed, but produced: "
            f"{[str(w.message) for w in record]}")


def assert_raises(exception_class, callable_obj=None, *args, **kwargs):
    """Like :func:`numpy.testing.assert_raises`."""
    if callable_obj is None:
        return _AssertRaisesContext(exception_class)
    try:
        callable_obj(*args, **kwargs)
    except exception_class:
        return
    raise AssertionError(f"{exception_class.__name__} not raised")


class _AssertRaisesContext:
    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise AssertionError(f"{self.expected.__name__} not raised")
        return issubclass(exc_type, self.expected)


def assert_warns(warning_class, func=None, *args, **kwargs):
    """Like :func:`numpy.testing.assert_warns`."""
    if func is None:
        return warnings.catch_warnings(),  # placeholder, unused by SciPy
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        func(*args, **kwargs)
    if not any(isinstance(w.message, warning_class) for w in record):
        raise AssertionError(f"{warning_class.__name__} not triggered")
    return record


def _make_testing_module():
    mod = types.ModuleType('rsnumpy.testing')
    mod.__doc__ = "NumPy-compatible testing helpers, implemented on rsnumpy."
    names = [
        'assert_', 'assert_allclose', 'assert_almost_equal', 'assert_approx_equal',
        'assert_array_almost_equal', 'assert_array_almost_equal_nulp',
        'assert_array_equal', 'assert_array_less', 'assert_array_max_ulp',
        'assert_equal', 'assert_no_warnings', 'assert_raises',
        'assert_string_equal', 'assert_warns',
    ]
    g = globals()
    for name in names:
        setattr(mod, name, g[name])
    mod.TestCase = unittest.TestCase
    mod.HAS_REFCOUNT = hasattr(sys, 'getrefcount')
    mod.assert_raises_regex = unittest.TestCase('run').assertRaisesRegex
    return mod


# ---------------------------------------------------------------------------
# top-level attributes rsnumpy does not define
# ---------------------------------------------------------------------------

class ufunc:  # noqa: N801 - mirrors numpy.ufunc
    """Placeholder mirroring :class:`numpy.ufunc`."""

    nin = 1
    nout = 1
    nargs = 2
    ntypes = 0
    identity = None
    signature = None

    def resolve_dtypes(self, dtypes=None, *args, **kwargs):
        """Find the dtypes that the ufunc would use for the given inputs."""
        raise NotImplementedError(
            "rsnumpy does not expose a NumPy ufunc C-API object")


def fromfile(file, dtype=float, count=-1, sep='', offset=0, *, like=None):
    """Like :func:`numpy.fromfile`."""
    import io
    if isinstance(file, (str, bytes)) or hasattr(file, '__fspath__'):
        with open(file, 'rb') as fh:
            data = fh.read()
    elif hasattr(file, 'read'):
        data = file.read()
    else:  # pragma: no cover
        raise TypeError(f"Unsupported file object: {type(file).__name__}")
    data = data[offset:]
    if sep == '':
        values = rsnumpy.frombuffer(bytearray(data), dtype=dtype, count=count)
    else:
        text = data.decode('latin-1').split(sep)
        values = rsnumpy.array([float(t) for t in text], dtype=dtype)
        if count >= 0:
            values = values[:count]
    return values


def matrix_power(a, n):
    """Like :func:`numpy.linalg.matrix_power`."""
    return rsnumpy.linalg.matrix_power(a, n)


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------

def install():
    """Register the compatibility objects on the :mod:`rsnumpy` package."""
    if getattr(rsnumpy, '_scipy_compat_installed', False):
        return rsnumpy

    testing = _make_testing_module()
    sys.modules.setdefault('rsnumpy.testing', testing)
    rsnumpy.testing = testing

    for name, obj in [('ufunc', ufunc), ('fromfile', fromfile),
                      ('memmap', getattr(rsnumpy, 'memmap', None)),
                      ('matrix_power', matrix_power)]:
        if obj is not None and not hasattr(rsnumpy, name):
            setattr(rsnumpy, name, obj)

    # 128-bit float aliases (rsnumpy uses `longdouble` internally)
    ld = getattr(rsnumpy, 'longdouble', rsnumpy.float64)
    for name in ('float96', 'float128'):
        if not hasattr(rsnumpy, name):
            setattr(rsnumpy, name, ld)
    for name in ('complex192', 'complex256'):
        if not hasattr(rsnumpy, name):
            setattr(rsnumpy, name, getattr(rsnumpy, 'clongdouble', None) or
                    rsnumpy.complex128)

    # make `asarray`-family accept arrays produced by NumPy-based extensions
    for name in ('asarray', 'asanyarray', 'ascontiguousarray',
                 'asfortranarray', 'asarray_chkfinite', 'atleast_1d',
                 'atleast_2d', 'atleast_3d', 'copyto', 'broadcast_to'):
        original = getattr(rsnumpy, name, None)
        if original is None or getattr(original, '_scipy_compat', False):
            continue
        setattr(rsnumpy, name, _make_adopting(original))

    # Incomplete object APIs that SciPy leans on but rsnumpy never shipped.
    _patch_dtype_and_flags()

    # rsnumpy 1.2.4 bugs that SciPy hits everywhere
    rsnumpy.allclose = _fixed_allclose()
    rsnumpy.where = _fixed_where(rsnumpy.where)

    # rsnumpy's 2-D+ boolean-mask getitem crashes with "0 is not a valid
    # dimension size" (internal reshape of the empty selection) when the mask
    # selects zero rows; return a correctly-shaped empty array instead.
    _patch_empty_bool_getitem()

    # rsnumpy's `pad` is broken (it misreads `pad_width` tuples, inserting
    # them as values or silently dropping entries); replace it with a
    # correct pure-Python implementation.
    rsnumpy.pad = _fixed_pad()

    # rsnumpy ships no lib.stride_tricks.sliding_window_view (no stride
    # tricks at all); SciPy's spectral and ndimage helpers rely on it.
    _stride = getattr(getattr(rsnumpy, 'lib', None), 'stride_tricks', None)
    if _stride is not None and not hasattr(_stride, 'sliding_window_view'):
        _stride.sliding_window_view = _sliding_window_view

    # Make rsnumpy.ndarray hashable (needed for use as dict keys).
    # rsnumpy defines __eq__ which sets __hash__ = None; override it.
    if rsnumpy.ndarray.__hash__ is None:
        def _ndarray_hash(self):
            try:
                flat = tuple(self.tolist())
            except TypeError:
                flat = (float(self),)
            return hash((self.shape, flat))
        rsnumpy.ndarray.__hash__ = _ndarray_hash

    rsnumpy._scipy_compat_installed = True
    return rsnumpy


def _sliding_window_view(x, window_shape, axis=None, writeable=None):
    """Pure rsnumpy replacement for ``lib.stride_tricks.sliding_window_view``.

    rsnumpy has no stride tricks, so the windowed array is built with basic
    slicing only: 1-D inputs are assembled through a list of lists (which
    ``asarray`` handles), higher-dimensional inputs through a zeros
    allocation filled row by row.
    """
    x = rsnumpy.asarray(x)
    if axis is None:
        axis = x.ndim - 1
    axis = axis % x.ndim
    w = int(window_shape)
    n = x.shape[axis]
    if n < w:
        raise ValueError(f"window shape {w} exceeds dimension size {n}")
    m = n - w + 1
    if x.ndim == 1:
        return rsnumpy.array([x[i:i + w].tolist() for i in range(m)])
    out_shape = x.shape[:axis] + (m, w) + x.shape[axis + 1:]
    out = rsnumpy.zeros(out_shape)
    for i in range(m):
        in_idx = [slice(None)] * x.ndim
        in_idx[axis] = slice(i, i + w)
        out_idx = [slice(None)] * x.ndim
        out_idx.insert(axis, i)
        out[tuple(out_idx)] = x[tuple(in_idx)]
    return out


def _fixed_pad():
    """Replace rsnumpy's broken `pad` with a correct pure-Python version.

    rsnumpy's implementation misreads the `pad_width` tuples (it inserts
    the tuples as values and silently drops 2-tuple entries).  SciPy's
    signal code pads only the last axis, which this covers: leading pad
    widths must be (0, 0).  Negative widths crop, as in NumPy.
    """
    def pad(array, pad_width, mode='constant', **kwargs):
        a = rsnumpy.asarray(array)
        if (a.ndim == 1 and len(pad_width) == 2
                and all(isinstance(v, (int,)) for v in pad_width)):
            widths = [tuple(pad_width)]
        else:
            widths = [tuple(w) for w in pad_width]
        for w in widths[:-1]:
            if any(v != 0 for v in w):
                raise NotImplementedError(
                    "compat pad only supports last-axis padding")
        left, right = widths[-1]

        if mode == 'constant':
            cv = kwargs.get('constant_values', 0)
            if hasattr(cv, '__len__'):
                lval, rval = float(cv[0]), float(cv[1])
            else:
                lval = rval = float(cv)
        elif mode in ('edge', 'wrap', 'reflect', 'symmetric'):
            lval = rval = 0.0
        else:
            raise NotImplementedError(f"compat pad: unsupported mode {mode!r}")
        reflect_type = kwargs.get('reflect_type', 'odd')

        def pad_left(row, n):
            p = left
            if p <= 0 or n == 0:
                return []
            if mode == 'constant':
                return [lval] * p
            if mode == 'edge':
                return [row[0]] * p
            if mode == 'wrap':
                return [row[(-i) % n] for i in range(p, 0, -1)]
            # 'reflect' (reflect_type) / 'symmetric' (even): bounce the
            # indices; even keeps the edge element, odd skips it.
            even = mode == 'symmetric' or reflect_type == 'even'
            lo, hi = (0, n - 1) if even else (1, n - 1)
            if lo == hi:
                return [row[lo]] * p
            seg = list(range(lo, hi + 1)) + list(range(hi - 1, lo, -1))
            seq = [seg[k % len(seg)] for k in range(p)]  # right-to-left
            return [row[i] for i in reversed(seq)]

        def pad_right(row, n):
            p = right
            if p <= 0 or n == 0:
                return []
            if mode == 'constant':
                return [rval] * p
            if mode == 'edge':
                return [row[-1]] * p
            if mode == 'wrap':
                return [row[i % n] for i in range(p)]
            even = mode == 'symmetric' or reflect_type == 'even'
            if n <= 1:
                return [row[0]] * p
            lo, hi = (0, n - 1) if even else (1, n - 1)
            if lo >= hi:
                return [row[hi if hi >= 0 else 0]] * p
            # mirror sequence starting closest to the right edge
            if even:
                seg = list(range(hi, lo - 1, -1)) + list(range(lo + 1, hi + 1))
            else:
                seg = list(range(hi - 1, lo - 1, -1)) + list(range(lo + 1, hi))
            seq = [seg[k % len(seg)] for k in range(p)]
            return [row[i] for i in seq]

        if a.ndim == 1:
            row = a.tolist()
        else:
            row = None
        # crop negative widths first
        if left < 0:
            crop_l = -left
            left = 0
        else:
            crop_l = 0
        if right < 0:
            crop_r = -right
            right = 0
        else:
            crop_r = 0

        if row is not None:
            core = row[crop_l:len(row) - crop_r if crop_r else len(row)]
            out = pad_left(core, len(core)) + core + pad_right(core, len(core))
        else:
            out = []
            for r in a.tolist():
                core = r[crop_l:len(r) - crop_r if crop_r else len(r)]
                out.append(pad_left(core, len(core)) + core
                           + pad_right(core, len(core)))
        return rsnumpy.asarray(out)

    return pad


def _patch_empty_bool_getitem():
    """Fix `arr[bool_mask]` for masks that select zero rows on 2-D+ arrays.

    rsnumpy's boolean-mask getitem computes an internal reshape of the
    selected rows and raises `ValueError: 0 is not a valid dimension size`
    when the selection is empty.  The selected data is empty, so a
    correctly-shaped zero array is an exact answer.
    """
    cls = type(rsnumpy.asarray([0.0]))
    orig_getitem = cls.__getitem__

    def getitem(self, key):
        try:
            return orig_getitem(self, key)
        except (ValueError, OverflowError) as e:
            if "0 is not a valid dimension size" not in str(e):
                raise
            if (isinstance(key, cls) and key.ndim == 1
                    and not bool(rsnumpy.any(key))):
                return rsnumpy.zeros((0,) + tuple(self.shape[1:]),
                                     dtype=self.dtype)
            raise

    cls.__getitem__ = getitem


def _patch_dtype_and_flags():
    """Teach `rsnumpy` the two attributes `scipy.linalg` cannot do without.

    * ``dtype.char`` - the ``?bBhHiIlLqQefdgFDGO`` type character NumPy exposes
      and `scipy.linalg.blas.get_blas_funcs` uses for its cache key.
    * ``flags.fortran`` - NumPy's alias of ``F_CONTIGUOUS``, used by the same
      cache lookup.
    """
    sample = rsnumpy.asarray([0.0])

    dtype_cls = type(sample.dtype)
    if not hasattr(dtype_cls, 'char'):
        dtype_cls.char = property(lambda self: _TYPE_CHAR.get(str(self), 'd'))

    flags_cls = type(sample.flags)
    if not hasattr(flags_cls, 'fortran'):
        flags_cls.fortran = property(lambda self: self.F_CONTIGUOUS)

    # NumPy's `a.flags` is dict-like; rsnumpy only exposes attributes.
    if not hasattr(flags_cls, '__getitem__'):
        flags_cls.__getitem__ = lambda self, key: _FLAG_ALIASES[key.upper()](self)

    # `int8`/`int16`/`int32` arrays `.tolist()` to floats instead of ints
    # (`int64`/`uint8` are fine).  Anything that iterates a low-width integer
    # array -- a LAPACK pivot vector, say -- then gets floats it cannot index
    # with, so restore the integer here.
    array_cls = type(sample)
    # `np.dtype(None)` means float64 in NumPy; rsnumpy raises instead.
    original_dtype = rsnumpy.dtype

    def dtype(obj=None, *args, **kwargs):
        if obj is None:
            obj = rsnumpy.float64
        return original_dtype(obj, *args, **kwargs)

    dtype.__name__ = 'dtype'
    rsnumpy.dtype = dtype

    # `np.finfo('d')` -- NumPy accepts a type character here, rsnumpy demands a
    # real dtype object.  `scipy.linalg._matfuncs` uses this spelling for eps.
    original_finfo = rsnumpy.finfo

    def finfo(dtype):
        return original_finfo(_resolve_type_char(dtype))

    finfo.__name__ = 'finfo'
    rsnumpy.finfo = finfo

    array_cls = type(sample)
    original_tolist = array_cls.tolist

    def tolist(self):
        result = original_tolist(self)
        if getattr(self.dtype, 'kind', '') in 'iu':
            return _restore_ints(result)
        return result

    tolist.__name__ = 'tolist'
    array_cls.tolist = tolist


def _restore_ints(value):
    """Recursively turn integral floats back into `int`."""
    if isinstance(value, list):
        return [_restore_ints(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _resolve_type_char(obj):
    """Turn a one-character type code such as `'d'` into a real dtype."""
    if isinstance(obj, str) and len(obj) == 1:
        name = _TYPE_NAME.get(obj)
        if name is not None:
            resolved = getattr(rsnumpy, name, None)
            if resolved is not None:
                return resolved
    return obj


def _fixed_allclose():
    """rsnumpy 1.2.4 returns ``False`` whenever ``b`` is a scalar/0-d array."""

    def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        a = _as(a)
        b = _as(b)
        with rsnumpy.errstate(invalid='ignore'):
            close = rsnumpy.isclose(a, b, rtol=rtol, atol=atol)
        if equal_nan:
            close = close | (_nan_safe_isnan(a) & _nan_safe_isnan(b))
        return bool(rsnumpy.all(close))

    allclose.__name__ = 'allclose'
    allclose.__doc__ = "NumPy-compatible ``allclose`` (rsnumpy bug workaround)."
    allclose._scipy_compat = True
    return allclose


def _fixed_where(original):
    """rsnumpy 1.2.4 rejects scalar branches in ``where(cond, x, y)``."""

    def where(condition, *args):
        condition = _as(condition)
        if len(args) == 0:
            return original(condition)
        x, y = args
        shape = rsnumpy.shape(condition)
        if _RS_ARRAY(x).ndim == 0:
            x = rsnumpy.broadcast_to(_RS_ARRAY(x), shape)
        else:
            x = _to_rs(x)
        if _RS_ARRAY(y).ndim == 0:
            y = rsnumpy.broadcast_to(_RS_ARRAY(y), shape)
        else:
            y = _to_rs(y)
        return original(condition, x, y)

    where.__name__ = 'where'
    where.__doc__ = "NumPy-compatible ``where`` (rsnumpy bug workaround)."
    where._scipy_compat = True
    return where


def _make_adopting(original):
    def wrapper(*args, **kwargs):
        # rsnumpy.asarray doesn't accept 'copy' or 'device'; strip them out
        kwargs.pop('copy', None)
        kwargs.pop('device', None)
        args = tuple(_to_rs(a) for a in args)
        # Convert dtype objects to rsnumpy-compatible form
        dt = kwargs.get('dtype')
        if dt is not None:
            # Convert any dtype-like object (rsnumpy DType, numpy dtype, etc.)
            # to a plain string that rsnumpy understands.
            if hasattr(dt, 'name'):
                dt = str(dt.name)
            elif hasattr(dt, '__name__'):
                dt = str(dt.__name__)
            else:
                dt = str(dt)
            kwargs['dtype'] = dt
        # rsnumpy bug: asarray(list_of_arrays, dtype=...) flattens the result
        # (e.g. [(2,8), (2,8)] -> (2,) instead of (2,2,8)); build without the
        # dtype and convert afterwards (astype preserves shape).
        if (args and isinstance(args[0], (list, tuple)) and dt is not None
                and any(isinstance(e, rsnumpy.ndarray) for e in args[0])):
            out = original(*args, **{k: v for k, v in kwargs.items()
                                     if k != 'dtype'})
            return out.astype(dt)
        kwargs = {k: _to_rs(v) for k, v in kwargs.items()}
        return original(*args, **kwargs)

    wrapper.__name__ = getattr(original, '__name__', 'wrapper')
    wrapper.__doc__ = getattr(original, '__doc__', None)
    wrapper._scipy_compat = True
    wrapper._scipy_compat_original = original
    return wrapper


install()
