"""Verify the Rust 1-D ndimage kernels against scipy (ground truth) and
against the pure-Python fallback path.

Checks uniform_filter1d / correlate1d / minimum_filter1d / maximum_filter1d
across modes, axes, sizes, origins, dims, dtypes (float64/complex/int) and
explicitly exercises the fallback path by disabling the Rust extension.
"""
import sys
import traceback

import rsnumpy as np
import numpy as npr
import scipy.ndimage as sni
import scicore.ndimage as snd
import scicore.ndimage._nd_image as nni

RUST_LOADED = nni._rust_ndimage is not None
print('rust extension loaded:', RUST_LOADED)
assert RUST_LOADED, 'Rust extension missing - rebuild wheel first'

failures = []
checked = 0


def _flat(a):
    out = []
    for v in a:
        if isinstance(v, list):
            out.extend(_flat(v))
        else:
            out.append(v)
    return out


def maxdiff(a, b):
    at = a.tolist() if hasattr(a, 'tolist') else list(a)
    bt = b.tolist() if hasattr(b, 'tolist') else list(b)
    if not isinstance(at, list):
        at = [at]
    if not isinstance(bt, list):
        bt = [bt]
    return max(abs(x - y) for x, y in zip(_flat(at), _flat(bt)))


def check(label, scicore_fn, scipy_fn, tol=1e-12):
    global checked
    try:
        r_sc = scicore_fn()
        r_sp = scipy_fn()
        d = maxdiff(r_sc, r_sp)
        checked += 1
        if d > tol:
            failures.append((label, f'maxdiff={d:.3e} > {tol}'))
    except Exception:
        checked += 1
        failures.append((label, traceback.format_exc(limit=3).splitlines()[-1]))


fdata = [[float((i * 7 + j * 13) % 11) / 11.0 for j in range(17)]
         for i in range(9)]
sdata = npr.array(fdata)
v1d = [float(i * 3 % 13) for i in range(23)]
v3d = [[[float((i * 5 + j * 3 + k) % 7) for k in range(6)]
        for j in range(5)] for i in range(4)]
s3d = npr.array(v3d)

MODES = ['nearest', 'wrap', 'reflect', 'mirror', 'constant',
         'grid-wrap', 'grid-constant']


def origins_for(size):
    return range(-(size // 2), size - size // 2)


# ---- uniform_filter1d -------------------------------------------------
for data, sdata_c, ndim in ((fdata, sdata, (0, 1)), (v1d, npr.array(v1d), (0,)),
                            (v3d, s3d, (0, 1, 2))):
    for mode in MODES:
        for axis in ndim:
            for size in (1, 2, 3, 5):
                for origin in origins_for(size):
                    check(
                        f'uniform {data is fdata and "2d" or data is v1d and "1d" or "3d"} '
                        f'm={mode} a={axis} s={size} o={origin}',
                        lambda d=data, mo=mode, ax=axis, s=size, o=origin:
                        snd.uniform_filter1d(d, s, axis=ax, mode=mo,
                                             cval=0.25, origin=o),
                        lambda d=sdata_c, mo=mode, ax=axis, s=size, o=origin:
                        sni.uniform_filter1d(d, s, axis=ax, mode=mo,
                                             cval=0.25, origin=o))

# ---- correlate1d ------------------------------------------------------
for w_py, w_np, wname in (([1.0, 2.0, 3.0, 2.0, 1.0], None, 'odd5'),
                          ([1.0, 3.0], None, 'even2'),
                          ([0.5, -1.0, 0.5], None, 'lap3')):
    w_np = npr.array(w_py)
    for mode in MODES:
        for axis in (0, 1):
            half = len(w_py) // 2
            for origin in range(-half, len(w_py) - half):
                check(
                    f'correlate {wname} m={mode} a={axis} o={origin}',
                    lambda d=fdata, w=w_py, mo=mode, ax=axis, o=origin:
                    snd.correlate1d(d, w, axis=ax, mode=mo, cval=-1.0,
                                    origin=o),
                    lambda d=sdata, w=w_np, mo=mode, ax=axis, o=origin:
                    sni.correlate1d(d, w, axis=ax, mode=mo, cval=-1.0,
                                    origin=o))

# ---- min/max filters --------------------------------------------------
for name, sc_func, sp_func in (('min', snd.minimum_filter1d,
                               sni.minimum_filter1d),
                              ('max', snd.maximum_filter1d,
                               sni.maximum_filter1d)):
    for mode in MODES:
        for axis in (0, 1):
            for size in (1, 2, 3, 5):
                check(
                    f'{name} m={mode} a={axis} s={size}',
                    lambda d=fdata, mo=mode, ax=axis, s=size, f=sc_func:
                    f(d, s, axis=ax, mode=mo, cval=100.0),
                    lambda d=sdata, mo=mode, ax=axis, s=size, f=sp_func:
                    f(d, s, axis=ax, mode=mo, cval=100.0))

# ---- complex input -----------------------------------------------------
# Pre-existing rsnumpy defects (NOT port regressions; scipy returns the
# true complex result, both scicore paths share the same broken behavior):
#   * np.asarray on a list of complex numbers silently drops the imaginary
#     part (float64 result), while np.array keeps complex128.
#   * .real/.imag views are read-only in rsnumpy (bulk AND element setitem
#     are silent no-ops), so complex array input yields an all-zero
#     complex128 output.
# The Rust port must preserve the current behavior exactly, so verify
# rust-path vs pure-Python-path here instead of rust vs scipy.
cdata = [[(fdata[i][j] + 0.5j) for j in range(17)] for i in range(9)]
carr = np.array(cdata)
saved_wb = nni._rust_ndimage


def complex_pair(label, data, mode, axis):
    global checked
    rr = snd.uniform_filter1d(data, 3, axis=axis, mode=mode)
    nni._rust_ndimage = None
    try:
        pp = snd.uniform_filter1d(data, 3, axis=axis, mode=mode)
    finally:
        nni._rust_ndimage = saved_wb
    checked += 1
    same_dtype = str(rr.dtype) == str(pp.dtype)
    same_vals = maxdiff(rr, pp) == 0.0
    if not (same_dtype and same_vals):
        failures.append(
            (label, f'dtype/behavior mismatch: {rr.dtype} vs {pp.dtype}, '
                    f'maxdiff={maxdiff(rr, pp):.3e}'))


for data, dname in ((cdata, 'list'), (carr, 'array')):
    for mode in ('reflect', 'constant'):
        for axis in (0, 1):
            complex_pair(f'complex {dname} m={mode} a={axis}', data, mode,
                         axis)

# ---- integer input (fallback path, dtype guard rejects f64 only) ------
idata = [[(i * 7 + j * 13) % 11 for j in range(17)] for i in range(9)]
for name, sc_func, sp_func in (('uniform', snd.uniform_filter1d,
                               sni.uniform_filter1d),
                              ('min', snd.minimum_filter1d,
                               sni.minimum_filter1d)):
    check(
        f'int input {name}',
        lambda d=idata: sc_func(d, 3),
        lambda d=npr.array(idata): sp_func(d, 3))

# ---- fallback path: disable Rust, must still match scipy ---------------
saved = nni._rust_ndimage
nni._rust_ndimage = None
try:
    check('fallback uniform', lambda: snd.uniform_filter1d(fdata, 3),
          lambda: sni.uniform_filter1d(sdata, 3))
    check('fallback correlate', lambda: snd.correlate1d(fdata, [1.0, 3.0]),
          lambda: sni.correlate1d(sdata, npr.array([1.0, 3.0])))
    check('fallback min', lambda: snd.minimum_filter1d(fdata, 5),
          lambda: sni.minimum_filter1d(sdata, 5))
    check('fallback max', lambda: snd.maximum_filter1d(fdata, 5),
          lambda: sni.maximum_filter1d(sdata, 5))
finally:
    nni._rust_ndimage = saved

# ---- docstring example (scicore _filters.py) ---------------------------
check('docstring correlate1d example',
      lambda: snd.correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3]),
      lambda: sni.correlate1d(npr.array([2, 8, 0, 4, 1, 9, 9, 0]),
                              weights=npr.array([1, 3])))
check('docstring uniform_filter1d example',
      lambda: snd.uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3),
      lambda: sni.uniform_filter1d(npr.array([2, 8, 0, 4, 1, 9, 9, 0]),
                                   size=3))

print(f'checked {checked} cases, failures: {len(failures)}')
for label, msg in failures:
    print(f'  FAIL {label}: {msg}')
sys.exit(1 if failures else 0)
