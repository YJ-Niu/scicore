"""Isolate the ndimage 1-D write-back cost: the old
``res -> np.asarray(list).reshape -> out[...] = arr`` bulk-setitem path versus
the new zero-copy ``_into`` buffer write.  Also verifies the ``_into`` result
matches the old path and scipy to 1e-12.
"""
import statistics
import time

import rsnumpy as np
import numpy as npr
import scipy.ndimage as sni
import scicore.ndimage._nd_image as nni

rust = nni._rust_ndimage
assert rust is not None, 'Rust extension missing'

# 100 x 1000 float64 strip (1e5 elements) — the benchmark case of record.
data2d = np.array(
    [[((i * 7 + j * 13) % 97) / 97.0 for j in range(1000)] for i in range(100)])
shape = list(data2d.shape)

W = [0.5, -1.0, 0.5]


def flat(a):
    out = []
    for v in a:
        if isinstance(v, list):
            out.extend(flat(v))
        else:
            out.append(float(v))
    return out


def timeit(fn, warmup=10, reps=50):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def old_uniform(out):
    # the pre-optimization write-back: flat Vec -> list -> asarray -> setitem
    res = rust._uniform_filter1d(data2d, shape, 5, 1, 2, 0.0, 0)  # reflect
    out[...] = np.asarray(res).reshape(shape)


def new_uniform(out):
    # the zero-copy path: write directly into `out`'s buffer
    rust._uniform_filter1d_into(data2d, out, shape, 5, 1, 2, 0.0, 0)


out_old = np.zeros(shape)
out_new = np.zeros(shape)

# correctness: old path, new path, scipy must all agree
old_uniform(out_old)
new_uniform(out_new)
ref = sni.uniform_filter1d(npr.array(data2d.tolist()), 5, axis=1, mode='reflect')
rf = flat(ref.ravel().tolist())
d_old = max(abs(a - b) for a, b in zip(flat(out_old.tolist()), rf))
d_new = max(abs(a - b) for a, b in zip(flat(out_new.tolist()), rf))
print(f'correctness: old-sp {d_old:.2e}  new-sp {d_new:.2e}  '
      f'({"OK" if max(d_old, d_new) <= 1e-12 else "BAD"})')

t_old = timeit(lambda: old_uniform(out_old))
t_new = timeit(lambda: new_uniform(out_new))
print(f'uniform s=5 axis=1 (1e5 elems, reflect):')
print(f'  old writeback (asarray+setitem): {t_old:8.3f} ms')
print(f'  new zerocopy (_into buffer)   : {t_new:8.3f} ms')
print(f'  speedup: {t_old / t_new:6.2f}x')
