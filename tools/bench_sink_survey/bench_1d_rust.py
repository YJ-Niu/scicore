"""Three-way benchmark of the Rust 1-D ndimage kernels:
scicore-Rust vs scicore-pure-Python vs scipy, on realistic workloads.
"""
import statistics
import time

import rsnumpy as np
import numpy as npr
import scipy.ndimage as sni
import scicore.ndimage as snd
import scicore.ndimage._nd_image as nni

assert nni._rust_ndimage is not None, 'Rust extension missing'
saved = nni._rust_ndimage

# 100 x 1000 float64 strip (1e5 elements)
data2d = np.array(
    [[((i * 7 + j * 13) % 97) / 97.0 for j in range(1000)] for i in range(100)])
# 20 x 30 x 40 volume (2.4e4 elements)
data3d = np.array(
    [[[((i * 5 + j * 3 + k) % 53) / 53.0 for k in range(40)]
      for j in range(30)] for i in range(20)])
npr_2d = npr.array(data2d.tolist())
npr_3d = npr.array(data3d.tolist())


def flat(a):
    out = []
    for v in a:
        if isinstance(v, list):
            out.extend(flat(v))
        else:
            out.append(float(v))
    return out


def timeit(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def rust_fn(f):
    return f()


def py_fn(f):
    nni._rust_ndimage = None
    try:
        return f()
    finally:
        nni._rust_ndimage = saved


def run_case(label, sp_fn, f_public):
    r = rust_fn(f_public)
    p = py_fn(f_public)
    s = sp_fn()
    rf, pf, sf = flat(r.tolist()), flat(p.tolist()), flat(s.ravel().tolist())
    d_rs = max(abs(a - b) for a, b in zip(rf, sf))
    d_ps = max(abs(a - b) for a, b in zip(pf, sf))
    t_rust = timeit(lambda: rust_fn(f_public))
    t_py = timeit(lambda: py_fn(f_public))
    t_sp = timeit(sp_fn)
    ok = 'OK ' if (d_rs <= 1e-12 and d_ps <= 1e-12) else 'BAD'
    print(f'{ok} {label:<34s} rust={t_rust:8.3f}ms  py={t_py:8.3f}ms  '
          f'sp={t_sp:8.3f}ms  vs py {t_py / t_rust:7.1f}x  '
          f'vs scipy {t_sp / t_rust:6.1f}x  (rust-sp {d_rs:.1e}, '
          f'py-sp {d_ps:.1e})')


W = [0.5, -1.0, 0.5]
run_case('uniform s=5 axis=1 (contig)',
         lambda: sni.uniform_filter1d(npr_2d, 5, axis=1),
         lambda: snd.uniform_filter1d(data2d, 5, axis=1))
run_case('uniform s=5 axis=0 (strided)',
         lambda: sni.uniform_filter1d(npr_2d, 5, axis=0),
         lambda: snd.uniform_filter1d(data2d, 5, axis=0))
run_case('correlate [0.5,-1,0.5] axis=1',
         lambda: sni.correlate1d(npr_2d, npr.array(W), axis=1),
         lambda: snd.correlate1d(data2d, W, axis=1))
run_case('minimum s=5 axis=1',
         lambda: sni.minimum_filter1d(npr_2d, size=5, axis=1),
         lambda: snd.minimum_filter1d(data2d, 5, axis=1))
run_case('maximum s=5 axis=1',
         lambda: sni.maximum_filter1d(npr_2d, size=5, axis=1),
         lambda: snd.maximum_filter1d(data2d, 5, axis=1))
run_case('uniform s=3 axis=2 (3d volume)',
         lambda: sni.uniform_filter1d(npr_3d, 3, axis=2),
         lambda: snd.uniform_filter1d(data3d, 3, axis=2))
