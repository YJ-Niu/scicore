"""三方对照：scipy / scicore-Rust / scicore-纯Python（禁用 Rust）。
定位偏差来自 Rust 移植还是 scicore Python 既有行为。"""
import numpy as npr
import scipy.ndimage as sni
import scicore.ndimage as snd
import scicore.ndimage._nd_image as nni

fdata = [[float((i * 7 + j * 13) % 11) / 11.0 for j in range(17)]
         for i in range(9)]
sdata = npr.array(fdata)

saved = nni._rust_ndimage


def show(label, rust_fn, py_fn, sp_fn):
    r = list(map(list, rust_fn()))
    nni._rust_ndimage = None
    try:
        p = list(map(list, py_fn()))
    finally:
        nni._rust_ndimage = saved
    s = list(map(list, sp_fn()))
    rust_vs_sp = max(abs(a - b) for ra, sa in zip(r, s) for a, b in zip(ra, sa))
    py_vs_sp = max(abs(a - b) for pa, sa in zip(p, s) for a, b in zip(pa, sa))
    rust_vs_py = max(abs(a - b) for ra, pa in zip(r, p) for a, b in zip(ra, pa))
    print(f'{label}:')
    print(f'  rust vs scipy = {rust_vs_sp:.6f}   py   vs scipy = {py_vs_sp:.6f}   '
          f'rust vs py = {rust_vs_py:.6f}')


for mode in ('reflect', 'mirror', 'wrap', 'nearest', 'constant'):
    for axis in (0, 1):
        show(f'uniform2d m={mode} a={axis} s=3',
             lambda m=mode, a=axis: snd.uniform_filter1d(fdata, 3, axis=a, mode=m),
             lambda m=mode, a=axis: snd.uniform_filter1d(fdata, 3, axis=a, mode=m),
             lambda m=mode, a=axis: sni.uniform_filter1d(sdata, 3, axis=a, mode=m))
