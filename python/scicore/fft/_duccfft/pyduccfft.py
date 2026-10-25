"""Pure-Python FFT backend replacing the deleted C ``pyduccfft`` extension.

Implements the ``pfft`` interface used by :mod:`scicore.fft._duccfft.basic`:
``c2c`` / ``r2c`` / ``c2r`` (1-D and N-D) plus the size helpers.  Radix-2 FFT
for power-of-two lengths and Bluestein's chirp-z algorithm for arbitrary
lengths (never worse than O(n log^2 n)).  As Rust kernels are added under
``rust/src/``, individual functions can be migrated back to compiled code.
"""

from __future__ import annotations

import cmath
import itertools
import math

import rsnumpy as np

__all__ = ['c2c', 'r2c', 'c2r', 'r2r_fftpack',
           'good_size', 'next_fast_len', 'prev_good_size']


# ---------------------------------------------------------------------------
# Core 1-D complex FFT on Python complex lists (in-place)
# ---------------------------------------------------------------------------

def _fft_pow2(x, inverse=False):
    """Iterative in-place radix-2 FFT; ``len(x)`` must be a power of two."""
    n = len(x)
    if n <= 1:
        return
    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    # butterflies
    length = 2
    while length <= n:
        ang = (2.0 if inverse else -2.0) * math.pi / length
        wlen = cmath.exp(1j * ang)
        half = length >> 1
        for start in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(start, start + half):
                t = w * x[k + half]
                u = x[k]
                x[k] = u + t
                x[k + half] = u - t
                w *= wlen
        length <<= 1


def _fft_bluestein(x, inverse=False):
    """Bluestein's chirp-z FFT for arbitrary ``len(x)`` (in-place).

    Reduces the n-point DFT to a circular convolution on Z_{2n} of the
    chirped input with the chirp kernel (which is even, so no parity
    factor appears), computed by zero-padding to the next power of two.
    """
    n = len(x)
    if n == 1:
        return
    two_n = 2 * n
    m = 1
    while m < 4 * n - 1:
        m <<= 1
    s_in = 1j if inverse else -1j      # input chirp e^{s_in i pi t^2 / n}
    s_out = -s_in                      # convolution chirp (opposite sign)
    w = [cmath.exp(s_out * cmath.pi * ((j * j) % (2 * n)) / n)
         for j in range(two_n)]
    a = [x[t] * cmath.exp(s_in * cmath.pi * ((t * t) % (2 * n)) / n)
         for t in range(n)]
    a += [0j] * (m - n)
    w += [0j] * (m - two_n)
    _fft_pow2(a)
    _fft_pow2(w)
    for k in range(m):
        a[k] *= w[k]
    _fft_pow2(a, inverse=True)
    # 1/m deconvolves the linear convolution; the 1/n DFT normalization is
    # applied by the caller (``_scale_factor``), matching ``_fft_pow2``.
    scale = 1.0 / m
    x[:] = [scale * (a[k] + a[k + two_n]) *
            cmath.exp(s_in * cmath.pi * ((k * k) % (2 * n)) / n)
            for k in range(n)]


def _fft1d(x, inverse=False):
    """FFT (``inverse=False``) or IFFT of a 1-D list, in place, any length."""
    n = len(x)
    if n == 1:
        return x
    if n & (n - 1) == 0:
        _fft_pow2(x, inverse)
    else:
        _fft_bluestein(x, inverse)
    return x


# ---------------------------------------------------------------------------
# Axis helpers for N-D transforms on nested Python lists
# ---------------------------------------------------------------------------

def _get(data, idx):
    for i in idx:
        data = data[i]
    return data


def _set(data, idx, value):
    for i in idx[:-1]:
        data = data[i]
    data[idx[-1]] = value


def _map_axis(data, axis, f, ndim, shape, new_len=None):
    """Apply ``f`` (1-D list -> 1-D list) to every slice along ``axis``
    (counted from the left).

    ``new_len=None`` keeps the slice length (written back in place); otherwise
    a fresh nested list is built with the axis length replaced by ``new_len``
    (in-place writing is impossible when the length changes).
    """
    n_axis = shape[axis]
    rest = [a for a in range(ndim) if a != axis]
    if new_len is None:
        for rest_idx in itertools.product(*[range(shape[a]) for a in rest]):
            p = [0] * ndim
            for a, v in zip(rest, rest_idx):
                p[a] = v
            sl = []
            for i in range(n_axis):
                p[axis] = i
                sl.append(_get(data, p))
            tf = f(sl)
            for i, v in enumerate(tf):
                p[axis] = i
                _set(data, p, v)
        return data
    values = {}
    for rest_idx in itertools.product(*[range(shape[a]) for a in rest]):
        p = [0] * ndim
        for a, v in zip(rest, rest_idx):
            p[a] = v
        sl = []
        for i in range(n_axis):
            p[axis] = i
            sl.append(_get(data, p))
        values[rest_idx] = f(sl)
    lens = [shape[a] for a in rest]

    def build(pos, prefix):
        if pos == len(rest):
            return values[prefix]
        return [build(pos + 1, prefix + (i,)) for i in range(lens[pos])]

    return build(0, ())


def _scale_factor(norm, n):
    """Per-axis scale for the ducc norm code (0=backward, 1=ortho, 2=forward)."""
    if norm == 1:
        return 1.0 / math.sqrt(n)
    if norm == 2:
        return 1.0 / n
    return 1.0


def _rows_transform(data, axis, ndim, shape, forward, norm,
                    slice_keep=None):
    """Transform every 1-D slice along ``axis`` (direction + per-axis scale).

    ``slice_keep`` (r2c forward) truncates each transformed row to the first
    ``slice_keep`` bins (the half spectrum).
    """
    s = _scale_factor(norm, shape[axis])
    inv = not forward

    if slice_keep is not None:
        def f(row, s=s, inv=inv, keep=slice_keep):
            row = _fft1d(list(row), inv)[:keep]
            if s != 1.0:
                row = [v * s for v in row]
            return row

        return _map_axis(data, axis, f, ndim, shape, new_len=slice_keep)

    def f(row, s=s, inv=inv):
        row = _fft1d(list(row), inv)
        if s != 1.0:
            row = [v * s for v in row]
        return row

    return _map_axis(data, axis, f, ndim, shape)


# ---------------------------------------------------------------------------
# pfft interface (called by scicore.fft._duccfft.basic)
# ---------------------------------------------------------------------------

def c2c(a, axes, forward, norm, out, workers):
    """Complex-to-complex transform along ``axes`` (1-D or N-D).

    ``norm`` is the ducc code: 0 = no scaling here, 1 = 1/sqrt(n), 2 = 1/n,
    applied per transformed axis.  ``out``/``workers`` are accepted for
    interface compatibility (results are always freshly allocated).
    """
    data = a.tolist()
    ndim = len(a.shape)
    shape = list(a.shape)
    for ax in axes:
        ax = ax % ndim
        data = _rows_transform(data, ax, ndim, shape, forward, norm)
    return np.array(data)


def r2c(a, axes, forward, norm, out, workers):
    """Real-to-complex: full complex on every axis except the last of
    ``axes``, which yields the half (n//2+1) spectrum for forward transforms.
    """
    data = a.tolist()
    ndim = len(a.shape)
    shape = list(a.shape)
    norm_axes = [ax % ndim for ax in axes]
    for ax in norm_axes[:-1]:
        data = _rows_transform(data, ax, ndim, shape, forward, norm)
    last = norm_axes[-1]
    n = shape[last]
    # Both directions yield the half spectrum (rfft / ihfft), so the last
    # axis is always truncated to n//2 + 1 bins.
    data = _rows_transform(data, last, ndim, shape, forward, norm,
                           slice_keep=n // 2 + 1)
    return np.array(data)


def c2r(a, axes, n, forward, norm, out, workers):
    """Complex-to-real: the last of ``axes`` holds the half spectrum and is
    expanded (Hermitian symmetry) to real output length ``n``; every other
    axis is a full complex transform.
    """
    data = a.tolist()
    ndim = len(a.shape)
    shape = list(a.shape)
    norm_axes = [ax % ndim for ax in axes]
    for ax in norm_axes[:-1]:
        data = _rows_transform(data, ax, ndim, shape, forward, norm)
    last = norm_axes[-1]
    s = _scale_factor(norm, n)
    inv = not forward

    def f(row, s=s, inv=inv, n=n):
        half = list(row)
        full = [0j] * n
        full[0] = half[0]
        if n % 2 == 0:
            full[n // 2] = half[n // 2]
            for k in range(1, n // 2):
                full[k] = half[k]
                full[n - k] = half[k].conjugate()
        else:
            for k in range(1, n // 2 + 1):
                full[k] = half[k]
                full[n - k] = half[k].conjugate()
        full = _fft1d(full, inv)
        real = [v.real for v in full]
        if s != 1.0:
            real = [v * s for v in real]
        return real

    data = _map_axis(data, last, f, ndim, shape, new_len=n)
    return np.array(data)


def r2r_fftpack(a, axes, forward, forward2, norm, out, workers):
    raise NotImplementedError("r2r_fftpack not yet ported (DCT/DST)")


# ---------------------------------------------------------------------------
# Size helpers: fast (5-smooth / complex-fast) FFT lengths
# ---------------------------------------------------------------------------

_COMPLEX_PRIMES = (2, 3, 5, 7, 11)
_REAL_PRIMES = (2, 3, 5)


def _is_fast_size(n, primes):
    """Return True if every prime factor of ``n`` is in ``primes``."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1


def good_size(n, real=False):
    """Smallest fast FFT length >= ``n`` (cf. ``scipy.fft.next_fast_len``)."""
    n = int(n)
    if n < 1:
        raise ValueError(f'next_fast_len requires positive input, got {n}')
    primes = _REAL_PRIMES if real else _COMPLEX_PRIMES
    m = n
    while not _is_fast_size(m, primes):
        m += 1
    return m


def prev_good_size(n, real=False):
    """Largest fast FFT length <= ``n`` (cf. ``scipy.fft.prev_fast_len``)."""
    n = int(n)
    if n < 1:
        raise ValueError(f'prev_fast_len requires positive input, got {n}')
    primes = _REAL_PRIMES if real else _COMPLEX_PRIMES
    m = n
    while m > 1 and not _is_fast_size(m, primes):
        m -= 1
    return m


next_fast_len = good_size


# Legacy names kept for interface compatibility; the public functions live
# in _duccfft.basic / _duccfft.realtransforms.
def _stub(*args, **kwargs):
    raise NotImplementedError("pyduccfft not yet ported to Rust/rsnumpy")


for _name in ('fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
              'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
              'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
              'dct', 'dst', 'idct', 'idst', 'dctn', 'dstn', 'idctn', 'idstn'):
    globals()[_name] = _stub
