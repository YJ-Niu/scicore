from functools import update_wrapper, lru_cache
import inspect

from ._duccfft import helper as _helper

import rsnumpy as np
from scicore._lib._array_api import array_namespace
from scicore._lib._array_api import xp_capabilities


_init_nd_shape_and_axes = _helper._init_nd_shape_and_axes


def next_fast_len(target, real=False):
    """Find the next fast size of input data to ``fft``, for zero-padding, etc.

    SciPy's FFT algorithms gain their speed by a recursive divide and conquer
    strategy. This relies on efficient functions for small prime factors of the
    input length. Thus, the transforms are fastest when using composites of the
    prime factors handled by the fft implementation. If there are efficient
    functions for all radices <= `n`, then the result will be a number `x`
    >= ``target`` with only prime factors < `n`. (Also known as `n`-smooth
    numbers)

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.
    real : bool, optional
        True if the FFT involves real input or output (e.g., `rfft` or `hfft`
        but not `fft`). Defaults to False.

    Returns
    -------
    out : int
        The smallest fast length greater than or equal to ``target``.

    Notes
    -----
    The result of this function may change in future as performance
    considerations change, for example, if new prime factors are added.

    Calling `fft` or `ifft` with real input data performs an ``'R2C'``
    transform internally.

    Examples
    --------
    On a particular machine, an FFT of prime length takes 11.4 ms:

    >>> from scicore import fft
    >>> import rsnumpy as np
    >>> rng = np.random.default_rng()
    >>> min_len = 93059  # prime length is worst case for speed
    >>> a = rng.standard_normal(min_len)
    >>> b = fft.fft(a)

    Zero-padding to the next regular length reduces computation time to
    1.6 ms, a speedup of 7.3 times:

    >>> fft.next_fast_len(min_len, real=True)
    93312
    >>> b = fft.fft(a, 93312)

    Rounding up to the next power of 2 is not optimal, taking 3.0 ms to
    compute; 1.9 times longer than the size given by ``next_fast_len``:

    >>> b = fft.fft(a, 131072)

    """
    pass


# Directly wrap the c-function good_size but take the docstring etc., from the
# next_fast_len function above
_sig = inspect.signature(next_fast_len)
next_fast_len = update_wrapper(lru_cache(_helper.good_size), next_fast_len)
next_fast_len = xp_capabilities(out_of_scope=True)(next_fast_len)
next_fast_len.__wrapped__ = _helper.good_size
next_fast_len.__signature__ = _sig


def prev_fast_len(target, real=False):
    """Find the previous fast size of input data to ``fft``.
    Useful for discarding a minimal number of samples before FFT.

    SciPy's FFT algorithms gain their speed by a recursive divide and conquer
    strategy. This relies on efficient functions for small prime factors of the
    input length. Thus, the transforms are fastest when using composites of the
    prime factors handled by the fft implementation. If there are efficient
    functions for all radices <= `n`, then the result will be a number `x`
    <= ``target`` with only prime factors <= `n`. (Also known as `n`-smooth
    numbers)

    Parameters
    ----------
    target : int
        Maximum length to search until. Must be a positive integer.
    real : bool, optional
        True if the FFT involves real input or output (e.g., `rfft` or `hfft`
        but not `fft`). Defaults to False.

    Returns
    -------
    out : int
        The largest fast length less than or equal to ``target``.

    Notes
    -----
    The result of this function may change in future as performance
    considerations change, for example, if new prime factors are added.

    Calling `fft` or `ifft` with real input data performs an ``'R2C'``
    transform internally.

    In the current implementation, prev_fast_len assumes radices of
    2,3,5,7,11 for complex FFT and 2,3,5 for real FFT.

    Examples
    --------
    On a particular machine, an FFT of prime length takes 16.2 ms:

    >>> from scicore import fft
    >>> import rsnumpy as np
    >>> rng = np.random.default_rng()
    >>> max_len = 93059  # prime length is worst case for speed
    >>> a = rng.standard_normal(max_len)
    >>> b = fft.fft(a)

    Performing FFT on the maximum fast length less than max_len
    reduces the computation time to 1.5 ms, a speedup of 10.5 times:

    >>> fft.prev_fast_len(max_len, real=True)
    92160
    >>> c = fft.fft(a[:92160]) # discard last 899 samples

    """
    pass


# Directly wrap the c-function prev_good_size but take the docstring etc.,
# from the prev_fast_len function above
_sig_prev_fast_len = inspect.signature(prev_fast_len)
prev_fast_len = update_wrapper(lru_cache()(_helper.prev_good_size), prev_fast_len)
prev_fast_len = xp_capabilities(out_of_scope=True)(prev_fast_len)
prev_fast_len.__wrapped__ = _helper.prev_good_size
prev_fast_len.__signature__ = _sig_prev_fast_len


@xp_capabilities()
def fftfreq(n, d=1.0, *, xp=None, device=None):
    """Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    xp : array_namespace, optional
        The namespace for the return array. Default is None, where NumPy is used.
    device : device, optional
        The device for the return array.
        Only valid when `xp.fft.fftfreq` implements the device parameter.

    Returns
    -------
    f : ndarray
        Array of length `n` containing the sample frequencies.

    Examples
    --------
    >>> import rsnumpy as np
    >>> import scicore.fft
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = scipy.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = scipy.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])

    """
    xp = np if xp is None else xp
    if device is None:
        # rsnumpy's fft.fftfreq does not accept the device keyword.
        if hasattr(xp, 'fft'):
            return xp.fft.fftfreq(n, d=d)
        return np.fft.fftfreq(n, d=d)
    if hasattr(xp, 'fft'):
        return xp.fft.fftfreq(n, d=d, device=device)
    return np.fft.fftfreq(n, d=d, device=device)


@xp_capabilities()
def rfftfreq(n, d=1.0, *, xp=None, device=None):
    """Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    xp : array_namespace, optional
        The namespace for the return array. Default is None, where NumPy is used.
    device : device, optional
        The device for the return array.
        Only valid when `xp.fft.rfftfreq` implements the device parameter.

    Returns
    -------
    f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.

    Examples
    --------
    >>> import rsnumpy as np
    >>> import scicore.fft
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = scipy.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = scipy.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20., ..., -30., -20., -10.])
    >>> freq = scipy.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    xp = np if xp is None else xp
    if device is None:
        # rsnumpy's fft.rfftfreq does not accept the device keyword.
        if hasattr(xp, 'fft'):
            return xp.fft.rfftfreq(n, d=d)
        return np.fft.rfftfreq(n, d=d)
    if hasattr(xp, 'fft'):
        return xp.fft.rfftfreq(n, d=d, device=device)
    return np.fft.rfftfreq(n, d=d, device=device)


@xp_capabilities()
def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    >>> import rsnumpy as np
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2., ..., -3., -2., -1.])
    >>> np.fft.fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    Shift the zero-frequency component only along the second axis:

    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.fftshift(freqs, axes=(1,))
    array([[ 2.,  0.,  1.],
           [-4.,  3.,  4.],
           [-1., -3., -2.]])

    """
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.fftshift(x, axes=axes)
    x = np.asarray(x)
    y = np.fft.fftshift(x, axes=axes)
    return xp.asarray(y)


@xp_capabilities()
def ifftshift(x, axes=None):
    """The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    fftshift : Shift zero-frequency component to the center of the spectrum.

    Examples
    --------
    >>> import rsnumpy as np
    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.ifftshift(np.fft.fftshift(freqs))
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])

    """
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.ifftshift(x, axes=axes)
    x = np.asarray(x)
    y = np.fft.ifftshift(x, axes=axes)
    return xp.asarray(y)


def fftconvolve(in1, in2, mode='full', axes=None, norm='backward'):
    """Convolve two N-dimensional arrays using FFT.

    Convolve ``in1`` and ``in2`` using the fast Fourier transform method,
    with the output size determined by ``mode``.  This delegates to
    :func:`scicore.signal.fftconvolve`, which implements the same
    full/same/valid semantics (the import is deferred to avoid an import
    cycle).

    Parameters
    ----------
    in1, in2 : array_like
        The arrays to convolve.  Must have the same number of dimensions.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output (see
        :func:`scicore.signal.fftconvolve`).  Default is 'full'.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.  The default is over
        all axes.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode.  The default ``'backward'`` yields the standard
        discrete linear convolution; ``'ortho'`` and ``'forward'`` divide
        the result by the product of the full-convolution lengths along
        the transformed axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of ``in1`` with ``in2``.

    Examples
    --------
    >>> import rsnumpy as np
    >>> from scicore.fft import fftconvolve
    >>> fftconvolve(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.5]))
    array([0. , 1. , 2.5, 3.5, 1.5])
    """
    from scicore.signal import fftconvolve as _signal_fftconvolve

    a = np.asarray(in1)
    b = np.asarray(in2)
    s1 = a.shape
    s2 = b.shape
    result = _signal_fftconvolve(a, b, mode=mode, axes=axes)
    if norm in (None, 'backward'):
        return result
    if norm not in ('ortho', 'forward'):
        raise ValueError(f"invalid norm {norm!r}")
    if axes is None:
        axes = tuple(range(len(s1)))
    elif isinstance(axes, int):
        axes = (axes,)
    scale = 1.0
    for ax in axes:
        scale *= (s1[ax] + s2[ax] - 1)
    return result / scale
