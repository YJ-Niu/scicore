"""scicore.fft / scicore.fftpack — functional tests against analytical refs."""
import pytest
import rsnumpy as np

import scicore.fft as sfft

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_fftshift():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(sfft.fftshift(x), [3.0, 4.0, 1.0, 2.0])


def test_ifftshift_roundtrip():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(sfft.ifftshift(sfft.fftshift(x)), x)


# ---------------- not yet ported / broken ----------------

@pytest.mark.xfail(reason="BackendNotImplementedError: no backend for rsnumpy.scipy.fft")
def test_fft_impulse():
    x = np.zeros(8)
    x[0] = 1.0
    assert np.allclose(sfft.fft(x), np.ones(8))


@pytest.mark.xfail(reason="BackendNotImplementedError: no backend for rsnumpy.scipy.fft")
def test_fft_roundtrip():
    x = np.array([1.0, 2.0, -0.5, 3.0, 0.25, -1.0, 2.0, 0.0])
    assert np.allclose(sfft.ifft(sfft.fft(x)), x)


@pytest.mark.xfail(reason="BackendNotImplementedError: no backend for rsnumpy.scipy.fft")
def test_rfft_roundtrip():
    x = np.array([1.0, 2.0, -0.5, 3.0, 0.25, -1.0])
    X = sfft.rfft(x)
    assert X.shape == (4,)
    assert np.allclose(sfft.irfft(X), x)


@pytest.mark.xfail(reason="BackendNotImplementedError: no backend for rsnumpy.scipy.fft")
def test_fftn_2d():
    A = sfft.fftn(np.ones((2, 2)))
    assert np.isclose(A[0, 0], 4.0)


def test_fftfreq():
    f = sfft.fftfreq(8)
    assert np.allclose(f, [0, 1 / 8, 2 / 8, 3 / 8, -4 / 8, -3 / 8, -2 / 8, -1 / 8])


def test_next_fast_len():
    # 100 = 2^2 * 5^2 and 256 = 2^8 are already "fast" lengths (all prime
    # factors in {2, 3, 5, 7, 11}), so they map to themselves.
    assert sfft.next_fast_len(100) == 100
    assert sfft.next_fast_len(256) == 256
    # prime 101 rounds up to the next fast length 105 = 3 * 5 * 7
    assert sfft.next_fast_len(101) == 105
    # documented example: real transforms use radices {2, 3, 5}
    assert sfft.next_fast_len(93059, real=True) == 93312


@pytest.mark.xfail(reason="good_size() takes 1 positional argument but 2 were given")
def test_fftconvolve():
    c = sfft.fftconvolve(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.5]), mode="full")
    assert np.allclose(c, [0.0, 1.0, 2.5, 3.5, 1.5])
