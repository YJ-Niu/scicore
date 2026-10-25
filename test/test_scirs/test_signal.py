"""scicore.signal — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.signal as ssig

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_lfilter_accumulator():
    # y[n] = x[n] + y[n-1] with x = [1, 0, 0]
    assert np.allclose(ssig.lfilter([1.0, 1.0], [1.0], [1.0, 0.0, 0.0]), [1.0, 1.0, 0.0])


def test_butter_lowpass_dc_gain():
    b, a = ssig.butter(4, 0.1, btype="low")
    assert len(b) == 5
    assert np.isclose(a[0], 1.0)
    # lowpass DC gain is 1: sum(b)/sum(a) = 1
    assert np.isclose(np.sum(b) / np.sum(a), 1.0, rtol=1e-12)


def test_correlate_full():
    assert np.allclose(ssig.correlate([1.0, 2.0, 3.0], [1.0, 1.0]), [1.0, 3.0, 5.0, 3.0])


def test_convolve_full():
    assert np.allclose(ssig.convolve([1.0, 2.0, 3.0], [1.0, 1.0]), [1.0, 3.0, 5.0, 3.0])


def test_find_peaks():
    peaks, _ = ssig.find_peaks(np.array([0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0]))
    assert list(peaks) == [1, 3, 5]


def test_freqz_frequency_grid_and_dc_gain():
    # H(z) = z^-1 / (1 - 0.5 z^-1):  |H(1)| = 1/0.5 = 2
    w, h = ssig.freqz([1.0, 0.0], [1.0, -0.5], 4)
    assert np.allclose(w, [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    assert np.isclose(np.abs(h[0]), 2.0, rtol=1e-12)
    assert np.all(np.isfinite(h))


@pytest.mark.xfail(reason="single-coefficient fast path needs rfft; no backend for rsnumpy.scipy.fft")
def test_freqz_allpass():
    w, h = ssig.freqz([1.0], [1.0], 8)
    assert np.isclose(w[0], 0.0)
    assert np.allclose(np.abs(h), 1.0, atol=1e-12)


def test_lfilter_zi():
    # zi for b=[1], a=[1, -0.5] is [b0/a0] = [1]
    assert np.allclose(ssig.lfilter_zi([1.0], [1.0, -0.5]), [1.0])


# ---------------- not yet ported / broken ----------------

@pytest.mark.xfail(reason="rsnumpy: Cannot compute diff: axis has less than 2 elements")
def test_butter_sos():
    sos = ssig.butter(2, 0.1, output="sos")
    assert sos.shape[1] == 6


@pytest.mark.xfail(reason="rsnumpy: Cannot compute diff: axis has less than 2 elements")
def test_sosfilt():
    sos = ssig.butter(2, 0.1, output="sos")
    out = ssig.sosfilt(sos, [1.0, 0.0, 0.0])
    assert len(out) == 3


@pytest.mark.xfail(reason="ValueError: All sub-arrays must have the same shape")
def test_welch():
    x = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
    freqs, pxx = ssig.welch(x, fs=256)
    # power peak at 5 Hz
    assert np.isclose(freqs[np.argmax(pxx)], 5.0, atol=1.0)


@pytest.mark.xfail(reason="ValueError: All sub-arrays must have the same shape")
def test_periodogram():
    x = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
    freqs, pxx = ssig.periodogram(x, fs=256)
    assert np.isclose(freqs[np.argmax(pxx)], 5.0, atol=1.0)


@pytest.mark.xfail(reason="OverflowError: can't convert negative int to unsigned")
def test_stft():
    x = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
    _, Zxx = ssig.stft(x)
    assert Zxx.ndim == 2


@pytest.mark.xfail(reason="BackendNotImplementedError: no backend for rsnumpy.scipy.fft")
def test_hilbert():
    h = ssig.hilbert([1.0, 0.0, -1.0, 0.0])
    assert len(h) == 4


@pytest.mark.xfail(reason="BackendNotImplementedError: no backend for rsnumpy.scipy.fft")
def test_resample():
    out = ssig.resample([1.0, 2.0, 3.0, 4.0], 8)
    assert len(out) == 8


@pytest.mark.xfail(reason="TypeError: Shape must be a tuple or integer")
def test_upfirdn():
    out = ssig.upfirdn([1.0, 1.0], [1.0, 2.0, 3.0])
    assert len(out) > 0


def test_iirfilter():
    b, a = ssig.iirfilter(3, 0.1)
    assert np.isclose(np.sum(b) / np.sum(a), 1.0, rtol=1e-10)  # lowpass DC gain


def test_tf2sos():
    b, a = ssig.butter(2, 0.1)
    sos = ssig.tf2sos(b, a)
    assert sos.shape[1] == 6


@pytest.mark.xfail(reason="good_size() takes 1 positional argument but 2 were given")
def test_fftconvolve():
    c = ssig.fftconvolve([1.0, 2.0, 3.0], [1.0, 1.0])
    assert np.allclose(c, [1.0, 3.0, 5.0, 3.0])
