"""scicore.ndimage — functional tests against analytical references."""
import math

import pytest
import rsnumpy as np

import scicore.ndimage as ndi

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_sum():
    assert np.isclose(ndi.sum(np.ones((2, 2))), 4.0)


# ---------------- not yet ported / broken ----------------

def test_gaussian_filter():
    out = ndi.gaussian_filter(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), 1.0)
    # kernel mass concentrates at the center
    assert out[1, 1] == out.max()


def test_sobel():
    # constant image has zero Sobel response
    assert np.allclose(ndi.sobel(np.ones((3, 3))), 0.0)


def test_median_filter():
    # scipy requires size or footprint; majority-1s image has median 1.0
    img = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    assert ndi.median_filter(img, size=3)[1, 1] == 1.0


def test_uniform_filter():
    out = ndi.uniform_filter(np.ones((4, 4)), size=2)
    assert np.isclose(out[1, 1], 1.0)


def test_label():
    labels, n = ndi.label(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
    assert n == 2
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 0] != labels[2, 2]


def test_center_of_mass():
    img = np.array([[0.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(ndi.center_of_mass(img), [1.0, 1.0])


def test_binary_erosion():
    # 3x3 ones eroded by 3x3 structure leaves the center 1
    assert ndi.binary_erosion(np.ones((3, 3), dtype=int))[1, 1] == 1


def test_binary_dilation():
    img = np.zeros((3, 3), dtype=int)
    img[1, 1] = 1
    assert np.all(ndi.binary_dilation(img, structure=np.ones((3, 3), dtype=int)))


def test_morphological_gradient():
    # scipy requires one of size/footprint/structure
    out = ndi.morphological_gradient(np.ones((3, 3), dtype=int), size=(3, 3))
    # interior of a uniform block has zero gradient
    assert out[1, 1] == 0


def test_zoom():
    out = ndi.zoom(np.array([[1.0]]), 2.0)
    assert out.shape == (2, 2)


def test_shift():
    out = ndi.shift(np.array([[1.0, 0.0]]), [0.0, 1.0])
    assert np.isclose(out[0, 1], 1.0)


def test_rotate():
    out = ndi.rotate(np.array([[1.0, 0.0], [0.0, 0.0]]), 90, order=0)
    assert out.shape == (2, 2)


def test_distance_transform_bf():
    mask = 1 - np.array([[1.0, 1.0], [1.0, 1.0]])  # all background
    out = ndi.distance_transform_bf(mask)
    assert out.shape == (2, 2)


def test_find_objects():
    # two isolated singletons with DIFFERENT labels -> two objects
    out = ndi.find_objects(np.array([[1, 0], [0, 2]]))
    assert len(out) == 2


# ---------------- fourier filters ----------------

def _freqs(n):
    return [float(t) for t in np.asarray(np.fft.fftfreq(n)).tolist()]


def _spec_1d(n, k0, re=2.0, im=1.0):
    # single unit component at bin k0
    X = [0.0 + 0.0j] * n
    X[k0] = re + 1j * im
    return np.array(X)


def test_fourier_gaussian_single_frequency():
    n = 16
    sigma = 1.5
    f = _freqs(n)
    for k0 in (0, 1, 4, 8, 13):  # DC, low, mid, Nyquist, negative freq
        Y = np.asarray(ndi.fourier_gaussian(_spec_1d(n, k0), sigma)).tolist()
        expected = (2.0 + 1j) * math.exp(-2.0 * math.pi**2 * sigma**2 * f[k0]**2)
        assert abs(Y[k0] - expected) < 1e-10
        assert all(abs(Y[j]) < 1e-14 for j in range(n) if j != k0)


def test_fourier_uniform_single_frequency():
    n = 16
    size = 2.0
    f = _freqs(n)
    for k0 in (0, 1, 5, 8):
        Y = np.asarray(ndi.fourier_uniform(_spec_1d(n, k0), size)).tolist()
        t = math.pi * size * f[k0]
        m = 1.0 if abs(t) < 1e-8 else math.sin(t) / t
        assert abs(Y[k0] - (2.0 + 1j) * m) < 1e-10


def test_fourier_ellipsoid_single_frequency():
    n = 16
    size = 1.0
    f = _freqs(n)
    for k0 in (0, 3, 8, 12):
        Y = np.asarray(ndi.fourier_ellipsoid(_spec_1d(n, k0), size)).tolist()
        m = math.exp(-(size * f[k0]) ** 2)
        assert abs(Y[k0] - (2.0 + 1j) * m) < 1e-10


def test_fourier_gaussian_2d_per_axis():
    n1, n2 = 8, 6
    X = [[0.0 + 0.0j] * n2 for _ in range(n1)]
    X[3][4] = 2.0 + 1.0j
    f1 = _freqs(n1)[3]
    f2 = _freqs(n2)[4]
    s1, s2 = 1.0, 2.0
    Y = np.asarray(ndi.fourier_gaussian(np.array(X), [s1, s2])).tolist()
    m = (math.exp(-2 * math.pi**2 * s1**2 * f1**2)
         * math.exp(-2 * math.pi**2 * s2**2 * f2**2))
    assert abs(Y[3][4] - (2.0 + 1j) * m) < 1e-10


def test_fourier_shift_roundtrip():
    # integer shift: ifft(fourier_shift(fft(x), s)) == circular roll by s
    xs = [1.0, 2.0, 1.5, 0.0, 3.0, 2.5, 0.0, 1.0, 0.5, 2.0, 1.0, 0.0]
    x = np.array(xs)
    y = np.asarray(np.fft.ifft(ndi.fourier_shift(np.fft.fft(x), 3))).tolist()
    for i in range(len(xs)):
        assert abs(y[i].real - xs[(i - 3) % len(xs)]) < 1e-10
        assert abs(y[i].imag) < 1e-10


def test_fourier_shift_2d_single_frequency():
    n1, n2 = 4, 6
    X = [[0.0 + 0.0j] * n2 for _ in range(n1)]
    X[2][3] = 1.0 + 0.0j
    f1 = _freqs(n1)[2]
    f2 = _freqs(n2)[3]
    s1, s2 = 1.5, -0.5
    Y = np.asarray(ndi.fourier_shift(np.array(X), [s1, s2])).tolist()
    ang = -2.0 * math.pi * (f1 * s1 + f2 * s2)
    expected = complex(math.cos(ang), math.sin(ang))
    assert abs(Y[2][3] - expected) < 1e-10
    for i in range(n1):
        for j in range(n2):
            if (i, j) != (2, 3):
                assert abs(Y[i][j]) < 1e-14


def test_fourier_gaussian_real_fft_path():
    # n >= 0: input is a real FFT along the axis; must agree with the
    # full-complex path and with the hand-built circular kernel
    N = 12
    xr = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
    x = np.array(xr)
    sig = 1.0
    Yr = ndi.fourier_gaussian(np.fft.rfft(x, N), sig, n=N, axis=0)
    yr = np.asarray(np.fft.irfft(Yr, N)).tolist()
    Yc = ndi.fourier_gaussian(np.fft.fft(x), sig)
    yc = np.asarray(np.fft.ifft(Yc)).tolist()
    for i in range(N):
        assert abs(yr[i].real - yc[i].real) < 1e-10
        assert abs(yr[i].imag) < 1e-10
    # hand kernel: upper half uses the mirrored (rfftfreq) frequencies
    h = []
    for m in range(N):
        acc = 0.0 + 0.0j
        for k in range(N):
            f = (k / N) if k <= N // 2 else ((N - k) / N)
            g = math.exp(-2.0 * math.pi**2 * sig**2 * f * f)
            acc += g * complex(math.cos(2 * math.pi * m * k / N),
                               math.sin(2 * math.pi * m * k / N))
        h.append(acc / N)
    expected = [sum(h[(i - m) % N].real * xr[m] for m in range(N))
                for i in range(N)]
    for i in range(N):
        assert abs(yr[i].real - expected[i]) < 1e-9
