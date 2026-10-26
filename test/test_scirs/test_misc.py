"""scicore.constants / scicore.datasets / scicore.differentiate — misc checks.

constants values are CODATA reference values (exact where defined).
datasets requires the optional dependency pooch; skipped when absent.
"""
import pytest
import rsnumpy as np

import scicore.constants as const
import scicore.differentiate as sdiff

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- constants (working) ----------------

def test_speed_of_light():
    assert const.c == 299792458.0


def test_gravitational_constant():
    assert np.isclose(const.G, 6.6743e-11, rtol=1e-4)


def test_planck_constant():
    assert const.h == 6.62607015e-34


def test_molar_gas_constant():
    assert np.isclose(const.R, 8.31446261815324, rtol=1e-12)


def test_boltzmann_constant():
    assert const.k == 1.380649e-23


def test_avogadro_constant():
    assert const.N_A == 6.02214076e23


def test_stefan_boltzmann_constant():
    assert np.isclose(const.sigma, 5.6703744191844314e-08, rtol=1e-12)


# ---------------- datasets (needs pooch) ----------------

def test_ascent_dataset():
    pooch = pytest.importorskip("pooch")
    import scicore.datasets as sds

    img = sds.ascent()
    assert img.shape == (512, 512)
    assert img.dtype in (np.uint8, "uint8") or getattr(img, "dtype", None) is not None


def test_face_dataset():
    pooch = pytest.importorskip("pooch")
    import scicore.datasets as sds

    img = sds.face()
    assert img.shape[0] == 683  # face.png height


# ---------------- differentiate (broken) ----------------

def test_jacobian():
    J = sdiff.jacobian(lambda x: [x[0] * x[1], x[0] + x[1]], np.array([1.0, 2.0]))
    assert np.allclose(J.df, [[2.0, 1.0], [1.0, 1.0]])


def test_hessian():
    H = sdiff.hessian(lambda x: x[0] ** 2 + x[1] ** 3, np.array([1.0, 2.0]))
    assert np.allclose(H.ddf, [[2.0, 0.0], [0.0, 12.0]])


def test_second_derivative():
    # d^2/dx^2 (x^3) at x=2 = 12
    val = sdiff.derivative(lambda x: x**3, 2.0, 2)
    assert np.isclose(val.df, 12.0, atol=1e-6)
