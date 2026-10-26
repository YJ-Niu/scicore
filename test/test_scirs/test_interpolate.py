"""scicore.interpolate — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.interpolate as sinterp

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_interp1d_linear():
    f = sinterp.interp1d(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    assert np.isclose(f(1.5), 2.5)
    assert np.allclose(f(np.array([0.0, 1.0, 2.0])), [0.0, 1.0, 4.0])


# ---------------- not yet ported / broken ----------------

def test_cubic_spline():
    cs = sinterp.CubicSpline(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    # interpolates the data points
    assert np.allclose(cs(np.array([0.0, 1.0, 2.0])), [0.0, 1.0, 4.0])
    # scipy's n=3 not-a-knot special case builds the parabola through the
    # three points: y = x^2, so cs(0.5) = 0.25
    assert np.isclose(cs(0.5), 0.25, atol=1e-10)


def test_pchip_interpolator():
    p = sinterp.PchipInterpolator(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    assert np.allclose(p(np.array([0.0, 1.0, 2.0])), [0.0, 1.0, 4.0])


def test_rbf():
    rbf = sinterp.Rbf(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    # default multiquadric with epsilon=(prod(edges)/N)^(1/1)=0.5:
    # A = [[1, sqrt(5)], [sqrt(5), 1]], w = A^-1 [0,1] -> w0=sqrt(5)/4,
    # w1=-1/4; rbf(0.5) = sqrt(2)(w0+w1) = (sqrt(10)-sqrt(2))/4
    assert np.isclose(rbf(0.5), (np.sqrt(10.0) - np.sqrt(2.0)) / 4.0,
                      atol=1e-10)


@pytest.mark.xfail(reason="TypeError: Unsupported data type")
def test_rect_bivariate_spline():
    rbs = sinterp.RectBivariateSpline(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([[0.0, 1.0], [1.0, 4.0]])
    )
    assert np.isclose(rbs(0.5, 0.5), 1.5)


@pytest.mark.xfail(reason="TypeError: Unsupported data type")
def test_univariate_spline():
    us = sinterp.UnivariateSpline(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    assert np.allclose(us(np.array([0.0, 1.0, 2.0])), [0.0, 1.0, 4.0], atol=1e-6)
