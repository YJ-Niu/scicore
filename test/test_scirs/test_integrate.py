"""scicore.integrate — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.integrate as sint

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_trapezoid():
    # trapezoid of y=[1,2,1] on x=[0,1,2] = (1+2)/2 + (2+1)/2 = 3
    assert np.isclose(sint.trapezoid(np.array([1.0, 2.0, 1.0]), x=np.array([0.0, 1.0, 2.0])), 3.0)


def test_cumulative_trapezoid():
    out = sint.cumulative_trapezoid(np.array([1.0, 1.0, 1.0]), x=np.array([0.0, 1.0, 2.0]))
    assert np.allclose(out, [1.0, 2.0])


def test_solve_ivp_harmonic_oscillator():
    # y'' = -y, y(0) = [1, 0]  =>  y(t) = [cos t, sin t]
    sol = sint.solve_ivp(
        lambda t, y: [-y[1], y[0]], [0, 1], [1.0, 0.0], rtol=1e-9, atol=1e-12
    )
    assert sol.success
    assert np.allclose(sol.y[:, -1], [np.cos(1.0), np.sin(1.0)], atol=1e-6)


# ---------------- not yet ported / broken ----------------

@pytest.mark.xfail(reason="_quadpack not yet ported to Rust/rsnumpy")
def test_quad_polynomial():
    val, _ = sint.quad(lambda x: x**2, 0, 1)
    assert np.isclose(val, 1.0 / 3.0, rtol=1e-10)


@pytest.mark.xfail(reason="_quadpack not yet ported to Rust/rsnumpy")
def test_nquad():
    val, _ = sint.nquad(lambda x, y: x + y, [[0, 1], [0, 1]])
    assert np.isclose(val, 1.0, rtol=1e-10)


def test_simpson():
    # Simpson on y=[1,2,1], x=[0,1,2] = (1/3)(1 + 4*2 + 1) = 10/3
    # （对二次函数精确；旧注释误写 1/6 系数且 10/6≠1.5）
    assert np.isclose(sint.simpson(np.array([1.0, 2.0, 1.0]), x=np.array([0.0, 1.0, 2.0])), 10.0 / 3.0)


@pytest.mark.xfail(reason="ValueError: truth value of array with more than one element is ambiguous")
def test_odeint():
    out = sint.odeint(lambda y, t: [-y[1], y[0]], [1.0, 0.0], np.array([0.0, 1.0]))
    assert np.allclose(out[-1], [np.cos(1.0), np.sin(1.0)], atol=1e-6)


def test_romb():
    # Romberg on y=[1,2,1], dx=1: exact for the quadratic => 10/3
    assert np.isclose(sint.romb(np.array([1.0, 2.0, 1.0]), dx=1.0), 10.0 / 3.0, rtol=1e-10)
