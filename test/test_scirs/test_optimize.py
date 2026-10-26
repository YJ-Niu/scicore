"""scicore.optimize — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.optimize as sopt

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_minimize_scalar_bounded():
    res = sopt.minimize_scalar(lambda x: (x - 2) ** 2, bounds=(0, 5))
    assert res.success
    assert np.isclose(res.x[0], 2.0, atol=1e-6)


def test_newton_quadratic():
    assert np.isclose(sopt.newton(lambda x: x**2 - 4, 3.0), 2.0, atol=1e-10)


def test_linear_sum_assignment_identity():
    row_ind, col_ind = sopt.linear_sum_assignment(np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]))
    assert list(row_ind) == [0, 1, 2]
    assert list(col_ind) == [0, 1, 2]


# ---------------- not yet ported / broken ----------------

def test_minimize_bfgs():
    res = sopt.minimize(lambda x: (x[0] - 1) ** 2 + x[1] ** 2, [0.0, 0.0])
    assert np.allclose(res.x, [1.0, 0.0], atol=1e-6)


def test_root_scalar_brentq():
    res = sopt.root_scalar(lambda x: x**2 - 4, bracket=[0, 3])
    assert np.isclose(res.root, 2.0, atol=1e-8)


def test_root_hybr():
    res = sopt.root(lambda x: x**2 - 4, [1.0, 1.0])
    assert np.allclose(res.x, [2.0], atol=1e-8)


@pytest.mark.xfail(reason="HiGHS solver not yet ported (_highspy/_core is a stub)")
def test_linprog():
    res = sopt.linprog(c=[-1, -1], A_ub=[[1, 1]], b_ub=[1])
    assert res.success
    assert np.allclose(res.x, [0.5, 0.5])


def test_curve_fit_exponential():
    t = np.arange(5)
    y = [5.0, 3.0, 1.0, 0.4, 0.15]
    popt, _ = sopt.curve_fit(lambda t, a, b: a * np.exp(-b * t), t, y)
    assert np.isclose(popt[0], 5.0, atol=0.5)
    assert np.isclose(popt[1], 0.5, atol=0.3)


def test_leastsq():
    out = sopt.leastsq(lambda p: [p[0] - 1.0, p[1] - 2.0], [0, 0])
    assert np.allclose(out[0], [1.0, 2.0], atol=1e-8)


def test_nnls():
    x, _ = sopt.nnls(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]))
    assert np.allclose(x, [1.0, 2.0])


@pytest.mark.xfail(reason="TypeError: ndarray cannot be interpreted as an integer")
def test_dual_annealing():
    res = sopt.dual_annealing(lambda x: (x[0] - 1) ** 2, [(-2.0, 2.0)])
    assert np.isclose(res.x[0], 1.0, atol=0.2)
