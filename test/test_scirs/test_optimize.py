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


def test_linear_sum_assignment_square():
    # min total cost is 5 via row0->col1, row1->col0, row2->col2
    row_ind, col_ind = sopt.linear_sum_assignment(np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]))
    assert list(row_ind) == [0, 1, 2]
    assert list(col_ind) == [1, 0, 2]


def test_linear_sum_assignment_rect():
    # n < m: all rows assigned
    row_ind, col_ind = sopt.linear_sum_assignment(np.array([[7.0, 3.0], [4.0, 2.0]]))
    assert list(row_ind) == [0, 1]
    assert list(col_ind) == [1, 0]
    # n > m: all columns assigned, min cost is col0->row2, col1->row1
    row_ind, col_ind = sopt.linear_sum_assignment(np.array([[7.0, 3.0], [4.0, 2.0], [1.0, 1.0]]))
    assert list(col_ind) == [0, 1]
    assert list(row_ind) == [2, 1]


def test_linear_sum_assignment_maximize():
    # max total cost is 11 via row0->col0, row1->col2, row2->col1
    row_ind, col_ind = sopt.linear_sum_assignment(np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]), maximize=True)
    assert list(row_ind) == [0, 1, 2]
    assert list(col_ind) == [0, 2, 1]


def test_isotonic_regression():
    y = [1.5, 1.0, 4.0, 6.0, 5.7, 5.0, 7.8, 9.0, 7.5, 9.5, 9.0]
    res = sopt.isotonic_regression(y)
    assert np.allclose(res.x, [1.25, 1.25, 4.0, 5.56666667, 5.56666667,
                               5.56666667, 7.8, 8.25, 8.25, 9.25, 9.25], atol=1e-6)
    assert list(res.blocks) == [0, 2, 3, 6, 7, 9, 11]
    assert np.allclose(res.weights, [2.0, 1.0, 3.0, 1.0, 2.0, 2.0])


def test_isotonic_regression_weights():
    # weighted pool of [2, 1] = (2*2 + 1*1)/3 = 5/3; point 1.0 stays separate
    res = sopt.isotonic_regression([1.0, 2.0, 1.0], weights=[1.0, 2.0, 1.0])
    assert np.allclose(res.x, [1.0, 5.0 / 3.0, 5.0 / 3.0])
    assert list(res.blocks) == [0, 1, 3]


def test_isotonic_regression_decreasing():
    res = sopt.isotonic_regression([1.0, 3.0, 2.0], increasing=False)
    assert np.allclose(res.x, [2.0, 2.0, 2.0])
    assert list(res.blocks) == [0, 2, 3]


def _quad2():
    # f = 0.5 (4 x0^2 + 9 x1^2), minimum at the origin
    def fun(x):
        x = [float(t) for t in np.asarray(x).tolist()]
        return 0.5 * (4.0 * x[0] ** 2 + 9.0 * x[1] ** 2)

    def jac(x):
        x = [float(t) for t in np.asarray(x).tolist()]
        return np.array([4.0 * x[0], 9.0 * x[1]])

    def hessp(x, p):
        p = [float(t) for t in np.asarray(p).tolist()]
        return np.array([4.0 * p[0], 9.0 * p[1]])

    def hess(x):
        return np.array([[4.0, 0.0], [0.0, 9.0]])

    return fun, jac, hessp, hess


def test_minimize_trust_krylov():
    fun, jac, hessp, hess = _quad2()
    # Cauchy-point subproblem (hessp only)
    res = sopt.minimize(fun, [3.0, -2.0], method='trust-krylov',
                        jac=jac, hessp=hessp)
    assert res.success
    assert np.allclose(res.x, [0.0, 0.0], atol=1e-3)
    # dogleg subproblem (full Hessian): quadratically fast
    res = sopt.minimize(fun, [3.0, -2.0], method='trust-krylov',
                        jac=jac, hess=hess)
    assert res.success
    assert np.allclose(res.x, [0.0, 0.0], atol=1e-5)
    assert res.nit < 10


def test_minimize_trust_ncg():
    fun, jac, hessp, hess = _quad2()
    for extra in (dict(hessp=hessp), dict(hess=hess)):
        res = sopt.minimize(fun, [3.0, -2.0], method='trust-ncg',
                            jac=jac, **extra)
        assert res.success
        assert np.allclose(res.x, [0.0, 0.0], atol=1e-3)


def test_minimize_dogleg():
    fun, jac, hessp, hess = _quad2()
    res = sopt.minimize(fun, [3.0, -2.0], method='dogleg',
                        jac=jac, hess=hess)
    assert res.success
    assert np.allclose(res.x, [0.0, 0.0], atol=1e-5)
    assert res.nit < 10


def test_quadratic_assignment():
    # brute-force optimum over all 24 permutations is 4.0
    a = [[0.0, 1.0, 1.0, 0.0],
         [1.0, 0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0, 1.0],
         [0.0, 1.0, 1.0, 0.0]]
    b = [[0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0]]
    for method in ('faq', '2opt'):
        res = sopt.quadratic_assignment(a, b, method=method)
        perm = [int(t) for t in np.asarray(res.col_ind).tolist()]
        assert sorted(perm) == [0, 1, 2, 3]
        assert np.isclose(res.fun, 4.0)
    # seeds must be respected by the solution
    res = sopt.quadratic_assignment(a, b,
                                    options={'partial_match': np.array([[0, 0], [1, 1]])})
    perm = [int(t) for t in np.asarray(res.col_ind).tolist()]
    assert perm[0] == 0 and perm[1] == 1


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


def test_linprog():
    # min -x-y s.t. x+y<=1: the optimum -1 is attained on the whole
    # segment x+y=1; a simplex solver returns a vertex, so check the
    # objective and the optimal face rather than a single point
    res = sopt.linprog(c=[-1, -1], A_ub=[[1, 1]], b_ub=[1])
    assert res.success
    assert np.allclose(res.fun, -1.0)
    x = np.asarray(res.x).tolist()
    assert x[0] >= -1e-9 and x[1] >= -1e-9
    assert abs(x[0] + x[1] - 1.0) < 1e-6


def _exponential_fit_reference():
    # scipy 1.18.1 reference for a*exp(-b*t) on t=[0..4],
    # y=[5.0, 3.0, 1.0, 0.4, 0.15]
    popt = [5.137438402744526, 0.7083734270360839]
    pcov = [[0.11643927240789177, 0.014280325328315057],
            [0.014280325328315057, 0.008798068400055115]]
    return popt, pcov


def _assert_pcov_close(pcov, ref, tol=1e-5):
    pc = [[float(v) for v in row] for row in np.asarray(pcov).tolist()]
    for row, row_ref in zip(pc, ref):
        for v, v_ref in zip(row, row_ref):
            assert abs(v - v_ref) <= tol, (v, v_ref)


def test_curve_fit_exponential():
    t = np.arange(5)
    y = [5.0, 3.0, 1.0, 0.4, 0.15]
    popt, pcov = sopt.curve_fit(lambda t, a, b: a * np.exp(-b * t), t, y)
    assert np.isclose(popt[0], 5.0, atol=0.5)
    assert np.isclose(popt[1], 0.5, atol=0.3)
    _assert_pcov_close(pcov, _exponential_fit_reference()[1])


def test_curve_fit_exponential_jac():
    # analytic-Jacobian (lmder) path: same fit and covariance
    t = np.arange(5)
    y = [5.0, 3.0, 1.0, 0.4, 0.15]
    tv = [0.0, 1.0, 2.0, 3.0, 4.0]

    def jac(t, a, b):
        return [[float(np.exp(-b * tv[i])),
                 -tv[i] * a * float(np.exp(-b * tv[i]))] for i in range(5)]

    popt, pcov = sopt.curve_fit(lambda t, a, b: a * np.exp(-b * t), t, y,
                                jac=jac)
    assert np.isclose(popt[0], 5.0, atol=0.5)
    assert np.isclose(popt[1], 0.5, atol=0.3)
    _assert_pcov_close(pcov, _exponential_fit_reference()[1])


def test_leastsq():
    out = sopt.leastsq(lambda p: [p[0] - 1.0, p[1] - 2.0], [0, 0])
    assert np.allclose(out[0], [1.0, 2.0], atol=1e-8)


def test_nnls():
    x, _ = sopt.nnls(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]))
    assert np.allclose(x, [1.0, 2.0])


def test_dual_annealing():
    res = sopt.dual_annealing(lambda x: (x[0] - 1) ** 2, [(-2.0, 2.0)])
    assert np.isclose(res.x[0], 1.0, atol=0.2)


def test_minimize_slsqp_eq_only():
    # min sum (xi-2)^2 s.t. x0+x1+x2=1, x0+x1-x2=0
    res = sopt.minimize(
        lambda x: (x[0] - 2) ** 2 + (x[1] - 2) ** 2 + (x[2] - 2) ** 2,
        [0.0, 0.0, 0.0], method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda x: x[0] + x[1] + x[2] - 1,
             "jac": lambda x: [1.0, 1.0, 1.0]},
            {"type": "eq", "fun": lambda x: x[0] + x[1] - x[2],
             "jac": lambda x: [1.0, 1.0, -1.0]},
        ],
    )
    assert res.success
    assert np.allclose(res.x, [0.25, 0.25, 0.5], atol=1e-6)


def test_minimize_slsqp_ineq_only():
    # min (x0-2)^2 + (x1-2)^2 s.t. 1 - x0 - x1 >= 0 -> [0.5, 0.5]
    res = sopt.minimize(
        lambda x: (x[0] - 2) ** 2 + (x[1] - 2) ** 2,
        [0.0, 0.0], method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda x: 1 - (x[0] + x[1]),
                      "jac": lambda x: [-1.0, -1.0]}],
    )
    assert res.success
    assert np.allclose(res.x, [0.5, 0.5], atol=1e-6)


def test_minimize_slsqp_mixed():
    # min (x0-3)^2 + (x1-3)^2 s.t. x0+x1 <= 4, 1 <= x0 <= 2, x1 >= 0 -> [2, 2]
    res = sopt.minimize(
        lambda x: (x[0] - 3) ** 2 + (x[1] - 3) ** 2,
        [0.0, 0.0], method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda x: 4 - (x[0] + x[1]),
                      "jac": lambda x: [-1.0, -1.0]}],
        bounds=[(1.0, 2.0), (0.0, None)],
    )
    assert res.success
    assert np.allclose(res.x, [2.0, 2.0], atol=1e-6)


def test_minimize_slsqp_eq_bounds():
    # min (x0-4)^2 + (x1-4)^2 s.t. x0+x1 = 3, x >= 0 -> [1.5, 1.5]
    res = sopt.minimize(
        lambda x: (x[0] - 4) ** 2 + (x[1] - 4) ** 2,
        [0.0, 0.0], method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda x: x[0] + x[1] - 3,
                      "jac": lambda x: [1.0, 1.0]}],
        bounds=[(0.0, None), (0.0, None)],
    )
    assert res.success
    assert np.allclose(res.x, [1.5, 1.5], atol=1e-6)
