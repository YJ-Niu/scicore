"""scicore.integrate — functional tests against analytical references."""
import math

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


# ---------------- dopri5 / dop853 (integrate._dop) ----------------

def test_dopri5_harmonic_oscillator():
    # y'' = -y, y(0) = [1, 0]  =>  y(t) = [cos t, sin t]
    o = sint.ode(lambda t, y: [-y[1], y[0]])
    o.set_integrator('dopri5', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0, 0.0], 0.0)
    y = o.integrate(1.0)
    assert np.allclose(np.asarray(y).tolist(),
                       [np.cos(1.0), np.sin(1.0)], atol=1e-7)
    assert o.successful()


def test_dop853_exponential_decay():
    # y' = -2y, y(0) = 1  =>  y(0.5) = e^-1
    o = sint.ode(lambda t, y: [-2.0 * y[0]])
    o.set_integrator('dop853', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0], 0.0)
    y = o.integrate(0.5)
    assert np.allclose(np.asarray(y).tolist(), [np.exp(-1.0)], atol=1e-7)
    assert o.successful()


def test_dopri5_backward_integration():
    o = sint.ode(lambda t, y: [-y[1], y[0]])
    o.set_integrator('dopri5', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0, 0.0], 0.0)
    y = o.integrate(-1.0)
    assert np.allclose(np.asarray(y).tolist(),
                       [np.cos(1.0), -np.sin(1.0)], atol=1e-7)
    assert o.successful()


def test_dop_nsteps_exhausted():
    # C dop driver: fails with ierr=-2 once more than nsteps steps were
    # taken (the check happens before the step, so nsteps=1 allows two)
    o = sint.ode(lambda t, y: [-y[1], y[0]])
    o.set_integrator('dopri5', rtol=1e-10, atol=1e-12, nsteps=1)
    o.set_initial_value([1.0, 0.0], 0.0)
    with pytest.warns(UserWarning):
        o.integrate(10.0)
    assert not o.successful()
    assert o.get_return_code() == -2


def test_dop_solout():
    seen = []
    o = sint.ode(lambda t, y: [-y[1], y[0]])
    o.set_integrator('dopri5', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0, 0.0], 0.0)
    o.set_solout(lambda t, yy: seen.append((t, list(np.asarray(yy)))))
    y = o.integrate(1.0)
    # solout fires once at t0 and after every accepted step
    assert len(seen) >= 2
    assert seen[0][0] == 0.0
    t_last, y_last = seen[-1]
    assert abs(t_last - 1.0) < 1e-12
    assert np.allclose(y_last, [np.cos(1.0), np.sin(1.0)], atol=1e-7)
    assert o.successful()


def test_dop_solout_stop():
    # returning -1 from solout stops the integration with istate=2
    stops = []
    o = sint.ode(lambda t, y: [-y[1], y[0]])
    o.set_integrator('dopri5', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0, 0.0], 0.0)
    o.set_solout(lambda t, yy: stops.append(t) or -1)
    o.integrate(1.0)
    assert len(stops) == 1
    assert o.get_return_code() == 2


def test_dop_f_params():
    # y' = -k y with k from set_f_params
    o = sint.ode(lambda t, y, k: [-k * y[0]])
    o.set_integrator('dop853', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0], 0.0)
    o.set_f_params(3.0)
    y = o.integrate(1.0)
    assert np.allclose(np.asarray(y).tolist(), [np.exp(-3.0)], atol=1e-7)
    assert o.successful()


def test_dop853_undamped_frequency():
    # y' = [y1, -0.9 y0] => undamped oscillator with w = sqrt(0.9)
    o = sint.ode(lambda t, y: [y[1], -0.9 * y[0]])
    o.set_integrator('dop853', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0, 0.0], 0.0)
    y = o.integrate(2.0)
    w = (0.9) ** 0.5
    expected = [np.cos(2.0 * w), -w * np.sin(2.0 * w)]
    assert np.allclose(np.asarray(y).tolist(), expected, atol=1e-6)
    assert o.successful()


# ---------------- lsoda (integrate._odepack) ----------------

def _rob_f(t, y):
    k1, k2, k3 = 0.04, 1e4, 3e3
    return [
        -k1 * y[0] + k3 * y[1] * y[2],
        k1 * y[0] - k2 * y[1] * y[1] - k3 * y[1] * y[2],
        k2 * y[1] * y[1],
    ]


# scipy.integrate.lsoda reference at t=10, rtol=1e-10, atol=1e-12
_ROB_T10 = [9.647295043727e-01, 3.564128693693e-04, 3.491408275789e-02]


def test_lsoda_harmonic_oscillator():
    # non-stiff; default integrator of ode() is lsoda
    o = sint.ode(lambda t, y: [-y[1], y[0]])
    o.set_integrator('lsoda', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0, 0.0], 0.0)
    y = o.integrate(2.0 * np.pi)
    assert np.allclose(np.asarray(y).tolist(), [1.0, 0.0], atol=1e-7)
    assert o.successful()


def test_lsoda_user_jacobian():
    o = sint.ode(
        lambda t, y: [-y[1], y[0]],
        lambda t, y: [[0.0, -1.0], [1.0, 0.0]],
    )
    o.set_integrator('lsoda', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0, 0.0], 0.0)
    y = o.integrate(2.0 * np.pi)
    assert np.allclose(np.asarray(y).tolist(), [1.0, 0.0], atol=1e-7)
    assert o.successful()


def test_lsoda_robertson_stiff_fd_jac():
    # stiff; switches Adams -> BDF internally (FD Jacobian, jt=2)
    o = sint.ode(_rob_f)
    o.set_integrator('lsoda', rtol=1e-10, atol=1e-12, nsteps=5000)
    o.set_initial_value([1.0, 0.0, 0.0], 0.0)
    y = o.integrate(10.0)
    assert np.allclose(np.asarray(y).tolist(), _ROB_T10, atol=1e-8)
    assert o.successful()


def test_lsoda_robertson_stiff_user_jac():
    # stiff with user Jacobian (jt=1)
    k1, k2, k3 = 0.04, 1e4, 3e3

    def jac(t, y):
        return [
            [-k1, k3 * y[2], k3 * y[1]],
            [k1, -2 * k2 * y[1] - k3 * y[2], -k3 * y[1]],
            [0.0, 2 * k2 * y[1], 0.0],
        ]

    o = sint.ode(_rob_f, jac)
    o.set_integrator('lsoda', rtol=1e-10, atol=1e-12, nsteps=5000)
    o.set_initial_value([1.0, 0.0, 0.0], 0.0)
    y = o.integrate(10.0)
    assert np.allclose(np.asarray(y).tolist(), _ROB_T10, atol=1e-8)
    assert o.successful()


def test_lsoda_continuation():
    # two consecutive integrate() calls must continue state (istate 1 -> 2)
    o = sint.ode(lambda t, y: [-y[1], y[0]])
    o.set_integrator('lsoda', rtol=1e-10, atol=1e-12)
    o.set_initial_value([1.0, 0.0], 0.0)
    o.integrate(np.pi / 2)
    y = o.integrate(np.pi)
    # y(t) = [cos t, sin t]  =>  y(pi) = [-1, 0]
    assert np.allclose(np.asarray(y).tolist(), [-1.0, 0.0], atol=1e-7)
    assert o.successful()


# ---------------- banded Jacobian (jt=4/5) + step/relax (itask=2/3) ----------------

_N_HEAT = 30
_H = 1.0 / (_N_HEAT + 1)
_H2 = _H * _H


def _heat_f(t, y):
    # 1D heat equation, _N_HEAT interior points, zero Dirichlet boundaries
    y = [0.0] + list(y) + [0.0]
    return [(y[i + 1] - 2.0 * y[i] + y[i - 1]) / _H2 for i in range(1, _N_HEAT + 1)]


def _heat_jac(t, y):
    # tridiagonal Jacobian in compressed banded form (lband+uband+1, n) = (3, n)
    J = [[0.0] * _N_HEAT for _ in range(3)]
    for i in range(_N_HEAT):
        J[1][i] = -2.0 / _H2
    for i in range(1, _N_HEAT):
        J[2][i - 1] = 1.0 / _H2
    for i in range(_N_HEAT - 1):
        J[0][i] = 1.0 / _H2
    return J


def _heat_y0():
    return [math.sin(math.pi * (i + 1) * _H) for i in range(_N_HEAT)]


def _heat_exact(t):
    # exact semi-discrete solution: the sine vector is the k=1 eigenvector of
    # the tridiagonal Laplacian, eigenvalue (-2 + 2cos(pi/(N+1)))/h^2
    lam1 = (-2.0 + 2.0 * math.cos(math.pi / (_N_HEAT + 1))) / _H2
    c = math.exp(lam1 * t)
    return [c * math.sin(math.pi * (i + 1) * _H) for i in range(_N_HEAT)]


def test_lsoda_banded_user_jac():
    # stiff tridiagonal heat equation, user banded Jacobian (jt=4), BDF mode
    o = sint.ode(_heat_f, _heat_jac)
    o.set_integrator('lsoda', lband=1, uband=1, with_jacobian=True,
                     rtol=1e-10, atol=1e-12, nsteps=5000)
    o.set_initial_value(_heat_y0(), 0.0)
    y = o.integrate(0.5)
    assert np.allclose(np.asarray(y).tolist(), _heat_exact(0.5), atol=1e-6)
    assert o.successful()


def test_lsoda_banded_fd_jac():
    # same banded problem, internally generated banded Jacobian (jt=5)
    o = sint.ode(_heat_f)
    o.set_integrator('lsoda', lband=1, uband=1, rtol=1e-10, atol=1e-12,
                     nsteps=5000)
    o.set_initial_value(_heat_y0(), 0.0)
    y = o.integrate(0.5)
    assert np.allclose(np.asarray(y).tolist(), _heat_exact(0.5), atol=1e-6)
    assert o.successful()


def test_lsoda_banded_continuation():
    # banded problem continued over three integrate() calls (istate 1 -> 2)
    o = sint.ode(_heat_f, _heat_jac)
    o.set_integrator('lsoda', lband=1, uband=1, with_jacobian=True,
                     rtol=1e-10, atol=1e-12, nsteps=5000)
    o.set_initial_value(_heat_y0(), 0.0)
    y = o.integrate(0.15)
    y = o.integrate(0.30)
    y = o.integrate(0.45)
    assert np.allclose(np.asarray(y).tolist(), _heat_exact(0.45), atol=1e-6)
    assert o.successful()


def test_lsoda_lowlevel_step():
    # itask=2 single-step mode: a few low-level steps stay on the orbit
    from scicore.integrate._odepack import lsoda as lsoda_low

    n = 2
    lrw = max(20 + 16 * n, 22 + 9 * n + n * n)
    liw = 20 + n
    rwork = np.zeros((lrw,), float)
    iwork = np.zeros((liw,), dtype=np.int32)
    iwork[5] = 10000
    iwork[7] = 12
    iwork[8] = 5
    sd = np.zeros(240, float)
    si = np.zeros(48, dtype=np.int32)
    f = lambda t, y: [-y[1], y[0]]
    y = np.array([1.0, 0.0], float)
    t, istate = 0.0, 1
    for _ in range(4):
        y, t, istate = lsoda_low(f, y, t, 1.0, 1e-10, 1e-12, 2, istate,
                                 rwork, iwork, None, 2, (), 1, (), sd, si)
    assert istate == 2
    assert 0.0 < t <= 1.0
    assert abs(float(y[0]) - math.cos(t)) < 1e-6
    assert abs(float(y[1]) - math.sin(t)) < 1e-6


def test_lsoda_lowlevel_relax():
    # itask=3 relax: from t=0.5 relax to t=1.0, lands at t >= 1.0 on the orbit
    from scicore.integrate._odepack import lsoda as lsoda_low

    n = 2
    lrw = max(20 + 16 * n, 22 + 9 * n + n * n)
    liw = 20 + n
    rwork = np.zeros((lrw,), float)
    iwork = np.zeros((liw,), dtype=np.int32)
    iwork[5] = 10000
    iwork[7] = 12
    iwork[8] = 5
    sd = np.zeros(240, float)
    si = np.zeros(48, dtype=np.int32)
    f = lambda t, y: [-y[1], y[0]]
    y = np.array([1.0, 0.0], float)
    t, istate = 0.0, 1
    y, t, istate = lsoda_low(f, y, t, 0.5, 1e-10, 1e-12, 1, istate,
                             rwork, iwork, None, 2, (), 1, (), sd, si)
    assert istate == 2 and abs(t - 0.5) < 1e-10
    y, t, istate = lsoda_low(f, y, t, 1.0, 1e-10, 1e-12, 3, 2,
                             rwork, iwork, None, 2, (), 1, (), sd, si)
    assert istate == 2
    assert t >= 1.0
    assert abs(float(y[0]) - math.cos(t)) < 1e-6
    assert abs(float(y[1]) - math.sin(t)) < 1e-6


# ---------------- not yet ported / broken ----------------

def test_quad_polynomial():
    val, _ = sint.quad(lambda x: x**2, 0, 1)
    assert np.isclose(val, 1.0 / 3.0, rtol=1e-10)


def test_nquad():
    val, _ = sint.nquad(lambda x, y: x + y, [[0, 1], [0, 1]])
    assert np.isclose(val, 1.0, rtol=1e-10)


def test_simpson():
    # Simpson on y=[1,2,1], x=[0,1,2] = (1/3)(1 + 4*2 + 1) = 10/3
    # （对二次函数精确；旧注释误写 1/6 系数且 10/6≠1.5）
    assert np.isclose(sint.simpson(np.array([1.0, 2.0, 1.0]), x=np.array([0.0, 1.0, 2.0])), 10.0 / 3.0)


def test_odeint():
    out = sint.odeint(lambda y, t: [-y[1], y[0]], [1.0, 0.0], np.array([0.0, 1.0]))
    assert np.allclose(out[-1], [np.cos(1.0), np.sin(1.0)], atol=1e-6)


def test_romb():
    # Romberg on y=[1,2,1], dx=1: exact for the quadratic => 10/3
    assert np.isclose(sint.romb(np.array([1.0, 2.0, 1.0]), dx=1.0), 10.0 / 3.0, rtol=1e-10)
