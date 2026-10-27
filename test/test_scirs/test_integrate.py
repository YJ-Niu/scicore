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
