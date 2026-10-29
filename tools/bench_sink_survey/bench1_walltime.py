"""scicore (rsnumpy) vs scipy (real numpy) wall-time comparison.

Same problems, same process. Measures only wall time; see bench2_profile.py
for the callback-vs-inner-loop breakdown.
"""
import time

import rsnumpy as np
import numpy as npr
import scipy.integrate as spi
import scipy.optimize as spo
import scipy.ndimage as sni
import scicore.integrate as si
import scicore.optimize as so
import scicore.ndimage as snd


# ---------------------------------------------------------------- problems
def osc_rhs(t, y):
    n = len(y)
    dy = [0.0] * n
    dy[0] = -y[1]
    for i in range(1, n - 1):
        dy[i] = y[i - 1] - y[i + 1]
    dy[n - 1] = y[n - 2]
    return dy


def heat_rhs(t, y):
    n = len(y)
    dy = [y[0] - y[1]]
    for i in range(1, n - 1):
        dy.append(y[i - 1] - 2.0 * y[i] + y[i + 1])
    dy.append(y[n - 2] - y[n - 1])
    return dy


def heat_jac_b(t, y):
    n = len(y)
    u = [0.0] * n
    d = [0.0] * n
    l = [0.0] * n
    d[0] = -1.0
    l[0] = 1.0
    for i in range(1, n - 1):
        u[i] = 1.0
        d[i] = -2.0
        l[i] = 1.0
    u[n - 1] = 1.0
    d[n - 1] = -1.0
    return [u, d, l]


def y_osc(n):
    return [0.01 * (i + 1) for i in range(n)]


def y_heat(n):
    return [1.0 if i % 3 == 0 else 0.0 for i in range(n)]


# NOTE: float() wraps work around the rsnumpy getitem quirk (scalar index
# returns a (1,) sub-array), which otherwise makes scicore fsolve/leastsq
# see shape (n,1).  Symmetric for both libraries; realistic user code.
def fsys(x):
    n = len(x)
    f = [0.0] * n
    for i in range(n):
        xi = float(x[i])
        xm = float(x[i - 1]) if i > 0 else 0.0
        xp = float(x[i + 1]) if i < n - 1 else 0.0
        f[i] = xi - 0.5 * (xm + xp) + (xi ** 3) / 10.0
    return f


def rb2_res(x):
    m = len(x) // 2
    r = [0.0] * (2 * m)
    for i in range(m):
        a = float(x[2 * i])
        b = float(x[2 * i + 1])
        r[2 * i] = 100.0 * (b - a * a)
        r[2 * i + 1] = 1.0 - a
    return r


def rb2_jac(x):
    m = len(x) // 2
    J = [[0.0] * (2 * m) for _ in range(2 * m)]
    for i in range(m):
        a = float(x[2 * i])
        b = float(x[2 * i + 1])
        J[2 * i][2 * i] = -200.0 * a * (b - a * a)
        J[2 * i][2 * i + 1] = 100.0
        J[2 * i + 1][2 * i] = -1.0
    return J


def slsqp_obj(x):
    s = 0.0
    for xi in x:
        xi = float(xi)
        s += (xi - 1.0) * (xi - 1.0)
    return s


def slsqp_grad(x):
    return [2.0 * (float(xi) - 1.0) for xi in x]


def ksum1d(input_line, output_line, filter_size):
    n = len(output_line)
    for i in range(n):
        s = 0.0
        for j in range(filter_size):
            s += float(input_line[i + j])
        output_line[i] = s


# ---------------------------------------------------------------- runners
def run_ode(mod, rhs, y0, tend, jac=None, banded=False, **kw):
    o = mod.ode(rhs, jac=jac)
    if banded:
        o.set_integrator('lsoda', with_jacobian=True, lband=1, uband=1,
                         nsteps=200000, **kw)
    else:
        o.set_integrator('lsoda', nsteps=200000, **kw)
    o.set_initial_value(y0, 0.0)
    return o.integrate(tend)


def bench_cases():
    N = 30
    M = 30

    cases = []

    # 1. LSODA non-stiff n=20 (oscillatory, ode old API)
    def c1_scicore():
        return run_ode(si, osc_rhs, y_osc(20), 10.0,
                       rtol=1e-10, atol=1e-10)

    def c1_scipy():
        return run_ode(spi, osc_rhs, npr.array(y_osc(20)), 10.0,
                       rtol=1e-10, atol=1e-10)

    cases.append(('LSODA nonstiff n=20 osc', c1_scicore, c1_scipy, 5))

    # 2. LSODA stiff N=60 banded (heat equation)
    def c2_scicore():
        return run_ode(si, heat_rhs, y_heat(60), 0.5, jac=heat_jac_b,
                       banded=True, rtol=1e-6, atol=1e-9)

    def c2_scipy():
        return run_ode(spi, heat_rhs, npr.array(y_heat(60)), 0.5,
                       jac=heat_jac_b, banded=True,
                       rtol=1e-6, atol=1e-9)

    cases.append(('LSODA stiff N=60 banded heat', c2_scicore, c2_scipy, 3))

    # 3. DOP853 n=50 (solve_ivp)
    def c3_scicore():
        return si.solve_ivp(osc_rhs, [0.0, 10.0], y_osc(50), method='DOP853',
                            rtol=1e-8, atol=1e-10)

    def c3_scipy():
        return spi.solve_ivp(osc_rhs, [0.0, 10.0], npr.array(y_osc(50)),
                             method='DOP853', rtol=1e-8, atol=1e-10)

    cases.append(('DOP853 n=50 osc', c3_scicore, c3_scipy, 5))

    # 4. fsolve n=50 (numerical Jacobian)
    x0f = [3.0] * 50

    def c4_scicore():
        return so.fsolve(fsys, x0f)

    def c4_scipy():
        return spo.fsolve(fsys, npr.array(x0f))

    cases.append(('fsolve n=50 num-jac', c4_scicore, c4_scipy, 5))

    # 5. leastsq N=30 (Rosenbrock pair, Dfun)
    x0r = [0.5] * 2 * (M // 2)

    def c5_scicore():
        return so.leastsq(rb2_res, x0r, Dfun=rb2_jac)

    def c5_scipy():
        return spo.leastsq(rb2_res, npr.array(x0r), Dfun=rb2_jac)

    cases.append(('leastsq n=30 Dfun', c5_scicore, c5_scipy, 5))

    # 6. SLSQP n=30 quadratic + linear equality + bounds
    x0s = [0.5] * M

    def c6_scicore():
        return so.minimize(slsqp_obj, x0s, jac=slsqp_grad, method='SLSQP',
                           bounds=[(0.0, 2.0)] * M,
                           constraints=[{'type': 'eq',
                                         'fun': lambda x: sum(x) - 15.0,
                                         'jac': lambda x: [1.0] * M}])

    def c6_scipy():
        return spo.minimize(slsqp_obj, npr.array(x0s), jac=slsqp_grad,
                            method='SLSQP', bounds=[(0.0, 2.0)] * M,
                            constraints=[{'type': 'eq',
                                          'fun': lambda x: sum(x) - 15.0,
                                          'jac': lambda x: [1.0] * M}])

    cases.append(('SLSQP n=30 QP', c6_scicore, c6_scipy, 5))

    # 7. ndimage generic_filter1d 1e5 (100x1000, window 5, axis=1)
    g1 = [[0.5 * ((i * 1000 + j) % 17) for j in range(1000)]
          for i in range(100)]

    def c7_scicore():
        return snd.generic_filter1d(g1, ksum1d, 5, axis=1,
                                    extra_arguments=(5,))

    def c7_scipy():
        return sni.generic_filter1d(npr.array(g1), ksum1d, 5, axis=1,
                                    extra_arguments=(5,))

    cases.append(('ndimage generic_filter1d 1e5', c7_scicore, c7_scipy, 5))

    # 8. ndimage distance_transform_bf 300x300
    img = [[0.0 if (i * 7 + j * 13) % 11 < 3 else 1.0 for j in range(300)]
           for i in range(300)]

    def c8_scicore():
        return snd.distance_transform_bf(img)

    def c8_scipy():
        return sni.distance_transform_bf(npr.array(img))

    cases.append(('ndimage dt_bf 300x300', c8_scicore, c8_scipy, 5))

    return cases


def timeit(fn, reps):
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        if t1 - t0 < best:
            best = t1 - t0
    return best


def main():
    print(f"{'problem':34s} {'scicore':>10s} {'scipy':>10s} {'ratio':>7s}")
    for name, fsc, fsp, reps in bench_cases():
        tsc = timeit(fsc, reps)
        tsp = timeit(fsp, reps)
        print(f"{name:34s} {tsc:10.4f} {tsp:10.4f} {tsc / tsp:7.1f}x")


if __name__ == '__main__':
    main()
