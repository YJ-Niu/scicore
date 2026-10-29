"""cProfile breakdown of the scicore hot paths.

For each case: total self-time, user-callback self-time fraction, and the
top internal functions. This shows how much of the scicore runtime is
pure-Python inner loop (Rust-movable) vs user callback (stays Python).
"""
import cProfile
import io
import pstats
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from bench1_walltime import (  # noqa: E402
    ksum1d, rb2_jac, rb2_res, run_ode, slsqp_grad, slsqp_obj,
    y_heat, y_osc, heat_jac_b, heat_rhs, fsys, osc_rhs,
)
import rsnumpy as np  # noqa: E402
import scicore.integrate as si  # noqa: E402
import scicore.optimize as so  # noqa: E402
import scicore.ndimage as snd  # noqa: E402

USER_FUNCS = {
    'osc_rhs', 'heat_rhs', 'heat_jac_b', 'fsys', 'rb2_res', 'rb2_jac',
    'slsqp_obj', 'slsqp_grad', 'ksum', '<lambda>',
}


def profile_case(label, user_names, fn):
    pr = cProfile.Profile()
    pr.enable()
    try:
        fn()
    finally:
        pr.disable()

    ps = pstats.Stats(pr)
    total = 0.0
    user = 0.0
    rows = []
    for (fn_, ln, name), (cc, nc, tt, ct, _callers) in ps.stats.items():
        total += tt
        if name in user_names:
            user += tt
        rows.append((tt, name, fn_.rsplit('/', 1)[-1], ln, cc))
    rows.sort(reverse=True)
    pct = 100.0 * user / total if total else 0.0
    print(f'== {label}: total_self={total:.3f}s  user_cb={user:.3f}s '
          f'({pct:.0f}%)')
    for tt, name, f, ln, cc in rows[:18]:
        print(f'   {tt:8.3f}  {name:38s} {f}:{ln}  calls={cc}')
    print()


def main():
    profile_case('LSODA stiff N=60 banded heat (worst case)', USER_FUNCS,
                 lambda: run_ode(si, heat_rhs, y_heat(60), 0.5,
                                 jac=heat_jac_b, banded=True,
                                 rtol=1e-6, atol=1e-9))

    profile_case('LSODA nonstiff n=20 osc', USER_FUNCS,
                 lambda: run_ode(si, osc_rhs, y_osc(20), 10.0,
                                 rtol=1e-10, atol=1e-10))

    def case_dop853():
        from bench1_walltime import osc_rhs
        return si.solve_ivp(osc_rhs, [0.0, 10.0], y_osc(50),
                            method='DOP853', rtol=1e-8, atol=1e-10)

    profile_case('DOP853 n=50 osc', USER_FUNCS, case_dop853)

    x0f = [3.0] * 50
    profile_case('fsolve n=50 num-jac', USER_FUNCS,
                 lambda: so.fsolve(fsys, x0f))

    x0r = [0.5] * 30
    profile_case('leastsq n=30 Dfun', USER_FUNCS,
                 lambda: so.leastsq(rb2_res, x0r, Dfun=rb2_jac))

    x0s = [0.5] * 30
    profile_case('SLSQP n=30 QP', USER_FUNCS,
                 lambda: so.minimize(slsqp_obj, x0s, jac=slsqp_grad,
                                     method='SLSQP',
                                     bounds=[(0.0, 2.0)] * 30,
                                     constraints=[{'type': 'eq',
                                                   'fun': lambda x: sum(x) - 15.0,
                                                   'jac': lambda x: [1.0] * 30}]))

    g1 = [[0.5 * ((i * 1000 + j) % 17) for j in range(1000)]
          for i in range(100)]
    profile_case('ndimage generic_filter1d 1e5', USER_FUNCS,
                 lambda: snd.generic_filter1d(g1, ksum1d, 5, axis=1,
                                              extra_arguments=(5,)))

    img = [[0.0 if (i * 7 + j * 13) % 11 < 3 else 1.0 for j in range(300)]
           for i in range(300)]
    profile_case('ndimage dt_bf 300x300', USER_FUNCS,
                 lambda: snd.distance_transform_bf(img))


def osc_rhs_wrapper():
    from bench1_walltime import osc_rhs
    return osc_rhs


if __name__ == '__main__':
    main()
