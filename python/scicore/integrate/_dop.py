"""Pure-Python replacement for the deleted C ``_dop`` extension.

Implements the DOPRI5 (5(4)) and DOP853 (8(5)) explicit Runge-Kutta
integrators with the interface of scipy's C ``dop`` module
(``scipy/integrate/src/dop.c``): the right-hand side is called as
``func(t, y, *f_params)``, the ``solout`` callback (when ``iout`` is
nonzero) is invoked once before the first step and after every
accepted step, and the returned ``istate`` follows the C ``ierr``
semantics::

    1   computation successful
    2   computation successful (interrupted by solout)
    -1  input is not consistent
    -2  larger nsteps is needed
    -3  step size becomes too small
    -4  problem is probably stiff (interrupted)
    -11..-16  inconsistent iwork/work parameters
"""

from __future__ import annotations

import math

import rsnumpy as np

__all__ = ['dop853', 'dopri5', 'dopri853']

_URound = 2.220446049250313e-16


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _tol_list(tol, n):
    """Scalar or array tolerance -> list of n floats."""
    if isinstance(tol, np.ndarray):
        return _flat_result(tol, n)
    return [float(tol)] * n


def _scalar_value(v):
    """Python/np scalar, possibly wrapped in 0-dim or (1,) sub-arrays."""
    try:
        return float(v)
    except (TypeError, ValueError):
        t = v.tolist() if hasattr(v, 'tolist') else v
        while isinstance(t, (list, tuple)):
            t = t[0]
        return float(t)


def _flat_result(r, n):
    """func() return (scalar, list, or array) -> flat list of n floats.

    rsnumpy ``tolist`` nesting is not stable (0-dim and (1,) sub-
    arrays add extra list levels), so flatten recursively."""
    a = np.asarray(r, dtype=np.float64)

    def _into(v, out):
        if isinstance(v, (list, tuple)):
            for item in v:
                _into(item, out)
        else:
            out.append(_scalar_value(v))

    vals = []
    _into(a.tolist(), vals)
    if len(vals) != n:
        raise ValueError('func returned %d values, expected %d'
                         % (len(vals), n))
    return vals


def _hinit(n, fcn, x, y, posneg, f0, hmax, atol, rtol, iord):
    """First step-size guess (scipy dop.c hinit / hinit853, identical
    except for iord)."""
    dnf = 0.0
    dny = 0.0
    for i in range(n):
        sk = atol[i] + rtol[i] * abs(y[i])
        dnf += (f0[i] / sk) ** 2
        dny += (y[i] / sk) ** 2
    if dnf <= 1e-10 or dny <= 1e-10:
        h = 1e-6
    else:
        h = math.sqrt(dny / dnf) * 0.01
    h = min(h, hmax)
    h = math.copysign(h, posneg)
    if h == 0.0:
        return h

    y1 = [y[i] + h * f0[i] for i in range(n)]
    f1 = fcn(x + h, y1)

    der2 = 0.0
    for i in range(n):
        sk = atol[i] + rtol[i] * abs(y[i])
        der2 += ((f1[i] - f0[i]) / sk) ** 2
    der2 = math.sqrt(der2) / h

    der12 = max(abs(der2), math.sqrt(dnf))
    if der12 <= 1e-15:
        h1 = max(1e-6, abs(h) * 1e-3)
    else:
        h1 = (0.01 / der12) ** (1.0 / iord)

    h = min(min(100.0 * abs(h), h1), hmax)
    return math.copysign(h, posneg)


def _driver_input(n, workl, iworkl, nsteps, x, xend, beta_default,
                  fac1_default, fac2_default):
    """Validate iwork/work parameters exactly as the C driver does.

    Returns (nmax, nstiff, uround, safe, fac1, fac2, beta, hmax, h) on
    success, or an error istate in -11..-16."""
    if nsteps < 0:
        return -11
    nmax = nsteps if nsteps != 0 else 100000

    meth = iworkl[1]
    if meth == 0:
        meth = 1
    if meth <= 0 or meth >= 4:
        return -12

    nstiff = iworkl[3]
    if nstiff == 0:
        nstiff = 1000
    if nstiff < 0:
        nstiff = nmax + 10

    nrdens = iworkl[4]
    if nrdens < 0 or nrdens > n:
        return -13

    uround = workl[0]
    if uround == 0.0:
        uround = _URound
    elif uround <= 1e-35 or uround >= 1.0:
        return -14

    safe = workl[1]
    if safe == 0.0:
        safe = 0.9
    elif safe <= 1e-4 or safe >= 1.0:
        return -15

    fac1 = fac1_default if workl[2] == 0.0 else workl[2]
    fac2 = fac2_default if workl[3] == 0.0 else workl[3]

    if workl[4] <= 0.0:
        beta = 0.0 if workl[4] < 0.0 else beta_default
    else:
        beta = workl[4]
        if beta > 0.2:
            return -16

    hmax = (xend - x) if workl[5] == 0.0 else workl[5]
    h = workl[6]
    return nmax, nstiff, uround, safe, fac1, fac2, beta, hmax, h


# ---------------------------------------------------------------------------
# DOPRI5
# ---------------------------------------------------------------------------

def dopri5(func, t0, y0, t1, rtol, atol, solout, iout, work, iwork,
           nsteps, verbosity, f_params):
    """DOPRI5 (Dormand-Prince 5(4)) integrator.

    Follows the scipy C ``dopri5`` driver: returns ``(x, y, istate)``
    with ``x`` the final time and ``y`` the state at ``x``.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    if y0.ndim != 1 or y0.size == 0:
        return float(t0), np.array([], dtype=np.float64), -1
    n = y0.size
    y = [float(v) for v in y0.tolist()]
    x = float(t0)
    xend = float(t1)

    workl = [float(v) for v in np.asarray(work, dtype=np.float64).tolist()]
    iworkl = [int(v) for v in np.asarray(iwork, dtype=np.int64).tolist()]

    inp = _driver_input(n, workl, iworkl, nsteps, x, xend,
                        beta_default=0.04, fac1_default=0.2, fac2_default=10.0)
    if isinstance(inp, int):
        return x, np.array(y, dtype=np.float64), inp
    nmax, nstiff, uround, safe, fac1, fac2, beta, hmax, h = inp

    atol_l = _tol_list(atol, n)
    rtol_l = _tol_list(rtol, n)

    def fcn(x_, y_):
        return _flat_result(func(x_, np.array(y_, dtype=np.float64),
                                 *f_params), n)

    posneg = math.copysign(1.0, xend - x)
    hmax = abs(hmax)

    k1 = fcn(x, y)
    if h == 0.0:
        h = _hinit(n, fcn, x, y, posneg, k1, hmax, atol_l, rtol_l, 5)
    nfcn = 2

    nstep = 0
    naccept = 0
    nreject = 0
    reject = 0
    last = 0
    facold = 1e-4
    expo1 = 0.2 - beta * 0.75
    facc1 = 1.0 / fac1
    facc2 = 1.0 / fac2
    xold = x
    iasti = 0
    nonsti = 0
    hlamb = 0.0
    istate = 0

    irtrn = 1
    if iout != 0:
        if (solout(x, np.array(y, dtype=np.float64)) or 0) < 0:
            return x, np.array(y, dtype=np.float64), 2
    else:
        irtrn = 0

    # Butcher tableau (scipy dop.c dopcor constants)
    C2, C3, C4, C5 = 0.2, 0.3, 0.8, 8.0 / 9.0
    A21 = 0.2
    A31, A32 = 3.0 / 40.0, 9.0 / 40.0
    A41, A42, A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    A51, A52, A53, A54 = (19372.0 / 6561.0, -25360.0 / 2187.0,
                          64448.0 / 6561.0, -212.0 / 729.0)
    A61, A62, A63, A64, A65 = (9017.0 / 3168.0, -355.0 / 33.0,
                               46732.0 / 5247.0, 49.0 / 176.0,
                               -5103.0 / 18656.0)
    A71, A73, A74, A75, A76 = (35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0,
                               -2187.0 / 6784.0, 11.0 / 84.0)
    E1, E3, E4, E5, E6, E7 = (71.0 / 57600.0, -71.0 / 16695.0, 71.0 / 1920.0,
                              -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0)

    while True:
        if nstep > nmax:
            istate = -2
            break
        if 0.1 * abs(h) <= abs(x) * uround:
            istate = -3
            break
        if (x + 1.01 * h - xend) * posneg > 0.0:
            h = xend - x
            last = 1
        nstep += 1

        if irtrn >= 2:
            k1 = fcn(x, y)

        y1 = [y[i] + h * A21 * k1[i] for i in range(n)]
        k2 = fcn(x + C2 * h, y1)
        y1 = [y[i] + h * (A31 * k1[i] + A32 * k2[i]) for i in range(n)]
        k3 = fcn(x + C3 * h, y1)
        y1 = [y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]) for i in range(n)]
        k4 = fcn(x + C4 * h, y1)
        y1 = [y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] +
                          A54 * k4[i]) for i in range(n)]
        k5 = fcn(x + C5 * h, y1)
        ysti = [y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] +
                            A64 * k4[i] + A65 * k5[i]) for i in range(n)]
        xph = x + h
        k6 = fcn(xph, ysti)
        y1 = [y[i] + h * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] +
                          A75 * k5[i] + A76 * k6[i]) for i in range(n)]
        k2 = fcn(xph, y1)
        nfcn += 6

        err = 0.0
        for i in range(n):
            sk = atol_l[i] + rtol_l[i] * max(abs(y[i]), abs(y1[i]))
            e = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] +
                     E6 * k6[i] + E7 * k2[i])
            err += (e / sk) ** 2
        err = math.sqrt(err / n)

        fac11 = err ** expo1
        fac = fac11 / facold ** beta
        fac = max(facc2, min(facc1, fac / safe))
        hnew = h / fac

        if err <= 1.0:
            facold = max(err, 1e-4)
            naccept += 1

            if naccept % nstiff == 0 or iasti > 0:
                stnum = 0.0
                stden = 0.0
                for i in range(n):
                    stnum += (k2[i] - k6[i]) ** 2
                    stden += (y1[i] - ysti[i]) ** 2
                if stden > 0.0:
                    hlamb = h * math.sqrt(stnum / stden)
                if hlamb > 3.25:
                    nonsti = 0
                    iasti += 1
                    if iasti == 15:
                        istate = -4
                        break
                else:
                    nonsti += 1
                    if nonsti == 6:
                        iasti = 0

            k1 = k2
            y = y1
            xold = x
            x = xph

            if iout != 0:
                if (solout(x, np.array(y, dtype=np.float64)) or 0) < 0:
                    istate = 2
                    break

            if last:
                h = hnew
                istate = 1
                break

            if abs(hnew) > hmax:
                hnew = posneg * hmax
            if reject:
                hnew = posneg * min(abs(hnew), abs(h))
            reject = 0
        else:
            hnew = h / min(facc1, fac11 / safe)
            reject = 1
            if naccept >= 1:
                nreject += 1
            last = 0
        h = hnew

    work[6] = h
    iwork[16] = nfcn
    iwork[17] = nstep
    iwork[18] = naccept
    iwork[19] = nreject
    if verbosity >= 0:
        print('dopri5: nsteps=%d naccept=%d nreject=%d nfcn=%d istate=%d'
              % (nstep, naccept, nreject, nfcn, istate))
    return x, np.array(y, dtype=np.float64), istate


# ---------------------------------------------------------------------------
# DOP853
# ---------------------------------------------------------------------------

def dopri853(func, t0, y0, t1, rtol, atol, solout, iout, work, iwork,
             nsteps, verbosity, f_params):
    """DOP853 (Hairer-Wanner 8(5)) integrator, scipy C ``dopri853``
    driver interface; returns ``(x, y, istate)``."""
    y0 = np.asarray(y0, dtype=np.float64)
    if y0.ndim != 1 or y0.size == 0:
        return float(t0), np.array([], dtype=np.float64), -1
    n = y0.size
    y = [float(v) for v in y0.tolist()]
    x = float(t0)
    xend = float(t1)

    workl = [float(v) for v in np.asarray(work, dtype=np.float64).tolist()]
    iworkl = [int(v) for v in np.asarray(iwork, dtype=np.int64).tolist()]

    inp = _driver_input(n, workl, iworkl, nsteps, x, xend,
                        beta_default=0.0, fac1_default=1.0 / 3.0,
                        fac2_default=6.0)
    if isinstance(inp, int):
        return x, np.array(y, dtype=np.float64), inp
    nmax, nstiff, uround, safe, fac1, fac2, beta, hmax, h = inp

    atol_l = _tol_list(atol, n)
    rtol_l = _tol_list(rtol, n)

    def fcn(x_, y_):
        return _flat_result(func(x_, np.array(y_, dtype=np.float64),
                                 *f_params), n)

    posneg = math.copysign(1.0, xend - x)
    hmax = abs(hmax)

    k1 = fcn(x, y)
    if h == 0.0:
        h = _hinit(n, fcn, x, y, posneg, k1, hmax, atol_l, rtol_l, 8)
    nfcn = 2

    nstep = 0
    naccept = 0
    nreject = 0
    reject = 0
    last = 0
    facold = 1e-4
    expo1 = 0.125 - beta * 0.2
    facc1 = 1.0 / fac1
    facc2 = 1.0 / fac2
    xold = x
    iasti = 0
    nonsti = 0
    hlamb = 0.0
    istate = 0

    irtrn = 0
    if iout >= 1:
        irtrn = 1
        if (solout(x, np.array(y, dtype=np.float64)) or 0) < 0:
            return x, np.array(y, dtype=np.float64), -3

    # Butcher tableau (scipy dop.c dp86co constants)
    c2 = 0.526001519587677318785587544488e-01
    c3 = 0.789002279381515978178381316732e-01
    c4 = 0.118350341907227396726757197510e+00
    c5 = 0.281649658092772603273242802490e+00
    c6 = 0.333333333333333333333333333333e+00
    c7 = 0.25e+00
    c8 = 0.307692307692307692307692307692e+00
    c9 = 0.651282051282051282051282051282e+00
    c10 = 0.6e+00
    c11 = 0.857142857142857142857142857142e+00
    b1 = 5.42937341165687622380535766363e-2
    b6 = 4.45031289275240888144113950566e0
    b7 = 1.89151789931450038304281599044e0
    b8 = -5.8012039600105847814672114227e0
    b9 = 3.1116436695781989440891606237e-1
    b10 = -1.52160949662516078556178806805e-1
    b11 = 2.01365400804030348374776537501e-1
    b12 = 4.47106157277725905176885569043e-2
    bhh1 = 0.244094488188976377952755905512e+00
    bhh2 = 0.733846688281611857341361741547e+00
    bhh3 = 0.220588235294117647058823529412e-01
    er1 = 0.1312004499419488073250102996e-01
    er6 = -0.1225156446376204440720569753e+01
    er7 = -0.4957589496572501915214079952e+00
    er8 = 0.1664377182454986536961530415e+01
    er9 = -0.3503288487499736816886487290e+00
    er10 = 0.3341791187130174790297318841e+00
    er11 = 0.8192320648511571246570742613e-01
    er12 = -0.2235530786388629525884427845e-01
    a21 = 5.26001519587677318785587544488e-2
    a31 = 1.97250569845378994544595329183e-2
    a32 = 5.91751709536136983633785987549e-2
    a41 = 2.95875854768068491816892993775e-2
    a43 = 8.87627564304205475450678981324e-2
    a51 = 2.41365134159266685502369798665e-1
    a53 = -8.84549479328286085344864962717e-1
    a54 = 9.24834003261792003115737966543e-1
    a61 = 3.7037037037037037037037037037e-2
    a64 = 1.70828608729473871279604482173e-1
    a65 = 1.25467687566822425016691814123e-1
    a71 = 3.7109375e-2
    a74 = 1.70252211019544039314978060272e-1
    a75 = 6.02165389804559606850219397283e-2
    a76 = -1.7578125e-2
    a81 = 3.70920001185047927108779319836e-2
    a84 = 1.70383925712239993810214054705e-1
    a85 = 1.07262030446373284651809199168e-1
    a86 = -1.53194377486244017527936158236e-2
    a87 = 8.27378916381402288758473766002e-3
    a91 = 6.24110958716075717114429577812e-1
    a94 = -3.36089262944694129406857109825e0
    a95 = -8.68219346841726006818189891453e-1
    a96 = 2.75920996994467083049415600797e1
    a97 = 2.01540675504778934086186788979e1
    a98 = -4.34898841810699588477366255144e1
    a101 = 4.77662536438264365890433908527e-1
    a104 = -2.48811461997166764192642586468e0
    a105 = -5.90290826836842996371446475743e-1
    a106 = 2.12300514481811942347288949897e1
    a107 = 1.52792336328824235832596922938e1
    a108 = -3.32882109689848629194453265587e1
    a109 = -2.03312017085086261358222928593e-2
    a111 = -9.3714243008598732571704021658e-1
    a114 = 5.18637242884406370830023853209e0
    a115 = 1.09143734899672957818500254654e0
    a116 = -8.14978701074692612513997267357e0
    a117 = -1.85200656599969598641566180701e1
    a118 = 2.27394870993505042818970056734e1
    a119 = 2.49360555267965238987089396762e0
    a1110 = -3.0467644718982195003823669022e0
    a121 = 2.27331014751653820792359768449e0
    a124 = -1.05344954667372501984066689879e1
    a125 = -2.00087205822486249909675718444e0
    a126 = -1.79589318631187989172765950534e1
    a127 = 2.79488845294199600508499808837e1
    a128 = -2.85899827713502369474065508674e0
    a129 = -8.87285693353062954433549289258e0
    a1210 = 1.23605671757943030647266201528e1
    a1211 = 6.43392746015763530355970484046e-1

    while True:
        if nstep > nmax:
            istate = -2
            break
        if 0.1 * abs(h) <= abs(x) * uround:
            istate = -3
            break
        if (x + 1.01 * h - xend) * posneg > 0.0:
            h = xend - x
            last = 1
        nstep += 1

        if irtrn >= 2:
            k1 = fcn(x, y)

        y1 = [y[i] + h * a21 * k1[i] for i in range(n)]
        k2 = fcn(x + c2 * h, y1)
        y1 = [y[i] + h * (a31 * k1[i] + a32 * k2[i]) for i in range(n)]
        k3 = fcn(x + c3 * h, y1)
        y1 = [y[i] + h * (a41 * k1[i] + a43 * k3[i]) for i in range(n)]
        k4 = fcn(x + c4 * h, y1)
        y1 = [y[i] + h * (a51 * k1[i] + a53 * k3[i] + a54 * k4[i]) for i in range(n)]
        k5 = fcn(x + c5 * h, y1)
        y1 = [y[i] + h * (a61 * k1[i] + a64 * k4[i] + a65 * k5[i]) for i in range(n)]
        k6 = fcn(x + c6 * h, y1)
        y1 = [y[i] + h * (a71 * k1[i] + a74 * k4[i] + a75 * k5[i] +
                          a76 * k6[i]) for i in range(n)]
        k7 = fcn(x + c7 * h, y1)
        y1 = [y[i] + h * (a81 * k1[i] + a84 * k4[i] + a85 * k5[i] +
                          a86 * k6[i] + a87 * k7[i]) for i in range(n)]
        k8 = fcn(x + c8 * h, y1)
        y1 = [y[i] + h * (a91 * k1[i] + a94 * k4[i] + a95 * k5[i] +
                          a96 * k6[i] + a97 * k7[i] + a98 * k8[i]) for i in range(n)]
        k9 = fcn(x + c9 * h, y1)
        y1 = [y[i] + h * (a101 * k1[i] + a104 * k4[i] + a105 * k5[i] +
                          a106 * k6[i] + a107 * k7[i] + a108 * k8[i] +
                          a109 * k9[i]) for i in range(n)]
        k10 = fcn(x + c10 * h, y1)
        y1 = [y[i] + h * (a111 * k1[i] + a114 * k4[i] + a115 * k5[i] +
                          a116 * k6[i] + a117 * k7[i] + a118 * k8[i] +
                          a119 * k9[i] + a1110 * k10[i]) for i in range(n)]
        k2 = fcn(x + c11 * h, y1)
        xph = x + h
        y1 = [y[i] + h * (a121 * k1[i] + a124 * k4[i] + a125 * k5[i] +
                          a126 * k6[i] + a127 * k7[i] + a128 * k8[i] +
                          a129 * k9[i] + a1210 * k10[i] + a1211 * k2[i])
              for i in range(n)]
        k3 = fcn(xph, y1)
        nfcn += 11

        k4 = [b1 * k1[i] + b6 * k6[i] + b7 * k7[i] + b8 * k8[i] + b9 * k9[i] +
              b10 * k10[i] + b11 * k2[i] + b12 * k3[i] for i in range(n)]
        k5 = [y[i] + h * k4[i] for i in range(n)]

        err = 0.0
        err2 = 0.0
        for i in range(n):
            sk = atol_l[i] + rtol_l[i] * max(abs(y[i]), abs(k5[i]))
            e2 = k4[i] - bhh1 * k1[i] - bhh2 * k9[i] - bhh3 * k3[i]
            err2 += (e2 / sk) ** 2
            e = (er1 * k1[i] + er6 * k6[i] + er7 * k7[i] + er8 * k8[i] +
                 er9 * k9[i] + er10 * k10[i] + er11 * k2[i] + er12 * k3[i])
            err += (e / sk) ** 2
        deno = err + 0.01 * err2
        if deno <= 0.0:
            deno = 1.0
        err = abs(h) * err * math.sqrt(1.0 / (n * deno))

        fac11 = err ** expo1
        fac = fac11 / facold ** beta
        fac = max(facc2, min(facc1, fac / safe))
        hnew = h / fac

        if err <= 1.0:
            facold = max(err, 1e-4)
            naccept += 1
            k4 = fcn(xph, k5)
            nfcn += 1

            if naccept % nstiff == 0 or iasti > 0:
                stnum = 0.0
                stden = 0.0
                for i in range(n):
                    stnum += (k4[i] - k3[i]) ** 2
                    stden += (k5[i] - y1[i]) ** 2
                if stden > 0.0:
                    hlamb = abs(h) * math.sqrt(stnum / stden)
                if hlamb > 6.1:
                    nonsti = 0
                    iasti += 1
                    if iasti == 15:
                        istate = -4
                        break
                else:
                    nonsti += 1
                    if nonsti == 6:
                        iasti = 0

            k1 = k4
            y = k5
            xold = x
            x = xph

            if iout >= 1:
                if (solout(x, np.array(y, dtype=np.float64)) or 0) < 0:
                    istate = 2
                    break

            if last:
                h = hnew
                istate = 1
                break

            if abs(hnew) > hmax:
                hnew = posneg * hmax
            if reject:
                hnew = posneg * min(abs(hnew), abs(h))
            reject = 0
        else:
            hnew = h / min(facc1, fac11 / safe)
            reject = 1
            if naccept >= 1:
                nreject += 1
            last = 0
        h = hnew

    work[6] = h
    iwork[16] = nfcn
    iwork[17] = nstep
    iwork[18] = naccept
    iwork[19] = nreject
    if verbosity >= 0:
        print('dopri853: nsteps=%d naccept=%d nreject=%d nfcn=%d istate=%d'
              % (nstep, naccept, nreject, nfcn, istate))
    return x, np.array(y, dtype=np.float64), istate


class _Runner:
    """Non-descriptor callable wrapper for the runner functions.

    ``_ode.py`` stores the runner as a class attribute
    (``runner = getattr(_dop, 'dopri5', None)``) and calls it through
    the integrator instance (``self.runner(...)``).  A plain Python
    function in a class dict becomes a bound method on instance access
    and would inject the integrator as a first argument; a plain
    callable object has no ``__get__`` and is returned unchanged.
    """

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args):
        return self._fn(*args)


_dopri5 = dopri5
_dopri853 = dopri853
dopri5 = _Runner(_dopri5)
dopri853 = _Runner(_dopri853)
dop853 = dopri853
