"""Pure-Python ODEPACK interface for scicore.

Replaces the deleted C ``_odepack`` extension with pure-Python implementations:

- ``odeint``: adaptive Runge-Kutta (RK4 step doubling) integrator.
- ``lsoda``:  the full LSODA algorithm (ODEPACK) as a line-faithful port of the
  scipy C sources ``lsoda.c`` / ``lsoda.h`` (which themselves replace the
  original Fortran LAMBDASP/LSODA).  Adams (non-stiff) + BDF (stiff) methods
  with automatic switching, user- or finite-difference Jacobians, full and
  banded Jacobian support, and state persistence between calls via
  ``state_doubles`` / ``state_ints``.

``zvode`` / ``dvode`` remain stubs (not yet ported).
"""

from __future__ import annotations

import math
import sys

import rsnumpy as np


__all__ = ['odeint', 'ODEintWarning', 'lsoda', 'zvode', 'dvode']


class ODEintWarning(Warning):
    pass


def odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, ml=None, mu=None,
           full_output=0, rtol=None, atol=None, tcrit=None, h0=0.0,
           hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
           mxords=5, printmessg=0, tfirst=False):
    """Pure-Python adaptive Runge-Kutta integrator (RK4 step doubling).

    Replaces the ODEPACK `odeint` stub.  Returns ``[Y, ier]``
    (``[Y, info, ier]`` when ``full_output`` is truthy), the convention
    expected by ``_odepack_py.odeint``.
    """
    if rtol is None:
        rtol = 1.49e-8
    if atol is None:
        atol = 1e-6

    tl = [float(v) for v in np.asarray(t).tolist()]
    y = [float(v) for v in y0]
    n = len(y)

    def f(t_val, yv):
        if tfirst:
            out = func(t_val, yv, *args)
        else:
            out = func(yv, t_val, *args)
        return [float(v) for v in out]

    def rk4(t_val, yv, h):
        k1 = f(t_val, yv)
        k2 = f(t_val + 0.5 * h, [yv[i] + 0.5 * h * k1[i] for i in range(n)])
        k3 = f(t_val + 0.5 * h, [yv[i] + 0.5 * h * k2[i] for i in range(n)])
        k4 = f(t_val + h, [yv[i] + h * k3[i] for i in range(n)])
        return [yv[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
                for i in range(n)]

    rows = [y[:]]
    for span in range(len(tl) - 1):
        t0, t1 = tl[span], tl[span + 1]
        if t1 == t0:
            rows.append(y[:])
            continue
        target = t1 - t0
        h = h0 if h0 != 0.0 else target / 10.0
        if abs(h) >= abs(target):
            h = target
        t_val = t0
        steps = 0
        limit = mxstep if mxstep > 0 else 100000
        while abs(t1 - t_val) > 1e-15 * max(1.0, abs(t1)):
            remaining = t1 - t_val
            if abs(h) > abs(remaining):
                h = remaining
            y1 = rk4(t_val, y, h)
            ya = rk4(t_val, y, 0.5 * h)
            y2 = rk4(t_val + 0.5 * h, ya, 0.5 * h)
            # Step doubling: y2 - y1 overestimates the local error by ~15x
            err = max(
                abs(y2[i] - y1[i])
                / (15.0 * (atol + rtol * max(abs(y[i]), abs(y2[i]))))
                for i in range(n))
            if err <= 1.0:
                t_val += h
                y = y2
                h *= 5.0 if err == 0.0 else 0.9 * err ** (-0.2)
            else:
                h *= 0.9 * err ** (-0.2)
            if hmax > 0.0:
                h = min(h, hmax) if h > 0 else max(h, -hmax)
            if hmin > 0.0 and abs(h) < hmin:
                h = hmin if h > 0 else -hmin
            steps += 1
            if steps >= limit:
                break
        rows.append(y[:])

    Y = np.array(rows)
    ier = 0
    if full_output:
        info = {'t': np.asarray(tl), 'y': Y, 'h': 0.0, 'message': ''}
        return [Y, info, ier]
    return [Y, ier]


def zvode(*args, **kwargs):
    """Stub for zvode."""
    raise NotImplementedError("zvode not yet ported to Rust/rsnumpy")


def dvode(*args, **kwargs):
    """Stub for dvode."""
    raise NotImplementedError("dvode not yet ported to Rust/rsnumpy")


# ============================================================================
# LSODA — pure-Python port of scipy integrate/src/lsoda.c + lsoda.h
#
# The C sources are themselves a port of the original Fortran ODEPACK
# (LAMBDASP + LSODA by C. Hindmarsh).  Field ordering of the state struct
# below matches lsoda_common_struct_t exactly (all doubles first, then all
# ints) so that pack/unpack of state_doubles/state_ints stays binary
# compatible with the C implementation.
# ============================================================================

LSODA_STATE_DOUBLE_SIZE = 240
LSODA_STATE_INT_SIZE = 48

# (field, count) pairs in declaration order of the C struct.
_DBL_LAYOUT = (
    ('conit', 1), ('crate', 1), ('el', 13), ('elco', 156), ('hold', 1),
    ('rmax', 1), ('tesco', 36), ('ccmax', 1), ('el0', 1), ('h', 1),
    ('hmin', 1), ('hmxi', 1), ('hu', 1), ('rc', 1), ('tn', 1), ('uround', 1),
    ('tsw', 1), ('pdest', 1), ('pdlast', 1), ('ratio', 1), ('cm1', 12),
    ('cm2', 5), ('pdnorm', 1),
)
_INT_LAYOUT = (
    ('illin', 1), ('init', 1), ('lyh', 1), ('lewt', 1), ('lacor', 1),
    ('lsavf', 1), ('lwm', 1), ('liwm', 1), ('mxstep', 1), ('mxhnil', 1),
    ('nhnil', 1), ('ntrep', 1), ('nslast', 1), ('nyh', 1), ('ialth', 1),
    ('ipup', 1), ('lmax', 1), ('meo', 1), ('nqnyh', 1), ('nslp', 1),
    ('icf', 1), ('ierpj', 1), ('iersl', 1), ('jcur', 1), ('jstart', 1),
    ('kflag', 1), ('l', 1), ('meth', 1), ('miter', 1), ('maxord', 1),
    ('maxcor', 1), ('msbp', 1), ('mxncf', 1), ('n', 1), ('nq', 1),
    ('nst', 1), ('nfe', 1), ('nje', 1), ('nqu', 1), ('insufr', 1),
    ('insufi', 1), ('ixpr', 1), ('icount', 1), ('irflag', 1), ('jtyp', 1),
    ('mused', 1), ('mxordn', 1), ('mxords', 1),
)


def _new_state():
    """A zeroed integrator state dict (matches `lsoda_common_struct_t S = {0}`)."
    """
    S = {}
    for name, cnt in _DBL_LAYOUT:
        S[name] = [0.0] * cnt if cnt > 1 else 0.0
    for name, cnt in _INT_LAYOUT:
        S[name] = 0
    return S


def _pack_state(S, d_state, i_state):
    """Serialize S into the flat state arrays (once per integration call)."""
    pos = 0
    for name, cnt in _DBL_LAYOUT:
        v = S[name]
        if cnt == 1:
            d_state[pos] = v
            pos += 1
        else:
            for j in range(cnt):
                d_state[pos + j] = v[j]
            pos += cnt
    pos = 0
    for name, cnt in _INT_LAYOUT:
        i_state[pos] = S[name]
        pos += cnt


def _unpack_state(d_state, i_state, S):
    """Restore S from the flat state arrays into the (pre-built) state dict."""
    dl = d_state.tolist()
    pos = 0
    for name, cnt in _DBL_LAYOUT:
        if cnt == 1:
            S[name] = float(dl[pos])
            pos += 1
        else:
            S[name] = [float(v) for v in dl[pos:pos + cnt]]
            pos += cnt
    il = i_state.tolist()
    pos = 0
    for name, cnt in _INT_LAYOUT:
        S[name] = int(il[pos])
        pos += cnt


class _LsodaCallbacks:
    """Wraps the user f / jac with tfirst and extra-parameter handling.

    Matches the C callback thunks: tfirst=1 -> func(t, y, *params),
    tfirst=0 -> func(y, t, *params).
    """

    def __init__(self, fun, jac, f_params, jac_params, tfirst, neq):
        self.fun = fun
        self.jac = jac
        self.f_params = tuple(f_params) if f_params else ()
        self.jac_params = tuple(jac_params) if jac_params else ()
        self.tfirst = tfirst
        self.neq = neq
        self.last_error = None

    def f(self, t, y):
        """Evaluate user f; returns a flat float list, or None on callback
        error (neq[0] set to -1, mirroring the C error flag).

        ``y`` is passed through as a plain Python list (not an array) so the
        user function's arithmetic stays on native floats; this mirrors the
        ``odeint`` convention and avoids per-element rsnumpy scalar overhead.
        """
        try:
            if self.tfirst:
                res = self.fun(t, y, *self.f_params)
            else:
                res = self.fun(y, t, *self.f_params)
            return _flat_floats(res)
        except Exception:
            self.neq[0] = -1
            self.last_error = sys.exc_info()[1]
            return None

    def jac(self, t, y):
        """Evaluate user Jacobian; returns a list of float lists, or None on
        callback error (neq[0] set to -1).

        ``y`` is passed through as a plain Python list (see ``f``).
        """
        try:
            if self.tfirst:
                res = self.jac(t, y, *self.jac_params)
            else:
                res = self.jac(y, t, *self.jac_params)
            return _rows_floats(res)
        except Exception:
            self.neq[0] = -1
            self.last_error = sys.exc_info()[1]
            return None


def _flat_floats(res):
    """Coerce a callback result to a flat Python float list."""
    if isinstance(res, (int, float)):
        return [float(res)]
    if hasattr(res, 'tolist'):
        res = res.tolist()
    if isinstance(res, (int, float)):
        return [float(res)]
    out = [float(v) for v in res]
    return out


def _rows_floats(res):
    """Coerce a 2-D callback result to a list of Python float lists."""
    if hasattr(res, 'tolist'):
        res = res.tolist()
    out = []
    for row in res:
        if hasattr(row, 'tolist'):
            row = row.tolist()
        out.append([float(v) for v in row])
    return out


# ----------------------------------------------------------------------------
# Norms (vmnorm / fnorm / bnorm)
# ----------------------------------------------------------------------------

def _vmnorm(n, v, vo, w, wo):
    """Weighted max norm max_i |v[vo+i]|*w[wo+i].  v/w are flat lists (they
    may be the same rwork list at different offsets, or the solution list)."""
    vm = 0.0
    for i in range(n):
        t = abs(v[vo + i]) * w[wo + i]
        if t > vm:
            vm = t
    return vm


def _fnorm(n, a, ao, w, wo):
    """Weighted max-norm of a full matrix, column-major at a[ao + i + j*n]."""
    an = 0.0
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += abs(a[ao + i + j * n]) / w[wo + j]
        t = s * w[wo + i]
        if t > an:
            an = t
    return an


# ----------------------------------------------------------------------------
# Method coefficients (cfode)
# ----------------------------------------------------------------------------

def _cfode(meth):
    """(elco[156], tesco[36]) for meth 1 (Adams, order<=12) or 2 (BDF, <=5)."""
    elco = [0.0] * 156
    tesco = [0.0] * 36
    if meth == 1:
        elco[0] = 1.0
        elco[1] = 1.0
        tesco[0] = 0.0
        tesco[1] = 2.0
        tesco[3] = 1.0
        tesco[35] = 0.0
        pc = [0.0] * 12
        pc[0] = 1.0
        rqfac = 1.0
        for nq in range(1, 12):
            rq1fac = rqfac
            rqfac = rqfac / (nq + 1)
            pc[nq] = 0.0
            for ib in range(nq):
                pc[nq - ib] = pc[nq - ib - 1] + nq * pc[nq - ib]
            pc[0] = nq * pc[0]
            # integral of p(x) and x*p(x) on [-1, 0]
            pint = pc[0]
            xpin = pc[0] * 0.5
            tsign = 1.0
            for i in range(1, nq + 1):
                tsign = -tsign
                pint = pint + tsign * pc[i] / (i + 1)
                xpin = xpin + tsign * pc[i] / (i + 2)
            elco[0 + nq * 13] = pint * rq1fac
            elco[1 + nq * 13] = 1.0
            for i in range(1, nq + 1):
                elco[(i + 1) + nq * 13] = rq1fac * pc[i] / (i + 1)
            agamq = rqfac * xpin
            ragq = 1.0 / agamq
            tesco[1 + 3 * nq] = ragq
            if nq < 11:
                tesco[3 * (nq + 1)] = ragq * rqfac / (nq + 2)
            tesco[2 + 3 * (nq - 1)] = ragq
        return elco, tesco

    # meth == 2 (BDF)
    pc = [0.0] * 12
    pc[0] = 1.0
    rq1fac = 1.0
    for nq in range(5):
        pc[nq + 1] = 0.0
        for ib in range(nq + 1):
            pc[nq - ib + 1] = pc[nq - ib] + (nq + 1) * pc[nq - ib + 1]
        pc[0] = (nq + 1) * pc[0]
        for i in range(nq + 2):
            elco[i + nq * 13] = pc[i] / pc[1]
        elco[1 + nq * 13] = 1.0
        tesco[0 + 3 * nq] = rq1fac
        tesco[1 + 3 * nq] = (nq + 2) / elco[nq * 13]
        tesco[2 + 3 * nq] = (nq + 3) / elco[nq * 13]
        rq1fac = rq1fac / (nq + 1)
    return elco, tesco


# ----------------------------------------------------------------------------
# Pure-Python LAPACK: dgetrf / dgetrs (LU factorization with partial pivoting)
# ----------------------------------------------------------------------------

def _dgetrf(n, a, lda, base=0):
    """LU factorization with partial pivoting (LAPACK dgetrf).

    ``a`` is a flat list in column-major order (element (i, j) at
    ``a[base + i + j*lda]``, lda >= n), overwritten in place with the L and
    U factors.  Returns ``(ipiv, ier)``: ipiv is the 1-based pivot list
    (ipiv[k] is the row interchanged with row k at step k, 1-based) and ier
    is the 1-based index of the first zero diagonal, or 0 on success.
    """
    ipiv = [0] * n
    for k in range(n):
        mk = k + 1
        p = k
        pv = abs(a[base + k + k * lda])
        for i in range(k + 1, n):
            v = abs(a[base + i + k * lda])
            if v > pv:
                pv = v
                p = i
        ipiv[k] = p + 1
        if p != k:
            for j in range(n):
                col = j * lda
                a[base + k + col], a[base + p + col] = a[base + p + col], a[base + k + col]
        if a[base + k + k * lda] == 0.0:
            return ipiv, mk
        inv = 1.0 / a[base + k + k * lda]
        for i in range(k + 1, n):
            a[base + i + k * lda] *= inv
    return ipiv, 0


def _dgetrs(n, a, lda, ipiv, x, base=0):
    """Solve A*x = b from a _dgetrf factorization (LAPACK dgetrs, TRANS='N').

    ``a``/``lda``/``ipiv`` as in _dgetrf; ``x`` (list, length n) is
    overwritten with the solution and returned.
    """
    # forward substitution with row interchanges
    for k in range(n):
        p = ipiv[k] - 1
        if p != k:
            x[k], x[p] = x[p], x[k]
        s = 0.0
        for i in range(k):
            s += a[base + i + k * lda] * x[i]
        x[k] -= s
    # back substitution
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += a[base + i + j * lda] * x[j]
        x[i] = (x[i] - s) / a[base + i + i * lda]
    return x


# [SECTION:LU]

# ----------------------------------------------------------------------------
# prja / solsy — Jacobian matrix P = I - h*el0*J, its LU factorization, and
# the chord-iteration linear solve.  Port of prja()/solsy() in lsoda.c.
#
# Deviation: banded Jacobians (jt=4/5) are expanded to a full n x n matrix
# and factored with _dgetrf/_dgetrs instead of the C code's custom banded
# LAPACK call.  This is numerically equivalent for exactly banded matrices
# (out-of-band entries stay exactly zero under elimination, so pivots and
# factors agree) and avoids banded-storage index subtleties in pure Python.
# ----------------------------------------------------------------------------

def _prja(cb, S, rw, iw, y, neq):
    """Compute P = I - h*el0*J and its LU factorization.

    On return the LU of P is stored in rw at offset S['lwm']+2 (column-major
    n x n) with pivots in iw[20:20+n].  y is restored on return (it is
    perturbed transiently by the finite-difference path).
    """
    S['nje'] += 1
    S['ierpj'] = 0
    S['jcur'] = 1
    n = S['n']
    miter = S['miter']
    if miter in (0, 3):
        return
    lwm = S['lwm']
    lewt = S['lewt']
    lsavf = S['lsavf']
    hl0 = S['h'] * S['el0']
    con = -hl0
    for i in range(n * n):
        rw[lwm + 2 + i] = 0.0

    if miter == 1:
        # user-supplied full Jacobian (row-major n x n from the callback)
        J = cb.jac(S['tn'], y)
        if neq[0] == -1:
            return
        for i in range(n):
            row = J[i]
            for j in range(n):
                rw[lwm + 2 + i + j * n] = row[j] * con
    elif miter == 4:
        # user-supplied banded Jacobian, compressed (mband, n): row i holds
        # subdiagonal (i - mu), i.e. A[j + i - mu, j]  (scipy convention).
        ml = iw[0]
        mu = iw[1]
        J = cb.jac(S['tn'], y)
        if neq[0] == -1:
            return
        for i in range(len(J)):
            row = J[i]
            for j in range(n):
                rr = j + i - mu
                if 0 <= rr < n:
                    rw[lwm + 2 + rr + j * n] = row[j] * con
    else:
        # miter 2/5: finite differences, n calls to f (full Jacobian)
        srur = math.sqrt(S['uround'])
        fac = _vmnorm(n, rw, lsavf, rw, lewt)
        r0 = 1000.0 * abs(S['h']) * S['uround'] * n * fac
        if r0 == 0.0:
            r0 = 1.0
        j1 = 0
        for j in range(n):
            yj = y[j]
            r = max(srur * abs(yj), r0 / rw[lewt + j])
            y[j] = yj + r
            fac = -hl0 / r
            ftem = cb.f(S['tn'], y)
            if neq[0] == -1:
                y[j] = yj
                return
            for i in range(n):
                rw[lwm + 2 + i + j1] = (ftem[i] - rw[lsavf + i]) * fac
            y[j] = yj
            j1 += n
        S['nfe'] += n

    S['pdnorm'] = _fnorm(n, rw, lwm + 2, rw, lewt) / abs(hl0)
    # add identity
    for i in range(n):
        rw[lwm + 2 + i * (n + 1)] += 1.0
    # LU factorization of P
    ipiv, ier = _dgetrf(n, rw, n, lwm + 2)
    if ier != 0:
        S['ierpj'] = 1
    for i in range(n):
        iw[20 + i] = ipiv[i]


def _solsy(S, rw, iw, x):
    """Solve P*x = b for x (in place); port of solsy().

    The LU of P must already be in rw at S['lwm']+2 with pivots in iw[20:].
    S['iersl'] is set to 1 if a singular diagonal is found (miter == 3 only).
    """
    S['iersl'] = 0
    n = S['n']
    miter = S['miter']
    lwm = S['lwm']
    if miter == 3:
        phl0 = rw[lwm + 1]
        hl0 = S['h'] * S['el0']
        rw[lwm + 1] = hl0
        if hl0 != phl0:
            r = hl0 / phl0
            for i in range(n):
                di = 1.0 - r * (1.0 - 1.0 / rw[lwm + 2 + i])
                if abs(di) == 0.0:
                    S['iersl'] = 1
                    return
                rw[lwm + 2 + i] = 1.0 / di
        for i in range(n):
            x[i] = rw[lwm + 2 + i] * x[i]
        return
    if miter in (1, 2, 4, 5):
        piv = [iw[20 + i] for i in range(n)]
        _dgetrs(n, rw, n, piv, x, lwm + 2)

# ----------------------------------------------------------------------------
# stoda — the one-step core integrator (Adams / BDF with automatic switching).
# Port of the stoda_* family in lsoda.c (itself the C port of stoda.f from
# ODEPACK).  All work arrays live in the flat rwork list ``rw`` at the offsets
# recorded in S (lyh: Nordsieck array, lewt: error weights, lsavf: saved f,
# lacor: corrector accumulation, lwm: Jacobian workspace).
# ----------------------------------------------------------------------------

# stability-region constants sm1[i] for orders i+1 (Adams)
_SM1 = [0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.20, 0.15, 0.10, 0.075, 0.050, 0.025]


def _stoda_first_call_init(S):
    """First-call initialization (jstart == 0); C stoda_first_call_init."""
    S['lmax'] = S['maxord'] + 1
    S['nq'] = 1
    S['l'] = 2
    S['ialth'] = 2
    S['rmax'] = 10000.0
    S['rc'] = 0.0
    S['el0'] = 1.0
    S['crate'] = 0.7
    S['hold'] = S['h']
    S['nslp'] = 0
    S['ipup'] = S['miter']
    # switching parameters (meth = 1 is assumed initially)
    S['icount'] = 20
    S['irflag'] = 0
    S['pdest'] = 0.0
    S['pdlast'] = 0.0
    S['ratio'] = 5.0
    # coefficients for both methods; cm2 comes from BDF, cm1 from Adams
    elco, tesco = _cfode(2)
    S['elco'] = elco
    S['tesco'] = tesco
    S['cm2'] = [tesco[1 + i * 3] * elco[i + 1 + i * 13] for i in range(5)]
    elco, tesco = _cfode(1)
    S['elco'] = elco
    S['tesco'] = tesco
    S['cm1'] = [tesco[1 + i * 3] * elco[i + 1 + i * 13] for i in range(12)]


def _stoda_reset(S):
    """Reset el and related constants after an order change (labels 150-160)."""
    for i in range(S['l']):
        S['el'][i] = S['elco'][i + (S['nq'] - 1) * 13]
    S['nqnyh'] = S['nq'] * S['nyh']
    S['rc'] = S['rc'] * S['el'][0] / S['el0']
    S['el0'] = S['el'][0]
    S['conit'] = 0.5 / (S['nq'] + 2)


def _stoda_adjust_step_size(S, rw, rh):
    """Scale the Nordsieck array and h by rh (labels 175-200).  rh is a
    one-element list, modified in place."""
    n = S['n']
    nyh = S['nyh']
    lyh = S['lyh']
    rh[0] = min(rh[0], S['rmax'])
    rh[0] = rh[0] / max(1.0, abs(S['h']) * S['hmxi'] * rh[0])
    # meth == 1: restrict by the stability region (set irflag if reduced)
    if S['meth'] != 2:
        S['irflag'] = 0
        pdh = max(abs(S['h']) * S['pdlast'], 0.000001)
        if rh[0] * pdh * 1.00001 >= _SM1[S['nq'] - 1]:
            rh[0] = _SM1[S['nq'] - 1] / pdh
            S['irflag'] = 1
    r = 1.0
    for j in range(1, S['l']):
        r = r * rh[0]
        for i in range(n):
            rw[lyh + i + j * nyh] = rw[lyh + i + j * nyh] * r
    S['h'] = S['h'] * rh[0]
    S['rc'] = S['rc'] * rh[0]
    S['ialth'] = S['l']


def _stoda_corrector_loop(cb, S, rw, iw, y, neq, pnorm):
    """Corrector iterations (labels 220-450).  Returns (status, m, del) with
    status in {'converged', 'retry', 'no_conv', 'error'}."""
    n = S['n']
    nyh = S['nyh']
    lyh = S['lyh']
    lewt = S['lewt']
    lsavf = S['lsavf']
    lacor = S['lacor']
    m = 0
    rate = 0.0
    del_ = 0.0
    delp = 0.0

    for i in range(n):
        y[i] = rw[lyh + i]
    savf = cb.f(S['tn'], y)
    if neq[0] == -1:
        return 'error', m, del_
    S['nfe'] += 1
    for i in range(n):
        rw[lsavf + i] = savf[i]

    if S['ipup'] > 0:
        # P = I - h*el0*J must be (re)evaluated and factored
        _prja(cb, S, rw, iw, y, neq)
        if neq[0] == -1:
            return 'error', m, del_
        S['ipup'] = 0
        S['rc'] = 1.0
        S['nslp'] = S['nst']
        S['crate'] = 0.7
        if S['ierpj'] != 0:
            return 'no_conv', m, del_

    for i in range(n):
        rw[lacor + i] = 0.0

    while True:
        if S['miter'] == 0:
            # functional iteration
            for i in range(n):
                rw[lsavf + i] = S['h'] * rw[lsavf + i] - rw[lyh + i + nyh]
                y[i] = rw[lsavf + i] - rw[lacor + i]
            del_ = _vmnorm(n, y, 0, rw, lewt)
            for i in range(n):
                y[i] = rw[lyh + i] + S['el'][0] * rw[lsavf + i]
                rw[lacor + i] = rw[lsavf + i]
        else:
            # chord method: solve the linear system for the correction
            for i in range(n):
                y[i] = S['h'] * rw[lsavf + i] - (rw[lyh + i + nyh] + rw[lacor + i])
            _solsy(S, rw, iw, y)
            if S['iersl'] < 0:
                return 'no_conv', m, del_
            if S['iersl'] > 0:
                if (S['miter'] != 0) and (S['jcur'] != 1):
                    S['icf'] = 1
                    S['ipup'] = S['miter']
                    return 'retry', m, del_
                return 'no_conv', m, del_
            del_ = _vmnorm(n, y, 0, rw, lewt)
            for i in range(n):
                rw[lacor + i] = rw[lacor + i] + y[i]
                y[i] = rw[lyh + i] + S['el'][0] * rw[lacor + i]

        # convergence tests (labels 400-410)
        if del_ <= 100.0 * pnorm * S['uround']:
            break
        if (m != 0) or (S['meth'] != 1):
            if m != 0:
                rm = 1024.0
                if del_ <= 1024.0 * delp:
                    rm = del_ / delp
                rate = max(rate, rm)
                S['crate'] = max(0.2 * S['crate'], rm)
            dcon = del_ * min(1.0, 1.5 * S['crate']) / (
                S['tesco'][1 + (S['nq'] - 1) * 3] * S['conit'])
            if dcon <= 1.0:
                S['pdest'] = max(S['pdest'], rate / abs(S['h'] * S['el'][0]))
                if S['pdest'] != 0.0:
                    S['pdlast'] = S['pdest']
                break
        m = m + 1
        if (m == S['maxcor']) or ((m >= 2) and (del_ > 2.0 * delp)):
            if S['miter'] != 0 and S['jcur'] != 1:
                S['icf'] = 1
                S['ipup'] = S['miter']
                return 'retry', m, del_
            return 'no_conv', m, del_
        delp = del_
        savf = cb.f(S['tn'], y)
        if neq[0] == -1:
            return 'error', m, del_
        S['nfe'] += 1
        for i in range(n):
            rw[lsavf + i] = savf[i]

    S['jcur'] = 0
    return 'converged', m, del_


def _stoda_handle_corrector_failure(S, rw, ncf, told, rh):
    """Recover from a corrector failure (labels 430-445 + retry setup).
    Returns 0 on successful recovery (retry the step), -1 on fatal error
    (S['kflag'] set).  ncf/rh are one-element lists, modified in place."""
    n = S['n']
    nyh = S['nyh']
    lyh = S['lyh']
    S['icf'] = 2
    ncf[0] += 1
    S['rmax'] = 2.0
    S['tn'] = told
    # retract the yh array to the pre-prediction values
    i1 = S['nqnyh']
    for jb in range(1, S['nq'] + 1):
        i1 -= nyh
        for i in range(i1, S['nqnyh']):
            rw[lyh + i] = rw[lyh + i] - rw[lyh + i + nyh]
    if S['ierpj'] < 0 or S['iersl'] < 0:
        S['kflag'] = -3
        return -1
    if abs(S['h']) <= S['hmin'] * 1.00001:
        S['kflag'] = -2
        return -1
    if ncf[0] == S['mxncf']:
        S['kflag'] = -2
        return -1
    rh[0] = 0.25
    S['ipup'] = S['miter']
    rh[0] = max(rh[0], S['hmin'] / abs(S['h']))
    _stoda_adjust_step_size(S, rw, rh)
    return 0


def _stoda_get_predicted_values(S, rw):
    """Advance to the predicted values: multiply the yh array by the pascal
    triangle (labels 205-233), update tn, and flag a Jacobian refresh."""
    n = S['n']
    nyh = S['nyh']
    lyh = S['lyh']
    if abs(S['rc'] - 1.0) > S['ccmax']:
        S['ipup'] = S['miter']
    if S['nst'] >= S['nslp'] + S['msbp']:
        S['ipup'] = S['miter']
    S['tn'] = S['tn'] + S['h']
    i1 = S['nqnyh']
    for jb in range(S['nq']):
        i1 -= nyh
        for i in range(i1, S['nqnyh']):
            rw[lyh + i] = rw[lyh + i] + rw[lyh + i + nyh]


def _stoda(cb, S, rw, iw, y, neq):
    """Perform one integration step (C stoda()).

    On return S['jstart'] is 1 (normal exit, success or recoverable failure)
    and S['kflag'] holds the outcome: 0 success, -1 error test failed at
    hmin, -2 convergence failures at hmin, -3 callback failure.
    """
    n = S['n']
    nyh = S['nyh']
    lyh = S['lyh']
    lewt = S['lewt']
    lsavf = S['lsavf']
    lacor = S['lacor']
    sm1 = _SM1

    rh = [0.0]
    ncf = [0]
    dsm = 0.0
    dup = 0.0
    ddn = 0.0
    dm1 = 0.0
    dm2 = 0.0
    exm1 = 0.0
    exm2 = 0.0
    exup = 0.0
    exdn = 0.0
    exsm = 0.0
    rh1 = 0.0
    rh2 = 0.0
    rh1it = 0.0
    rhup = 0.0
    rhdn = 0.0
    rhsm = 0.0
    pdh = 0.0
    newq = 0
    nqm1 = 0
    nqm2 = 0
    lm1 = 0
    lm2 = 0
    told = S['tn']
    should_reset_rmax = 0

    S['kflag'] = 0
    S['ierpj'] = 0
    S['iersl'] = 0
    S['jcur'] = 0
    S['icf'] = 0

    # ---- jstart dispatch (labels 100-160) ----
    if S['jstart'] == 0:
        # first call
        _stoda_first_call_init(S)
        _stoda_reset(S)
    elif S['jstart'] == -1:
        # restart, possibly with a method and/or step change
        S['ipup'] = S['miter']
        S['lmax'] = S['maxord'] + 1
        if S['ialth'] == 1:
            S['ialth'] = 2
        if S['meth'] != S['mused']:
            elco, tesco = _cfode(S['meth'])
            S['elco'] = elco
            S['tesco'] = tesco
            S['ialth'] = S['l']
            _stoda_reset(S)
        if S['h'] != S['hold']:
            rh[0] = S['h'] / S['hold']
            S['h'] = S['hold']
            rh[0] = max(rh[0], S['hmin'] / abs(S['h']))
            _stoda_adjust_step_size(S, rw, rh)
    elif S['jstart'] == -2:
        # step size change only
        if S['h'] != S['hold']:
            rh[0] = S['h'] / S['hold']
            S['h'] = S['hold']
            _stoda_adjust_step_size(S, rw, rh)
    # jstart > 0: normal continuation

    # ---- 200: predict ----
    _stoda_get_predicted_values(S, rw)
    pnorm = _vmnorm(n, rw, lyh, rw, lewt)

    while True:
        # ---- corrector loop (220-450) ----
        status, m_corr, del_corr = _stoda_corrector_loop(
            cb, S, rw, iw, y, neq, pnorm)

        if status == 'error':
            S['kflag'] = -3
            S['hold'] = S['h']
            S['jstart'] = 1
            return

        if status == 'retry':
            continue

        if status == 'no_conv':
            if _stoda_handle_corrector_failure(S, rw, ncf, told, rh) != 0:
                S['hold'] = S['h']
                S['jstart'] = 1
                return
            # recovered: repredict and retry the step
            _stoda_get_predicted_values(S, rw)
            pnorm = _vmnorm(n, rw, lyh, rw, lewt)
            continue

        # ---- converged: local error test (450) ----
        S['jcur'] = 0
        if m_corr == 0:
            dsm = del_corr / S['tesco'][1 + (S['nq'] - 1) * 3]
        else:
            dsm = _vmnorm(n, rw, lacor, rw, lewt) / S['tesco'][1 + (S['nq'] - 1) * 3]

        if dsm > 1.0:
            # ---- 500: the step failed the error test ----
            S['kflag'] -= 1
            S['tn'] = told
            i1 = S['nqnyh']
            for jb in range(S['nq']):
                i1 -= nyh
                for i in range(i1, S['nqnyh']):
                    rw[lyh + i] = rw[lyh + i] - rw[lyh + i + nyh]
            S['rmax'] = 2.0
            if abs(S['h']) <= S['hmin'] * 1.00001:
                S['kflag'] = -1
                S['hold'] = S['h']
                S['jstart'] = 1
                return
            if S['kflag'] <= -3:
                if S['kflag'] == -10:
                    S['kflag'] = -1
                    S['hold'] = S['h']
                    S['jstart'] = 1
                    return
                # 3+ failures: reset to order 1, cut h by 10, recompute f
                rh[0] = 0.1
                rh[0] = max(S['hmin'] / abs(S['h']), rh[0])
                S['h'] = S['h'] * rh[0]
                for i in range(n):
                    y[i] = rw[lyh + i]
                savf = cb.f(S['tn'], y)
                if neq[0] == -1:
                    return
                S['nfe'] += 1
                for i in range(n):
                    rw[lyh + i + nyh] = S['h'] * savf[i]
                S['ipup'] = S['miter']
                S['ialth'] = 5
                if S['nq'] == 1:
                    _stoda_get_predicted_values(S, rw)
                    pnorm = _vmnorm(n, rw, lyh, rw, lewt)
                    continue
                S['nq'] = 1
                S['l'] = 2
                _stoda_reset(S)
                _stoda_get_predicted_values(S, rw)
                pnorm = _vmnorm(n, rw, lyh, rw, lewt)
                continue
            rhup = 0.0
            # fall through to 540 (compute new step size)
        else:
            # ---- 460: successful step: update yh, consider switching ----
            S['kflag'] = 0
            should_reset_rmax = 1
            S['nst'] += 1
            S['hu'] = S['h']
            S['nqu'] = S['nq']
            S['mused'] = S['meth']
            for j in range(S['l']):
                for i in range(n):
                    rw[lyh + i + j * nyh] = (
                        rw[lyh + i + j * nyh] + S['el'][j] * rw[lacor + i])
            S['icount'] -= 1
            if S['icount'] >= 0:
                pass  # 488: no method change for now
            elif S['meth'] == 1:
                # Adams: consider switching to BDF
                polluted = (dsm > 100.0 * pnorm * S['uround']) and (
                    S['pdest'] != 0.0)
                if not (S['nq'] > 5 or (not polluted and S['irflag'] == 0)):
                    else_branch = 0
                    if not polluted:
                        rh2 = 2.0
                        nqm2 = min(S['nq'], S['mxords'])
                    else:
                        else_branch = 1
                        exsm = 1.0 / S['l']
                        rh1 = 1.0 / (1.2 * dsm ** exsm + 0.0000012)
                        rh1it = 2.0 * rh1
                        pdh = S['pdlast'] * abs(S['h'])
                        if pdh * rh1 > 0.00001:
                            rh1it = sm1[S['nq'] - 1] / pdh
                        rh1 = min(rh1, rh1it)
                        if S['nq'] > S['mxords']:
                            nqm2 = S['mxords']
                            lm2 = S['mxords'] + 1
                            exm2 = 1.0 / lm2
                            dm2 = _vmnorm(n, rw, lyh + lm2 * nyh, rw, lewt) / \
                                S['cm2'][S['mxords'] - 1]
                            rh2 = 1.0 / (1.2 * dm2 ** exm2 + 0.0000012)
                        else:
                            dm2 = dsm * (S['cm1'][S['nq'] - 1] /
                                         S['cm2'][S['nq'] - 1])
                            rh2 = 1.0 / (1.2 * dm2 ** exsm + 0.0000012)
                            nqm2 = S['nq']
                    if (else_branch and rh2 >= S['ratio'] * rh1) or (
                            not else_branch):
                        # 478: switch to BDF
                        rh[0] = rh2
                        S['icount'] = 20
                        S['meth'] = 2
                        S['miter'] = S['jtyp']
                        S['pdlast'] = 0.0
                        S['nq'] = nqm2
                        S['l'] = S['nq'] + 1
                        rh[0] = max(rh[0], S['hmin'] / abs(S['h']))
                        _stoda_adjust_step_size(S, rw, rh)
                        should_reset_rmax = 1
                        break
                # else: 488, no switch
            else:
                # BDF: consider switching to Adams
                exsm = 1.0 / S['l']
                if S['mxordn'] >= S['nq']:
                    dm1 = dsm * (S['cm2'][S['nq'] - 1] / S['cm1'][S['nq'] - 1])
                    rh1 = 1.0 / (1.2 * dm1 ** exsm + 0.0000012)
                    nqm1 = S['nq']
                    exm1 = exsm
                else:
                    nqm1 = S['mxordn']
                    lm1 = S['mxordn'] + 1
                    exm1 = 1.0 / lm1
                    dm1 = _vmnorm(n, rw, lyh + lm1 * nyh, rw, lewt) / \
                        S['cm1'][S['mxordn'] - 1]
                    rh1 = 1.0 / (1.2 * dm1 ** exm1 + 0.0000012)
                rh1it = 2.0 * rh1
                pdh = S['pdnorm'] * abs(S['h'])
                if pdh * rh1 > 0.00001:
                    rh1it = sm1[nqm1 - 1] / pdh
                rh1 = min(rh1, rh1it)
                rh2 = 1.0 / (1.2 * dsm ** exsm + 0.0000012)
                if rh1 * S['ratio'] >= 5.0 * rh2:
                    dm1 = max(0.001, rh1) ** exm1 * dm1
                    if dm1 > 1000.0 * S['uround'] * pnorm:
                        # 480: switch to Adams (functional iteration)
                        rh[0] = rh1
                        S['icount'] = 20
                        S['meth'] = 1
                        S['miter'] = 0
                        S['pdlast'] = 0.0
                        S['nq'] = nqm1
                        S['l'] = S['nq'] + 1
                        rh[0] = max(rh[0], S['hmin'] / abs(S['h']))
                        _stoda_adjust_step_size(S, rw, rh)
                        should_reset_rmax = 1
                        break

            # ---- 488: no method switch: usual step/order selection ----
            S['ialth'] -= 1
            if S['ialth'] == 0:
                # 520: consider an order increase
                rhup = 0.0
                if S['l'] != S['lmax']:
                    for i in range(n):
                        rw[lsavf + i] = (
                            rw[lacor + i] -
                            rw[lyh + i + (S['lmax'] - 1) * nyh])
                    dup = _vmnorm(n, rw, lsavf, rw, lewt) / \
                        S['tesco'][2 + (S['nq'] - 1) * 3]
                    exup = 1.0 / (S['l'] + 1.0)
                    rhup = 1.0 / (1.4 * dup ** exup + 0.0000014)
                # fall through to 540
            else:
                if not ((S['ialth'] > 1) or (S['l'] == S['lmax'])):
                    for i in range(n):
                        rw[lyh + i + (S['lmax'] - 1) * nyh] = rw[lacor + i]
                # 700: exit with the current step
                r = 1.0 / S['tesco'][1 + (S['nqu'] - 1) * 3]
                for i in range(n):
                    rw[lacor + i] = rw[lacor + i] * r
                S['hold'] = S['h']
                S['jstart'] = 1
                return

        # ---- 540: choose the next step size / order ----
        exsm = 1.0 / S['l']
        rhsm = 1.0 / (1.2 * dsm ** exsm + 0.0000012)
        rhdn = 0.0
        if S['nq'] != 1:
            ddn = _vmnorm(n, rw, lyh + (S['l'] - 1) * nyh, rw, lewt) / \
                S['tesco'][(S['nq'] - 1) * 3]
            exdn = 1.0 / S['nq']
            rhdn = 1.0 / (1.3 * ddn ** exdn + 0.0000013)

        # 550: Adams stability-region restrictions
        if S['meth'] == 1:
            pdh = max(abs(S['h']) * S['pdlast'], 0.000001)
            if S['l'] < S['lmax']:
                rhup = min(rhup, sm1[S['l'] - 1] / pdh)
            rhsm = min(rhsm, sm1[S['nq'] - 1] / pdh)
            if S['nq'] > 1:
                rhdn = min(rhdn, sm1[S['nq'] - 2] / pdh)
            S['pdest'] = 0.0

        # 560
        if rhsm >= rhup:
            if rhsm >= rhdn:
                newq = S['nq']
                rh[0] = rhsm
            else:
                # 580: decrease the order
                newq = S['nq'] - 1
                rh[0] = rhdn
                if (S['kflag'] < 0) and (rh[0] > 1.0):
                    rh[0] = 1.0
        else:
            if rhup > rhdn:
                # 590: increase the order
                newq = S['l']
                rh[0] = rhup
                if rh[0] < 1.1:
                    S['ialth'] = 3
                    should_reset_rmax = 0
                    break
                r = S['el'][S['l'] - 1] / S['l']
                for i in range(n):
                    rw[lyh + i + newq * nyh] = rw[lacor + i] * r
                S['nq'] = newq
                S['l'] = S['nq'] + 1
                _stoda_reset(S)
                rh[0] = max(rh[0], S['hmin'] / abs(S['h']))
                _stoda_adjust_step_size(S, rw, rh)
                should_reset_rmax = 1
                break
            else:
                # 580
                newq = S['nq'] - 1
                rh[0] = rhdn
                if (S['kflag'] < 0) and (rh[0] > 1.0):
                    rh[0] = 1.0

        # 620: bypass the 10% test for a stability-restricted Adams step
        bypass = (S['meth'] == 1) and (
            rh[0] * pdh * 1.00001 >= sm1[newq - 1])
        if (not bypass) and (S['kflag'] == 0) and (rh[0] < 1.1):
            # 610: improvement too small
            S['ialth'] = 3
            should_reset_rmax = 0
            break

        # 625
        if S['kflag'] <= -2:
            rh[0] = min(rh[0], 0.2)

        if newq != S['nq']:
            S['nq'] = newq
            S['l'] = S['nq'] + 1
            _stoda_reset(S)

        # 170
        rh[0] = max(rh[0], S['hmin'] / abs(S['h']))
        _stoda_adjust_step_size(S, rw, rh)

        if S['kflag'] == 0:
            should_reset_rmax = 1
            break

        # failed step: repredict with the new h and retry
        _stoda_get_predicted_values(S, rw)
        pnorm = _vmnorm(n, rw, lyh, rw, lewt)
        continue

    # ---- 690 ----
    if should_reset_rmax:
        S['rmax'] = 10.0
    # ---- 700 ----
    r = 1.0 / S['tesco'][1 + (S['nqu'] - 1) * 3]
    for i in range(n):
        rw[lacor + i] = rw[lacor + i] * r
    S['hold'] = S['h']
    S['jstart'] = 1
    return

# ----------------------------------------------------------------------------
# intdy / ewset — interpolation and error weights (ports of intdy()/ewset())
# ----------------------------------------------------------------------------

def _intdy(S, rw, t, k, y):
    """Interpolate the k-th derivative of the solution at time t into y
    (port of intdy()).  Returns 0 on success, -1 if k is out of range,
    -2 if t is out of range."""
    n = S['n']
    nyh = S['nyh']
    lyh = S['lyh']
    if k < 0 or k > S['nq']:
        return -1
    hu = S['hu']
    tp = S['tn'] - hu - 100.0 * S['uround'] * (S['tn'] + hu)
    if (t - tp) * (t - S['tn']) > 0.0:
        return -2
    s = (t - S['tn']) / S['h']
    ic = 1
    if k != 0:
        for jj in range(S['l'] - k, S['nq'] + 1):
            ic *= jj
    c = float(ic)
    for i in range(n):
        y[i] = c * rw[lyh + i + (S['l'] - 1) * nyh]
    if k != S['nq']:
        for jb in range(1, S['nq'] - k + 1):
            if k != 0:
                for jj in range(S['nq'] - jb + 1 - k, S['nq'] - jb + 1):
                    ic *= jj
            c = float(ic)
            for i in range(n):
                y[i] = c * rw[lyh + i + (S['nq'] - jb) * nyh] + s * y[i]
        if k == 0:
            return 0
    r = S['h'] ** (-k)
    for i in range(n):
        y[i] = r * y[i]
    return 0


def _ewset(n, itol, rtol, atol, rw, yoff, ewtoff):
    """Set the error weights rw[ewtoff+i] = rtol*|ycur[i]| + atol
    (port of ewset()); ycur is rw[yoff+i].  rtol/atol are scalars (itol 1)
    or length-n lists (itol 2/3/4)."""
    if itol == 1:
        for i in range(n):
            rw[ewtoff + i] = rtol * abs(rw[yoff + i]) + atol
    elif itol == 2:
        for i in range(n):
            rw[ewtoff + i] = rtol * abs(rw[yoff + i]) + atol[i]
    elif itol == 3:
        for i in range(n):
            rw[ewtoff + i] = rtol[i] * abs(rw[yoff + i]) + atol
    else:
        for i in range(n):
            rw[ewtoff + i] = rtol[i] * abs(rw[yoff + i]) + atol[i]


def _mark_error(istate_ref, S):
    """Error-flag escalation (C lsoda_mark_error)."""
    if S['illin'] == 5:
        istate_ref[0] = -8
        return
    S['illin'] += 1
    istate_ref[0] = -3


def _lsoda_core(cb, S, rw, iw, y, neq, itask, istate_ref, rtol, atol, tout,
                tref, jt):
    """Top-level LSODA driver (C lsoda()); one step towards ``tout``.

    itol is fixed to 1 (scalar rtol/atol) for this interface, matching the
    C wrapper used by the ode class.  On return the updated solution is in
    y, the time in tref[0], the state flag in istate_ref[0]; the rwork/iwork
    outputs (rwork[10:15], iwork[10:20]) are written in place.
    """
    itol = 1
    iflag = 0
    ihit = 0
    initial_jump = 1
    ml = 0
    mu = 0
    hmx = 0.0
    hmin = 0.0
    h0 = 0.0
    hmax = 0.0
    tcrit = 0.0
    tnext = 0.0
    tolsf = 0.0
    rtoli = 0.0
    atoli = 0.0
    lrw = len(rw)
    liw = len(iw)

    def finish():
        # C "exit" label: reset illin and publish optional outputs
        S['illin'] = 0
        rw[10] = S['hu']
        rw[11] = S['h']
        rw[12] = S['tn']
        rw[14] = S['tsw']
        iw[10] = S['nst']
        iw[11] = S['nfe']
        iw[12] = S['nje']
        iw[13] = S['nqu']
        iw[14] = S['nq']
        iw[18] = S['mused']
        iw[19] = S['meth']

    # ---- block a: istate / itask legality ----
    if istate_ref[0] < 1 or istate_ref[0] > 3:
        _mark_error(istate_ref, S)
        return
    if itask < 1 or itask > 5:
        _mark_error(istate_ref, S)
        return
    if istate_ref[0] != 1 and S['init'] == 0:
        _mark_error(istate_ref, S)
        return

    if istate_ref[0] == 1:
        # ---- first call: full initialization (blocks b + c) ----
        S['init'] = 0
        if tout == tref[0]:
            S['ntrep'] += 1
            if S['ntrep'] < 5:
                return
            istate_ref[0] = -8
            return
        S['ntrep'] = 0

        neq_in = len(y)
        if neq_in <= 0:
            _mark_error(istate_ref, S)
            return
        S['n'] = neq_in
        if jt == 3 or jt < 1 or jt > 5:
            _mark_error(istate_ref, S)
            return
        S['jtyp'] = jt
        if jt > 2:
            ml = iw[0]
            mu = iw[1]
            if ml < 0 or ml >= neq_in:
                _mark_error(istate_ref, S)
                return
            if mu < 0 or mu >= neq_in:
                _mark_error(istate_ref, S)
                return

        # optional inputs (iopt is always 1 for this interface)
        S['ixpr'] = iw[4]
        if S['ixpr'] < 0 or S['ixpr'] > 1:
            _mark_error(istate_ref, S)
            return
        S['mxstep'] = iw[5]
        if S['mxstep'] < 0:
            _mark_error(istate_ref, S)
            return
        if S['mxstep'] == 0:
            S['mxstep'] = 500
        S['mxhnil'] = iw[6]
        if S['mxhnil'] < 0:
            _mark_error(istate_ref, S)
            return
        if S['mxhnil'] == 0:
            S['mxhnil'] = 10
        h0 = rw[4]
        S['mxordn'] = iw[7]
        if S['mxordn'] < 0:
            _mark_error(istate_ref, S)
            return
        if S['mxordn'] == 0:
            S['mxordn'] = 100
        S['mxordn'] = min(S['mxordn'], 12)
        S['mxords'] = iw[8]
        if S['mxords'] < 0:
            _mark_error(istate_ref, S)
            return
        if S['mxords'] == 0:
            S['mxords'] = 100
        S['mxords'] = min(S['mxords'], 5)
        if (tout - tref[0]) * h0 < 0.0:
            _mark_error(istate_ref, S)
            return
        hmax = rw[5]
        if hmax < 0.0:
            _mark_error(istate_ref, S)
            return
        S['hmxi'] = 0.0
        if hmax > 0.0:
            S['hmxi'] = 1.0 / hmax
        hmin = rw[6]
        if hmin < 0.0:
            _mark_error(istate_ref, S)
            return

        # work array layout
        S['meth'] = 1
        S['nyh'] = S['n']
        S['lyh'] = 20
        len1n = 20 + (S['mxordn'] + 1) * S['nyh']
        len1s = 20 + (S['mxords'] + 1) * S['nyh']
        S['lwm'] = len1s
        # full n x n LU for all Jacobian types (see _prja for the deviation
        # from the C code's banded storage)
        lenwm = S['n'] * S['n'] + 2
        len1s += lenwm
        len1c = len1n
        if S['meth'] == 2:
            len1c = len1s
        len1 = max(len1n, len1s)
        len2 = 3 * S['n']
        lenrw = len1 + len2
        lenrwc = len1c + len2
        iw[16] = lenrw
        S['liwm'] = 0
        leniw = 20 + S['n']
        leniwc = 20
        if S['meth'] == 2:
            leniwc = leniw
        iw[17] = leniw
        if lrw < lenrwc:
            _mark_error(istate_ref, S)
            return
        if liw < leniwc:
            _mark_error(istate_ref, S)
            return
        S['lewt'] = len1
        S['insufr'] = 0
        if lrw < lenrw:
            S['insufr'] = 2
            S['lewt'] = len1c
        S['lsavf'] = S['lewt'] + S['n']
        S['lacor'] = S['lsavf'] + S['n']
        S['insufi'] = 0
        if liw < leniw:
            S['insufi'] = 2

        # 70: tolerance checks
        rtoli = rtol
        atoli = atol
        for i in range(S['n']):
            if itol >= 3:
                rtoli = rtol[i]
            if itol in (2, 4):
                atoli = atol[i]
            if rtoli < 0.0:
                _mark_error(istate_ref, S)
                return
            if atoli < 0.0:
                _mark_error(istate_ref, S)
                return

        # block c: remaining initializations, initial f call, initial h0
        S['uround'] = 2.220446049250313e-16  # DBL_EPSILON
        S['tn'] = tref[0]
        S['tsw'] = tref[0]
        S['maxord'] = S['mxordn']

        tcrit = 0.0
        if itask in (4, 5):
            tcrit = rw[0]
            if (tcrit - tout) * (tout - tref[0]) < 0.0:
                _mark_error(istate_ref, S)
                return
            if h0 != 0.0 and (tref[0] + h0 - tcrit) * h0 > 0.0:
                h0 = tcrit - tref[0]

        S['jstart'] = 0
        S['nhnil'] = 0
        S['nst'] = 0
        S['nje'] = 0
        S['nslast'] = 0
        S['hu'] = 0.0
        S['nqu'] = 0
        S['mused'] = 0
        S['miter'] = 0
        S['ccmax'] = 0.3
        S['maxcor'] = 3
        S['msbp'] = 20
        S['mxncf'] = 10

        # initial call to f (lf0 points at yh(*,2))
        lf0 = S['lyh'] + S['nyh']
        fv = cb.f(tref[0], y)
        if neq[0] == -1:
            istate_ref[0] = -1
            return
        S['nfe'] = 1
        for i in range(S['n']):
            rw[lf0 + i] = fv[i]
        for i in range(S['n']):
            rw[i + S['lyh']] = y[i]

        # load and invert the error weights (h temporarily 1.0)
        S['nq'] = 1
        S['h'] = 1.0
        _ewset(S['n'], itol, rtol, atol, rw, S['lyh'], S['lewt'])
        for i in range(S['n']):
            if rw[S['lewt'] + i] <= 0.0:
                _mark_error(istate_ref, S)
                return
            rw[S['lewt'] + i] = 1.0 / rw[S['lewt'] + i]

        # 120: compute the initial step size h0
        if h0 == 0.0:
            tdist = abs(tout - tref[0])
            w0 = max(abs(tref[0]), abs(tout))
            if tdist < 2.0 * S['uround'] * w0:
                _mark_error(istate_ref, S)
                return
            tol = rtol
            if itol > 2:
                for i in range(S['n']):
                    tol = max(tol, rtol[i])
            if tol <= 0.0:
                atoli = atol
                for i in range(S['n']):
                    if itol in (2, 4):
                        atoli = atol[i]
                    if abs(y[i]) != 0.0:
                        tol = max(tol, atoli / abs(y[i]))
            tol = max(tol, 100.0 * S['uround'])
            tol = min(tol, 0.001)
            ssum = _vmnorm(S['n'], rw, lf0, rw, S['lewt'])
            ssum = 1.0 / (tol * w0 * w0) + tol * ssum * ssum
            h0 = 1.0 / math.sqrt(ssum)
            h0 = min(h0, tdist)
            h0 = math.copysign(h0, tout - tref[0])

        # 180: honor the hmax bound
        rh_ = abs(h0) * S['hmxi']
        if rh_ > 1.0:
            h0 = h0 / rh_
        S['h'] = h0
        for i in range(S['n']):
            rw[i + lf0] = h0 * rw[i + lf0]
        initial_jump = 0

    elif istate_ref[0] == 3:
        # ---- change of parameters (block b, restart) ----
        S['ntrep'] = 0

        neq_in = len(y)
        if neq_in <= 0:
            _mark_error(istate_ref, S)
            return
        if neq_in > S['n']:
            _mark_error(istate_ref, S)
            return
        S['n'] = neq_in
        if jt == 3 or jt < 1 or jt > 5:
            _mark_error(istate_ref, S)
            return
        S['jtyp'] = jt
        if jt > 2:
            ml = iw[0]
            mu = iw[1]
            if ml < 0 or ml >= neq_in:
                _mark_error(istate_ref, S)
                return
            if mu < 0 or mu >= neq_in:
                _mark_error(istate_ref, S)
                return

        # optional inputs (iopt is always 1 here)
        S['ixpr'] = iw[4]
        if S['ixpr'] < 0 or S['ixpr'] > 1:
            _mark_error(istate_ref, S)
            return
        S['mxstep'] = iw[5]
        if S['mxstep'] < 0:
            _mark_error(istate_ref, S)
            return
        if S['mxstep'] == 0:
            S['mxstep'] = 500
        S['mxhnil'] = iw[6]
        if S['mxhnil'] < 0:
            _mark_error(istate_ref, S)
            return
        if S['mxhnil'] == 0:
            S['mxhnil'] = 10
        hmax = rw[5]
        if hmax < 0.0:
            _mark_error(istate_ref, S)
            return
        S['hmxi'] = 0.0
        if hmax > 0.0:
            S['hmxi'] = 1.0 / hmax
        hmin = rw[6]
        if hmin < 0.0:
            _mark_error(istate_ref, S)
            return

        # work array layout (sizes still based on the old nyh, as in C)
        S['lyh'] = 20
        len1n = 20 + (S['mxordn'] + 1) * S['nyh']
        len1s = 20 + (S['mxords'] + 1) * S['nyh']
        S['lwm'] = len1s
        lenwm = S['n'] * S['n'] + 2
        len1s += lenwm
        len1c = len1n
        if S['meth'] == 2:
            len1c = len1s
        len1 = min(len1n, len1s)
        len2 = 3 * S['n']
        lenrw = len1 + len2
        lenrwc = len1c + len2
        iw[16] = lenrw
        S['liwm'] = 0
        leniw = 20 + S['n']
        leniwc = 20
        if S['meth'] == 2:
            leniwc = leniw
        iw[17] = leniw
        if lrw < lenrwc or liw < leniwc:
            istate_ref[0] = -7
            for i in range(S['n']):
                y[i] = rw[i + S['lyh']]
            tref[0] = S['tn']
            finish()
            return
        S['lewt'] = len1
        S['insufr'] = 0
        if lrw < lenrw:
            S['insufr'] = 2
            S['lewt'] = len1c
        S['lsavf'] = S['lewt'] + S['n']
        S['lacor'] = S['lsavf'] + S['n']
        S['insufi'] = 0
        if liw < leniw:
            S['insufi'] = 2

        # 70: tolerance checks
        rtoli = rtol
        atoli = atol
        for i in range(S['n']):
            if itol >= 3:
                rtoli = rtol[i]
            if itol in (2, 4):
                atoli = atol[i]
            if rtoli < 0.0:
                _mark_error(istate_ref, S)
                return
            if atoli < 0.0:
                _mark_error(istate_ref, S)
                return

        # flag a parameter change to stoda; zero unused part of yh if n shrank
        S['jstart'] = -1
        if S['n'] != S['nyh']:
            i1 = S['lyh'] + S['l'] * S['nyh']
            i2 = S['lyh'] + (S['maxord'] + 1) * S['nyh']
            if i1 <= i2:
                for i in range(i1, i2):
                    rw[i] = 0.0
    # istate == 2: plain continuation (fall into the pre-step checks)

    # ---- pre-step stop checks (continuation calls only) ----
    if istate_ref[0] != 1:
        S['nslast'] = S['nst']

        if itask == 1:
            if (S['tn'] - tout) * S['h'] < 0.0:
                pass  # go to the main loop (250)
            else:
                iflag = _intdy(S, rw, tout, 0, y)
                if iflag != 0:
                    _mark_error(istate_ref, S)
                    return
                tref[0] = tout
                istate_ref[0] = 2
                finish()
                return
        elif itask == 3:
            tp = S['tn'] - S['hu'] * (1.0 + 100.0 * S['uround'])
            if (tp - tout) * S['h'] > 0.0:
                _mark_error(istate_ref, S)
                return
            if (S['tn'] - tout) * S['h'] < 0.0:
                pass  # go to the main loop
            else:
                tref[0] = S['tn']
                for i in range(S['n']):
                    y[i] = rw[i + S['lyh']]
                if itask in (4, 5) and ihit:
                    tref[0] = tcrit
                istate_ref[0] = 2
                finish()
                return
        elif itask == 4:
            tcrit = rw[0]
            if (S['tn'] - tcrit) * S['h'] > 0.0:
                _mark_error(istate_ref, S)
                return
            if (tcrit - tout) * S['h'] < 0.0:
                _mark_error(istate_ref, S)
                return
            if (S['tn'] - tout) * S['h'] < 0.0:
                hmx = abs(S['tn']) + abs(S['h'])
                ihit = 1 if (abs(S['tn'] - tcrit) <=
                             100.0 * S['uround'] * hmx) else 0
                if ihit:
                    for i in range(S['n']):
                        y[i] = rw[i + S['lyh']]
                    tref[0] = tcrit
                    istate_ref[0] = 2
                    finish()
                    return
                tnext = S['tn'] + S['h'] * (1.0 + 4.0 * S['uround'])
                if (tnext - tcrit) * S['h'] > 0.0:
                    S['h'] = (tcrit - S['tn']) * (1.0 - 4.0 * S['uround'])
                    if istate_ref[0] == 2 and S['jstart'] >= 0:
                        S['jstart'] = -2
                # go to the main loop
            else:
                iflag = _intdy(S, rw, tout, 0, y)
                if iflag != 0:
                    _mark_error(istate_ref, S)
                    return
                tref[0] = tout
                istate_ref[0] = 2
                finish()
                return
        elif itask == 5:
            tcrit = rw[0]
            if (S['tn'] - tcrit) * S['h'] > 0.0:
                _mark_error(istate_ref, S)
                return
            hmx = abs(S['tn']) + abs(S['h'])
            ihit = 1 if (abs(S['tn'] - tcrit) <=
                         100.0 * S['uround'] * hmx) else 0
            if ihit:
                for i in range(S['n']):
                    y[i] = rw[i + S['lyh']]
                tref[0] = tcrit
                istate_ref[0] = 2
                finish()
                return
            tnext = S['tn'] + S['h'] * (1.0 + 4.0 * S['uround'])
            if (tnext - tcrit) * S['h'] > 0.0:
                S['h'] = (tcrit - S['tn']) * (1.0 - 4.0 * S['uround'])
                if istate_ref[0] == 2 and S['jstart'] >= 0:
                    S['jstart'] = -2
            # go to the main loop

    # ---- block e: the main step loop (label 250) ----
    while True:
        if initial_jump:
            # method switched since the last successful step: check the
            # workspace can now accommodate the new method
            if S['meth'] != S['mused']:
                if S['insufr'] == 1 or S['insufi'] == 1:
                    istate_ref[0] = -7
                    for i in range(S['n']):
                        y[i] = rw[i + S['lyh']]
                    tref[0] = S['tn']
                    finish()
                    return
            # too many steps on this call
            if (S['nst'] - S['nslast']) >= S['mxstep']:
                istate_ref[0] = -1
                for i in range(S['n']):
                    y[i] = rw[i + S['lyh']]
                tref[0] = S['tn']
                finish()
                return
            # refresh the error weights (inverted)
            _ewset(S['n'], itol, rtol, atol, rw, S['lyh'], S['lewt'])
            for i in range(S['n']):
                if rw[i + S['lewt']] <= 0.0:
                    istate_ref[0] = -6
                    for j in range(S['n']):
                        y[j] = rw[j + S['lyh']]
                    tref[0] = S['tn']
                    finish()
                    return
                rw[i + S['lewt']] = 1.0 / rw[i + S['lewt']]
        initial_jump = 1

        # 270: check for excessive accuracy
        tolsf = S['uround'] * _vmnorm(S['n'], rw, S['lyh'], rw, S['lewt'])
        if tolsf > 0.01:
            tolsf = tolsf * 200.0
            if S['nst'] == 0:
                rw[13] = tolsf
                _mark_error(istate_ref, S)
                return
            istate_ref[0] = -2
            for j in range(S['n']):
                y[j] = rw[j + S['lyh']]
            tref[0] = S['tn']
            rw[13] = tolsf
            finish()
            return

        # 280: count steps that are nil at the roundoff level of t
        if abs(S['h']) <= (math.nextafter(abs(S['tn']), 2.0 * abs(S['tn']))
                           - abs(S['tn'])):
            S['nhnil'] += 1

        # 290: the core one-step integrator
        _stoda(cb, S, rw, iw, y, neq)
        if neq[0] == -1:
            istate_ref[0] = -3
            return

        # block f: result processing
        if S['kflag'] == -1 or S['kflag'] == -2:
            istate_ref[0] = -4 if S['kflag'] == -1 else -5
            big = 0.0
            imxer = 0
            for i in range(S['n']):
                size = abs(rw[i + S['lacor']] * rw[i + S['lewt']])
                if size > big:
                    big = size
                    imxer = i + 1
            iw[15] = imxer
            for j in range(S['n']):
                y[j] = rw[j + S['lyh']]
            tref[0] = S['tn']
            finish()
            return

        # 300: a successful step
        S['init'] = 1
        if S['meth'] != S['mused']:
            S['tsw'] = S['tn']
            S['maxord'] = S['mxords'] if S['meth'] == 2 else S['mxordn']
            if S['meth'] == 2:
                rw[S['lwm']] = math.sqrt(S['uround'])
            S['insufr'] = min(S['insufr'], 1)
            S['insufi'] = min(S['insufi'], 1)
            S['jstart'] = -1

        if itask == 1:
            # integrate to tout: interpolate when it has been reached
            if (S['tn'] - tout) * S['h'] < 0.0:
                continue
            iflag = _intdy(S, rw, tout, 0, y)
            if iflag != 0:
                _mark_error(istate_ref, S)
                return
            tref[0] = tout
            istate_ref[0] = 2
            finish()
            return
        elif itask == 2:
            for i in range(S['n']):
                y[i] = rw[i + S['lyh']]
            tref[0] = S['tn']
            if itask in (4, 5) and ihit:
                tref[0] = tcrit
            istate_ref[0] = 2
            finish()
            return
        elif itask == 3:
            if (S['tn'] - tout) * S['h'] >= 0.0:
                for i in range(S['n']):
                    y[i] = rw[i + S['lyh']]
                tref[0] = S['tn']
                if itask in (4, 5) and ihit:
                    tref[0] = tcrit
                istate_ref[0] = 2
                finish()
                return
            continue
        elif itask == 4:
            # stop at the earlier of tout and tcrit
            if (S['tn'] - tout) * S['h'] >= 0.0:
                iflag = _intdy(S, rw, tout, 0, y)
                tref[0] = tout
                istate_ref[0] = 2
                finish()
                return
            hmx = abs(S['tn']) + abs(S['h'])
            ihit = 1 if (abs(S['tn'] - tcrit) <=
                         100.0 * S['uround'] * hmx) else 0
            if ihit:
                for i in range(S['n']):
                    y[i] = rw[i + S['lyh']]
                tref[0] = tcrit
                istate_ref[0] = 2
                finish()
                return
            tnext = S['tn'] + S['h'] * (1.0 + 4.0 * S['uround'])
            if (tnext - tcrit) * S['h'] <= 0.0:
                continue
            S['h'] = (tcrit - S['tn']) * (1.0 - 4.0 * S['uround'])
            if S['jstart'] >= 0:
                S['jstart'] = -2
            continue
        else:  # itask == 5
            hmx = abs(S['tn']) + abs(S['h'])
            ihit = 1 if (abs(S['tn'] - tcrit) <=
                         100.0 * S['uround'] * hmx) else 0
            for i in range(S['n']):
                y[i] = rw[i + S['lyh']]
            tref[0] = S['tn']
            if ihit:
                tref[0] = tcrit
            istate_ref[0] = 2
            finish()
            return


def lsoda(fun, y, t, tout, rtol, atol, itask, istate, rwork, iwork, jac, jt,
          f_params, tfirst, jac_params, state_doubles, state_ints):
    """Pure-Python LSODA one-step wrapper.

    Signature mirrors the C ``_odepack.lsoda`` wrapper used by the ode class:

        lsoda(fun, y, t, tout, rtol, atol, itask, istate, rwork, iwork,
              jac, jt, f_params, tfirst, jac_params, state_doubles,
              state_ints)

    ``y`` is updated in place; returns ``(y, t, istate)``.
    """
    n = len(y)
    ylist = [float(v) for v in np.asarray(y).tolist()]
    rw = [float(v) for v in rwork.tolist()]
    iw = [int(v) for v in iwork.tolist()]

    S = _new_state()
    _unpack_state(state_doubles, state_ints, S)

    neq = [1]
    cb = _LsodaCallbacks(fun, jac, f_params, jac_params, tfirst, neq)
    istate_ref = [int(istate)]
    tref = [float(t)]

    _lsoda_core(cb, S, rw, iw, ylist, neq, itask, istate_ref, rtol, atol,
                float(tout), tref, jt)

    # write back the work arrays and the internal state
    for i in range(len(rw)):
        rwork[i] = rw[i]
    for i in range(len(iw)):
        iwork[i] = iw[i]
    _pack_state(S, state_doubles, state_ints)
    for i in range(n):
        y[i] = ylist[i]

    if neq[0] == -1 and cb.last_error is not None:
        # the C wrapper prints the callback traceback and reports the
        # failure through istate; do the same
        import traceback
        traceback.print_exception(cb.last_error)

    return y, tref[0], istate_ref[0]


class _Runner:
    """Non-descriptor callable wrapper for the runner functions.

    ``_ode.py`` stores the runner as a class attribute
    (``runner = lsoda_step``) and calls it through
    the integrator instance (``self.runner(...)``).  A plain Python
    function in a class dict becomes a bound method on instance access
    and would inject the integrator as a first argument; a plain
    callable object has no ``__get__`` and is returned unchanged.
    """

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args):
        return self._fn(*args)


_lsoda = lsoda
lsoda = _Runner(_lsoda)
