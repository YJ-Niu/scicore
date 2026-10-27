"""Pure-Python replacement for the deleted C ``_quadpack`` extension.

Implements the QUADPACK drivers used by ``scicore.integrate._quadpack_py``:

* ``_qagse`` — adaptive 10-point Gauss-Legendre quadrature with interval
  bisection on finite intervals (the error indicator compares the rule on
  an interval with the sum of its two halves);
* ``_qagpe`` — same, with user-supplied break points;
* ``_qagie`` — infinite intervals via a rational change of variables,
  delegated to ``_qagse``.

The weighted drivers (``_qawoe``/``_qawse``) are not yet ported.

Return conventions match the deleted C extension: ``full_output=0`` returns
``(result, abserr, ier)``; ``full_output=1`` returns
``(result, abserr, infodict, message, ier)``.
"""

from __future__ import annotations

import math

__all__ = ['_qagse', '_qagie', '_qagpe', '_qawoe', '_qawse']


def _stub(*args, **kwargs):
    raise NotImplementedError("_quadpack weighted drivers not yet ported")


_qawoe = _stub
_qawse = _stub


# ---------------------------------------------------------------------------
# 10-point Gauss-Legendre rule: nodes/weights are derived at import time
# from the Legendre three-term recurrence (Newton root finding), so there is
# no hard-coded table that could drift out of sync.
# ---------------------------------------------------------------------------

def _legendre_p_d(n):
    """Return (P_n, P'_n) evaluators built from the three-term recurrence."""

    def pval(x):
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        return p1

    def dval(x):
        # Q_n = P'_n satisfies P'_n = ((2n-1)(P_{n-1} + x Q_{n-1})
        #                             - (n-1) Q_{n-2}) / n.
        q0, q1, p0, p1 = 0.0, 1.0, 1.0, x
        for k in range(2, n + 1):
            qn = ((2 * k - 1) * (p1 + x * q1) - (k - 1) * q0) / k
            q0, q1 = q1, qn
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        return q1

    return pval, dval


def _gauss_legendre(n):
    """Nodes and weights of the n-point Gauss-Legendre rule on [-1, 1]."""
    pval, dval = _legendre_p_d(n)
    # One Chebyshev starting value per positive root; the j=0 value starts
    # just below 1 and converges to the largest root.
    m = (n + 1) // 2
    starts = [math.cos(math.pi * (j + 0.75) / (n + 0.5)) for j in range(m)]
    roots = []
    for s in starts:
        x = s
        for _ in range(80):
            f = pval(x)
            d = dval(x)
            if d == 0.0:
                break
            step = f / d
            x -= step
            if abs(step) <= 1.0e-15 * max(1.0, abs(x)):
                break
        if abs(pval(x)) > 1.0e-12:
            raise RuntimeError("Gauss-Legendre node computation failed")
        roots.append(x)
    pos_desc = sorted({r for r in roots}, reverse=True)
    if len(pos_desc) != m:
        raise RuntimeError("Gauss-Legendre node computation failed")
    nodes = [-x for x in pos_desc][::-1] + ([0.0] if n % 2 else []) + pos_desc[::-1]
    weights = [2.0 / ((1.0 - x * x) * dval(x) ** 2) for x in nodes]
    if abs(sum(weights) - 2.0) > 1.0e-9:
        raise RuntimeError("Gauss-Legendre weight computation failed")
    return nodes, weights


_NODES, _WEIGHTS = _gauss_legendre(10)


def _g10(f, a, b):
    """10-point Gauss-Legendre of ``f`` over [a, b]."""
    m = 0.5 * (b - a)
    c = 0.5 * (a + b)
    s = 0.0
    for x, w in zip(_NODES, _WEIGHTS):
        s += w * f(c + m * x)
    return m * s


def _qagse(func, a, b, args, full_output, epsabs, epsrel, limit):
    """Adaptive Gauss-Legendre quadrature on [a, b]."""
    a = float(a)
    b = float(b)
    args = tuple(args)
    if limit < 1:
        limit = 1
    neval = 0

    def feval(x):
        nonlocal neval
        neval += 1
        return float(func(x, *args))

    if a == b:
        result = 0.0
        abserr = 0.0
        ier = 0
        ivals = []
    else:
        result_whole = _g10(feval, a, b)
        xm = 0.5 * (a + b)
        v1 = _g10(feval, a, xm)
        v2 = _g10(feval, xm, b)
        result = v1 + v2
        abserr = abs(result_whole - result)

        def child_est(x1, x2, v):
            q = 0.5 * (x1 + x2)
            return abs(v - (_g10(feval, x1, q) + _g10(feval, q, x2)))

        e1 = child_est(a, xm, v1)
        e2 = child_est(xm, b, v2)
        ivals = [(a, xm, v1, e1), (xm, b, v2, e2)]
        ier = 0
        while True:
            gtol = max(float(epsabs), float(epsrel) * abs(result))
            if abserr <= gtol:
                ier = 0
                break
            if len(ivals) >= limit:
                ier = 1
                break
            kmax = max(range(len(ivals)), key=lambda i: ivals[i][3])
            x1, x2, val, est = ivals[kmax]
            if est <= gtol:
                ier = 0
                break
            xq1 = 0.5 * (x1 + x2)
            xq2 = 0.5 * (x2 + xq1)
            c1 = _g10(feval, x1, xq1)
            c2 = _g10(feval, xq1, x2)
            q1 = 0.5 * (x1 + xq1)
            q2 = 0.5 * (xq1 + x2)
            e1 = abs(c1 - (_g10(feval, x1, q1) + _g10(feval, q1, xq1)))
            e2 = abs(c2 - (_g10(feval, xq1, q2) + _g10(feval, q2, x2)))
            new_est = e1 + e2
            if new_est <= 0.5 * est:
                # The indicator does not decrease: roundoff level reached.
                ivals[kmax:kmax + 1] = [(x1, xq1, c1, e1), (xq1, x2, c2, e2)]
                result = sum(v for _, _, v, _ in ivals)
                abserr = sum(e for _, _, _, e in ivals)
                ier = 2
                break
            if est > 0.0 and new_est > est * (1.0 + 1.0e-4):
                # Error grows on subdivision: bad integrand behavior.
                ier = 3
                break
            ivals[kmax:kmax + 1] = [(x1, xq1, c1, e1), (xq1, x2, c2, e2)]
            result = sum(v for _, _, v, _ in ivals)
            abserr = sum(e for _, _, _, e in ivals)

    if ier == 0:
        message = ""
    elif ier == 1:
        message = "The maximum number of subdivisions has been achieved."
    elif ier == 2:
        message = "The occurrence of roundoff error is detected."
    else:
        message = "Extremely bad integrand behavior occurs at some points."

    if full_output:
        infodict = {
            'neval': neval,
            'nint': len(ivals),
            'last': (ivals[-1][0], ivals[-1][1]) if ivals else (a, b),
            'ptab': [v for iv in ivals for v in (iv[0], iv[1])],
            'elev': [1] * len(ivals),
            'errbnd': [iv[3] for iv in ivals],
            'iord': list(range(len(ivals))),
            'msg': message,
        }
        return (result, abserr, infodict, message, ier)
    return (result, abserr, ier)


def _qagpe(func, a, b, points, args, full_output, epsabs, epsrel, limit):
    """Adaptive quadrature on [a, b] with user-supplied break points."""
    a = float(a)
    b = float(b)
    pts = [float(p) for p in np_points(points)]
    pts = sorted({p for p in pts if a < p < b})
    bounds = [a] + pts + [b]
    n_int = len(bounds) - 1
    per = max(1, int(limit) // n_int)
    out = [
        _qagse(func, x1, x2, args, 1, epsabs, epsrel, per)
        for x1, x2 in zip(bounds[:-1], bounds[1:])
    ]
    result = sum(r[0] for r in out)
    abserr = sum(r[1] for r in out)
    ier = max(r[4] for r in out)
    message = max((r[3] for r in out if r[4] != 0), default="")
    if full_output:
        infodict = dict(out[0][2])
        infodict['neval'] = sum(r[2]['neval'] for r in out)
        infodict['nint'] = sum(r[2]['nint'] for r in out)
        infodict['msg'] = message
        return (result, abserr, infodict, message, ier)
    return (result, abserr, ier)


def _qagie(func, bound, infbounds, args, full_output, epsabs, epsrel, limit):
    """Adaptive quadrature on an infinite interval (via change of variable)."""
    bound = float(bound)

    def wrapped(t):
        if infbounds == 1:      # [bound, +inf)
            x = bound + t / (1.0 - t)
            jac = 1.0 / (1.0 - t) ** 2
        elif infbounds == -1:   # (-inf, bound]
            x = bound - t / (1.0 - t)
            jac = 1.0 / (1.0 - t) ** 2
        else:                   # (-inf, +inf)
            x = (2.0 * t - 1.0) / (t * (1.0 - t))
            jac = (2.0 * t * t - 2.0 * t + 1.0) / (t * (1.0 - t)) ** 2
        return float(func(x, *args)) * jac

    return _qagse(wrapped, 0.0, 1.0, (), full_output, epsabs, epsrel, limit)


def np_points(points):
    """Flatten ``points`` (rsnumpy array or sequence) to a list of floats."""
    try:
        return points.tolist()
    except AttributeError:
        return [float(p) for p in points]
