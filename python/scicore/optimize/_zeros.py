"""Pure-Python replacement for the deleted C ``_zeros`` extension.

Provides the C-convention root-finding kernels (``_brentq``, ``_brenth``,
``_ridder``, ``_bisect``) that ``_zeros_py`` calls, implemented in pure
Python.  Each kernel takes
``(f, a, b, xtol, rtol, maxiter, args, full_output, disp)`` and returns
either the root or ``(root, funcalls, iterations, flag)`` depending on
``full_output``; ``flag`` is 1 on convergence, 0 otherwise.  When the
iteration limit is hit without convergence and ``disp`` is truthy, a
warning message is printed (matching the C extension's behaviour).
"""

from __future__ import annotations

import rsnumpy as np


def _report(failure_msg):
    print(failure_msg)


def _bisect(f, a, b, xtol, rtol, maxiter, args, full_output, disp):
    a = float(a)
    b = float(b)
    fa = float(f(a, *args))
    fb = float(f(b, *args))
    funcalls = 2
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError("f(a) and f(b) must be finite.")
    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs.")
    for it in range(1, maxiter + 1):
        d = 0.5 * (b - a)
        x = a + d
        if abs(d) <= max(xtol, rtol * abs(x)) or fa == 0.0:
            root = a if fa == 0.0 else x
            if full_output:
                return (root, funcalls, it, 1)
            return root
        fx = float(f(x, *args))
        funcalls += 1
        if fx * fa <= 0.0:
            b, fb = x, fx
        else:
            a, fa = x, fx
    if disp:
        _report(f"Failed to converge after {maxiter} iterations.")
    root = 0.5 * (a + b)
    if full_output:
        return (root, funcalls, maxiter, 0)
    return root


def _ridder(f, a, b, xtol, rtol, maxiter, args, full_output, disp):
    a = float(a)
    b = float(b)
    fa = float(f(a, *args))
    fb = float(f(b, *args))
    funcalls = 2
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError("f(a) and f(b) must be finite.")
    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs.")
    for it in range(1, maxiter + 1):
        d = 0.5 * (b - a)
        xm = a + d
        fm = float(f(xm, *args))
        funcalls += 1
        if fm * fa <= 0.0:
            b, fb = xm, fm
        else:
            a, fa = xm, fm
        if abs(d) <= max(xtol, rtol * abs(xm)) or fm == 0.0:
            root = xm
            if full_output:
                return (root, funcalls, it, 1)
            return root
        e = d * (fb - fa) / (fb - fm)
        if (fa - fm) * (fb - fm) > 0.0:
            x = xm + e
            if x <= a or x >= b:
                x = 0.5 * (a + b)
        else:
            x = 0.5 * (a + b)
        fx = float(f(x, *args))
        funcalls += 1
        if fx * fa <= 0.0:
            b, fb = x, fx
        else:
            a, fa = x, fx
    if disp:
        _report(f"Failed to converge after {maxiter} iterations.")
    root = 0.5 * (a + b)
    if full_output:
        return (root, funcalls, maxiter, 0)
    return root


def _brentq(f, a, b, xtol, rtol, maxiter, args, full_output, disp):
    a = float(a)
    b = float(b)
    fa = float(f(a, *args))
    fb = float(f(b, *args))
    funcalls = 2
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError("f(a) and f(b) must be finite.")
    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs.")
    if fa == 0.0:
        if full_output:
            return (a, funcalls, 0, 1)
        return a
    if fb == 0.0:
        if full_output:
            return (b, funcalls, 0, 1)
        return b

    # c holds the function value of the bracket endpoint opposite to the
    # "current" estimate a (the C extension's convention).
    c = a
    fc = fa
    d = e = b - a
    for it in range(1, maxiter + 1):
        xr = 0.5 * (b - a)
        if abs(xr) <= max(xtol, rtol * abs(xr)) or fc == 0.0:
            root = a
            if full_output:
                return (root, funcalls, it, 1)
            return root
        if abs(e) <= max(xtol, rtol * abs(e)) or fc * fb >= 0.0:
            # bisection
            e_new = xr
        else:
            # inverse quadratic interpolation (the C extension's form)
            A = fc
            B = fb
            if A == B:
                e_new = d
            else:
                C = A / B
                A = 3.0 * xr * C / (B + C) + d * B
                e_new = A / (2.0 * (A - 0.5 * B))
            if abs(e_new) < 0.25 * abs(d) or e_new < 0.5 * abs(e):
                e_new = xr
        d = e
        e = xr - e_new
        x = a + e_new
        fx = float(f(x, *args))
        funcalls += 1
        if fx * fa <= 0.0:
            b, fb = x, fx
        else:
            c, fc = a, fa
            a, fa = x, fx
    if disp:
        _report(f"Failed to converge after {maxiter} iterations.")
    root = 0.5 * (a + b)
    if full_output:
        return (root, funcalls, maxiter, 0)
    return root


def _brenth(f, a, b, xtol, rtol, maxiter, args, full_output, disp):
    a = float(a)
    b = float(b)
    fa = float(f(a, *args))
    fb = float(f(b, *args))
    funcalls = 2
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError("f(a) and f(b) must be finite.")
    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs.")
    if fa == 0.0:
        if full_output:
            return (a, funcalls, 0, 1)
        return a
    if fb == 0.0:
        if full_output:
            return (b, funcalls, 0, 1)
        return b

    c = a
    fc = fa
    d = e = b - a
    for it in range(1, maxiter + 1):
        xr = 0.5 * (b - a)
        if abs(xr) <= max(xtol, rtol * abs(xr)) or fc == 0.0:
            root = a
            if full_output:
                return (root, funcalls, it, 1)
            return root
        if abs(e) <= max(xtol, rtol * abs(e)):
            # bisection
            e_new = xr
        else:
            # hyperbolic interpolation (the C extension's form)
            A = fc
            B = fb
            if A == B:
                e_new = d
            elif A == -B:
                e_new = 2.0 * xr
            else:
                C = A / B
                A = 3.0 * xr * C / (B + C) + d * B
                e_new = 2.0 * xr * C / (C + 1.0)
                if A < 0.0:
                    A = -A
                if e_new < 0.0:
                    e_new = -e_new
                if A + B < C:
                    e_new = (A + B) / C * (e_new - xr) + xr
                else:
                    e_new = 2.0 * (A + B) / C * e_new + xr
            if abs(e_new) < 0.25 * abs(d) or e_new < 0.5 * abs(e):
                e_new = xr
        d = e
        e = xr - e_new
        x = a + e_new
        fx = float(f(x, *args))
        funcalls += 1
        if fx * fa <= 0.0:
            b, fb = x, fx
        else:
            c, fc = a, fa
            a, fa = x, fx
    if disp:
        _report(f"Failed to converge after {maxiter} iterations.")
    root = 0.5 * (a + b)
    if full_output:
        return (root, funcalls, maxiter, 0)
    return root
