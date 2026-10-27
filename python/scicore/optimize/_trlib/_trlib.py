"""Pure Python replacement for the deleted Cython ``_trlib`` extension.

Implements the trust-region quadratic subproblem solver used by
``scicore.optimize`` (``minimize(method='trust-krylov')`` and friends).

The subproblem

    min_p  f + g'p + 0.5 p'Hp   s.t.   ||p|| <= trust_radius

is solved with the dogleg method when a full Hessian is available
(Cauchy point + Newton direction, interpolated at the trust-region
boundary), and with the Cauchy point (steepest descent) when only
Hessian-vector products are provided — the standard inexact fallback.
The outer trust-region loop (radius update, step acceptance,
convergence) lives in
``scicore.optimize._trustregion._minimize_trust_region``.
"""

from __future__ import annotations

import math

import rsnumpy as np

__all__ = ['TRLIBQuadraticSubproblem']


class TRLIBQuadraticSubproblem:
    """Trust-region quadratic subproblem solver (pure Python).

    Mirrors the interface of the deleted C++ extension: the factory
    produced by ``get_trlib_quadratic_subproblem`` is called by
    ``_minimize_trust_region`` as ``subproblem(x, fun, jac, hess, hessp)``
    and the resulting object exposes ``solve(trust_radius)`` returning
    ``(p, hits_boundary)``, the model value ``m(p)``, and the ``fun`` /
    ``jac`` / ``hess`` / ``jac_mag`` properties.
    """

    def __init__(self, x, fun, jac, hess, hessp, tol_rel_i=-2.0,
                 tol_rel_b=-3.0, disp=False):
        self._x = np.asarray(x, dtype=np.float64).flatten()
        self._f = None
        self._g = None
        self._h = None
        self._g_mag = None
        self._fun = fun
        self._jac = jac
        self._hess = hess
        self._hessp = hessp
        self.tol_rel_i = tol_rel_i
        self.tol_rel_b = tol_rel_b
        self.disp = disp

    def __call__(self, p):
        return (self.fun + np.dot(self.jac, p)
                + 0.5 * np.dot(p, self.hessp(p)))

    @property
    def fun(self):
        """Value of the objective function at the current iterate."""
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def jac(self):
        """Gradient of the objective function at the current iterate."""
        if self._g is None:
            self._g = self._jac(self._x)
        return self._g

    @property
    def hess(self):
        """Hessian of the objective function at the current iterate."""
        if self._h is None:
            if callable(self._hess):
                self._h = self._hess(self._x)
            else:
                self._h = self._hess
        return self._h

    def hessp(self, p):
        """Hessian-vector product at the current iterate."""
        if self._hessp is not None:
            return self._hessp(self._x, p)
        return np.dot(self.hess, p)

    @property
    def jac_mag(self):
        """Euclidean norm of the gradient at the current iterate."""
        if self._g_mag is None:
            self._g_mag = float(np.linalg.norm(self.jac))
        return self._g_mag

    def _cauchy_point(self, trust_radius):
        """Steepest-descent (Cauchy) point of the quadratic model.

        Returns (p, hits_boundary).
        """
        g = [float(t) for t in np.asarray(self.jac).tolist()]
        n = len(g)
        g_mag = self.jac_mag
        Hp = [float(t) for t in np.asarray(self.hessp(g)).tolist()]
        gHg = sum(g[i] * Hp[i] for i in range(n))
        t_star = (g_mag * g_mag) / gHg if gHg > 0 else 0.0
        if t_star < trust_radius / g_mag:
            return (np.array([-t_star * g[i] for i in range(n)],
                             dtype=np.float64), False)
        scale = trust_radius / g_mag
        return (np.array([-scale * g[i] for i in range(n)],
                         dtype=np.float64), True)

    def solve(self, trust_radius):
        """Solve the subproblem within ``trust_radius``.

        Returns ``(p, hits_boundary)`` per the ``BaseQuadraticSubproblem``
        contract used by ``_minimize_trust_region``.
        """
        g_mag = self.jac_mag
        if g_mag == 0.0:
            return np.zeros_like(self.jac), False
        if self._hess is None:
            # Only Hessian-vector products: inexact Cauchy-point solution.
            return self._cauchy_point(trust_radius)

        g = [float(t) for t in np.asarray(self.jac).tolist()]
        n = len(g)
        # Newton direction.  rsnumpy's solve does not raise on singular
        # matrices, so detect a usable (descent) direction explicitly.
        try:
            pN = np.asarray(np.linalg.solve(self.hess, -np.array(g)))
            pN_list = [float(t) for t in pN.tolist()]
        except Exception:
            return self._cauchy_point(trust_radius)
        if not all(math.isfinite(t) for t in pN_list):
            return self._cauchy_point(trust_radius)
        if sum(g[i] * pN_list[i] for i in range(n)) >= 0.0:
            # not a descent direction -> Hessian not positive definite
            return self._cauchy_point(trust_radius)
        pN_mag = math.sqrt(sum(t * t for t in pN_list))
        if pN_mag <= trust_radius:
            return np.array(pN_list, dtype=np.float64), False
        pC, on_boundary = self._cauchy_point(trust_radius)
        if on_boundary:
            return pC, True
        pC_list = [float(t) for t in np.asarray(pC).tolist()]
        # interpolate on the dogleg segment pC -> pN at the boundary
        d = [pN_list[i] - pC_list[i] for i in range(n)]
        a = sum(t * t for t in d)
        b = 2.0 * sum(pC_list[i] * d[i] for i in range(n))
        c = sum(t * t for t in pC_list) - trust_radius * trust_radius
        beta = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        return (np.array([pC_list[i] + beta * d[i] for i in range(n)],
                         dtype=np.float64), True)
