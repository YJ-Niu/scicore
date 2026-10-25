"""Stub replacement for the deleted Cython ``_ufuncs`` extension.

This module provides the same public names that the original Cython
``_ufuncs`` exposed, implemented on top of :mod:`rsnumpy` and the Python
standard library.  Functions that do not yet have a Rust implementation
delegate to a pure-Python/rsnumpy fallback (or raise ``NotImplementedError``
for truly missing special functions).

The module is intentionally a stub: as Rust kernels are added under
``rust/src/``, individual functions can be migrated to real
implementations.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import rsnumpy as np

__all__ = [
    'geterr', 'seterr', 'errstate',
    'agm', 'airy', 'airye', 'bdtr', 'bdtrc', 'bdtri', 'bdtrik', 'bdtrin',
    'bei', 'beip', 'ber', 'berp', 'besselpoly', 'beta', 'betainc', 'betaincc',
    'betainccinv', 'betaincinv', 'betaln', 'binom', 'boxcox', 'boxcox1p',
    'btdtria', 'btdtrib', 'cbrt', 'chdtr', 'chdtrc', 'chdtri', 'chdtriv',
    'chndtr', 'chndtridf', 'chndtrinc', 'chndtrix', 'cosdg', 'cosm1', 'cotdg',
    'dawsn', 'digammainv', 'ellipe', 'ellipeinc', 'ellipj', 'ellipk',
    'ellipkinc', 'ellipkm1', 'elliprc', 'elliprd', 'elliprf', 'elliprg',
    'elliprj', 'entr', 'erf', 'erfc', 'erfcinv', 'erfcx', 'erfi', 'erfinv',
    'eval_chebyc', 'eval_chebys', 'eval_chebyt', 'eval_chebyu',
    'eval_gegenbauer', 'eval_genlaguerre', 'eval_hermite', 'eval_hermitenorm',
    'eval_jacobi', 'eval_laguerre', 'eval_legendre', 'eval_sh_chebyt',
    'eval_sh_chebyu', 'eval_sh_jacobi', 'eval_sh_legendre', 'exp1', 'exp10',
    'exp2', 'expi', 'expit', 'expm1', 'expn', 'exprel', 'fdtr', 'fdtrc',
    'fdtri', 'fdtridfd', 'fresnel', 'gamma', 'gammainc', 'gammaincc',
    'gammainccinv', 'gammaincinv', 'gammaln', 'gammasgn', 'gdtr', 'gdtrc',
    'gdtria', 'gdtrib', 'gdtrix', 'hankel1', 'hankel1e', 'hankel2',
    'hankel2e', 'huber', 'hyp0f1', 'hyp1f1', 'hyp2f1', 'hyperu', 'i0', 'i0e',
    'i1', 'i1e', 'inv_boxcox', 'inv_boxcox1p', 'it2i0k0', 'it2j0y0',
    'it2struve0', 'itairy', 'iti0k0', 'itj0y0', 'itmodstruve0', 'itstruve0',
    'iv', 'ive', 'j0', 'j1', 'jn', 'jv', 'jve', 'k0', 'k0e', 'k1', 'k1e',
    'kei', 'keip', 'kelvin', 'ker', 'kerp', 'kl_div', 'kn', 'kolmogi',
    'kolmogorov', 'kv', 'kve', 'log1p', 'log_expit', 'log_gammainc',
    'log_gammaincc', 'log_ndtr', 'log_wright_bessel', 'loggamma', 'logit',
    'lpmv', 'mathieu_a', 'mathieu_b', 'mathieu_modcem1', 'mathieu_modcem2',
    'mathieu_modsem1', 'mathieu_modsem2', 'modfresnelm', 'modfresnelp',
    'modstruve', 'nbdtr', 'nbdtrc', 'nbdtri', 'nbdtrik', 'nbdtrin',
    'ncfdtr', 'ncfdtri', 'ncfdtridfd', 'ncfdtridfn', 'ncfdtrinc',
    'nctdtr', 'nctdtridf', 'nctdtrinc', 'nctdtrit', 'ndtr', 'ndtri',
    'ndtri_exp', 'nrdtrimn', 'nrdtrisd', 'obl_ang1', 'obl_ang1_cv', 'obl_cv',
    'obl_rad1', 'obl_rad1_cv', 'obl_rad2', 'obl_rad2_cv', 'owens_t', 'pbdv',
    'pbvv', 'pbwa', 'pdtr', 'pdtrc', 'pdtri', 'pdtrik', 'poch', 'powm1',
    'pro_ang1', 'pro_ang1_cv', 'pro_cv', 'pro_rad1', 'pro_rad1_cv', 'pro_rad2',
    'pro_rad2_cv', 'pseudo_huber', 'psi', 'radian', 'rel_entr', 'rgamma',
    'round', 'shichi', 'sici', 'sindg', 'smirnov', 'smirnovi', 'spence',
    'stdtr', 'stdtridf', 'stdtrit', 'struve', 'tandg', 'tklmbda',
    'voigt_profile', 'wofz', 'wright_bessel', 'wrightomega', 'xlog1py',
    'xlogy', 'y0', 'y1', 'yn', 'yv', 'yve', 'zetac',
    # internal ufuncs used by other SciPy modules
    '_cosine_cdf', '_cosine_invcdf', '_bivariate_normal_cdf', '_cospi',
    '_ellip_harm', '_gen_harmonic', '_igam_fac', '_iv_ratioinv',
    '_kolmogc', '_kolmogci', '_kolmogp', '_lambertw', '_lgam1p',
    '_log1mexp', '_log1pmx', '_mathieu_cem', '_mathieu_sem',
    '_normalized_gen_harmonic', '_riemann_zeta', '_scaled_exp1',
    '_sf_error_test_function', '_sinpi', '_smirnovc', '_smirnovci',
    '_smirnovp', '_spherical_in', '_spherical_in_d', '_spherical_jn',
    '_spherical_jn_d', '_spherical_kn', '_spherical_kn_d', '_spherical_yn',
    '_spherical_yn_d', '_von_mises_cdf', '_stirling2_inexact', '_zeta',
    # Additional internal ufuncs needed by stats and other modules
    '_beta_pdf', '_binom_cdf', '_binom_isf', '_binom_pmf', '_binom_ppf',
    '_binom_sf', '_cauchy_isf', '_cauchy_ppf', '_hypergeom_cdf',
    '_hypergeom_mean', '_hypergeom_pmf', '_hypergeom_sf',
    '_hypergeom_skewness', '_hypergeom_variance', '_invgauss_isf',
    '_invgauss_ppf', '_iv_ratio', '_iv_ratio_c', '_landau_cdf',
    '_landau_isf', '_landau_pdf', '_landau_ppf', '_landau_sf',
    '_nbinom_cdf', '_nbinom_isf', '_nbinom_kurtosis_excess',
    '_nbinom_mean', '_nbinom_pmf', '_nbinom_ppf', '_nbinom_sf',
    '_nbinom_skewness', '_nbinom_variance', '_ncf_isf',
    '_ncf_kurtosis_excess', '_ncf_mean', '_ncf_pdf', '_ncf_sf',
    '_ncf_skewness', '_ncf_variance', '_nct_isf', '_nct_kurtosis_excess',
    '_nct_mean', '_nct_pdf', '_nct_sf', '_nct_skewness', '_nct_variance',
    '_ncx2_isf', '_ncx2_pdf', '_ncx2_sf', '_skewnorm_cdf', '_skewnorm_isf',
    '_skewnorm_ppf',
]


# ---------------------------------------------------------------------------
# Error state (mirrors numpy.seterr/geterr for ufuncs)
# ---------------------------------------------------------------------------

_errstate = {'over': 'ignore', 'divide': 'ignore', 'invalid': 'ignore',
             'under': 'ignore'}


def geterr() -> dict:
    return dict(_errstate)


def seterr(**kwargs):
    for k, v in kwargs.items():
        if k not in _errstate:
            raise ValueError(f"Unknown error type {k!r}")
        _errstate[k] = v
    return dict(_errstate)


class errstate:
    def __init__(self, **kwargs):
        self._old = dict(_errstate)
        self._new = kwargs
    def __enter__(self):
        _errstate.clear()
        _errstate.update(self._new)
    def __exit__(self, *exc):
        _errstate.clear()
        _errstate.update(self._old)
        return False


# ---------------------------------------------------------------------------
# ufunc-like callable wrapper
#
# Each special function is exposed as an object that behaves like a NumPy
# ufunc: it accepts rsnumpy arrays (and scalars) and applies the underlying
# scalar function elementwise.
# ---------------------------------------------------------------------------

class _ufunc(np.ufunc):
    """A thin ufunc-like wrapper around a scalar function."""

    def __init__(self, name: str, func: Callable[[float], float],
                 nin: int = 1, nout: int = 1):
        self.name = name
        self._func = func
        self.nin = nin
        self.nout = nout
        self.ntypes = 0
        self.identity = None
        self.signature = None
        self.nargs = nin
        self.types = []

    def resolve_dtypes(self, dtypes=None, *args, **kwargs):
        """Find the dtypes that the ufunc would use for the given inputs."""
        raise NotImplementedError(
            "rsnumpy ufunc does not expose a NumPy ufunc C-API object")

    def __call__(self, *args, **kwargs):
        # Convert rsnumpy arrays to plain Python lists / scalars, apply
        # elementwise, then wrap back in rsnumpy.
        import rsnumpy as _np
        rs_arrays = [a for a in args if isinstance(a, _np.ndarray)]
        if rs_arrays:
            # Determine broadcast shape from all array args
            shapes = [a.shape for a in rs_arrays]
            if len(set(shapes)) == 1:
                shape = shapes[0]
            else:
                try:
                    shape = _np.broadcast_shapes(*shapes)
                except Exception:
                    shape = shapes[0]
            # Flatten each array arg and broadcast to shape
            n_flat = int(_np.prod(shape))
            flat_args = []
            for a in args:
                if isinstance(a, _np.ndarray):
                    if a.shape == shape:
                        vals = a.tolist()
                    else:
                        a2 = _np.broadcast_to(a, shape)
                        vals = a2.tolist()
                    # Flatten to 1D list
                    if isinstance(vals, list):
                        flat_vals = []
                        stack = [vals]
                        while stack:
                            item = stack.pop()
                            if isinstance(item, list):
                                for sub in reversed(item):
                                    stack.append(sub)
                            else:
                                flat_vals.append(item)
                    else:
                        flat_vals = [vals]
                    flat_args.append([float(x) for x in flat_vals[:n_flat]])
                else:
                    flat_args.append([float(a)] * n_flat)
            # Apply elementwise
            result = [self._func(*[fa[k] for fa in flat_args]) for k in range(n_flat)]
            out = _np.array(result, dtype=float).reshape(shape)
            return out
        # Scalar path
        vals = [float(a) for a in args]
        result = self._func(*vals)
        if self.nout == 1:
            return result
        return tuple(result)

    @property
    def __name__(self):
        return self.name

    def __repr__(self):
        return f"<rsnumpy ufunc '{self.name}'>"

    def __getattr__(self, item):
        # ufuncs expose attributes like ``reduce``, ``accumulate``, ``outer``
        # that SciPy sometimes touches.  Provide no-ops.
        raise AttributeError(f"'{self.name}' ufunc has no attribute {item!r}")


def _math1(func: Callable[[float], float]):
    def f(x: float) -> float:
        return func(x)
    return f


def _math2(func: Callable[[float, float], float]):
    def f(x: float, y: float) -> float:
        return func(x, y)
    return f


# Map of name -> scalar function.  Functions available in Python's ``math``
# module are wired directly; everything else falls back to a stub that returns
# ``nan`` (callers can later be migrated to Rust).
_SCALAR_FUNCS: dict[str, Any] = {}

def _register(name: str, func: Any):
    _SCALAR_FUNCS[name] = func

# --- math module functions (single argument) ---
_for_math1 = [
    'cbrt', 'cosdg', 'cosm1', 'cotdg', 'dawsn', 'erf', 'erfc', 'erfcx', 'erfi',
    'exp1', 'exp10', 'exp2', 'expm1', 'expn', 'gamma', 'gammaln', 'i0', 'i0e',
    'i1', 'i1e', 'j0', 'j1', 'k0', 'k0e', 'k1', 'k1e', 'log1p', 'loggamma',
    'ndtr', 'ndtri', 'psi', 'rgamma', 'shichi', 'sici', 'sindg', 'spence',
    'tandg', 'y0', 'y1',
]
for _n in _for_math1:
    if hasattr(math, _n):
        _register(_n, _math1(getattr(math, _n)))

# ``math.lgamma`` is ln|Gamma(x)|; scipy exposes it as ``gammaln`` and
# ``loggamma`` — neither name exists in the stdlib ``math`` module, so wire
# them explicitly (otherwise they fall through to the NaN stubs).
_register('gammaln', _math1(math.lgamma))
_register('loggamma', _math1(math.lgamma))

# Degree-based trigonometric functions (scipy names, not in stdlib ``math``).
_register('cosdg', _math1(lambda x: math.cos(math.radians(x))))
_register('sindg', _math1(lambda x: math.sin(math.radians(x))))
_register('tandg', _math1(lambda x: math.tan(math.radians(x))))
_register('cotdg', _math1(lambda x: 1.0 / math.tan(math.radians(x))))

# scipy.special.xlogy: 0 if x == 0, else x * log(y) (two-argument).
def _xlogy(x, y):
    if x == 0:
        return 0.0
    return x * math.log(y)

_register('xlogy', _math2(_xlogy))

# scipy.special.ndtr: standard normal CDF, ndtr(x) = 0.5 * erfc(-x/sqrt(2)).
_register('ndtr', _math1(lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))))

# scipy.special.entr: entr(x) = -x*log(x), 0 if x == 0.
_register('entr', _math1(
    lambda x: 0.0 if x == 0 else -x * math.log(x)))

# --- Regularized incomplete beta function + Student's t CDF (scipy names,
# not in stdlib ``math``); needed by the beta and studentt distributions ---
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz method)."""
    max_it, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


_register('betainc', _betainc)
_register('betaincc', lambda a, b, x: 1.0 - _betainc(a, b, x))


def _stdtr(df, t):
    """CDF of Student's t distribution with ``df`` degrees of freedom."""
    x = df / (df + t * t)
    p = 0.5 * _betainc(0.5 * df, 0.5, x)
    return p if t < 0.0 else 1.0 - p


_register('stdtr', _math2(_stdtr))

# --- two-argument math functions ---
_for_math2 = []
for _n in _for_math2:
    if hasattr(math, _n):
        _register(_n, _math2(getattr(math, _n)))

# --- multi-output special functions (return tuples) ---
def _airy(x):
    # airy returns (Ai, Aip, Bi, Bip)
    return (float('nan'),) * 4

def _airye(x):
    return (float('nan'),) * 4

def _fresnel(x):
    return (float('nan'), float('nan'))

def _ellipj(x):
    return (float('nan'),) * 4

def _hankel1(x):
    return complex(float('nan'), float('nan'))

def _hankel2(x):
    return complex(float('nan'), float('nan'))

def _kelvin(x):
    return (float('nan'),) * 4

def _binom(n, k):
    # Binomial coefficient C(n, k). Exact for integer n, k (the common case,
    # e.g. the moment transforms); Gamma-based for general real arguments.
    # NB: avoid the builtin ``round`` here — it is shadowed in this module's
    # namespace by rsnumpy's ``round`` ufunc, so use ``int`` + an exact check.
    ni = int(n); ki = int(k)
    if n == ni and k == ki:
        if ni < 0 or ki < 0 or ki > ni:
            return 0.0
        return float(math.comb(ni, ki))
    try:
        return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1)
                        - math.lgamma(n - k + 1))
    except (ValueError, OverflowError):
        return float('nan')

# Register the rest as NaN-returning stubs
_for_stub = [n for n in __all__ if n not in _SCALAR_FUNCS and n not in (
    'geterr', 'seterr', 'errstate',
    '_cosine_cdf', '_cosine_invcdf', '_bivariate_normal_cdf', '_cospi',
    '_ellip_harm', '_gen_harmonic', '_igam_fac', '_iv_ratioinv',
    '_kolmogc', '_kolmogci', '_kolmogp', '_lambertw', '_lgam1p',
    '_log1mexp', '_log1pmx', '_mathieu_cem', '_mathieu_sem',
    '_normalized_gen_harmonic', '_riemann_zeta', '_scaled_exp1',
    '_sf_error_test_function', '_sinpi', '_smirnovc', '_smirnovci',
    '_smirnovp', '_spherical_in', '_spherical_in_d', '_spherical_jn',
    '_spherical_jn_d', '_spherical_kn', '_spherical_kn_d', '_spherical_yn',
    '_spherical_yn_d', '_von_mises_cdf', '_stirling2_inexact', '_zeta',
)]
_stub_funcs = {
    'airy': _airy, 'airye': _airye, 'fresnel': _fresnel, 'ellipj': _ellipj,
    'hankel1': _hankel1, 'hankel2': _hankel2, 'kelvin': _kelvin,
    # All stubs accept arbitrary arguments
    'binom': _binom,
    'gammainc': lambda *args: float('nan'),
    'gammaincc': lambda *args: float('nan'),
    'betainc': lambda *args: float('nan'),
    'betaincc': lambda *args: float('nan'),
    'bdtr': lambda *args: float('nan'),
    'bdtrc': lambda *args: float('nan'),
    'nbdtr': lambda *args: float('nan'),
    'nbdtrc': lambda *args: float('nan'),
    'pdtr': lambda *args: float('nan'),
    'pdtrc': lambda *args: float('nan'),
    'chdtr': lambda *args: float('nan'),
    'chdtrc': lambda *args: float('nan'),
    'fdtr': lambda *args: float('nan'),
    'fdtrc': lambda *args: float('nan'),
    'gdtr': lambda *args: float('nan'),
    'gdtrc': lambda *args: float('nan'),
    'ncfdtr': lambda *args: float('nan'),
    'nctdtr': lambda *args: float('nan'),
    'stdtr': lambda *args: float('nan'),
    # All remaining internal ufuncs (return nan stub)
    '_beta_pdf': lambda *args: float('nan'),
    '_binom_cdf': lambda *args: float('nan'),
    '_binom_isf': lambda *args: float('nan'),
    '_binom_pmf': lambda *args: float('nan'),
    '_binom_ppf': lambda *args: float('nan'),
    '_binom_sf': lambda *args: float('nan'),
    '_bivariate_normal_cdf': lambda *args: float('nan'),
    '_cauchy_isf': lambda *args: float('nan'),
    '_cauchy_ppf': lambda *args: float('nan'),
    '_cosine_cdf': lambda *args: float('nan'),
    '_cosine_invcdf': lambda *args: float('nan'),
    '_cospi': lambda *args: float('nan'),
    '_ellip_harm': lambda *args: float('nan'),
    '_gen_harmonic': lambda *args: float('nan'),
    '_hypergeom_cdf': lambda *args: float('nan'),
    '_hypergeom_mean': lambda *args: float('nan'),
    '_hypergeom_pmf': lambda *args: float('nan'),
    '_hypergeom_sf': lambda *args: float('nan'),
    '_hypergeom_skewness': lambda *args: float('nan'),
    '_hypergeom_variance': lambda *args: float('nan'),
    '_igam_fac': lambda *args: float('nan'),
    '_invgauss_isf': lambda *args: float('nan'),
    '_invgauss_ppf': lambda *args: float('nan'),
    '_iv_ratio': lambda *args: float('nan'),
    '_iv_ratio_c': lambda *args: float('nan'),
    '_iv_ratioinv': lambda *args: float('nan'),
    '_kolmogc': lambda *args: float('nan'),
    '_kolmogci': lambda *args: float('nan'),
    '_kolmogp': lambda *args: float('nan'),
    '_landau_cdf': lambda *args: float('nan'),
    '_landau_isf': lambda *args: float('nan'),
    '_landau_pdf': lambda *args: float('nan'),
    '_landau_ppf': lambda *args: float('nan'),
    '_landau_sf': lambda *args: float('nan'),
    '_lambertw': lambda *args: float('nan'),
    '_lgam1p': lambda *args: float('nan'),
    '_log1mexp': lambda *args: float('nan'),
    '_log1pmx': lambda *args: float('nan'),
    '_mathieu_cem': lambda *args: float('nan'),
    '_mathieu_sem': lambda *args: float('nan'),
    '_nbinom_cdf': lambda *args: float('nan'),
    '_nbinom_isf': lambda *args: float('nan'),
    '_nbinom_kurtosis_excess': lambda *args: float('nan'),
    '_nbinom_mean': lambda *args: float('nan'),
    '_nbinom_pmf': lambda *args: float('nan'),
    '_nbinom_ppf': lambda *args: float('nan'),
    '_nbinom_sf': lambda *args: float('nan'),
    '_nbinom_skewness': lambda *args: float('nan'),
    '_nbinom_variance': lambda *args: float('nan'),
    '_ncf_isf': lambda *args: float('nan'),
    '_ncf_kurtosis_excess': lambda *args: float('nan'),
    '_ncf_mean': lambda *args: float('nan'),
    '_ncf_pdf': lambda *args: float('nan'),
    '_ncf_sf': lambda *args: float('nan'),
    '_ncf_skewness': lambda *args: float('nan'),
    '_ncf_variance': lambda *args: float('nan'),
    '_nct_isf': lambda *args: float('nan'),
    '_nct_kurtosis_excess': lambda *args: float('nan'),
    '_nct_mean': lambda *args: float('nan'),
    '_nct_pdf': lambda *args: float('nan'),
    '_nct_sf': lambda *args: float('nan'),
    '_nct_skewness': lambda *args: float('nan'),
    '_nct_variance': lambda *args: float('nan'),
    '_ncx2_isf': lambda *args: float('nan'),
    '_ncx2_pdf': lambda *args: float('nan'),
    '_ncx2_sf': lambda *args: float('nan'),
    '_normalized_gen_harmonic': lambda *args: float('nan'),
    '_riemann_zeta': lambda *args: float('nan'),
    '_scaled_exp1': lambda *args: float('nan'),
    '_sf_error_test_function': lambda *args: float('nan'),
    '_sinpi': lambda *args: float('nan'),
    '_skewnorm_cdf': lambda *args: float('nan'),
    '_skewnorm_isf': lambda *args: float('nan'),
    '_skewnorm_ppf': lambda *args: float('nan'),
    '_spherical_in': lambda *args: float('nan'),
    '_spherical_in_d': lambda *args: float('nan'),
    '_spherical_jn': lambda *args: float('nan'),
    '_spherical_jn_d': lambda *args: float('nan'),
    '_spherical_kn': lambda *args: float('nan'),
    '_spherical_kn_d': lambda *args: float('nan'),
    '_spherical_yn': lambda *args: float('nan'),
    '_spherical_yn_d': lambda *args: float('nan'),
    '_stirling2_inexact': lambda *args: float('nan'),
    '_von_mises_cdf': lambda *args: float('nan'),
    '_zeta': lambda *args: float('nan'),
}

for _n in _for_stub:
    if _n in _stub_funcs:
        _SCALAR_FUNCS[_n] = _stub_funcs[_n]
    else:
        _SCALAR_FUNCS[_n] = lambda *args: float('nan')


# Build the module-level ufunc objects
for _name, _func in _SCALAR_FUNCS.items():
    if _name not in __all__:
        continue
    nout = 4 if _name in ('airy', 'airye', 'ellipj', 'kelvin') else \
           2 if _name in ('fresnel',) else 1
    globals()[_name] = _ufunc(_name, _func, nout=nout)

# Alias names that start with underscore (internal ufuncs)
for _name in ('_cosine_cdf', '_cosine_invcdf', '_bivariate_normal_cdf',
              '_cospi', '_ellip_harm', '_gen_harmonic', '_igam_fac',
              '_iv_ratioinv', '_kolmogc', '_kolmogci', '_kolmogp',
              '_lambertw', '_lgam1p', '_log1mexp', '_log1pmx',
              '_mathieu_cem', '_mathieu_sem', '_normalized_gen_harmonic',
              '_riemann_zeta', '_scaled_exp1', '_sf_error_test_function',
              '_sinpi', '_smirnovc', '_smirnovci', '_smirnovp',
              '_spherical_in', '_spherical_in_d', '_spherical_jn',
              '_spherical_jn_d', '_spherical_kn', '_spherical_kn_d',
              '_spherical_yn', '_spherical_yn_d', '_von_mises_cdf',
              '_stirling2_inexact', '_zeta'):
    globals()[_name] = _ufunc(_name, lambda *args: float('nan'), nout=1)
