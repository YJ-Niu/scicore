"""Pure-rsnumpy reimplementation of the deleted C ``_nd_image`` extension.

The functions below are faithful ports of the scipy.ndimage C core
(``ni_filters.c``, ``ni_interpolation.c``, ``ni_morphology.c``,
``ni_splines.c``, ``ni_support.c``) onto :mod:`rsnumpy`.  They provide
the n-dimensional image processing routines used by
:mod:`scicore.ndimage`.

Extension-mode codes (see :func:`_ni_support._extend_mode_to_code`):

    0 nearest, 1 wrap, 2 reflect, 3 mirror, 4 constant,
    5 grid-wrap, 6 grid-constant
"""

from __future__ import annotations

import heapq
import math
from itertools import product

import rsnumpy as np

# ---------------------------------------------------------------------------
# extension-mode codes
# ---------------------------------------------------------------------------
_MODE_NEAREST = 0
_MODE_WRAP = 1
_MODE_REFLECT = 2
_MODE_MIRROR = 3
_MODE_CONSTANT = 4
_MODE_GRID_WRAP = 5
_MODE_GRID_CONSTANT = 6


# ---------------------------------------------------------------------------
# generic array helpers
# ---------------------------------------------------------------------------
def _to_float_list(arr):
    if isinstance(arr, list):
        return [float(v) for v in arr]
    if hasattr(arr, "tolist"):
        return [float(v) for v in arr.tolist()]
    return [float(arr[i]) for i in range(len(arr))]


def _get_elem(arr, idx):
    # Direct tuple indexing: rsnumpy chained getitem (arr[r][c]) returns a
    # copy, so chained *setitem* on the "row" would be a silent no-op.
    return arr[idx]


def _set_elem(arr, idx, val):
    arr[idx] = val


def _iter_lines(arr, axis):
    """Yield (other_axes_index_tuple, line_as_list) along ``axis``."""
    ndim = arr.ndim
    if ndim == 0:
        yield (), [float(arr)]
        return
    shape = list(arr.shape)
    others = [d for d in range(ndim) if d != axis]
    n_lines = 1
    for d in others:
        n_lines *= shape[d]
    for flat in range(n_lines):
        idx = [0] * ndim
        rem = flat
        for d in reversed(others):
            idx[d] = rem % shape[d]
            rem //= shape[d]
        line = []
        for i in range(shape[axis]):
            k = idx[:]
            k[axis] = i
            line.append(float(_get_elem(arr, tuple(k))))
        yield tuple(idx[d] for d in others), line


def _unravel(flat, shape):
    idx = []
    for dim in reversed(shape):
        idx.append(flat % dim)
        flat //= dim
    return tuple(reversed(idx))


def _normalize_origin(origin, ndim):
    if isinstance(origin, int):
        return [origin] * ndim
    return list(origin)


# ---------------------------------------------------------------------------
# boundary extension
# (scipy ni_support.c NI_InitFilterOffsets / NI_ExtendLine semantics)
# ---------------------------------------------------------------------------
def _boundary_index(cc, n, mode):
    """Map a possibly out-of-range integer index ``cc`` into ``[0, n)``.

    Implements the per-axis boundary conditions of the scipy C core
    (integer form).  Returns ``None`` when the index is outside the array
    and the extension mode is constant (caller substitutes ``cval``).
    """
    if mode in (_MODE_CONSTANT, _MODE_GRID_CONSTANT):
        if 0 <= cc < n:
            return cc
        return None
    if n <= 1:
        return 0
    if 0 <= cc < n:
        return cc
    if mode == _MODE_MIRROR:
        # scipy mirror: d e d c b a | a b c d e d
        # (edge duplicated, period 2*n)
        sz2 = 2 * n
        t = cc % sz2
        return t if t < n else sz2 - 1 - t
    if mode == _MODE_REFLECT:
        # scipy reflect: d e d c b | a b c d e d
        # (edge not duplicated, period 2*(n-1))
        sz2 = 2 * n - 2
        t = cc % sz2
        return t if t <= n - 1 else sz2 - t
    if mode in (_MODE_WRAP, _MODE_GRID_WRAP):
        return cc % n
    if mode == _MODE_NEAREST:
        return 0 if cc < 0 else n - 1
    return None


def _line_value(line, k, n, mode, cval):
    idx = _boundary_index(k, n, mode)
    if idx is None:
        return float(cval)
    return line[idx]


# ---------------------------------------------------------------------------
# 1-D filters (scipy ni_filters.c)
# ---------------------------------------------------------------------------
def correlate1d(input, weights, axis, output, mode, cval, origin=0):
    """Correlate the lines of ``input`` along ``axis`` with ``weights``."""
    w = _to_float_list(weights)
    W = len(w)
    half = W // 2
    for other, line in _iter_lines(input, axis):
        n = len(line)
        idx = list(other)
        idx.insert(axis, 0)
        for i in range(n):
            s = 0.0
            for j in range(W):
                s += w[j] * _line_value(line, i + j - half - origin, n, mode, cval)
            k = idx[:]
            k[axis] = i
            _set_elem(output, tuple(k), s)


def uniform_filter1d(input, size, axis, output, mode, cval, origin=0):
    """Uniform (moving-average) filter along ``axis`` with window ``size``."""
    size = int(size)
    half = size // 2
    for other, line in _iter_lines(input, axis):
        n = len(line)
        idx = list(other)
        idx.insert(axis, 0)
        for i in range(n):
            s = 0.0
            for j in range(size):
                s += _line_value(line, i + j - half - origin, n, mode, cval)
            k = idx[:]
            k[axis] = i
            _set_elem(output, tuple(k), s / size)


def min_or_max_filter1d(input, size, axis, output, mode, cval, origin=0,
                        minimum=1):
    """Sliding min/max filter along ``axis`` with window ``size``."""
    size = int(size)
    half = size // 2
    for other, line in _iter_lines(input, axis):
        n = len(line)
        idx = list(other)
        idx.insert(axis, 0)
        for i in range(n):
            if size == 1:
                best = line[i]
            else:
                best = None
                for j in range(size):
                    v = _line_value(line, i + j - half - origin, n, mode, cval)
                    if best is None:
                        best = v
                    elif minimum:
                        if v < best:
                            best = v
                    else:
                        if v > best:
                            best = v
            k = idx[:]
            k[axis] = i
            _set_elem(output, tuple(k), best)


# ---------------------------------------------------------------------------
# n-D filters with a footprint (scipy ni_filters.c NI_Correlate /
# NI_MinOrMaxFilter)
# ---------------------------------------------------------------------------
def correlate(input, weights, output, mode, cval, origin=0):
    """Full n-D correlation with a (possibly non-separable) weight array."""
    wshape = list(weights.shape)
    ndim = input.ndim
    origins = _normalize_origin(origin, ndim)
    in_shape = list(input.shape)
    terms = []
    for sp in product(*[range(d) for d in wshape]):
        wv = float(_get_elem(weights, sp))
        if wv != 0.0:
            terms.append((sp, wv))
    n_out = 1
    for d in in_shape:
        n_out *= d
    for flat in range(n_out):
        p = _unravel(flat, in_shape)
        s = 0.0
        for sp, wv in terms:
            mapped = []
            is_cval = False
            for h in range(ndim):
                cc = p[h] + sp[h] - wshape[h] // 2 - origins[h]
                e = _boundary_index(cc, in_shape[h], mode)
                if e is None:
                    is_cval = True
                    break
                mapped.append(e)
            if is_cval:
                s += wv * float(cval)
            else:
                s += wv * float(_get_elem(input, tuple(mapped)))
        _set_elem(output, p, s)


def min_or_max_filter(input, footprint, structure, output, mode, cval,
                      origins=(), minimum=1):
    """Min/max filter with a (possibly non-separable) footprint."""
    origins = list(origins)
    fshape = list(footprint.shape)
    ndim = input.ndim
    if structure is not None:
        sshape = list(structure.shape)
        terms = []
        for flat in range(footprint.size):
            sp = _unravel(flat, fshape)
            if footprint[sp]:
                sv = float(_get_elem(structure, sp))
                terms.append((sp, -sv if minimum else sv))
    else:
        terms = []
        for flat in range(footprint.size):
            sp = _unravel(flat, fshape)
            if footprint[sp]:
                terms.append((sp, 0.0))
    in_shape = list(input.shape)
    n_out = 1
    for d in in_shape:
        n_out *= d
    for flat in range(n_out):
        p = _unravel(flat, in_shape)
        best = None
        for sp, sv in terms:
            mapped = []
            is_cval = False
            for h in range(ndim):
                cc = p[h] + sp[h] - fshape[h] // 2 - origins[h]
                e = _boundary_index(cc, in_shape[h], mode)
                if e is None:
                    is_cval = True
                    break
                mapped.append(e)
            if is_cval:
                v = float(cval)
            else:
                v = float(_get_elem(input, tuple(mapped)))
            v += sv
            if best is None:
                best = v
            elif minimum:
                if v < best:
                    best = v
            else:
                if v > best:
                    best = v
        _set_elem(output, p, best)


# ---------------------------------------------------------------------------
# spline filter machinery (scipy ni_splines.c)
# ---------------------------------------------------------------------------
_SPLINE_POLES = {
    2: [-0.171572875253809902396622551580603843],
    3: [-0.267949192431122706472553658494127633],
    4: [-0.361341225900220177092212841325675255,
        -0.013725429297339121360331226939128204],
    5: [-0.430575347099973791851434783493520110,
        -0.043096288203264653822712376822550182],
}


def _spline_weights(x, order):
    """Interpolation weights at position ``x`` for a spline of ``order``.

    Port of get_spline_interpolation_weights (ni_splines.c).
    """
    x -= math.floor(x if (order & 1) else x + 0.5)
    y = x
    z = 1.0 - x
    w = [0.0] * (order + 1)
    if order == 1:
        w[0] = 1.0 - x
    elif order == 2:
        w[1] = 0.75 - x * x
        y = 0.5 - x
        w[0] = 0.5 * y * y
    elif order == 3:
        w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0
        w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0
        w[0] = z * z * z / 6.0
    elif order == 4:
        t = x * x
        w[2] = t * (t * 0.25 - 0.625) + 115.0 / 192.0
        y = 1.0 + x
        w[1] = y * (y * (y * (5.0 - y) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0
        z = 1.0 - x
        w[3] = z * (z * (z * (5.0 - z) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0
        y = 0.5 - x
        t = y * y
        w[0] = t * t / 24.0
    else:  # order == 5
        t = y * y
        w[2] = t * (t * (0.25 - y / 12.0) - 0.5) + 0.55
        t = z * z
        w[3] = t * (t * (0.25 - z / 12.0) - 0.5) + 0.55
        y += 1.0
        w[1] = y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425
        z += 1.0
        w[4] = z * (z * (z * (z * (z / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425
        y = 1.0 - x
        t = y * y
        w[0] = y * t * t / 120.0
    # All interpolation weights add to 1.0, so use it for the last one.
    w[order] = 1.0
    for i in range(order):
        w[order] -= w[i]
    return w


def _init_causal_mirror(c, n, z):
    z_i = z
    z_n_1 = z ** (n - 1)
    c[0] = c[0] + z_n_1 * c[n - 1]
    for i in range(1, n - 1):
        c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i])
        z_i *= z
    c[0] /= 1 - z_n_1 * z_n_1


def _init_anticausal_mirror(c, n, z):
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1)


def _init_causal_wrap(c, n, z):
    z_i = z
    for i in range(1, n):
        c[0] += z_i * c[n - i]
        z_i *= z
    c[0] /= 1 - z_i  # z_i == z**n


def _init_anticausal_wrap(c, n, z):
    z_i = z
    for i in range(0, n - 1):
        c[n - 1] += z_i * c[i]
        z_i *= z
    c[n - 1] *= z / (z_i - 1)  # z_i == z**n


def _init_causal_reflect(c, n, z):
    z_i = z
    z_n = z ** n
    s = c[0] + z_n * c[n - 1]
    for i in range(1, n):
        s += z_i * (c[i] + z_n * c[n - 1 - i])
        z_i *= z
    c[0] += s * z / (1 - z_n * z_n)


def _init_anticausal_reflect(c, n, z):
    c[n - 1] *= z / (z - 1)


def _spline_apply_filter(c, n, z, causal_init, anticausal_init):
    causal_init(c, n, z)
    for i in range(1, n):
        c[i] += z * c[i - 1]
    anticausal_init(c, n, z)
    for i in range(n - 2, -1, -1):
        c[i] = z * (c[i + 1] - c[i])


def spline_filter1d(input, order, axis, output, mode):
    """1-D spline filter (Thévenaz IIR prefilter) along ``axis``."""
    order = int(order)
    poles = _SPLINE_POLES[order]
    if mode in (_MODE_GRID_CONSTANT, _MODE_CONSTANT, _MODE_MIRROR, _MODE_WRAP):
        causal, anticausal = _init_causal_mirror, _init_anticausal_mirror
    elif mode == _MODE_GRID_WRAP:
        causal, anticausal = _init_causal_wrap, _init_anticausal_wrap
    else:  # NEAREST, REFLECT
        causal, anticausal = _init_causal_reflect, _init_anticausal_reflect
    for other, line in _iter_lines(input, axis):
        c = list(line)
        n = len(c)
        if n > 1:
            gain = 1.0
            for z in poles:
                gain *= (1.0 - z) * (1.0 - 1.0 / z)
            for i in range(n):
                c[i] *= gain
            for z in poles:
                _spline_apply_filter(c, n, z, causal, anticausal)
        idx = list(other)
        idx.insert(axis, 0)
        for i, v in enumerate(c):
            k = idx[:]
            k[axis] = i
            _set_elem(output, tuple(k), v)


# ---------------------------------------------------------------------------
# geometric transforms (scipy ni_interpolation.c)
# ---------------------------------------------------------------------------
def _map_coordinate(val, n, mode):
    """Map a (possibly fractional) coordinate outside [0, n) per boundary
    condition.  Port of map_coordinate (ni_interpolation.c)."""
    if val < 0:
        if mode == _MODE_MIRROR:
            if n <= 1:
                return 0.0
            sz2 = 2.0 * n - 2
            val = math.fmod(val, sz2)
            val = val + sz2 if val <= 1 - n else -val
        elif mode == _MODE_REFLECT:
            if n <= 1:
                return 0.0
            sz2 = 2.0 * n
            if val < -sz2:
                val = math.fmod(val, sz2)
            val = val + sz2 if val < -n else (1e-15 if val > -1e-15 else -val) - 1
        elif mode == _MODE_WRAP:
            if n <= 1:
                return 0.0
            sz = n - 1
            val = math.fmod(val, sz) + sz
        elif mode == _MODE_GRID_WRAP:
            if n <= 1:
                return 0.0
            val = math.fmod(val, n)
            if val < 0:
                val += n
        elif mode == _MODE_NEAREST:
            val = 0.0
        elif mode == _MODE_CONSTANT:
            val = -1.0
    elif val > n - 1:
        if mode == _MODE_MIRROR:
            if n <= 1:
                return 0.0
            sz2 = 2.0 * n - 2
            val = math.fmod(val, sz2)
            if val >= n:
                val = sz2 - val
        elif mode == _MODE_REFLECT:
            if n <= 1:
                return 0.0
            sz2 = 2.0 * n
            val = math.fmod(val, sz2)
            if val >= n:
                val = sz2 - val - 1
        elif mode == _MODE_WRAP:
            if n <= 1:
                return 0.0
            val = math.fmod(val, n - 1)
        elif mode == _MODE_GRID_WRAP:
            if n <= 1:
                return 0.0
            val = math.fmod(val, n)
        elif mode == _MODE_NEAREST:
            val = n - 1
        elif mode == _MODE_CONSTANT:
            val = -1.0
    return val


def _spline_boundary_mode(mode):
    # Modes without an analytic prefilter use mirror extension for the
    # spline footprint (scipy _get_spline_boundary_mode).
    if mode in (_MODE_CONSTANT, _MODE_WRAP):
        return _MODE_MIRROR
    return mode


def _axis_setup(cc, n, order, mode, spline_mode):
    """Per-axis setup shared by geometric_transform / zoom_shift.

    Returns ``(constant_flag, idxs, weights)`` where ``idxs`` is a list of
    ``order + 1`` input indices (raw, possibly out of range, for
    grid-constant) and ``weights`` the spline weights (None if order == 0).
    """
    if mode != _MODE_GRID_CONSTANT and mode != _MODE_NEAREST:
        cc = _map_coordinate(cc, n, mode)
    else:
        cc = min(max(cc, -order - 1.0), n + order + 1.0)
    if cc > -1.0 or mode == _MODE_GRID_CONSTANT or mode == _MODE_NEAREST:
        if order & 1:
            start = int(math.floor(cc)) - order // 2
        else:
            start = int(math.floor(cc + 0.5)) - order // 2
        if start < 0 or start + order >= n:
            if mode == _MODE_GRID_CONSTANT:
                idxs = [start + ll for ll in range(order + 1)]
            else:
                idxs = [int(_map_coordinate(start + ll, n, spline_mode))
                        for ll in range(order + 1)]
        else:
            idxs = [start + ll for ll in range(order + 1)]
        w = _spline_weights(cc, order) if order > 0 else None
        return 0, idxs, w
    return 1, None, None


def _accumulate(input, idim, per_axis, order, mode, cval):
    """Sum the (order+1)**ndim filter contributions for one output pixel."""
    ndim = len(idim)
    t = 0.0
    for ff in product(range(order + 1), repeat=ndim):
        is_cval = 0
        if mode == _MODE_GRID_CONSTANT:
            for hh in range(ndim):
                ix = per_axis[hh][0][ff[hh]]
                if ix < 0 or ix >= idim[hh]:
                    is_cval = 1
                    break
        if is_cval:
            coeff = float(cval)
        else:
            coeff = float(_get_elem(input, tuple(
                per_axis[hh][0][ff[hh]] for hh in range(ndim))))
        for hh in range(ndim):
            if order > 0:
                coeff *= per_axis[hh][1][ff[hh]]
        t += coeff
    return t


def geometric_transform(input, mapping, matrix, coordinates, offset,
                        output, order, mode, cval, nprepad,
                        extra_arguments=(), extra_keywords=None):
    """Apply an arbitrary geometric transform (spline interpolation).

    Matches the ported calling convention: ``matrix`` carries the coordinate
    array (from ``map_coordinates``, shape ``(ndim, *output_shape)``) and
    ``coordinates`` carries the affine matrix (from ``affine_transform``,
    shape ``(ndim, ndim)``) together with ``offset``.
    """
    if extra_keywords is None:
        extra_keywords = {}
    order = int(order)
    idim = list(input.shape)
    ndim = input.ndim
    oshape = list(output.shape)
    spline_mode = _spline_boundary_mode(mode)
    if offset is None:
        offset_list = [0.0] * ndim
    else:
        try:
            offset_list = [float(offset[i]) for i in range(ndim)]
        except (TypeError, IndexError):
            offset_list = [float(offset)] * ndim
    n_out = 1
    for d in oshape:
        n_out *= d
    for flat in range(n_out):
        out_idx = _unravel(flat, oshape)
        if mapping is not None:
            icoor = [float(v) for v in mapping(
                tuple(out_idx), *(extra_arguments or ()), **extra_keywords)]
        elif matrix is not None:
            # coordinate array (map_coordinates)
            icoor = [float(_get_elem(matrix, (ax,) + out_idx))
                     for ax in range(ndim)]
        elif coordinates is not None:
            # affine matrix (affine_transform): icoor = M @ out_idx + offset
            icoor = []
            for ax in range(ndim):
                s = offset_list[ax]
                for j in range(ndim):
                    s += float(_get_elem(coordinates, (ax, j))) * out_idx[j]
                icoor.append(s)
        else:
            raise RuntimeError(
                "one of mapping, matrix or coordinates must be provided")
        for hh in range(ndim):
            if not math.isfinite(icoor[hh]):
                raise ValueError("coordinates must be finite")
        constant = 0
        per_axis = []
        for hh in range(ndim):
            flag, idxs, w = _axis_setup(icoor[hh] + nprepad, idim[hh],
                                        order, mode, spline_mode)
            if flag:
                constant = 1
                break
            per_axis.append((idxs, w))
        if constant:
            val = float(cval)
        else:
            val = _accumulate(input, idim, per_axis, order, mode, cval)
        _set_elem(output, out_idx, val)


def zoom_shift(input, zoom, shift, output, order, mode, cval, nprepad,
               grid_mode):
    """Zoom/shift transform (NI_ZoomShift): linear output->input mapping."""
    order = int(order)
    idim = list(input.shape)
    rank = input.ndim
    oshape = list(output.shape)
    spline_mode = _spline_boundary_mode(mode)
    zooms = [float(v) for v in zoom] if zoom is not None else None
    shifts = [float(v) for v in shift] if shift is not None else None
    axis_data = []
    for jj in range(rank):
        zeros = [0] * oshape[jj]
        idxs_per = [None] * oshape[jj]
        w_per = [None] * oshape[jj]
        for kk in range(oshape[jj]):
            cc = float(kk)
            if shifts is not None:
                cc += shifts[jj]
            if zooms is not None:
                if grid_mode:
                    cc += 0.5
                    cc *= zooms[jj]
                    cc -= 0.5
                else:
                    cc *= zooms[jj]
            cc += nprepad
            if not math.isfinite(cc):
                raise ValueError("coordinates must be finite")
            flag, idxs, w = _axis_setup(cc, idim[jj], order, mode, spline_mode)
            if flag:
                zeros[kk] = 1
            else:
                idxs_per[kk] = idxs
                w_per[kk] = w
        axis_data.append((zeros, idxs_per, w_per))
    n_out = 1
    for d in oshape:
        n_out *= d
    for flat in range(n_out):
        out_idx = _unravel(flat, oshape)
        if any(axis_data[hh][0][out_idx[hh]] for hh in range(rank)):
            val = float(cval)
        else:
            per_axis = [(axis_data[hh][1][out_idx[hh]],
                         axis_data[hh][2][out_idx[hh]]) for hh in range(rank)]
            val = _accumulate(input, idim, per_axis, order, mode, cval)
        _set_elem(output, out_idx, val)


# ---------------------------------------------------------------------------
# binary morphology (scipy ni_morphology.c NI_BinaryErosion / 2)
# ---------------------------------------------------------------------------
def _structure_deltas(structure, origins):
    """Multi-index deltas of the true structure positions."""
    shape = list(structure.shape)
    ndim = structure.ndim
    deltas = []
    for flat in range(structure.size):
        sp = _unravel(flat, shape)
        if structure[sp]:
            deltas.append(tuple(sp[h] - origins[h] - shape[h] // 2
                                for h in range(ndim)))
    return deltas


def binary_erosion(input, structure, mask, output, bdr_value, origins,
                   invert, cit, flag):
    """Single binary erosion pass.

    ``flag == 0``: fill ``output`` (scipy returns nothing).
    ``flag == 1``: fill ``output`` and return ``(changed, coords)`` where
    ``coords`` is the list of multi-indices whose value changed.
    """
    origins = _normalize_origin(origins, input.ndim)
    deltas = _structure_deltas(structure, origins)
    in_shape = list(input.shape)
    ndim = input.ndim
    bdr = bool(bdr_value)
    if invert:
        bdr = not bdr

    def logical(p):
        v = bool(_get_elem(input, p))
        return (not v) if invert else v

    changed = 0
    changed_coords = []
    n_in = 1
    for d in in_shape:
        n_in *= d
    for flat in range(n_in):
        p = _unravel(flat, in_shape)
        in_l = logical(p)
        msk = bool(_get_elem(mask, p)) if mask is not None else True
        if not msk or (cit and not in_l):
            out_l = in_l
            chg = 0
        else:
            out_l = True
            for o in deltas:
                q = tuple(p[h] + o[h] for h in range(ndim))
                if not all(0 <= q[h] < in_shape[h] for h in range(ndim)):
                    v_l = bdr
                else:
                    v = bool(_get_elem(input, q))
                    v_l = (not v) if invert else v
                if not v_l:
                    out_l = False
                    break
            chg = 1 if out_l != in_l else 0
        if chg:
            changed = 1
            if flag:
                changed_coords.append(p)
        _set_elem(output, p, (not out_l) if invert else out_l)
    if flag:
        return changed, changed_coords
    return None


def binary_erosion2(array, structure, mask, niter, origins, invert,
                    coordinate_list):
    """Iterated erosion on the coordinates changed by the previous pass
    (scipy NI_BinaryErosion2).  ``array`` is modified in place."""
    origins = _normalize_origin(origins, array.ndim)
    deltas = _structure_deltas(structure, origins)
    a_shape = list(array.shape)
    ndim = array.ndim

    def is_true(p):
        v = bool(_get_elem(array, p))
        return (not v) if invert else v

    def set_false(p):
        _set_elem(array, p, (1 if invert else 0))

    # scipy: non-mask pixels are erased up front (their original values are
    # stashed in the mask array) and restored after the erosion loop.
    if mask is not None:
        saved = {}
        n_px = 1
        for d in a_shape:
            n_px *= d
        for flat in range(n_px):
            p = _unravel(flat, a_shape)
            if bool(_get_elem(mask, p)):
                continue
            saved[p] = bool(_get_elem(array, p))
            set_false(p)

    cur = [tuple(c) for c in coordinate_list]
    it = 0
    while cur:
        if niter > 0 and it >= niter:
            break
        build = (niter <= 0) or (it + 1 < niter)
        nxt = []
        for p in cur:
            for o in deltas:
                q = tuple(p[h] + o[h] for h in range(ndim))
                if not all(0 <= q[h] < a_shape[h] for h in range(ndim)):
                    continue
                if is_true(q):
                    set_false(q)
                    if build:
                        nxt.append(q)
        cur = nxt
        it += 1

    if mask is not None:
        for p, v in saved.items():
            _set_elem(array, p, 1 if v else 0)


# ---------------------------------------------------------------------------
# remaining routines not yet ported (kept for interface compatibility)
# ---------------------------------------------------------------------------
def rank_filter(input, rank, footprint, output, mode, cval, origin=0):
    """Rank filter: pick the ``rank``-th smallest value in the (boundary-
    extended) footprint neighbourhood of each pixel."""
    fshape = list(footprint.shape)
    ndim = input.ndim
    origins = _normalize_origin(origin, ndim)
    in_shape = list(input.shape)
    positions = []
    for flat in range(footprint.size):
        sp = _unravel(flat, fshape)
        if footprint[sp]:
            positions.append(sp)
    rank = int(rank)
    n_out = 1
    for d in in_shape:
        n_out *= d
    for flat in range(n_out):
        p = _unravel(flat, in_shape)
        vals = []
        for sp in positions:
            mapped = []
            is_cval = False
            for h in range(ndim):
                cc = p[h] + sp[h] - fshape[h] // 2 - origins[h]
                e = _boundary_index(cc, in_shape[h], mode)
                if e is None:
                    is_cval = True
                    break
                mapped.append(e)
            vals.append(float(cval) if is_cval
                        else float(_get_elem(input, tuple(mapped))))
        vals.sort()
        _set_elem(output, p, vals[rank])


def generic_filter1d(input, function, filter_size, axis, output, mode, cval,
                     origin=0, extra_arguments=(), extra_keywords=None):
    """1-D generic filter along ``axis`` (scipy ni_filters.c
    NI_GenericFilter1D).  ``function`` is called once per line as
    ``function(input_line, output_line, *extra_arguments,
    **extra_keywords)``: ``input_line`` is the line extended to
    ``L + filter_size - 1`` samples according to the boundary mode,
    ``output_line`` (length ``L``) must be filled in place.
    """
    input = np.asarray(input)
    output = np.asarray(output)
    if extra_keywords is None:
        extra_keywords = {}
    ndim = input.ndim
    shape = list(input.shape)
    L = shape[axis]
    fs = int(filter_size)
    origin = int(origin)
    line_data = input.tolist()

    others = [h for h in range(ndim) if h != axis]
    for oidx in product(*[range(shape[h]) for h in others]):
        base = [0] * ndim
        for h, o in zip(others, oidx):
            base[h] = o
        line = []
        for k in range(L):
            node = line_data
            for h in range(ndim):
                node = node[k if h == axis else base[h]]
            line.append(node)
        ext = [_line_value(line, j - fs // 2 - origin, L, mode, cval)
               for j in range(L + fs - 1)]
        in_arr = np.array(ext, dtype=np.float64)
        out_arr = np.zeros(L, dtype=np.float64)
        function(in_arr, out_arr, *extra_arguments, **extra_keywords)
        vals = [float(t) for t in np.asarray(out_arr).tolist()]
        for off in range(L):
            q = base[:]
            q[axis] = off
            _set_elem(output, tuple(q), vals[off])


def generic_filter(input, function, footprint, output, mode, cval, origins,
                   extra_arguments=(), extra_keywords=None):
    """N-D generic filter (scipy ni_filters.c NI_GenericFilter).
    ``function`` is called once per pixel as
    ``function(buffer, *extra_arguments, **extra_keywords)`` where
    ``buffer`` is the 1-D array of the (boundary-extended) footprint
    values at that pixel; its return value is the output pixel.
    """
    input = np.asarray(input)
    output = np.asarray(output)
    footprint = np.asarray(footprint)
    if extra_keywords is None:
        extra_keywords = {}
    ndim = input.ndim
    fshape = list(footprint.shape)
    origins = _normalize_origin(origins, ndim)
    in_shape = list(input.shape)
    line_data = input.tolist()

    positions = []
    for flat in range(footprint.size):
        sp = _unravel(flat, fshape)
        if bool(footprint[sp]):
            positions.append(sp)

    n_out = 1
    for d in in_shape:
        n_out *= d
    for flat in range(n_out):
        p = _unravel(flat, in_shape)
        buf = []
        for sp in positions:
            mapped = []
            is_cval = False
            for h in range(ndim):
                cc = p[h] + sp[h] - fshape[h] // 2 - origins[h]
                e = _boundary_index(cc, in_shape[h], mode)
                if e is None:
                    is_cval = True
                    break
                mapped.append(e)
            if is_cval:
                buf.append(float(cval))
            else:
                node = line_data
                for h in range(ndim):
                    node = node[mapped[h]]
                buf.append(float(node))
        buf_arr = np.array(buf, dtype=np.float64)
        res = function(buf_arr, *extra_arguments, **extra_keywords)
        _set_elem(output, p, float(res))


# ---------------------------------------------------------------------------
# fourier filters (port of the scipy ``ni_fourier.c`` core)
# ---------------------------------------------------------------------------

_FOURIER_GAUSSIAN = 0
_FOURIER_UNIFORM = 1
_FOURIER_ELLIPSOID = 2


def _fourier_gauss_factor(s, f):
    return math.exp(-2.0 * math.pi * math.pi * s * s * f * f)


def _fourier_uniform_factor(s, f):
    t = math.pi * s * f
    if abs(t) < 1e-8:
        return 1.0
    return math.sin(t) / t


def _fourier_ellipsoid_factor(s, f):
    return math.exp(-(s * f) * (s * f))


def _fourier_axis_freqs(length, real_n, is_real_axis):
    """Spectral frequencies (cycles/sample) of one axis."""
    if is_real_axis:
        fr = np.fft.rfftfreq(real_n)
    else:
        fr = np.fft.fftfreq(length)
    return [float(t) for t in np.asarray(fr).tolist()]


def fourier_filter(input, sigmas, n, axis, output, type):
    """Multiply the FFT of `input` with the separable filter of `type`
    (0 gaussian, 1 uniform, 2 ellipsoid), storing the result in `output`.

    Per-axis multipliers (f = fftfreq(L) per axis, or rfftfreq(n) on
    `axis` when n >= 0, i.e. `input` is a real FFT of an n-sample signal
    along that axis):

        gaussian  exp(-2 pi^2 sigma^2 f^2)
        uniform   sin(pi size f) / (pi size f)      (1 for |pi size f| < 1e-8)
        ellipsoid exp(-(size f)^2)
    """
    input = np.asarray(input)
    output = np.asarray(output)
    ndim = input.ndim
    shape = list(input.shape)
    sig = [float(t) for t in np.asarray(sigmas).tolist()]
    n = -1 if n is None else int(n)
    axis = 0 if axis is None else int(axis)
    if type == _FOURIER_GAUSSIAN:
        fct = _fourier_gauss_factor
    elif type == _FOURIER_UNIFORM:
        fct = _fourier_uniform_factor
    elif type == _FOURIER_ELLIPSOID:
        fct = _fourier_ellipsoid_factor
    else:
        raise ValueError("invalid fourier filter type %r" % (type,))

    facs = []
    for k in range(ndim):
        L = shape[k]
        is_real = n >= 0 and k == axis
        fr = _fourier_axis_freqs(L, n, is_real)
        if len(fr) != L:
            raise RuntimeError(
                "invalid real transform length %d for axis %d (length %d)"
                % (n, k, L))
        facs.append([fct(sig[k], t) for t in fr])

    data = input.tolist()
    for idx in product(*[range(L) for L in shape]):
        m = 1.0
        node = data
        for k in range(ndim):
            m *= facs[k][idx[k]]
            node = node[idx[k]]
        z = node
        if isinstance(z, complex):
            val = complex(z.real * m, z.imag * m)
        else:
            val = float(z) * m
        if ndim == 1:
            output[idx[0]] = val
        else:
            output[idx] = val


def fourier_shift(input, shifts, n, axis, output):
    """Multiply the FFT of `input` with the phase
    exp(-2 pi i sum_k f_k shift_k) — the Fourier transform of a circular
    shift (the inverse FFT of the result is `input` shifted by `shifts`).
    The n/axis conventions match :func:`fourier_filter`; the result is
    complex.
    """
    input = np.asarray(input)
    output = np.asarray(output)
    ndim = input.ndim
    shape = list(input.shape)
    sh = [float(t) for t in np.asarray(shifts).tolist()]
    n = -1 if n is None else int(n)
    axis = 0 if axis is None else int(axis)

    facs = []
    for k in range(ndim):
        L = shape[k]
        is_real = n >= 0 and k == axis
        fr = _fourier_axis_freqs(L, n, is_real)
        if len(fr) != L:
            raise RuntimeError(
                "invalid real transform length %d for axis %d (length %d)"
                % (n, k, L))
        facs.append([complex(math.cos(-2.0 * math.pi * t * sh[k]),
                             math.sin(-2.0 * math.pi * t * sh[k]))
                     for t in fr])

    data = input.tolist()
    for idx in product(*[range(L) for L in shape]):
        m = 1.0 + 0.0j
        node = data
        for k in range(ndim):
            m *= facs[k][idx[k]]
            node = node[idx[k]]
        z = node
        if not isinstance(z, complex):
            z = complex(float(z), 0.0)
        val = z * m
        if ndim == 1:
            output[idx[0]] = val
        else:
            output[idx] = val


def find_objects(input, max_label):
    """Bounding-box slices for labels 1..max_label (scipy semantics: the
    label ``l`` is at index ``l-1``; a missing label yields ``None``)."""
    shape = list(input.shape)
    ndim = input.ndim
    n = 1
    for d in shape:
        n *= d
    max_label = int(max_label)
    boxes = {}
    for flat in range(n):
        p = _unravel(flat, shape)
        lbl = int(_get_elem(input, p))
        if max_label == 0 and lbl > max_label:
            max_label = lbl
        if lbl <= 0:
            continue
        if lbl in boxes:
            lo, hi = boxes[lbl]
            for h in range(ndim):
                if p[h] < lo[h]:
                    lo[h] = p[h]
                if p[h] > hi[h]:
                    hi[h] = p[h]
        else:
            boxes[lbl] = ([p[h] for h in range(ndim)],
                          [p[h] for h in range(ndim)])
    results = []
    for lbl in range(1, max_label + 1):
        if lbl in boxes:
            lo, hi = boxes[lbl]
            results.append(tuple(slice(lo[h], hi[h] + 1) for h in range(ndim)))
        else:
            results.append(None)
    return results


def value_indices(arr, ignoreIsNone, ignore_value_arr):
    return np.array([]), np.array([])


def watershed_ift(input, markers, structure, output):
    """Image foresting transform watershed (scipy ni_measurements.c
    NI_WatershedIFT): a multi-source Dijkstra over the structure
    neighbourhood.  The IFT cost of a pixel is the minimum, over all
    paths from any marker, of the sum of the input values of the pixels
    entered along the path (a marker pixel itself costs 0).  Each pixel
    receives the label of the marker that reaches it with the lowest
    cost; positive markers are grown first and negative (background)
    markers afterwards.  Ties keep the earlier label."""
    input = np.asarray(input)
    markers = np.asarray(markers)
    structure = np.asarray(structure, dtype=bool)
    output = np.asarray(output)
    ndim = input.ndim
    shape = list(input.shape)
    shp_mk = list(structure.shape)

    in_vals = input.tolist()
    mk_vals = markers.tolist()

    strides = [1] * ndim
    for h in range(ndim - 2, -1, -1):
        strides[h] = strides[h + 1] * shape[h + 1]

    def flat_of(idx):
        f = 0
        for h in range(ndim):
            f += idx[h] * strides[h]
        return f

    center = tuple(s // 2 for s in shp_mk)
    deltas = []
    for sp in product(*[range(s) for s in shp_mk]):
        if sp != center and bool(structure[sp]):
            deltas.append(tuple(a - b for a, b in zip(sp, center)))

    n_px = 1
    for d in shape:
        n_px *= d

    def multi_of(f):
        idx = [0] * ndim
        for h in range(ndim - 1, -1, -1):
            idx[h] = f % shape[h]
            f //= shape[h]
        # tuple key = element index in rsnumpy; a list key is fancy
        # row indexing and would broadcast the value over whole rows
        return tuple(idx)

    def elem(vals, f):
        idx = multi_of(f)
        node = vals
        for h in range(ndim):
            node = node[idx[h]]
        return node

    costs = [None] * n_px
    labels = [0] * n_px

    def phase(positive):
        heap = []
        for f in range(n_px):
            mk = int(elem(mk_vals, f))
            if mk != 0 and (mk > 0) == positive:
                costs[f] = 0.0
                labels[f] = mk
                heapq.heappush(heap, (0.0, f))
        while heap:
            c, f = heapq.heappop(heap)
            if c > costs[f]:
                continue
            idx = multi_of(f)
            lab = labels[f]
            for d in deltas:
                q = [idx[h] + d[h] for h in range(ndim)]
                if any(a < 0 or a >= shape[h] for h, a in enumerate(q)):
                    continue
                g = flat_of(q)
                nc = c + float(elem(in_vals, g))
                if costs[g] is None or nc < costs[g]:
                    costs[g] = nc
                    labels[g] = lab
                    heapq.heappush(heap, (nc, g))

    phase(True)
    phase(False)

    for f in range(n_px):
        _set_elem(output, multi_of(f), labels[f])


# Aliases commonly expected by ndimage submodules (the Python-level
# convolve/convolve1d wrappers flip the weights themselves before calling
# the correlate1d path).
convolve1d = correlate1d
convolve = correlate


# ---------------------------------------------------------------------------
# exact distance transform (scipy ``distance_transform_bf``)
# ---------------------------------------------------------------------------

_DT_INF = float('inf')


def _dt_coords(lin, shape):
    c = [0] * len(shape)
    for ax in range(len(shape) - 1, -1, -1):
        c[ax] = lin % shape[ax]
        lin //= shape[ax]
    return tuple(c)


def _dt_linear(c, shape):
    # Row-major (C order): must be the exact inverse of _dt_coords, i.e.
    # lin = c[0]*shape[1]*... + c[1]*... + c[-1].  A descending Horner loop
    # transposes the axes and breaks neighbour lookups in the BFS.
    lin = 0
    for k in range(len(shape)):
        lin = lin * shape[k] + c[k]
    return lin


def _edt1d(fvals, fids, n, scale=1.0):
    """1-D Felzenszwalb-Huttenlocher parabolic distance transform.

    fvals[i] is the source cost at line position i (inf = no source);
    fids[i] carries the identity of that source.  Returns (g, ids) with
    g[i] = min_j (i - j)**2 * scale**2 + fvals[j].
    """
    srcs = [q for q in range(n) if fvals[q] < _DT_INF]
    if not srcs:
        return [_DT_INF] * n, [0] * n
    v = [0] * n
    vid = [0] * n
    z = [0.0] * (n + 1)
    k = 0
    v[0] = srcs[0]
    vid[0] = fids[srcs[0]]
    z[0] = -_DT_INF
    z[1] = _DT_INF

    def elev(q):
        return fvals[q] + (q * scale) ** 2

    for q in srcs[1:]:
        s = (elev(q) - elev(v[k])) / (2.0 * scale * (q - v[k]))
        while s <= z[k]:
            k -= 1
            s = (elev(q) - elev(v[k])) / (2.0 * scale * (q - v[k]))
        k += 1
        v[k] = q
        vid[k] = fids[q]
        # z[k] is the left boundary of v[k]'s region (Felzenszwalb-Huttenlocher
        # convention: z[k] = s, z[k+1] = +inf after inserting at position k)
        z[k] = s
        z[k + 1] = _DT_INF
    g = [0.0] * n
    ids = [0] * n
    k = 0
    for i in range(n):
        x = i * scale
        while z[k + 1] < x:
            k += 1
        dx = x - v[k] * scale
        g[i] = dx * dx + fvals[v[k]]
        ids[i] = vid[k]
    return g, ids


def _l1_1d(fvals, fids, n, scale=1.0):
    """1-D city-block (L1) distance transform via forward/backward passes."""
    fwd_v = [_DT_INF] * n
    fwd_id = [0] * n
    bwd_v = [_DT_INF] * n
    bwd_id = [0] * n
    for q in range(n):
        best = fvals[q]
        bid = fids[q]
        if q > 0 and fwd_v[q - 1] + scale < best:
            best = fwd_v[q - 1] + scale
            bid = fwd_id[q - 1]
        fwd_v[q] = best
        fwd_id[q] = bid
    for q in range(n - 1, -1, -1):
        best = fvals[q]
        bid = fids[q]
        if q < n - 1 and bwd_v[q + 1] + scale < best:
            best = bwd_v[q + 1] + scale
            bid = bwd_id[q + 1]
        bwd_v[q] = best
        bwd_id[q] = bid
    out_v = [0.0] * n
    out_id = [0] * n
    for q in range(n):
        if bwd_v[q] < fwd_v[q]:
            out_v[q] = bwd_v[q]
            out_id[q] = bwd_id[q]
        else:
            out_v[q] = fwd_v[q]
            out_id[q] = fwd_id[q]
    return out_v, out_id


def _dt_axis_transform(vals, ids, shape, ax, scale, transform):
    """Apply a 1-D transform along every line of axis ``ax``."""
    out_v = list(vals)
    out_id = list(ids)
    total = 1
    for s in shape:
        total *= s
    for base in range(total):
        c0 = _dt_coords(base, shape)
        if c0[ax] != 0:
            continue
        n = shape[ax]
        line_v = [0.0] * n
        line_id = [0] * n
        for q in range(n):
            c = list(c0)
            c[ax] = q
            lin = _dt_linear(tuple(c), shape)
            line_v[q] = vals[lin]
            line_id[q] = ids[lin]
        gv, gid = transform(line_v, line_id, n, scale)
        for q in range(n):
            c = list(c0)
            c[ax] = q
            lin = _dt_linear(tuple(c), shape)
            out_v[lin] = gv[q]
            out_id[lin] = gid[q]
    return out_v, out_id


def distance_transform_bf(input, metric, sampling, distances, features):
    """Exact distance transform: distance to the nearest background pixel.

    Mirrors the scipy C entry point.  ``input`` is the wrapper-encoded
    array (interior foreground 1, boundary foreground 0, background -1);
    entries < 0 are the distance sources.  metric: 1 = euclidean,
    2 = city block (L1), 3 = chessboard (L-inf).  ``distances`` (float64
    for metric 1, uint32 for 2/3) and ``features`` (int32 linear index of
    the nearest source) are filled in place; either may be None.
    """
    a = np.asarray(input)
    nd = a.ndim
    shape = tuple(a.shape)
    total = 1
    for s in shape:
        total *= s
    if sampling is not None:
        sampling = [float(s) for s in np.asarray(sampling, dtype=np.float64)]
    else:
        sampling = [1.0] * nd

    values = [0.0] * total
    for lin in range(total):
        values[lin] = float(a[_dt_coords(lin, shape)])
    srcs = [lin for lin in range(total) if values[lin] < 0.0]

    d_out = [0.0] * total
    f_out = list(range(total))

    if srcs:
        if metric in (1, 2):
            transform = _edt1d if metric == 1 else _l1_1d
            cur_v = [0.0] * total
            cur_id = list(range(total))
            for lin in range(total):
                if values[lin] < 0.0:
                    cur_v[lin] = 0.0
                else:
                    cur_v[lin] = _DT_INF
            for ax in range(nd - 1, -1, -1):
                cur_v, cur_id = _dt_axis_transform(
                    cur_v, cur_id, shape, ax, sampling[ax], transform)
            for lin in range(total):
                if values[lin] >= 0.0:
                    if metric == 1:
                        d_out[lin] = math.sqrt(max(cur_v[lin], 0.0))
                    else:
                        d_out[lin] = cur_v[lin]
                    f_out[lin] = cur_id[lin]
        else:  # chessboard: multi-source BFS with unit Chebyshev steps
            from collections import deque
            dist = [_DT_INF] * total
            s_id = [-1] * total
            dq = deque()
            for b in srcs:
                dist[b] = 0.0
                s_id[b] = b
                dq.append(b)
            deltas = [d for d in product((-1, 0, 1), repeat=nd) if any(d)]
            while dq:
                p = dq.popleft()
                c0 = _dt_coords(p, shape)
                base_d = dist[p]
                for dlt in deltas:
                    nc = tuple(c0[kk] + dlt[kk] for kk in range(nd))
                    if any(cc < 0 or cc > shape[kk] - 1
                           for kk, cc in enumerate(nc)):
                        continue
                    nlin = _dt_linear(nc, shape)
                    step = max(abs(dlt[kk]) * sampling[kk]
                               for kk in range(nd))
                    ndist = base_d + step
                    if ndist < dist[nlin]:
                        dist[nlin] = ndist
                        s_id[nlin] = s_id[p]
                        dq.append(nlin)
            for lin in range(total):
                if values[lin] >= 0.0 and s_id[lin] >= 0:
                    d_out[lin] = dist[lin]
                    f_out[lin] = s_id[lin]

    if distances is not None:
        for lin in range(total):
            distances[_dt_coords(lin, shape)] = d_out[lin]
    if features is not None:
        for lin in range(total):
            features[_dt_coords(lin, shape)] = f_out[lin]
