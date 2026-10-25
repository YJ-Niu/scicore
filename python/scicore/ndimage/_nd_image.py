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
    if cc < 0:
        if mode == _MODE_MIRROR:
            sz2 = 2 * n - 2
            cc = sz2 * ((-cc) // sz2) + cc
            return cc + sz2 if cc <= 1 - n else -cc
        if mode == _MODE_REFLECT:
            sz2 = 2 * n
            if cc < -sz2:
                cc = sz2 * ((-cc) // sz2) + cc
            return cc + sz2 if cc < -n else -cc - 1
        if mode in (_MODE_WRAP, _MODE_GRID_WRAP):
            cc += n * ((-cc) // n)
            if cc < 0:
                cc += n
            return cc
        if mode == _MODE_NEAREST:
            return 0
        return None
    if cc >= n:
        if mode == _MODE_MIRROR:
            sz2 = 2 * n - 2
            cc -= sz2 * (cc // sz2)
            return sz2 - cc if cc >= n else cc
        if mode == _MODE_REFLECT:
            sz2 = 2 * n
            cc -= sz2 * (cc // sz2)
            return sz2 - cc - 1 if cc >= n else cc
        if mode in (_MODE_WRAP, _MODE_GRID_WRAP):
            return cc - n * (cc // n)
        if mode == _MODE_NEAREST:
            return n - 1
        return None
    return cc


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
                     origin=0):
    raise NotImplementedError("generic_filter1d not yet ported to Rust/rsnumpy")


def generic_filter(input, function, footprint, output, mode, cval, origin=0):
    raise NotImplementedError("generic_filter not yet ported to Rust/rsnumpy")


def fourier_filter(input, sigmas, n, axis, output, type):
    raise NotImplementedError("fourier_filter not yet ported to Rust/rsnumpy")


def fourier_shift(input, shifts, n, axis, output):
    raise NotImplementedError("fourier_shift not yet ported to Rust/rsnumpy")


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
    raise NotImplementedError("watershed_ift not yet ported to Rust/rsnumpy")


# Aliases commonly expected by ndimage submodules (the Python-level
# convolve/convolve1d wrappers flip the weights themselves before calling
# the correlate1d path).
convolve1d = correlate1d
convolve = correlate
