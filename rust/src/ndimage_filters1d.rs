//! Rust port of the pure-rsnumpy 1-D filter kernels from
//! `scicore/ndimage/_nd_image.py` (`correlate1d`, `uniform_filter1d`,
//! `min_or_max_filter1d`; scipy `ni_filters.c` semantics).
//!
//! rsnumpy only exposes a **read-only** float64 buffer, so the kernels read
//! the input zero-copy through `PyBuffer<f64>` and return a flat C-ordered
//! result that the Python layer writes back with
//! `out[...] = np.asarray(result).reshape(shape)`.

use std::borrow::Cow;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Extension-mode codes (see `_ni_support._extend_mode_to_code`):
/// 0 nearest, 1 wrap, 2 reflect, 3 mirror, 4 constant, 5 grid-wrap,
/// 6 grid-constant.
///
/// Maps a possibly out-of-range index `cc` into `[0, n)`, mirroring the
/// scipy C core (ni_support.c `NI_ExtendLine` integer form).  Returns
/// `None` when the caller must substitute `cval` (constant modes).
fn boundary_index(cc: isize, n: usize, mode: usize) -> Option<usize> {
    if mode == 4 || mode == 6 {
        // constant / grid-constant
        return (0..n as isize).contains(&cc).then_some(cc as usize);
    }
    if n <= 1 {
        return Some(0);
    }
    if (0..n as isize).contains(&cc) {
        return Some(cc as usize);
    }
    match mode {
        2 => {
            // reflect: abcddcba|abcd|dcbaabcd (edge duplicated, period
            // 2*n) — ni_support.c NI_ExtendLine NI_EXTEND_REFLECT.  The
            // scipy docs ASCII patterns for reflect/mirror are swapped
            // relative to the C implementation; follow the C code.
            let sz2 = 2 * n as isize;
            let t = cc.rem_euclid(sz2);
            Some(if t < n as isize {
                t as usize
            } else {
                (sz2 - 1 - t) as usize
            })
        }
        3 => {
            // mirror: cbabcdcb|abcd|cbabcdcb (edge not duplicated,
            // period 2*(n-1)) — ni_support.c NI_ExtendLine
            // NI_EXTEND_MIRROR.
            let sz2 = 2 * n as isize - 2;
            let t = cc.rem_euclid(sz2);
            Some(if t < n as isize - 1 {
                t as usize
            } else {
                (sz2 - t) as usize
            })
        }
        1 | 5 => Some(cc.rem_euclid(n as isize) as usize), // wrap / grid-wrap
        0 => Some(cc.clamp(0, n as isize - 1) as usize),   // nearest
        _ => None,
    }
}

#[inline]
fn line_value(line: &[f64], cc: isize, mode: usize, cval: f64) -> f64 {
    boundary_index(cc, line.len(), mode).map_or(cval, |i| line[i])
}

fn total_size(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Borrow a C-contiguous float64 buffer as a slice, without copying.
fn borrow_f64<'a>(view: &'a PyBuffer<f64>, shape: &[usize]) -> PyResult<&'a [f64]> {
    if !view.is_c_contiguous() {
        return Err(PyValueError::new_err("expected a C-contiguous buffer"));
    }
    let n = total_size(shape);
    if view.item_count() != n {
        return Err(PyValueError::new_err(format!(
            "buffer holds {} elements, expected {}",
            view.item_count(),
            n
        )));
    }
    // SAFETY: `PyBuffer<f64>` validated the format/itemsize, the view lives
    // at least as long as this call, and the slice is never written through.
    Ok(unsafe { std::slice::from_raw_parts(view.buf_ptr() as *const f64, n) })
}

/// Borrow a C-contiguous **writable** float64 buffer as a mutable slice,
/// without copying.  Requires the source to have exported a writable buffer —
/// rsnumpy `ndarray` always does (owned data, no shared views), so this backs
/// zero-copy in-place write-back for the kernels below.
fn borrow_f64_mut<'a>(view: &'a mut PyBuffer<f64>, shape: &[usize]) -> PyResult<&'a mut [f64]> {
    if view.readonly() {
        return Err(PyValueError::new_err("expected a writable buffer"));
    }
    if !view.is_c_contiguous() {
        return Err(PyValueError::new_err("expected a C-contiguous buffer"));
    }
    let n = total_size(shape);
    if view.item_count() != n {
        return Err(PyValueError::new_err(format!(
            "buffer holds {} elements, expected {}",
            view.item_count(),
            n
        )));
    }
    // SAFETY: the buffer is writable (checked above), C-contiguous, holds
    // exactly `n` f64 items, and the view lives at least as long as this call.
    // The returned slice is the sole writer for the duration of the call.
    Ok(unsafe { std::slice::from_raw_parts_mut(view.buf_ptr() as *mut f64, n) })
}

/// Geometry of the lines along `axis` in a C-contiguous array of the given
/// shape.  Lines are enumerated row-major over the non-axis axes.
struct LineGeometry {
    line_len: usize,
    line_stride: usize,
    n_lines: usize,
    /// Non-axis axes in ascending order.
    axes: Vec<usize>,
    shape: Vec<usize>,
    /// Full C memory stride of each axis.
    mem_stride: Vec<usize>,
    /// Row-major divisor of each non-axis axis in the line index `l`.
    l_stride: Vec<usize>,
}

impl LineGeometry {
    /// Offset of the first element of line `l`: unpack `l` into the
    /// non-axis multi-index and sum `idx_k * mem_stride[k]`.
    fn base(&self, l: usize) -> usize {
        let mut base = 0usize;
        for &k in &self.axes {
            let idx = (l / self.l_stride[k]) % self.shape[k];
            base += idx * self.mem_stride[k];
        }
        base
    }
}

fn line_geometry(shape: &[usize], axis: isize) -> PyResult<LineGeometry> {
    let ndim = shape.len();
    if axis < 0 || axis as usize >= ndim {
        return Err(PyValueError::new_err("axis out of range"));
    }
    let a = axis as usize;
    let line_len = shape[a];
    let mut mem_stride = vec![1usize; ndim];
    for k in (0..ndim - 1).rev() {
        mem_stride[k] = mem_stride[k + 1] * shape[k + 1];
    }
    let total: usize = shape.iter().product();
    let axes: Vec<usize> = (0..ndim).filter(|&k| k != a).collect();
    let mut l_stride = vec![0usize; ndim];
    let mut prod = 1usize;
    for &k in axes.iter().rev() {
        l_stride[k] = prod;
        prod *= shape[k];
    }
    Ok(LineGeometry {
        line_len,
        line_stride: mem_stride[a],
        n_lines: total.checked_div(line_len).unwrap_or(0),
        axes,
        shape: shape.to_vec(),
        mem_stride,
        l_stride,
    })
}

/// The line starting at flat offset `base` as a contiguous slice (borrowed
/// when the line is contiguous in memory, gathered otherwise).
fn line_at<'a>(
    data: &'a [f64],
    base: usize,
    line_len: usize,
    line_stride: usize,
) -> Cow<'a, [f64]> {
    if line_stride == 1 {
        Cow::Borrowed(&data[base..base + line_len])
    } else {
        Cow::Owned(
            (0..line_len)
                .map(|k| data[base + k * line_stride])
                .collect(),
        )
    }
}

/// All three 1-D kernels write into `out` (same length as `input`).  The
/// caller MUST guarantee `input` and `out` do not alias — the `_into`
/// pyfunctions copy the input to a temporary when they are the same buffer,
/// and the `Vec`-returning pyfunctions pass a freshly allocated `out`.
#[allow(clippy::too_many_arguments)]
fn correlate1d_core(
    input: &[f64],
    out: &mut [f64],
    shape: &[usize],
    weights: &[f64],
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
) -> PyResult<()> {
    if weights.is_empty() {
        return Err(PyValueError::new_err("no filter weights given"));
    }
    let geo = line_geometry(shape, axis)?;
    let half = (weights.len() / 2) as isize;
    for l in 0..geo.n_lines {
        let base = geo.base(l);
        let line = line_at(input, base, geo.line_len, geo.line_stride);
        for i in 0..geo.line_len {
            let mut s = 0.0f64;
            for (j, w) in weights.iter().enumerate() {
                s += w * line_value(&line, i as isize + j as isize - half - origin, mode, cval);
            }
            out[base + i * geo.line_stride] = s;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn uniform_filter1d_core(
    input: &[f64],
    out: &mut [f64],
    shape: &[usize],
    size: usize,
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
) -> PyResult<()> {
    if size == 0 {
        return Err(PyValueError::new_err("incorrect filter size"));
    }
    let geo = line_geometry(shape, axis)?;
    let half = (size / 2) as isize;
    for l in 0..geo.n_lines {
        let base = geo.base(l);
        let line = line_at(input, base, geo.line_len, geo.line_stride);
        for i in 0..geo.line_len {
            let mut s = 0.0f64;
            for j in 0..size {
                s += line_value(&line, i as isize + j as isize - half - origin, mode, cval);
            }
            out[base + i * geo.line_stride] = s / size as f64;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn min_or_max_filter1d_core(
    input: &[f64],
    out: &mut [f64],
    shape: &[usize],
    size: usize,
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
    minimum: bool,
) -> PyResult<()> {
    if size == 0 {
        return Err(PyValueError::new_err("incorrect filter size"));
    }
    let geo = line_geometry(shape, axis)?;
    let half = (size / 2) as isize;
    for l in 0..geo.n_lines {
        let base = geo.base(l);
        let line = line_at(input, base, geo.line_len, geo.line_stride);
        for i in 0..geo.line_len {
            // size == 1 reads line[i] directly (the valid origin range then
            // forces origin == 0, so this matches the windowed form).
            let best = if size == 1 {
                line[i]
            } else {
                let mut best = line_value(&line, i as isize - half - origin, mode, cval);
                for j in 1..size {
                    let v = line_value(&line, i as isize + j as isize - half - origin, mode, cval);
                    if minimum {
                        if v < best {
                            best = v;
                        }
                    } else if v > best {
                        best = v;
                    }
                }
                best
            };
            out[base + i * geo.line_stride] = best;
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (input, shape, weights, axis, mode, cval, origin))]
fn _correlate1d(
    input: PyBuffer<f64>,
    shape: Vec<usize>,
    weights: Vec<f64>,
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
) -> PyResult<Vec<f64>> {
    let data = borrow_f64(&input, &shape)?;
    let mut result = vec![0.0f64; data.len()];
    correlate1d_core(
        data,
        &mut result,
        &shape,
        &weights,
        axis,
        mode,
        cval,
        origin,
    )?;
    Ok(result)
}

/// Zero-copy variant: write the result directly into `output` (a writable,
/// C-contiguous f64 buffer of the same shape as `input`).
#[pyfunction]
#[pyo3(signature = (input, output, shape, weights, axis, mode, cval, origin))]
#[allow(clippy::too_many_arguments)]
fn _correlate1d_into(
    input: PyBuffer<f64>,
    mut output: PyBuffer<f64>,
    shape: Vec<usize>,
    weights: Vec<f64>,
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
) -> PyResult<()> {
    if input.buf_ptr() == output.buf_ptr() {
        // In-place: copy the input first so each line is fully read before it
        // is written back.
        let d = borrow_f64(&input, &shape)?;
        let tmp = d.to_vec();
        let out = borrow_f64_mut(&mut output, &shape)?;
        correlate1d_core(
            tmp.as_slice(),
            out,
            &shape,
            &weights,
            axis,
            mode,
            cval,
            origin,
        )
    } else {
        let src = borrow_f64(&input, &shape)?;
        let out = borrow_f64_mut(&mut output, &shape)?;
        correlate1d_core(src, out, &shape, &weights, axis, mode, cval, origin)
    }
}

#[pyfunction]
#[pyo3(signature = (input, shape, size, axis, mode, cval, origin))]
fn _uniform_filter1d(
    input: PyBuffer<f64>,
    shape: Vec<usize>,
    size: usize,
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
) -> PyResult<Vec<f64>> {
    let data = borrow_f64(&input, &shape)?;
    let mut result = vec![0.0f64; data.len()];
    uniform_filter1d_core(data, &mut result, &shape, size, axis, mode, cval, origin)?;
    Ok(result)
}

/// Zero-copy variant: write the result directly into `output`.
#[pyfunction]
#[pyo3(signature = (input, output, shape, size, axis, mode, cval, origin))]
#[allow(clippy::too_many_arguments)]
fn _uniform_filter1d_into(
    input: PyBuffer<f64>,
    mut output: PyBuffer<f64>,
    shape: Vec<usize>,
    size: usize,
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
) -> PyResult<()> {
    if input.buf_ptr() == output.buf_ptr() {
        let d = borrow_f64(&input, &shape)?;
        let tmp = d.to_vec();
        let out = borrow_f64_mut(&mut output, &shape)?;
        uniform_filter1d_core(tmp.as_slice(), out, &shape, size, axis, mode, cval, origin)
    } else {
        let src = borrow_f64(&input, &shape)?;
        let out = borrow_f64_mut(&mut output, &shape)?;
        uniform_filter1d_core(src, out, &shape, size, axis, mode, cval, origin)
    }
}

#[pyfunction]
#[pyo3(signature = (input, shape, size, axis, mode, cval, origin, minimum))]
#[allow(clippy::too_many_arguments)]
fn _min_or_max_filter1d(
    input: PyBuffer<f64>,
    shape: Vec<usize>,
    size: usize,
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
    minimum: bool,
) -> PyResult<Vec<f64>> {
    let data = borrow_f64(&input, &shape)?;
    let mut result = vec![0.0f64; data.len()];
    min_or_max_filter1d_core(
        data,
        &mut result,
        &shape,
        size,
        axis,
        mode,
        cval,
        origin,
        minimum,
    )?;
    Ok(result)
}

/// Zero-copy variant: write the result directly into `output`.
#[pyfunction]
#[pyo3(signature = (input, output, shape, size, axis, mode, cval, origin, minimum))]
#[allow(clippy::too_many_arguments)]
fn _min_or_max_filter1d_into(
    input: PyBuffer<f64>,
    mut output: PyBuffer<f64>,
    shape: Vec<usize>,
    size: usize,
    axis: isize,
    mode: usize,
    cval: f64,
    origin: isize,
    minimum: bool,
) -> PyResult<()> {
    if input.buf_ptr() == output.buf_ptr() {
        let d = borrow_f64(&input, &shape)?;
        let tmp = d.to_vec();
        let out = borrow_f64_mut(&mut output, &shape)?;
        min_or_max_filter1d_core(
            tmp.as_slice(),
            out,
            &shape,
            size,
            axis,
            mode,
            cval,
            origin,
            minimum,
        )
    } else {
        let src = borrow_f64(&input, &shape)?;
        let out = borrow_f64_mut(&mut output, &shape)?;
        min_or_max_filter1d_core(src, out, &shape, size, axis, mode, cval, origin, minimum)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_correlate1d, module)?)?;
    module.add_function(wrap_pyfunction!(_correlate1d_into, module)?)?;
    module.add_function(wrap_pyfunction!(_uniform_filter1d, module)?)?;
    module.add_function(wrap_pyfunction!(_uniform_filter1d_into, module)?)?;
    module.add_function(wrap_pyfunction!(_min_or_max_filter1d, module)?)?;
    module.add_function(wrap_pyfunction!(_min_or_max_filter1d_into, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{boundary_index, line_geometry, line_value};

    // Reference values hand-checked against the scipy C implementation
    // (n = 4, a=0 b=1 c=2 d=3):
    //   reflect: abcddcba|abcd|dcbaabcd (edge duplicated, period 8)
    //   mirror:  cbabcdcb|abcd|cbabcdcb (edge not duplicated, period 6)
    #[test]
    fn boundary_index_matches_python_semantics() {
        // reflect: period 8; -1 % 8 = 7 -> 8-1-7 = 0 (edge duplicated);
        // 4 % 8 = 4 -> 8-1-4 = 3; -5 % 8 = 3 -> 3 (in range)
        assert_eq!(boundary_index(-1, 4, 2), Some(0));
        assert_eq!(boundary_index(4, 4, 2), Some(3));
        assert_eq!(boundary_index(-5, 4, 2), Some(3));
        // mirror: period 6; -1 % 6 = 5 -> 6-5 = 1; -5 % 6 = 1;
        // 4 % 6 = 4 -> 6-4 = 2; 7 % 6 = 1
        assert_eq!(boundary_index(-1, 4, 3), Some(1));
        assert_eq!(boundary_index(4, 4, 3), Some(2));
        assert_eq!(boundary_index(-5, 4, 3), Some(1));
        assert_eq!(boundary_index(7, 4, 3), Some(1));
    }

    #[test]
    fn boundary_index_wrap_nearest_constant() {
        assert_eq!(boundary_index(-1, 4, 1), Some(3));
        assert_eq!(boundary_index(5, 4, 1), Some(1));
        assert_eq!(boundary_index(-3, 4, 5), Some(1)); // grid-wrap
        assert_eq!(boundary_index(-1, 4, 0), Some(0)); // nearest
        assert_eq!(boundary_index(9, 4, 0), Some(3));
        assert_eq!(boundary_index(-1, 4, 4), None); // constant
        assert_eq!(boundary_index(2, 4, 4), Some(2));
        assert_eq!(boundary_index(-1, 4, 6), None); // grid-constant
        // unknown mode -> cval
        assert_eq!(boundary_index(0, 4, 9), Some(0)); // in range, any mode
        assert_eq!(boundary_index(-1, 4, 9), None);
    }

    #[test]
    fn degenerate_lines() {
        assert_eq!(boundary_index(-7, 1, 2), Some(0));
        assert_eq!(boundary_index(-7, 1, 3), Some(0));
        assert_eq!(boundary_index(-7, 1, 1), Some(0));
        assert_eq!(boundary_index(-7, 1, 0), Some(0));
        assert_eq!(boundary_index(3, 1, 0), Some(0));
        // empty line: only constant modes are well defined
        assert_eq!(boundary_index(0, 0, 4), None);
    }

    #[test]
    fn line_geometry_bases_match_c_order_offsets() {
        // 2d (9, 17): axis 0 -> column l starts at flat offset l (stride 17)
        let g = line_geometry(&[9, 17], 0).unwrap();
        assert_eq!((g.n_lines, g.line_len, g.line_stride), (17, 9, 17));
        for l in 0..17 {
            assert_eq!(g.base(l), l);
        }
        // axis 1 -> row l starts at offset 17*l (stride 1)
        let g = line_geometry(&[9, 17], 1).unwrap();
        assert_eq!((g.n_lines, g.line_len, g.line_stride), (9, 17, 1));
        for l in 0..9 {
            assert_eq!(g.base(l), 17 * l);
        }
        // 3d (4, 5, 6): memory strides (30, 6, 1)
        let g = line_geometry(&[4, 5, 6], 0).unwrap();
        assert_eq!((g.n_lines, g.line_len, g.line_stride), (30, 4, 30));
        assert_eq!(g.base(0), 0); // (i1=0, i2=0)
        assert_eq!(g.base(5), 5); // (i1=0, i2=5)
        assert_eq!(g.base(6), 6); // (i1=1, i2=0)
        let g = line_geometry(&[4, 5, 6], 1).unwrap();
        assert_eq!((g.n_lines, g.line_len, g.line_stride), (24, 5, 6));
        assert_eq!(g.base(0), 0); // (i0=0, i2=0)
        assert_eq!(g.base(1), 1); // (i0=0, i2=1)
        assert_eq!(g.base(6), 30); // (i0=1, i2=0)
        assert_eq!(g.base(11), 35); // (i0=1, i2=5)
        let g = line_geometry(&[4, 5, 6], 2).unwrap();
        assert_eq!((g.n_lines, g.line_len, g.line_stride), (20, 6, 1));
        assert_eq!(g.base(1), 6); // (i0=0, i1=1)
        assert_eq!(g.base(5), 30); // (i0=1, i1=0)
        assert_eq!(g.base(6), 36); // (i0=1, i1=1)
    }

    #[test]
    fn line_value_falls_back_to_cval() {
        let line = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(line_value(&line, -1, 4, 9.5), 9.5);
        assert_eq!(line_value(&line, 5, 6, 9.5), 9.5);
        assert_eq!(line_value(&line, 1, 2, 9.5), 2.0);
        assert_eq!(line_value(&line, 3, 3, 9.5), 4.0); // mirror, in range
    }
}
