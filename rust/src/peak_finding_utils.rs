//! Rust port of `scipy/signal/_peak_finding_utils.pyx`.
//!
//! Four computational kernels used by `scipy.signal._peak_finding`:
//!
//! * `_local_maxima_1d` — find local maxima in a 1-D signal
//! * `_select_by_peak_distance` — boolean mask by peak priority/distance
//! * `_peak_prominences` — prominence of each peak
//! * `_peak_widths` — width of each peak at a relative height
//!
//! The functions operate on raw slices obtained from the rsnumpy buffer
//! protocol and return plain Python lists; the Python-side wrapper converts
//! them to rsnumpy arrays.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

type PeakWidthsOutput<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Extract a C-contiguous f64 buffer from any object that supports the buffer
/// protocol.  Mirrors the `const np.float64_t[::1] x not None` signature of
/// the original Cython code: wrong dimensionality raises ValueError, wrong
/// dtype raises TypeError, non-array-like raises TypeError, None raises
/// TypeError.
/// Extract a 1-D f64 slice from any object that supports the buffer
/// protocol (rsnumpy float64 arrays expose one) or, as a fallback, can be
/// converted to a Python list via `tolist()`.
///
/// Returns the data as a `Vec<f64>` (owned) because rsnumpy only exposes a
/// zero-copy buffer for float64 arrays.
fn extract_f64_slice<'py>(_py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    if obj.is_none() {
        return Err(PyTypeError::new_err("'x' must not be None"));
    }

    // Fast path: zero-copy buffer (only available for float64 in rsnumpy).
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err(
                "wrong number of dimensions, expected 1",
            ));
        }
        return Ok(unsafe { f64_slice(&buf) }.to_vec());
    }

    // Fallback: if it's already a list, use it directly; otherwise call tolist().
    let values: Bound<'py, PyList> = if obj.is_instance_of::<PyList>() {
        obj.cast::<PyList>()?.clone()
    } else {
        let list = obj.call_method0("tolist")?;
        list.cast::<PyList>()?.clone()
    };
    let mut result = Vec::with_capacity(values.len());
    for item in values.iter() {
        let v: f64 = item.extract()?;
        result.push(v);
    }
    Ok(result)
}

/// Extract a 1-D integer slice from any object that supports the buffer
/// protocol (float64 arrays expose one via rsnumpy) or, as a fallback, can be
/// converted to a Python list via `tolist()`.
///
/// Returns the data as a `Vec<isize>` (owned) because rsnumpy only exposes a
/// zero-copy buffer for float64 arrays, not for integer dtypes.
fn extract_intp_slice<'py>(_py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Vec<isize>> {
    if obj.is_none() {
        return Err(PyTypeError::new_err("'x' must not be None"));
    }

    // Fast path: zero-copy buffer (only available for float64 in rsnumpy).
    if let Ok(buf) = PyBuffer::<isize>::get(obj) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err(
                "wrong number of dimensions, expected 1",
            ));
        }
        return Ok(unsafe { isize_slice(&buf) }.to_vec());
    }

    // Fallback: if it's already a list, use it directly; otherwise call tolist().
    let values: Bound<'py, PyList> = if obj.is_instance_of::<PyList>() {
        obj.cast::<PyList>()?.clone()
    } else {
        let list = obj.call_method0("tolist")?;
        list.cast::<PyList>()?.clone()
    };
    let mut result = Vec::with_capacity(values.len());
    for item in values.iter() {
        let v: isize = item.extract()?;
        result.push(v);
    }
    Ok(result)
}

// SAFETY: the caller must ensure the buffer is alive for the lifetime of the
// returned slice, and that the buffer is C-contiguous with the correct
// itemsize (guaranteed by `PyBuffer<T>::get`).
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn f64_slice(buf: &PyBuffer<f64>) -> &[f64] {
    std::slice::from_raw_parts(buf.buf_ptr() as *mut f64, buf.item_count())
}
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn isize_slice(buf: &PyBuffer<isize>) -> &[isize] {
    std::slice::from_raw_parts(buf.buf_ptr() as *mut isize, buf.item_count())
}

// ---------------------------------------------------------------------------
// _local_maxima_1d
// ---------------------------------------------------------------------------

/// Find local maxima in a 1-D array.
///
/// Returns `(midpoints, left_edges, right_edges)` as Python lists of ints.
#[pyfunction]
fn _local_maxima_1d<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let x_data = extract_f64_slice(py, x)?;
    let data: &[f64] = &x_data;

    let n = data.len();
    // Preallocate, there can't be more maxima than half the size of `x`.
    let mut midpoints = Vec::with_capacity(n / 2);
    let mut left_edges = Vec::with_capacity(n / 2);
    let mut right_edges = Vec::with_capacity(n / 2);

    if n == 0 {
        return Ok((
            midpoints.into_pyobject(py)?.into_any(),
            left_edges.into_pyobject(py)?.into_any(),
            right_edges.into_pyobject(py)?.into_any(),
        ));
    }

    let mut i: usize = 1; // first sample can't be maxima
    let i_max = n - 1; // last sample can't be maxima

    while i < i_max {
        if data[i - 1] < data[i] {
            let mut i_ahead = i + 1;

            // Find next sample that is unequal to data[i].
            while i_ahead < i_max && data[i_ahead] == data[i] {
                i_ahead += 1;
            }

            // Maxima is found if next unequal sample is smaller than data[i].
            if data[i_ahead] < data[i] {
                left_edges.push(i as isize);
                right_edges.push((i_ahead - 1) as isize);
                midpoints.push(((i + i_ahead - 1) / 2) as isize);
                // Skip samples that can't be maximum.
                i = i_ahead;
            }
        }
        i += 1;
    }

    Ok((
        midpoints.into_pyobject(py)?.into_any(),
        left_edges.into_pyobject(py)?.into_any(),
        right_edges.into_pyobject(py)?.into_any(),
    ))
}

// ---------------------------------------------------------------------------
// _select_by_peak_distance
// ---------------------------------------------------------------------------

/// Evaluate which peaks fulfill the distance condition.
#[pyfunction]
fn _select_by_peak_distance<'py>(
    py: Python<'py>,
    peaks: &Bound<'py, PyAny>,
    priority: &Bound<'py, PyAny>,
    distance: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let peaks_data = extract_intp_slice(py, peaks)?;
    let pri_data = extract_f64_slice(py, priority)?;

    let peaks_data: &[isize] = &peaks_data;
    let pri_data: &[f64] = &pri_data;

    let peaks_size = peaks_data.len();
    // Round up because actual peak distance can only be natural number.
    let distance_ = distance.ceil() as isize;
    let mut keep = vec![1u8; peaks_size];

    // Create map from `i` (index for `peaks` sorted by `priority`) to `j`
    // (index for `peaks` sorted by position).  We use argsort of priority.
    let mut priority_to_position: Vec<usize> = (0..peaks_size).collect();
    priority_to_position.sort_by(|&a, &b| {
        pri_data[a]
            .partial_cmp(&pri_data[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Highest priority first -> iterate in reverse order (decreasing).
    for i in (0..peaks_size).rev() {
        let j = priority_to_position[i];
        if keep[j] == 0 {
            continue;
        }

        let mut k = j as isize - 1;
        while k >= 0 && peaks_data[j] - peaks_data[k as usize] < distance_ {
            keep[k as usize] = 0;
            k -= 1;
        }

        let mut k = j + 1;
        while k < peaks_size && peaks_data[k] - peaks_data[j] < distance_ {
            keep[k] = 0;
            k += 1;
        }
    }

    // Return as a list of bools (the Python wrapper converts to uint8 array).
    Ok(keep
        .iter()
        .map(|&b| b != 0)
        .collect::<Vec<bool>>()
        .into_pyobject(py)?
        .into_any())
}

// ---------------------------------------------------------------------------
// _peak_prominences
// ---------------------------------------------------------------------------

/// Calculate the prominence of each peak in a signal.
#[pyfunction]
fn _peak_prominences<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    peaks: &Bound<'py, PyAny>,
    wlen: isize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let x_data = extract_f64_slice(py, x)?;
    let peaks_data = extract_intp_slice(py, peaks)?;

    let x_data: &[f64] = &x_data;
    let peaks_data: &[isize] = &peaks_data;

    let n_peaks = peaks_data.len();
    let mut prominences = Vec::with_capacity(n_peaks);
    let mut left_bases = Vec::with_capacity(n_peaks);
    let mut right_bases = Vec::with_capacity(n_peaks);

    for &peak in peaks_data {
        let peak = peak as usize;
        let mut i_min = 0usize;
        let mut i_max = x_data.len() - 1;

        if !(i_min <= peak && peak <= i_max) {
            return Err(PyValueError::new_err(format!(
                "peak {} is not a valid index for `x`",
                peak
            )));
        }

        if 2 <= wlen {
            i_min = peak.saturating_sub((wlen / 2) as usize);
            i_max = std::cmp::min(peak + (wlen / 2) as usize, x_data.len() - 1);
        }

        // Find the left base in interval [i_min, peak].
        // Cython: i = left_bases[peak_nr] = peak; left_min = x[peak]
        //        while i_min <= i and x[i] <= x[peak]:
        //            if x[i] < left_min: left_min = x[i]; left_bases[peak_nr] = i
        //            i -= 1
        let mut left_base = peak;
        let mut left_min = x_data[peak];
        let mut idx = peak as isize;
        while idx >= i_min as isize && x_data[idx as usize] <= x_data[peak] {
            if x_data[idx as usize] < left_min {
                left_min = x_data[idx as usize];
                left_base = idx as usize;
            }
            idx -= 1;
        }

        // Find the right base in interval [peak, i_max].
        // Cython: i = right_bases[peak_nr] = peak; right_min = x[peak]
        //        while i <= i_max and x[i] <= x[peak]:
        //            if x[i] < right_min: right_min = x[i]; right_bases[peak_nr] = i
        //            i += 1
        let mut right_base = peak;
        let mut right_min = x_data[peak];
        idx = peak as isize;
        while idx <= i_max as isize && x_data[idx as usize] <= x_data[peak] {
            if x_data[idx as usize] < right_min {
                right_min = x_data[idx as usize];
                right_base = idx as usize;
            }
            idx += 1;
        }

        prominences.push(x_data[peak] - left_min.max(right_min));
        left_bases.push(left_base as isize);
        right_bases.push(right_base as isize);
    }

    Ok((
        prominences.into_pyobject(py)?.into_any(),
        left_bases.into_pyobject(py)?.into_any(),
        right_bases.into_pyobject(py)?.into_any(),
    ))
}

// ---------------------------------------------------------------------------
// _peak_widths
// ---------------------------------------------------------------------------

/// Calculate the width of each peak at a relative height.
#[pyfunction]
fn _peak_widths<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    peaks: &Bound<'py, PyAny>,
    rel_height: f64,
    prominences: &Bound<'py, PyAny>,
    left_bases: &Bound<'py, PyAny>,
    right_bases: &Bound<'py, PyAny>,
) -> PyResult<PeakWidthsOutput<'py>> {
    if rel_height < 0.0 {
        return Err(PyValueError::new_err(
            "`rel_height` must be greater or equal to 0.0",
        ));
    }

    let x_data = extract_f64_slice(py, x)?;
    let peaks_data = extract_intp_slice(py, peaks)?;
    let prom_data = extract_f64_slice(py, prominences)?;
    let lb_data = extract_intp_slice(py, left_bases)?;
    let rb_data = extract_intp_slice(py, right_bases)?;

    let x_data: &[f64] = &x_data;
    let peaks_data: &[isize] = &peaks_data;
    let prom_data: &[f64] = &prom_data;
    let lb_data: &[isize] = &lb_data;
    let rb_data: &[isize] = &rb_data;

    let n_peaks = peaks_data.len();
    if !(n_peaks == prom_data.len() && n_peaks == lb_data.len() && n_peaks == rb_data.len()) {
        return Err(PyValueError::new_err(
            "arrays in `prominence_data` must have the same shape as `peaks`",
        ));
    }

    let mut widths = Vec::with_capacity(n_peaks);
    let mut width_heights = Vec::with_capacity(n_peaks);
    let mut left_ips = Vec::with_capacity(n_peaks);
    let mut right_ips = Vec::with_capacity(n_peaks);

    for p in 0..n_peaks {
        let i_min = lb_data[p] as usize;
        let i_max = rb_data[p] as usize;
        let peak = peaks_data[p] as usize;

        if !(i_min <= peak && peak <= i_max && i_max < x_data.len()) {
            return Err(PyValueError::new_err(format!(
                "prominence data is invalid for peak {}",
                peak
            )));
        }

        let height = x_data[peak] - prom_data[p] * rel_height;
        width_heights.push(height);

        // Find intersection point on left side.
        let mut i = peak;
        while i_min < i && height < x_data[i] {
            i -= 1;
        }
        let mut left_ip = i as f64;
        if x_data[i] < height {
            // Interpolate if true intersection height is between samples.
            if i + 1 < x_data.len() && x_data[i + 1] != x_data[i] {
                left_ip += (height - x_data[i]) / (x_data[i + 1] - x_data[i]);
            }
        }

        // Find intersection point on right side.
        i = peak;
        while i < i_max && height < x_data[i] {
            i += 1;
        }
        let mut right_ip = i as f64;
        if x_data[i] < height {
            // Interpolate if true intersection height is between samples.
            if i > 0 && x_data[i - 1] != x_data[i] {
                right_ip -= (height - x_data[i]) / (x_data[i - 1] - x_data[i]);
            }
        }

        widths.push(right_ip - left_ip);
        left_ips.push(left_ip);
        right_ips.push(right_ip);
    }

    Ok((
        widths.into_pyobject(py)?.into_any(),
        width_heights.into_pyobject(py)?.into_any(),
        left_ips.into_pyobject(py)?.into_any(),
        right_ips.into_pyobject(py)?.into_any(),
    ))
}

// ---------------------------------------------------------------------------
// PeakPropertyWarning
// ---------------------------------------------------------------------------

/// Warning raised when a calculated peak property has an unexpected value.
#[pyclass]
struct PeakPropertyWarning;

// ---------------------------------------------------------------------------
// module
// ---------------------------------------------------------------------------

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_local_maxima_1d, m)?)?;
    m.add_function(wrap_pyfunction!(_select_by_peak_distance, m)?)?;
    m.add_function(wrap_pyfunction!(_peak_prominences, m)?)?;
    m.add_function(wrap_pyfunction!(_peak_widths, m)?)?;
    m.add_class::<PeakPropertyWarning>()?;
    Ok(())
}
