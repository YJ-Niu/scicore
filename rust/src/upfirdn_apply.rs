//! Polyphase upsample/filter/downsample kernels and signal extensions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Clone, Copy, Default)]
struct Value {
    re: f64,
    im: f64,
}

impl Value {
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
}

fn value_from_python(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    Ok(Value {
        re: value.getattr("real")?.extract()?,
        im: value.getattr("imag")?.extract()?,
    })
}

fn values_from_python(value: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| value_from_python(&item))
        .collect()
}

fn values_to_python(values: &[Value]) -> (Vec<f64>, Vec<f64>) {
    let real = values.iter().map(|value| value.re).collect();
    let imag = values.iter().map(|value| value.im).collect();
    (real, imag)
}

fn encode_mode(mode: &str) -> PyResult<usize> {
    match mode {
        "constant" => Ok(0),
        "symmetric" => Ok(1),
        "edge" => Ok(2),
        "smooth" => Ok(3),
        "wrap" => Ok(4),
        "reflect" => Ok(5),
        "antisymmetric" => Ok(6),
        "antireflect" => Ok(7),
        "line" => Ok(8),
        _ => Err(PyValueError::new_err(format!("Unknown mode: {mode}"))),
    }
}

fn extend_left(x: &[Value], idx: isize, mode: usize, cval: Value) -> Value {
    let len = x.len() as isize;
    match mode {
        0 => cval,
        1 => {
            let offset = -idx - 1;
            let period_idx = offset.rem_euclid(2 * len);
            x[if period_idx < len {
                period_idx
            } else {
                2 * len - 1 - period_idx
            } as usize]
        }
        2 => x[0],
        3 => x[0].add(x[1].sub(x[0]).scale(idx as f64)),
        4 => x[(-idx - 1).rem_euclid(len) as usize],
        5 => {
            let period = 2 * (len - 1);
            let period_idx = (-idx - 1).rem_euclid(period);
            x[if period_idx < len - 1 {
                period_idx + 1
            } else {
                2 * len - 2 - period_idx
            } as usize]
        }
        6 => {
            let offset = -idx - 1;
            let period_idx = offset.rem_euclid(2 * len);
            if period_idx < len {
                x[period_idx as usize].scale(-1.0)
            } else {
                x[(2 * len - 1 - period_idx) as usize]
            }
        }
        7 => {
            if -idx < len {
                x[0].scale(2.0).sub(x[-idx as usize])
            } else {
                let slope_count = (-idx - 1).div_euclid(len - 1) as f64;
                let level = x[0].add(x[0].sub(x[len as usize - 1]).scale(slope_count));
                let period_idx = (-idx - 1).rem_euclid(2 * (len - 1));
                if period_idx < len - 1 {
                    level.sub(x[(period_idx + 1) as usize].sub(x[0]))
                } else {
                    level.sub(x[len as usize - 1].sub(x[(2 * len - 3 - period_idx) as usize]))
                }
            }
        }
        8 => {
            let slope = x[len as usize - 1].sub(x[0]).scale(1.0 / (len - 1) as f64);
            x[0].add(slope.scale(idx as f64))
        }
        _ => unreachable!("mode was validated by the Python entry point"),
    }
}

fn extend_right(x: &[Value], idx: usize, mode: usize, cval: Value) -> Value {
    let len = x.len();
    match mode {
        0 => cval,
        1 => {
            let period_idx = idx % (2 * len);
            x[if period_idx < len {
                period_idx
            } else {
                2 * len - 1 - period_idx
            }]
        }
        2 => x[len - 1],
        3 => x[len - 1].add(x[len - 1].sub(x[len - 2]).scale((idx - len + 1) as f64)),
        4 => x[idx % len],
        5 => {
            let period_idx = idx % (2 * (len - 1));
            x[if period_idx < len - 1 {
                period_idx
            } else {
                2 * len - 2 - period_idx
            }]
        }
        6 => {
            let period_idx = idx % (2 * len);
            if period_idx < len {
                x[period_idx]
            } else {
                x[2 * len - 1 - period_idx].scale(-1.0)
            }
        }
        7 => {
            if idx < 2 * len - 1 {
                x[len - 1].scale(2.0).sub(x[2 * len - 2 - idx])
            } else {
                let level =
                    x[len - 1].add(x[len - 1].sub(x[0]).scale((idx / (len - 1) - 1) as f64));
                let period_idx = idx % (2 * (len - 1));
                if period_idx < len - 1 {
                    level.add(x[period_idx].sub(x[0]))
                } else {
                    level.add(x[len - 1].sub(x[2 * len - 2 - period_idx]))
                }
            }
        }
        8 => {
            let slope = x[len - 1].sub(x[0]).scale(1.0 / (len - 1) as f64);
            x[len - 1].add(slope.scale((idx - len + 1) as f64))
        }
        _ => unreachable!("mode was validated by the Python entry point"),
    }
}

fn extended_value(x: &[Value], idx: isize, mode: usize, cval: Value) -> Value {
    if idx < 0 {
        extend_left(x, idx, mode, cval)
    } else if idx >= x.len() as isize {
        extend_right(x, idx as usize, mode, cval)
    } else {
        x[idx as usize]
    }
}

fn output_len(len_h: usize, input_len: usize, up: usize, down: usize) -> PyResult<usize> {
    if len_h == 0 || input_len == 0 || up == 0 || down == 0 {
        return Err(PyValueError::new_err(
            "filter and input lengths and up/down factors must be positive",
        ));
    }
    let last = (input_len - 1)
        .checked_mul(up)
        .and_then(|value| value.checked_add(len_h - 1))
        .ok_or_else(|| PyValueError::new_err("output length overflow"))?;
    Ok(last / down + 1)
}

fn apply_row(
    x: &[Value],
    filter: &[Value],
    up: usize,
    down: usize,
    mode: usize,
    cval: Value,
    output: &mut [Value],
) {
    let filter_per_phase = filter.len() / up;
    for (output_index, result) in output.iter_mut().enumerate() {
        let time = output_index * down;
        let phase = time % up;
        let latest_x = (time / up) as isize;
        let earliest_x = ((time as isize - filter.len() as isize + 1).div_euclid(up as isize))
            + i32::from((time as isize - filter.len() as isize + 1).rem_euclid(up as isize) != 0)
                as isize;

        let mut sum = Value::default();
        for input_index in earliest_x..=latest_x {
            let filter_index = time as isize - input_index * up as isize;
            let phase_filter_index =
                phase * filter_per_phase + filter_per_phase - 1 - filter_index as usize / up;
            sum =
                sum.add(extended_value(x, input_index, mode, cval).mul(filter[phase_filter_index]));
        }
        *result = sum;
    }
}

#[pyfunction]
fn _output_len(len_h: usize, input_len: usize, up: usize, down: usize) -> PyResult<usize> {
    output_len(len_h, input_len, up, down)
}

#[pyfunction]
fn mode_enum(mode: &str) -> PyResult<usize> {
    encode_mode(&mode.to_lowercase())
}

#[pyfunction]
fn _pad_test<'py>(
    data: &Bound<'py, PyAny>,
    npre: usize,
    npost: usize,
    mode: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let values = values_from_python(data)?;
    if values.is_empty() {
        return Err(PyValueError::new_err("data must not be empty"));
    }
    if mode > 8 {
        return Err(PyValueError::new_err("invalid extension mode"));
    }
    if matches!(mode, 3 | 5 | 7 | 8) && values.len() < 2 {
        return Err(PyValueError::new_err(
            "this extension mode requires at least two data points",
        ));
    }
    let mut output = Vec::with_capacity(npre + values.len() + npost);
    for idx in -(npre as isize)..values.len() as isize + npost as isize {
        output.push(extended_value(&values, idx, mode, Value::default()));
    }
    Ok(values_to_python(&output))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _apply<'py>(
    data: &Bound<'py, PyAny>,
    filter: &Bound<'py, PyAny>,
    shape: Vec<usize>,
    output_shape: Vec<usize>,
    up: usize,
    down: usize,
    axis: usize,
    mode: usize,
    cval: &Bound<'py, PyAny>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if shape.is_empty() || shape.len() != output_shape.len() {
        return Err(PyValueError::new_err(
            "data and output must have the same number of dimensions",
        ));
    }
    if axis >= shape.len() {
        return Err(PyValueError::new_err("axis is out of bounds"));
    }
    if up == 0 || down == 0 || mode > 8 {
        return Err(PyValueError::new_err(
            "up/down factors and extension mode are invalid",
        ));
    }

    let input_values = values_from_python(data)?;
    let filter_values = values_from_python(filter)?;
    let input_count = shape.iter().product::<usize>();
    let output_count = output_shape.iter().product::<usize>();
    if input_values.len() != input_count || output_count == 0 {
        return Err(PyValueError::new_err("data and shape do not match"));
    }
    if filter_values.is_empty() || filter_values.len() % up != 0 {
        return Err(PyValueError::new_err(
            "filter length must be a positive multiple of up",
        ));
    }
    if output_shape[axis] > output_len(filter_values.len(), shape[axis], up, down)? {
        return Err(PyValueError::new_err(
            "output length exceeds the filter result length",
        ));
    }
    if matches!(mode, 3 | 5 | 7 | 8) && shape[axis] < 2 {
        return Err(PyValueError::new_err(
            "this extension mode requires at least two data points",
        ));
    }

    let cval = value_from_python(cval)?;
    let mut output = vec![Value::default(); output_count];
    let num_rows = output_count / output_shape[axis];
    let mut row_input = vec![Value::default(); shape[axis]];
    let mut row_output = vec![Value::default(); output_shape[axis]];

    for row in 0..num_rows {
        let mut reduced = row;
        let mut input_offset = 0;
        let mut output_offset = 0;
        for dim in (0..shape.len()).rev() {
            if dim != axis {
                let index = reduced % output_shape[dim];
                reduced /= output_shape[dim];
                input_offset += index * shape[dim + 1..].iter().product::<usize>();
                output_offset += index * output_shape[dim + 1..].iter().product::<usize>();
            }
        }

        let input_stride = shape[axis + 1..].iter().product::<usize>();
        let output_stride = output_shape[axis + 1..].iter().product::<usize>();
        for index in 0..shape[axis] {
            row_input[index] = input_values[input_offset + index * input_stride];
        }
        apply_row(
            &row_input,
            &filter_values,
            up,
            down,
            mode,
            cval,
            &mut row_output,
        );
        for index in 0..output_shape[axis] {
            output[output_offset + index * output_stride] = row_output[index];
        }
    }

    Ok(values_to_python(&output))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_output_len, module)?)?;
    module.add_function(wrap_pyfunction!(mode_enum, module)?)?;
    module.add_function(wrap_pyfunction!(_pad_test, module)?)?;
    module.add_function(wrap_pyfunction!(_apply, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Value, apply_row, encode_mode, extend_left, extend_right, output_len};

    fn real_values(values: &[f64]) -> Vec<Value> {
        values.iter().map(|&re| Value { re, im: 0.0 }).collect()
    }

    #[test]
    fn computes_output_lengths_without_overflow() {
        assert_eq!(output_len(31, 10_000, 1, 2).unwrap(), 5_015);
        assert_eq!(output_len(3, 3, 2, 3).unwrap(), 3);
        assert!(output_len(0, 1, 1, 1).is_err());
    }

    #[test]
    fn maps_all_supported_extension_modes() {
        for (name, value) in [
            ("constant", 0),
            ("symmetric", 1),
            ("edge", 2),
            ("smooth", 3),
            ("wrap", 4),
            ("reflect", 5),
            ("antisymmetric", 6),
            ("antireflect", 7),
            ("line", 8),
        ] {
            assert_eq!(encode_mode(name).unwrap(), value);
        }
        assert!(encode_mode("invalid").is_err());
    }

    #[test]
    fn extends_signal_using_antireflect_and_symmetric_modes() {
        let x = real_values(&[1.0, 2.0, 3.0, 1.0]);
        let antireflect = (-(6isize)..10)
            .map(|idx| {
                if idx < 0 {
                    super::extend_left(&x, idx, 7, Value::default()).re
                } else if idx >= x.len() as isize {
                    extend_right(&x, idx as usize, 7, Value::default()).re
                } else {
                    x[idx as usize].re
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            &antireflect[..16],
            &[
                1.0, 2.0, 3.0, 1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 1.0
            ]
        );

        assert_eq!(extend_left(&x, -1, 1, Value::default()).re, 1.0);
        assert_eq!(extend_right(&x, 4, 1, Value::default()).re, 1.0);
    }

    #[test]
    fn applies_polyphase_filter_to_real_and_complex_data() {
        let x = real_values(&[1.0, -2.0, 3.0]);
        let filter = real_values(&[0.75, -0.25, 0.5, 0.0, 2.0, 1.0]);
        let mut output = vec![Value::default(); output_len(5, 3, 2, 3).unwrap()];
        apply_row(&x, &filter, 2, 3, 0, Value::default(), &mut output);
        assert_eq!(
            output.iter().map(|value| value.re).collect::<Vec<_>>(),
            vec![0.5, 0.0, -2.25]
        );

        let complex_x = [Value { re: 1.0, im: 1.0 }];
        let complex_filter = [Value { re: 2.0, im: -1.0 }];
        let mut complex_output = [Value::default()];
        apply_row(
            &complex_x,
            &complex_filter,
            1,
            1,
            0,
            Value::default(),
            &mut complex_output,
        );
        assert_eq!(complex_output[0].re, 3.0);
        assert_eq!(complex_output[0].im, 1.0);
    }
}
