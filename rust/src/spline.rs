//! Spline filtering and mirror-symmetric IIR boundary kernels.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::f64::consts::PI;

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

    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

fn values_from_python(value: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| {
            Ok(Value {
                re: item.getattr("real")?.extract()?,
                im: item.getattr("imag")?.extract()?,
            })
        })
        .collect()
}

fn values_to_python(values: &[Value]) -> (Vec<f64>, Vec<f64>) {
    (
        values.iter().map(|value| value.re).collect(),
        values.iter().map(|value| value.im).collect(),
    )
}

fn signal_shape(shape: &[usize], data_len: usize) -> PyResult<(usize, usize)> {
    if !(1..=2).contains(&shape.len()) {
        return Err(PyValueError::new_err(
            "input must be one- or two-dimensional",
        ));
    }
    let (rows, len) = if shape.len() == 1 {
        (1, shape[0])
    } else {
        (shape[0], shape[1])
    };
    if rows.checked_mul(len) != Some(data_len) || len == 0 {
        return Err(PyValueError::new_err("input shape does not match data"));
    }
    Ok((rows, len))
}

fn effective_precision(precision: f64, single: bool) -> f64 {
    if precision <= 0.0 || precision > 1.0 {
        if single { 1e-6 } else { 1e-11 }
    } else {
        precision
    }
}

fn symiirorder1(
    input: &[Value],
    rows: usize,
    len: usize,
    mut z1: Value,
    precision: f64,
    single: bool,
    complex_input: bool,
) -> PyResult<Vec<Value>> {
    if !complex_input {
        z1.im = 0.0;
    }
    if z1.abs() >= 1.0 {
        return Err(PyValueError::new_err("|z1| must be less than 1.0"));
    }
    let threshold = effective_precision(precision, single).powi(2);
    let mut output = vec![Value::default(); rows];
    for (row, value) in output.iter_mut().enumerate() {
        *value = input[row * len];
    }
    let mut power = Value { re: 1.0, im: 0.0 };
    let mut k = 0;
    loop {
        power = power.mul(z1);
        for row in 0..rows {
            output[row] = output[row].add(power.mul(input[row * len + k]));
        }
        k += 1;
        if power.abs().powi(2) <= threshold || k >= len {
            break;
        }
    }
    if k >= len {
        return Err(PyValueError::new_err(
            "Sum to find symmetric boundary conditions did not converge.",
        ));
    }
    Ok(output)
}

fn hs(k: i32, cs: f64, rsq: f64, omega: f64) -> f64 {
    let k = k.abs();
    let rsupk = rsq.powf(k as f64 / 2.0);
    if omega == 0.0 || omega == PI {
        let c0 = (1.0 + rsq) / (1.0 - rsq).powi(3) * cs * cs;
        let sign = if omega == PI && k % 2 != 0 { -1.0 } else { 1.0 };
        let gamma = (1.0 - rsq) / (1.0 + rsq) * sign;
        return c0 * rsupk * (1.0 + gamma * k as f64);
    }
    let c0 =
        cs * cs * (1.0 + rsq) / (1.0 - rsq) / (1.0 - 2.0 * rsq * (2.0 * omega).cos() + rsq * rsq);
    let gamma = (1.0 - rsq) / (1.0 + rsq) / omega.tan();
    c0 * rsupk * ((omega * k as f64).cos() + gamma * (omega * k as f64).sin())
}

fn hc(k: i32, cs: f64, r: f64, omega: f64) -> f64 {
    if k < 0 {
        return 0.0;
    }
    let kf = k as f64;
    if omega == 0.0 {
        return cs * r.powf(kf) * (kf + 1.0);
    }
    if omega == PI {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        return cs * r.powf(kf) * (kf + 1.0) * sign;
    }
    cs * r.powf(kf) * (omega * (kf + 1.0)).sin() / omega.sin()
}

fn symiirorder2_forward(
    input: &[f64],
    rows: usize,
    len: usize,
    r: f64,
    omega: f64,
    precision: f64,
    single: bool,
) -> PyResult<Vec<f64>> {
    if len < 2 {
        return Err(PyValueError::new_err(
            "input must contain at least two samples",
        ));
    }
    let cs = 1.0 - 2.0 * r * omega.cos() + r * r;
    let threshold = effective_precision(precision, single).powi(2);
    let mut output = vec![0.0; rows * 2];
    for row in 0..rows {
        output[2 * row] = hc(0, cs, r, omega) * input[row * len];
    }

    let mut k = 0usize;
    loop {
        let diff = hc(k as i32 + 1, cs, r, omega);
        for row in 0..rows {
            output[2 * row] += diff * input[row * len + k];
        }
        k += 1;
        if diff * diff <= threshold || k >= len {
            break;
        }
    }
    if k >= len {
        return Err(PyValueError::new_err(
            "Sum to find symmetric boundary conditions did not converge.",
        ));
    }

    for row in 0..rows {
        output[2 * row + 1] =
            hc(0, cs, r, omega) * input[row * len + 1] + hc(1, cs, r, omega) * input[row * len];
    }
    k = 0;
    loop {
        let diff = hc(k as i32 + 2, cs, r, omega);
        for row in 0..rows {
            output[2 * row + 1] += diff * input[row * len + k];
        }
        k += 1;
        if diff * diff <= threshold || k >= len {
            break;
        }
    }
    if k >= len {
        return Err(PyValueError::new_err(
            "Sum to find symmetric boundary conditions did not converge.",
        ));
    }
    Ok(output)
}

fn symiirorder2_backward(
    input: &[f64],
    rows: usize,
    len: usize,
    r: f64,
    omega: f64,
    precision: f64,
    single: bool,
) -> PyResult<Vec<f64>> {
    let cs = 1.0 - 2.0 * r * omega.cos() + r * r;
    let threshold = effective_precision(precision, single);
    let mut output = vec![0.0; rows * 2];
    let mut k = 0;
    loop {
        let diff = hs(k as i32, cs, r * r, omega) + hs(k as i32 + 1, cs, r * r, omega);
        for row in 0..rows {
            output[2 * row] += diff * input[row * len + len - 1 - k];
        }
        k += 1;
        if diff * diff <= threshold || k >= len {
            break;
        }
    }
    if k >= len {
        return Err(PyValueError::new_err(
            "Sum to find symmetric boundary conditions did not converge.",
        ));
    }

    k = 0;
    loop {
        let diff = hs(k as i32 - 1, cs, r * r, omega) + hs(k as i32 + 2, cs, r * r, omega);
        for row in 0..rows {
            output[2 * row + 1] += diff * input[row * len + len - 1 - k];
        }
        k += 1;
        if diff * diff <= threshold || k >= len {
            break;
        }
    }
    if k >= len {
        return Err(PyValueError::new_err(
            "Sum to find symmetric boundary conditions did not converge.",
        ));
    }
    Ok(output)
}

fn mirror_index(index: isize, len: usize) -> usize {
    if index < 0 {
        (-index - 1) as usize
    } else if index >= len as isize {
        (2 * len as isize - 1 - index) as usize
    } else {
        index as usize
    }
}

fn sepfir2d(
    image: &[Value],
    rows: usize,
    cols: usize,
    hrow: &[Value],
    hcol: &[Value],
) -> PyResult<Vec<Value>> {
    if hrow.is_empty() || hcol.is_empty() || hrow.len() & 1 == 0 || hcol.len() & 1 == 0 {
        return Err(PyValueError::new_err("hrow and hcol must be odd length"));
    }
    if cols < hrow.len() - 1 {
        return Err(PyValueError::new_err(
            "hrow must not be longer than input.shape[1] + 1",
        ));
    }
    if rows < hcol.len() - 1 {
        return Err(PyValueError::new_err(
            "hcol must not be longer than input.shape[0] + 1",
        ));
    }
    if image.len() != rows * cols {
        return Err(PyValueError::new_err("image shape does not match data"));
    }

    let mut horizontal = vec![Value::default(); rows * cols];
    let row_half = (hrow.len() / 2) as isize;
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = Value::default();
            for (filter_index, coefficient) in hrow.iter().enumerate() {
                let source_col =
                    mirror_index(col as isize + row_half - filter_index as isize, cols);
                sum = sum.add(image[row * cols + source_col].mul(*coefficient));
            }
            horizontal[row * cols + col] = sum;
        }
    }

    let mut output = vec![Value::default(); rows * cols];
    let col_half = (hcol.len() / 2) as isize;
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = Value::default();
            for (filter_index, coefficient) in hcol.iter().enumerate() {
                let source_row =
                    mirror_index(row as isize + col_half - filter_index as isize, rows);
                sum = sum.add(horizontal[source_row * cols + col].mul(*coefficient));
            }
            output[row * cols + col] = sum;
        }
    }
    Ok(output)
}

#[pyfunction]
fn _sepfir2d<'py>(
    image: &Bound<'py, PyAny>,
    image_shape: Vec<usize>,
    hrow: &Bound<'py, PyAny>,
    hcol: &Bound<'py, PyAny>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if image_shape.len() != 2 {
        return Err(PyValueError::new_err("image must be two-dimensional"));
    }
    let image = values_from_python(image)?;
    let hrow = values_from_python(hrow)?;
    let hcol = values_from_python(hcol)?;
    let result = sepfir2d(&image, image_shape[0], image_shape[1], &hrow, &hcol)?;
    Ok(values_to_python(&result))
}

#[pyfunction]
fn _symiirorder1_ic<'py>(
    input: &Bound<'py, PyAny>,
    shape: Vec<usize>,
    z1: &Bound<'py, PyAny>,
    precision: f64,
    single: bool,
    complex_input: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let input = values_from_python(input)?;
    let (rows, len) = signal_shape(&shape, input.len())?;
    if len < 2 {
        return Err(PyValueError::new_err(
            "input must contain at least two samples",
        ));
    }
    let z1 = Value {
        re: z1.getattr("real")?.extract()?,
        im: z1.getattr("imag")?.extract()?,
    };
    let result = symiirorder1(&input, rows, len, z1, precision, single, complex_input)?;
    Ok(values_to_python(&result))
}

#[pyfunction]
fn _symiirorder2_ic_fwd<'py>(
    input: &Bound<'py, PyAny>,
    shape: Vec<usize>,
    r: f64,
    omega: f64,
    precision: f64,
    single: bool,
) -> PyResult<Vec<f64>> {
    let input = values_from_python(input)?;
    let input = input.iter().map(|value| value.re).collect::<Vec<_>>();
    let (rows, len) = signal_shape(&shape, input.len())?;
    symiirorder2_forward(&input, rows, len, r, omega, precision, single)
}

#[pyfunction]
fn _symiirorder2_ic_bwd<'py>(
    input: &Bound<'py, PyAny>,
    shape: Vec<usize>,
    r: f64,
    omega: f64,
    precision: f64,
    single: bool,
) -> PyResult<Vec<f64>> {
    let input = values_from_python(input)?;
    let input = input.iter().map(|value| value.re).collect::<Vec<_>>();
    let (rows, len) = signal_shape(&shape, input.len())?;
    symiirorder2_backward(&input, rows, len, r, omega, precision, single)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_sepfir2d, module)?)?;
    module.add_function(wrap_pyfunction!(_symiirorder1_ic, module)?)?;
    module.add_function(wrap_pyfunction!(_symiirorder2_ic_fwd, module)?)?;
    module.add_function(wrap_pyfunction!(_symiirorder2_ic_bwd, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Value, sepfir2d, symiirorder1, symiirorder2_backward, symiirorder2_forward};

    fn real(values: &[f64]) -> Vec<Value> {
        values.iter().map(|&re| Value { re, im: 0.0 }).collect()
    }

    #[test]
    fn applies_separable_mirror_filter() {
        let image = real(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let output = sepfir2d(&image, 2, 3, &real(&[0.5, 1.0, 0.5]), &real(&[1.0])).unwrap();
        assert_eq!(
            output.iter().map(|value| value.re).collect::<Vec<_>>(),
            vec![2.5, 4.0, 5.5, 2.5, 4.0, 5.5]
        );
    }

    #[test]
    fn computes_first_order_real_and_complex_initial_conditions() {
        let input = real(&[1.0; 200]);
        let result = symiirorder1(
            &input,
            1,
            200,
            Value { re: 0.85, im: 0.0 },
            -1.0,
            false,
            false,
        )
        .unwrap();
        assert!((result[0].re - 1.0 / (1.0 - 0.85)).abs() < 1e-9);

        let complex = [Value { re: 1.0, im: 1.0 }; 100];
        let result = symiirorder1(
            &complex,
            1,
            100,
            Value { re: 0.5, im: 0.25 },
            0.001,
            false,
            true,
        )
        .unwrap();
        assert!(result[0].im > 1.0);
    }

    #[test]
    fn computes_second_order_boundary_conditions() {
        let input = vec![1.0; 100];
        let omega = std::f64::consts::PI / 3.0;
        let fwd = symiirorder2_forward(&input, 1, 100, 0.5, omega, -1.0, false).unwrap();
        assert!((fwd[0] - 1.125).abs() < 1e-9);
        let bwd = symiirorder2_backward(&input, 1, 100, 0.5, omega, -1.0, false).unwrap();
        assert!(bwd.iter().all(|value| value.is_finite()));
    }
}
