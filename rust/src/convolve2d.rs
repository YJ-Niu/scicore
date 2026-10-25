use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

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
}

#[derive(Clone, Copy)]
struct ConvolveOptions {
    input_shape: [usize; 2],
    kernel_shape: [usize; 2],
    flip: bool,
    mode: usize,
    boundary: usize,
    fill: Value,
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

fn reflect_index(index: isize, len: usize) -> usize {
    let period = 2 * len as isize;
    let wrapped = index.rem_euclid(period);
    if wrapped >= len as isize {
        (period - wrapped - 1) as usize
    } else {
        wrapped as usize
    }
}

fn boundary_index(index: isize, len: usize, boundary: usize) -> Option<usize> {
    if (0..len as isize).contains(&index) {
        return Some(index as usize);
    }
    match boundary {
        0 => None,
        4 => Some(reflect_index(index, len)),
        8 => Some(index.rem_euclid(len as isize) as usize),
        _ => None,
    }
}

fn convolve_2d(
    input: &[Value],
    kernel: &[Value],
    options: ConvolveOptions,
) -> PyResult<(Vec<f64>, Vec<f64>, [usize; 2])> {
    let ConvolveOptions {
        input_shape: [rows, cols],
        kernel_shape: [kernel_rows, kernel_cols],
        flip,
        mode,
        boundary,
        fill,
    } = options;
    if rows.checked_mul(cols) != Some(input.len())
        || kernel_rows.checked_mul(kernel_cols) != Some(kernel.len())
    {
        return Err(PyValueError::new_err("array shape does not match data"));
    }
    if rows == 0 || cols == 0 || kernel_rows == 0 || kernel_cols == 0 {
        return Err(PyValueError::new_err(
            "inputs must have non-zero dimensions",
        ));
    }
    if mode > 2 {
        return Err(PyValueError::new_err(
            "mode must be 0 (valid), 1 (same), or 2 (full)",
        ));
    }
    if !matches!(boundary, 0 | 4 | 8) {
        return Err(PyValueError::new_err("Incorrect boundary value."));
    }
    if mode == 0 && (kernel_rows > rows || kernel_cols > cols) {
        return Err(PyValueError::new_err(
            "no part of the output is valid, use option 1 (same) or 2 (full) for third argument",
        ));
    }

    let output_shape = match mode {
        0 => [rows - kernel_rows + 1, cols - kernel_cols + 1],
        1 => [rows, cols],
        _ => [rows + kernel_rows - 1, cols + kernel_cols - 1],
    };
    let output_len = output_shape[0]
        .checked_mul(output_shape[1])
        .ok_or_else(|| PyValueError::new_err("output shape is too large"))?;
    let mut output = vec![Value::default(); output_len];

    for out_row in 0..output_shape[0] {
        let base_row = match mode {
            0 if flip => out_row as isize + kernel_rows as isize - 1,
            0 => out_row as isize,
            1 if flip => out_row as isize + ((kernel_rows - 1) >> 1) as isize,
            1 => out_row as isize - ((kernel_rows - 1) >> 1) as isize,
            _ if flip => out_row as isize,
            _ => out_row as isize - kernel_rows as isize + 1,
        };
        for out_col in 0..output_shape[1] {
            let base_col = match mode {
                0 if flip => out_col as isize + kernel_cols as isize - 1,
                0 => out_col as isize,
                1 if flip => out_col as isize + ((kernel_cols - 1) >> 1) as isize,
                1 => out_col as isize - ((kernel_cols - 1) >> 1) as isize,
                _ if flip => out_col as isize,
                _ => out_col as isize - kernel_cols as isize + 1,
            };

            let mut sum = Value::default();
            for kernel_row in 0..kernel_rows {
                let row_offset = if flip {
                    base_row - kernel_row as isize
                } else {
                    base_row + kernel_row as isize
                };
                let source_row = boundary_index(row_offset, rows, boundary);
                for kernel_col in 0..kernel_cols {
                    let col_offset = if flip {
                        base_col - kernel_col as isize
                    } else {
                        base_col + kernel_col as isize
                    };
                    let source_col = boundary_index(col_offset, cols, boundary);
                    let sample = match (source_row, source_col) {
                        (Some(row), Some(col)) => input[row * cols + col],
                        _ => fill,
                    };
                    let coefficient = kernel[kernel_row * kernel_cols + kernel_col];
                    sum = sum.add(sample.mul(coefficient));
                }
            }
            output[out_row * output_shape[1] + out_col] = sum;
        }
    }

    let real = output.iter().map(|value| value.re).collect();
    let imag = output.iter().map(|value| value.im).collect();
    Ok((real, imag, output_shape))
}

#[pyfunction]
fn _convolve2d(
    input: &Bound<'_, PyAny>,
    kernel: &Bound<'_, PyAny>,
    options: &Bound<'_, PyTuple>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let input_shape = options.get_item(0)?.extract::<Vec<usize>>()?;
    let kernel_shape = options.get_item(1)?.extract::<Vec<usize>>()?;
    let flip = options.get_item(2)?.extract::<bool>()?;
    let mode = options.get_item(3)?.extract::<usize>()?;
    let boundary = options.get_item(4)?.extract::<usize>()?;
    let fill_value = options.get_item(5)?;
    if input_shape.len() != 2 || kernel_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "convolve2d inputs must both be 2-D arrays",
        ));
    }
    let input_shape = [input_shape[0], input_shape[1]];
    let kernel_shape = [kernel_shape[0], kernel_shape[1]];
    let input = values_from_python(input)?;
    let kernel = values_from_python(kernel)?;
    let fill = value_from_python(&fill_value)?;
    let options = ConvolveOptions {
        input_shape,
        kernel_shape,
        flip,
        mode,
        boundary,
        fill,
    };
    let (real, imag, shape) = convolve_2d(&input, &kernel, options)?;
    Ok((real, imag, shape.to_vec()))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_convolve2d, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ConvolveOptions, Value, convolve_2d};

    fn real(values: &[f64]) -> Vec<Value> {
        values.iter().map(|&re| Value { re, im: 0.0 }).collect()
    }

    #[test]
    fn matches_full_same_and_valid_convolution() {
        let input = real(&[1.0, 2.0, 3.0, 3.0, 4.0, 5.0]);
        let kernel = real(&[2.0, 3.0, 4.0, 4.0, 5.0, 6.0]);
        let fill = Value::default();
        let (full, _, shape) = convolve_2d(
            &input,
            &kernel,
            ConvolveOptions {
                input_shape: [2, 3],
                kernel_shape: [2, 3],
                flip: true,
                mode: 2,
                boundary: 0,
                fill,
            },
        )
        .unwrap();
        assert_eq!(shape, [3, 5]);
        assert_eq!(
            full,
            [
                2.0, 7.0, 16.0, 17.0, 12.0, 10.0, 30.0, 62.0, 58.0, 38.0, 12.0, 31.0, 58.0, 49.0,
                30.0
            ]
        );

        let (same, _, shape) = convolve_2d(
            &input,
            &kernel,
            ConvolveOptions {
                input_shape: [2, 3],
                kernel_shape: [2, 3],
                flip: true,
                mode: 1,
                boundary: 0,
                fill,
            },
        )
        .unwrap();
        assert_eq!(shape, [2, 3]);
        assert_eq!(same, [7.0, 16.0, 17.0, 30.0, 62.0, 58.0]);

        let kernel = real(&[2.0, 3.0, 4.0, 5.0]);
        let (valid, _, shape) = convolve_2d(
            &input,
            &kernel,
            ConvolveOptions {
                input_shape: [2, 3],
                kernel_shape: [2, 2],
                flip: true,
                mode: 0,
                boundary: 0,
                fill,
            },
        )
        .unwrap();
        assert_eq!(shape, [1, 2]);
        assert_eq!(valid, [30.0, 44.0]);
    }

    #[test]
    fn supports_wrapping_and_symmetric_extensions() {
        let input = real(&[1.0, 2.0, 3.0, 3.0, 4.0, 5.0]);
        let kernel = real(&[2.0, 3.0, 4.0, 4.0, 5.0, 6.0]);
        let fill = Value::default();
        let options = |boundary| ConvolveOptions {
            input_shape: [2, 3],
            kernel_shape: [2, 3],
            flip: true,
            mode: 2,
            boundary,
            fill,
        };
        let (wrapped, _, _) = convolve_2d(&input, &kernel, options(8)).unwrap();
        assert_eq!(
            wrapped,
            [
                80.0, 80.0, 74.0, 80.0, 80.0, 68.0, 68.0, 62.0, 68.0, 68.0, 80.0, 80.0, 74.0, 80.0,
                80.0
            ]
        );

        let (symmetric, _, _) = convolve_2d(&input, &kernel, options(4)).unwrap();
        assert_eq!(
            symmetric,
            [
                34.0, 30.0, 44.0, 62.0, 66.0, 52.0, 48.0, 62.0, 80.0, 84.0, 82.0, 78.0, 92.0,
                110.0, 114.0
            ]
        );
    }

    #[test]
    fn computes_correlation_without_flipping_the_kernel() {
        let input = real(&[1.0, 2.0]);
        let kernel = real(&[3.0, 4.0]);
        let (output, _, shape) = convolve_2d(
            &input,
            &kernel,
            ConvolveOptions {
                input_shape: [1, 2],
                kernel_shape: [1, 2],
                flip: false,
                mode: 2,
                boundary: 0,
                fill: Value::default(),
            },
        )
        .unwrap();
        assert_eq!(shape, [1, 3]);
        assert_eq!(output, [4.0, 11.0, 6.0]);
    }
}
