use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

#[derive(Clone, Copy)]
struct CorrelationOptions<'a> {
    input_shape: &'a [usize],
    kernel_shape: &'a [usize],
    mode: usize,
    single_precision: bool,
}

#[derive(Clone, Copy, Default)]
struct Value {
    re: f64,
    im: f64,
}

fn values_from_python(value: &Bound<'_, PyAny>, itemsize: usize) -> PyResult<Vec<f64>> {
    let bytes = value.cast::<PyBytes>()?.as_bytes();
    match itemsize {
        4 if bytes.len() % 4 == 0 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk) as f64)
            .collect()),
        8 if bytes.len() % 8 == 0 => Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| f64::from_le_bytes(*chunk))
            .collect()),
        _ => Err(PyValueError::new_err("unsupported floating-point width")),
    }
}

fn checked_size(shape: &[usize]) -> PyResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |size, &dimension| size.checked_mul(dimension))
        .ok_or_else(|| PyValueError::new_err("array shape is too large"))
}

fn shapes_and_size(
    input_shape: &[usize],
    kernel_shape: &[usize],
    mode: usize,
) -> PyResult<(Vec<usize>, usize)> {
    if input_shape.is_empty() || input_shape.len() != kernel_shape.len() {
        return Err(PyValueError::new_err(
            "Arrays must have the same non-zero number of dimensions.",
        ));
    }
    if input_shape.contains(&0) || kernel_shape.contains(&0) {
        return Err(PyValueError::new_err("Arrays cannot be empty."));
    }
    if mode > 2 {
        return Err(PyValueError::new_err(
            "mode must be 0 (valid), 1 (same), or 2 (full)",
        ));
    }
    if mode == 0 && input_shape.iter().zip(kernel_shape).any(|(x, y)| x < y) {
        return Err(PyValueError::new_err(
            "For 'valid' mode, the first input must be at least as large as the second in every dimension.",
        ));
    }

    let output_shape = input_shape
        .iter()
        .zip(kernel_shape)
        .map(|(&input, &kernel)| match mode {
            0 => Ok(input - kernel + 1),
            1 => Ok(input),
            _ => input
                .checked_add(kernel)
                .and_then(|size| size.checked_sub(1))
                .ok_or_else(|| PyValueError::new_err("output shape is too large")),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let output_size = checked_size(&output_shape)
        .map_err(|_| PyValueError::new_err("output shape is too large"))?;
    Ok((output_shape, output_size))
}

fn coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = index % shape[axis];
        index /= shape[axis];
    }
    result
}

fn correlate_float(
    input_real: &[f64],
    input_imag: &[f64],
    kernel_real: &[f64],
    kernel_imag: &[f64],
    options: CorrelationOptions<'_>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let CorrelationOptions {
        input_shape,
        kernel_shape,
        mode,
        single_precision,
    } = options;
    let input_size = checked_size(input_shape)?;
    let kernel_size = checked_size(kernel_shape)?;
    if input_real.len() != input_size
        || input_imag.len() != input_size
        || kernel_real.len() != kernel_size
        || kernel_imag.len() != kernel_size
    {
        return Err(PyValueError::new_err("array shape does not match data"));
    }
    let (output_shape, output_size) = shapes_and_size(input_shape, kernel_shape, mode)?;
    let mut output = vec![Value::default(); output_size];
    let kernel_coordinates = (0..kernel_size)
        .map(|index| coordinates(index, kernel_shape))
        .collect::<Vec<_>>();

    for (output_index, result) in output.iter_mut().enumerate() {
        let output_coordinate = coordinates(output_index, &output_shape);
        let mut sum = Value::default();
        for (kernel_index, kernel_coordinate) in kernel_coordinates.iter().enumerate() {
            let mut input_index = 0;
            let mut in_bounds = true;
            for axis in 0..input_shape.len() {
                let start = match mode {
                    0 => kernel_shape[axis] - 1,
                    1 => (kernel_shape[axis] - 1) / 2,
                    _ => 0,
                };
                let coordinate = output_coordinate[axis] as i128 + start as i128
                    - (kernel_shape[axis] - 1) as i128
                    + kernel_coordinate[axis] as i128;
                if coordinate < 0 || coordinate >= input_shape[axis] as i128 {
                    in_bounds = false;
                    break;
                }
                input_index = input_index * input_shape[axis] + coordinate as usize;
            }
            if !in_bounds {
                continue;
            }

            let xr = input_real[input_index];
            let xi = input_imag[input_index];
            let yr = kernel_real[kernel_index];
            let yi = kernel_imag[kernel_index];
            if single_precision {
                let real_product = (xr as f32 * yr as f32) + (xi as f32 * yi as f32);
                let imag_product = (xi as f32 * yr as f32) - (xr as f32 * yi as f32);
                sum.re = (sum.re as f32 + real_product) as f64;
                sum.im = (sum.im as f32 + imag_product) as f64;
            } else {
                sum.re += xr * yr + xi * yi;
                sum.im += xi * yr - xr * yi;
            }
        }
        *result = sum;
    }

    let real = output.iter().map(|value| value.re).collect();
    let imag = output.iter().map(|value| value.im).collect();
    Ok((real, imag, output_shape))
}

trait Integer: Copy + Default {
    fn multiply(self, rhs: Self, bits: u32) -> Self;
    fn add(self, rhs: Self, bits: u32) -> Self;
}

fn signed_value(value: i64, bits: u32) -> i64 {
    if bits == 64 {
        value
    } else {
        let shift = 64 - bits;
        (value << shift) >> shift
    }
}

fn unsigned_value(value: u64, bits: u32) -> u64 {
    if bits == 64 {
        value
    } else {
        value & ((1u64 << bits) - 1)
    }
}

impl Integer for i64 {
    fn multiply(self, rhs: Self, bits: u32) -> Self {
        signed_value(self.wrapping_mul(rhs), bits)
    }

    fn add(self, rhs: Self, bits: u32) -> Self {
        signed_value(self.wrapping_add(rhs), bits)
    }
}

impl Integer for u64 {
    fn multiply(self, rhs: Self, bits: u32) -> Self {
        unsigned_value(self.wrapping_mul(rhs), bits)
    }

    fn add(self, rhs: Self, bits: u32) -> Self {
        unsigned_value(self.wrapping_add(rhs), bits)
    }
}

fn correlate_integer<T: Integer>(
    input: &[T],
    kernel: &[T],
    input_shape: &[usize],
    kernel_shape: &[usize],
    mode: usize,
    bits: u32,
) -> PyResult<(Vec<T>, Vec<usize>)> {
    let input_size = checked_size(input_shape)?;
    let kernel_size = checked_size(kernel_shape)?;
    if input.len() != input_size || kernel.len() != kernel_size {
        return Err(PyValueError::new_err("array shape does not match data"));
    }
    if !matches!(bits, 8 | 16 | 32 | 64) {
        return Err(PyValueError::new_err("unsupported integer width"));
    }
    let (output_shape, output_size) = shapes_and_size(input_shape, kernel_shape, mode)?;
    let kernel_coordinates = (0..kernel_size)
        .map(|index| coordinates(index, kernel_shape))
        .collect::<Vec<_>>();
    let mut output = vec![T::default(); output_size];

    for (output_index, result) in output.iter_mut().enumerate() {
        let output_coordinate = coordinates(output_index, &output_shape);
        let mut sum = T::default();
        for (kernel_index, kernel_coordinate) in kernel_coordinates.iter().enumerate() {
            let mut input_index = 0;
            let mut in_bounds = true;
            for axis in 0..input_shape.len() {
                let start = match mode {
                    0 => kernel_shape[axis] - 1,
                    1 => (kernel_shape[axis] - 1) / 2,
                    _ => 0,
                };
                let coordinate = output_coordinate[axis] as i128 + start as i128
                    - (kernel_shape[axis] - 1) as i128
                    + kernel_coordinate[axis] as i128;
                if coordinate < 0 || coordinate >= input_shape[axis] as i128 {
                    in_bounds = false;
                    break;
                }
                input_index = input_index * input_shape[axis] + coordinate as usize;
            }
            if in_bounds {
                sum = sum.add(
                    input[input_index].multiply(kernel[kernel_index], bits),
                    bits,
                );
            }
        }
        *result = sum;
    }
    Ok((output, output_shape))
}

fn signed_values(value: &Bound<'_, PyAny>, bits: u32) -> PyResult<Vec<i64>> {
    let bytes = value.cast::<PyBytes>()?.as_bytes();
    let width = (bits / 8) as usize;
    if !matches!(bits, 8 | 16 | 32 | 64) || bytes.len() % width != 0 {
        return Err(PyValueError::new_err("invalid signed integer buffer"));
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| match bits {
            8 => i8::from_le_bytes([chunk[0]]) as i64,
            16 => i16::from_le_bytes(chunk.try_into().unwrap()) as i64,
            32 => i32::from_le_bytes(chunk.try_into().unwrap()) as i64,
            _ => i64::from_le_bytes(chunk.try_into().unwrap()),
        })
        .collect())
}

fn unsigned_values(value: &Bound<'_, PyAny>, bits: u32) -> PyResult<Vec<u64>> {
    let bytes = value.cast::<PyBytes>()?.as_bytes();
    let width = (bits / 8) as usize;
    if !matches!(bits, 8 | 16 | 32 | 64) || bytes.len() % width != 0 {
        return Err(PyValueError::new_err("invalid unsigned integer buffer"));
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| match bits {
            8 => u8::from_le_bytes([chunk[0]]) as u64,
            16 => u16::from_le_bytes(chunk.try_into().unwrap()) as u64,
            32 => u32::from_le_bytes(chunk.try_into().unwrap()) as u64,
            _ => u64::from_le_bytes(chunk.try_into().unwrap()),
        })
        .collect())
}

#[pyfunction]
fn _correlate_nd_float(
    input_real: &Bound<'_, PyAny>,
    input_imag: &Bound<'_, PyAny>,
    kernel_real: &Bound<'_, PyAny>,
    kernel_imag: &Bound<'_, PyAny>,
    itemsize: usize,
    options: &Bound<'_, PyTuple>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let input_shape = options.get_item(0)?.extract::<Vec<usize>>()?;
    let kernel_shape = options.get_item(1)?.extract::<Vec<usize>>()?;
    let mode = options.get_item(2)?.extract::<usize>()?;
    let single_precision = options.get_item(3)?.extract::<bool>()?;
    correlate_float(
        &values_from_python(input_real, itemsize)?,
        &values_from_python(input_imag, itemsize)?,
        &values_from_python(kernel_real, itemsize)?,
        &values_from_python(kernel_imag, itemsize)?,
        CorrelationOptions {
            input_shape: &input_shape,
            kernel_shape: &kernel_shape,
            mode,
            single_precision,
        },
    )
}

#[pyfunction]
fn _correlate_nd_signed(
    input: &Bound<'_, PyAny>,
    kernel: &Bound<'_, PyAny>,
    input_shape: Vec<usize>,
    kernel_shape: Vec<usize>,
    mode: usize,
    bits: u32,
) -> PyResult<(Vec<i64>, Vec<usize>)> {
    correlate_integer(
        &signed_values(input, bits)?,
        &signed_values(kernel, bits)?,
        &input_shape,
        &kernel_shape,
        mode,
        bits,
    )
}

#[pyfunction]
fn _correlate_nd_unsigned(
    input: &Bound<'_, PyAny>,
    kernel: &Bound<'_, PyAny>,
    input_shape: Vec<usize>,
    kernel_shape: Vec<usize>,
    mode: usize,
    bits: u32,
) -> PyResult<(Vec<u64>, Vec<usize>)> {
    correlate_integer(
        &unsigned_values(input, bits)?,
        &unsigned_values(kernel, bits)?,
        &input_shape,
        &kernel_shape,
        mode,
        bits,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_correlate_nd_float, module)?)?;
    module.add_function(wrap_pyfunction!(_correlate_nd_signed, module)?)?;
    module.add_function(wrap_pyfunction!(_correlate_nd_unsigned, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CorrelationOptions, correlate_float, correlate_integer};

    #[test]
    fn computes_full_same_and_valid_multidimensional_correlation() {
        let (full, _, shape) = correlate_float(
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0; 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0; 4],
            CorrelationOptions {
                input_shape: &[2, 2],
                kernel_shape: &[2, 2],
                mode: 2,
                single_precision: false,
            },
        )
        .unwrap();
        assert_eq!(shape, [3, 3]);
        assert_eq!(full, [4.0, 11.0, 6.0, 14.0, 30.0, 14.0, 6.0, 11.0, 4.0]);

        let (same, _, shape) = correlate_float(
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0; 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0; 4],
            CorrelationOptions {
                input_shape: &[2, 2],
                kernel_shape: &[2, 2],
                mode: 1,
                single_precision: false,
            },
        )
        .unwrap();
        assert_eq!(shape, [2, 2]);
        assert_eq!(same, [4.0, 11.0, 14.0, 30.0]);

        let (valid, _, shape) = correlate_float(
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0; 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0; 4],
            CorrelationOptions {
                input_shape: &[2, 2],
                kernel_shape: &[2, 2],
                mode: 0,
                single_precision: false,
            },
        )
        .unwrap();
        assert_eq!(shape, [1, 1]);
        assert_eq!(valid, [30.0]);
    }

    #[test]
    fn correlates_complex_values_with_conjugation() {
        let (real, imag, _) = correlate_float(
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[2.0],
            &[3.0],
            CorrelationOptions {
                input_shape: &[2],
                kernel_shape: &[1],
                mode: 1,
                single_precision: false,
            },
        )
        .unwrap();
        assert_eq!(real, [5.0, 3.0]);
        assert_eq!(imag, [-1.0, 2.0]);
    }

    #[test]
    fn handles_unequal_kernel_shapes_across_modes() {
        let full = correlate_float(
            &[1.0, 2.0, 3.0],
            &[0.0; 3],
            &[2.0, 1.0],
            &[0.0; 2],
            CorrelationOptions {
                input_shape: &[3],
                kernel_shape: &[2],
                mode: 2,
                single_precision: false,
            },
        )
        .unwrap();
        assert_eq!(full.0, [1.0, 4.0, 7.0, 6.0]);
        assert_eq!(full.2, [4]);

        let same = correlate_float(
            &[1.0, 2.0, 3.0],
            &[0.0; 3],
            &[2.0, 1.0],
            &[0.0; 2],
            CorrelationOptions {
                input_shape: &[3],
                kernel_shape: &[2],
                mode: 1,
                single_precision: false,
            },
        )
        .unwrap();
        assert_eq!(same.0, [1.0, 4.0, 7.0]);

        let valid = correlate_float(
            &[1.0, 2.0, 3.0],
            &[0.0; 3],
            &[2.0, 1.0],
            &[0.0; 2],
            CorrelationOptions {
                input_shape: &[3],
                kernel_shape: &[2],
                mode: 0,
                single_precision: false,
            },
        )
        .unwrap();
        assert_eq!(valid.0, [4.0, 7.0]);
    }

    #[test]
    fn integer_accumulation_wraps_at_the_input_width() {
        let (result, shape) = correlate_integer(&[120i64, 120], &[2i64], &[2], &[1], 1, 8).unwrap();
        assert_eq!(shape, [2]);
        assert_eq!(result, [-16, -16]);
    }
}
