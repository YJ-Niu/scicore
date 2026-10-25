use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Clone, Copy, Default)]
struct Value {
    re: f64,
    im: f64,
}

type PythonOutput = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

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

fn linear_filter(
    b: &[Value],
    a: &[Value],
    x: &[Value],
    initial_state: &[Value],
    rows: usize,
    samples: usize,
) -> PyResult<(Vec<Value>, Vec<Value>)> {
    if a.is_empty() || b.is_empty() {
        return Err(PyValueError::new_err(
            "filter coefficients must not be empty",
        ));
    }
    if x.len() != rows * samples {
        return Err(PyValueError::new_err("input size does not match its shape"));
    }

    let order = b.len().max(a.len()) - 1;
    if !initial_state.is_empty() && initial_state.len() != rows * order {
        return Err(PyValueError::new_err(
            "initial state size does not match its shape",
        ));
    }
    if a[0].re == 0.0 && a[0].im == 0.0 {
        return Err(PyValueError::new_err(
            "BUG: filter coefficient a[0] == 0 not supported yet",
        ));
    }

    let mut normalized_b = vec![Value::default(); order + 1];
    let mut normalized_a = vec![Value::default(); order + 1];
    for (dst, &coefficient) in normalized_b.iter_mut().zip(b) {
        *dst = coefficient;
    }
    for (dst, &coefficient) in normalized_a.iter_mut().zip(a) {
        *dst = coefficient;
    }
    let reciprocal = Value {
        re: a[0].re / (a[0].re * a[0].re + a[0].im * a[0].im),
        im: -a[0].im / (a[0].re * a[0].re + a[0].im * a[0].im),
    };
    for coefficient in &mut normalized_b {
        *coefficient = coefficient.mul(reciprocal);
    }
    for coefficient in &mut normalized_a {
        *coefficient = coefficient.mul(reciprocal);
    }

    let mut output = vec![Value::default(); x.len()];
    let mut final_state = vec![Value::default(); rows * order];
    for row in 0..rows {
        let state_start = row * order;
        let mut state = if initial_state.is_empty() {
            vec![Value::default(); order]
        } else {
            initial_state[state_start..state_start + order].to_vec()
        };
        for sample in 0..samples {
            let index = row * samples + sample;
            let current = x[index];
            let filtered = normalized_b[0]
                .mul(current)
                .add(state.first().copied().unwrap_or_default());
            output[index] = filtered;
            for tap in 0..order {
                let next_state = state.get(tap + 1).copied().unwrap_or_default();
                state[tap] = normalized_b[tap + 1]
                    .mul(current)
                    .sub(normalized_a[tap + 1].mul(filtered))
                    .add(next_state);
            }
        }
        final_state[state_start..state_start + order].copy_from_slice(&state);
    }
    Ok((output, final_state))
}

#[pyfunction]
fn _linear_filter(
    b: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    initial_state: &Bound<'_, PyAny>,
    rows: usize,
    samples: usize,
) -> PyResult<PythonOutput> {
    let b = values_from_python(b)?;
    let a = values_from_python(a)?;
    let x = values_from_python(x)?;
    let initial_state = values_from_python(initial_state)?;
    let (output, final_state) = linear_filter(&b, &a, &x, &initial_state, rows, samples)?;
    let (output_real, output_imag) = values_to_python(&output);
    let (state_real, state_imag) = values_to_python(&final_state);
    Ok((output_real, output_imag, state_real, state_imag))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_linear_filter, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Value, linear_filter};

    #[test]
    fn filters_real_input_and_returns_final_state() {
        let b = [Value { re: 0.5, im: 0.0 }, Value { re: 0.5, im: 0.0 }];
        let a = [Value { re: 1.0, im: 0.0 }];
        let x = [
            Value { re: 1.0, im: 0.0 },
            Value { re: 2.0, im: 0.0 },
            Value { re: 3.0, im: 0.0 },
        ];
        let initial = [Value { re: 2.0, im: 0.0 }];
        let (output, state) = linear_filter(&b, &a, &x, &initial, 1, 3).unwrap();
        assert_eq!(
            output.iter().map(|v| v.re).collect::<Vec<_>>(),
            [2.5, 1.5, 2.5]
        );
        assert_eq!(state[0].re, 1.5);
    }

    #[test]
    fn normalizes_coefficients_and_filters_complex_values() {
        let b = [Value { re: 2.0, im: 0.0 }];
        let a = [Value { re: 2.0, im: 0.0 }];
        let x = [Value { re: 1.0, im: 2.0 }];
        let (output, _) = linear_filter(&b, &a, &x, &[], 1, 1).unwrap();
        assert_eq!(output[0].re, 1.0);
        assert_eq!(output[0].im, 2.0);
    }
}
