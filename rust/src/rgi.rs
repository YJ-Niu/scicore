use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
}

fn read_complex(value: &Bound<'_, PyAny>) -> PyResult<Vec<Complex>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| {
            Ok(Complex {
                re: item.getattr("real")?.extract()?,
                im: item.getattr("imag")?.extract()?,
            })
        })
        .collect()
}

fn interval(grid: &[f64], value: f64, previous: usize) -> isize {
    if grid.len() == 1 || value.is_nan() {
        return -1;
    }
    let last = grid.len() - 1;
    if value < grid[0] {
        return 0;
    }
    if value > grid[last] {
        return last as isize - 1;
    }
    if value == grid[last] {
        return last as isize - 1;
    }

    let mut low = if value >= grid[previous.min(last - 1)] {
        previous.min(last - 1)
    } else {
        0
    };
    let mut high = if value >= grid[low + 1] {
        last - 1
    } else {
        low
    };
    while low < high {
        let middle = (low + high) / 2;
        if value < grid[middle] {
            high = middle;
        } else if value >= grid[middle + 1] {
            low = middle + 1;
        } else {
            low = middle;
            break;
        }
    }
    low as isize
}

#[pyfunction]
fn _find_indices(
    grid: Vec<Vec<f64>>,
    xi: Vec<f64>,
    npoints: usize,
) -> PyResult<(Vec<isize>, Vec<f64>)> {
    if grid.len().checked_mul(npoints) != Some(xi.len()) {
        return Err(PyValueError::new_err("xi shape does not match data"));
    }
    if grid.iter().any(Vec::is_empty) {
        return Err(PyValueError::new_err("grid axes must not be empty"));
    }

    let mut indices = vec![0isize; xi.len()];
    let mut distances = vec![0.0; xi.len()];
    for (axis, coordinates) in grid.iter().enumerate() {
        let mut previous = 0;
        for point in 0..npoints {
            let offset = axis * npoints + point;
            let value = xi[offset];
            let index = interval(coordinates, value, previous);
            indices[offset] = index;
            if index >= 0 {
                previous = index as usize;
                if value.is_nan() {
                    distances[offset] = f64::NAN;
                } else {
                    let lower = coordinates[previous];
                    distances[offset] = (value - lower) / (coordinates[previous + 1] - lower);
                }
            } else if coordinates.len() > 1 {
                distances[offset] = f64::NAN;
            }
        }
    }
    Ok((indices, distances))
}

#[pyfunction]
fn _evaluate_linear_2d(
    values: &Bound<'_, PyAny>,
    indices: Vec<isize>,
    norm_distances: Vec<f64>,
    rows: usize,
    columns: usize,
    npoints: usize,
) -> PyResult<Vec<(f64, f64)>> {
    let values = read_complex(values)?;
    if rows.checked_mul(columns) != Some(values.len())
        || indices.len() != npoints.saturating_mul(2)
        || norm_distances.len() != indices.len()
    {
        return Err(PyValueError::new_err("array shape does not match data"));
    }
    let nan = Complex {
        re: f64::NAN,
        im: f64::NAN,
    };
    let mut output = vec![nan; npoints];
    for point in 0..npoints {
        let row_index = indices[point];
        let column_index = indices[npoints + point];
        let row_distance = norm_distances[point];
        let column_distance = norm_distances[npoints + point];
        if rows == 1 && columns == 1 {
            output[point] = values[0];
        } else if columns == 1 {
            if row_index >= 0 {
                let row = row_index as usize;
                output[point] = values[row]
                    .scale(1.0 - row_distance)
                    .add(values[row + 1].scale(row_distance));
            }
        } else if rows == 1 {
            if column_index >= 0 {
                let column = column_index as usize;
                output[point] = values[column]
                    .scale(1.0 - column_distance)
                    .add(values[column + 1].scale(column_distance));
            }
        } else if row_index >= 0 && column_index >= 0 {
            let row = row_index as usize;
            let column = column_index as usize;
            output[point] = values[row * columns + column]
                .scale((1.0 - row_distance) * (1.0 - column_distance))
                .add(
                    values[row * columns + column + 1]
                        .scale((1.0 - row_distance) * column_distance),
                )
                .add(
                    values[(row + 1) * columns + column]
                        .scale(row_distance * (1.0 - column_distance)),
                )
                .add(
                    values[(row + 1) * columns + column + 1].scale(row_distance * column_distance),
                );
        }
    }
    Ok(output
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_find_indices, module)?)?;
    module.add_function(wrap_pyfunction!(_evaluate_linear_2d, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Complex, interval};

    #[test]
    fn finds_intervals_with_extrapolation_and_nan() {
        let grid = [0.0, 1.0, 3.0];
        assert_eq!(interval(&grid, -1.0, 0), 0);
        assert_eq!(interval(&grid, 1.0, 0), 1);
        assert_eq!(interval(&grid, 3.0, 0), 1);
        assert_eq!(interval(&grid, 4.0, 0), 1);
        assert_eq!(interval(&grid, f64::NAN, 0), -1);
    }

    #[test]
    fn computes_weighted_complex_values() {
        let left = Complex { re: 1.0, im: 2.0 };
        let right = Complex { re: 3.0, im: 4.0 };
        let result = left.scale(0.25).add(right.scale(0.75));
        assert_eq!(result.re, 2.5);
        assert_eq!(result.im, 3.5);
    }
}
