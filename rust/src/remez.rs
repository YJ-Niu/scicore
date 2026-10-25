use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

const PI: f64 = std::f64::consts::PI;

#[derive(Clone, Copy)]
struct RemezOptions {
    numtaps: usize,
    filter_type: usize,
    fs: f64,
    maxiter: usize,
    grid_density: usize,
}

fn chebyshev_values(frequency: f64, count: usize) -> Vec<f64> {
    let x = (2.0 * PI * frequency).cos();
    let mut values = vec![1.0; count];
    if count > 1 {
        values[1] = x;
        for index in 2..count {
            values[index] = 2.0 * x * values[index - 1] - values[index - 2];
        }
    }
    values
}

fn solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let size = rhs.len();
    for column in 0..size {
        let pivot = (column..size).max_by(|&left, &right| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right][column].abs())
        })?;
        if matrix[pivot][column].abs() < 1e-14 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let diagonal = matrix[column][column];
        for entry in matrix[column].iter_mut().skip(column) {
            *entry /= diagonal;
        }
        rhs[column] /= diagonal;
        let pivot_row = matrix[column].clone();
        for (row_index, row) in matrix.iter_mut().enumerate().take(size) {
            if row_index == column {
                continue;
            }
            let factor = row[column];
            for (entry, pivot_entry) in row.iter_mut().zip(&pivot_row).skip(column) {
                *entry -= factor * pivot_entry;
            }
            rhs[row_index] -= factor * rhs[column];
        }
    }
    Some(rhs)
}

fn solve_extremals(
    frequencies: &[f64],
    desired: &[f64],
    weights: &[f64],
    extremals: &[usize],
    nfcns: usize,
) -> Option<Vec<f64>> {
    let size = nfcns + 1;
    let mut matrix = Vec::with_capacity(size);
    let mut rhs = Vec::with_capacity(size);
    for (index, &point) in extremals.iter().enumerate() {
        let mut row = chebyshev_values(frequencies[point], nfcns);
        row.push(if index % 2 == 0 {
            1.0 / weights[point]
        } else {
            -1.0 / weights[point]
        });
        matrix.push(row);
        rhs.push(desired[point]);
    }
    solve(matrix, rhs)
}

fn alternating_extremals(
    errors: &[f64],
    bands: &[std::ops::Range<usize>],
    count: usize,
) -> Vec<usize> {
    let mut candidates = Vec::new();
    for band in bands {
        if band.is_empty() {
            continue;
        }
        let start = band.start;
        let end = band.end - 1;
        candidates.push(start);
        for index in (start + 1)..end {
            if errors[index].abs() >= errors[index - 1].abs()
                && errors[index].abs() >= errors[index + 1].abs()
            {
                candidates.push(index);
            }
        }
        if end != start {
            candidates.push(end);
        }
    }
    candidates.sort_unstable();
    candidates.dedup();

    let mut alternating = Vec::<usize>::new();
    for candidate in candidates {
        if errors[candidate] == 0.0 {
            continue;
        }
        if let Some(&last) = alternating.last()
            && errors[last].is_sign_positive() == errors[candidate].is_sign_positive()
        {
            if errors[candidate].abs() > errors[last].abs() {
                *alternating.last_mut().unwrap() = candidate;
            }
            continue;
        }
        alternating.push(candidate);
    }
    if alternating.len() <= count {
        return alternating;
    }

    let n = alternating.len();
    let mut scores = vec![vec![f64::NEG_INFINITY; n]; count];
    let mut previous = vec![vec![None; n]; count];
    for index in 0..n {
        scores[0][index] = errors[alternating[index]].abs();
    }
    for selected in 1..count {
        let mut best_previous = None;
        for index in selected..n {
            let prev = index - 1;
            if best_previous
                .is_none_or(|best| scores[selected - 1][prev] > scores[selected - 1][best])
            {
                best_previous = Some(prev);
            }
            if let Some(best) = best_previous {
                scores[selected][index] =
                    scores[selected - 1][best] + errors[alternating[index]].abs();
                previous[selected][index] = Some(best);
            }
        }
    }
    let mut index = (count - 1..n)
        .max_by(|&left, &right| scores[count - 1][left].total_cmp(&scores[count - 1][right]));
    let mut selected = vec![0; count];
    for row in (0..count).rev() {
        let Some(current) = index else {
            return Vec::new();
        };
        selected[row] = alternating[current];
        index = previous[row][current];
    }
    selected
}

fn coefficients_to_taps(alpha: &[f64], numtaps: usize, antisymmetric: bool) -> Vec<f64> {
    let nfcns = alpha.len();
    let odd = numtaps % 2 == 1;
    let mut taps = vec![0.0; numtaps];
    if !antisymmetric {
        if odd {
            for j in 1..nfcns {
                taps[j - 1] = 0.5 * alpha[nfcns - j];
            }
            taps[nfcns - 1] = alpha[0];
        } else {
            taps[0] = 0.25 * alpha[nfcns - 1];
            for j in 2..nfcns {
                taps[j - 1] = 0.25 * (alpha[nfcns - j] + alpha[nfcns + 1 - j]);
            }
            taps[nfcns - 1] = 0.5 * alpha[0] + 0.25 * alpha[1];
        }
        for j in 0..(numtaps / 2) {
            taps[numtaps - 1 - j] = taps[j];
        }
    } else if !odd {
        taps[0] = 0.25 * alpha[nfcns - 1];
        for j in 2..nfcns {
            taps[j - 1] = 0.25 * (alpha[nfcns - j] - alpha[nfcns + 1 - j]);
        }
        taps[nfcns - 1] = 0.5 * alpha[0] - 0.25 * alpha[2];
        for j in 0..nfcns {
            taps[numtaps - 1 - j] = -taps[j];
        }
    } else {
        taps[0] = 0.25 * alpha[nfcns - 1];
        if nfcns > 1 {
            taps[1] = 0.25 * alpha[nfcns - 2];
        }
        for j in 3..nfcns {
            taps[j - 1] = 0.25 * (alpha[nfcns - j] - alpha[nfcns + 2 - j]);
        }
        if nfcns > 2 {
            taps[nfcns - 1] = 0.5 * alpha[0] - 0.25 * alpha[2];
        } else {
            taps[nfcns - 1] = 0.5 * alpha[0];
        }
        for j in 0..nfcns {
            taps[numtaps - 1 - j] = -taps[j];
        }
        taps[nfcns] = 0.0;
    }
    taps
}

fn design_remez(
    band_edges: &[f64],
    desired_by_band: &[f64],
    band_weights: &[f64],
    options: RemezOptions,
) -> PyResult<Vec<f64>> {
    let RemezOptions {
        numtaps,
        filter_type,
        fs,
        maxiter,
        grid_density,
    } = options;
    if !matches!(filter_type, 1..=3) {
        return Err(PyValueError::new_err(
            "The type must be BANDPASS, DIFFERENTIATOR, or HILBERT.",
        ));
    }
    if numtaps < 2 {
        return Err(PyValueError::new_err(
            "The number of taps must be greater than 1.",
        ));
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(PyValueError::new_err("fs must be a finite positive scalar"));
    }
    if grid_density == 0 || maxiter == 0 {
        return Err(PyValueError::new_err(
            "maxiter and grid_density must be positive integers",
        ));
    }
    if band_edges.len() != desired_by_band.len() * 2
        || desired_by_band.len() != band_weights.len()
        || desired_by_band.is_empty()
    {
        return Err(PyValueError::new_err(
            "The inputs desired and weight must have same length.\n  The input bands must have twice this length.",
        ));
    }
    let mut previous = 0.0;
    for edge in band_edges {
        if *edge < previous {
            return Err(PyValueError::new_err(
                "Bands must be monotonic starting at zero.",
            ));
        }
        if *edge * 2.0 > fs {
            return Err(PyValueError::new_err(
                "Band edges should be less than 1/2 the sampling frequency",
            ));
        }
        previous = *edge;
    }
    if band_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(PyValueError::new_err("weights must be finite and positive"));
    }

    let antisymmetric = filter_type == 3;
    let odd = numtaps % 2 == 1;
    let nfcns = numtaps / 2 + usize::from(odd && !antisymmetric);
    let grid_step = 0.5 / (grid_density as f64 * nfcns as f64);
    let mut frequencies = Vec::new();
    let mut desired = Vec::new();
    let mut weights = Vec::new();
    let mut band_ranges = Vec::new();

    for band_index in 0..desired_by_band.len() {
        let low = band_edges[2 * band_index] / fs;
        let high = band_edges[2 * band_index + 1] / fs;
        let mut frequency = low;
        if antisymmetric && frequency < grid_step {
            frequency = grid_step;
        }
        let start = frequencies.len();
        while frequency <= high {
            frequencies.push(frequency);
            desired.push(desired_by_band[band_index]);
            weights.push(band_weights[band_index]);
            frequency += grid_step;
            if frequencies.len() > grid_density.saturating_mul(nfcns).saturating_mul(4) {
                return Err(PyValueError::new_err("dense frequency grid is too large"));
            }
        }
        if frequencies.len() == start {
            frequencies.push(high);
            desired.push(desired_by_band[band_index]);
            weights.push(band_weights[band_index]);
        } else {
            let end = frequencies.len() - 1;
            frequencies[end] = high;
            desired[end] = desired_by_band[band_index];
            weights[end] = band_weights[band_index];
        }
        band_ranges.push(start..frequencies.len());
    }

    if antisymmetric == odd
        && frequencies
            .last()
            .is_some_and(|frequency| *frequency > 0.5 - grid_step)
    {
        frequencies.pop();
        desired.pop();
        weights.pop();
        if let Some(last_band) = band_ranges.last_mut() {
            last_band.end = last_band.end.saturating_sub(1);
        }
    }

    if frequencies.len() < nfcns + 1 {
        return Err(PyValueError::new_err(format!(
            "Band edges are too close together to build the dense frequency grid: it has only {} point(s) but at least {} (one per extremal frequency) are required. Widen the bands, reduce numtaps, or increase grid_density.",
            frequencies.len(),
            nfcns + 1
        )));
    }

    let cosine_transform = !antisymmetric && !odd;
    let sine_transform = antisymmetric;
    for index in 0..frequencies.len() {
        let frequency = frequencies[index];
        if filter_type == 2 {
            if desired_by_band[band_ranges
                .iter()
                .position(|range| range.contains(&index))
                .unwrap()]
                >= 0.0001
            {
                weights[index] /= frequency;
            }
            desired[index] *= frequency;
        }
        let factor = if cosine_transform {
            (PI * frequency).cos()
        } else if sine_transform && odd {
            (2.0 * PI * frequency).sin()
        } else if sine_transform {
            (PI * frequency).sin()
        } else {
            1.0
        };
        if factor.abs() < 1e-14 {
            return Err(PyValueError::new_err(
                "Band edges are too close together to build the dense frequency grid.",
            ));
        }
        desired[index] /= factor;
        weights[index] *= factor.abs();
    }

    let mut extremals = (0..nfcns)
        .map(|index| index * (frequencies.len() - 1) / nfcns)
        .collect::<Vec<_>>();
    extremals.push(frequencies.len() - 1);
    let mut alpha = Vec::new();
    let mut converged = false;
    for _ in 0..maxiter {
        let Some(solution) = solve_extremals(&frequencies, &desired, &weights, &extremals, nfcns)
        else {
            break;
        };
        alpha = solution[..nfcns].to_vec();
        let errors = frequencies
            .iter()
            .enumerate()
            .map(|(index, &frequency)| {
                (chebyshev_values(frequency, nfcns)
                    .iter()
                    .zip(&alpha)
                    .map(|(basis, coefficient)| basis * coefficient)
                    .sum::<f64>()
                    - desired[index])
                    * weights[index]
            })
            .collect::<Vec<_>>();
        let next = alternating_extremals(&errors, &band_ranges, nfcns + 1);
        if next.len() != nfcns + 1 {
            break;
        }
        let old_ripple = extremals
            .iter()
            .map(|&index| errors[index].abs())
            .sum::<f64>()
            / (nfcns + 1) as f64;
        let new_ripple =
            next.iter().map(|&index| errors[index].abs()).sum::<f64>() / (nfcns + 1) as f64;
        let unchanged = next == extremals;
        extremals = next;
        if unchanged || (old_ripple - new_ripple).abs() <= 1e-12 * old_ripple.max(1.0) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(PyValueError::new_err(
            "Failure to converge; try reducing transition band width.",
        ));
    }
    Ok(coefficients_to_taps(&alpha, numtaps, antisymmetric))
}

fn float_list(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| item.extract::<f64>())
        .collect()
}

#[pyfunction]
#[pyo3(signature = (bands, desired, weight, options))]
fn _remez(
    bands: &Bound<'_, PyAny>,
    desired: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    options: &Bound<'_, PyTuple>,
) -> PyResult<Vec<f64>> {
    design_remez(
        &float_list(bands)?,
        &float_list(desired)?,
        &float_list(weight)?,
        RemezOptions {
            numtaps: options.get_item(0)?.extract()?,
            filter_type: options.get_item(1)?.extract()?,
            fs: options.get_item(2)?.extract()?,
            maxiter: options.get_item(3)?.extract()?,
            grid_density: options.get_item(4)?.extract()?,
        },
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_remez, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{RemezOptions, design_remez};

    #[test]
    fn designs_reference_type_two_lowpass_filter() {
        pyo3::Python::initialize();
        let taps = design_remez(
            &[0.0, 0.3, 0.5, 1.0],
            &[1.0, 0.0],
            &[1.0, 1.0],
            RemezOptions {
                numtaps: 12,
                filter_type: 1,
                fs: 2.0,
                maxiter: 25,
                grid_density: 16,
            },
        )
        .unwrap();
        let expected = [
            0.024590270518440,
            -0.041314581814658,
            -0.075943803756711,
            -0.003530911231040,
            0.193140296954975,
            0.373400753484939,
        ];
        for index in 0..expected.len() {
            assert!((taps[index] - expected[index]).abs() < 1e-8);
            assert_eq!(taps[index], taps[taps.len() - 1 - index]);
        }
    }

    #[test]
    fn designs_odd_symmetric_bandpass_filter() {
        pyo3::Python::initialize();
        let taps = design_remez(
            &[0.0, 0.8, 0.9, 1.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            RemezOptions {
                numtaps: 21,
                filter_type: 1,
                fs: 2.0,
                maxiter: 25,
                grid_density: 16,
            },
        )
        .unwrap();
        let expected_first = -0.038976016082299;
        assert!((taps[0] - expected_first).abs() < 3e-8);
        for index in 0..taps.len() {
            assert!((taps[index] - taps[taps.len() - 1 - index]).abs() < 1e-14);
        }
    }

    #[test]
    fn designs_antisymmetric_hilbert_filter() {
        pyo3::Python::initialize();
        let taps = design_remez(
            &[0.1, 0.4],
            &[1.0],
            &[1.0],
            RemezOptions {
                numtaps: 11,
                filter_type: 3,
                fs: 1.0,
                maxiter: 25,
                grid_density: 16,
            },
        )
        .unwrap();
        for index in 0..taps.len() / 2 {
            assert!((taps[index] + taps[taps.len() - 1 - index]).abs() < 1e-14);
        }
        assert_eq!(taps[5], 0.0);
    }
}
