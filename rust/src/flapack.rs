//! Dense double-precision LAPACK kernels for `scipy.linalg._flapack`.
//!
//! SciPy's wrappers force every input to Fortran order before handing it to
//! LAPACK.  rsnumpy is row-major, so each kernel transposes its input into
//! column-major on the way in and back on the way out.  That costs `O(n^2)`
//! against LAPACK's `O(n^3)`, which is negligible and keeps the index
//! arithmetic obvious.
//!
//! Porting order follows what the rest of `scipy.linalg` actually needs first:
//! the LU family (`getrf`, `getrs`, `getri`) underpins `lu`, `solve` and `inv`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn dgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn dgetrs_(
        trans: *const u8,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        ipiv: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn dgetri_(
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

// ============================================================================
// Pure-Rust LU backend (non-macOS platforms)
//
// Re-implements the Netlib reference algorithms DGETRF / DGETRS / DGETRI so
// no system LAPACK is required.  On macOS the Accelerate symbols above are
// used instead; the Fortran-signature shims below keep every call site
// identical across platforms.
// ============================================================================

/// DGETRF: LU factorisation with partial pivoting, `P A = L U`.
///
/// `a` is column-major (m, n) with leading dimension `lda` (>= m.max(1)).
/// On return the strictly lower triangle holds the multipliers of L (unit
/// diagonal implicit) and the upper triangle holds U.  `ipiv` (length
/// m.min(n)) holds 1-based pivot rows; `info` is the 1-based index of the
/// first exactly-zero pivot, or 0.
#[allow(dead_code)]
fn dgetrf_rust(m: usize, n: usize, a: &mut [f64], lda: usize, ipiv: &mut [i32], info: &mut i32) {
    *info = 0;
    let minmn = m.min(n);
    for j in 0..minmn {
        // Partial pivoting: first occurrence of the largest |a(i, j)|, i >= j.
        let mut pivot_row = j;
        let mut best = a[j * lda + j].abs();
        for i in (j + 1)..m {
            let v = a[j * lda + i].abs();
            if v > best {
                best = v;
                pivot_row = i;
            }
        }
        ipiv[j] = (pivot_row + 1) as i32;
        if pivot_row != j {
            // Swap the ENTIRE rows (all columns 0..n), not just the active
            // columns j..n: the multipliers already stored in columns < j must
            // move with their rows so the final buffer satisfies P*A = L*U.
            for col in 0..n {
                a.swap(col * lda + pivot_row, col * lda + j);
            }
        }
        let pivot = a[j * lda + j];
        if pivot != 0.0 {
            for i in (j + 1)..m {
                let factor = a[j * lda + i] / pivot;
                a[j * lda + i] = factor;
                for col in (j + 1)..n {
                    a[col * lda + i] -= factor * a[col * lda + j];
                }
            }
        } else if *info == 0 {
            *info = (j + 1) as i32;
        }
    }
}

/// DGETRS: solve `op(A) X = B` from a DGETRF factorisation.
///
/// `lu` is column-major (n, n) with leading dimension `lda` (>= n.max(1));
/// `ipiv` is 1-based; `b` is column-major (n, nrhs) with leading dimension
/// `ldb` (>= n.max(1)) and is replaced in place by the solution.  `trans` is
/// `b'N'`/`b'C'` (no transpose) or `b'T'`.  `info` is the 1-based index of
/// the first zero diagonal of U, or 0.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn dgetrs_rust(
    trans: u8,
    n: usize,
    nrhs: usize,
    lu: &[f64],
    lda: usize,
    ipiv: &[i32],
    b: &mut [f64],
    ldb: usize,
    info: &mut i32,
) {
    *info = 0;
    if trans == b'T' {
        // A = P^T L U, so A^T = U^T L^T P.  Solve U^T Y = B, then L^T X = Y,
        // then undo the pivoting.  `lu` is column-major (A(i, k) sits at
        // lu[k*lda + i]); `b` is column-major (B(i, j) sits at b[j*ldb + i]).
        // 1) Solve U^T Y = B.  U^T is lower triangular; U^T(i, k) = U(k, i)
        //    = lu[i*lda + k] for k <= i.
        for i in 0..n {
            for j in 0..nrhs {
                let mut val = b[j * ldb + i];
                for k in 0..i {
                    val -= lu[i * lda + k] * b[j * ldb + k];
                }
                let d = lu[i * lda + i];
                b[j * ldb + i] = if d == 0.0 {
                    if *info == 0 {
                        *info = (i + 1) as i32;
                    }
                    val
                } else {
                    val / d
                };
            }
        }
        // 2) Solve L^T X = Y.  L^T is upper triangular with unit diagonal;
        //    L^T(i, k) = L(k, i) = lu[i*lda + k] for k > i.
        for i in (0..n).rev() {
            for j in 0..nrhs {
                let mut val = b[j * ldb + i];
                for k in (i + 1)..n {
                    val -= lu[i * lda + k] * b[j * ldb + k];
                }
                b[j * ldb + i] = val;
            }
        }
        // 3) Undo the pivoting (reverse of the application order).
        for i in (0..n).rev() {
            let p = (ipiv[i] - 1) as usize;
            if p != i {
                for j in 0..nrhs {
                    b.swap(j * ldb + i, j * ldb + p);
                }
            }
        }
        return;
    }
    // trans == b'N' (or b'C'): A X = B with A = P^T (L U), i.e. L U X = P B.
    // 1) Apply the row permutations to B (forward order).
    #[allow(clippy::needless_range_loop)]
    for (i, item) in ipiv.iter().enumerate().take(n) {
        let p = (item - 1) as usize;
        if p != i {
            for j in 0..nrhs {
                b.swap(j * ldb + i, j * ldb + p);
            }
        }
    }
    // 2) Forward-substitute L (unit lower): L(i, k) = lu[k*lda + i], k < i.
    for i in 0..n {
        for j in 0..nrhs {
            let mut val = b[j * ldb + i];
            for k in 0..i {
                val -= lu[k * lda + i] * b[j * ldb + k];
            }
            b[j * ldb + i] = val;
        }
    }
    // 3) Back-substitute U: U(i, k) = lu[k*lda + i], k > i.
    for i in (0..n).rev() {
        for j in 0..nrhs {
            let mut val = b[j * ldb + i];
            for k in (i + 1)..n {
                val -= lu[k * lda + i] * b[j * ldb + k];
            }
            let d = lu[i * lda + i];
            b[j * ldb + i] = if d == 0.0 {
                if *info == 0 {
                    *info = (i + 1) as i32;
                }
                val
            } else {
                val / d
            };
        }
    }
}

/// DGETRI: inverse of A from a DGETRF factorisation.
///
/// `lu` (column-major (n, n), leading dimension `lda`) is replaced by
/// A^(-1).  `work` must hold at least n*n elements (the reference's optimal
/// size).  When `lwork < 0` this is the workspace query: `work[0]` receives
/// max(1, n*n) and no other argument is read or written.
#[allow(dead_code)]
fn dgetri_rust(
    n: usize,
    lu: &mut [f64],
    lda: usize,
    ipiv: &[i32],
    work: &mut [f64],
    lwork: i32,
    info: &mut i32,
) {
    if lwork < 0 {
        *info = 0;
        if !work.is_empty() {
            work[0] = ((n * n).max(1)) as f64;
        }
        return;
    }
    // Same reduction as the reference: set the right-hand side to the
    // identity and solve A X = I with the LU factors (n right-hand sides);
    // X = A^(-1) is written into `work`, then copied back into `lu`.
    for i in 0..n {
        for j in 0..n {
            work[i + j * n] = if i == j { 1.0 } else { 0.0 };
        }
    }
    dgetrs_rust(b'N', n, n, lu, lda, ipiv, work, n, info);
    for i in 0..n {
        for j in 0..n {
            lu[j * lda + i] = work[i + j * n];
        }
    }
}

// Fortran-signature shims so the call sites above are identical on every
// platform; on macOS these names bind to the Accelerate symbols instead.
#[cfg(not(target_os = "macos"))]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dgetrf_(
    m: *const i32,
    n: *const i32,
    a: *mut f64,
    lda: *const i32,
    ipiv: *mut i32,
    info: *mut i32,
) {
    let (m, n, lda) = (*m as usize, *n as usize, *lda as usize);
    unsafe {
        dgetrf_rust(
            m,
            n,
            std::slice::from_raw_parts_mut(a, m * n),
            lda,
            std::slice::from_raw_parts_mut(ipiv, m.min(n)),
            &mut *info,
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dgetrs_(
    trans: *const u8,
    n: *const i32,
    nrhs: *const i32,
    a: *const f64,
    lda: *const i32,
    ipiv: *const i32,
    b: *mut f64,
    ldb: *const i32,
    info: *mut i32,
) {
    let (n, nrhs) = (*n as usize, *nrhs as usize);
    unsafe {
        dgetrs_rust(
            *trans,
            n,
            nrhs,
            std::slice::from_raw_parts(a, n * n),
            *lda as usize,
            std::slice::from_raw_parts(ipiv, n),
            std::slice::from_raw_parts_mut(b, n * nrhs),
            *ldb as usize,
            &mut *info,
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dgetri_(
    n: *const i32,
    a: *mut f64,
    lda: *const i32,
    ipiv: *const i32,
    work: *mut f64,
    lwork: *const i32,
    info: *mut i32,
) {
    let n = *n as usize;
    if *lwork < 0 {
        // Workspace query: do not touch `a` / `ipiv` (they may be null).
        unsafe {
            *work = ((n * n).max(1)) as f64;
        }
        *info = 0;
        return;
    }
    unsafe {
        dgetri_rust(
            n,
            std::slice::from_raw_parts_mut(a, n * n),
            *lda as usize,
            std::slice::from_raw_parts(ipiv, n),
            std::slice::from_raw_parts_mut(work, n * n),
            *lwork,
            &mut *info,
        );
    }
}

fn to_f64(buf: &[u8]) -> PyResult<Vec<f64>> {
    if !buf.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(
            "float64 buffer length is not a multiple of 8",
        ));
    }
    Ok(buf
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(c);
            f64::from_le_bytes(arr)
        })
        .collect())
}

fn to_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Row-major `(rows, cols)` -> column-major buffer of the same matrix, i.e. the
/// transposition both directions need.
fn transpose<T: Copy>(src: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            out.push(src[row * cols + col]);
        }
    }
    out
}

fn leading_dim(rows: usize) -> i32 {
    i32::try_from(rows.max(1)).unwrap_or(i32::MAX)
}

/// getrf(a_bytes, m, n) -> (lu_bytes, pivots, info)
///
/// LU factorisation `A = P L U` with partial pivoting.  `pivots` is returned
/// **0-based**, matching SciPy's convention; LAPACK itself is 1-based.
#[pyfunction]
fn getrf_f64<'py>(
    py: Python<'py>,
    a_bytes: &Bound<'_, PyBytes>,
    m: usize,
    n: usize,
) -> PyResult<(Bound<'py, PyBytes>, Vec<i32>, i32)> {
    let row_major = to_f64(a_bytes.as_bytes())?;
    if row_major.len() != m * n {
        return Err(PyValueError::new_err(
            "buffer size does not match the declared shape",
        ));
    }

    let mut a_cm = transpose(&row_major, m, n);
    let lda = leading_dim(m);
    let npiv = m.min(n);
    let mut pivots = vec![0i32; npiv];
    let mut info: i32 = 0;

    unsafe {
        dgetrf_(
            &(m as i32),
            &(n as i32),
            a_cm.as_mut_ptr(),
            &lda,
            pivots.as_mut_ptr(),
            &mut info,
        );
    }

    for p in pivots.iter_mut() {
        *p -= 1; // SciPy hands 0-based permutation indices back to Python
    }

    let lu = PyBytes::new(py, &to_bytes(&transpose(&a_cm, n, m)));
    Ok((lu, pivots, info))
}

/// getrs(lu_bytes, n, pivots, b_bytes, nrhs, trans) -> (x_bytes, info)
///
/// Solves `op(A) x = b` from the `getrf` factorisation; `trans` is SciPy's
/// 0 = N / 1 = T / 2 = C.
#[pyfunction]
fn getrs_f64<'py>(
    py: Python<'py>,
    lu_bytes: &Bound<'_, PyBytes>,
    n: usize,
    pivots: Vec<i32>,
    b_bytes: &Bound<'_, PyBytes>,
    nrhs: usize,
    trans: i32,
) -> PyResult<(Bound<'py, PyBytes>, i32)> {
    let b_row = to_f64(b_bytes.as_bytes())?;
    if b_row.len() != n * nrhs {
        return Err(PyValueError::new_err(
            "b shape is incompatible with the factorisation",
        ));
    }

    let trans_char: u8 = match trans {
        0 => b'N',
        1 => b'T',
        2 => b'C',
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid trans value {other}"
            )));
        }
    };

    let lu_row = to_f64(lu_bytes.as_bytes())?;
    if lu_row.len() != n * n {
        return Err(PyValueError::new_err("lu must be square"));
    }
    if pivots.len() != n {
        return Err(PyValueError::new_err("pivots length must match n"));
    }

    // The factorisation is used as-is for `trans == 0`.  SciPy's caller always
    // supplies the LU of A itself, so the row-major -> column-major transpose is
    // all that differs from the Fortran layout LAPACK wants.
    let lu_cm = transpose(&lu_row, n, n);
    let mut b_cm = transpose(&b_row, n, nrhs);
    let shifted: Vec<i32> = pivots.iter().map(|p| p + 1).collect();

    let lda = leading_dim(n);
    let ldb = leading_dim(n);
    let mut info: i32 = 0;

    unsafe {
        dgetrs_(
            &trans_char,
            &(n as i32),
            &(nrhs as i32),
            lu_cm.as_ptr(),
            &lda,
            shifted.as_ptr(),
            b_cm.as_mut_ptr(),
            &ldb,
            &mut info,
        );
    }

    let x = PyBytes::new(py, &to_bytes(&transpose(&b_cm, nrhs, n)));
    Ok((x, info))
}

/// getri(lu_bytes, n, pivots, lwork) -> (inv_bytes, info)
#[pyfunction]
fn getri_f64<'py>(
    py: Python<'py>,
    lu_bytes: &Bound<'_, PyBytes>,
    n: usize,
    pivots: Vec<i32>,
    lwork: usize,
) -> PyResult<(Bound<'py, PyBytes>, i32)> {
    let lu_row = to_f64(lu_bytes.as_bytes())?;
    if lu_row.len() != n * n {
        return Err(PyValueError::new_err("lu must be square"));
    }
    if pivots.len() != n {
        return Err(PyValueError::new_err("pivots length must match n"));
    }
    // The inverse needs an n-by-n workspace; the caller's hint (SciPy asks
    // for 3*n by default) may be smaller, so allocate the maximum of the two.
    let lwork_eff = lwork.max(n * n).max(1);
    let lwork_i = i32::try_from(lwork_eff)
        .map_err(|_| PyValueError::new_err("lwork does not fit in a 32-bit BLAS integer"))?;

    let mut lu_cm = transpose(&lu_row, n, n);
    let shifted: Vec<i32> = pivots.iter().map(|p| p + 1).collect();
    let mut work = vec![0.0f64; lwork_eff];
    let lda = leading_dim(n);
    let mut info: i32 = 0;

    unsafe {
        dgetri_(
            &(n as i32),
            lu_cm.as_mut_ptr(),
            &lda,
            shifted.as_ptr(),
            work.as_mut_ptr(),
            &lwork_i,
            &mut info,
        );
    }

    let inv = PyBytes::new(py, &to_bytes(&transpose(&lu_cm, n, n)));
    Ok((inv, info))
}

/// getri_lwork(n) -> (optimal_lwork, info): the workspace size query.
#[pyfunction]
fn getri_lwork_f64(n: usize) -> PyResult<(f64, i32)> {
    let mut work = 0.0f64;
    let mut info: i32 = 0;
    let n_i = n as i32;
    let lda = leading_dim(n);
    unsafe {
        dgetri_(
            &n_i,
            std::ptr::null_mut(),
            &lda,
            std::ptr::null(),
            &mut work,
            &-1,
            &mut info,
        );
    }
    Ok((work, info))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(getrf_f64, m)?)?;
    m.add_function(wrap_pyfunction!(getrs_f64, m)?)?;
    m.add_function(wrap_pyfunction!(getri_f64, m)?)?;
    m.add_function(wrap_pyfunction!(getri_lwork_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::testutil::max_abs_diff;
    use crate::testutil::{Rng, max_abs};

    fn random_matrix(rng: &mut Rng, rows: usize, cols: usize) -> Vec<f64> {
        (0..rows * cols).map(|_| rng.f64()).collect()
    }

    fn factorize(rng: &mut Rng, n: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        let a_row = random_matrix(rng, n, n);
        let a_cm = transpose(&a_row, n, n);
        let mut lu = a_cm.clone();
        let mut piv = vec![0i32; n];
        let mut info = -1;
        dgetrf_rust(n, n, &mut lu, n, &mut piv, &mut info);
        assert_eq!(info, 0, "random {n}x{n} matrix should be nonsingular");
        (a_row, lu, piv)
    }

    // ------------------------------------------------------------------
    // Property tests (run on every platform, CI included)
    // ------------------------------------------------------------------

    /// P A = L U: apply the forward row swaps to A, compare with L * U.
    #[test]
    fn getrf_rust_reconstructs_a() {
        let mut rng = Rng::new(0xABCD);
        for &(m, n) in &[(3usize, 3), (5, 5), (2, 7), (7, 2), (1, 1)] {
            let a_row = random_matrix(&mut rng, m, n);
            let a_cm = transpose(&a_row, m, n);
            let lda = m.max(1);
            let mut lu = a_cm.clone();
            let mut piv = vec![0i32; m.min(n)];
            let mut info = 0;
            dgetrf_rust(m, n, &mut lu, lda, &mut piv, &mut info);
            // P A: apply the pivoting swaps in forward order.
            let mut pa = a_cm.clone();
            for (i, item) in piv.iter().enumerate().take(m.min(n)) {
                let p = (item - 1) as usize;
                if p != i {
                    for col in 0..n {
                        pa.swap(col * lda + p, col * lda + i);
                    }
                }
            }
            let scale = max_abs(&lu).max(1.0);
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for t in 0..=i.min(j) {
                        let l = if t == i { 1.0 } else { lu[t * lda + i] };
                        s += l * lu[j * lda + t];
                    }
                    assert!(
                        (s - pa[j * lda + i]).abs() <= 1e-12 * scale * (i.min(j) + 1) as f64,
                        "m={m} n={n} ({i}, {j}): {s} vs {}",
                        pa[j * lda + i]
                    );
                }
            }
        }
    }

    #[test]
    fn getrf_rust_singular_info() {
        // A = [[1, 2, 3], [0, 0, 1], [0, 0, 1]]: exactly-zero pivot in
        // column 1 -> info == 2 (1-based), pivots unchanged.
        let a_row = vec![1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let a_cm = transpose(&a_row, 3, 3);
        let mut lu = a_cm.clone();
        let mut piv = vec![0i32; 3];
        let mut info = 0;
        dgetrf_rust(3, 3, &mut lu, 3, &mut piv, &mut info);
        assert_eq!(info, 2);
        assert_eq!(piv, vec![1, 2, 3]);
        // getrs reports the same zero diagonal.
        let mut b = vec![1.0, 2.0, 3.0];
        let mut info2 = 0;
        dgetrs_rust(b'N', 3, 1, &lu, 3, &piv, &mut b, 3, &mut info2);
        assert_eq!(info2, 2);
    }

    #[test]
    fn getri_rust_lwork_query() {
        let mut work = vec![9.0f64];
        let mut info = -1;
        dgetri_rust(5, &mut [], 5, &[], &mut work, -1, &mut info);
        assert_eq!(work[0], 25.0);
        assert_eq!(info, 0);
        dgetri_rust(0, &mut [], 1, &[], &mut work, -1, &mut info);
        assert_eq!(work[0], 1.0);
        assert_eq!(info, 0);
    }

    #[test]
    fn getrs_rust_solves() {
        let mut rng = Rng::new(0x1234);
        for &n in &[1usize, 2, 3, 5, 8] {
            for &nrhs in &[1usize, 2, 3] {
                for &trans in &[0i32, 1, 2] {
                    let (a_row, lu, piv) = factorize(&mut rng, n);
                    let b_row = random_matrix(&mut rng, n, nrhs);
                    let b_cm = transpose(&b_row, n, nrhs);
                    let trans_char = match trans {
                        0 => b'N',
                        _ => b'T',
                    };
                    let mut x = b_cm.clone();
                    let mut info = 0;
                    dgetrs_rust(trans_char, n, nrhs, &lu, n, &piv, &mut x, n, &mut info);
                    assert_eq!(info, 0);
                    // Relative residual ||op(A) X - B|| / (||op(A) X|| + ||B||).
                    let mut res_max: f64 = 0.0;
                    let mut ax_max: f64 = 0.0;
                    for j in 0..nrhs {
                        for i in 0..n {
                            let mut s = 0.0;
                            for k in 0..n {
                                let aik = if trans == 0 {
                                    a_row[i * n + k]
                                } else {
                                    a_row[k * n + i]
                                };
                                s += aik * x[j * n + k];
                            }
                            res_max = res_max.max((s - b_row[i * nrhs + j]).abs());
                            ax_max = ax_max.max(s.abs());
                        }
                    }
                    let cond = (ax_max + max_abs(&b_row)).max(1.0);
                    assert!(
                        res_max <= 1e-10 * cond,
                        "n={n} nrhs={nrhs} trans={trans}: residual {res_max} / {cond}"
                    );
                }
            }
        }
    }

    #[test]
    fn getri_rust_inverts() {
        let mut rng = Rng::new(0x9999);
        for &n in &[1usize, 2, 3, 5, 8] {
            let (a_row, lu, piv) = factorize(&mut rng, n);
            let mut inv = lu.clone();
            let mut work = vec![0.0f64; n * n];
            let mut info = 0;
            dgetri_rust(n, &mut inv, n, &piv, &mut work, (n * n) as i32, &mut info);
            assert_eq!(info, 0);
            // ||A * Ainv - I|| (Ainv is column-major in `inv`).
            let mut err: f64 = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let mut s: f64 = 0.0;
                    for k in 0..n {
                        s += a_row[i * n + k] * inv[j * n + k];
                    }
                    err = err.max((s - f64::from(i == j)).abs());
                }
            }
            let scale = max_abs(&inv).max(1.0);
            assert!(
                err <= 1e-9 * scale * n as f64,
                "n={n}: ||A Ainv - I|| = {err}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Cross-validation against Accelerate (macOS only)
    // ------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn getrf_rust_matches_accelerate() {
        let mut rng = Rng::new(0xD00D);
        for &(m, n) in &[(3usize, 3), (5, 5), (2, 7), (7, 2), (1, 1), (4, 4)] {
            let a_row = random_matrix(&mut rng, m, n);
            let a_cm = transpose(&a_row, m, n);
            let lda = m.max(1);
            let mut lu_r = a_cm.clone();
            let mut lu_a = a_cm.clone();
            let mut piv_r = vec![0i32; m.min(n)];
            let mut piv_a = vec![0i32; m.min(n)];
            let mut info_r = -1;
            let mut info_a = -1;
            dgetrf_rust(m, n, &mut lu_r, lda, &mut piv_r, &mut info_r);
            unsafe {
                dgetrf_(
                    &(m as i32),
                    &(n as i32),
                    lu_a.as_mut_ptr(),
                    &(lda as i32),
                    piv_a.as_mut_ptr(),
                    &mut info_a,
                );
            }
            assert_eq!(info_r, info_a, "m={m} n={n}: info {info_r} vs {info_a}");
            assert_eq!(piv_r, piv_a, "m={m} n={n}: pivots differ");
            let scale = max_abs(&lu_a).max(1.0);
            assert!(
                max_abs_diff(&lu_r, &lu_a) <= 1e-12 * scale,
                "m={m} n={n}: LU diff {}",
                max_abs_diff(&lu_r, &lu_a)
            );
        }
        // Singular: zero second column -> info == 2 on both backends.
        let a_row = vec![1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let a_cm = transpose(&a_row, 3, 3);
        let mut lu_r = a_cm.clone();
        let mut lu_a = a_cm.clone();
        let mut piv_r = vec![0i32; 3];
        let mut piv_a = vec![0i32; 3];
        let mut info_r = -1;
        let mut info_a = -1;
        dgetrf_rust(3, 3, &mut lu_r, 3, &mut piv_r, &mut info_r);
        #[allow(clippy::unnecessary_cast)]
        unsafe {
            dgetrf_(
                &(3 as i32),
                &(3 as i32),
                lu_a.as_mut_ptr(),
                &(3 as i32),
                piv_a.as_mut_ptr(),
                &mut info_a,
            );
        }
        assert_eq!(info_r, 2);
        assert_eq!(info_a, 2);
        assert_eq!(piv_r, piv_a);
        assert!(max_abs_diff(&lu_r, &lu_a) <= 1e-12);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn getrs_rust_matches_accelerate() {
        let mut rng = Rng::new(0xCAFE);
        for &n in &[1usize, 2, 3, 5] {
            for &nrhs in &[1usize, 2] {
                for &trans in &[0i32, 1] {
                    let (_a_row, lu, piv) = factorize(&mut rng, n);
                    let b_row = random_matrix(&mut rng, n, nrhs);
                    let b_cm0 = transpose(&b_row, n, nrhs);
                    let mut b_r = b_cm0.clone();
                    let mut b_a = b_cm0.clone();
                    let trans_char = if trans == 0 { b'N' } else { b'T' };
                    let mut info = 0;
                    dgetrs_rust(trans_char, n, nrhs, &lu, n, &piv, &mut b_r, n, &mut info);
                    assert_eq!(info, 0);
                    info = -1;
                    unsafe {
                        dgetrs_(
                            &trans_char,
                            &(n as i32),
                            &(nrhs as i32),
                            lu.as_ptr(),
                            &(n as i32),
                            piv.as_ptr(),
                            b_a.as_mut_ptr(),
                            &(n as i32),
                            &mut info,
                        );
                    }
                    let scale = max_abs(&b_a).max(1.0);
                    assert!(
                        max_abs_diff(&b_r, &b_a) <= 1e-10 * scale,
                        "n={n} nrhs={nrhs} trans={trans}: diff {}",
                        max_abs_diff(&b_r, &b_a)
                    );
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn getri_rust_matches_accelerate() {
        let mut rng = Rng::new(0x7777);
        for &n in &[1usize, 2, 3, 5] {
            let (_a_row, lu, piv) = factorize(&mut rng, n);
            let lu_r0 = lu.clone();
            let mut inv_r = lu_r0.clone();
            let mut inv_a = lu_r0.clone();
            let mut work_r = vec![0.0f64; n * n];
            let mut work_a = vec![0.0f64; n * n];
            let mut info = 0;
            dgetri_rust(
                n,
                &mut inv_r,
                n,
                &piv,
                &mut work_r,
                (n * n) as i32,
                &mut info,
            );
            assert_eq!(info, 0);
            info = -1;
            unsafe {
                dgetri_(
                    &(n as i32),
                    inv_a.as_mut_ptr(),
                    &(n as i32),
                    piv.as_ptr(),
                    work_a.as_mut_ptr(),
                    &((n * n) as i32),
                    &mut info,
                );
            }
            let scale = max_abs(&inv_a).max(1.0);
            assert!(
                max_abs_diff(&inv_r, &inv_a) <= 1e-10 * scale,
                "n={n}: diff {}",
                max_abs_diff(&inv_r, &inv_a)
            );
        }
    }
}
