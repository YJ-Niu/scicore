"""Stub replacement for the deleted C ``_sparsetools`` extension.

Provides the sparse-matrix low-level routines used by ``scipy.sparse``,
implemented on top of :mod:`rsnumpy`.  This is a stub: performance-critical
routines should later be ported to Rust under ``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np


def _as_rs(arr):
    return np.asarray(arr)


# ----- CSR -----

def csr_matvec(indptr, indices, data, x, y):
    """y = A x for CSR matrix."""
    m = len(indptr) - 1
    n = len(x)
    y = np.zeros(m)
    for i in range(m):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * x[indices[j]]
        y[i] = s
    return y


def csr_matvecs(indptr, indices, data, x, y):
    """Y = A X for CSR matrix (X is 2-D)."""
    m = len(indptr) - 1
    n_cols = x.shape[1] if x.ndim == 2 else 1
    for k in range(n_cols):
        col = x[:, k] if x.ndim == 2 else x
        out = csr_matvec(indptr, indices, data, col, None)
        if x.ndim == 2:
            y[:, k] = out
        else:
            y[:] = out
    return y


def csr_todense(M, N, indptr, indices, data, out):
    """Scipy-compatible ``csr_todense``: scatter CSR rows into a 2-D buffer.

    ``out`` is a 2-D array (C-contiguous, possibly a transposed view);
    logical entry ``(i, j)`` receives ``data[j]`` for each stored column
    ``j`` of row ``i``.
    """
    out[:] = 0.0
    for i in range(M):
        for j in range(indptr[i], indptr[i + 1]):
            out[i, indices[j]] = data[j]
    return out


def csr_count_blocks(indptr, indices, blockdim):
    # stub: return number of blocks
    return 0


def csr_tocsc(indptr, indices, data, shape):
    # Convert CSR to CSC: return new indptr, indices, data
    m, n = shape
    # Count entries per column
    col_counts = [0] * n
    for j_idx in indices:
        col_counts[j_idx] += 1
    csc_indptr = [0] * (n + 1)
    for j in range(n):
        csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
    csc_indices = [0] * len(indices)
    csc_data = [0.0] * len(data)
    col_pos = list(csc_indptr[:n])
    for i in range(m):
        for jj in range(indptr[i], indptr[i + 1]):
            j = indices[jj]
            pos = col_pos[j]
            csc_indices[pos] = i
            csc_data[pos] = data[jj]
            col_pos[j] += 1
    return np.array(csc_indptr), np.array(csc_indices), np.array(csc_data)


def csr_tobsr(indptr, indices, data, shape, blockdim):
    raise NotImplementedError("csr_tobsr not yet ported to Rust/rsnumpy")


def csr_hstack(blocks, col_starts, m, n):
    raise NotImplementedError("csr_hstack not yet ported to Rust/rsnumpy")


def get_csr_submatrix(indptr, indices, data, shape, row_start, row_end, col_start, col_end):
    raise NotImplementedError("get_csr_submatrix not yet ported")


def csr_sample_offsets(indptr, indices, data, shape, sample_size, seed):
    raise NotImplementedError("csr_sample_offsets not yet ported")


# ----- BSR -----

def bsr_matvec(indptr, indices, data, x, y, blockdim):
    raise NotImplementedError("bsr_matvec not yet ported")


def bsr_matvecs(indptr, indices, data, x, y, blockdim):
    raise NotImplementedError("bsr_matvecs not yet ported")


def bsr_diagonal(k, rows, cols, R, C, indptr, indices, data, out):
    raise NotImplementedError("bsr_diagonal not yet ported")


def csr_eliminate_zeros(M, N, indptr, indices, data):
    raise NotImplementedError("csr_eliminate_zeros not yet ported")


# ----- COO -----

def coo_tocsr(M, N, nnz, major, minor, data, indptr, indices, out_data):
    """Scipy-compatible ``coo_tocsr``: fill pre-allocated CSR buffers.

    Pure rsnumpy replacement for the C ``coo_tocsr``.  Entries are
    scattered in input order (stable within each row) and duplicate
    coordinates are preserved, mirroring the C implementation.
    """
    counts = np.zeros(M, dtype=np.intp)
    counts[major.astype(np.intp)] += 1
    indptr[0] = 0
    indptr[1:] = np.cumsum(counts)
    pos = indptr[0:M].copy()
    for k in range(nnz):
        r = int(major[k])
        p = int(pos[r])
        pos[r] = p + 1
        indices[p] = int(minor[k])
        out_data[p] = data[k]


def coo_todense(M, N, nnz, row, col, data, out, fortran):
    """Scatter COO entries into a flat dense buffer (2-D).

    Pure rsnumpy replacement for the C ``coo_todense``.  ``out`` is the
    raveled dense array (length ``M*N``); the flat index is computed for the
    C or Fortran layout selected by ``fortran``.
    """
    if fortran:
        idx = col * M + row
    else:
        idx = row * N + col
    out[idx.astype(np.intp)] += data


def coo_todense_nd(strides, nnz, ndim, coords, data, out, fortran):
    """N-dimensional analogue of :func:`coo_todense`.

    ``coords`` is the concatenation of the per-dimension index arrays
    (``ndim * nnz`` long); the flat index is each dimension's index dotted
    with ``strides``.
    """
    idx = np.zeros(nnz, dtype=np.intp)
    for d in range(ndim):
        s = d * nnz
        idx += coords[s:s + nnz].astype(np.intp) * strides[d]
    out[idx] += data


# ----- DIA -----

def dia_matmat(offsets, data, shape, B):
    raise NotImplementedError("dia_matmat not yet ported")


def dia_matvec(offsets, data, shape, x, y):
    raise NotImplementedError("dia_matvec not yet ported")


def dia_matvecs(offsets, data, shape, x, y):
    raise NotImplementedError("dia_matvecs not yet ported")


def dia_tocsr(offsets, data, shape):
    raise NotImplementedError("dia_tocsr not yet ported")


# ----- CSC -----

def expandptr(indptr, indices, data, shape):
    raise NotImplementedError("expandptr not yet ported")


# ----- CSR count blocks -----

def csr_count_blocks(indptr, indices, blockdim):
    raise NotImplementedError("csr_count_blocks not yet ported")


def csr_sample_values(M, N, indptr, indices, data, nnz, row, col, val):
    """Sample values from a CSR matrix at given (row, col) positions."""
    import rsnumpy as _np
    indptr = _np.asarray(indptr)
    indices = _np.asarray(indices)
    data = _np.asarray(data)
    row = _np.asarray(row)
    col = _np.asarray(col)
    for k in range(len(row)):
        r = int(row[k])
        c = int(col[k])
        v = 0.0
        for j in range(int(indptr[r]), int(indptr[r + 1])):
            if int(indices[j]) == c:
                v = float(data[j])
                break
        val[k] = v
    return val


def csr_row_index(M, indices, indptr, sindices, sdata, res_indices, res_data):
    """Gather row indices/data for CSR fancy indexing (stub)."""
    import rsnumpy as _np
    res_indices[:] = _np.asarray(sindices)[: len(res_indices)]
    res_data[:] = _np.asarray(sdata)[: len(res_data)]


def csr_row_slice(start, stop, step, indptr, indices, data, res_indices, res_data):
    """Slice rows from a CSR matrix (stub)."""
    raise NotImplementedError("csr_row_slice not yet ported")


def csr_column_index1(k, idx, M, N, indptr, indices, col_offsets, res_indptr):
    """Pass 1 of column-based CSR fancy indexing (stub)."""
    raise NotImplementedError("csr_column_index1 not yet ported")


def csr_column_index2(col_order, col_offsets, nnz, indices, data, res_indices, res_data):
    """Pass 2 of column-based CSR fancy indexing (stub)."""
    raise NotImplementedError("csr_column_index2 not yet ported")


def csr_diagonal(M, N, indptr, indices, data, out):
    """Extract diagonal from CSR matrix (stub)."""
    raise NotImplementedError("csr_diagonal not yet ported")


def csr_has_canonical_format(indptr, indices, data):
    """Check if CSR matrix has canonical format (stub)."""
    return True


def csr_has_sorted_indices(indptr, indices):
    """Check if CSR indices are sorted (stub)."""
    return True


def csr_sort_indices(M, N, indptr, indices, data):
    """Sort CSR indices in-place (stub)."""
    pass


def csr_matmat_maxnnz(M, N, nnz, indptr, indices, data, B_indices, B_data, M2, N2):
    """Max nnz for CSR x dense matrix multiply (stub)."""
    return 0


def csr_matmat(M, N, nnz, indptr, indices, data, B_indices, B_data, B_indptr, M2, N2, C_indices, C_data, C_indptr):
    """CSR x dense matrix multiply (stub)."""
    pass


def coo_matvec(M, N, row, col, data, x, y):
    """COO matrix-vector multiply (stub)."""
    raise NotImplementedError("coo_matvec not yet ported")


def coo_matvec_nd(M, N, row, col, data, x, y):
    """COO matrix-vector multiply with 2-D output (stub)."""
    raise NotImplementedError("coo_matvec_nd not yet ported")


def coo_matmat_dense(M, N, row, col, data, B, C):
    """COO x dense matrix multiply (stub)."""
    raise NotImplementedError("coo_matmat_dense not yet ported")


def coo_matmat_dense_nd(M, N, row, col, data, B, C):
    """COO x dense matrix multiply with ND output (stub)."""
    raise NotImplementedError("coo_matmat_dense_nd not yet ported")


def bsr_matmat(M, N, blockdim, indptr, indices, data, B_indices, B_data, B_indptr, M2, N2, C_indices, C_data, C_indptr):
    """BSR x dense matrix multiply (stub)."""
    pass


def bsr_transpose(M, N, blockdim, indptr, indices, data, out_indices, out_data):
    """Transpose BSR matrix (stub)."""
    pass


def bsr_sort_indices(M, N, blockdim, indptr, indices, data):
    """Sort BSR indices (stub)."""
    pass


def bsr_tocsr(M, N, blockdim, indptr, indices, data, out_indptr, out_indices, out_data):
    """Convert BSR to CSR (stub)."""
    pass


# Alias names commonly expected
csr_sum_duplicates = csr_eliminate_zeros
