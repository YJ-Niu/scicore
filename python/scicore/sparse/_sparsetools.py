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

def csr_matvec(M, N, indptr, indices, data, x, y):
    """y = A x for CSR matrix (scipy-compatible signature).

    ``y`` is pre-allocated (length M) and filled in place.
    """
    for i in range(M):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * x[indices[j]]
        y[i] = s


def csr_matvecs(M, N, n_vecs, indptr, indices, data, x, y):
    """Y = A X for CSR matrix (scipy-compatible signature).

    ``x``/``y`` are raveled row-major buffers of shape (N, n_vecs) and
    (M, n_vecs); ``y`` is filled in place.
    """
    for k in range(n_vecs):
        for i in range(M):
            s = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                s += x[indices[j] * n_vecs + k] * data[j]
            y[k * M + i] = s


def csc_matvec(M, N, indptr, indices, data, x, y):
    """y = A x for CSC matrix A (scipy-compatible signature).

    ``indptr`` is per column, ``indices[k]`` is the row; ``y`` is
    pre-allocated (length M) and filled in place.
    """
    for j in range(N):
        xj = x[j]
        for k in range(indptr[j], indptr[j + 1]):
            i = indices[k]
            y[i] = y[i] + data[k] * xj


def csc_matvecs(M, N, n_vecs, indptr, indices, data, x, y):
    """Y = A X for CSC matrix A (scipy-compatible signature).

    ``x``/``y`` are raveled row-major buffers of shape (N, n_vecs) and
    (M, n_vecs); ``y`` is filled in place.
    """
    for j in range(N):
        for k in range(indptr[j], indptr[j + 1]):
            i = indices[k]
            d = data[k]
            for v in range(n_vecs):
                y[i * n_vecs + v] = y[i * n_vecs + v] + d * x[j * n_vecs + v]


def _csr_merge_rows(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices, B_data,
                    C_indptr, C_indices, C_data, op, inner=False):
    """Row-wise merge of two CSR matrices (scipy ``csr_*_csr`` core).

    Outer (default) emits every column present in either row, treating
    missing entries as 0; inner emits only columns present in both rows
    (scipy ``csr_eldiv_csr`` semantics).  Returns the actual nnz; the
    caller's ``_binopt`` wrapper prunes the pre-allocated tail.
    """
    nnz = 0
    for i in range(M):
        C_indptr[i] = nnz
        k, ja = A_indptr[i], A_indptr[i + 1]
        l, jb = B_indptr[i], B_indptr[i + 1]
        while k < ja and l < jb:
            a_idx = int(A_indices[k])
            b_idx = int(B_indices[l])
            if a_idx < b_idx:
                if not inner:
                    C_indices[nnz] = a_idx
                    C_data[nnz] = A_data[k]
                    nnz += 1
                k += 1
            elif a_idx > b_idx:
                if not inner:
                    C_indices[nnz] = b_idx
                    C_data[nnz] = B_data[l]
                    nnz += 1
                l += 1
            else:
                C_indices[nnz] = a_idx
                C_data[nnz] = op(A_data[k], B_data[l])
                nnz += 1
                k += 1
                l += 1
        if not inner:
            while k < ja:
                C_indices[nnz] = int(A_indices[k])
                C_data[nnz] = A_data[k]
                nnz += 1
                k += 1
            while l < jb:
                C_indices[nnz] = int(B_indices[l])
                C_data[nnz] = B_data[l]
                nnz += 1
                l += 1
    C_indptr[M] = nnz
    return nnz


def csr_plus_csr(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices, B_data,
                 C_indptr, C_indices, C_data):
    return _csr_merge_rows(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices,
                           B_data, C_indptr, C_indices, C_data,
                           lambda a, b: a + b)


def csr_minus_csr(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices, B_data,
                  C_indptr, C_indices, C_data):
    return _csr_merge_rows(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices,
                           B_data, C_indptr, C_indices, C_data,
                           lambda a, b: a - b)


def csr_multiply_csr(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices, B_data,
                     C_indptr, C_indices, C_data):
    return _csr_merge_rows(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices,
                           B_data, C_indptr, C_indices, C_data,
                           lambda a, b: a * b)


def csr_eldiv_csr(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices, B_data,
                  C_indptr, C_indices, C_data):
    """Elementwise division, keeping only entries present in both (scipy)."""
    return _csr_merge_rows(M, N, A_indptr, A_indices, A_data, B_indptr, B_indices,
                           B_data, C_indptr, C_indices, C_data,
                           lambda a, b: a / b, inner=True)


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


def csr_tocsc(M, N, indptr, indices, data, csc_indptr, csc_indices, csc_data):
    """Convert CSR(M, N) to CSC (scipy-compatible, in-place fill).

    Input is CSR storage; the three output buffers are pre-allocated by
    the caller and filled here (rows of the CSC = columns of the CSR).
    """
    indptr = [int(v) for v in indptr]
    indices = [int(v) for v in indices]
    data = list(data)
    col_count = [0] * N
    for i in range(M):
        for k in range(indptr[i], indptr[i + 1]):
            col_count[indices[k]] += 1
    cursor = 0
    csc_indptr[0] = 0
    for j in range(N):
        cursor += col_count[j]
        csc_indptr[j + 1] = cursor
    pos = [int(csc_indptr[j]) for j in range(N)]
    for i in range(M):
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            p = pos[j]
            csc_indices[p] = i
            csc_data[p] = data[k]
            pos[j] += 1


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
    # Count entries per row with bincount: rsnumpy's in-place fancy
    # indexing with repeated indices (``c[[0, 0]] += 1``) applies the
    # update only once, which would undercount rows that appear more
    # than once.  bincount is correct and available in rsnumpy.
    counts = np.bincount(major.astype(np.intp), minlength=M)
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
    # Scatter with a plain loop: rsnumpy's in-place fancy indexing with
    # repeated indices (``out[idx] += data``) only applies each update
    # once, so duplicate coordinates would lose contributions.  scipy's
    # C coo_todense accumulates duplicates.
    for k in range(nnz):
        if fortran:
            p = int(col[k]) * M + int(row[k])
        else:
            p = int(row[k]) * N + int(col[k])
        out[p] = out[p] + data[k]


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
    for k in range(nnz):
        out[int(idx[k])] = out[int(idx[k])] + data[k]


# ----- DIA -----

def dia_matmat(offsets, data, shape, B):
    raise NotImplementedError("dia_matmat not yet ported")


def dia_matvec(offsets, data, shape, x, y):
    raise NotImplementedError("dia_matvec not yet ported")


def dia_matvecs(offsets, data, shape, x, y):
    raise NotImplementedError("dia_matvecs not yet ported")


def dia_tocsr(M, N, num_diags, num_diag_vals, offsets, data, order,
              out_data, out_indices, out_indptr):
    """scipy-compatible ``dia_tocsr``: fill pre-allocated CSR buffers.

    Diagonals are emitted in ``order`` (argsort of offsets) so columns
    come out sorted; explicit zeros are dropped.  Returns the actual
    nnz written.
    """
    idx = 0
    out_indptr[0] = 0
    for i in range(M):
        for k in range(num_diags):
            d = int(order[k])
            off = int(offsets[d])
            col = i + off
            if col < 0 or col >= N:
                continue
            p = i + (max(0, -off))
            if p >= num_diag_vals:
                continue
            v = data[d, p]
            if v != 0:
                out_data[idx] = v
                out_indices[idx] = col
                idx += 1
        out_indptr[i + 1] = idx
    return idx


# ----- CSC -----

def expandptr(N, indptr, indices):
    """Fill ``indices`` so that ``indices[indptr[i]:indptr[i + 1]] == i``.

    scipy-compatible ``expandptr`` (used when converting compressed
    formats back to COO).
    """
    for i in range(N):
        for j in range(indptr[i], indptr[i + 1]):
            indices[j] = i


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


def csr_diagonal(k, M, N, indptr, indices, data, out):
    """k-th diagonal of a CSR matrix (scipy-compatible, fills ``out``).

    ``out[i] = A[i, i + k]`` for ``k >= 0`` and ``out[i] = A[i - k, i]``
    for ``k < 0``; positions without a stored entry become 0.  ``out`` is
    pre-allocated by the caller with the correct length.
    """
    indptr = [int(v) for v in indptr]
    indices = [int(v) for v in indices]
    data = list(data)

    def row_get(ia, ib, col):
        lo, hi = 0, ib - ia
        while lo < hi:
            mid = (lo + hi) // 2
            if indices[ia + mid] < col:
                lo = mid + 1
            else:
                hi = mid
        if lo < (ib - ia) and indices[ia + lo] == col:
            return data[ia + lo]
        return 0.0

    if k >= 0:
        length = min(M, N - k)
        for i in range(length):
            out[i] = row_get(indptr[i], indptr[i + 1], i + k)
    else:
        kk = -k
        length = min(M - kk, N)
        for i in range(length):
            out[i] = row_get(indptr[i + kk], indptr[i + kk + 1], i)


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
