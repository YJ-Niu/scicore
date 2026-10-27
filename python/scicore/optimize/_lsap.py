"""Pure Python replacement for the deleted C ``_lsap`` extension.

Provides the linear sum assignment (LSAP) solver used by
``scicore.optimize``: the Hungarian algorithm (Kuhn-Munkres) with
potentials, O(n^2 m) for an n x m cost matrix.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['linear_sum_assignment']


def _hungarian_min(cost):
    """Min-cost assignment of all rows to distinct columns.

    Parameters
    ----------
    cost : list of n lists of m floats (n <= m)

    Returns
    -------
    list of n ints
        For each row, the index of its assigned column.
    """
    n = len(cost)
    m = len(cost[0]) if n else 0
    INF = float("inf")
    u = [0.0] * (n + 1)          # row potentials
    v = [0.0] * (m + 1)          # column potentials
    p = [0] * (m + 1)            # p[j] = 1-based row matched to column j
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # augment along the way chain
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def linear_sum_assignment(cost_matrix, maximize=False):
    """Solve the linear sum assignment problem.

    Find the assignment of rows to columns that minimizes the total cost
    ``sum(cost_matrix[i, col_ind[i]])`` (maximizes it when ``maximize``
    is True).

    Parameters
    ----------
    cost_matrix : (n, m) array_like
        Cost matrix.
    maximize : bool, optional
        If True, find the assignment that maximizes the total cost.
        Default is False.

    Returns
    -------
    row_ind : (k,) int array
    col_ind : (k,) int array
        ``k = min(n, m)``.  The i-th pair ``(row_ind[i], col_ind[i])`` is an
        assigned row/column.  All rows are assigned when ``n <= m``,
        otherwise all columns are.
    """
    cost = np.asarray(cost_matrix)
    n, m = cost.shape
    c = cost.tolist()
    if n > m:
        # Solve the transpose: assign all m columns to distinct rows.
        data = [[c[i][j] for i in range(n)] for j in range(m)]
        if maximize:
            data = [[-t for t in row] for row in data]
        rows_of_col = _hungarian_min(data)
        return np.array(rows_of_col), np.arange(m)
    data = [list(row) for row in c]
    if maximize:
        data = [[-t for t in row] for row in data]
    col_of_row = _hungarian_min(data)
    return np.arange(n), np.array(col_of_row)
