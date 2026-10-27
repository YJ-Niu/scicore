"""Pure-Python replacement for the deleted Cython ``_hierarchy`` extension.

Implements the low-level routines used by
``scicore.cluster.hierarchy._hierarchy_impl``:

* ``fast_linkage`` / ``nn_chain`` — naive O(n^2) agglomerative merging
  (the C nn-chain speed-up is not available in this port);
* ``mst_single_linkage`` — single linkage (minimum rule);
* ``cluster_*`` — dendrogram-cutting routines used by ``fcluster``;
* ``calculate_cluster_sizes``.
"""

from __future__ import annotations

import math

import rsnumpy as np

__all__ = [
    'mst_single_linkage', 'nn_chain', 'fast_linkage',
    'calculate_cluster_sizes',
    'cluster_in', 'cluster_dist', 'cluster_maxclust_dist',
    'cluster_maxclust_monocrit', 'cluster_monocrit',
]

# method codes (must match ``_LINKAGE_METHODS`` in ``_hierarchy_impl``)
_SINGLE, _COMPLETE, _AVERAGE, _CENTROID, _MEDIAN, _WARD, _WEIGHTED = range(7)


def _condensed_to_full(yl, n):
    """Condensed distance vector (length n(n-1)/2) -> n x n nested list."""
    D = [[0.0] * n for _ in range(n)]
    k = 0
    for i in range(n - 1):
        row = D[i]
        for j in range(i + 1, n):
            d = yl[k]
            k += 1
            row[j] = d
            D[j][i] = d
    return D


def _update(d_im, d_jm, sa, sb, sm, d_ij, code):
    """Distance from the new cluster (i+j) to a third cluster m.

    Sizes: sa/sb for i/j, sm for m; d_im/d_jm/d_ij are the current
    (method-specific) cluster distances.
    """
    if code == _SINGLE:
        return min(d_im, d_jm)
    if code == _COMPLETE:
        return max(d_im, d_jm)
    if code == _AVERAGE or code == _CENTROID:
        return (sa * d_im + sb * d_jm) / (sa + sb)
    if code == _WEIGHTED:
        return 0.5 * (d_im + d_jm)
    if code == _MEDIAN:
        if sa == sb:
            return 0.5 * (d_im + d_jm)
        v = (sa * d_im - sb * d_jm) / (sa - sb)
        return v if v > 0.0 else 0.0
    # _WARD: exact Lance update (preserves the within-cluster sum of
    # squares definition of Ward's method).
    denom = sa + sb + sm
    v = (d_im * d_im * (sa + sm) + d_jm * d_jm * (sb + sm)) / denom \
        - sm * d_ij * d_ij / denom
    if v < 0.0:
        v = 0.0
    return math.sqrt(v)


def _agglomerate(D, n, code):
    """Naive O(n^2) agglomerative merging; returns (n-1) linkage rows.

    The merge at row k creates linkage-matrix cluster ``n + k``; later
    rows must reference that id, never the surviving leaf slot.
    """
    clusters = list(range(n))
    size = {c: 1.0 for c in range(n)}
    lid = {c: c for c in range(n)}  # storage slot -> linkage-matrix id
    Z = [[0.0, 0.0, 0.0, 0.0] for _ in range(n - 1)]
    for k in range(n - 1):
        m = len(clusters)
        best = None
        for ai in range(m - 1):
            ci = clusters[ai]
            row = D[ci]
            for aj in range(ai + 1, m):
                d = row[clusters[aj]]
                if best is None or d < best[0]:
                    best = (d, ai, aj)
        d, ai, aj = best
        ci, cj = clusters[ai], clusters[aj]
        li, lj = lid[ci], lid[cj]
        sa, sb = size[ci], size[cj]
        Z[k] = [float(li), float(lj), d, sa + sb]
        for cm in clusters:
            if cm == ci or cm == cj:
                continue
            newd = _update(D[ci][cm], D[cj][cm], sa, sb, size[cm], d, code)
            D[ci][cm] = newd
            D[cm][ci] = newd
        del clusters[aj]
        size[ci] = sa + sb
        lid[ci] = n + k
    return Z


def _linkage_from(y, n, code):
    n = int(n)
    yl = [float(v) for v in np.asarray(y).tolist()]
    D = _condensed_to_full(yl, n)
    Z = _agglomerate(D, n, code)
    return np.array(Z, dtype=float)


def fast_linkage(y, n, method_code):
    return _linkage_from(y, n, method_code)


def nn_chain(y, n, method_code):
    # The C nn-chain heuristic is a speed-up only; the naive merge loop
    # produces the same linkage matrix.
    return _linkage_from(y, n, method_code)


def mst_single_linkage(y, n):
    return _linkage_from(y, n, _SINGLE)


def calculate_cluster_sizes(Z):
    rows = np.asarray(Z).tolist()
    out = [float(row[3]) for row in rows]
    return np.array(out, dtype=float)


# ---------------------------------------------------------------------------
# Dendrogram cutting (fcluster backends)
# ---------------------------------------------------------------------------

def _children_map(Z):
    """Row k merges children (i, j) into node n+k; returns (rows, n, children)."""
    rows = np.asarray(Z).tolist()
    n = len(rows) + 1
    children = {}
    for k, row in enumerate(rows):
        children[n + k] = (int(row[0]), int(row[1]))
    return rows, n, children


def _leaf_sets(rows, n, children):
    leafsets = {c: [c] for c in range(n)}
    for k in range(len(rows)):
        i, j = children[n + k]
        leafsets[n + k] = leafsets[i] + leafsets[j]
    return leafsets


def _write_labels(T, nodes, leafsets, n):
    lab = [0] * n
    for c, cl in enumerate(nodes):
        for leaf in leafsets[cl]:
            lab[leaf] = c + 1
    for leaf in range(n):
        T[leaf] = lab[leaf]


def cluster_maxclust_dist(Z, T, n, t):
    rows, n, children = _children_map(Z)
    leafsets = _leaf_sets(rows, n, children)
    # Undo merges from the top until exactly t clusters remain.
    nodes = [2 * n - 2]
    t = int(t)
    while len(nodes) < t:
        idx = max(range(len(nodes)), key=lambda p: nodes[p] - n)
        i, j = children[nodes[idx]]
        nodes[idx:idx + 1] = [i, j]
    nodes.sort(key=lambda c: leafsets[c][0])
    _write_labels(T, nodes, leafsets, n)


def cluster_dist(Z, T, t, n):
    rows, n, children = _children_map(Z)
    leafsets = _leaf_sets(rows, n, children)
    # Keep the largest subtrees whose merge height is at most t.
    active = {2 * n - 2}
    for k in range(len(rows) - 1, -1, -1):
        node = n + k
        if node in active and rows[k][2] > float(t):
            active.discard(node)
            active.update(children[node])
    nodes = sorted(active, key=lambda c: leafsets[c][0])
    _write_labels(T, nodes, leafsets, n)


def _node_in_values(R, rows, n):
    Rl = np.asarray(R).tolist()
    in_val = {c: 0.0 for c in range(n)}
    for k in range(len(rows)):
        r = Rl[k]
        in_val[n + k] = max(float(r[1]), float(r[2]), float(r[3]))
    return in_val


def cluster_in(Z, R, T, t, n):
    rows, n, children = _children_map(Z)
    leafsets = _leaf_sets(rows, n, children)
    in_val = _node_in_values(R, rows, n)
    # Keep the largest subtrees whose inconsistent value is at most t.
    active = {2 * n - 2}
    for k in range(len(rows) - 1, -1, -1):
        node = n + k
        if node in active and in_val[node] > float(t):
            active.discard(node)
            active.update(children[node])
    nodes = sorted(active, key=lambda c: leafsets[c][0])
    _write_labels(T, nodes, leafsets, n)


def cluster_monocrit(Z, monocrit, T, t, n):
    rows, n, children = _children_map(Z)
    leafsets = _leaf_sets(rows, n, children)
    # Group leaves by the ancestor t levels above them.
    parent = {}
    for k in range(len(rows)):
        i, j = children[n + k]
        parent[i] = n + k
        parent[j] = n + k
    depth = int(float(t))
    targets = set()
    for leaf in range(n):
        node = leaf
        for _ in range(depth):
            node = parent.get(node)
            if node is None:
                break
        targets.add(node)
    nodes = sorted(targets, key=lambda c: leafsets[c][0])
    _write_labels(T, nodes, leafsets, n)


def cluster_maxclust_monocrit(Z, monocrit, T, n, t):
    rows, n, children = _children_map(Z)
    leafsets = _leaf_sets(rows, n, children)
    in_val = _node_in_values(monocrit, rows, n)
    # Form exactly t clusters, always splitting the most inconsistent one.
    nodes = [2 * n - 2]
    t = int(t)
    while len(nodes) < t:
        idx = max(range(len(nodes)),
                  key=lambda p: (in_val[nodes[p]], nodes[p] - n))
        i, j = children[nodes[idx]]
        nodes[idx:idx + 1] = [i, j]
    nodes.sort(key=lambda c: leafsets[c][0])
    _write_labels(T, nodes, leafsets, n)
