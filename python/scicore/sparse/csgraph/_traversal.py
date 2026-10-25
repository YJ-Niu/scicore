"""Stub replacement for the deleted Cython ``_traversal`` extension.

Provides graph traversal algorithms used by ``scipy.sparse.csgraph``.
This is a stub: real graph algorithms should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['breadth_first_order', 'depth_first_order', 'breadth_first_tree',
           'depth_first_tree', 'connected_components']


def breadth_first_order(csgraph, i, *args, **kwargs):
    """Stub for breadth_first_order."""
    return np.array([i]), np.array([0])


def depth_first_order(csgraph, i, *args, **kwargs):
    """Stub for depth_first_order."""
    return np.array([i]), np.array([0])


def breadth_first_tree(csgraph, i, *args, **kwargs):
    """Stub for breadth_first_tree."""
    return np.zeros_like(csgraph)


def depth_first_tree(csgraph, i, *args, **kwargs):
    """Stub for depth_first_tree."""
    return np.zeros_like(csgraph)


def connected_components(csgraph, directed=False, connection=1, return_labels=True):
    """Number of connected components and per-node component labels.

    scipy-compatible: ``connection`` must be 1; for undirected graphs the
    graph is symmetrized (``G + G.T``) before traversing.
    """
    if connection != 1:
        raise NotImplementedError("Only connection=1 is supported")
    g = csgraph.tocsr()
    if not directed:
        g = g + g.T
    n = g.shape[0]
    indptr = [int(v) for v in g.indptr]
    indices = [int(v) for v in g.indices]
    labels = [-1] * n
    n_components = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        labels[start] = n_components
        stack = [start]
        while stack:
            u = stack.pop()
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if labels[v] == -1:
                    labels[v] = n_components
                    stack.append(v)
        n_components += 1
    if return_labels:
        return n_components, np.array(labels, dtype=int)
    return n_components
