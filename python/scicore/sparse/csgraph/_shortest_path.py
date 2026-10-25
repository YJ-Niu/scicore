"""Stub replacement for the deleted Cython ``_shortest_path`` extension.

Provides graph shortest-path algorithms used by ``scipy.sparse.csgraph``.
This is a stub: real graph algorithms should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['shortest_path', 'floyd_warshall', 'dijkstra', 'bellman_ford',
           'johnson', 'yen', 'NegativeCycleError']


class NegativeCycleError(ValueError):
    """Raised when a negative cycle is detected."""
    pass


def shortest_path(csgraph, *args, **kwargs):
    """Stub for shortest_path."""
    return np.zeros(csgraph.shape)


def floyd_warshall(csgraph, *args, **kwargs):
    """Stub for floyd_warshall."""
    return np.zeros(csgraph.shape), np.zeros(csgraph.shape, dtype=int)


def dijkstra(csgraph, directed=True, return_predecessors=True, indices=None):
    """Shortest paths from every source (scipy-compatible ``dijkstra``).

    Heap-based Dijkstra over the CSR storage of ``csgraph``; stored
    entries are the weighted edges.  Returns ``(dist_matrix,
    predecessors)`` — or just ``dist_matrix`` when
    ``return_predecessors`` is False.  ``indices`` restricts the sources.
    """
    import heapq

    g = csgraph.tocsr()
    n = g.shape[0]
    indptr = [int(v) for v in g.indptr]
    indices_arr = [int(v) for v in g.indices]
    data = [float(v) for v in g.data]
    sources = range(n) if indices is None else [int(i) for i in indices]

    inf = float("inf")
    dist = [[inf] * n for _ in range(n)]
    preds = [[-1] * n for _ in range(n)]
    for src in sources:
        dist[src][src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[src][u]:
                continue
            for k in range(indptr[u], indptr[u + 1]):
                v = indices_arr[k]
                nd = d + data[k]
                if nd < dist[src][v]:
                    dist[src][v] = nd
                    preds[src][v] = u
                    heapq.heappush(heap, (nd, v))
    dist_mat = np.array(dist)
    if return_predecessors:
        return dist_mat, np.array(preds, dtype=int)
    return dist_mat


def bellman_ford(csgraph, *args, **kwargs):
    """Stub for bellman_ford."""
    return np.zeros(csgraph.shape[0]), np.zeros(csgraph.shape[0], dtype=int)


def johnson(csgraph, *args, **kwargs):
    """Stub for johnson."""
    return np.zeros(csgraph.shape)


def yen(*args, **kwargs):
    """Stub for yen."""
    return []
