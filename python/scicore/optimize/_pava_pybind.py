"""Pure Python replacement for the deleted C++ ``_pava_pybind`` extension.

Provides PAVA (pool adjacent violators algorithm) used by
``scicore.optimize`` — weighted isotonic regression in O(n).
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['pava']


def pava(x, w, r):
    """Isotonic regression via pool adjacent violators (PAVA).

    Parameters
    ----------
    x : (n,) float array_like
        Observations.
    w : (n,) float array_like
        Strictly positive case weights.
    r : (n+1,) int array
        Caller-allocated scratch; see return value for the layout written
        into it.

    Returns
    -------
    x_iso : (n,) float array
        Isotonic (non-decreasing) weighted least-squares fit of ``x``.
    w_block : (n,) float array
        First ``b`` entries: sum of the case weights of each block.
    r : (n+1,) int array
        First ``b+1`` entries: the block boundaries ``[0, s1, s2, ..., n]``
        (start index of each block, plus the end of the last one).
    b : int
        Number of blocks.
    """
    xv = [float(t) for t in np.asarray(x).tolist()]
    wv = [float(t) for t in np.asarray(w).tolist()]
    n = len(xv)
    # Pool adjacent violators: blocks of [start, sum_y*w, sum_w]; the block
    # mean is the weighted mean sum(y*w)/sum(w).  Weights are strictly
    # positive, so means are compared by cross-multiplication.
    blocks = []
    for i in range(n):
        blocks.append([i, xv[i] * wv[i], wv[i]])
        while (len(blocks) > 1 and
               blocks[-2][1] * blocks[-1][2] > blocks[-1][1] * blocks[-2][2]):
            prev, cur = blocks[-2], blocks[-1]
            prev[1] += cur[1]
            prev[2] += cur[2]
            blocks.pop()
    b = len(blocks)
    x_list = [0.0] * n
    w_list = [0.0] * n
    r_list = [0] * (n + 1)
    for k, (start, sy, sw) in enumerate(blocks):
        end = blocks[k + 1][0] if k + 1 < b else n
        mean = sy / sw
        for i in range(start, end):
            x_list[i] = mean
        w_list[k] = sw
    for k in range(1, b):
        r_list[k] = blocks[k][0]
    if b > 0:
        r_list[b] = n
    return (np.array(x_list, dtype=np.float64),
            np.array(w_list, dtype=np.float64),
            np.array(r_list),
            b)
