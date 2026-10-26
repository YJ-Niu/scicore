"""Replacements for the deleted C ``_qhull`` extension.

``ConvexHull``, ``Delaunay`` and ``Voronoi`` are backed by the pure-Rust
geometry core (``rust/src/geometry.rs``) for 2-D input.  Remaining qhull
functionality (N > 2 dimensions, ``HalfspaceIntersection``, raw ``Qhull``)
is not yet available.
"""

import rsnumpy as np

from scicore._scicore_rs import spatial as _rust_spatial

__all__ = [
    "Qhull",
    "QhullError",
    "Delaunay",
    "ConvexHull",
    "Voronoi",
    "HalfspaceIntersection",
]


class QhullError(Exception):
    """Exception raised when a qhull operation fails."""


class Qhull:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Qhull is not yet ported")


class HalfspaceIntersection:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HalfspaceIntersection is not yet ported")


def _as_2d_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise NotImplementedError(
            "the Rust geometry core only supports 2-D point sets"
        )
    if int(points.shape[0]) == 0:
        raise QhullError("no input points")
    return np.ascontiguousarray(points)


class ConvexHull:
    """Convex hull of a 2-D point set (Graham scan in the Rust core)."""

    def __init__(self, points, qhull_options=None, copy_input=True, **kwargs):
        pts = _as_2d_points(points)
        vertices, area, eq = _rust_spatial.geometry_convex_hull_2d(
            pts.tobytes(), int(pts.shape[0])
        )
        vertices = [int(i) for i in vertices]
        n_h = len(vertices)
        self.points = pts.copy() if copy_input else pts
        self.ndim = 2
        self.area = float(area)
        self.volume = float(area)
        self.vertices = np.array(vertices, dtype=int)
        self.simplices = np.array(
            [[vertices[i], vertices[(i + 1) % n_h]] for i in range(n_h)],
            dtype=int,
        )
        self.equations = np.array(eq, dtype=float).reshape(n_h, 3)
        self.coplanar = {}
        self.qhull_options = qhull_options
        self.copy_input = copy_input

    def __repr__(self):
        return f"ConvexHull(vertices={self.vertices.tolist()})"


class Delaunay:
    """Delaunay triangulation of a 2-D point set (Bowyer-Watson in the Rust
    core)."""

    def __init__(self, points, qhull_options=None, copy=True, **kwargs):
        pts = _as_2d_points(points)
        self._pts = [
            (float(x), float(y)) for x, y in np.asarray(pts).tolist()
        ]
        simplices, vertices, cc = _rust_spatial.geometry_delaunay_2d(
            pts.tobytes(), int(pts.shape[0])
        )
        self._simplices = np.array(simplices, dtype=int).reshape(-1, 3)
        self._vertices = np.array(vertices, dtype=int)
        self.points = pts.copy() if copy else pts
        self.simplices = self._simplices
        self.vertices = self._vertices
        self.equidistant = np.array(cc, dtype=float).reshape(-1, 2)
        self.ndim = 2
        self.coplanar = {}
        self.qhull_options = qhull_options
        self.copy_input = copy
        # Neighbors: for simplex i and vertex j, the index of the simplex
        # sharing the opposite edge (-1 at the hull boundary).
        n = len(self._simplices)
        edge_map = {}
        for si in range(n):
            tri = [int(v) for v in self._simplices[si]]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = (tri[a], tri[b]) if tri[a] < tri[b] else (tri[b], tri[a])
                edge_map[key] = si
        nbr = []
        for si in range(n):
            tri = [int(v) for v in self._simplices[si]]
            row = []
            for j in range(3):
                a, b = (j + 1) % 3, (j + 2) % 3
                key = (tri[a], tri[b]) if tri[a] < tri[b] else (tri[b], tri[a])
                row.append(edge_map.get(key, -1))
            nbr.append(row)
        self.neighbors = np.array(nbr, dtype=int)

    def find_simplex(self, x, max_dist=np.inf):
        """Index of the simplex containing each point in ``x`` (-1 if none
        or farther than ``max_dist`` from the point)."""
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 1
        if scalar:
            x = x.reshape(1, 2)
        xs = [float(v) for v in np.asarray(x).tolist()
              for row in ([v] if isinstance(v, float) else v)]
        out = []
        n = len(self._simplices)
        for i in range(0, len(xs), 2):
            px, py = xs[i], xs[i + 1]
            found = -1
            for si in range(n):
                a, b, c = [int(v) for v in self._simplices[si]]
                (ax, ay) = self._pts[a]
                (bx, by) = self._pts[b]
                (cx, cy) = self._pts[c]
                d = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
                if abs(d) < 1e-15:
                    continue
                w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / d
                w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / d
                w2 = 1.0 - w0 - w1
                if w0 >= -1e-9 and w1 >= -1e-9 and w2 >= -1e-9:
                    if max_dist == np.inf or (
                        ((ax + bx + cx) / 3.0 - px) ** 2
                        + ((ay + by + cy) / 3.0 - py) ** 2
                    ) ** 0.5 <= max_dist:
                        found = si
                        break
            out.append(found)
        if scalar:
            return np.array(out[0], dtype=int)
        return np.array(out, dtype=int)

    def __repr__(self):
        return f"Delaunay({len(self._pts)} points, {len(self._simplices)} simplices)"


class Voronoi:
    """Voronoi diagram of a 2-D point set (dual of the Delaunay
    triangulation in the Rust core)."""

    def __init__(self, points, qhull_options=None, **kwargs):
        pts = _as_2d_points(points)
        (
            verts,
            vertices,
            ridges,
            ridge_points,
            regions,
        ) = _rust_spatial.geometry_voronoi_2d(pts.tobytes(), int(pts.shape[0]))
        self.points = pts.copy()
        self.vertices = np.array(vertices, dtype=float).reshape(-1, 2)
        self.verts = np.array(verts, dtype=float).reshape(-1, 2)
        self.ridges = np.array(ridges, dtype=int).reshape(-1, 2)
        self.ridge_points = np.array(ridge_points, dtype=float).reshape(-1, 2)
        self.regions = [list(map(int, r)) for r in regions]
        self.ndim = 2
        self.qhull_options = qhull_options
        self.copy_input = True

    def __repr__(self):
        return f"Voronoi({len(self.points)} points, {len(self.verts)} vertices)"
