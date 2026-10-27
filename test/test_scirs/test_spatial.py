"""scicore.spatial — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.spatial as sspa
import scicore.spatial.distance as sdist
import scicore.spatial.transform as stf

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_cdist_euclidean():
    d = sdist.cdist(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
    assert np.allclose(d, [[5.0]])


def test_pdist():
    d = sdist.pdist(np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]]))
    # (0,1)=5, (0,2)=1, (1,2)=sqrt(18)
    assert np.allclose(d, [5.0, 1.0, np.sqrt(18.0)], rtol=1e-12)


def test_squareform():
    sq = sdist.squareform(np.array([5.0, 1.0, 4.0]))
    assert np.allclose(sq, [[0.0, 5.0, 1.0], [5.0, 0.0, 4.0], [1.0, 4.0, 0.0]])


def test_mahalanobis():
    # u=[1,2], v=[0,0], inv-cov = diag(1, 0.25) => sqrt(1 + 4*0.25) = sqrt(2)
    d = sdist.mahalanobis(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.25]]))
    assert np.isclose(d, np.sqrt(2.0), rtol=1e-12)


def test_kdtree_query():
    # Rust-backed kd-tree: nearest neighbor of [1,1] in [[0,0],[3,4]] is [0,0].
    # A single-point k=1 query returns scalars (matching scipy's cKDTree/KDTree).
    tree = sspa.KDTree(np.array([[0.0, 0.0], [3.0, 4.0]]))
    dist, idx = tree.query(np.array([1.0, 1.0]))
    assert np.isclose(dist, np.sqrt(2.0), rtol=1e-12)
    assert int(idx) == 0


# ---------------- not yet ported / broken ----------------


def test_delaunay():
    tri = sspa.Delaunay(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    assert tri.simplices.shape == (1, 3)


def test_voronoi():
    vor = sspa.Voronoi(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert len(vor.verts) > 0


def test_convex_hull():
    hull = sspa.ConvexHull(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
    # interior point (0.5, 0.5) is not on the hull: volume of triangle = 0.5
    assert np.isclose(hull.volume, 0.5)


def test_rotation_from_quat():
    rot = stf.Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(rot.as_quat(), [0.0, 0.0, 0.0, 1.0])


def test_rotation_apply_90deg_z():
    rot = stf.Rotation.from_euler("z", 90, degrees=True)
    assert np.allclose(rot.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_slerp_midpoint():
    # current Slerp API takes a single (batched) Rotation instance;
    # from_euler's last dim must match the number of seq axes
    s = stf.Slerp(
        [0.0, 1.0],
        stf.Rotation.from_euler("z", [[0.0], [90.0]], degrees=True),
    )
    mid = s(0.5)
    assert np.allclose(mid.apply([1.0, 0.0, 0.0]), [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0], atol=1e-12)
