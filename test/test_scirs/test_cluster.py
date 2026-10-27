"""scicore.cluster — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.cluster.hierarchy as shier
import scicore.cluster.vq as svq

pytestmark = pytest.mark.filterwarnings("default")


def _two_blobs():
    return np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])


# ---------------- working ----------------

def test_kmeans2_two_blobs():
    # Seeded RNG: kmeans2's default init samples from OS entropy; without a
    # seed it occasionally raises ClusterError on an empty cluster (flaky in
    # CI — e.g. seed 7 fails, seed 42 is stable).
    centroids, _ = svq.kmeans2(
        _two_blobs(), 2, missing="raise", rng=np.random.default_rng(42)
    )
    c = sorted(map(tuple, np.round(centroids, 1).tolist()))
    assert c == [(0.1, 0.0), (10.1, 10.0)] or c == [(0.0, 0.0), (10.0, 10.0)]


def test_vq_assignment():
    codebook = np.array([[0.0, 0.0], [10.0, 10.0]])
    data = np.array([[0.1, 0.0], [9.9, 10.0]])
    labels, dist = svq.vq(codebook, data)
    assert list(labels) == [0, 1]


# ---------------- not yet ported ----------------

def test_linkage_ward():
    Z = shier.linkage(_two_blobs(), method="ward")
    assert Z.shape == (3, 4)
    # the last merge joins the two blobs
    assert Z[-1, 2] > 5.0


def test_fcluster_two_clusters():
    Z = shier.linkage(_two_blobs(), method="ward")
    labels = shier.fcluster(Z, t=2, criterion="maxclust")
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
