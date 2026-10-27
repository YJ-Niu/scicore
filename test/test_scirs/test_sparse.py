"""scicore.sparse — functional tests against analytical references.

The whole sparse stack is currently broken: matrix construction hits
rsnumpy conversion-arity bugs, and the linalg/csgraph layers still
import upstream scipy (not installed).
"""
import pytest
import rsnumpy as np

import scicore.sparse as sps

pytestmark = pytest.mark.filterwarnings("default")


def _dense_2x2():
    return np.array([[1.0, 0.0], [0.0, 2.0]])


# ---------------- broken ----------------

def test_csr_matrix_roundtrip():
    m = sps.csr_matrix(_dense_2x2())
    assert np.allclose(m.toarray(), _dense_2x2())


def test_diags():
    m = sps.diags([1.0, 2.0])
    assert np.allclose(m.toarray(), [[1.0, 0.0], [0.0, 2.0]])


def test_eye():
    m = sps.eye(3, format="csr")
    assert np.allclose(m.toarray(), np.eye(3))


def test_addition():
    a = sps.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    b = sps.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert np.allclose((a + b).toarray(), np.array([[2.0, 1.0], [1.0, 3.0]]))


def test_matvec():
    a = sps.csr_matrix(_dense_2x2())
    out = a @ np.array([1.0, 1.0])
    assert np.allclose(np.asarray(out), [1.0, 2.0])


def test_spsolve():
    a = sps.csr_matrix(np.array([[4.0, 3.0], [6.0, 3.0]]))
    x = sps.linalg.spsolve(a, np.array([10.0, 12.0]))
    assert np.allclose(x, [1.0, 2.0])


def test_eigsh():
    # scipy requires k < N for eigsh; use 3x3 so k=2 is valid
    a = sps.csr_matrix(np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]))
    # which="LA" = largest algebraic eigenvalues: {3, 4}
    w, v = sps.linalg.eigsh(a, k=2, which="LA")
    assert np.allclose(np.sort(w), [3.0, 4.0])
    assert np.asarray(v).shape == (3, 2)


def test_eigs():
    # 5x5 upper triangular (eigenvalues 2..6); ARPACK requires k < N - 1
    a = sps.csr_matrix(np.array([[2., 1., 0., 0., 0.],
                                 [0., 3., 1., 0., 0.],
                                 [0., 0., 4., 1., 0.],
                                 [0., 0., 0., 5., 1.],
                                 [0., 0., 0., 0., 6.]]))
    w, v = sps.linalg.eigs(a, k=2, which="LM")
    w_list = [complex(x) for x in np.asarray(w).tolist()]
    assert np.allclose(sorted(x.real for x in w_list), [5.0, 6.0])
    A = [[2., 1., 0., 0., 0.], [0., 3., 1., 0., 0.], [0., 0., 4., 1., 0.],
         [0., 0., 0., 5., 1.], [0., 0., 0., 0., 6.]]
    v_rows = np.asarray(v).tolist()
    for i in range(2):
        col = [v_rows[r][i] for r in range(5)]
        Av = [sum(A[r][c] * col[c] for c in range(5)) for r in range(5)]
        wv = [w_list[i] * col[c] for c in range(5)]
        assert max(abs(Av[r] - wv[r]) for r in range(5)) < 1e-8
    w_sm = sps.linalg.eigs(a, k=2, which="SM", return_eigenvectors=False)
    assert np.allclose(sorted(complex(x).real for x in np.asarray(w_sm).tolist()),
                       [2.0, 3.0])


def test_svds():
    # diagonal 4x4: singular values {3, 2, 1, 0}
    A = [[1., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 3., 0.], [0., 0., 0., 0.]]
    u, s, vh = sps.linalg.svds(np.array(A), k=2)
    s_list = np.asarray(s).tolist()
    assert np.allclose(s_list, [2.0, 3.0])
    # rank-2 reconstruction = A with the smallest singular direction removed
    u_rows = np.asarray(u).tolist()
    vh_rows = np.asarray(vh).tolist()
    expected = [[0., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 3., 0.],
                [0., 0., 0., 0.]]
    for r in range(4):
        for c in range(4):
            val = sum(u_rows[r][j] * s_list[j] * vh_rows[j][c] for j in range(2))
            assert abs(val - expected[r][c]) < 1e-8
    s_sp = sps.linalg.svds(sps.csr_matrix(np.array(A)), k=2,
                           return_singular_vectors=False)
    assert np.allclose(np.asarray(s_sp).tolist(), [2.0, 3.0])


def test_dijkstra():
    # directed chain 0 -> 1 (w=1) -> 2 (w=2)
    g = sps.csr_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]))
    dist, _ = sps.csgraph.dijkstra(g)
    assert np.isclose(dist[0, 2], 3.0)


def test_connected_components():
    g = sps.csr_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    n, labels = sps.csgraph.connected_components(g)
    assert n == 2


def test_laplacian():
    g = sps.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    L = sps.csgraph.laplacian(g)
    assert np.allclose(L.toarray(), [[1.0, -1.0], [-1.0, 1.0]])
