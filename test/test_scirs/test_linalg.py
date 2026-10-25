"""scicore.linalg — functional tests against analytical references.

xfail markers document functions that are not yet ported or currently
broken; they will XPASS once the port catches up (strict=False, so an
XPASS is reported, not a failure).
"""
import pytest
import rsnumpy as np

import scicore.linalg as sla

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_qr():
    a = np.array([[2.0, 0.0, 1.0], [0.4, 0.0, 3.0]])
    q, r = sla.qr(a)
    assert np.allclose(q.T @ q, np.eye(2))
    assert np.allclose(q @ r, a)


def test_block_diag():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0, 7.0]])
    bd = sla.block_diag(a, b)
    assert bd.shape == (3, 5)
    assert np.allclose(bd[:2, :2], a)
    assert np.allclose(bd[2:, 2:], b)
    assert np.allclose(bd[:2, 2:], 0.0)


def test_expm_zero():
    assert np.allclose(sla.expm(np.zeros((2, 2))), np.eye(2))


def test_expm_diagonal():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert np.allclose(sla.expm(a), np.array([[np.e, 0.0], [0.0, np.e**2]]))


def test_sqrtm_diagonal():
    a = np.array([[4.0, 0.0], [0.0, 9.0]])
    assert np.allclose(sla.sqrtm(a), np.array([[2.0, 0.0], [0.0, 3.0]]))


# ---------------- not yet ported / broken ----------------

def test_solve():
    a = np.array([[4.0, 3.0], [6.0, 3.0]])
    # hand-verified: 4x1+3x2=10, 6x1+3x2=12  ->  x1=1, x2=2
    x = sla.solve(a, np.array([10.0, 12.0]))
    assert np.allclose(x, [1.0, 2.0])


def test_inv():
    a = np.array([[4.0, 3.0], [6.0, 3.0]])
    assert np.allclose(a @ sla.inv(a), np.eye(2))


def test_det():
    a = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    # hand-verified: 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
    assert np.isclose(sla.det(a), 1.0)


def test_eig():
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    w, v = sla.eig(a)
    assert np.allclose(np.sort(np.real(w)), [1.0, 3.0])


def test_eigvalsh():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    expected = ((5.0 - np.sqrt(5.0)) / 2, (5.0 + np.sqrt(5.0)) / 2)
    assert np.allclose(np.sort(sla.eigvalsh(a)), expected)


def test_svd_roundtrip():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    u, s, vt = sla.svd(a)
    assert np.allclose(u @ np.diag(s) @ vt, a)


def test_cholesky():
    a = np.array([[4.0, 2.0], [2.0, 3.0]])
    # default lower=False returns the upper factor U with A = U.T @ U
    u = sla.cholesky(a)
    assert np.allclose(u.T @ u, a)


def test_lstsq():
    a = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    # rank-1 system with exact solutions x1+2x2=1; LAPACK's gelsd returns
    # the minimum-2-norm solution [0.2, 0.4]
    x, *_ = sla.lstsq(a, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(x, [0.2, 0.4], atol=1e-10)


def test_lu():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    p, l, u = sla.lu(a)
    assert np.allclose(p.T @ l @ u, a)


def test_norm_vector():
    assert np.isclose(sla.norm(np.array([3.0, 4.0])), 5.0)


def test_norm_matrix_ords():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(sla.norm(m, "fro"), np.sqrt(30.0))
    assert np.isclose(sla.norm(m, 1), 6.0)
    assert np.isclose(sla.norm(m, np.inf), 7.0)


def test_toeplitz():
    t = sla.toeplitz(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(t, np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]]))


def test_hankel():
    # scipy.linalg.hankel: c is the first COLUMN, r the last ROW
    # (default r = zeros). Documented: hankel([1,17,99]) ->
    # [[1,17,99],[17,99,0],[99,0,0]]
    h = sla.hankel(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(h, np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 0.0], [3.0, 0.0, 0.0]]))
    # documented example with an explicit last row
    h2 = sla.hankel([1, 2, 3, 4], [4, 7, 7, 8, 9])
    assert np.allclose(h2, np.array([[1, 2, 3, 4, 7],
                                     [2, 3, 4, 7, 7],
                                     [3, 4, 7, 7, 8],
                                     [4, 7, 7, 8, 9]]))


def test_logm():
    a = np.array([[4.0, 0.0], [0.0, 9.0]])
    assert np.allclose(sla.logm(a), np.array([[np.log(4.0), 0.0], [0.0, np.log(9.0)]]))
