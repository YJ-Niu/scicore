"""scicore.io — file-format roundtrips.

The top-level io module exposes loadmat/savemat/mmread/mmwrite/FortranFile/
netcdf; the wavfile submodule exists but is not re-exported at the top
level (tested via direct submodule import).
"""
import os
import tempfile

import pytest
import rsnumpy as np

import scicore.io as sio

pytestmark = pytest.mark.filterwarnings("default")


@pytest.mark.xfail(reason="loadmat: could not convert string to float (MAT-file header parsing bug)")
def test_matlab_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.mat")
        sio.savemat(p, {"a": np.array([1.0, 2.0])})
        d = sio.loadmat(p)
        assert np.allclose(d["a"], [1.0, 2.0])


def test_wavfile_roundtrip():
    import scicore.io.wavfile as swav

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "t.wav")
        swav.write(fp, 8000, np.array([100, -200, 300], dtype=np.int16))
        rate, data = swav.read(fp)
        assert rate == 8000
        assert list(data) == [100, -200, 300]


@pytest.mark.xfail(reason="coo_tocsr() takes 4 positional arguments but 9 were given (rsnumpy)")
def test_mmfile_roundtrip():
    import scicore.sparse as sps

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "t.mtx")
        sio.mmwrite(fp, sps.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]])))
        m = sio.mmread(fp)
        assert np.allclose(m.toarray(), [[1.0, 0.0], [0.0, 2.0]])


@pytest.mark.xfail(reason="AttributeError: 'FortranFile' object has no attribute 'write'")
def test_fortran_file_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "t.dat")
        with sio.FortranFile(fp, "w") as f:
            f.write_record(np.array([1.0, 2.0, 3.0]))
        with sio.FortranFile(fp, "r") as f:
            out = f.read_reals(1)[0]
        assert np.allclose(out, [1.0, 2.0, 3.0])


def test_io_module_surface():
    # the public file-format entry points exist
    for name in ("loadmat", "savemat", "mmread", "mmwrite", "mminfo", "FortranFile"):
        assert hasattr(sio, name)
