"""scicore.ndimage — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.ndimage as ndi

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_sum():
    assert np.isclose(ndi.sum(np.ones((2, 2))), 4.0)


# ---------------- not yet ported / broken ----------------

def test_gaussian_filter():
    out = ndi.gaussian_filter(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), 1.0)
    # kernel mass concentrates at the center
    assert out[1, 1] == out.max()


def test_sobel():
    # constant image has zero Sobel response
    assert np.allclose(ndi.sobel(np.ones((3, 3))), 0.0)


def test_median_filter():
    # scipy requires size or footprint; majority-1s image has median 1.0
    img = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    assert ndi.median_filter(img, size=3)[1, 1] == 1.0


def test_uniform_filter():
    out = ndi.uniform_filter(np.ones((4, 4)), size=2)
    assert np.isclose(out[1, 1], 1.0)


def test_label():
    labels, n = ndi.label(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
    assert n == 2
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 0] != labels[2, 2]


def test_center_of_mass():
    img = np.array([[0.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(ndi.center_of_mass(img), [1.0, 1.0])


def test_binary_erosion():
    # 3x3 ones eroded by 3x3 structure leaves the center 1
    assert ndi.binary_erosion(np.ones((3, 3), dtype=int))[1, 1] == 1


def test_binary_dilation():
    img = np.zeros((3, 3), dtype=int)
    img[1, 1] = 1
    assert np.all(ndi.binary_dilation(img, structure=np.ones((3, 3), dtype=int)))


def test_morphological_gradient():
    # scipy requires one of size/footprint/structure
    out = ndi.morphological_gradient(np.ones((3, 3), dtype=int), size=(3, 3))
    # interior of a uniform block has zero gradient
    assert out[1, 1] == 0


def test_zoom():
    out = ndi.zoom(np.array([[1.0]]), 2.0)
    assert out.shape == (2, 2)


def test_shift():
    out = ndi.shift(np.array([[1.0, 0.0]]), [0.0, 1.0])
    assert np.isclose(out[0, 1], 1.0)


def test_rotate():
    out = ndi.rotate(np.array([[1.0, 0.0], [0.0, 0.0]]), 90, order=0)
    assert out.shape == (2, 2)


@pytest.mark.xfail(reason="AttributeError: 'ArrayFlags' object has no attribute 'contiguous'")
def test_distance_transform_bf():
    mask = 1 - np.array([[1.0, 1.0], [1.0, 1.0]])  # all background
    out = ndi.distance_transform_bf(mask)
    assert out.shape == (2, 2)


def test_find_objects():
    # two isolated singletons with DIFFERENT labels -> two objects
    out = ndi.find_objects(np.array([[1, 0], [0, 2]]))
    assert len(out) == 2
