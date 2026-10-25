"""scicore.stats — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.stats as sst

pytestmark = pytest.mark.filterwarnings("default")


def test_norm_rvs_deterministic():
    v1 = sst.norm.rvs(5, random_state=42)
    v2 = sst.norm.rvs(5, random_state=42)
    assert np.allclose(np.asarray(v1), np.asarray(v2))
    assert np.all(np.isfinite(np.asarray(v1, dtype=np.float64)))


def test_zscore():
    z = sst.zscore(np.array([1.0, 2.0, 3.0]))
    expected = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0 / 3.0)
    assert np.allclose(z, expected, rtol=1e-12)


def test_qmc_latin_hypercube_reproducible():
    s1 = sst.qmc.LatinHypercube(2, seed=1).random(4)
    s2 = sst.qmc.LatinHypercube(2, seed=1).random(4)
    assert np.allclose(s1, s2)
    assert np.all(s1 >= 0) and np.all(s1 < 1)


def test_norm_pdf_cdf():
    assert np.isclose(sst.norm.pdf(0.0), 1.0 / np.sqrt(2 * np.pi), rtol=1e-12)
    assert np.isclose(sst.norm.cdf(1.96), 0.9750021048517795, rtol=1e-9)
    assert np.isclose(sst.norm.mean(1.0, 2.0), 1.0)
    # norm.std(loc, scale) returns scale
    assert np.isclose(sst.norm.std(1.0, 4.0), 4.0)


def test_poisson_pmf():
    assert np.isclose(sst.poisson.pmf(2, 4.0), np.exp(-4.0) * 4.0**2 / 2.0)
    assert np.isclose(sst.poisson.pmf(1, 1.0), np.exp(-1.0))


def test_tmean():
    assert np.isclose(sst.tmean(np.array([1.0, 2.0, 3.0, 4.0])), 2.5)


def test_sem():
    # sem = std(ddof=1)/sqrt(n) = sqrt(5/3)/2
    assert np.isclose(sst.sem(np.array([1.0, 2.0, 3.0, 4.0])),
                      0.6454972243679028, rtol=1e-9)


def test_linregress():
    res = sst.linregress(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert np.isclose(res.slope, 2.0, rtol=1e-12)
    assert np.isclose(res.intercept, 0.0, atol=1e-12)


def test_ttest_ind():
    res = sst.ttest_ind(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0]))
    assert np.isclose(res.statistic, -1.0954451150103321, rtol=1e-9)


def test_chi2_contingency():
    chi2, p, dof, _ = sst.chi2_contingency(np.array([[10.0, 20.0], [20.0, 30.0]]))
    assert dof == 1
    assert chi2 > 0


def test_entropy():
    # sst.entropy defaults to base=e (nats); 1.0 is the base-2 (bits) value
    assert np.isclose(sst.entropy(np.array([0.5, 0.5]), base=2), 1.0, rtol=1e-12)


def test_chisquare():
    res = sst.chisquare(np.array([10.0, 20.0, 30.0, 40.0]))
    assert res.statistic > 0


def test_pearsonr():
    res = sst.pearsonr(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert np.isclose(res.statistic, 1.0, rtol=1e-12)


def test_skew():
    assert sst.skew(np.array([1.0, 2.0, 3.0, 4.0, 10.0])) > 0


def test_describe():
    res = sst.describe(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    # DescribeResult is a namedtuple: attribute access, not string indexing
    assert np.isclose(res.mean, 3.0)


def test_gmean():
    assert np.isclose(sst.gmean(np.array([1.0, 4.0])), 2.0)


def test_mode():
    res = sst.mode(np.array([1.0, 1.0, 2.0, 2.0, 2.0]))
    assert np.isclose(res.mode[0], 2.0)
