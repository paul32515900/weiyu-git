import math

import numpy as np
import pytest

from src.sampling import (
    beta_pdf,
    rejection_sampling_beta,
    inverse_transform_cubic_pdf,
    inverse_transform_exponential,
    bernoulli_via_inversion,
)


def test_beta_pdf_values_and_domain():
    # Inside domain
    xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = np.array([
        0.0,              # 6*0*(1-0)
        6*0.25*0.75,
        6*0.5*0.5,        # 1.5
        6*0.75*0.25,
        0.0,              # 6*1*(0)
    ])
    got = beta_pdf(xs)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    # Outside domain -> 0
    xs_out = np.array([-0.1, 1.1, 2.0])
    got_out = beta_pdf(xs_out)
    assert np.all(got_out == 0.0)


def test_rejection_sampling_beta_length_and_bounds():
    rng = np.random.default_rng(123)
    n = 2000
    samples = rejection_sampling_beta(n, rng=rng)
    assert samples.shape == (n,)
    assert np.all((samples > 0.0) & (samples < 1.0))


def test_inverse_transform_cubic_mean_close_to_theoretical():
    # Theoretical mean of f(x)=3x^2 on (0,1) is E[X] = 3/4 = 0.75
    rng = np.random.default_rng(321)
    n = 20000
    samples = inverse_transform_cubic_pdf(n, rng=rng)
    mean = samples.mean()
    assert abs(mean - 0.75) < 0.02  # tolerance for randomness


def test_inverse_transform_exponential_mean_and_nonnegativity():
    lam = 2.5
    rng = np.random.default_rng(44)
    n = 20000
    samples = inverse_transform_exponential(n, lam=lam, rng=rng)
    assert np.all(samples >= 0.0)
    # Theoretical mean is 1/lam
    assert abs(samples.mean() - (1.0 / lam)) < 0.02


def test_bernoulli_via_inversion_values_and_mean():
    p = 0.37
    rng = np.random.default_rng(96)
    n = 20000
    samples = bernoulli_via_inversion(n, p=p, rng=rng)
    assert set(np.unique(samples)).issubset({0, 1})
    assert abs(samples.mean() - p) < 0.02


def test_reproducibility_across_rng():
    n = 1000
    seed = 777

    rng1 = np.random.default_rng(seed)
    a1 = rejection_sampling_beta(n, rng=rng1)

    rng2 = np.random.default_rng(seed)
    a2 = rejection_sampling_beta(n, rng=rng2)

    np.testing.assert_allclose(a1, a2, rtol=0, atol=0)  # bitwise equal

    # Check for exponential and cubic too
    rng3 = np.random.default_rng(seed)
    rng4 = np.random.default_rng(seed)
    e1 = inverse_transform_exponential(n, lam=3.0, rng=rng3)
    e2 = inverse_transform_exponential(n, lam=3.0, rng=rng4)
    np.testing.assert_allclose(e1, e2, rtol=0, atol=0)

    rng5 = np.random.default_rng(seed)
    rng6 = np.random.default_rng(seed)
    c1 = inverse_transform_cubic_pdf(n, rng=rng5)
    c2 = inverse_transform_cubic_pdf(n, rng=rng6)
    np.testing.assert_allclose(c1, c2, rtol=0, atol=0)


def test_beta_pdf_scalar_and_array_types():
    # Scalar input returns scalar
    assert isinstance(beta_pdf(0.5), (int, float, np.floating))
    # Array input returns ndarray with same shape
    xs = np.linspace(-0.5, 1.5, 11)
    ys = beta_pdf(xs)
    assert isinstance(ys, np.ndarray)
    assert ys.shape == xs.shape
    # Check that values outside [0,1] are zero and inside follow formula
    mask = (xs >= 0) & (xs <= 1)
    np.testing.assert_allclose(ys[~mask], 0.0)
    np.testing.assert_allclose(ys[mask], 6 * xs[mask] * (1 - xs[mask]))


def test_rejection_sampling_beta_empirical_pdf_shape():
    # Empirical histogram should roughly follow the Beta(2,2) shape (symmetric around 0.5)
    rng = np.random.default_rng(2024)
    n = 60_000
    s = rejection_sampling_beta(n, rng=rng)
    hist, edges = np.histogram(s, bins=20, range=(0, 1), density=True)
    left = hist[: len(hist)//2].mean()
    right = hist[len(hist)//2 :].mean()
    assert abs(left - right) < 0.15
    # Peak should be around middle bin
    peak_bin = np.argmax(hist)
    mid_bin = len(hist)//2
    assert abs(peak_bin - mid_bin) <= 2


def test_inverse_transform_cubic_distribution_properties():
    rng = np.random.default_rng(135)
    n = 50_000
    s = inverse_transform_cubic_pdf(n, rng=rng)
    # Support in (0,1)
    assert np.all((s >= 0) & (s <= 1))
    # Theoretical variance of f(x)=3x^2 is Var = E[X^2]-E[X]^2, where E[X]=3/4, E[X^2]=3/5
    var_theory = 3/5 - (3/4)**2
    assert abs(s.var() - var_theory) < 0.02


def test_inverse_transform_exponential_quantiles():
    lam = 1.3
    rng = np.random.default_rng(246)
    n = 60_000
    s = inverse_transform_exponential(n, lam=lam, rng=rng)
    # Median for Exp(lam) is ln(2)/lam
    median_theory = math.log(2) / lam
    assert abs(np.median(s) - median_theory) < 0.03


def test_bernoulli_via_inversion_edge_probabilities_and_dtype():
    rng = np.random.default_rng(864)
    n = 10_000
    # p=0 -> all zeros
    z = bernoulli_via_inversion(n, p=0.0, rng=rng)
    assert np.all(z == 0)
    # p=1 -> all ones
    o = bernoulli_via_inversion(n, p=1.0, rng=rng)
    assert np.all(o == 1)
    # dtype should be integer-ish
    assert np.issubdtype(z.dtype, np.integer)


@pytest.mark.parametrize(
    "n,lam",
    [(-1, 1.0), (0, 1.0), (10, 0.0), (10, -3.0)],
)
def test_invalid_params_exponential(n, lam):
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        inverse_transform_exponential(n, lam=lam, rng=rng)


@pytest.mark.parametrize("n,p", [(-1, 0.1), (0, 0.1), (10, -0.1), (10, 1.1)])
def test_invalid_params_bernoulli(n, p):
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        bernoulli_via_inversion(n, p=p, rng=rng)


def test_invalid_params_rejection_beta():
    with pytest.raises(ValueError):
        rejection_sampling_beta(0)
    with pytest.raises(ValueError):
        rejection_sampling_beta(-5)


@pytest.mark.timeout(5)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.parametrize("n", [10_000, 50_000])
def test_performance_sanity(n):
    # Not a strict perf test; ensures functions run reasonably fast for larger n.
    rng = np.random.default_rng(12345)
    _ = inverse_transform_cubic_pdf(n, rng=rng)
    _ = inverse_transform_exponential(n, lam=1.7, rng=rng)
    _ = bernoulli_via_inversion(n, p=0.41, rng=rng)

