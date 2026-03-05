from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, int, np.ndarray]


def _asarray(x: ArrayLike) -> np.ndarray:
    """Convert input to numpy array without copying when possible."""
    return np.asarray(x)


def _return_same_type(original: ArrayLike, arr: np.ndarray) -> ArrayLike:
    """Return a Python scalar if original was scalar, else np.ndarray.

    Parameters
    ----------
    original: ArrayLike
        The original input value used to determine the return type.
    arr: np.ndarray
        The computed array result to convert if needed.
    """
    if np.isscalar(original):
        return arr.item()
    return arr


def beta_pdf(x: ArrayLike) -> ArrayLike:
    """Beta(2,2) probability density function evaluated at x.

    f(x) = 6 * x * (1 - x) for x in [0, 1], else 0.

    Accepts scalars or numpy arrays and returns the same type.
    """
    xa = _asarray(x)
    y = np.zeros_like(xa, dtype=float)
    mask = (xa >= 0.0) & (xa <= 1.0)
    y[mask] = 6.0 * xa[mask] * (1.0 - xa[mask])
    return _return_same_type(x, y)


def rejection_sampling_beta(n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate Beta(2,2) samples via rejection sampling using Uniform(0,1) proposal.

    The target density is f(x) = 6*x*(1-x) on (0,1), with maximum f(0.5) = 1.5.
    We use M = 1.5 with Uniform(0,1) proposal (density = 1), so acceptance probability is
    f(y)/M = 4*y*(1-y).

    Parameters
    ----------
    n_samples : int
        Number of samples to generate (> 0).
    rng : numpy.random.Generator, optional
        RNG to use for reproducibility. If None, a default generator is created.

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if rng is None:
        rng = np.random.default_rng()

    M = 1.5
    out = np.empty(n_samples, dtype=float)
    filled = 0
    # Vectorized batch sampling to be efficient
    while filled < n_samples:
        remaining = n_samples - filled
        # Slight oversampling factor to reduce loops; bounded for efficiency
        batch = max(remaining * 2, 1024)
        y = rng.random(batch)
        u = rng.random(batch)
        accept_prob = 4.0 * y * (1.0 - y)
        acc = u <= accept_prob
        n_acc = int(np.count_nonzero(acc))
        if n_acc:
            take = min(n_acc, remaining)
            out[filled:filled + take] = y[acc][:take]
            filled += take
    return out


def inverse_transform_cubic_pdf(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Inverse-transform sampling for density f(x) = 3*x^2 on (0,1).

    The CDF is F(x) = x^3, so X = U^{1/3} for U ~ Uniform(0,1).

    Parameters
    ----------
    n : int
        Number of samples to generate (> 0).
    rng : numpy.random.Generator, optional
        RNG to use for reproducibility. If None, a default generator is created.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    return np.power(u, 1.0 / 3.0)


def inverse_transform_exponential(n: int, lam: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Inverse-transform sampling for Exponential(rate=lam).

    X = -log(U) / lam for U ~ Uniform(0,1), lam > 0.

    Parameters
    ----------
    n : int
        Number of samples to generate (> 0).
    lam : float
        Rate parameter (> 0).
    rng : numpy.random.Generator, optional
        RNG to use for reproducibility. If None, a default generator is created.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if lam <= 0:
        raise ValueError("lam must be positive")
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    return -np.log(u) / float(lam)


def bernoulli_via_inversion(n: int, p: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate Bernoulli(p) samples using inversion method.

    Parameters
    ----------
    n : int
        Number of samples to generate (> 0).
    p : float
        Success probability in [0, 1].
    rng : numpy.random.Generator, optional
        RNG to use for reproducibility. If None, a default generator is created.

    Returns
    -------
    np.ndarray of shape (n,), with values 0 or 1.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    return (u <= float(p)).astype(int)
