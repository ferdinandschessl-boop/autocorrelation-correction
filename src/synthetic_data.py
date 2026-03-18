"""
Synthetic data generators for testing autocorrelation correction methods.

Generates AR(1) time series with known autocorrelation structure, grouped
into synthetic conversations with binary labels. This allows controlled
experiments to verify that:

    1. Naive tests produce inflated significance on autocorrelated data
    2. The Chelton + bootstrap correction properly controls false positive rates
    3. True effects are still detectable after correction

Example:
    >>> from src.synthetic_data import generate_synthetic_dataset
    >>> dataset = generate_synthetic_dataset(
    ...     n_conversations=50, turns_per_conv=80, rho=0.9,
    ...     effect_size=0.0, seed=42
    ... )
    >>> # With effect_size=0.0, no true association exists
    >>> # Naive tests will still find "significant" correlations
"""

import numpy as np


def generate_ar1_series(n, rho, sigma=1.0, seed=None):
    """Generate an AR(1) time series with specified lag-1 autocorrelation.

    The process is:
        x[0] = eps[0]
        x[t] = rho * x[t-1] + eps[t]

    where eps ~ N(0, sigma^2 * (1 - rho^2)) so that the marginal variance
    of x is sigma^2.

    Parameters
    ----------
    n : int
        Length of the time series.
    rho : float
        Lag-1 autocorrelation coefficient, in (-1, 1).
    sigma : float, optional
        Marginal standard deviation (default: 1.0).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        AR(1) time series of length n.

    Examples
    --------
    >>> x = generate_ar1_series(1000, rho=0.9, seed=42)
    >>> # Empirical lag-1 autocorrelation should be close to 0.9
    """
    rng = np.random.default_rng(seed)
    innovation_std = sigma * np.sqrt(max(1e-10, 1 - rho ** 2))
    eps = rng.normal(0, innovation_std, size=n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def generate_synthetic_conversation(n_turns, rho, mal_fraction=0.0,
                                    effect_size=0.0, seed=None):
    """Generate a single synthetic conversation with AR(1) metric and binary labels.

    Parameters
    ----------
    n_turns : int
        Number of turns in the conversation.
    rho : float
        Lag-1 autocorrelation of the metric.
    mal_fraction : float, optional
        Fraction of turns labeled as 1 (e.g., malicious). Default: 0.0.
    effect_size : float, optional
        Mean shift in metric for turns labeled 1. Default: 0.0 (null).
    seed : int or None, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'binary': np.ndarray of shape (n_turns,) with 0/1 labels
        - 'metric': np.ndarray of shape (n_turns,) with AR(1) values
        - 'rho_true': the target autocorrelation
    """
    rng = np.random.default_rng(seed)

    # Generate AR(1) metric
    metric = generate_ar1_series(n_turns, rho, sigma=1.0, seed=seed)

    # Generate binary labels
    binary = np.zeros(n_turns)
    if mal_fraction > 0:
        n_mal = max(1, int(n_turns * mal_fraction))
        mal_indices = rng.choice(n_turns, size=n_mal, replace=False)
        binary[mal_indices] = 1

        # Add effect: shift metric for malicious turns
        if effect_size != 0:
            metric[mal_indices] += effect_size

    return {
        'binary': binary,
        'metric': metric,
        'rho_true': rho,
    }


def generate_synthetic_dataset(n_conversations=50, turns_per_conv=80,
                                rho=0.9, mal_fraction=0.3,
                                effect_size=0.0, turns_std=20,
                                seed=42):
    """Generate a full synthetic dataset of multiple conversations.

    Creates a dataset suitable for testing the full correction pipeline:
    block bootstrap + Chelton correction.

    Parameters
    ----------
    n_conversations : int, optional
        Number of conversations (default: 50).
    turns_per_conv : int, optional
        Mean number of turns per conversation (default: 80).
    rho : float, optional
        Lag-1 autocorrelation of the metric (default: 0.9).
    mal_fraction : float, optional
        Fraction of turns labeled as 1 in each conversation (default: 0.3).
    effect_size : float, optional
        Mean shift for labeled turns. 0.0 = null hypothesis (default: 0.0).
    turns_std : int, optional
        Standard deviation of turns per conversation (default: 20).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    dict
        Dictionary mapping conversation IDs ('conv_0', 'conv_1', ...) to
        dicts with 'binary' and 'metric' arrays. Suitable for passing
        directly to block_bootstrap_correlation() or combined_test().

    Examples
    --------
    >>> # Null dataset (no true effect, high autocorrelation)
    >>> null_data = generate_synthetic_dataset(rho=0.9, effect_size=0.0)
    >>> len(null_data)
    50

    >>> # Dataset with true effect
    >>> effect_data = generate_synthetic_dataset(rho=0.9, effect_size=0.5)
    """
    rng = np.random.default_rng(seed)
    dataset = {}

    for i in range(n_conversations):
        # Vary conversation length
        n_turns = max(10, int(rng.normal(turns_per_conv, turns_std)))
        conv_seed = seed + i + 1 if seed is not None else None

        conv = generate_synthetic_conversation(
            n_turns=n_turns,
            rho=rho,
            mal_fraction=mal_fraction,
            effect_size=effect_size,
            seed=conv_seed,
        )

        dataset[f'conv_{i:03d}'] = conv

    return dataset
