"""
Chelton (1983) correction for effective degrees of freedom.

When observations in a time series are autocorrelated, the effective number of
independent observations is smaller than the nominal sample size. Chelton (1983)
proposed adjusting the degrees of freedom as:

    n_eff = n * (1 - rho) / (1 + rho)

where rho is the lag-1 autocorrelation coefficient of the metric time series.

This correction is then used to compute a corrected p-value via the t-distribution
with df = n_eff - 2.

Reference:
    Chelton, D. B. (1983). Effects of sampling errors in statistical estimation.
    Deep Sea Research Part A, 30(10), 1083-1103.

Example:
    >>> from src.chelton_correction import chelton_correct
    >>> result = chelton_correct(r_obs=0.15, n=500, rho=0.9)
    >>> print(f"n_eff={result['n_eff']:.0f}, p={result['p_corrected']:.4f}")
    n_eff=26, p=0.4627
"""

import math
import numpy as np
from scipy.stats import t as t_dist


def autocorr_lag1(x):
    """Compute lag-1 autocorrelation for a time series.

    Parameters
    ----------
    x : array-like
        Time series of metric values (one per turn).

    Returns
    -------
    float
        Lag-1 autocorrelation coefficient. Returns 0.0 if the series is
        too short (< 4 observations) or has zero variance.

    Examples
    --------
    >>> autocorr_lag1([1, 2, 3, 4, 5])  # trending series
    0.4
    >>> autocorr_lag1([1, -1, 1, -1])  # alternating series
    -1.0
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    x = x[mask]
    if len(x) < 4:
        return 0.0
    mu = np.mean(x)
    var = np.var(x, ddof=0)
    if var == 0:
        return 0.0
    return float(np.mean((x[:-1] - mu) * (x[1:] - mu)) / var)


def effective_n(n, rho, n_min=2):
    """Compute effective sample size using Chelton (1983) formula.

    Parameters
    ----------
    n : int
        Nominal sample size (number of observations).
    rho : float
        Lag-1 autocorrelation coefficient. Clamped to [-0.99, 0.99].
    n_min : int, optional
        Minimum effective sample size (default: 2).

    Returns
    -------
    float
        Effective sample size, always >= n_min.

    Examples
    --------
    >>> effective_n(1000, rho=0.0)  # no autocorrelation
    1000.0
    >>> effective_n(1000, rho=0.9)  # high autocorrelation
    52.631...
    >>> effective_n(1000, rho=0.95)  # very high
    25.641...
    """
    rho = max(-0.99, min(0.99, rho))
    n_eff = n * (1 - rho) / (1 + rho)
    return max(float(n_min), n_eff)


def corrected_p_value(r_obs, n_eff):
    """Compute two-sided p-value using t-distribution with n_eff degrees of freedom.

    Parameters
    ----------
    r_obs : float
        Observed correlation coefficient.
    n_eff : float
        Effective sample size (from Chelton correction).

    Returns
    -------
    float
        Two-sided p-value.
    """
    df = max(2, n_eff - 2)
    denom = 1 - r_obs ** 2
    if denom <= 0:
        denom = 1e-12
    t_stat = r_obs * math.sqrt(df / denom)
    return float(2 * t_dist.sf(abs(t_stat), df=df))


def chelton_correct(r_obs, n, rho, n_min=2):
    """Full Chelton correction pipeline: compute n_eff and corrected p-value.

    Given an observed correlation `r_obs` from `n` observations with lag-1
    autocorrelation `rho`, compute the effective sample size and a corrected
    p-value that accounts for the reduced degrees of freedom.

    Parameters
    ----------
    r_obs : float
        Observed correlation coefficient (e.g., point-biserial r).
    n : int
        Nominal sample size.
    rho : float
        Lag-1 autocorrelation of the metric time series.
    n_min : int, optional
        Minimum effective sample size (default: 2).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'r_obs': the input correlation
        - 'n': nominal sample size
        - 'rho': lag-1 autocorrelation
        - 'n_eff': effective sample size
        - 'p_corrected': corrected two-sided p-value
        - 'significant': bool, True if p_corrected < 0.05

    Examples
    --------
    >>> result = chelton_correct(r_obs=0.10, n=1000, rho=0.05)
    >>> result['significant']
    True
    >>> result = chelton_correct(r_obs=0.10, n=1000, rho=0.93)
    >>> result['significant']
    False
    """
    n_eff_val = effective_n(n, rho, n_min=n_min)
    p_val = corrected_p_value(r_obs, n_eff_val)
    return {
        'r_obs': r_obs,
        'n': n,
        'rho': rho,
        'n_eff': n_eff_val,
        'p_corrected': p_val,
        'significant': p_val < 0.05,
    }
