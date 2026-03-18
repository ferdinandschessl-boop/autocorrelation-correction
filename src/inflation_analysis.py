"""
Inflation rate analysis for autocorrelation-induced false positives.

The inflation rate (IR) quantifies how many nominally significant findings
are spurious due to autocorrelation:

    IR = (n_naive_sig - n_robust) / n_naive_sig

where n_naive_sig is the number of tests significant at p<0.05 using standard
(uncorrected) methods, and n_robust is the number surviving cluster-robust
correction (block bootstrap + Chelton).

A permutation test shuffles binary labels within each conversation to establish
a null distribution of expected significant correlations under the null
hypothesis of no association. Comparing observed significant count to this
distribution yields a global p-value.

Example:
    >>> results = compare_naive_vs_robust(test_results)
    >>> print(f"IR = {results['inflation_rate']:.0%}")
    IR = 43%
"""

import numpy as np
from scipy.stats import pointbiserialr


def inflation_rate(n_naive_sig, n_robust):
    """Compute the inflation rate.

    Parameters
    ----------
    n_naive_sig : int
        Number of tests significant with naive (uncorrected) p-values.
    n_robust : int
        Number of tests significant after cluster-robust correction.

    Returns
    -------
    float
        Inflation rate, in [0, 1]. Returns 0.0 if n_naive_sig == 0.

    Examples
    --------
    >>> inflation_rate(56, 32)
    0.42857142857142855
    >>> inflation_rate(10, 10)
    0.0
    >>> inflation_rate(10, 0)
    1.0
    """
    if n_naive_sig == 0:
        return 0.0
    return (n_naive_sig - n_robust) / n_naive_sig


def compare_naive_vs_robust(test_results, alpha=0.05):
    """Compare naive vs. cluster-robust significance across multiple tests.

    Parameters
    ----------
    test_results : list of dict
        Each dict must have keys:
        - 'label': str, test name
        - 'p_naive': float, uncorrected p-value
        - 'p_final': float, cluster-robust p-value (from combined_test)

    alpha : float, optional
        Significance level (default: 0.05).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'n_total': total number of tests
        - 'n_naive_sig': number significant with naive p
        - 'n_robust': number significant with cluster-robust p
        - 'n_inflated': difference (naive_sig - robust)
        - 'inflation_rate': IR as a proportion
        - 'naive_sig_tests': list of test labels significant naively
        - 'robust_tests': list of test labels significant after correction
        - 'inflated_tests': tests significant naively but not after correction
    """
    naive_sig = [t for t in test_results if t['p_naive'] < alpha]
    robust = [t for t in test_results if t['p_final'] < alpha]

    naive_labels = set(t['label'] for t in naive_sig)
    robust_labels = set(t['label'] for t in robust)
    inflated_labels = naive_labels - robust_labels

    n_naive = len(naive_sig)
    n_robust = len(robust)

    return {
        'n_total': len(test_results),
        'n_naive_sig': n_naive,
        'n_robust': n_robust,
        'n_inflated': n_naive - n_robust,
        'inflation_rate': inflation_rate(n_naive, n_robust),
        'naive_sig_tests': sorted(naive_labels),
        'robust_tests': sorted(robust_labels),
        'inflated_tests': sorted(inflated_labels),
    }


def permutation_test(conversations, metric_keys, n_perm=1000, seed=42):
    """Permutation test: shuffle labels within conversations, count significant hits.

    For each permutation, binary labels (e.g., malicious/benign) are shuffled
    independently within each conversation, and the number of significant
    correlations (p < 0.05) across all (conversation, metric) pairs is counted.
    The observed count is compared to the permutation distribution to yield
    a global p-value.

    Parameters
    ----------
    conversations : dict
        Maps conversation IDs to dicts with keys:
        - 'binary': array-like of binary labels (0/1) per turn
        - 'metrics': dict mapping metric names to arrays of values per turn
    metric_keys : list of str
        Which metrics to test.
    n_perm : int, optional
        Number of permutations (default: 1000).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'observed_sig': number of significant correlations observed
        - 'perm_mean': mean significant correlations under null
        - 'perm_std': std of null distribution
        - 'perm_max': maximum under null
        - 'p_value': permutation p-value
        - 'ratio': observed / perm_mean (enrichment ratio)
        - 'significant': True if p_value < 0.05
    """
    rng = np.random.default_rng(seed)

    def count_sig_one_conv(binary, metrics_dict):
        n_sig = 0
        for key in metric_keys:
            if key not in metrics_dict:
                continue
            vals = np.asarray(metrics_dict[key], dtype=float)
            mask = ~np.isnan(binary) & ~np.isnan(vals)
            if np.sum(mask) < 5:
                continue
            b, m = binary[mask], vals[mask]
            if np.std(b) == 0 or np.std(m) == 0:
                continue
            _, p = pointbiserialr(b, m)
            if p < 0.05:
                n_sig += 1
        return n_sig

    # Count observed significant correlations
    observed_total = 0
    for cid, conv in conversations.items():
        binary = np.asarray(conv['binary'], dtype=float)
        observed_total += count_sig_one_conv(binary, conv['metrics'])

    # Permutation null distribution
    perm_counts = []
    for _ in range(n_perm):
        perm_total = 0
        for cid, conv in conversations.items():
            binary = np.asarray(conv['binary'], dtype=float).copy()
            rng.shuffle(binary)
            perm_total += count_sig_one_conv(binary, conv['metrics'])
        perm_counts.append(perm_total)

    perm_counts = np.array(perm_counts)
    p_perm = float(np.mean(perm_counts >= observed_total))
    perm_mean = float(np.mean(perm_counts))

    return {
        'observed_sig': observed_total,
        'perm_mean': perm_mean,
        'perm_std': float(np.std(perm_counts)),
        'perm_max': int(np.max(perm_counts)),
        'p_value': p_perm,
        'ratio': observed_total / perm_mean if perm_mean > 0 else float('inf'),
        'significant': p_perm < 0.05,
    }
