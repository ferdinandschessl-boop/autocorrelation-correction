"""
Block bootstrap for clustered conversational data.

Standard bootstrap resamples individual observations, which breaks the
temporal structure within conversations. Block bootstrap resamples entire
conversations (clusters), preserving intra-conversation autocorrelation.

The procedure:
    1. Pool all turns across conversations, recording which conversation
       each turn belongs to.
    2. For each bootstrap iteration, resample conversations with replacement.
    3. Pool all turns from the resampled conversations.
    4. Compute the test statistic (point-biserial r) on the pooled sample.
    5. The bootstrap p-value is the proportion of iterations where the
       correlation has opposite sign to the observed value (two-sided).

The combined test takes the maximum (most conservative) of the bootstrap
p-value and the Chelton-corrected p-value.

Example:
    >>> from src.block_bootstrap import combined_test
    >>> conversations = {
    ...     'chat_1': {'binary': [0,0,0,1,1], 'metric': [0.1,0.2,0.15,0.8,0.9]},
    ...     'chat_2': {'binary': [0,0,0,0,0], 'metric': [0.1,0.1,0.2,0.15,0.1]},
    ... }
    >>> result = combined_test(conversations, n_boot=500)
    >>> print(f"r={result['r_obs']:.3f}, p_final={result['p_final']:.4f}")
"""

import math
import numpy as np
from scipy.stats import pointbiserialr

from .chelton_correction import autocorr_lag1, effective_n, corrected_p_value


def block_bootstrap_correlation(conversations, n_boot=2000, seed=42):
    """Block bootstrap test for point-biserial correlation.

    Resamples entire conversations (not individual turns) to preserve
    within-conversation autocorrelation structure.

    Parameters
    ----------
    conversations : dict
        Dictionary mapping conversation IDs to dicts with keys:
        - 'binary': list/array of binary labels (0/1) per turn
        - 'metric': list/array of metric values per turn
        NaN values in 'metric' are excluded automatically.
    n_boot : int, optional
        Number of bootstrap iterations (default: 2000).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'r_obs': observed point-biserial correlation
        - 'p_naive': naive (uncorrected) p-value
        - 'p_boot': bootstrap p-value (two-sided)
        - 'boot_se': bootstrap standard error of r
        - 'boot_ci_95': 95% confidence interval (2.5th, 97.5th percentile)
        - 'n_total': total number of valid turn pairs
        - 'n_conversations': number of conversations
        - 'mean_autocorr': mean lag-1 autocorrelation across conversations

    Raises
    ------
    ValueError
        If fewer than 2 conversations or fewer than 20 valid observations.
    """
    rng = np.random.default_rng(seed)
    chat_ids = list(conversations.keys())
    n_chats = len(chat_ids)

    if n_chats < 2:
        raise ValueError(f"Need at least 2 conversations, got {n_chats}")

    # Pool data and compute autocorrelations
    all_binary = []
    all_metric = []
    autocorrelations = []

    for cid in chat_ids:
        conv = conversations[cid]
        b = np.asarray(conv['binary'], dtype=float)
        m = np.asarray(conv['metric'], dtype=float)

        # Filter NaN
        mask = ~np.isnan(b) & ~np.isnan(m)
        b = b[mask]
        m = m[mask]

        if len(b) > 0:
            all_binary.extend(b.tolist())
            all_metric.extend(m.tolist())
            autocorrelations.append(autocorr_lag1(m))

    all_binary = np.array(all_binary)
    all_metric = np.array(all_metric)
    n_total = len(all_binary)

    if n_total < 20:
        raise ValueError(f"Need at least 20 valid observations, got {n_total}")

    if np.std(all_binary) == 0 or np.std(all_metric) == 0:
        raise ValueError("Zero variance in binary labels or metric values")

    # Observed correlation
    r_obs, p_naive = pointbiserialr(all_binary, all_metric)

    # Block bootstrap
    boot_rs = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(chat_ids, size=n_chats, replace=True)
        boot_b = []
        boot_m = []
        for cid in sampled_ids:
            conv = conversations[cid]
            b = np.asarray(conv['binary'], dtype=float)
            m = np.asarray(conv['metric'], dtype=float)
            mask = ~np.isnan(b) & ~np.isnan(m)
            boot_b.extend(b[mask].tolist())
            boot_m.extend(m[mask].tolist())

        if len(boot_b) < 20:
            continue

        bba = np.array(boot_b)
        bma = np.array(boot_m)
        if np.std(bba) == 0 or np.std(bma) == 0:
            boot_rs.append(0.0)
            continue

        br, _ = pointbiserialr(bba, bma)
        boot_rs.append(br)

    boot_rs = np.array(boot_rs)
    if len(boot_rs) < 100:
        raise ValueError(f"Too few successful bootstrap iterations: {len(boot_rs)}")

    boot_se = float(np.std(boot_rs, ddof=1))

    # Bootstrap p-value: proportion where sign flips (two-sided)
    if r_obs > 0:
        p_boot = float(np.mean(boot_rs <= 0) * 2)
    else:
        p_boot = float(np.mean(boot_rs >= 0) * 2)
    p_boot = min(1.0, p_boot)

    # Confidence interval
    ci_lo = float(np.percentile(boot_rs, 2.5))
    ci_hi = float(np.percentile(boot_rs, 97.5))

    mean_ac = float(np.mean([ac for ac in autocorrelations if not math.isnan(ac)])) if autocorrelations else 0.0

    return {
        'r_obs': float(r_obs),
        'p_naive': float(p_naive),
        'p_boot': p_boot,
        'boot_se': boot_se,
        'boot_ci_95': (ci_lo, ci_hi),
        'n_total': n_total,
        'n_conversations': n_chats,
        'mean_autocorr': mean_ac,
    }


def combined_test(conversations, n_boot=2000, seed=42):
    """Combined Chelton + block bootstrap test.

    Runs both corrections and takes the most conservative (highest) p-value.
    A test is considered cluster-robust only if BOTH corrections agree on
    significance at alpha=0.05.

    Parameters
    ----------
    conversations : dict
        Same format as block_bootstrap_correlation.
    n_boot : int, optional
        Number of bootstrap iterations (default: 2000).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    dict
        All keys from block_bootstrap_correlation, plus:
        - 'n_eff': effective sample size (Chelton)
        - 'p_corrected': Chelton-corrected p-value
        - 'p_final': max(p_boot, p_corrected) -- most conservative
        - 'significant': True if p_final < 0.05
        - 'status': 'ROBUST' or 'weak'
    """
    boot_result = block_bootstrap_correlation(conversations, n_boot=n_boot, seed=seed)

    # Chelton correction
    n_eff = effective_n(
        boot_result['n_total'],
        boot_result['mean_autocorr'],
        n_min=boot_result['n_conversations'],
    )
    p_corrected = corrected_p_value(boot_result['r_obs'], n_eff)

    # Conservative: take the worst p-value
    p_final = max(boot_result['p_boot'], p_corrected)

    result = dict(boot_result)
    result.update({
        'n_eff': n_eff,
        'p_corrected': p_corrected,
        'p_final': p_final,
        'significant': p_final < 0.05,
        'status': 'ROBUST' if p_final < 0.05 else 'weak',
    })
    return result
