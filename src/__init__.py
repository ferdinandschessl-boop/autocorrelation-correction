"""
Autocorrelation Correction for Turn-Level Conversational Metrics.

This package provides tools for detecting and correcting autocorrelation-induced
inflation in turn-level conversational analysis. Standard statistical tests
(point-biserial r, Spearman rho) applied to sequential turn-level data produce
inflated significance because consecutive turns within a conversation are not
independent observations.

Methods implemented:
    1. Chelton (1983) effective degrees of freedom correction
    2. Block bootstrap with conversations as clusters
    3. Inflation rate analysis
    4. Permutation testing for global significance

Reference:
    Schessl, F. (2026). The Autocorrelation Blind Spot: Why 43% of Turn-Level
    Findings in LLM Conversation Analysis May Be Spurious. [arXiv:XXXX.XXXXX]
"""

from .chelton_correction import (
    autocorr_lag1,
    effective_n,
    corrected_p_value,
    chelton_correct,
)

from .block_bootstrap import (
    block_bootstrap_correlation,
    combined_test,
)

from .inflation_analysis import (
    inflation_rate,
    compare_naive_vs_robust,
)

from .synthetic_data import (
    generate_ar1_series,
    generate_synthetic_conversation,
    generate_synthetic_dataset,
)

__version__ = "0.1.0"
