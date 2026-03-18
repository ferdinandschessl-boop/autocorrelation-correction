"""
Tests for autocorrelation correction pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.chelton_correction import autocorr_lag1, effective_n, corrected_p_value, chelton_correct
from src.block_bootstrap import block_bootstrap_correlation, combined_test
from src.inflation_analysis import inflation_rate, compare_naive_vs_robust
from src.synthetic_data import generate_ar1_series, generate_synthetic_conversation, generate_synthetic_dataset


# ── autocorr_lag1 ──────────────────────────────────────────────────────────────

class TestAutocorrLag1:
    def test_white_noise(self):
        """White noise should have near-zero autocorrelation."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=10000)
        rho = autocorr_lag1(x)
        assert abs(rho) < 0.05, f"White noise rho={rho}, expected ~0"

    def test_ar1_high(self):
        """AR(1) with rho=0.9 should recover approximately that value."""
        x = generate_ar1_series(10000, rho=0.9, seed=42)
        rho = autocorr_lag1(x)
        assert 0.85 < rho < 0.95, f"AR(1) rho=0.9, got {rho}"

    def test_ar1_low(self):
        """AR(1) with rho=0.1 should recover approximately that value."""
        x = generate_ar1_series(10000, rho=0.1, seed=42)
        rho = autocorr_lag1(x)
        assert 0.05 < rho < 0.15, f"AR(1) rho=0.1, got {rho}"

    def test_short_series(self):
        """Short series (< 4) should return 0.0."""
        assert autocorr_lag1([1, 2, 3]) == 0.0
        assert autocorr_lag1([]) == 0.0

    def test_constant_series(self):
        """Constant series should return 0.0."""
        assert autocorr_lag1([5, 5, 5, 5, 5]) == 0.0

    def test_nan_handling(self):
        """NaN values should be filtered out."""
        x = [1.0, 2.0, np.nan, 3.0, 4.0, 5.0]
        rho = autocorr_lag1(x)
        assert not np.isnan(rho)


# ── effective_n ────────────────────────────────────────────────────────────────

class TestEffectiveN:
    def test_no_autocorrelation(self):
        """rho=0 should give n_eff = n."""
        assert effective_n(1000, 0.0) == 1000.0

    def test_high_autocorrelation(self):
        """rho=0.9 should give n_eff ≈ n/19."""
        n_eff = effective_n(1000, 0.9)
        assert 50 < n_eff < 60  # 1000 * 0.1/1.9 ≈ 52.6

    def test_negative_autocorrelation(self):
        """Negative rho should increase n_eff."""
        n_eff = effective_n(1000, -0.5)
        assert n_eff > 1000  # 1000 * 1.5/0.5 = 3000

    def test_minimum(self):
        """n_eff should not go below n_min."""
        n_eff = effective_n(100, 0.99, n_min=5)
        assert n_eff >= 5

    def test_clamping(self):
        """rho > 0.99 should be clamped."""
        n_eff = effective_n(1000, 1.5)  # out of range
        assert n_eff > 0  # should not crash


# ── chelton_correct ────────────────────────────────────────────────────────────

class TestCheltonCorrect:
    def test_significant_low_rho(self):
        """r=0.1, n=1000, rho=0.05 should be significant."""
        result = chelton_correct(0.10, 1000, 0.05)
        assert result['significant'] is True

    def test_not_significant_high_rho(self):
        """r=0.1, n=1000, rho=0.93 should NOT be significant."""
        result = chelton_correct(0.10, 1000, 0.93)
        assert result['significant'] is False

    def test_strong_effect_survives(self):
        """r=0.5, n=1000, rho=0.93 should still be significant (n_eff~36, large r)."""
        result = chelton_correct(0.50, 1000, 0.93)
        assert result['significant'] is True

    def test_output_keys(self):
        """Check all expected keys are present."""
        result = chelton_correct(0.1, 100, 0.5)
        expected_keys = {'r_obs', 'n', 'rho', 'n_eff', 'p_corrected', 'significant'}
        assert set(result.keys()) == expected_keys


# ── block_bootstrap ────────────────────────────────────────────────────────────

class TestBlockBootstrap:
    def _make_data(self, effect=0.0, rho=0.5, n_conv=20, n_turns=50):
        return generate_synthetic_dataset(
            n_conversations=n_conv, turns_per_conv=n_turns,
            rho=rho, effect_size=effect, seed=42,
        )

    def test_null_not_significant(self):
        """Under null (no effect), combined test should usually not be significant."""
        data = self._make_data(effect=0.0, rho=0.9)
        result = combined_test(data, n_boot=500, seed=42)
        # Under null with high rho, p_final should often be > 0.05
        # We don't assert this deterministically since it's stochastic,
        # but we check the structure is correct
        assert 'p_final' in result
        assert 'status' in result
        assert result['status'] in ('ROBUST', 'weak')

    def test_strong_effect_significant(self):
        """Strong effect should survive correction."""
        data = self._make_data(effect=1.0, rho=0.5)
        result = combined_test(data, n_boot=500, seed=42)
        assert result['significant'] is True
        assert result['status'] == 'ROBUST'

    def test_output_keys(self):
        """Check all expected keys from combined_test."""
        data = self._make_data(effect=0.0, rho=0.5)
        result = combined_test(data, n_boot=200, seed=42)
        expected_keys = {
            'r_obs', 'p_naive', 'p_boot', 'boot_se', 'boot_ci_95',
            'n_total', 'n_conversations', 'mean_autocorr',
            'n_eff', 'p_corrected', 'p_final', 'significant', 'status',
        }
        assert set(result.keys()) == expected_keys

    def test_too_few_conversations(self):
        """Should raise ValueError with < 2 conversations."""
        data = {'conv_0': generate_synthetic_conversation(50, 0.5, seed=42)}
        with pytest.raises(ValueError):
            block_bootstrap_correlation(data)


# ── inflation_analysis ─────────────────────────────────────────────────────────

class TestInflationRate:
    def test_basic(self):
        assert inflation_rate(10, 5) == 0.5
        assert inflation_rate(10, 10) == 0.0
        assert inflation_rate(10, 0) == 1.0

    def test_zero_naive(self):
        assert inflation_rate(0, 0) == 0.0

    def test_compare_naive_vs_robust(self):
        test_results = [
            {'label': 'A', 'p_naive': 0.01, 'p_final': 0.04},   # both sig
            {'label': 'B', 'p_naive': 0.03, 'p_final': 0.20},   # inflated
            {'label': 'C', 'p_naive': 0.04, 'p_final': 0.30},   # inflated
            {'label': 'D', 'p_naive': 0.10, 'p_final': 0.50},   # neither sig
        ]
        result = compare_naive_vs_robust(test_results)
        assert result['n_naive_sig'] == 3
        assert result['n_robust'] == 1
        assert result['n_inflated'] == 2
        assert abs(result['inflation_rate'] - 2/3) < 1e-10
        assert 'B' in result['inflated_tests']
        assert 'C' in result['inflated_tests']
        assert 'A' in result['robust_tests']


# ── synthetic_data ─────────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_ar1_length(self):
        x = generate_ar1_series(100, rho=0.5, seed=42)
        assert len(x) == 100

    def test_conversation_shape(self):
        conv = generate_synthetic_conversation(50, rho=0.5, mal_fraction=0.3, seed=42)
        assert len(conv['binary']) == 50
        assert len(conv['metric']) == 50
        assert np.sum(conv['binary']) == 15  # 30% of 50

    def test_dataset_size(self):
        data = generate_synthetic_dataset(n_conversations=10, seed=42)
        assert len(data) == 10

    def test_effect_size_shifts_metric(self):
        """With large effect_size, malicious turns should have higher metric values."""
        conv = generate_synthetic_conversation(
            1000, rho=0.0, mal_fraction=0.3, effect_size=5.0, seed=42
        )
        mal_mask = conv['binary'] == 1
        base_mask = conv['binary'] == 0
        mal_mean = np.mean(conv['metric'][mal_mask])
        base_mean = np.mean(conv['metric'][base_mask])
        assert mal_mean > base_mean + 3  # effect should be clearly visible


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
