#!/usr/bin/env python3
"""
Demo: Autocorrelation correction for turn-level conversational metrics.

This script demonstrates the full correction pipeline on synthetic data,
showing how autocorrelation inflates naive p-values and how the Chelton +
block bootstrap correction controls false positive rates.

Run:
    python examples/demo_correction.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    autocorr_lag1,
    chelton_correct,
    block_bootstrap_correlation,
    combined_test,
    inflation_rate,
    generate_synthetic_dataset,
    generate_ar1_series,
)


def demo_ar1_autocorrelation():
    """Show how AR(1) processes have predictable autocorrelation."""
    print("=" * 60)
    print("  1. AR(1) Process: Autocorrelation Structure")
    print("=" * 60)
    print()

    for rho_target in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]:
        series = generate_ar1_series(5000, rho=rho_target, seed=42)
        rho_empirical = autocorr_lag1(series)
        print(f"  rho_target={rho_target:.2f}  ->  rho_empirical={rho_empirical:.3f}")

    print()


def demo_chelton_correction():
    """Show how Chelton correction deflates significance for autocorrelated metrics."""
    print("=" * 60)
    print("  2. Chelton Correction: Effect on p-values")
    print("=" * 60)
    print()

    r_obs = 0.10  # moderate observed correlation
    n = 1000      # typical conversation pooled sample

    print(f"  Observed r = {r_obs}, n = {n}")
    print(f"  {'rho':<6} {'n_eff':>8} {'p_corrected':>14} {'Significant':>12}")
    print("  " + "-" * 44)

    for rho in [0.0, 0.05, 0.2, 0.5, 0.7, 0.9, 0.93, 0.95]:
        result = chelton_correct(r_obs, n, rho)
        sig = "YES" if result['significant'] else "no"
        print(f"  {rho:<6.2f} {result['n_eff']:>8.1f} {result['p_corrected']:>14.6f} {sig:>12}")

    print()
    print("  -> Higher autocorrelation = fewer effective observations = weaker evidence")
    print()


def demo_null_hypothesis():
    """Show inflation under null hypothesis (no true effect)."""
    print("=" * 60)
    print("  3. Null Hypothesis: False Positive Inflation")
    print("=" * 60)
    print()

    n_experiments = 20
    naive_sig_count = 0
    robust_sig_count = 0

    print(f"  Running {n_experiments} experiments with rho=0.9, effect_size=0.0 (NULL)...")
    print(f"  {'Exp':<5} {'r_obs':>7} {'p_naive':>10} {'p_boot':>10} {'p_chelton':>10} {'p_final':>10} {'Status':>8}")
    print("  " + "-" * 62)

    for i in range(n_experiments):
        data = generate_synthetic_dataset(
            n_conversations=30, turns_per_conv=60,
            rho=0.9, effect_size=0.0,
            seed=i * 100,
        )

        result = combined_test(data, n_boot=500, seed=i)

        naive_sig = result['p_naive'] < 0.05
        if naive_sig:
            naive_sig_count += 1
        if result['significant']:
            robust_sig_count += 1

        print(f"  {i+1:<5} {result['r_obs']:>+.4f} {result['p_naive']:>10.4f} "
              f"{result['p_boot']:>10.4f} {result['p_corrected']:>10.4f} "
              f"{result['p_final']:>10.4f} {result['status']:>8}")

    print()
    print(f"  Naive: {naive_sig_count}/{n_experiments} significant "
          f"({100*naive_sig_count/n_experiments:.0f}%, expected ~5%)")
    print(f"  Robust: {robust_sig_count}/{n_experiments} significant "
          f"({100*robust_sig_count/n_experiments:.0f}%)")
    ir = inflation_rate(naive_sig_count, robust_sig_count)
    print(f"  Inflation Rate: {ir:.0%}")
    print()


def demo_true_effect():
    """Show that true effects survive correction."""
    print("=" * 60)
    print("  4. True Effect: Detectable After Correction")
    print("=" * 60)
    print()

    print("  Generating data with TRUE effect (effect_size=0.5, rho=0.9)...")
    data = generate_synthetic_dataset(
        n_conversations=50, turns_per_conv=80,
        rho=0.9, effect_size=0.5,
        seed=42,
    )

    result = combined_test(data, n_boot=2000, seed=42)

    print(f"  r_obs     = {result['r_obs']:+.4f}")
    print(f"  p_naive   = {result['p_naive']:.2e}")
    print(f"  p_boot    = {result['p_boot']:.4f}")
    print(f"  p_chelton = {result['p_corrected']:.4f}")
    print(f"  p_final   = {result['p_final']:.4f}")
    print(f"  n_eff     = {result['n_eff']:.0f} (nominal n = {result['n_total']})")
    print(f"  mean rho  = {result['mean_autocorr']:.3f}")
    print(f"  Status    = {result['status']}")
    print(f"  95% CI    = [{result['boot_ci_95'][0]:+.4f}, {result['boot_ci_95'][1]:+.4f}]")
    print()
    print("  -> True effects survive both corrections")
    print()


if __name__ == '__main__':
    print()
    print("  Autocorrelation Correction for Conversational Metrics")
    print("  " + "=" * 55)
    print()

    demo_ar1_autocorrelation()
    demo_chelton_correction()
    demo_null_hypothesis()
    demo_true_effect()

    print("  Done.")
