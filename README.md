# Autocorrelation Correction for Turn-Level Conversational Metrics

Statistical correction pipeline for autocorrelation-induced false positives in LLM conversation analysis.

**Paper:** Schessl, F. (2026). *The Autocorrelation Blind Spot: Why 43% of Turn-Level Findings in LLM Conversation Analysis May Be Spurious.* [arXiv:XXXX.XXXXX]

## The Problem

Turn-level metrics in LLM conversations (e.g., semantic velocity, frame distance, lexical diversity) are highly autocorrelated: consecutive turns within a conversation are not independent observations. Standard statistical tests (point-biserial *r*, Spearman rho) ignore this dependency and produce inflated significance.

In our study of 150 conversations (9,599 turn pairs, 4 LLM providers), **43% of nominally significant findings were spurious** after correcting for autocorrelation.

## Methods

This package implements two complementary corrections:

### 1. Chelton (1983) Effective Degrees of Freedom

Adjusts the sample size based on lag-1 autocorrelation:

```
n_eff = n * (1 - rho) / (1 + rho)
```

A metric with rho = 0.93 (e.g., cumulative budget scores) reduces n = 10,000 to n_eff = 362. The corrected p-value uses a t-distribution with df = n_eff - 2.

### 2. Block Bootstrap (Conversations as Clusters)

Resamples entire conversations (not individual turns), preserving within-conversation autocorrelation. The bootstrap p-value is the proportion of iterations where the correlation sign flips.

### Combined Test

A finding is considered **cluster-robust** only if it survives BOTH corrections:

```
p_final = max(p_bootstrap, p_chelton)
```

## Quick Start

```bash
pip install numpy scipy pandas
```

```python
from src import combined_test, generate_synthetic_dataset

# Generate synthetic data with known autocorrelation
data = generate_synthetic_dataset(
    n_conversations=50,
    turns_per_conv=80,
    rho=0.9,           # high autocorrelation
    effect_size=0.0,    # no true effect (null hypothesis)
)

# Run combined test
result = combined_test(data, n_boot=2000)
print(f"r = {result['r_obs']:+.4f}")
print(f"p_naive   = {result['p_naive']:.4f}")    # likely < 0.05 (inflated!)
print(f"p_final   = {result['p_final']:.4f}")    # likely > 0.05 (corrected)
print(f"Status    = {result['status']}")          # likely 'weak'
print(f"n_eff     = {result['n_eff']:.0f} (nominal: {result['n_total']})")
```

## Usage with Real Data

Structure your data as a dictionary of conversations:

```python
from src import combined_test

conversations = {
    'chat_001': {
        'binary': [0, 0, 0, 1, 1, 0, 1, ...],  # per-turn labels (0/1)
        'metric': [0.12, 0.15, 0.14, 0.82, ...], # per-turn metric values
    },
    'chat_002': {
        'binary': [...],
        'metric': [...],
    },
    # ... more conversations
}

result = combined_test(conversations, n_boot=2000)

if result['significant']:
    print(f"ROBUST: r={result['r_obs']:+.3f}, p={result['p_final']:.4f}")
else:
    print(f"weak: r={result['r_obs']:+.3f}, p={result['p_final']:.4f} (inflated)")
```

## Running the Demo

```bash
python examples/demo_correction.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Autocorrelation by Metric Family

| Metric Family | Mean rho | n_eff/n | Risk |
|:---|:---:|:---:|:---:|
| Embedding velocity | 0.05 | 90% | Low |
| Compression (NCD) | 0.12 | 79% | Low |
| Directional | 0.19 | 68% | Low |
| Thermodynamic | 0.35 | 48% | Medium |
| Impulse (1st diff.) | 0.44 | 39% | Medium |
| Frame distance | 0.45 | 38% | Medium |
| Lexical/structural | 0.50 | 33% | High |
| Normalized cumul. | 0.73 | 16% | High |
| Rolling (W=20) | 0.93 | 4% | Very High |
| Cumulative | 0.93 | 4% | Very High |

## Citation

```bibtex
@article{schessl2026autocorrelation,
  title={The Autocorrelation Blind Spot: Why 43\% of Turn-Level Findings
         in LLM Conversation Analysis May Be Spurious},
  author={Schessl, Ferdinand},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
