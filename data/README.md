# Full Results Data

## `full_results_table.csv`

Persisted results entries from the study underlying *The Autocorrelation
Blind Spot*. The CSV combines four complementary slices; the full 81
pooled-significant / 47 cluster-robust inventory from the final n=202 run
is only printed to stdout by `layer_mapping_validation.py` and not currently
persisted as a single CSV — the entries below document what is stable
across runs.

### Source tags

| `source` value | What it is | n rows |
|---|---|---|
| `paper_tab_full_representative` | 6 representative pooled-sig pairs shown in Appendix B (Table 6 in the paper) covering the spectrum from low-ρ (embedding velocity, compression) to high-ρ (EWMA, timestamp) | 6 |
| `paper_table3_aggregate_n202` | Category-level aggregation from Table 3 in the paper: for each of the 12 metric families, counts of pooled-sig/cluster-robust findings and mean ρ̄ | 14 (12 categories + 2 totals) |
| `preregistered_exploratory_n152` | The six pre-registered directional hypotheses H1–H6 evaluated on the exploratory set | 6 |
| `holdout_tierA_n36` | All cluster-robust correlations (p_boot < 0.05) from the strictly-unseen Tier-A hold-out (36 chats, 1,829 turn pairs) | 25 |

### Columns

- `code` — identifier
- `category` — metric family / metric name
- `label` — gold label: `malicious`, `typx`, `rewiring`, or `all`
- `r_obs` — point-biserial correlation
- `p_pooled` — naive pooled p-value
- `rho_bar` — mean lag-1 autocorrelation of the metric
- `n_eff` — Chelton effective sample size
- `p_final` — max(p_Chelton, p_bootstrap) after cluster-robust correction
- `status` — `robust`, `weak`, `confirmed` (pre-registered), or free-text
  counts for aggregate rows
- `source` — which study phase the row comes from

### Relation to headline numbers

- Paper headline: **IR = 42%** = (81 − 47) / 81 for all three labels on the
  full n = 202 corpus.
- Malicious-only IR: **39%** = (44 − 27) / 44.
- Hold-out replication: cluster-robust entries replicate at **57%** in
  Tier-A vs. **30%** for pooled-only entries (paper Table 4).

### Full 81-row inventory

The exhaustive per-metric-pair results for the final n=202 run are produced
by `layer_mapping_validation.py` via `report_cluster_robust()` and
`report_typx_cluster_robust()` (printed to stdout). If you need the full
table, re-run the validation pipeline and pipe the output to a log file,
then parse the section headers. This will be added to the repo as a CSV in
a future release.
