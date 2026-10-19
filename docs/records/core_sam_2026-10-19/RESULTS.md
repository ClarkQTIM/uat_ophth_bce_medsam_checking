# Core SAM ViT-H results, 2026-10-19 (80 runs, complete)

`configs/errata/parity_sam_core.yaml`: cup and GA; baseline, entropy, variance and the
conformal arm shown in Figures 1–2; 10 runs (seeds 1001–1010) per arm; every run
trained until all three stopping criteria had exhausted patience 5.
Analysis: `scripts/select_offline.py` (DECISIONS D16–D18). Raw output and per-run
tables in this folder (`*_summary.csv`, `*_selections.csv`, `select_offline_output.txt`).

Test soft Dice, mean ± SD (n = 10). Δ = arm − baseline under the same criterion.
P = Mann-Whitney vs baseline (Holm across the three UQ arms in parentheses);
paired = Wilcoxon signed-rank by seed (D18, supplementary).

## Q1 — Parity: does the new pipeline reproduce the paper's runs? (criterion `as_run`)

| Cell | Re-run | Paper-era runs (n = 10) | P (re-run vs paper runs) |
|---|---|---|---|
| cup entropy γ5 W5 | 0.785 ± 0.007 | 0.781 | 0.35 |
| cup variance γ3 W5 | 0.747 ± 0.016 | 0.742 | 0.39 |
| cup Conf-0.20 γ2 W1 | 0.737 ± 0.020 | 0.739 | 0.97 |
| GA entropy γ5 W5 | 0.807 ± 0.036 | 0.824 | 0.31 |
| GA variance γ5 W3 | 0.767 ± 0.071 | 0.787 | 0.97 |
| GA Conf-0.25 γ4 W2 | 0.785 ± 0.035 | 0.801 | 0.39 |

All six UQ cells are statistically indistinguishable from the paper's own runs.
Baselines (no paper-era baseline runs survive; reference = figure readings ±0.003):
cup 0.725 vs 0.705 (Δ 0.0204, tolerance 0.02), GA 0.714 vs 0.728.

**D12 verdict (mechanical):** GA **PASS**; cup **FAIL** on the baseline criterion only,
by 0.0004 beyond the tolerance, against a figure reading with ±0.003 uncertainty. Entropy
mean and entropy > baseline (P = 0.0002) pass for cup. **Interpretation:** the pipeline
reproduces the paper-era procedure; the marginal cup baseline miss is within the
uncertainty of the reference value. (Stated here as an interpretation, not as a change
to the pre-registered criterion.)

## Q2 / Q3 — Does UAT improve Dice? (pre-registered, D17)

### Cup

| Criterion | Baseline | Entropy | Δ, P (Holm), paired P | Variance | Δ, P (Holm), paired P | Conf-0.20 | Δ, P |
|---|---|---|---|---|---|---|---|
| as run (weighted) | 0.725 | 0.785 | +0.059, 0.0002 (0.0005), 0.002 | 0.747 | +0.022, 0.031 (0.062), 0.027 | 0.737 | +0.011, 0.24 |
| **Q2 stated (unweighted)** | 0.725 | 0.702 | **−0.023, 0.075 (0.23)**, 0.063 | 0.723 | −0.003, 0.97 | 0.722 | −0.004, 0.47 |
| **Q3 common (val Dice)** | **0.781** | 0.785 | **+0.004, 0.31 (0.62)**, 0.43 | 0.791 | +0.010, 0.017 (0.052), 0.020 | 0.781 | 0.000, 1.0 |

Mean selected epoch (as-run / Q3): baseline 5.7 / 19.4; entropy 11.5 / 14.0.

### GA

| Criterion | Baseline | Entropy | Δ, P (Holm), paired P | Variance | Δ, P (Holm), paired P | Conf-0.25 | Δ, P (Holm), paired P |
|---|---|---|---|---|---|---|---|
| as run (weighted) | 0.714 | 0.807 | +0.093, 0.0006 (0.002), 0.002 | 0.767 | +0.053, 0.045 (0.045), 0.065 | 0.785 | +0.071, 0.005 (0.009), 0.002 |
| **Q2 stated (unweighted)** | 0.714 | 0.746 | **+0.032, 0.34 (0.34)**, 0.078 | 0.758 | +0.044, 0.076 (0.15), 0.084 | 0.759 | +0.045, 0.026 (0.077), 0.002 |
| **Q3 common (val Dice)** | 0.749 | 0.813 | **+0.065, 0.0046 (0.014)**, 0.004 | 0.784 | +0.035, 0.064 (0.11), 0.027 | 0.782 | +0.033, 0.054 (0.11), 0.014 |

Mean selected epoch (as-run / Q3): baseline 6.9 / 12.3; entropy 9.9 / 10.1.

### Pre-registered verdict (D17: upheld only if entropy > baseline, P < 0.05, under both Q2 and Q3)

- **Cup: NOT upheld.** No entropy gain under Q2 (−0.023) or Q3 (+0.004). The reported
  cup gain (+0.059) appears only under the paper-era weighted stopping rule, which stops
  the baseline at epoch ≈ 6; a baseline stopped on validation Dice (epoch ≈ 19) reaches
  0.781, the paper's entropy-UAT value.
- **GA: NOT upheld under the conjunctive D17 rule** (Q2: +0.032, P = 0.34), **but the
  common-criterion test Q3 shows a clear gain**: +0.065, P = 0.005 (Holm 0.014; paired
  P = 0.004), with similar training length in both arms (selected epoch 12.3 vs 10.1).

### Q2 is degenerate for the entropy arm (observation made after seeing the results)

Under the stated unweighted rule, the entropy arm (W = 5: epochs 1–5 unweighted)
selects a checkpoint from **inside warmup in 10 of 10 cup runs and 4 of 10 GA runs**
(variance: 5/10 cup, 1/10 GA; conformal 0/10 in both). A warmup checkpoint has never
been trained with the weighted loss: it lies on the same training trajectory as the
same-seed baseline, only stopped earlier (selected epoch: cup entropy 4.2 vs baseline
5.7). Once weighting starts, entropy-weighted training makes predictions more extreme,
the unweighted validation BCE rises, and patience runs out. So for cup, Q2 compares the
baseline with an earlier checkpoint of itself and **cannot test UAT at all**; Q3 is the
informative test. This does not change the pre-registered verdicts above; it explains
them. (Counts from `*_selections.csv`, `best_epoch ≤ n_warmup`, epochs 1-based.)

Corollary for the paper: had early stopping been done as the Methods state, every cup
entropy model reported would have been a model trained without UAT.

### Plain reading

1. The published numbers are reproducible: they are what the procedure that was actually
   run produces.
2. **OC (cup):** the reported UAT improvement is explained by the stopping criterion, not
   by uncertainty-aware training. With the same stopping rule for every arm there is no
   entropy gain; variance shows a small gain (+0.010, Holm P = 0.052).
3. **GA:** entropy-UAT improves Dice by ≈ 0.065 over a baseline given the same stopping
   rule — a genuine effect, smaller than the reported +0.097 (0.825 vs 0.728).
4. Early stopping on unweighted BCE (the paper's stated rule) is unusable for the
   entropy arm: it selects a pre-UAT checkpoint in every cup run and 4 of 10 GA runs
   (see above).
5. Variance and conformal arms: no reliable gain under Q3 in either dataset after Holm
   (cup variance +0.010, Holm P = 0.052; GA variance +0.035, conformal +0.033, Holm
   P = 0.11; paired-by-seed P < 0.05 for these three, supplementary).

Still to come: MedSAM (same 80 cells), FAZ (split file not found), the remaining
conformal α arms, and the K14 ablation if adopted.
