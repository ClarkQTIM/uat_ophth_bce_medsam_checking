# Core MedSAM results, 2026-10-20 (78 of 80 runs; INTERIM for cup Conf-0.20, n = 8)

`configs/errata/medsam_core.yaml`: the same 8 cells as `parity_sam_core`, on MedSAM
(`wanglab/medsam-vit-base`, 93.7M parameters, asserted per run). Seeds 1001–1010.
Analysis: `scripts/select_offline.py` (DECISIONS D16–D18). Two cup Conf-0.20 runs were
still training; every other arm has n = 10. Raw output and per-run tables in this folder.

Note on the script output: its "paper runs" and "D12 parity" lines compare with the
paper-era runs, which were SAM ViT-H (ERRATA E1). For MedSAM they show how MedSAM differs
from the published numbers, not pipeline parity (parity is a SAM-only question, D12/D17 Q1).

Test soft Dice, mean (n = 10 unless stated). Δ = arm − baseline under the same criterion.
P = Mann-Whitney vs baseline (Holm across the three UQ arms); paired = Wilcoxon by seed (D18).

## Cup (MedSAM)

| Criterion | Baseline | Entropy γ5 W5 | Δ, P (Holm), paired P | Variance γ3 W5 | Δ, P (Holm), paired P | Conf-0.20 γ2 W1 (n = 8) | Δ, P |
|---|---|---|---|---|---|---|---|
| as run (weighted) | 0.701 | 0.770 | +0.069, 0.0002 (0.0005), 0.002 | 0.717 | +0.016, 0.12 (0.24) | 0.708 | +0.007, 0.41 |
| **Q2 stated (unweighted)** | 0.701 | 0.664 | **−0.037, 0.031 (0.093)**, 0.039 | 0.704 | +0.003, 0.68 | 0.713 | +0.012, 0.41 |
| **Q3 common (val Dice)** | 0.761 | 0.775 | **+0.014, 0.054 (0.11)**, 0.049 | 0.777 | +0.016, 0.038 (0.11), 0.027 | 0.757 | −0.004, 0.90 |

Selected epoch (as run / Q3): baseline 7.3 / 19.0; entropy 12.9 / 10.9.
Q2: entropy selects a warmup (pre-UAT) checkpoint in **10/10** runs (as on SAM).

## GA (MedSAM)

| Criterion | Baseline | Entropy γ5 W5 | Δ, P (Holm), paired P | Variance γ5 W3 | Δ, P (Holm), paired P | Conf-0.25 γ4 W2 | Δ, P (Holm), paired P |
|---|---|---|---|---|---|---|---|
| as run (weighted) | 0.699 | 0.767 | +0.068, 0.011 (0.034), 0.010 | 0.726 | +0.027, 0.21 | 0.733 | +0.034, 0.24, 0.065 |
| **Q2 stated (unweighted)** | 0.699 | 0.731 | **+0.032, 0.17 (0.35)**, 0.20 | 0.726 | +0.027, 0.21 | 0.743 | +0.044, 0.089 (0.27), 0.004 |
| **Q3 common (val Dice)** | 0.747 | 0.796 | **+0.049, 0.006 (0.017)**, 0.010 | 0.757 | +0.009, 0.79 | 0.767 | +0.020, 0.35, 0.049 |

Selected epoch (as run / Q3): baseline 9.1 / 15.3; entropy 9.3 / 10.9.
Q2: entropy selects a warmup checkpoint in 2/10 runs.

## Pre-registered verdict (D17: entropy > baseline, P < 0.05, under both Q2 and Q3)

- **Cup: NOT upheld.** Q2 is degenerate (warmup checkpoints, 10/10) and goes the wrong
  way (−0.037). Q3: +0.014, P = 0.054 — a small effect, not significant (paired P = 0.049,
  supplementary). The as-run gain (+0.069) again comes with the baseline stopped at
  epoch ≈ 7 on BCE.
- **GA: NOT upheld under the conjunction** (Q2 +0.032, P = 0.17), **upheld under Q3**:
  +0.049, P = 0.006 (Holm 0.017, paired 0.010).
- Variance and conformal: no arm significant after Holm under Q3 in either dataset.

## MedSAM vs SAM ViT-H vs the paper (entropy cell and baseline)

| | Paper (= SAM ViT-H runs) | SAM re-run, as run | MedSAM, as run | SAM, Q3 Δ | MedSAM, Q3 Δ |
|---|---|---|---|---|---|
| Cup baseline | 0.705 (figure) | 0.725 | 0.701 | | |
| Cup entropy | 0.781 | 0.785 | 0.770 (vs paper P = 0.011) | +0.004 | +0.014 |
| GA baseline | 0.728 (figure) | 0.714 | 0.699 | | |
| GA entropy | 0.824 | 0.807 | 0.767 (vs paper P = 0.002) | +0.065 | +0.049 |

## Plain reading

1. **MedSAM reproduces the pattern found on SAM.** Under the procedure as run, entropy-UAT
   beats the baseline in both datasets by ≈ 0.07, as published. Under one stopping rule
   for all arms, the cup gain shrinks to +0.014 (not significant) and the GA gain stays
   (+0.049, significant after Holm).
2. **The paper's numbers are not MedSAM numbers.** With MedSAM the same procedure gives
   lower Dice than published, most clearly for GA entropy (0.767 vs 0.824).
3. **The stated stopping rule (Methods) is unusable for the entropy arm on either backbone:**
   in cup it selects a model never trained with UAT in every run.
4. Across both backbones the evidence supports one claim: entropy-UAT improves GA
   segmentation under a fair comparison (≈ +0.05 to +0.065). It does not support a
   consistent improvement across datasets; FAZ is untested on the original split.

Final numbers for cup Conf-0.20 (n = 10) will replace the interim n = 8 when the last two
runs finish; no other cell is affected.
