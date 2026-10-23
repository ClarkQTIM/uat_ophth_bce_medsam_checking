# Patient-level splits and FAZ, MedSAM — results 2026-10-23

Two experiments, both MedSAM (`wanglab/medsam-vit-base`), both on the H100 NVL machine
(ophlslcec08), recorded per run in `run_manifest.json`:

- **Patient-level split sensitivity** (DECISIONS D20): GA and cup, the 8 core cells, 10 runs
  each, splits from `scripts/make_patient_split.py --seed 1020` (no patient in more than one
  split). 80/80 runs. Output root `results/patient_split_v1/`.
- **FAZ on the seed-2026 split** (DECISIONS D19; ERRATA E15): baseline, entropy γ4 W2,
  variance γ4 W3, Conf-0.05 γ1 W1, 10 runs each. 39/40 runs (Conf-0.05 n = 9; final run
  pending). OCTA-500 has one eye per subject, so this split is patient-level by construction.
  Tables: `docs/records/faz_medsam_2026-10-23/`.

Per-image-split comparison runs (`results/errata_v2`, cup and GA) were made on the A100
machine (ophlappc05). Across split types, therefore, hardware also differs; within each
experiment every arm ran on the same hardware.

Test soft Dice, mean ± SD (n = 10). Δ = arm − baseline under the same criterion; P =
Mann-Whitney (Holm across the 3 UQ arms); paired = Wilcoxon by seed (D18, supplementary).
Criteria as in D17: **Q3 = validation-Dice stopping for every arm (primary)**; Q2 = stated
unweighted BCE; as_run = the paper-era weighted rule.

## GA: per-image split (A100) vs patient-level split (H100)

| Criterion | Split | Baseline | Entropy | Δ | P (Holm) | paired P |
|---|---|---|---|---|---|---|
| **Q3 (val Dice)** | per-image | 0.747 ± 0.038 | 0.796 ± 0.027 | +0.049 | 0.006 (0.017) | 0.010 |
| **Q3 (val Dice)** | **patient** | 0.797 ± 0.030 | 0.823 ± 0.032 | **+0.026** | **0.064 (0.19)** | 0.006 |
| Q2 (unweighted) | per-image | 0.699 | 0.731 | +0.032 | 0.17 | 0.20 |
| Q2 (unweighted) | patient | 0.772 | 0.800 | +0.028 | 0.27 | 0.28 |
| as run (weighted) | per-image | 0.699 | 0.767 | +0.068 | 0.011 (0.034) | 0.010 |
| as run (weighted) | patient | 0.772 | 0.828 | +0.055 | 0.011 (0.034) | 0.002 |

Variance and Conf-0.25 on the patient split: Q3 −0.019 and −0.010 (P ≥ 0.79); no gain.

## Cup: per-image split (A100) vs patient-level split (H100)

| Criterion | Split | Baseline | Entropy | Δ | P (Holm) | paired P |
|---|---|---|---|---|---|---|
| **Q3 (val Dice)** | per-image | 0.761 ± 0.019 | 0.774 ± 0.009 | +0.014 | 0.054 (0.11) | 0.049 |
| **Q3 (val Dice)** | **patient** | 0.749 ± 0.027 | 0.765 ± 0.009 | **+0.017** | **0.12 (0.36)** | 0.065 |
| Q2 (unweighted) | per-image | 0.701 | 0.664 | −0.037 | 0.031 | 0.039 |
| Q2 (unweighted) | patient | 0.691 | 0.677 | −0.013 | 0.52 | 0.38 |
| as run (weighted) | per-image | 0.701 | 0.770 | +0.069 | 0.0002 | 0.002 |
| as run (weighted) | patient | 0.691 | 0.752 | +0.061 | 0.0002 | 0.002 |

Q2 on the patient split: entropy again selects a warmup (pre-UAT) checkpoint in 8/10 runs.
Variance and Conf-0.20: Q3 +0.000 and −0.004; no gain.

## FAZ, seed-2026 split (H100), MedSAM — interim, 39/40

| Criterion | Baseline | Entropy γ4 W2 | Δ | P (Holm) | paired P |
|---|---|---|---|---|---|
| **Q3 (val Dice)** | 0.911 ± 0.005 | 0.920 ± 0.005 | **+0.009** | **0.006 (0.017)** | 0.004 |
| Q2 (unweighted) | 0.901 | 0.905 | +0.004 | 0.43 | 0.43 |
| as run (weighted) | 0.901 | 0.916 | +0.015 | 0.0002 | 0.002 |

Variance: as run −0.011 (P = 0.054), Q3 −0.005. Conf-0.05 (n = 9): Q3 −0.001. Selected
epochs under Q3 are similar for baseline and entropy (26.5 vs 23.5). Paper (SAM ViT-H,
original split, as run): baseline 0.888, entropy 0.907 (+0.019).

## Reading

1. **GA's gain shrinks on patient-level splits.** Under the common stopping rule the entropy
   gain falls from +0.049 (P = 0.006) to +0.026, and the pre-registered test is no longer
   significant (P = 0.064; Holm 0.19). Seed-paired it remains positive (P = 0.006, 9 of 10
   seeds better; supplementary). The baseline is much stronger on the patient split (0.797 vs
   0.747), so part of the per-image gain reflected a weak baseline on that particular test set.
2. **Cup is unchanged:** a small positive trend (+0.014 to +0.017) that is not significant on
   either split; the as-run gain (+0.06 to +0.07) again comes with the baseline stopped at
   epoch ≈ 7.
3. **FAZ shows a small, significant entropy gain under the common rule** (+0.009, Holm
   0.017), half the size of the paper's reported FAZ gap, on a different split.
4. **Variance and conformal arms show no gain anywhere under the common rule** on MedSAM.
5. **Pre-registered D17 verdict (Q2 and Q3 both): not upheld for any dataset.** Under Q3
   alone, entropy is significant for FAZ and for GA on the per-image split only.

Observation, not pre-registered: under Q3 every entropy-vs-baseline difference measured so
far is positive (MedSAM cup +0.014/+0.017, GA +0.049/+0.026, FAZ +0.009; SAM cup +0.004,
GA +0.065). These comparisons share datasets and are not independent, so this is a
direction, not a test. The defensible summary: entropy-UAT may give a small improvement,
far smaller than reported and significant only in some settings; the reported gains are
largely an effect of the stopping rule.
