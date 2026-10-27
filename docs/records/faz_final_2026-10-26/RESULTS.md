# FAZ (OCTA-500), seed-2026 split — final, both backbones (2026-10-26)

DECISIONS D19 (new split, seed fixed in advance; ERRATA E15). OCTA-500 has one eye per
subject, so the split is patient-level by construction. Cells unchanged from the paper:
baseline, entropy γ4 W2, variance γ4 W3, Conf-0.05 γ1 W1; 10 runs each (seeds 1001–1010);
all 80 runs on the H100 NVL machine (ophlslcec08). Supersedes the interim MedSAM numbers
(39/40) in docs/records/patient_split_2026-10-23/RESULTS.md; only Conf-0.05 changed.

Test soft Dice, Δ = arm − baseline under the same criterion; P = Mann-Whitney (Holm across
the 3 UQ arms); paired = Wilcoxon by seed (D18, supplementary).

## Entropy vs baseline

| Backbone | Criterion | Baseline | Entropy | Δ | P (Holm) | paired P |
|---|---|---|---|---|---|---|
| SAM ViT-H | as run (weighted) | 0.901 | 0.922 | +0.022 | 0.0003 (0.001) | 0.002 |
| SAM ViT-H | Q2 stated (unweighted) | 0.901 | 0.921 | **+0.020** | **0.0006 (0.002)** | 0.002 |
| SAM ViT-H | **Q3 common (val Dice)** | 0.921 ± 0.009 | 0.926 ± 0.005 | **+0.005** | **0.39 (1.0)** | 0.037 |
| MedSAM | as run (weighted) | 0.901 | 0.916 | +0.015 | 0.0002 (0.0005) | 0.002 |
| MedSAM | Q2 stated (unweighted) | 0.901 | 0.905 | +0.004 | 0.43 (0.86) | 0.43 |
| MedSAM | **Q3 common (val Dice)** | 0.911 ± 0.005 | 0.920 ± 0.005 | **+0.009** | **0.006 (0.017)** | 0.004 |

Paper (SAM ViT-H, original split, as run): baseline 0.888, entropy 0.907 (+0.019).

## Variance and conformal

| Backbone | Arm | as run Δ (P) | Q3 Δ (P) |
|---|---|---|---|
| SAM ViT-H | variance γ4 W3 | +0.013 (0.007, Holm 0.015) | +0.003 (0.62) |
| SAM ViT-H | Conf-0.05 γ1 W1 | +0.005 (0.19) | +0.003 (0.91) |
| MedSAM | variance γ4 W3 | −0.011 (0.054) | −0.005 (0.79) |
| MedSAM | Conf-0.05 γ1 W1 | +0.004 (0.19) | +0.000 (0.85) |

## Reading

- **SAM ViT-H, the model the paper ran, reproduces the paper's FAZ gain under its as-run
  rule** (+0.022 vs the paper's +0.019) on a different split. Here, unlike cup and GA, the
  gain also holds under the stated unweighted rule (W = 2: entropy never selects a warmup
  checkpoint). **Under the common validation-Dice rule it shrinks to +0.005 and is not
  significant** (P = 0.39; 7/10 seeds positive, paired P = 0.037, supplementary): the
  baseline trains to epoch ≈ 24 and reaches 0.921.
- **MedSAM:** the opposite pattern: no gain under the stated rule, a small significant gain
  under the common rule (+0.009).
- **D17 verdict (Q2 and Q3 both): not upheld for FAZ on either backbone.**
- Variance and conformal: no gain under the common rule on either backbone.
