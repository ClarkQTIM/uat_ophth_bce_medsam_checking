# 3-class UAT on MedSAM (PAPILA background / rim / cup) — results 2026-10-29

Pre-registered in DECISIONS D21. Patient-level split (D20, seed 1020); STAPLE 3-class map;
MedSAM; settings chosen on validation only (`tuning_selection.txt`, committed before the
final runs): **entropy γ3 W2, conformal α0.20 γ1 W1**. 10 runs per arm (seeds 1001–1010);
all 30 final runs and all 36 tuning runs on the H100 NVL machine. Primary outcome: **test
cup soft Dice under validation-Dice stopping**. Raw output: `select_offline_output.txt`.

Δ = arm − baseline under the same stopping criterion; P = Mann-Whitney (Holm across the 2
UQ arms); paired = Wilcoxon by seed (supplementary).

## Primary: cup Dice, validation-Dice stopping

| Arm | Cup Dice | Δ | P (Holm) | paired P | seeds better |
|---|---|---|---|---|---|
| Baseline | 0.764 ± 0.009 | — | — | — | — |
| Entropy γ3 W2 | 0.761 ± 0.019 | −0.003 | 0.79 (1.0) | 1.0 | 6/10 |
| Conformal γ1 W1 | 0.763 ± 0.006 | −0.001 | 0.79 (1.0) | 0.77 | 4/10 |

**Pre-registered verdict: not upheld.** Neither UQ arm improves cup Dice under the common
stopping rule. Disc (baseline 0.960), rim (0.911) and mean (0.862): all |Δ| ≤ 0.001, P ≥ 0.57.

## The other stopping rules (same runs)

| Stopping rule | Cup: baseline | Entropy Δ (P, Holm) | Conformal Δ (P) | Disc: entropy Δ (P) |
|---|---|---|---|---|
| Weighted validation CE (as the paper's code did) | 0.697 | **+0.057 (0.0002, 0.0005)** | +0.005 (0.91) | +0.009 (0.0002) |
| Unweighted validation CE (as the paper's Methods say) | 0.697 | +0.026 (0.021, 0.042) | +0.009 (0.52) | +0.006 (0.003) |
| **Validation Dice (common rule, primary)** | **0.764** | **−0.003 (0.79)** | **−0.001 (0.79)** | +0.001 (0.57) |

Mean selected epoch. CE-based rules: baseline 8.1, entropy 5.4–9.0, conformal 8.3–8.5.
Validation-Dice rule: baseline **25.4**, entropy **13.7**, conformal 24.5.

## Reading

1. **No accuracy gain in 3-class segmentation.** With every model stopped on validation Dice,
   entropy and conformal UAT end at the baseline's cup and disc Dice.
2. **The binary pattern repeats.** Stopped on validation cross-entropy, weighted or not, all
   models stop around epoch 8, while Dice keeps improving until about epoch 25 for the baseline.
   At epoch 8 the entropy model is well ahead (+0.057 cup), so CE-based stopping shows a large,
   highly significant "gain" that disappears when every model may train to its best.
3. **Entropy-UAT gets there faster.** Under the common rule, entropy reaches its best validation
   Dice at epoch 13.7 on average vs 25.4 for the baseline (epochs: entropy 5–21, baseline 14–36;
   Mann-Whitney P = 0.004), at the same final Dice. Conformal does not (24.5, P = 0.88).
   *Not pre-registered.* It matches what was seen in all 8 binary settings (entropy peaked earlier
   under the common rule every time), so across 9 of 9 settings the consistent effect of
   entropy-UAT is faster convergence, not higher final accuracy. That is a hypothesis to test in
   advance on new data, not yet a result.
